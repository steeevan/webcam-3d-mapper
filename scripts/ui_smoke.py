"""Headless browser smoke test for the whole UI.

Optional developer tool. Chromium's fake media device stands in for a webcam, so the
ready -> scanning -> processing path runs for real, and the developer image import drives the
processing -> result path including the WebGL viewer.

    pip install playwright && python -m playwright install chromium
    python app.py &
    python scripts/make_test_scene.py --out .cache/testscene
    python scripts/ui_smoke.py --images .cache/testscene --screenshots .cache/shots

Exits non-zero if any stage fails, a page error is raised, or the viewer draws nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    sys.exit("playwright is not installed:  pip install playwright && playwright install chromium")

CHROMIUM_ARGS = [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    # SwiftShader gives headless Chromium a working WebGL implementation.
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--images", required=True, help="folder of images for the dev import")
    parser.add_argument("--screenshots", default="", help="folder to write screenshots into")
    parser.add_argument("--timeout", type=int, default=600, help="reconstruction timeout (s)")
    args = parser.parse_args()

    shots = Path(args.screenshots) if args.screenshots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    page_errors: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {label}{f'  ({detail})' if detail else ''}")
        if not condition:
            failures.append(label)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=CHROMIUM_ARGS)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900}, permissions=["camera"]
        )
        page = context.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        def snap(name: str) -> None:
            if shots:
                page.screenshot(path=str(shots / f"{name}.png"))

        print("\nREADY")
        page.goto(args.url, wait_until="networkidle")
        time.sleep(2.5)
        check("camera preview is live", page.evaluate(
            "() => { const v = document.getElementById('preview'); return v.videoWidth > 0; }"
        ), page.evaluate("() => document.getElementById('preview').videoWidth + 'px wide'"))
        check("COLMAP pill is green", page.get_attribute("#pill-engine", "data-ok") == "true")
        check("Start Scan enabled", not page.is_disabled("#btn-start"))
        snap("1-ready")

        print("\nSCANNING")
        page.click("#btn-start")
        time.sleep(6)
        frames = int(page.inner_text("#scan-frames") or 0)
        check("frames are being accepted", frames > 0, f"{frames} good frames in ~6 s")
        check("scan overlay visible", page.is_visible("#scan-overlay"))
        check("timer running", page.inner_text("#scan-elapsed") != "00:00")
        snap("2-scanning")

        print("\nPROCESSING")
        page.click("#btn-finish")
        time.sleep(3)
        check("switched to processing", page.evaluate("document.documentElement.dataset.state")
              in ("processing", "error", "result"))
        stage_states = page.eval_on_selector_all("#stage-list li", "els => els.map(e => e.dataset.state)")
        check("stage list rendered", len(stage_states) == 5, str(stage_states))
        snap("3-processing")

        # The fake camera has no real parallax, so this scan is expected to fail - what
        # matters is that it fails with guidance instead of a stack trace.
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            state = page.evaluate("document.documentElement.dataset.state")
            if state in ("result", "error"):
                break
            time.sleep(2)
        if state == "error":
            check("failure explains what to do", bool(page.inner_text("#error-message").strip()),
                  page.inner_text("#error-title"))
            snap("4-error")
            page.click("#btn-error-retry")
            time.sleep(1)
            check("retry returns to capture",
                  page.evaluate("document.documentElement.dataset.state") == "capture")

        print("\nRESULT (developer import)")
        page.click("#btn-settings")
        page.fill("#import-folder", str(Path(args.images).resolve()))
        page.click("#btn-import")

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            state = page.evaluate("document.documentElement.dataset.state")
            if state in ("result", "error"):
                break
            time.sleep(1)

        check("reconstruction completed", state == "result",
              "" if state == "result" else page.inner_text("#error-message"))
        if state != "result":
            browser.close()
            return report(failures, page_errors)

        time.sleep(3)
        points = page.inner_text("#stat-points")
        check("point count shown", points not in ("", "0"), f"{points} points")
        check("registered images shown", page.inner_text("#stat-images") not in ("", "0"))
        drawn = page.evaluate(
            """() => {
                const canvas = document.getElementById('viewer-canvas');
                const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                return { width: canvas.width, height: canvas.height, context: !!gl };
            }"""
        )
        check("canvas sized and WebGL live", drawn["context"] and drawn["width"] > 100,
              f"{drawn['width']}x{drawn['height']}")
        snap("5-result")

        page.evaluate(
            """() => {
                const el = document.getElementById('point-size');
                el.value = '3.0';
                el.dispatchEvent(new Event('input', { bubbles: true }));
            }"""
        )
        page.uncheck("#show-cameras")
        page.mouse.move(700, 450)
        page.mouse.down()
        page.mouse.move(900, 370, steps=12)
        page.mouse.up()
        page.mouse.wheel(0, -350)
        time.sleep(1)
        snap("6-interacted")
        page.click("#btn-reset-view")
        time.sleep(1)

        with page.expect_download() as download_info:
            page.click("#btn-export")
        download = download_info.value
        check("export downloads a .ply", download.suggested_filename.endswith(".ply"),
              download.suggested_filename)

        page.click("#btn-new-scan")
        time.sleep(2)
        check("New scan returns to capture",
              page.evaluate("document.documentElement.dataset.state") == "capture")
        check("counters cleared", page.inner_text("#scan-frames") == "0"
              and page.inner_text("#progress-pct") == "0%")
        check("preview still live",
              page.evaluate("() => !!document.getElementById('preview').srcObject"))
        snap("7-new-scan")

        browser.close()

    return report(failures, page_errors)


def report(failures: list[str], page_errors: list[str]) -> int:
    print()
    if page_errors:
        print("Uncaught page errors:")
        for error in page_errors:
            print(f"  {error}")
    if failures or page_errors:
        print(f"FAILED: {len(failures)} check(s), {len(page_errors)} page error(s)")
        return 1
    print("All UI checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
