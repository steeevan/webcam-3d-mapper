"""Download COLMAP into ``vendor/colmap/``, picking the build that suits this computer.

    python scripts/get_colmap.py              # CUDA build if an NVIDIA GPU is found, else CPU-only
    python scripts/get_colmap.py --no-cuda    # force the CPU-only build
    python scripts/get_colmap.py --force      # replace an existing vendor/colmap/

Windows only: the official releases ship ready-to-run Windows zips. On macOS and Linux COLMAP
comes from the package manager, and this script says which command to run.

The app searches ``vendor/colmap/`` automatically, so nothing else needs configuring. A
``COLMAP_PATH`` variable or ``colmap_path.txt`` still takes precedence if one is set.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.colmap_runner import _detect_gpu, _probe, describe_start_failure  # noqa: E402

#: The release the app's option handling was verified against. Newer ones should work too,
#: because the runner adapts to whatever options ``colmap <command> -h`` advertises.
DEFAULT_VERSION = "4.2.0"
RELEASE_URL = "https://github.com/colmap/colmap/releases/download/{version}/colmap-x64-windows-{flavour}.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--cuda", action="store_true", help="download the CUDA (NVIDIA) build")
    gpu.add_argument("--no-cuda", action="store_true", help="download the CPU-only build")
    parser.add_argument("--version", default=DEFAULT_VERSION, help=f"default {DEFAULT_VERSION}")
    parser.add_argument("--dest", type=Path, default=config.APP_DIR / "vendor" / "colmap")
    parser.add_argument("--force", action="store_true", help="replace an existing install")
    args = parser.parse_args()

    if os.name != "nt":
        print("COLMAP ships Windows builds only; install it with your package manager:")
        print("  macOS   brew install colmap")
        print("  Linux   sudo apt install colmap   (or build from source for CUDA support)")
        return 1

    dest: Path = args.dest.resolve()
    if dest.exists() and not args.force:
        print(f"{dest} already exists. Re-run with --force to replace it.")
        return 1

    gpu_name = _detect_gpu()
    if args.cuda or args.no_cuda:
        cuda = args.cuda
    else:
        cuda = gpu_name != "unknown"
    print(f"GPU: {gpu_name if gpu_name != 'unknown' else 'no NVIDIA GPU found'}")
    print(f"Build: {'CUDA' if cuda else 'CPU-only'} (COLMAP {args.version})")

    url = RELEASE_URL.format(version=args.version, flavour="cuda" if cuda else "nocuda")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".colmap-download-") as work:
        archive = Path(work) / "colmap.zip"
        _download(url, archive)
        extracted = Path(work) / "extracted"
        print("Extracting...")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        root = _install_root(extracted)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(root), str(dest))

    return _verify(dest)


def _download(url: str, target: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done >> 20} / {total >> 20} MB", end="", flush=True)
    print()


def _install_root(extracted: Path) -> Path:
    """The folder holding COLMAP.bat: the zip root, or its single top-level folder."""
    for candidate in (extracted, *sorted(extracted.iterdir())):
        if (candidate / "COLMAP.bat").is_file() or (candidate / "bin" / "colmap.exe").is_file():
            return candidate
    raise SystemExit(f"Unexpected archive layout: no COLMAP.bat under {extracted}")


def _verify(dest: Path) -> int:
    executable = dest / "COLMAP.bat"
    if not executable.is_file():
        executable = dest / "bin" / "colmap.exe"
    usable, output, returncode = _probe(executable)
    if usable:
        banner = next((line for line in output.splitlines() if "COLMAP" in line), "").strip()
        print(f"Installed to {dest}")
        print(f"  {banner}")
        return 0
    reason = describe_start_failure(returncode)
    print(f"Installed to {dest}, but it does not start: {reason or output.strip()[-300:]}")
    if reason and "blocked" in reason:
        print("See 'COLMAP was found but could not start' under Troubleshooting in README.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
