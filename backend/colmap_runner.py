"""Discovery and execution of the COLMAP command-line interface.

Two jobs:

1. **Find COLMAP.** PATH first, then a list of conventional install directories, then an
   explicit override (``COLMAP_PATH`` env var or ``colmap_path.txt`` next to ``app.py``).
   Windows ships COLMAP as ``COLMAP.bat`` alongside ``colmap.exe``; both are handled.

2. **Build commands that the installed version actually understands.** COLMAP option names
   drift between releases, so rather than hardcoding flags we parse ``colmap <cmd> -h`` once,
   cache the set of supported ``--Section.Option`` tokens, and silently drop anything the
   local build does not advertise. A missing tuning flag degrades quality slightly; a wrong
   tuning flag aborts the whole pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import config

logger = logging.getLogger(__name__)

#: Matches `--Foo.bar` / `--image_path` style option names in `-h` output.
_OPTION_RE = re.compile(r"--([A-Za-z_][\w.]*)")
_VERSION_RE = re.compile(r"COLMAP\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

#: Suppress the console window COLMAP would otherwise flash on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

LogSink = Callable[[str], None]


class ColmapNotFoundError(RuntimeError):
    """Raised when no COLMAP executable could be located."""


class ColmapCommandError(RuntimeError):
    """A COLMAP subprocess exited non-zero."""

    def __init__(self, command: str, returncode: int, tail: str) -> None:
        super().__init__(f"colmap {command} exited with code {returncode}")
        self.command = command
        self.returncode = returncode
        self.tail = tail


@dataclass
class ColmapInfo:
    """What we know about the local COLMAP install, surfaced by ``GET /api/system``."""

    available: bool
    path: str | None = None
    version: str | None = None
    source: str | None = None
    gpu: str = "unknown"
    #: Official Windows builds ship in CUDA and no-CUDA flavours; the banner says which.
    gpu_build: bool = True
    searched: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "path": self.path,
            "version": self.version,
            "source": self.source,
            "gpu": self.gpu,
            "gpuBuild": self.gpu_build,
            "searched": self.searched,
            "error": self.error,
        }


def _candidate_paths() -> list[tuple[Path, str]]:
    """Every location we are willing to look in, paired with a human-readable source label."""
    candidates: list[tuple[Path, str]] = []

    override = os.environ.get("COLMAP_PATH", "").strip()
    if override:
        candidates.append((Path(override), "COLMAP_PATH environment variable"))

    if config.COLMAP_PATH_FILE.exists():
        try:
            stored = config.COLMAP_PATH_FILE.read_text(encoding="utf-8").strip()
            if stored:
                candidates.append((Path(stored), "colmap_path.txt"))
        except OSError as exc:  # pragma: no cover - unreadable config file
            logger.warning("Could not read %s: %s", config.COLMAP_PATH_FILE, exc)

    for name in config.COLMAP_EXE_NAMES:
        found = shutil.which(name)
        if found:
            candidates.append((Path(found), "PATH"))

    for directory in config.COLMAP_SEARCH_DIRS:
        base = Path(directory)
        # Releases are commonly extracted as <dir>/ or <dir>/bin/.
        for sub in (base, base / "bin"):
            for name in config.COLMAP_EXE_NAMES:
                candidates.append((sub / name, f"search path ({sub})"))

    return candidates


#: Windows exit codes (NTSTATUS) that mean "the file is there but was not allowed or able to
#: start", worth telling the user about instead of "not found".
_START_FAILURES = {
    0xC0E90002: "Windows blocked it (Smart App Control or an application control policy)",
    0xC0000135: "a DLL it needs is missing",
    0xC000007B: "a DLL it needs is the wrong architecture",
    0xC0000022: "Windows denied access to it",
}


def describe_start_failure(returncode: int | None) -> str | None:
    """Plain-language reason for a Windows start-up failure code, or None if not one."""
    if returncode is None:
        return None
    reason = _START_FAILURES.get(returncode & 0xFFFFFFFF)
    return f"{reason} (exit code 0x{returncode & 0xFFFFFFFF:08X})" if reason else None


def _probe(executable: Path) -> tuple[bool, str, int | None]:
    """Run ``<exe> -h``: (looks_like_colmap, combined output, exit code or None)."""
    try:
        completed = subprocess.run(
            [str(executable), "-h"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc), None

    output = f"{completed.stdout}\n{completed.stderr}"
    usable = "feature_extractor" in output or "COLMAP" in output
    return usable, output, completed.returncode


def _detect_gpu() -> str:
    """Best-effort GPU label. Never raises, never blocks for long."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=_NO_WINDOW,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


class ColmapRunner:
    """Locates COLMAP and runs its sub-commands as async subprocesses."""

    def __init__(self) -> None:
        self._executable: Path | None = None
        self._info: ColmapInfo | None = None
        self._option_cache: dict[str, set[str]] = {}
        self._gpu: str | None = None
        #: Set when a GPU stage failed; later runs on this server stay on the CPU.
        self.gpu_failed = False

    # -- discovery ------------------------------------------------------------

    def detect(self, force: bool = False) -> ColmapInfo:
        """Locate COLMAP, caching the answer. ``force=True`` re-runs the search."""
        if self._info is not None and not force:
            return self._info

        if force:
            self._option_cache.clear()
            self._executable = None
            self.gpu_failed = False

        searched: list[str] = []
        seen: set[str] = set()
        #: Candidates that exist but would not start, with the reason.
        blocked: list[str] = []

        for path, source in _candidate_paths():
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            searched.append(str(path))

            if not path.exists() or path.is_dir():
                continue

            usable, output, returncode = _probe(path)
            if not usable:
                reason = describe_start_failure(returncode)
                if reason:
                    blocked.append(f"{path}: {reason}")
                    logger.warning("COLMAP at %s could not start: %s", path, reason)
                else:
                    logger.debug("Rejected COLMAP candidate %s", path)
                continue

            match = _VERSION_RE.search(output)
            self._executable = path
            if self._gpu is None:
                self._gpu = _detect_gpu()
            # e.g. "COLMAP 4.2.0 (Commit ... without GPU support)"
            gpu_build = "without gpu support" not in output.lower()
            self._info = ColmapInfo(
                available=True,
                path=str(path),
                version=match.group(1) if match else "unknown",
                source=source,
                gpu=self._gpu if gpu_build else "CPU-only build",
                gpu_build=gpu_build,
                searched=searched,
            )
            logger.info("COLMAP found at %s (version %s)", path, self._info.version)
            return self._info

        if self._gpu is None:
            self._gpu = _detect_gpu()
        self._info = ColmapInfo(
            available=False,
            gpu=self._gpu,
            searched=searched,
            error=(
                "COLMAP was found but could not start: " + "; ".join(blocked)
                if blocked
                else "No COLMAP executable found on PATH or in the standard install locations."
            ),
        )
        logger.warning("COLMAP not found. Searched %d locations.", len(searched))
        return self._info

    def set_path(self, path: str) -> ColmapInfo:
        """Persist a user-supplied executable path and re-detect."""
        candidate = Path(path.strip().strip('"'))
        if not candidate.exists():
            raise ColmapNotFoundError(f"Path does not exist: {candidate}")

        if candidate.is_dir():
            resolved: Path | None = None
            for name in config.COLMAP_EXE_NAMES:
                for sub in (candidate, candidate / "bin"):
                    if (sub / name).is_file():
                        resolved = sub / name
                        break
                if resolved is not None:
                    break
            if resolved is None:
                raise ColmapNotFoundError(f"No COLMAP executable inside {candidate}")
            candidate = resolved

        usable, _, returncode = _probe(candidate)
        if not usable:
            reason = describe_start_failure(returncode)
            raise ColmapNotFoundError(
                f"{candidate} could not start: {reason}"
                if reason
                else f"{candidate} does not look like a COLMAP executable."
            )

        config.COLMAP_PATH_FILE.write_text(str(candidate), encoding="utf-8")
        return self.detect(force=True)

    @property
    def executable(self) -> Path:
        info = self.detect()
        if not info.available or self._executable is None:
            raise ColmapNotFoundError(info.error or "COLMAP not available")
        return self._executable

    @property
    def available(self) -> bool:
        return self.detect().available

    # -- option introspection --------------------------------------------------

    def supported_options(self, command: str) -> set[str]:
        """Parse ``colmap <command> -h`` once and cache the option names it advertises."""
        if command in self._option_cache:
            return self._option_cache[command]

        options: set[str] = set()
        try:
            completed = subprocess.run(
                [str(self.executable), command, "-h"],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
                creationflags=_NO_WINDOW,
            )
            options = {
                match.group(1)
                for match in _OPTION_RE.finditer(f"{completed.stdout}\n{completed.stderr}")
            }
        except (OSError, subprocess.SubprocessError, ColmapNotFoundError) as exc:
            logger.warning("Could not introspect 'colmap %s -h': %s", command, exc)

        self._option_cache[command] = options
        logger.debug("colmap %s advertises %d options", command, len(options))
        return options

    def filter_options(
        self, command: str, options: Sequence[tuple[str | Sequence[str], object]]
    ) -> list[str]:
        """Turn ``[(name_or_aliases, value), ...]`` into argv for the installed COLMAP.

        Option names drift between releases - COLMAP 4.x renamed ``SiftExtraction.use_gpu`` to
        ``FeatureExtraction.use_gpu``, ``Mapper.ba_global_images_ratio`` to
        ``...ba_global_frames_ratio``, and so on. Each entry therefore carries every spelling we
        know of, and the first one this build advertises wins.

        If introspection produced nothing (unparseable help text), the first spelling is kept:
        we would rather attempt the documented flag than silently run an untuned pipeline.
        """
        supported = self.supported_options(command)
        argv: list[str] = []

        for name, value in options:
            aliases = (name,) if isinstance(name, str) else tuple(name)
            if not supported:
                argv.extend([f"--{aliases[0]}", str(value)])
                continue

            chosen = next((alias for alias in aliases if alias in supported), None)
            if chosen is None:
                logger.info(
                    "Dropping option %s - 'colmap %s' advertises none of these spellings",
                    "/".join(aliases),
                    command,
                )
                continue
            argv.extend([f"--{chosen}", str(value)])

        return argv

    # -- execution -------------------------------------------------------------

    def build_command(self, command: str, args: Iterable[str]) -> list[str]:
        return [str(self.executable), command, *args]

    async def run(
        self,
        command: str,
        args: Sequence[str],
        log_sink: LogSink | None = None,
        line_callback: Callable[[str], None] | None = None,
        timeout: float = config.COLMAP_STAGE_TIMEOUT_S,
    ) -> str:
        """Run one COLMAP sub-command, streaming output. Never blocks the event loop."""
        argv = self.build_command(command, args)
        logger.info("Running: %s", " ".join(argv))
        if log_sink:
            log_sink(f"\n$ {' '.join(argv)}\n")

        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=_NO_WINDOW,
        )

        tail: list[str] = []

        async def pump() -> None:
            assert process.stdout is not None
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if log_sink:
                    log_sink(line + "\n")
                if line_callback:
                    try:
                        line_callback(line)
                    except Exception:  # pragma: no cover - a callback must not kill the pump
                        logger.exception("progress callback failed")
                tail.append(line)
                if len(tail) > 60:
                    del tail[0]

        try:
            await asyncio.wait_for(pump(), timeout=timeout)
            returncode = await asyncio.wait_for(process.wait(), timeout=60)
        except asyncio.TimeoutError:
            await _kill_tree(process)
            raise ColmapCommandError(command, -1, "\n".join(tail[-20:]) + "\n[timed out]")
        except asyncio.CancelledError:
            await _kill_tree(process)
            raise

        if returncode != 0:
            raise ColmapCommandError(command, returncode, "\n".join(tail[-20:]))
        return "\n".join(tail)


async def _kill_tree(process: asyncio.subprocess.Process) -> None:
    """Stop a COLMAP run *and its children*.

    On Windows COLMAP is launched through ``COLMAP.bat``, so ``process`` is ``cmd.exe`` and the
    real work happens in a child ``colmap.exe``. ``process.kill()`` alone ends only the shell:
    measured here, the orphaned mapper kept a CPU core busy for another ~20 s after "cancel".
    ``taskkill /T`` takes the whole tree down.
    """
    if process.returncode is None and os.name == "nt":
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                check=False,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            logger.warning("taskkill failed for pid %s: %s", process.pid, exc)
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover - exited in between
            pass
    await process.wait()


#: Module-level singleton; detection is cheap after the first call.
runner = ColmapRunner()
