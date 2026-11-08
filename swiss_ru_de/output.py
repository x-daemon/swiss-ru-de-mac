"""Console output and transcript files (.srt / .txt)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .pipeline import Line


def _enable_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:  # turn on VT100 escape codes in the Windows console
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        pass
    return False


def setup_console() -> None:
    """Make sure Cyrillic and umlauts print even when output is redirected."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8")
            else:
                stream.reconfigure(errors="replace")
        except Exception:
            pass


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def line_texts(line: Line, target: str) -> list[str]:
    """What to show for a line: target ru -> Russian, de -> German, both -> both.
    The dialect "as spoken" line, if there is one, comes first."""
    head = [line.dialect] if line.dialect else []
    if target == "de" or line.translation is None:
        return head + [line.german]
    if target == "both":
        return head + [line.german, line.translation or "…"]
    return head + [line.translation or f"({line.german})"]


class ConsolePrinter:
    def __init__(self, target: str, stream=None):
        self.target = target
        self.stream = stream or sys.stdout
        color = _enable_ansi() if stream is None else False
        self.dim = "\033[2m" if color else ""
        self.bold = "\033[1m" if color else ""
        self.reset = "\033[0m" if color else ""
        self._lock = threading.Lock()

    def __call__(self, line: Line) -> None:
        texts = line_texts(line, self.target)
        stamp = f"{self.dim}[{clock(line.start)}]{self.reset}"
        indent = " " * (len(clock(line.start)) + 2)
        with self._lock:
            for i, text in enumerate(texts):
                style = self.bold if i == len(texts) - 1 else self.dim
                prefix = stamp if i == 0 else indent
                print(f"{prefix} {style}{text}{self.reset}", file=self.stream)
            self.stream.flush()


class TranscriptWriter:
    """Appends every line to a .srt (subtitles) or plain text file as it arrives.
    ``target=None`` uses each line's own target (it can change while running)."""

    def __init__(self, path: str, target: str | None):
        self.path = Path(path)
        self.srt = self.path.suffix.lower() == ".srt"
        self.target = target
        self.n = 0
        self._lock = threading.Lock()
        self.f = self.path.open("w", encoding="utf-8")

    def __call__(self, line: Line) -> None:
        texts = line_texts(line, self.target or line.target)
        with self._lock:
            self.n += 1
            if self.srt:
                end = max(line.end, line.start + 1.0)
                self.f.write(f"{self.n}\n{srt_time(line.start)} --> {srt_time(end)}\n")
                self.f.write("\n".join(texts) + "\n\n")
            else:
                self.f.write(f"[{clock(line.start)}] " + "\n    ".join(texts) + "\n")
            self.f.flush()

    def close(self) -> None:
        self.f.close()
