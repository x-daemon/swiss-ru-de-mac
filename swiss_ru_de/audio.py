"""Audio input: PC speaker loopback (Teams, YouTube in the browser, any app),
microphone / virtual input, and audio downloaded from a URL."""

from __future__ import annotations

import importlib
import sys
import threading
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE


S_FALSE_ERROR = "0x100000001"  # how soundcard reports S_FALSE ("COM already initialised")


def _ole32():
    import ctypes

    return ctypes.windll.ole32


def _ensure_com() -> None:
    """Windows: join the multithreaded COM apartment in this thread (needed for WASAPI).
    Harmless if the thread already is initialised (returns S_FALSE, which we ignore)."""
    if sys.platform == "win32":
        _ole32().CoInitializeEx(None, 0)  # COINIT_MULTITHREADED


def _import_soundcard():
    """Import soundcard and make COM usable in the calling thread.

    On Windows, soundcard initialises COM itself when it is first imported, and treats
    S_FALSE ("already initialised in this thread") as a fatal error ("Error 0x100000001").
    So COM must not be initialised before that first import; other threads join it after.
    """
    try:
        soundcard = importlib.import_module("soundcard")
    except RuntimeError as e:
        if sys.platform != "win32" or S_FALSE_ERROR not in str(e):
            raise _no_audio(e) from e
        # Something else already initialised COM in this thread: give back that one
        # reference so soundcard's own initialisation succeeds, then import again.
        _ole32().CoUninitialize()
        try:
            soundcard = importlib.import_module("soundcard")
        except Exception as e2:
            raise _no_audio(e2) from e2
    except Exception as e:  # no PulseAudio/PipeWire/WASAPI/CoreAudio
        raise _no_audio(e) from e
    _ensure_com()  # the module may have been imported by another thread
    return soundcard


def _no_audio(e: Exception) -> RuntimeError:
    hint = " Is PulseAudio/PipeWire running?" if sys.platform.startswith("linux") else ""
    return RuntimeError(f"Could not open the audio system.{hint} ({e!r})")


def list_devices() -> str:
    sc = _import_soundcard()
    lines = ["Speakers - capture what plays on them (default: *):"]
    default = sc.default_speaker()
    for s in sc.all_speakers():
        lines.append(f"  {'*' if s.id == default.id else ' '} {s.name}")
    lines.append("")
    lines.append("Microphones / virtual inputs - use with --mic (default: *):")
    try:
        default_mic = sc.default_microphone()
    except Exception:
        default_mic = None
    for m in sc.all_microphones(include_loopback=False):
        mark = "*" if default_mic is not None and m.id == default_mic.id else " "
        lines.append(f"  {mark} {m.name}")
    lines.append("")
    lines.append('Pick one with --device "part of the name".')
    return "\n".join(lines)


def device_names() -> tuple[list[str], list[str]]:
    """(speakers, microphones) for the settings dialog; empty lists if unavailable."""
    result: tuple[list[str], list[str]] = ([], [])

    def work():  # own thread: keeps COM away from the GUI thread on Windows
        try:
            sc = _import_soundcard()
            result[0].extend(s.name for s in sc.all_speakers())
            result[1].extend(m.name for m in sc.all_microphones(include_loopback=False))
        except Exception:
            pass

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout=5)
    return result


class SoundcardSource:
    """Streams 16 kHz mono float32 blocks from a speaker (loopback) or a microphone.

    Loopback works on Windows (WASAPI) and Linux (PulseAudio/PipeWire monitor).
    On macOS install BlackHole and use ``--mic --device BlackHole``.
    """

    def __init__(self, loopback: bool = True, device: str | None = None, block_s: float = 0.1):
        self.loopback = loopback
        self.device = device
        self.block = int(SAMPLE_RATE * block_s)
        self.name = "?"

    def _open(self, sc):
        if self.loopback:
            speaker = sc.get_speaker(self.device) if self.device else sc.default_speaker()
            self.name = f"PC audio: {speaker.name}"
            return sc.get_microphone(id=str(speaker.name), include_loopback=True)
        mic = sc.get_microphone(self.device) if self.device else sc.default_microphone()
        self.name = f"input: {mic.name}"
        return mic

    def run(self, emit: Callable[[np.ndarray], None], stop: threading.Event,
            on_open: Callable[[str], None] | None = None) -> None:
        # Imported here so that on Windows COM is initialised in the capture thread.
        sc = _import_soundcard()
        warnings.filterwarnings("ignore", message="data discontinuity")
        mic = self._open(sc)
        # Windows: soundcard asks WASAPI for a buffer of one device period (~10 ms) unless
        # told otherwise. Whenever this thread is a little late (the speech model, the window
        # and the translation all run in this process), audio was lost - words missing and
        # the stream clock falling behind. Half a second of buffer avoids that.
        extra = {"blocksize": SAMPLE_RATE // 2} if sys.platform == "win32" else {}
        with mic.recorder(samplerate=SAMPLE_RATE, **extra) as rec:
            if on_open:
                on_open(self.name)
            while not stop.is_set():
                data = rec.record(numframes=self.block)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                emit(np.ascontiguousarray(data, dtype=np.float32))


def download_audio(url: str, out_dir: Path) -> tuple[Path, str]:
    """Download the audio track of a YouTube (or any yt-dlp supported) video."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)), info.get("title") or url
