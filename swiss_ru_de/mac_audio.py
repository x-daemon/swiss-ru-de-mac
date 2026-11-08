"""System audio on macOS: everything the Mac plays (Teams, Zoom, YouTube, …).

macOS has no "speaker loopback" like Windows. The small helper ``audiotap`` (Swift,
``mac/audiotap.swift``, built by the installer) records the system audio with Apple's
ScreenCaptureKit and writes 16 kHz mono float32 samples to its standard output, which this
source reads. You keep hearing the audio normally - no virtual audio device, no change of
the sound output.

The first time, macOS asks for "Screen & System Audio Recording" permission (System
Settings → Privacy & Security) for the app (or Terminal) that started it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np

HELPER = "audiotap"
ROOT = Path(__file__).resolve().parent.parent
PERMISSION_HELP = ("macOS did not allow recording the system audio. Open System Settings → "
                   "Privacy & Security → Screen & System Audio Recording, switch on "
                   "\"Swiss RU-DE\" (or Terminal), then start swiss-ru-de again.")


def helper_path() -> str | None:
    for candidate in (ROOT / "bin" / HELPER, ROOT / "mac" / ".build" / HELPER):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(HELPER)


class SystemAudioSource:
    """Same interface as audio.SoundcardSource."""

    def __init__(self, command: list[str] | None = None, block_s: float = 0.1):
        helper = helper_path()
        self.command = command or ([helper] if helper else None)
        self.block_bytes = int(16000 * block_s) * 4
        self.name = "Mac system audio"

    def run(self, emit: Callable[[np.ndarray], None], stop: threading.Event,
            on_open: Callable[[str], None] | None = None) -> None:
        if not self.command:
            raise RuntimeError("The system-audio helper is not built. Run install-mac.command "
                               "again (it needs Apple's command line tools).")
        proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errors: list[str] = []
        threading.Thread(target=lambda: errors.extend(
            proc.stderr.read().decode("utf-8", "replace").splitlines()), daemon=True).start()
        # end the helper when asked to stop (the read below would otherwise wait for data)
        threading.Thread(target=lambda: (stop.wait(), proc.terminate()), daemon=True).start()
        if on_open:
            on_open(self.name)
        rest = b""
        try:
            while not stop.is_set():
                chunk = proc.stdout.read1(self.block_bytes) if hasattr(proc.stdout, "read1") \
                    else proc.stdout.read(self.block_bytes)
                if not chunk:
                    break
                data = rest + chunk
                usable = len(data) - len(data) % 4  # a sample can be split across reads
                rest = data[usable:]
                if usable:
                    emit(np.frombuffer(data[:usable], dtype=np.float32).copy())
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if stop.is_set():
            return
        message = " ".join(errors[-3:]).strip()
        if proc.returncode == 3 or "permission" in message.lower() or "declined" in message.lower():
            raise RuntimeError(PERMISSION_HELP)
        raise RuntimeError(f"System audio stopped ({message or f'exit code {proc.returncode}'}).")


def blackhole_device() -> str | None:
    """Name of a BlackHole virtual input, if one is installed (alternative to audiotap)."""
    try:
        from .audio import _import_soundcard

        sc = _import_soundcard()
        for mic in sc.all_microphones():
            if "blackhole" in mic.name.lower():
                return mic.name
    except Exception:
        pass
    return None


def make_source(loopback: bool = True, device: str | None = None):
    """The audio source for this Mac: system audio (audiotap, or BlackHole), or an input."""
    from .audio import SoundcardSource

    if not loopback:
        return SoundcardSource(loopback=False, device=device)
    if helper_path():
        return SystemAudioSource()
    name = blackhole_device()
    if name:
        return SoundcardSource(loopback=False, device=name)
    return SystemAudioSource()  # explains how to build the helper when started
