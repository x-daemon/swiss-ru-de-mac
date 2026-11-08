"""Adapt live recognition to the speed of this PC, and describe its hardware.

Every Whisper pass costs about the same (a 30 s audio window, however short the speech),
so the time of one pass, measured at startup, decides what is affordable live:

* fast hardware (GPU):  wider search (beam 5) and the dialect line
* medium:               dialect line only when the PC is idle, longer segments
* slow CPU:             no dialect line (it doubles the work and holds up the next
                        sentence), segments long enough that recognition keeps up
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

SPEEDS = ("auto", "fast", "precise")


@dataclass
class Tuning:
    profile: str  # "precise" | "balanced" | "fast"
    beam: int
    dialect: bool  # run the dialect line (2nd model pass)
    min_segment_s: float

    def describe(self, pass_s: float) -> str:
        what = {
            "precise": "precise mode (beam 5, dialect line on)",
            "balanced": "balanced mode (dialect line when the PC is idle)",
            "fast": "fast mode (dialect line paused, longer segments)",
        }[self.profile]
        return f"{pass_s:.1f} s per speech-model pass → {what}"


def choose(speed: str, pass_s: float, device: str = "cpu") -> Tuning:
    """speed: "auto" | "fast" | "precise"; pass_s: measured seconds per Whisper pass."""
    long_segments = round(min(8.0, max(3.0, 2 * pass_s)), 1)
    if speed == "fast":
        return Tuning("fast", 1, False, long_segments)
    if speed == "precise":
        return Tuning("precise", 5 if (device in ("cuda", "mlx") or pass_s < 1.0) else 2, True, 3.0)
    if pass_s <= 1.0:
        return Tuning("precise", 5, True, 2.0)
    if pass_s <= 2.5:
        return Tuning("balanced", 1, True, long_segments)
    return Tuning("fast", 1, False, long_segments)


# --------------------------------------------------------------------- hardware
def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:
        return ""


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    return platform.processor() or "unknown CPU"


def gpu_names() -> list[str]:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
    elif sys.platform.startswith("linux"):
        out = "\n".join(l.split(":", 2)[-1] for l in _run(["lspci"]).splitlines()
                        if "VGA" in l or "3D controller" in l)
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])  # Apple M-chip: GPU on the chip
        out = f"{out.strip()} GPU" if out.strip().startswith("Apple") else ""
    else:
        out = ""
    return [l.strip() for l in out.splitlines() if l.strip()]


def cuda_devices() -> int:
    try:
        from .asr import add_nvidia_libraries

        add_nvidia_libraries()
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def report(pass_s: float | None = None, device: str | None = None) -> list[str]:
    """Hardware summary plus concrete advice for making it faster."""
    gpus = gpu_names()
    lines = [f"CPU: {cpu_name()} ({os.cpu_count()} threads)",
             "Graphics: " + (", ".join(gpus) if gpus else "not detected")]
    if pass_s is not None:
        lines.append(f"Speech model: {pass_s:.1f} s per pass on {(device or '?').upper()} "
                     f"(real time needs less than about half the length of a sentence)")
    nvidia = any("nvidia" in g.lower() for g in gpus) or shutil.which("nvidia-smi")
    if device == "mlx":
        lines.append("✓ The Apple GPU is used (MLX) - this is the fastest setup on a Mac.")
    elif device == "cuda":
        lines.append("✓ The NVIDIA GPU is used - this is the fastest local setup.")
    elif nvidia:
        if cuda_devices():
            lines.append("→ An NVIDIA GPU is usable: set Settings → Run on → cuda (or auto).")
        else:
            lines.append("→ NVIDIA GPU found but its CUDA libraries are missing: run "
                         "start-windows.bat again (installs them), or pip install -e \".[gpu]\". "
                         "Expect about 10x faster recognition.")
    elif sys.platform == "darwin":
        lines.append("→ Speech recognition runs on the processor. On an Apple Silicon Mac, run "
                     "install-mac.command again (installs MLX for the Apple GPU).")
    elif any("intel" in g.lower() for g in gpus):
        lines.append("→ Only Intel graphics: CTranslate2 cannot use it. An OpenVINO version could "
                     "run several times faster on Intel Arc/Iris Xe/NPU (not built in yet).")
    elif any(("amd" in g.lower() or "radeon" in g.lower()) for g in gpus):
        lines.append("→ AMD graphics: not usable by CTranslate2 on Windows.")
    if pass_s is not None and device not in ("cuda", "mlx") and pass_s > 2.5:
        lines.append("→ On this CPU: keep the top part on Standard German (no 2nd model), and "
                     "close other heavy programs.")
    return lines
