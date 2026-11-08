"""Settings of the window app, stored as JSON in the user's config folder."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import DEFAULT_DIALECT_MODEL, DEFAULT_MODEL, DEFAULT_OFFLINE_MODEL


@dataclass
class Settings:
    target: str = "ru"  # bottom pane: "ru" (Russian) or "de" (Standard German)
    top: str = "dialect"  # top pane: "dialect" (as spoken) or "german" (Standard German)
    translator: str = "auto"  # auto | claude | deepl | offline
    dialect: str = "auto"  # dialect hint, see config.DIALECTS
    font_size: int = 18
    always_on_top: bool = False
    source: str = "loopback"  # "loopback" (PC audio) or "mic"
    device: str = ""  # speaker / microphone name, "" = system default
    model: str = DEFAULT_MODEL  # applies at next start
    dialect_model: str = DEFAULT_DIALECT_MODEL  # applies at next start
    compute: str = "auto"  # auto | mlx (Apple GPU) | cpu | cuda, applies at next start
    recognizer: str = "local"  # "local" (Whisper on this PC) or "azure" (cloud, de-CH)
    speed: str = "auto"  # auto | fast | precise - see tuning.py
    azure_region: str = "switzerlandnorth"
    claude_model: str = ""
    openai_model: str = ""  # "" = provider default (translate.PROVIDERS)
    gemini_model: str = ""
    grok_model: str = ""
    local_model: str = ""  # Ollama / LM Studio model, "" = first installed of gemma3:4b, gemma3:12b, ...
    local_url: str = "http://localhost:11434/v1"
    offline_model: str = DEFAULT_OFFLINE_MODEL
    geometry: str = ""  # window size/position, remembered


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "swiss-ru-de"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    known = {f.name: f.type for f in fields(Settings)}
    defaults = Settings()
    values = {}
    for k, v in data.items() if isinstance(data, dict) else ():
        if k in known and type(v) is type(getattr(defaults, k)):
            values[k] = v
    return Settings(**values)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
