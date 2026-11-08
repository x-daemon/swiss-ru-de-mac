"""API keys: from environment variables, or saved in the system keychain (Windows
Credential Manager / macOS Keychain / Secret Service) via the Settings dialog."""

from __future__ import annotations

import os

SERVICE = "swiss-ru-de"
ENV = {"anthropic": "ANTHROPIC_API_KEY", "deepl": "DEEPL_AUTH_KEY", "azure": "AZURE_SPEECH_KEY",
       "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "grok": "XAI_API_KEY"}
ALT_ENV = {"gemini": "GOOGLE_API_KEY"}


def get_key(name: str) -> str | None:
    value = os.environ.get(ENV[name]) or os.environ.get(ALT_ENV.get(name, ""), "")
    if value:
        return value
    if name == "anthropic" and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return None  # the SDK picks up the token itself
    try:
        import keyring

        return keyring.get_password(SERVICE, name) or None
    except Exception:  # no keyring backend available
        return None


def has_key(name: str) -> bool:
    if name == "anthropic" and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return bool(get_key(name))


def set_key(name: str, value: str) -> bool:
    """Save (or with an empty value: delete) a key. Returns False if no keychain works."""
    try:
        import keyring

        if value:
            keyring.set_password(SERVICE, name, value)
        else:
            try:
                keyring.delete_password(SERVICE, name)
            except keyring.errors.PasswordDeleteError:
                pass
        return True
    except Exception:
        return False


def saved_in_keychain(name: str) -> bool:
    try:
        import keyring

        return bool(keyring.get_password(SERVICE, name))
    except Exception:
        return False
