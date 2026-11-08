"""Windows COM handling around `import soundcard`, simulated so it runs on any OS."""

from types import SimpleNamespace

import pytest

from swiss_ru_de import audio


@pytest.fixture
def windows(monkeypatch):
    calls = []
    ole32 = SimpleNamespace(CoInitializeEx=lambda *a: calls.append("CoInitializeEx") or 0,
                            CoUninitialize=lambda: calls.append("CoUninitialize"))
    monkeypatch.setattr(audio.sys, "platform", "win32")
    monkeypatch.setattr(audio, "_ole32", lambda: ole32)
    return calls


def fake_importer(calls, results):
    """import_module that returns/raises the given results in order."""
    results = list(results)

    def import_module(name):
        assert name == "soundcard"
        calls.append("import")
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    return import_module


def test_com_is_not_initialised_before_first_import(windows, monkeypatch):
    module = object()
    monkeypatch.setattr(audio.importlib, "import_module", fake_importer(windows, [module]))
    assert audio._import_soundcard() is module
    # soundcard must initialise COM itself first (it fails on S_FALSE otherwise)
    assert windows == ["import", "CoInitializeEx"]


def test_recovers_when_com_was_already_initialised(windows, monkeypatch):
    module = object()
    s_false = RuntimeError("Error 0x100000001")
    monkeypatch.setattr(audio.importlib, "import_module", fake_importer(windows, [s_false, module]))
    assert audio._import_soundcard() is module
    assert windows == ["import", "CoUninitialize", "import", "CoInitializeEx"]


def test_other_errors_are_reported_clearly(windows, monkeypatch):
    monkeypatch.setattr(audio.importlib, "import_module",
                        fake_importer(windows, [OSError("no WASAPI")]))
    with pytest.raises(RuntimeError, match="Could not open the audio system"):
        audio._import_soundcard()
    assert "CoUninitialize" not in windows
