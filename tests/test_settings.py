import json

from swiss_ru_de.asr import DIALECT_PROMPTS, Transcriber, _echoes_prompt, dialect_prompt
from swiss_ru_de.config import DIALECTS
from swiss_ru_de.output import line_texts
from swiss_ru_de.pipeline import Line
from swiss_ru_de.settings import Settings, load_settings, save_settings
from swiss_ru_de.translate import Passthrough, TranslatorHub


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "s" / "settings.json"
    s = Settings(target="de", top="german", font_size=24, device="Headset (Jabra)")
    save_settings(s, path)
    assert load_settings(path) == s


def test_settings_survive_bad_files(tmp_path):
    path = tmp_path / "settings.json"
    assert load_settings(path) == Settings()  # missing
    path.write_text("{not json", encoding="utf-8")
    assert load_settings(path) == Settings()
    path.write_text(json.dumps({"target": "de", "font_size": "huge", "unknown": 1}), encoding="utf-8")
    s = load_settings(path)
    assert s.target == "de" and s.font_size == Settings().font_size  # wrong type ignored


def test_every_dialect_has_a_prompt():
    assert set(DIALECTS) == set(DIALECT_PROMPTS)
    assert dialect_prompt("basel") != dialect_prompt("bern")
    assert dialect_prompt(None) == dialect_prompt("nonexistent") == DIALECT_PROMPTS["auto"]


def test_prompt_echo_is_dropped():
    prompt = dialect_prompt("zurich")
    assert _echoes_prompt("Hüt isch es schöns Wätter.", prompt)
    assert not _echoes_prompt("Mir gönd morn uf Grindelwald.", prompt)

    from types import SimpleNamespace

    import numpy as np

    t = Transcriber.__new__(Transcriber)
    t.hint, t.beam_size, t.prompt = None, 1, prompt
    seg = SimpleNamespace(text=" Grüezi mitenand.", no_speech_prob=0.1)
    t.model = SimpleNamespace(transcribe=lambda audio, **kw: (iter([seg]), None))
    assert t.transcribe(np.zeros(16000, np.float32)) == ""


def test_line_texts_with_dialect():
    line = Line(0, 0, 1, "Wir gehen heute.", target="ru", translation="Мы идём сегодня.",
                dialect="Mir gönd hüt.", translation_pending=False)
    assert line_texts(line, "ru") == ["Mir gönd hüt.", "Мы идём сегодня."]
    assert line_texts(line, "both") == ["Mir gönd hüt.", "Wir gehen heute.", "Мы идём сегодня."]


def test_translator_hub_caches_and_reconfigures(monkeypatch):
    from swiss_ru_de import translate

    made = []

    def fake_make(kind, target, claude_model, offline_model, dialect, log, models=None):
        made.append((kind, dialect))
        return Passthrough()

    monkeypatch.setattr(translate, "make_translator", fake_make)
    hub = TranslatorHub("offline", dialect="auto")
    hub.get("ru"), hub.get("ru")
    assert made == [("offline", "auto")]
    hub.configure(kind="claude", dialect="wallis")
    hub.get("ru")
    assert made[-1] == ("claude", "wallis")
    assert isinstance(hub.get("de"), Passthrough) and len(made) == 2
