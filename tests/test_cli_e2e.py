"""Run the real CLI with the speech model and the sound card replaced by fakes."""

import numpy as np
import pytest

from swiss_ru_de import asr, audio, cli

from .test_pipeline import FakeSource, FakeTranscriber, silence, tone


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    created = {}

    def make_transcriber(model, **kw):
        created["model"], created["kw"] = model, kw
        return FakeTranscriber()

    monkeypatch.setattr(asr, "Transcriber", make_transcriber)
    sig = np.concatenate([silence(0.3), tone(1.0), silence(1.0), tone(1.0), silence(1.0)])
    monkeypatch.setattr(audio, "SoundcardSource", lambda loopback, device: FakeSource(sig))
    return created


def test_live_german(fakes, capsys, tmp_path):
    out = tmp_path / "live.txt"
    assert cli.main(["--to", "de", "--save", str(out), "--cpu", "--min-seconds", "0"]) == 0
    printed = capsys.readouterr().out
    assert "Satz 1" in printed and "Satz 2" in printed
    assert fakes["kw"]["device"] == "cpu" and fakes["kw"]["beam_size"] is None  # auto
    assert out.read_text(encoding="utf-8").count("Satz") == 2


def test_file_mode_with_translation_to_srt(fakes, monkeypatch, capsys, tmp_path):
    from swiss_ru_de import translate

    class FakeOffline(translate.Translator):
        name = "fake offline"

        def __init__(self, model=None, log=print):
            pass

        def translate(self, text, context=()):
            return "RU:" + text

    monkeypatch.setattr(translate, "OfflineTranslator", FakeOffline)
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"")
    srt = tmp_path / "talk.srt"
    assert cli.main(["--file", str(media), "--to", "both", "--save", str(srt)]) == 0
    assert fakes["kw"]["beam_size"] == 5
    assert "RU:Satz 3" in capsys.readouterr().out
    assert srt.read_text(encoding="utf-8").count("-->") == 3


def test_missing_file(fakes, tmp_path):
    assert cli.main(["--file", str(tmp_path / "nope.wav"), "--to", "de"]) == 2
