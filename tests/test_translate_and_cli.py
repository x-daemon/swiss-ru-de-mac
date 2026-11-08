from types import SimpleNamespace

import pytest

from swiss_ru_de import translate
from swiss_ru_de.cli import build_parser
from swiss_ru_de.translate import ClaudeTranslator, Passthrough, TranslationError, make_translator


def test_german_target_needs_no_translator(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert isinstance(make_translator("auto", "de"), Passthrough)


def test_auto_picks_claude_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    t = make_translator("auto", "ru", dialect="berner-oberland")
    assert isinstance(t, ClaudeTranslator)
    assert t.model == "claude-opus-5-5"
    assert "Grindelwald" in t.system


def test_auto_picks_deepl_then_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DEEPL_AUTH_KEY", "abc:fx")
    assert isinstance(make_translator("auto", "ru"), translate.DeepLTranslator)
    monkeypatch.delenv("DEEPL_AUTH_KEY")
    seen = {}
    monkeypatch.setattr(translate, "OfflineTranslator", lambda model, log: seen.setdefault("m", model))
    make_translator("auto", "ru", offline_model="m1")
    assert seen["m"] == "m1"


def test_deepl_without_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    with pytest.raises(TranslationError, match="DeepL key in Settings"):
        make_translator("deepl", "ru")


class FakeMessages:
    def __init__(self, stop_reason="end_turn",
                 text='{"german": "Hoi zäme, zwei Gipfeli bitte.", "russian": "Привет всем, два круассана, пожалуйста."}'):
        self.kwargs = None
        self.stop_reason = stop_reason
        self.text = text

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.text)],
        )


def test_claude_corrects_and_translates_in_one_json_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    t = ClaudeTranslator()
    fake = FakeMessages()
    t.client = SimpleNamespace(beta=SimpleNamespace(messages=fake), messages=None)
    fixed, russian = t.process("Hoi zäme, zwei Gipfel bitte.", "Hoi zäme, zwöi Gipfeli bitte.",
                               ["Guten Morgen"], "ru")
    assert fixed == "Hoi zäme, zwei Gipfeli bitte." and russian.startswith("Привет всем")
    kw = fake.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["effort"] == "low"
    schema = kw["output_config"]["format"]["schema"]
    assert schema["required"] == ["german", "russian"] and schema["additionalProperties"] is False
    content = kw["messages"][0]["content"]
    assert "Guten Morgen" in content
    assert "Recognised: Hoi zäme, zwei Gipfel bitte." in content
    assert content.endswith("As heard: Hoi zäme, zwöi Gipfeli bitte.")


def test_claude_german_only_target_corrects_without_translating(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    t = ClaudeTranslator(model="claude-haiku-5-5")
    fake = FakeMessages(text='{"german": "Ich habe zwei Gipfeli gegessen."}')
    t.client = SimpleNamespace(beta=None, messages=fake)
    assert t.process("Ich habe zwei Gipfel gegessen.", None, (), "de") == ("Ich habe zwei Gipfeli gegessen.", None)
    assert "fallbacks" not in fake.kwargs  # Haiku: plain endpoint
    assert fake.kwargs["output_config"]["format"]["schema"]["required"] == ["german"]
    assert "As heard" not in fake.kwargs["messages"][0]["content"]


def test_claude_translate_still_works(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    t = ClaudeTranslator()
    t.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    assert t.translate("Hoi zäme") == "Привет всем, два круассана, пожалуйста."


def test_claude_refusal_or_bad_json_returns_empty(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    t = ClaudeTranslator()
    t.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages("refusal", "")))
    assert t.process("x") == (None, "")
    t.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(text="not json")))
    assert t.process("x", target="de") == (None, None)


def test_hub_uses_claude_for_german_only_when_claude(monkeypatch):
    from swiss_ru_de.translate import TranslatorHub

    hub = TranslatorHub("offline")
    assert isinstance(hub.get("de"), Passthrough)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hub = TranslatorHub("auto")
    assert isinstance(hub.get("de"), ClaudeTranslator)
    assert hub.get("de") is hub.get("ru")  # one client for both


def test_cli_defaults_and_flags():
    p = build_parser()
    a = p.parse_args([])
    assert (a.to, a.translator, a.model, a.url, a.file, a.mic) == ("ru", "auto", "swiss-turbo", None, None, False)
    a = p.parse_args(["--to", "both", "--url", "https://youtu.be/x", "--dialect", "basel", "--cpu"])
    assert a.to == "both" and a.url and a.dialect == "basel" and a.cpu
    with pytest.raises(SystemExit):
        p.parse_args(["--url", "u", "--file", "f"])  # only one input
    with pytest.raises(SystemExit):
        p.parse_args(["--cpu", "--gpu"])
