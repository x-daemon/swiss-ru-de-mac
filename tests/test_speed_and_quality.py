"""Speed (single-pass live decoding, best-effort dialect line, GPU libraries) and
translation quality (keys, Swiss words, fast offline translator)."""

from types import SimpleNamespace

import numpy as np

from swiss_ru_de import asr, keys, translate
from swiss_ru_de.pipeline import LineBook, LivePipeline, _STOP
from swiss_ru_de.segmenter import Segment
from swiss_ru_de.translate import OfflineTranslator, normalize_swiss_terms

from .test_pipeline import FakeSource, FakeTranscriber, FakeRu, silence


def test_live_decoding_is_single_pass_files_keep_fallback():
    t = asr.Transcriber.__new__(asr.Transcriber)
    t.hint, t.prompt = None, None
    assert t._options(1, live=True)["temperature"] == 0.0
    assert t._options(5)["temperature"] == [0.0, 0.2, 0.4, 0.6]


def test_dialect_line_is_skipped_while_speech_is_waiting():
    lines = []
    dialect = FakeTranscriber("Satzli")
    p = LivePipeline(FakeSource(silence(0.1)), FakeTranscriber(), FakeRu(),
                     LineBook(on_line=lines.append), lambda s: None,
                     dialect_transcriber=dialect)
    p.options.dialect = True
    for i in range(3):  # 20 s each: cannot be merged, so a backlog remains
        p.segments.put(Segment(i * 20.0, i * 20.0 + 20, np.zeros(16000, np.float32)))
    p.segments.put(_STOP)
    p.worker.start()
    p._dialect.start()
    p._asr_loop()
    p.stop()
    assert sorted(l.dialect or "" for l in lines) == ["", "", "Satzli 1"]  # only once idle
    assert all(l.translation for l in lines)  # translation never waits for it


def test_recognition_does_not_wait_for_the_dialect_line():
    import threading
    import time

    lines = []
    release = threading.Event()

    class SlowDialect(FakeTranscriber):
        def transcribe(self, audio):
            release.wait(5)  # e.g. on the processor
            return super().transcribe(audio)

    p = LivePipeline(FakeSource(silence(0.1)), FakeTranscriber(), FakeRu(),
                     LineBook(on_line=lines.append), lambda s: None,
                     dialect_transcriber=SlowDialect("Satzli"))
    p.options.dialect = True
    p.worker.start()
    p._dialect.start()
    asr = threading.Thread(target=p._asr_loop)
    asr.start()
    p.segments.put(Segment(0, 2, np.zeros(16000, np.float32)))
    time.sleep(0.3)  # the dialect pass of line 1 is now running (and blocked)
    p.segments.put(Segment(3, 5, np.zeros(16000, np.float32)))
    p.segments.put(_STOP)
    asr.join(2)
    assert not asr.is_alive()  # recognition went on while the dialect line was busy
    time.sleep(0.2)
    assert [l.german for l in lines] == ["Satz 2"]  # line 2 had no dialect pass: done at once
    release.set()
    p.stop()
    assert [l.dialect for l in sorted(lines, key=lambda l: l.id)] == ["Satzli 1", None]


def test_stats_and_lag_warning():
    from swiss_ru_de.pipeline import Stats, TranslationWorker

    stats = Stats()
    assert stats.summary() == "No sentences yet."
    stats.add("recognition", 1.0)
    stats.add("translation", 10.0)
    stats.add("translation", 20.0)
    assert stats.avg["translation"] == 13.0 and stats.slowest() == "translation"
    status = []
    worker = TranslationWorker(lambda t: FakeRu(), LineBook(), status.append, stats=stats)
    worker.origin = time_now() - 100  # the stream started 100 s ago
    line = SimpleNamespace(end=30.0)
    worker._check_speed(line, FakeRu())
    assert 69 < stats.lag < 72
    assert "70 s behind" in status[0] and "slowest step" in status[0] and "translation 13.0 s" in status[0]
    worker._check_speed(line, FakeRu())
    assert len(status) == 1  # not repeated every sentence


def time_now():
    import time

    return time.monotonic()


def test_gpu_libraries_missing_is_fine():
    assert isinstance(asr.add_nvidia_libraries(), list)


def test_keys_env_first_then_keychain(monkeypatch, fake_keyring):
    assert keys.get_key("deepl") is None and not keys.has_key("deepl")
    assert keys.set_key("deepl", "abc:fx")
    assert keys.get_key("deepl") == "abc:fx" and keys.saved_in_keychain("deepl")
    monkeypatch.setenv("DEEPL_AUTH_KEY", "from-env")
    assert keys.get_key("deepl") == "from-env"
    assert keys.set_key("deepl", "")  # remove
    assert not keys.saved_in_keychain("deepl")


def test_auto_translator_uses_saved_claude_key():
    keys.set_key("anthropic", "sk-ant-test")
    t = translate.make_translator("auto", "ru")
    assert isinstance(t, translate.ClaudeTranslator)
    assert t.client.api_key == "sk-ant-test"


def test_swiss_words_are_normalised_for_machine_translators():
    assert normalize_swiss_terms("Zmorge mit Gipfeli, dann mit dem Velo ins Spital.") == \
        "Frühstück mit Croissant, dann mit dem Fahrrad ins Krankenhaus."
    assert normalize_swiss_terms("Merci vilmal, Grüezi mitenand!") == "vielen Dank, Guten Tag!"
    assert normalize_swiss_terms("Spitalgasse und Velodrom bleiben.") == "Spitalgasse und Velodrom bleiben."


class FakeTokenizer:
    def encode(self, text, truncation=True, max_length=512):
        return [1, 2, 3]

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def convert_tokens_to_ids(self, tokens):
        self.decoded_tokens = tokens
        return [7, 8]

    def decode(self, ids, skip_special_tokens=True):
        return " Привет "


def make_offline(nllb=True):
    t = OfflineTranslator.__new__(OfflineTranslator)
    t.nllb, t.log, t.tok = nllb, lambda m: None, FakeTokenizer()
    t._torch_model = None
    return t


def test_offline_ct2_strips_target_language_token():
    t = make_offline()
    calls = {}

    def translate_batch(batch, target_prefix=None, **options):
        calls["batch"], calls["prefix"], calls["options"] = batch, target_prefix, options
        return [SimpleNamespace(hypotheses=[["rus_Cyrl", "▁При", "вет"]])]

    t._ct2 = SimpleNamespace(translate_batch=translate_batch)
    assert t.translate("Zmorge") == "Привет"
    assert calls["batch"] == [["t1", "t2", "t3"]] and calls["prefix"] == [["rus_Cyrl"]]
    assert calls["options"]["no_repeat_ngram_size"] == 3  # no "есть есть есть"
    assert t.tok.decoded_tokens == ["▁При", "вет"]


def test_offline_falls_back_to_pytorch_when_ct2_fails(monkeypatch):
    t = make_offline()

    def broken(*a, **k):
        raise RuntimeError("bad model")

    t._ct2 = SimpleNamespace(translate_batch=broken)
    monkeypatch.setattr(OfflineTranslator, "_load_torch", lambda self: None)
    monkeypatch.setattr(OfflineTranslator, "_translate_torch",
                        lambda self, sentences: [f"torch:{s}" for s in sentences])
    assert t.translate("Velo") == "torch:Fahrrad"
    assert t._ct2 is None  # stays on PyTorch afterwards


def test_translator_label_shows_what_auto_uses():
    assert translate.resolve_kind("auto") == "offline"
    keys.set_key("deepl", "abc:fx")
    assert translate.resolve_kind("auto") == "deepl"
    keys.set_key("anthropic", "sk-ant-x")
    assert translate.resolve_kind("auto") == "claude"
    assert translate.resolve_kind("offline") == "offline"
    assert "basic" in translate.QUALITY_LABEL["offline"]


def test_beam_size_auto_depends_on_device(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, path, device, compute_type, cpu_threads):
            seen["device"] = device
            self.model = SimpleNamespace(n_mels=None)

        def transcribe(self, audio, **kw):
            return iter([]), None

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(asr, "resolve_model", lambda name, log: "local")
    for device, beam in (("cpu", 1), ("cuda", 5)):
        t = asr.Transcriber("x", device=device, log=lambda m: None)
        assert t.beam_size == beam
    assert asr.Transcriber("x", device="cpu", beam_size=3, log=lambda m: None).beam_size == 3


def test_gipfeli_and_gipfel_in_food_talk_are_croissants():
    n = normalize_swiss_terms
    assert n("Zum Frühstück esse ich zwei Gipfel.") == "Zum Frühstück esse ich zwei Croissants."
    assert n("Ich habe ein Gipfeli gekauft.") == "Ich habe ein Croissant gekauft."
    assert n("Zwei Gipfelis und ein Weggli.") == "Zwei Croissants und ein Brötchen."
    assert n("Nur einen Gipfel bitte.", ["Was möchtest du zum Zmorge?"]) == "Nur einen Croissant bitte."
    assert n("Wir standen auf dem Gipfel des Eigers.") == "Wir standen auf dem Gipfel des Eigers."
    assert n("Peperoni mit Rahm") == "Paprika mit Sahne"


def test_offline_translator_uses_context_for_gipfel(monkeypatch):
    t = make_offline()
    seen = {}
    t._ct2 = None
    monkeypatch.setattr(OfflineTranslator, "_translate_torch",
                        lambda self, sentences: seen.setdefault("t", sentences))
    t.translate("Zwei Gipfel bitte.", ["Ich hätte gern einen Kaffee."])
    assert seen["t"] == ["Zwei Croissants bitte."]


def test_claude_prompt_knows_gipfel_at_breakfast():
    assert "misrecognised \"Gipfeli\"" in translate.CLAUDE_SYSTEM


def test_offline_translates_sentence_by_sentence():
    t = make_offline()
    calls = {}

    def translate_batch(batch, target_prefix=None, **options):
        calls["n"] = len(batch)
        return [SimpleNamespace(hypotheses=[["rus_Cyrl", f"S{i}"]]) for i in range(len(batch))]

    t._ct2 = SimpleNamespace(translate_batch=translate_batch)
    text = ("Und wo essen Sie nie etwas zum Morgen? Ich schätze nichts am Morgen. "
            "Müsli und Zopf mit Honig. Und du? Ein Joghurt.")
    t.translate(text)
    assert calls["n"] == 5  # nothing is dropped: every sentence is translated


def test_split_and_breakfast_rules():
    from swiss_ru_de.translate import split_sentences

    assert split_sentences("Blutorange. Ein Blutorange? Nicht mehr.") == ["Blutorange.", "Ein Blutorange?", "Nicht mehr."]
    assert split_sentences("Er kam um 3.5 Uhr. z.B. heute. Ja!") == ["Er kam um 3.5 Uhr. z.B. heute.", "Ja!"]
    assert normalize_swiss_terms("Weil ich nie zum Morgen esse.") == "Weil ich nie zum Frühstück esse."
    assert normalize_swiss_terms("Ich stehe am Morgen früh auf.") == "Ich stehe am Morgen früh auf."


def test_gipfel_compounds_and_neuter_are_croissants():
    n = normalize_swiss_terms
    assert n("Ehm, ein Gipfel, also Buttergipfel. Oder noch das Gipfel allein?") == \
        "Ehm, ein Croissant, also Buttercroissant. Oder noch das Croissant allein?"
    assert n("Zwei Schoggigipfeli und ein Kafi.") == "Zwei Schoggicroissants und ein Kaffee."
    assert n("Hast du etwas dazu getrunken? Nur ein Gipfel.") == \
        "Hast du etwas dazu getrunken? Nur ein Croissant."
    assert n("Der Gipfel ist hoch, wir trinken oben Tee.") == "Der Gipfel ist hoch, wir trinken oben Tee."
    assert n("Das Gipfeltreffen war lang.") == "Das Gipfeltreffen war lang."


def test_time_words_for_small_models():
    n = normalize_swiss_terms
    assert n("Eine Viertelstunde oder so 20 Minuten.") == "15 Minuten oder so 20 Minuten."
    assert n("Etwa eine Dreiviertelstunde, mega langsam.") == "Etwa 45 Minuten, sehr langsam."
    assert n("Wie lange braucht ihr zum Morgenessen?") == "Wie lange braucht ihr zum Frühstück?"
    assert n("Wie lange nehmen Sie sich Zeit, um morgen zu essen?") == \
        "Wie lange nehmen Sie sich Zeit, um zu frühstücken?"


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds):
    return np.full(int(seconds * 16000), 0.001, np.float32)


def test_segments_are_split_into_utterances():
    # short answer, pause, question, short breath, rest of the question
    audio = np.concatenate([tone(1.0), quiet(0.6), tone(3.0), quiet(0.15), tone(2.0)])
    clips = asr.utterance_clips(audio)
    assert len(clips) == 2  # cut in the 0.6 s pause only
    assert 1.0 * 16000 < clips[0][1] < 1.6 * 16000 and clips[1][1] == len(audio)
    assert asr.utterance_clips(tone(1.2)) == [(0, len(tone(1.2)))]
    yes = np.concatenate([tone(0.3), quiet(0.5), tone(3.0)])  # "Ja." alone is too short
    assert len(asr.utterance_clips(yes)) == 1
    many = np.concatenate([np.concatenate([tone(1.0), quiet(0.5)]) for _ in range(12)])
    assert len(asr.utterance_clips(many)) == 8  # one batch at most


def test_gpu_recognises_each_utterance_in_one_batch():
    t = asr.Transcriber.__new__(asr.Transcriber)
    t.hint, t.prompt, t.beam_size, t.log = None, None, 5, lambda m: None
    t.split_utterances = True
    seen = {}

    def batched(audio, clip_timestamps, batch_size, **kw):
        seen.update(clips=clip_timestamps, batch=batch_size, vad="vad_parameters" in kw)
        return iter([SimpleNamespace(text=" Orangensaft.", no_speech_prob=0.1),
                     SimpleNamespace(text=" Und am Wochenende?", no_speech_prob=0.1)]), None

    t._batched = SimpleNamespace(transcribe=batched)
    audio = np.concatenate([tone(1.0), quiet(0.6), tone(3.0)])
    assert t.transcribe(audio) == "Orangensaft. Und am Wochenende?"
    assert seen["batch"] == 2 and len(seen["clips"]) == 2 and not seen["vad"]

    def broken(*a, **k):
        raise TypeError("old faster-whisper")

    t._batched = SimpleNamespace(transcribe=broken)
    t.model = SimpleNamespace(transcribe=lambda audio, **kw: (
        iter([SimpleNamespace(text=" Ganzer Satz.", no_speech_prob=0.1)]), None))
    assert t.transcribe(audio) == "Ganzer Satz." and not t.split_utterances  # falls back


def test_lag_is_measured_by_the_clock():
    import time

    from swiss_ru_de.pipeline import Line, Stats, TranslationWorker

    stats = Stats()
    worker = TranslationWorker(lambda t: FakeRu(), LineBook(), lambda m: None, stats=stats)
    worker.origin = time.monotonic() - 500  # audio samples say: 470 s behind (lost audio)
    line = Line(0, 0, 30.0, "Satz", heard_at=time.monotonic() - 3)
    worker._check_speed(line, FakeRu())
    assert 2.9 < stats.lag < 4  # the real delay


def test_more_time_words():
    n = normalize_swiss_terms
    assert n("Dann geht es, ja, dreiviertel Stunde.") == "Dann geht es, ja, 45 Minuten."
    assert n("eine super Sauce") == "eine toll Sauce"
