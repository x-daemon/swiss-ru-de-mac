"""Azure cloud recognition: speech gate, partial/final text, translation, dialect line."""


import numpy as np

from swiss_ru_de.cloud import CloudPipeline, SpeechGate
from swiss_ru_de.config import SAMPLE_RATE
from swiss_ru_de.pipeline import LineBook, Options

from .test_pipeline import FakeRu, FakeTranscriber, silence, tone


def test_speech_gate_sends_speech_with_margins_but_not_silence():
    gate = SpeechGate(pre_roll_ms=300, hangover_ms=1200)
    out = gate.feed(np.concatenate([silence(5.0), tone(1.0), silence(5.0)]))
    seconds = out.size / SAMPLE_RATE
    assert 2.3 < seconds < 2.7  # 0.3 before + 1.0 speech + 1.2 after; the 9 s of silence are skipped
    assert gate.feed(silence(3.0)).size == 0


class FakeAzure:
    """Stands in for AzureSpeech; the test drives the SDK callbacks."""

    def __init__(self, key, region, on_partial, on_final, on_error, phrases=None):
        self.key, self.region, self.phrases = key, region, phrases
        self.partial, self.final, self.error = on_partial, on_final, on_error
        self.sent = 0
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def write(self, audio):
        self.sent += audio.size

    def stop(self):
        self.stopped = True


class IdleSource:
    def run(self, emit, stop, on_open=None):
        stop.wait()


def make(dialect=None, options=None):
    updates, lines = [], []
    book = LineBook(on_update=updates.append, on_line=lines.append)
    p = CloudPipeline(IdleSource(), FakeRu(), book, lambda s: None, key="k", region="switzerlandnorth",
                      options=options, dialect_transcriber=dialect, phrases=["Grindelwald"],
                      speech_factory=FakeAzure)
    return p, updates, lines


def test_partial_then_final_text_is_translated():
    p, updates, lines = make()
    assert p.speech.key == "k" and p.speech.phrases == ["Grindelwald"]
    p.worker.start()
    p._partial("Satz", 1.0)
    p._partial("Satz eins", 1.0)
    p._final("Satz eins.", 1.0, 2.5)
    p.worker.close()
    assert [u.german for u in updates][:2] == ["Satz", "Satz eins"]  # live while speaking
    assert len(lines) == 1
    assert (lines[0].german, lines[0].translation) == ("Satz eins.", "Предложение eins.")
    assert (lines[0].start, lines[0].end) == (1.0, 2.5)


def test_no_final_text_keeps_the_partial():
    p, _, lines = make()
    p.worker.start()
    p._partial("Hallo zäme", 0.0)
    p._final("", 0.0, 0.0)
    p.worker.close()
    assert [l.german for l in lines] == ["Hallo zäme"]


def test_dialect_line_from_the_audio_sent_to_azure():
    dialect = FakeTranscriber("Satzli")
    p, _, lines = make(dialect=dialect, options=Options(dialect=True))
    p._emit(np.concatenate([silence(2.0), tone(2.0), silence(2.0)]))  # gate sends ~3.5 s
    assert p.speech.sent == p._sent_total > 0
    p.worker.start()
    p._dialect.start()
    p._final("Satz eins.", 0.5, 2.5)
    p._dialect_jobs.put(None)
    p._dialect.join(5)
    p.worker.close()
    assert lines[0].dialect == "Satzli 1"
    assert dialect.calls == [2.0]  # exactly the sentence's 2 s of sent audio


def test_paused_sends_nothing():
    p, _, _ = make(options=Options(paused=True))
    p._emit(tone(2.0))
    assert p.speech.sent == 0


def test_azure_error_stops_the_pipeline():
    status = []
    p, _, _ = make()
    p.on_status = status.append
    p._fail("Azure Speech: key or region is wrong")
    assert p.stop_event.is_set() and "region" in status[0]


def test_dialect_worker_keeps_only_the_newest_sentence():
    dialect = FakeTranscriber("Satzli")
    p, _, lines = make(dialect=dialect, options=Options(dialect=True))
    p._emit(np.concatenate([silence(1.0), tone(6.0), silence(2.0)]))
    p.worker.start()
    for i in range(3):  # three sentences arrive while the dialect model is busy
        p._final(f"Satz {i}.", 0.5 + i * 2, 2.0 + i * 2)
    p._dialect_jobs.put(None)
    p._dialect.start()
    p._dialect.join(5)
    p.worker.close()
    assert [l.dialect for l in sorted(lines, key=lambda l: l.id)] == [None, None, "Satzli 1"]
    assert len(dialect.calls) == 1


def test_azure_errors_are_readable():
    from swiss_ru_de.cloud import friendly_error

    assert "key or the region" in friendly_error("WebSocket upgrade failed: Authentication error (401)")
    assert "internet" in friendly_error("Connection failed (no connection to the remote host).\nwss://x")
    assert "5 hours" in friendly_error("Quota exceeded (429)")
    assert friendly_error("Something else\nsecond line") == "Something else"
