import threading
import time
from types import SimpleNamespace

import numpy as np

from swiss_ru_de.config import SAMPLE_RATE
from swiss_ru_de.output import TranscriptWriter
from swiss_ru_de.pipeline import LineBook, LivePipeline, Options, process_segment, run_file
from swiss_ru_de.segmenter import Segment
from swiss_ru_de.translate import Passthrough, TranslationError, Translator


def tone(seconds, amp=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class FakeSource:
    """Plays a fixed signal in 100 ms blocks at 10x real time, then stops."""

    def __init__(self, audio):
        self.audio = audio

    def run(self, emit, stop, on_open=None):
        if on_open:
            on_open("fake")
        for i in range(0, len(self.audio), 1600):
            if stop.is_set():
                return
            emit(self.audio[i : i + 1600])
            time.sleep(0.01)


class FakeTranscriber:
    pass_seconds = 0.5  # measured speed of one pass (see tuning.py)
    device = "cpu"
    beam_size = None

    def __init__(self, word="Satz"):
        self.word = word
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio) / SAMPLE_RATE)
        return f"{self.word} {len(self.calls)}"

    def transcribe_file(self, path, beam_size=5):
        for i in range(3):
            yield SimpleNamespace(start=i * 2.0, end=i * 2.0 + 1.5, text=f"Satz {i + 1}")


class FakeRu(Translator):
    name = "fake"

    def __init__(self):
        self.contexts = []

    def translate(self, text, context=()):
        self.contexts.append(list(context))
        return text.replace("Satz", "Предложение")


def run_live(audio, translator, options=None, dialect=None):
    lines, updates, status = [], [], []
    book = LineBook(on_update=updates.append, on_line=lines.append)
    p = LivePipeline(FakeSource(audio), FakeTranscriber(), translator, book, status.append,
                     options=options, dialect_transcriber=dialect).start()
    assert p.wait(10)
    p.stop()
    return lines, updates, status


TWO_SENTENCES = np.concatenate([silence(0.5), tone(1.2), silence(1.0), tone(1.0), silence(1.0)])


def test_live_pipeline_end_to_end():
    tr = FakeRu()
    lines, _, status = run_live(TWO_SENTENCES, tr)
    assert [l.german for l in lines] == ["Satz 1", "Satz 2"]
    assert [l.translation for l in lines] == ["Предложение 1", "Предложение 2"]
    assert tr.contexts == [[], ["Satz 1"]]  # previous line passed as context
    assert lines[0].start < lines[1].start
    assert all(l.dialect is None for l in lines)
    assert "Listening to fake" in status


def test_dialect_line_and_partial_updates():
    lines, updates, _ = run_live(TWO_SENTENCES, FakeRu(), Options(dialect=True),
                                 dialect=FakeTranscriber("Satzli"))
    assert [l.dialect for l in lines] == ["Satzli 1", "Satzli 2"]
    assert [l.translation for l in lines] == ["Предложение 1", "Предложение 2"]
    first = [u for u in updates if u.id == lines[0].id]
    assert first[0].translation_pending and first[0].dialect_pending  # shown immediately
    assert first[-1].done and len(first) == 3  # then filled in twice


def test_dialect_transcriber_ignored_when_option_off():
    lines, _, _ = run_live(TWO_SENTENCES, FakeRu(), Options(dialect=False),
                           dialect=FakeTranscriber("Satzli"))
    assert all(l.dialect is None and l.done for l in lines)


def test_german_only_skips_translation():
    lines, _, _ = run_live(np.concatenate([tone(1.0), silence(1.0)]), FakeRu(), Options(target="de"))
    assert len(lines) == 1 and lines[0].translation is None


def test_target_can_change_while_running():
    options = Options(target="ru")
    translators = {"ru": FakeRu(), "de": Passthrough()}
    seen = []
    book = LineBook(on_line=seen.append)
    p = LivePipeline(FakeSource(silence(0.1)), FakeTranscriber(), lambda t: translators[t], book,
                     lambda s: None, options=options)
    p.worker.start()
    process_segment(0, 1, tone(1), p.transcriber, None, book, p.worker, options.target, print)
    options.target = "de"
    process_segment(1, 2, tone(1), p.transcriber, None, book, p.worker, options.target, print)
    p.worker.close()
    assert [l.translation for l in seen] == ["Предложение 1", None]


def test_paused_audio_is_dropped():
    lines, _, _ = run_live(TWO_SENTENCES, FakeRu(), Options(paused=True))
    assert lines == []


def test_translation_errors_do_not_stop_subtitles():
    class Broken(Translator):
        def translate(self, text, context=()):
            raise TranslationError("no key")

    lines, _, status = run_live(TWO_SENTENCES, Broken())
    assert len(lines) == 2 and all(l.translation == "" for l in lines)
    assert sum("no key" in s for s in status) == 1  # reported once, not per line


def test_backlog_is_merged_into_one_window():
    p = LivePipeline(FakeSource(silence(0.1)), FakeTranscriber(), Passthrough(), LineBook(),
                     lambda s: None)
    for i in range(4):
        p.segments.put(Segment(i * 8.0, i * 8.0 + 7.0, tone(7.0)))
    batch, done = p._next_batch()
    assert len(batch) == 3 and not done  # 3 x 7 s fit into 25 s
    batch, done = p._next_batch()
    assert [s.start for s in batch] == [24.0] and not done  # 4th was carried over, not lost


def test_run_file_writes_srt(tmp_path):
    out = tmp_path / "x.srt"
    writer = TranscriptWriter(str(out), "both")
    lines = []

    def sink(line):
        lines.append(line)
        writer(line)

    run_file(FakeTranscriber(), FakeRu(), "dummy.wav", LineBook(on_line=sink), lambda s: None)
    writer.close()
    text = out.read_text(encoding="utf-8")
    assert len(lines) == 3
    assert "1\n00:00:00,000 --> 00:00:01,500\nSatz 1\nПредложение 1\n" in text
    assert text.count("-->") == 3


def test_run_file_with_dialect_uses_matching_audio_slices():
    dialect = FakeTranscriber("Satzli")
    lines = []
    run_file(FakeTranscriber(), FakeRu(), "x.wav", LineBook(on_line=lines.append), lambda s: None,
             options=Options(dialect=True), dialect_transcriber=dialect,
             load_audio=lambda path: silence(10))
    assert [l.dialect for l in lines] == ["Satzli 1", "Satzli 2", "Satzli 3"]
    assert dialect.calls == [1.5, 1.5, 1.5]  # each segment's own 1.5 s of audio


def test_run_file_can_be_stopped():
    stop = threading.Event()
    lines = []

    def sink(line):
        lines.append(line)
        stop.set()

    class Slow(FakeTranscriber):
        def transcribe_file(self, path, beam_size=5):
            for i in range(100):
                yield SimpleNamespace(start=i, end=i + 1, text=f"Satz {i}")
                stop.wait(1)

    run_file(Slow(), Passthrough(), "x", LineBook(on_line=sink), lambda s: None,
             options=Options(target="de"), stop=stop)
    assert len(lines) < 100


class FakeClaude(Translator):
    corrects = True

    def __init__(self):
        self.seen = []

    def process(self, german, dialect=None, context=(), target="ru", correct=True):
        self.seen.append((german, dialect, list(context), target))
        fixed = german.replace("Gipfel", "Gipfeli")
        return fixed, ("RU:" + fixed if target == "ru" else None)


def test_worker_uses_dialect_line_and_stores_corrected_german():
    from swiss_ru_de.pipeline import TranslationWorker

    lines = []
    book = LineBook(on_line=lines.append)
    claude = FakeClaude()
    worker = TranslationWorker(lambda t: claude, book, lambda s: None, dialect_wait=2.0).start()
    line = book.new(0, 1, "zwei Gipfel", "ru", dialect=True)
    worker.put(line)

    def dialect_ready():
        time.sleep(0.2)
        book.update(line.id, dialect="zwöi Gipfeli", dialect_pending=False)

    threading.Thread(target=dialect_ready).start()
    worker.close()
    assert claude.seen[0][:2] == ("zwei Gipfel", "zwöi Gipfeli")  # waited for the 2nd hearing
    assert (lines[0].german, lines[0].translation) == ("zwei Gipfeli", "RU:zwei Gipfeli")


def test_worker_does_not_wait_forever_for_the_dialect_line():
    from swiss_ru_de.pipeline import TranslationWorker

    lines = []
    book = LineBook(on_line=lines.append)
    claude = FakeClaude()
    worker = TranslationWorker(lambda t: claude, book, lambda s: None, dialect_wait=0.2).start()
    line = book.new(0, 1, "zwei Gipfel", "de", dialect=True)
    started = time.monotonic()
    worker.put(line)
    time.sleep(0.5)
    book.update(line.id, dialect=None, dialect_pending=False)
    worker.close()
    assert claude.seen[0][1] is None and time.monotonic() - started < 2
    assert lines[0].german == "zwei Gipfeli" and lines[0].translation is None  # German corrected
