"""Wire audio -> segments -> text -> translation -> output, using threads so capture,
recognition and translation never block each other.

Every utterance becomes a :class:`Line`. Its parts arrive at different times:

* ``german``      Standard German from the Swiss German model (always first)
* ``translation`` Russian, from the translation thread (target ``ru``)
* ``dialect``     Swiss German as spoken, from the optional second (dialect) model

:class:`LineBook` collects the parts and reports every change (``on_update``, used by
the GUI) and every finished line (``on_line``, used by console and transcript files).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE
from .segmenter import Segment, Segmenter
from .translate import TranslationError, Translator

_STOP = object()
MAX_BATCH_S = 25.0  # Whisper sees 30 s windows; merge backlog up to this length


@dataclass
class Line:
    id: int
    start: float
    end: float
    german: str
    target: str = "ru"
    translation: str | None = None  # None when the target is German
    dialect: str | None = None  # Swiss German as spoken, if the dialect model is on
    translation_pending: bool = True
    dialect_pending: bool = False
    heard_at: float | None = None  # time.monotonic() when the speech had been heard (live)

    @property
    def done(self) -> bool:
        return not (self.translation_pending or self.dialect_pending)


OnLine = Callable[[Line], None]
OnStatus = Callable[[str], None]


@dataclass
class Options:
    """Settings that may change while running (from the GUI settings dialog)."""

    target: str = "ru"  # "ru", "de" or "both"
    dialect: bool = False  # also transcribe the dialect as spoken
    paused: bool = False
    correct_german: bool = True  # AI translators also return the corrected German (if shown)


class Stats:
    """How long each step takes (moving average) and how far subtitles lag behind speech."""

    STEPS = ("recognition", "dialect", "translation")

    def __init__(self):
        self.avg: dict[str, float] = {}
        self.lag = 0.0
        self._lock = threading.Lock()

    def add(self, step: str, seconds: float) -> None:
        with self._lock:
            old = self.avg.get(step)
            self.avg[step] = seconds if old is None else 0.7 * old + 0.3 * seconds

    def summary(self) -> str:
        with self._lock:
            parts = [f"{s} {self.avg[s]:.1f} s" for s in self.STEPS if s in self.avg]
        text = "Per sentence: " + ", ".join(parts) if parts else "No sentences yet"
        return text + (f". Subtitles are {self.lag:.0f} s behind the speech." if self.lag >= 3 else ".")

    def slowest(self) -> str | None:
        with self._lock:
            return max(self.avg, key=self.avg.get) if self.avg else None


class LineBook:
    def __init__(self, on_update: OnLine | None = None, on_line: OnLine | None = None):
        self.on_update = on_update
        self.on_line = on_line
        self._lines: dict[int, Line] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def new(self, start: float, end: float, german: str, target: str, dialect: bool,
            heard_at: float | None = None) -> Line:
        with self._lock:
            line = Line(self._next_id, start, end, german, target, dialect_pending=dialect,
                        heard_at=heard_at)
            self._next_id += 1
            self._lines[line.id] = line
            snap = replace(line)
        if self.on_update:
            self.on_update(snap)
        return snap

    def get(self, line_id: int) -> Line | None:
        """Current state of a line that is not finished yet."""
        with self._lock:
            line = self._lines.get(line_id)
            return replace(line) if line else None

    def update(self, line_id: int, **fields) -> None:
        with self._lock:
            line = self._lines.get(line_id)
            if line is None:
                return
            for k, v in fields.items():
                setattr(line, k, v)
            snap = replace(line)
            if line.done:
                del self._lines[line_id]
        if self.on_update:
            self.on_update(snap)
        if snap.done and self.on_line:
            self.on_line(snap)


class TranslationWorker:
    """Translates German lines in the background, keeping a little context."""

    def __init__(self, translators: Callable[[str], Translator], book: LineBook,
                 on_status: OnStatus, context_lines: int = 3, dialect_wait: float = 1.5,
                 options: Options | None = None, stats: Stats | None = None):
        self.translators = translators
        self.book = book
        self.on_status = on_status
        self.context: deque[str] = deque(maxlen=context_lines)
        self.dialect_wait = dialect_wait
        self.options = options
        self.stats = stats or Stats()
        self.origin: float | None = None  # monotonic time when the audio stream started
        self._dialect_misses = 0
        self._lag_warned_at = 0.0
        self.q: queue.Queue = queue.Queue()
        self._last_error = ""
        self.thread = threading.Thread(target=self._run, name="translate", daemon=True)

    def start(self) -> "TranslationWorker":
        self.thread.start()
        return self

    def put(self, line: Line) -> None:
        self.q.put(line)

    def close(self, wait: bool = True) -> None:
        self.q.put(_STOP)
        if wait and self.thread.ident is not None:
            self.thread.join()

    def _error(self, msg: str) -> None:
        if msg != self._last_error:  # don't repeat the same error for every line
            self._last_error = msg
            self.on_status(f"Translation failed: {msg}")

    def _dialect_of(self, line: Line) -> str | None:
        """The dialect line is a second hearing of the same audio; the AI uses it to fix
        recognition errors. Wait a moment for it - but never long, and not at all while it
        keeps being late (e.g. computed on the processor)."""
        wait = self.dialect_wait if self._dialect_misses < 2 else 0.0
        deadline = time.monotonic() + wait
        while True:
            current = self.book.get(line.id)
            if current is None or not current.dialect_pending:
                self._dialect_misses = 0
                return current.dialect if current else None
            if time.monotonic() >= deadline:
                self._dialect_misses += 1
                return None
            time.sleep(0.05)

    def _check_speed(self, line: Line, translator: Translator) -> None:
        """Tell the user when subtitles fall behind, and why."""
        # from the end of the speech to the finished subtitle (by the clock, not by audio
        # samples: those fall behind whenever audio is lost)
        if getattr(line, "heard_at", None) is not None:
            self.stats.lag = max(0.0, time.monotonic() - line.heard_at)
        elif self.origin is not None:
            self.stats.lag = max(0.0, time.monotonic() - (self.origin + line.end))
        check = getattr(translator, "gpu_warning", None)
        warning = check() if callable(check) else None
        if warning:
            self.on_status(warning)
            return
        now = time.monotonic()
        if self.stats.lag > 15 and now - self._lag_warned_at > 60:
            self._lag_warned_at = now
            slowest = self.stats.slowest()
            hint = {"translation": " The translation is the slowest step: try a smaller or "
                                   "faster model (Settings → Model).",
                    "dialect": " The dialect line is the slowest step: set the top part to "
                               "Standard German.",
                    "recognition": " Speech recognition is the slowest step: see Settings → "
                                   "Check this PC."}.get(slowest or "", "")
            self.on_status(self.stats.summary() + hint)

    def _run(self) -> None:
        while (line := self.q.get()) is not _STOP:
            german, text = line.german, None
            failed = "" if line.target != "de" else None
            translator = None
            try:
                translator = self.translators(line.target)
                dialect = self._dialect_of(line) if translator.uses_dialect else None
                correct = getattr(self.options, "correct_german", True)
                started = time.monotonic()
                fixed, text = translator.process(line.german, dialect, list(self.context), line.target,
                                                 correct=correct)
                if line.target != "de" or translator.corrects:
                    self.stats.add("translation", time.monotonic() - started)
                german = fixed or german
                self._last_error = ""
            except TranslationError as e:
                text = failed
                self._error(str(e))
            except Exception as e:  # keep the subtitles running whatever happens
                text = failed
                self._error(repr(e))
            self.context.append(german)
            self.book.update(line.id, german=german, translation=text, translation_pending=False)
            if translator is not None:
                self._check_speed(line, translator)


def merge_segments(segments: list[Segment]) -> Segment:
    """Join several utterances (with short gaps) into one recognition window."""
    if len(segments) == 1:
        return segments[0]
    gap = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, s in enumerate(segments):
        if i:
            parts.append(gap)
        parts.append(s.audio)
    return Segment(segments[0].start, segments[-1].end, np.concatenate(parts), segments[-1].heard_at)


def _join(thread: threading.Thread, timeout: float) -> None:
    if thread.ident is not None:  # started
        thread.join(timeout)


def _as_translators(t) -> Callable[[str], Translator]:
    return t if callable(t) and not isinstance(t, Translator) else (lambda target: t)


class LivePipeline:
    """Live mode: capture thread -> recognition thread -> translation thread."""

    def __init__(self, source, transcriber, translators, book: LineBook, on_status: OnStatus,
                 options: Options | None = None, segmenter: Segmenter | None = None,
                 dialect_transcriber=None):
        self.source = source
        self.transcriber = transcriber
        self.dialect_transcriber = dialect_transcriber  # may be set later (loads slowly)
        self.book = book
        self.options = options or Options()
        self.segmenter = segmenter or Segmenter()
        self.on_status = on_status
        self.segments: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.stats = Stats()
        self.worker = TranslationWorker(_as_translators(translators), book, on_status,
                                        options=self.options, stats=self.stats)
        self._capture = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._asr = threading.Thread(target=self._asr_loop, name="asr", daemon=True)
        # The dialect line has its own thread, so recognition never waits for it.
        self._dialect_jobs: queue.Queue = queue.Queue()
        self._dialect = threading.Thread(target=self._dialect_loop, name="dialect", daemon=True)
        self._dialect_busy = False
        self._lag_warned = False
        self._carry: Segment | None = None
        self.error: BaseException | None = None

    def start(self) -> "LivePipeline":
        self.worker.start()
        self._dialect.start()
        self._asr.start()
        self._capture.start()
        return self

    def stop(self, wait: bool = True) -> None:
        self.stop_event.set()
        if wait:
            _join(self._capture, 3)
            _join(self._asr, 60)
        self._dialect_jobs.put(_STOP)
        if wait:
            _join(self._dialect, 60)
            self.worker.close()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the pipeline stops by itself (e.g. audio device error)."""
        return self.stop_event.wait(timeout)

    # -- capture -----------------------------------------------------------------
    def _emit(self, audio: np.ndarray) -> None:
        for seg in self.segmenter.feed(audio):
            seg.heard_at = time.monotonic()
            if not self.options.paused:
                self.segments.put(seg)

    def _opened(self, name: str) -> None:
        self.worker.origin = time.monotonic()  # segment times count from here
        self.on_status(f"Listening to {name}")

    def _capture_loop(self) -> None:
        try:
            self.source.run(self._emit, self.stop_event, on_open=self._opened)
        except Exception as e:
            self.error = e
            self.on_status(f"Audio capture stopped: {e}")
        finally:
            last = self.segmenter.flush()
            if last is not None and not self.options.paused:
                self.segments.put(last)
            self.segments.put(_STOP)
            self.stop_event.set()

    # -- recognition ---------------------------------------------------------------
    def _next_batch(self) -> tuple[list[Segment], bool]:
        """Take one segment, plus any backlog that fits in one Whisper window."""
        first, self._carry = (self._carry or self.segments.get()), None
        if first is _STOP:
            return [], True
        batch, total = [first], first.duration
        while True:
            try:
                nxt = self.segments.get_nowait()
            except queue.Empty:
                return batch, False
            if nxt is _STOP:
                return batch, True
            if total + nxt.duration > MAX_BATCH_S:
                self._carry = nxt  # keep it for the next round
                return batch, False
            batch.append(nxt)
            total += nxt.duration

    def _asr_loop(self) -> None:
        done = False
        while not done:
            batch, done = self._next_batch()
            if not batch:
                break
            if len(batch) > 2 and not self._lag_warned:
                self._lag_warned = True
                self.on_status("This PC is too slow for real time; sentences are merged to "
                               "catch up. An NVIDIA GPU makes it about 10x faster.")
            seg = merge_segments(batch)
            # The dialect line costs a second speech-model pass. It is "best effort": only
            # done when no other speech is waiting and the previous one is finished, so it
            # never delays recognition or translation.
            idle = self.segments.empty() and self._carry is None
            want = (self.options.dialect and self.dialect_transcriber is not None and idle
                    and not self._dialect_busy)
            line = process_segment(seg.start, seg.end, seg.audio, self.transcriber, None,
                                   self.book, self.worker, self.options.target, self.on_status,
                                   dialect_later=want, heard_at=seg.heard_at)
            if line is not None and want:
                self._dialect_busy = True
                self._dialect_jobs.put((line.id, seg.audio))

    def _dialect_loop(self) -> None:
        while (job := self._dialect_jobs.get()) is not _STOP:
            line_id, audio = job
            text = transcribe_dialect(self.dialect_transcriber, audio, self.worker.stats,
                                      self.on_status)
            self.book.update(line_id, dialect=text or None, dialect_pending=False)
            self._dialect_busy = False


def transcribe_dialect(transcriber, audio, stats: Stats | None, on_status: OnStatus) -> str:
    if transcriber is None:
        return ""
    started = time.monotonic()
    try:
        text = transcriber.transcribe(audio)
    except Exception as e:
        on_status(f"Dialect recognition error: {e!r}")
        return ""
    if stats is not None:
        stats.add("dialect", time.monotonic() - started)
    return text


def process_segment(start, end, audio, transcriber, dialect_transcriber, book, worker,
                    target, on_status, german: str | None = None,
                    dialect_later: bool = False, heard_at: float | None = None) -> Line | None:
    """Standard German first (so translation can start), then the dialect line - here,
    or (``dialect_later``) by the caller."""
    stats = getattr(worker, "stats", None)
    if german is None:
        started = time.monotonic()
        try:
            german = transcriber.transcribe(audio)
        except Exception as e:
            on_status(f"Recognition error: {e!r}")
            return None
        if stats is not None:
            stats.add("recognition", time.monotonic() - started)
    if not german:
        return None
    line = book.new(start, end, german, target,
                    dialect=dialect_later or dialect_transcriber is not None, heard_at=heard_at)
    worker.put(line)
    if dialect_transcriber is not None:
        dialect = transcribe_dialect(dialect_transcriber, audio, stats, on_status)
        book.update(line.id, dialect=dialect or None, dialect_pending=False)
    return line


def run_file(transcriber, translators, path: str, book: LineBook, on_status: OnStatus,
             options: Options | None = None, beam_size: int = 5,
             stop: threading.Event | None = None, dialect_transcriber=None,
             load_audio: Callable[[str], np.ndarray] | None = None) -> None:
    """File / URL mode: recognise the whole file, translate while recognising."""
    options = options or Options()
    worker = TranslationWorker(_as_translators(translators), book, on_status,
                               options=options).start()
    audio = None
    if dialect_transcriber is not None:
        if load_audio is None:
            from faster_whisper import decode_audio as load_audio
        audio = load_audio(path)
    try:
        for s in transcriber.transcribe_file(path, beam_size=beam_size):
            if stop is not None and stop.is_set():
                break
            chunk = None
            if audio is not None:
                chunk = audio[int(s.start * SAMPLE_RATE) : int(s.end * SAMPLE_RATE)]
            process_segment(s.start, s.end, chunk, transcriber,
                            dialect_transcriber if chunk is not None and chunk.size else None,
                            book, worker, options.target, on_status, german=s.text)
    finally:
        worker.close()
