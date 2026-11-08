"""Cloud speech recognition with Azure AI Speech (locale de-CH).

Microsoft's de-CH model recognises the Swiss German dialects and writes Standard German.
Audio is streamed while it is spoken: text appears during the sentence (partial results)
and the final sentence a moment after the speaker pauses. The PC's CPU stays free, so this
is the fast option for computers without an NVIDIA GPU.

Only speech is sent (a local level gate skips silence), which keeps the cost down: Azure
bills per second of audio sent (free tier: 5 hours per month).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE
from .pipeline import Line, LineBook, OnStatus, Options, Stats, TranslationWorker, _as_translators
from .segmenter import frame_db

TICKS_PER_SECOND = 10_000_000  # Azure offsets are in 100 ns units


class SpeechGate:
    """Passes audio only while someone speaks: 300 ms before speech starts until 1.2 s after
    it stops (so Azure still sees the pause that ends the sentence)."""

    def __init__(self, frame_ms: int = 30, pre_roll_ms: int = 300, hangover_ms: int = 1200,
                 margin_db: float = 8.0, floor_db: float = -55.0, max_noise_db: float = -35.0):
        self.frame = SAMPLE_RATE * frame_ms // 1000
        self.margin_db, self.floor_db, self.max_noise_db = margin_db, floor_db, max_noise_db
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self._history: deque[float] = deque(maxlen=166)  # 5 s
        self._hangover = max(1, hangover_ms // frame_ms)
        self._open_for = 0

    def _noise_db(self) -> float:
        if not self._history:
            return self.floor_db - self.margin_db
        return min(float(np.percentile(self._history, 10)), self.max_noise_db)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).reshape(-1)])
        n = len(buf) // self.frame
        out: list[np.ndarray] = []
        for i in range(n):
            frame = buf[i * self.frame : (i + 1) * self.frame]
            db = frame_db(frame)
            speech = db > max(self._noise_db() + self.margin_db, self.floor_db)
            self._history.append(db)
            if speech:
                if self._open_for == 0:  # opening: include what came just before
                    out.extend(self._pre)
                    self._pre.clear()
                self._open_for = self._hangover
                out.append(frame)
            elif self._open_for > 0:
                self._open_for -= 1
                out.append(frame)
            else:
                self._pre.append(frame)
        self._pending = buf[n * self.frame :].copy()
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def friendly_error(details: str) -> str:
    if "401" in details or "Authentication" in details or "403" in details:
        return "the key or the region is wrong (check Settings)."
    if "Connection failed" in details or "WS_OPEN_ERROR" in details:
        return "cannot connect - check the internet connection and the region in Settings."
    if "429" in details or "Quota" in details or "quota" in details:
        return "quota used up (free tier: 5 hours per month)."
    return details.splitlines()[0][:200]


class AzureSpeech:
    """Thin wrapper around the Azure Speech SDK: push audio in, get callbacks out."""

    def __init__(self, key: str, region: str, on_partial: Callable[[str, float], None],
                 on_final: Callable[[str, float, float], None], on_error: Callable[[str], None],
                 phrases: list[str] | None = None, pause_ms: int = 600):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as e:
            raise RuntimeError('Azure needs: pip install "swiss-ru-de[azure]"') from e
        self.sdk = speechsdk
        config = speechsdk.SpeechConfig(subscription=key, region=region)
        config.speech_recognition_language = "de-CH"  # Swiss German dialects -> Standard German
        config.set_profanity(speechsdk.ProfanityOption.Raw)
        config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, str(pause_ms))
        stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=SAMPLE_RATE,
                                                          bits_per_sample=16, channels=1)
        self.stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=speechsdk.audio.AudioConfig(stream=self.stream))
        if phrases:  # names and terms that should be recognised
            grammar = speechsdk.PhraseListGrammar.from_recognizer(self.recognizer)
            for p in phrases:
                grammar.addPhrase(p)
        self._on_partial, self._on_final, self._on_error = on_partial, on_final, on_error
        self.recognizer.recognizing.connect(self._recognizing)
        self.recognizer.recognized.connect(self._recognized)
        self.recognizer.canceled.connect(self._canceled)

    def start(self) -> None:
        self.recognizer.start_continuous_recognition()

    def write(self, audio: np.ndarray) -> None:
        if audio.size:
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            self.stream.write(pcm.tobytes())

    def stop(self) -> None:
        try:
            self.stream.close()
            self.recognizer.stop_continuous_recognition()
        except Exception:
            pass

    def _recognizing(self, evt) -> None:
        if evt.result.text:
            self._on_partial(evt.result.text, evt.result.offset / TICKS_PER_SECOND)

    def _recognized(self, evt) -> None:
        r = evt.result
        if r.reason == self.sdk.ResultReason.RecognizedSpeech:
            start = r.offset / TICKS_PER_SECOND
            self._on_final(r.text, start, start + r.duration / TICKS_PER_SECOND)
        elif r.reason == self.sdk.ResultReason.NoMatch:
            self._on_final("", r.offset / TICKS_PER_SECOND, r.offset / TICKS_PER_SECOND)

    def _canceled(self, evt) -> None:
        details = evt.cancellation_details
        if details.reason == self.sdk.CancellationReason.Error:
            self._on_error(f"Azure Speech: {friendly_error(details.error_details or str(details.code))}")


class CloudPipeline:
    """Live mode with Azure: capture thread -> speech gate -> Azure (partial + final text)
    -> translation thread; optional dialect line from the local model, best effort."""

    def __init__(self, source, translators, book: LineBook, on_status: OnStatus, key: str,
                 region: str, options: Options | None = None, dialect_transcriber=None,
                 phrases: list[str] | None = None, speech_factory=AzureSpeech):
        self.source = source
        self.book = book
        self.on_status = on_status
        self.options = options or Options()
        self.dialect_transcriber = dialect_transcriber  # may be set later
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.stats = Stats()  # no lag here: Azure only hears speech, so its times skip silence
        self.worker = TranslationWorker(_as_translators(translators), book, on_status,
                                        options=self.options, stats=self.stats)
        self.gate = SpeechGate()
        self._lock = threading.Lock()
        self._current: Line | None = None  # sentence being spoken (partial text)
        # Audio exactly as sent to Azure, to cut out each sentence for the dialect line.
        self._sent = np.zeros(0, dtype=np.float32)
        self._sent_start = 0  # sample index of _sent[0] in the sent stream
        self._sent_total = 0
        self._dialect_jobs: queue.Queue = queue.Queue()
        self.speech = speech_factory(key, region, self._partial, self._final, self._fail,
                                     phrases=phrases)
        self._capture = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._dialect = threading.Thread(target=self._dialect_loop, name="dialect", daemon=True)

    def start(self) -> "CloudPipeline":
        self.worker.start()
        self._dialect.start()
        self.speech.start()
        self._capture.start()
        return self

    def stop(self, wait: bool = True) -> None:
        self.stop_event.set()
        self.speech.stop()
        self._dialect_jobs.put(None)
        if wait:
            self._capture.join(timeout=3)
            self.worker.close()

    def wait(self, timeout: float | None = None) -> bool:
        return self.stop_event.wait(timeout)

    # -- audio -------------------------------------------------------------------
    def _emit(self, audio: np.ndarray) -> None:
        if self.options.paused:
            return
        speech = self.gate.feed(audio)
        if not speech.size:
            return
        with self._lock:
            self._sent = np.concatenate([self._sent, speech])
            self._sent_total += speech.size
            excess = self._sent.size - 120 * SAMPLE_RATE  # keep the last 2 minutes
            if excess > 0:
                self._sent = self._sent[excess:]
                self._sent_start += excess
        self.speech.write(speech)

    def _capture_loop(self) -> None:
        try:
            self.source.run(self._emit, self.stop_event,
                            on_open=lambda name: self.on_status(f"Listening to {name} (Azure)"))
        except Exception as e:
            self.error = e
            self.on_status(f"Audio capture stopped: {e}")
        finally:
            self.stop_event.set()

    def _audio(self, start: float, end: float) -> np.ndarray:
        with self._lock:
            a = int(start * SAMPLE_RATE) - self._sent_start
            b = int(end * SAMPLE_RATE) - self._sent_start
            return self._sent[max(0, a) : max(0, b)].copy()

    # -- Azure callbacks (SDK threads) --------------------------------------------
    def _partial(self, text: str, start: float) -> None:
        with self._lock:
            current = self._current
        if current is None:
            line = self.book.new(start, start, text, self.options.target, dialect=False)
            with self._lock:
                self._current = line
        else:
            self.book.update(current.id, german=text)

    def _final(self, text: str, start: float, end: float) -> None:
        with self._lock:
            current, self._current = self._current, None
        text = text.strip()
        if not text:
            if current is None:
                return
            text = current.german  # partial text but no final result: keep the partial
        want_dialect = self.options.dialect and self.dialect_transcriber is not None
        target = self.options.target
        if current is None:
            current = self.book.new(start, end, text, target, dialect=want_dialect)
        self.book.update(current.id, german=text, end=end, target=target,
                         dialect_pending=want_dialect)
        self.worker.put(Line(current.id, start, end, text, target))
        if want_dialect:
            self._dialect_jobs.put((current.id, start, end))

    def _fail(self, message: str) -> None:
        self.error = RuntimeError(message)
        self.on_status(message)
        self.stop_event.set()

    # -- dialect line (local model, best effort) ------------------------------------
    def _dialect_loop(self) -> None:
        stopping = False
        while not stopping:
            job = self._dialect_jobs.get()
            if job is None:
                return
            # Only the newest sentence is worth the CPU time; skip older waiting ones.
            while not self._dialect_jobs.empty():
                nxt = self._dialect_jobs.get()
                if nxt is None:
                    stopping = True
                    break
                self.book.update(job[0], dialect_pending=False)
                job = nxt
            line_id, start, end = job
            dialect = ""
            tr = self.dialect_transcriber
            audio = self._audio(start, end)
            if tr is not None and audio.size:
                started = time.monotonic()
                try:
                    dialect = tr.transcribe(audio)
                except Exception as e:
                    self.on_status(f"Dialect recognition error: {e!r}")
                self.stats.add("dialect", time.monotonic() - started)
            self.book.update(line_id, dialect=dialect or None, dialect_pending=False)
