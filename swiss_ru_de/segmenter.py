"""Cut a continuous audio stream into speech segments (utterances).

A small energy-based voice activity detector with an adaptive noise floor. It is
cheap (pure numpy) and good enough to find pauses between sentences; faster-whisper's
Silero VAD then removes any remaining non-speech inside each segment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE


@dataclass
class Segment:
    start: float  # seconds since the stream started
    end: float
    audio: np.ndarray  # float32 mono, SAMPLE_RATE
    heard_at: float | None = None  # time.monotonic() when the segment was complete (live)

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
    return max(20.0 * np.log10(rms + 1e-10), -100.0)


class Segmenter:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
        silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_segment_s: float = 12.0,
        pre_roll_ms: int = 300,
        margin_db: float = 8.0,
        floor_db: float = -55.0,
        max_noise_db: float = -35.0,
        noise_window_s: float = 5.0,
        min_segment_s: float = 0.0,
        long_silence_ms: int = 1500,
    ):
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = max(2, int(max_segment_s * 1000 / frame_ms))
        # Short utterances are kept open (up to a long pause) and joined with the next one:
        # Whisper is more accurate with more context, and every recognition pass costs the
        # same 30 s window however short the audio is - so fewer, longer segments are both
        # more precise and faster.
        self.min_frames = int(min_segment_s * 1000 / frame_ms)
        self.long_silence_frames = max(self.silence_frames, long_silence_ms // frame_ms)
        self.keep_tail_frames = max(1, 200 // frame_ms)  # keep a little silence at the end
        self.margin_db = margin_db
        self.floor_db = floor_db
        self.max_noise_db = max_noise_db
        self._history: deque[float] = deque(maxlen=max(10, int(noise_window_s * 1000 / frame_ms)))

        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet framed
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self._frames: list[np.ndarray] = []  # frames of the current segment
        self._flags: list[bool] = []  # speech flag per frame
        self._dbs: list[float] = []
        self._seg_start = 0  # sample index of the first frame in _frames
        self._n = 0  # samples consumed so far
        self._silence_run = 0
        self.in_speech = False

    # ------------------------------------------------------------------ public
    def feed(self, audio: np.ndarray) -> list[Segment]:
        """Add audio, return the segments that were completed by it."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._pending, audio]) if self._pending.size else audio
        out: list[Segment] = []
        n_frames = len(buf) // self.frame
        for i in range(n_frames):
            seg = self._process(buf[i * self.frame : (i + 1) * self.frame])
            if seg is not None:
                out.append(seg)
        self._pending = buf[n_frames * self.frame :].copy()
        return out

    def flush(self) -> Segment | None:
        """Close the current segment (e.g. when the stream stops)."""
        seg = self._emit(len(self._frames)) if self.in_speech else None
        self._reset()
        return seg

    # ----------------------------------------------------------------- private
    @property
    def noise_db(self) -> float:
        """Background level: 10th percentile of the last few seconds. Speech always has
        short gaps between words, so this follows the room/background, not the voice.
        Capped so that a loud music bed is not mistaken for silence forever."""
        if not self._history:
            return self.floor_db - self.margin_db
        return min(float(np.percentile(self._history, 10)), self.max_noise_db)

    def _is_speech(self, db: float) -> bool:
        threshold = max(self.noise_db + self.margin_db, self.floor_db)
        self._history.append(db)
        return db > threshold

    def _process(self, frame: np.ndarray) -> Segment | None:
        db = frame_db(frame)
        speech = self._is_speech(db)
        start_sample = self._n
        self._n += len(frame)

        if not self.in_speech:
            if not speech:
                self._pre.append(frame)
                return None
            # speech starts: include a bit of audio from before the onset
            self.in_speech = True
            self._frames = list(self._pre)
            self._flags = [False] * len(self._frames)
            self._dbs = [-100.0] * len(self._frames)
            self._seg_start = start_sample - len(self._frames) * self.frame
            self._pre.clear()
            self._silence_run = 0

        self._frames.append(frame)
        self._flags.append(speech)
        self._dbs.append(db)
        self._silence_run = 0 if speech else self._silence_run + 1

        spoken = len(self._frames) - self._silence_run
        if self._silence_run >= self.silence_frames and (
            spoken >= self.min_frames or self._silence_run >= self.long_silence_frames
        ):
            keep = spoken + self.keep_tail_frames
            seg = self._emit(keep)
            self._reset()
            return seg

        if len(self._frames) >= self.max_frames:
            return self._split()
        return None

    def _split(self) -> Segment | None:
        """Segment too long: cut at the quietest point of its last 40 %."""
        n = len(self._frames)
        lo = int(n * 0.6)
        dbs = np.convolve(np.asarray(self._dbs[lo:]), np.ones(3) / 3, mode="same")
        cut = lo + int(np.argmin(dbs)) + 1
        cut = min(max(cut, 1), n - 1)
        seg = self._emit(cut)
        self._seg_start += cut * self.frame
        self._frames = self._frames[cut:]
        self._flags = self._flags[cut:]
        self._dbs = self._dbs[cut:]
        return seg

    def _emit(self, n_frames: int) -> Segment | None:
        frames = self._frames[:n_frames]
        if sum(self._flags[:n_frames]) < self.min_speech_frames or not frames:
            return None
        audio = np.concatenate(frames)
        start = self._seg_start / self.sr
        return Segment(start=start, end=start + len(audio) / self.sr, audio=audio)

    def _reset(self) -> None:
        self.in_speech = False
        self._frames, self._flags, self._dbs = [], [], []
        self._silence_run = 0
