import numpy as np

from swiss_ru_de.config import SAMPLE_RATE
from swiss_ru_de.segmenter import Segmenter

rng = np.random.default_rng(0)


def tone(seconds, amp=0.2, freq=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds, amp=0.001):
    return (amp * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def feed_in_blocks(seg, audio, block=1600):
    out = []
    for i in range(0, len(audio), block):
        out += seg.feed(audio[i : i + block])
    return out


def test_two_utterances_separated_by_pause():
    audio = np.concatenate([noise(1.0), tone(1.5), noise(1.0), tone(2.0), noise(1.0)])
    segs = feed_in_blocks(Segmenter(), audio)
    assert len(segs) == 2
    assert abs(segs[0].start - 1.0) < 0.4 and abs(segs[1].start - 3.5) < 0.4
    assert 1.4 < segs[0].duration < 2.3
    assert 1.9 < segs[1].duration < 2.8


def test_silence_and_short_clicks_produce_nothing():
    audio = np.concatenate([noise(2.0), tone(0.06), noise(2.0)])
    seg = Segmenter()
    assert feed_in_blocks(seg, audio) == []
    assert seg.flush() is None


def test_long_speech_is_split_at_max_length():
    # continuous speech-like signal with small dips, 30 s long
    parts = []
    for _ in range(15):
        parts += [tone(1.9), tone(0.1, amp=0.02)]
    segs = feed_in_blocks(Segmenter(max_segment_s=10), np.concatenate(parts))
    assert len(segs) >= 2
    assert all(s.duration <= 10.05 for s in segs)
    # segments are contiguous: nothing lost
    for a, b in zip(segs, segs[1:]):
        assert abs(a.end - b.start) < 0.05


def test_flush_returns_open_segment():
    seg = Segmenter()
    assert feed_in_blocks(seg, np.concatenate([noise(0.5), tone(1.0)])) == []
    last = seg.flush()
    assert last is not None and last.duration > 0.9


def test_adapts_to_background_noise():
    # loud steady background (e.g. fan / room tone) must not count as speech forever
    bg = noise(4.0, amp=0.01)
    speech = tone(1.5, amp=0.3) + noise(1.5, amp=0.01)
    audio = np.concatenate([bg, speech, noise(2.0, amp=0.01)])
    segs = feed_in_blocks(Segmenter(), audio)
    assert len(segs) == 1
    assert abs(segs[0].start - 4.0) < 0.4


def test_short_utterances_are_joined_until_a_long_pause():
    # "Nüüd." (0.6 s) - 0.8 s pause - "Warum nüüd?" (0.8 s) - long pause
    audio = np.concatenate([noise(1.0), tone(0.6), noise(0.8), tone(0.8), noise(2.0)])
    joined = feed_in_blocks(Segmenter(min_segment_s=3.0), audio)
    assert len(joined) == 1 and joined[0].duration > 2.0
    separate = feed_in_blocks(Segmenter(), audio)
    assert len(separate) == 2


def test_long_pause_still_ends_a_short_segment():
    audio = np.concatenate([noise(1.0), tone(0.8), noise(2.0), tone(0.8), noise(2.0)])
    segs = feed_in_blocks(Segmenter(min_segment_s=3.0, long_silence_ms=1500), audio)
    assert len(segs) == 2
