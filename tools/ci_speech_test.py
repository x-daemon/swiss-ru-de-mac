"""CI on a Mac: the real speech models on real speech, with timings.

The speech is made with macOS text-to-speech (`say`, a German voice), so it is Standard
German, not dialect - this checks that the models download, convert, load and recognise,
and how fast each way is on this machine.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

TEXT = "Guten Morgen. Ich hätte gern zwei Gipfeli und einen Kaffee, bitte. Danke vielmal."
EXPECT = ("morgen", "kaffee")


def german_voice() -> str | None:
    voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    for line in voices.splitlines():
        if "de_DE" in line or "de_CH" in line:
            return line.split("  ")[0].strip()
    return None


def speech() -> "np.ndarray":
    from faster_whisper import decode_audio

    tmp = Path(tempfile.mkdtemp())
    voice = german_voice()
    print(f"voice: {voice or 'default (no German voice installed)'}")
    subprocess.run(["say", *(["-v", voice] if voice else []), "-o", str(tmp / "s.aiff"), TEXT], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(tmp / "s.aiff"), str(tmp / "s.wav")], check=True)
    return decode_audio(str(tmp / "s.wav"), sampling_rate=16000), voice is not None


def check(name: str, make) -> bool:
    started = time.perf_counter()
    try:
        tr = make()
    except Exception as e:
        print(f"[{name}] could not load: {e!r}")
        return False
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    text = tr.transcribe(AUDIO)
    took = time.perf_counter() - started
    ok = all(w in text.lower() for w in EXPECT) if GERMAN else bool(text)
    print(f"[{name}] {'OK ' if ok else 'BAD'} device={tr.device} load={loaded:.1f}s "
          f"pass={tr.pass_seconds:.2f}s sentence={took:.2f}s -> {text!r}")
    return ok


if __name__ == "__main__":
    import numpy as np  # noqa: F401

    from swiss_ru_de import asr, mlx_asr

    AUDIO, GERMAN = speech()
    print(f"audio: {len(AUDIO) / 16000:.1f} s, MLX available: {mlx_asr.mlx_available()}")
    results = {
        "MLX turbo": check("MLX turbo", lambda: mlx_asr.MLXTranscriber("turbo", log=print)),
        "MLX swiss-turbo": check("MLX swiss-turbo", lambda: mlx_asr.MLXTranscriber("swiss-turbo", log=print)),
        "CPU swiss-turbo int8": check("CPU swiss-turbo int8", lambda: asr.Transcriber(
            "swiss-turbo", device="cpu", compute_type="int8", log=print)),
        "CPU swiss-turbo float32": check("CPU swiss-turbo float32", lambda: asr.Transcriber(
            "swiss-turbo", device="cpu", compute_type="float32", log=print)),
    }
    factory = asr.make_transcriber("swiss-turbo", log=print)
    print(f"make_transcriber('swiss-turbo') -> {type(factory).__name__} on {factory.device}")
    print(results)
    # the app works if the factory's choice recognises the speech
    sys.exit(0 if check("app default", lambda: factory) else 1)
