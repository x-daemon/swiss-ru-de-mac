import inspect
import os
from types import SimpleNamespace

import numpy as np

from swiss_ru_de import asr


def test_transcribe_options_match_faster_whisper_api():
    from faster_whisper import WhisperModel

    t = asr.Transcriber.__new__(asr.Transcriber)  # skip model loading
    t.hint, t.prompt = "Grindelwald, Eiger", asr.dialect_prompt("bern")
    sig = inspect.signature(WhisperModel.transcribe)
    sig.bind(None, np.zeros(16000, np.float32), without_timestamps=True, **t._options(1))
    sig.bind(None, "file.mp4", **t._options(5))


def test_mel_bins_fixed_for_large_v3_models():
    from faster_whisper.feature_extractor import FeatureExtractor

    model = SimpleNamespace(model=SimpleNamespace(n_mels=128), feature_extractor=FeatureExtractor(),
                            feat_kwargs={})
    asr._fix_mel_bins(model)
    assert model.feature_extractor.mel_filters.shape[0] == 128


def test_resolve_preset_downloads_only_ct2_subfolder(monkeypatch, tmp_path):
    import huggingface_hub

    calls = {}

    class FakeApi:
        def list_repo_files(self, repo):
            calls["repo"] = repo
            return ["README.md", "model.safetensors", "ct2/model.bin", "ct2/config.json"]

    def fake_download(repo, allow_patterns, ignore_patterns):
        calls["allow"] = allow_patterns
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    path = asr.resolve_model("swiss-large", log=lambda m: None)
    assert calls["repo"] == "gcoli/whisper-large-v3-swiss-german-mit"
    assert calls["allow"] == ["ct2/*"]
    assert path == os.path.join(str(tmp_path), "ct2")


def test_resolve_alias_and_local_folder(tmp_path):
    assert asr.resolve_model("turbo") == "large-v3-turbo"
    sub = tmp_path / "my-model" / "ct2"
    sub.mkdir(parents=True)
    (sub / "model.bin").write_bytes(b"")
    assert asr.resolve_model(str(tmp_path / "my-model")) == str(sub)


def test_transcribe_filters_hallucinations():
    t = asr.Transcriber.__new__(asr.Transcriber)
    t.hint, t.beam_size, t.prompt = None, 1, None
    segs = [SimpleNamespace(text=" Grüezi mitenand,", no_speech_prob=0.1),
            SimpleNamespace(text=" Untertitel im Auftrag des ZDF", no_speech_prob=0.2),
            SimpleNamespace(text=" wie gohts?", no_speech_prob=0.1),
            SimpleNamespace(text=" leise", no_speech_prob=0.95)]
    t.model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(segs), None))
    assert t.transcribe(np.zeros(16000, np.float32)) == "Grüezi mitenand, wie gohts?"


def test_audio_files_can_be_decoded(tmp_path):
    """faster-whisper decodes files with PyAV; PyAV 19 broke it (pinned in pyproject)."""
    import wave

    from faster_whisper import decode_audio

    path = tmp_path / "t.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    assert len(decode_audio(str(path), sampling_rate=16000)) == 16000
