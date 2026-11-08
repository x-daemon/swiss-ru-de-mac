"""Mac parts: MLX model conversion, the recogniser factory, system audio capture."""

import json
import sys
import textwrap
import threading

import numpy as np
import pytest

from swiss_ru_de import mlx_asr

TINY = {"num_mel_bins": 80, "max_source_positions": 1500, "d_model": 64,
        "encoder_attention_heads": 2, "encoder_layers": 2, "vocab_size": 51866,
        "max_target_positions": 448, "decoder_attention_heads": 2, "decoder_layers": 2}


def hf_whisper_weights(cfg=TINY, rng=np.random.default_rng(0)):
    """Parameter names and shapes exactly as transformers' WhisperForConditionalGeneration."""
    d, mels, ff = cfg["d_model"], cfg["num_mel_bins"], 4 * cfg["d_model"]
    w = {}

    def add(name, *shape):
        w[name] = (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def attn(prefix):
        add(f"{prefix}.k_proj.weight", d, d)
        for p in ("q_proj", "v_proj", "out_proj"):
            add(f"{prefix}.{p}.weight", d, d)
            add(f"{prefix}.{p}.bias", d)

    def norm(name):
        w[f"{name}.weight"], w[f"{name}.bias"] = np.ones(d, np.float32), np.zeros(d, np.float32)

    add("model.encoder.conv1.weight", d, mels, 3)
    add("model.encoder.conv1.bias", d)
    add("model.encoder.conv2.weight", d, d, 3)
    add("model.encoder.conv2.bias", d)
    add("model.encoder.embed_positions.weight", cfg["max_source_positions"], d)
    for side, n in (("encoder", cfg["encoder_layers"]), ("decoder", cfg["decoder_layers"])):
        for i in range(n):
            layer = f"model.{side}.layers.{i}"
            attn(f"{layer}.self_attn")
            norm(f"{layer}.self_attn_layer_norm")
            if side == "decoder":
                attn(f"{layer}.encoder_attn")
                norm(f"{layer}.encoder_attn_layer_norm")
            add(f"{layer}.fc1.weight", ff, d)
            add(f"{layer}.fc1.bias", ff)
            add(f"{layer}.fc2.weight", d, ff)
            add(f"{layer}.fc2.bias", d)
            norm(f"{layer}.final_layer_norm")
        norm(f"model.{side}.layer_norm")
    add("model.decoder.embed_tokens.weight", cfg["vocab_size"], d)
    add("model.decoder.embed_positions.weight", cfg["max_target_positions"], d)
    add("proj_out.weight", cfg["vocab_size"], d)
    return w


def test_hf_names_map_to_mlx_whisper_names():
    names = {mlx_asr._hf_key(k) for k in hf_whisper_weights()} - {None}
    assert "encoder.blocks.1.attn.query.weight" in names
    assert "encoder.blocks.0.attn_ln.bias" in names
    assert "decoder.blocks.0.cross_attn.key.weight" in names
    assert "decoder.blocks.1.cross_attn_ln.weight" in names
    assert "decoder.blocks.0.mlp1.weight" in names and "decoder.blocks.0.mlp_ln.bias" in names
    assert {"encoder.ln_post.weight", "decoder.ln.bias", "decoder.token_embedding.weight",
            "decoder.positional_embedding"} <= names
    assert not any("embed_positions" in n or "proj_out" in n for n in names)
    assert mlx_asr.dims_from_hf_config(TINY)["n_text_layer"] == 2


def test_convert_and_transcribe_with_mlx(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_whisper")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    src = tmp_path / "hf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(TINY))
    mx.save_safetensors(str(src / "model.safetensors"),
                        {k: mx.array(v) for k, v in hf_whisper_weights().items()})
    out = mlx_asr.convert_hf_whisper(src, tmp_path / "mlx")
    assert json.loads((out / "config.json").read_text())["n_audio_layer"] == 2
    weights = mx.load(str(out / "weights.safetensors"))
    assert weights["encoder.conv1.weight"].shape == (64, 3, 80)  # MLX conv layout
    assert weights["decoder.token_embedding.weight"].dtype == mx.float16

    tr = mlx_asr.MLXTranscriber(str(out), log=lambda m: None)  # loads and runs it
    assert tr.pass_seconds > 0 and tr.beam_size == 1
    assert isinstance(tr.transcribe(np.zeros(16000 * 3, np.float32)), str)
    # a folder with Transformers weights is converted automatically
    assert mlx_asr.resolve_mlx_model(str(src)).endswith("mlx-hf")


def test_mlx_models_and_fallback(monkeypatch):
    assert mlx_asr.resolve_mlx_model("turbo") == "mlx-community/whisper-large-v3-turbo"
    from swiss_ru_de import asr

    made = []

    class FakeCT2:
        def __init__(self, model, **kw):
            made.append(("ct2", model, kw.get("device")))

    def broken(model, **kw):
        raise RuntimeError("download failed")

    monkeypatch.setattr(asr, "Transcriber", FakeCT2)
    monkeypatch.setattr(mlx_asr, "mlx_available", lambda: True)
    monkeypatch.setattr(mlx_asr, "MLXTranscriber", broken)
    log = []
    asr.make_transcriber("swiss-turbo", device="auto", log=log.append)
    assert made == [("ct2", "swiss-turbo", "cpu")] and "processor" in log[-1]
    asr.make_transcriber("swiss-turbo", device="cpu", log=log.append)  # MLX not even tried
    assert made[-1] == ("ct2", "swiss-turbo", "cpu")
    monkeypatch.setattr(mlx_asr, "MLXTranscriber", lambda model, **kw: ("mlx", model))
    assert asr.make_transcriber("turbo", log=log.append) == ("mlx", "turbo")
    monkeypatch.setattr(mlx_asr, "mlx_available", lambda: False)
    asr.make_transcriber("turbo", device="auto", log=log.append)
    assert made[-1] == ("ct2", "turbo", "auto")


# ------------------------------------------------------------------ system audio
def fake_helper(tmp_path, body: str) -> list[str]:
    script = tmp_path / "audiotap.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


def test_system_audio_source_streams_the_helper_output(tmp_path):
    from swiss_ru_de.mac_audio import SystemAudioSource

    cmd = fake_helper(tmp_path, """
        import sys, numpy as np
        out = sys.stdout.buffer
        data = (np.arange(16000, dtype=np.float32) / 16000).tobytes()
        for i in range(0, len(data), 1001):  # odd chunk sizes: samples split across reads
            out.write(data[i:i + 1001]); out.flush()
    """)
    got, opened = [], []
    src = SystemAudioSource(command=cmd)
    with pytest.raises(RuntimeError):  # the helper ended: reported like a device error
        src.run(got.append, threading.Event(), on_open=opened.append)
    audio = np.concatenate(got)
    assert len(audio) == 16000 and np.allclose(audio, np.arange(16000) / 16000)
    assert opened == ["Mac system audio"]


def test_missing_permission_is_explained(tmp_path):
    from swiss_ru_de.mac_audio import SystemAudioSource

    cmd = fake_helper(tmp_path, """
        import sys
        print("audiotap: permission: The user declined TCCs for application", file=sys.stderr)
        sys.exit(3)
    """)
    with pytest.raises(RuntimeError, match="Screen & System Audio Recording"):
        SystemAudioSource(command=cmd).run(lambda a: None, threading.Event())


def test_stop_ends_a_running_helper(tmp_path):
    from swiss_ru_de.mac_audio import SystemAudioSource

    cmd = fake_helper(tmp_path, """
        import sys, time
        while True:
            sys.stdout.buffer.write(b"\\0" * 6400); sys.stdout.flush(); time.sleep(0.1)
    """)
    stop = threading.Event()
    got = []

    def emit(a):
        got.append(a)
        if len(got) == 3:
            stop.set()

    SystemAudioSource(command=cmd).run(emit, stop)  # returns once stopped, no error
    assert len(got) >= 3
