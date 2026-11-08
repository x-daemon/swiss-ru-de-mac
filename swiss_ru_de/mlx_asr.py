"""Speech recognition on the Apple GPU with MLX (Apple Silicon Macs).

Whisper on MLX runs on the GPU of an M-series chip: on an M3/M4 Max a recognition pass
takes a fraction of a second. faster-whisper (CTranslate2) has no Mac GPU support and runs
on the processor there.

Models:

* ``turbo`` (dialect line, fallback): the official MLX conversion of Whisper
  large-v3-turbo (``mlx-community/whisper-large-v3-turbo``).
* ``swiss-turbo``: a Swiss German fine-tune of large-v3-turbo. It is published in
  Hugging Face Transformers format, so it is converted to MLX once (about a minute) and
  kept in the settings folder.

If MLX is not available (Intel Mac, missing package) or a model cannot be prepared, the
app falls back to faster-whisper on the processor (``asr.Transcriber``).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .config import SAMPLE_RATE

Log = Callable[[str], None]

# Ready-made MLX models
MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
# Swiss German fine-tunes in Transformers format, converted to MLX on first use (in order)
MLX_CONVERT = {
    "swiss-turbo": ["Flurin17/whisper-large-v3-turbo-swiss-german"],
    "swiss-turbo-int8": ["Flurin17/whisper-large-v3-turbo-swiss-german"],
}

# mlx_whisper keeps one model in a class-level holder and reloads it whenever another
# model is asked for. The app uses two (Swiss German + dialect line), so every call sets
# the holder to the caller's model - under one lock, which also keeps the GPU work of the
# two recognition threads from interleaving.
_MLX_LOCK = threading.Lock()


def mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_whisper  # noqa: F401
    except Exception:
        return False
    return True


def models_dir() -> Path:
    from .settings import config_dir

    return config_dir() / "models"


# --------------------------------------------------------------- conversion
def _hf_key(key: str) -> str | None:
    """Hugging Face Whisper parameter name -> mlx_whisper name (None = not needed)."""
    key = key.removeprefix("model.")
    if key.startswith("proj_out") or key == "encoder.embed_positions.weight":
        return None  # output layer = token embedding; encoder positions are fixed sinusoids
    for old, new in ((".layers.", ".blocks."), (".self_attn_layer_norm", ".attn_ln"),
                     (".encoder_attn_layer_norm", ".cross_attn_ln"), (".self_attn.", ".attn."),
                     (".encoder_attn.", ".cross_attn."), (".final_layer_norm", ".mlp_ln"),
                     (".q_proj.", ".query."), (".k_proj.", ".key."), (".v_proj.", ".value."),
                     (".out_proj.", ".out."), (".fc1.", ".mlp1."), (".fc2.", ".mlp2."),
                     ("decoder.embed_positions.weight", "decoder.positional_embedding"),
                     ("decoder.embed_tokens.", "decoder.token_embedding."),
                     ("encoder.layer_norm.", "encoder.ln_post."),
                     ("decoder.layer_norm.", "decoder.ln.")):
        key = key.replace(old, new)
    return key


def dims_from_hf_config(cfg: dict) -> dict:
    return {
        "n_mels": cfg["num_mel_bins"],
        "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"],
        "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"],
        "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"],
        "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"],
        "n_text_layer": cfg["decoder_layers"],
    }


def convert_hf_whisper(src: Path, out: Path, dtype: str = "float16") -> Path:
    """Convert a Transformers Whisper checkpoint (config.json + *.safetensors) to the
    mlx_whisper format (config.json + weights.safetensors). Checked by loading it."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_whisper import whisper

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    dims = dims_from_hf_config(cfg)
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"{src} has no .safetensors weights")
    weights = {}
    for f in files:
        for key, value in mx.load(str(f)).items():
            new = _hf_key(key)
            if new is None:
                continue
            if new.endswith(("conv1.weight", "conv2.weight")):
                value = value.swapaxes(1, 2)  # PyTorch (out, in, k) -> MLX (out, k, in)
            weights[new] = value.astype(getattr(mx, dtype))
    model = whisper.Whisper(whisper.ModelDimensions(**dims), getattr(mx, dtype))
    # alignment_heads: computed by mlx_whisper (word timestamps), not a weight
    expected = {k for k, _ in tree_flatten(model.parameters()) if k != "alignment_heads"}
    missing, extra = expected - set(weights), set(weights) - expected
    if missing or extra:
        raise RuntimeError(f"conversion mismatch: missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(extra)[:5]}")
    model.update(tree_unflatten(list(weights.items())))  # shapes are checked when used
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "weights.safetensors"), weights)
    (tmp / "config.json").write_text(json.dumps({"model_type": "whisper", **dims}), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def resolve_mlx_model(name: str, log: Log = print) -> str:
    """Preset / Hugging Face repo / local folder -> what mlx_whisper can load."""
    if os.path.isdir(name):
        if (Path(name) / "weights.safetensors").exists() or (Path(name) / "weights.npz").exists():
            return name
        out = models_dir() / ("mlx-" + re.sub(r"[^\w.-]", "_", Path(name).name))
        if not (out / "weights.safetensors").exists():
            convert_hf_whisper(Path(name), out)
        return str(out)
    if name in MLX_MODELS:
        return MLX_MODELS[name]
    sources = MLX_CONVERT.get(name, [name] if "/" in name and not name.startswith("mlx-community/") else [])
    if not sources:
        if name.startswith("mlx-community/"):
            return name
        raise RuntimeError(f"no MLX version of '{name}'")
    errors = []
    for repo in sources:
        out = models_dir() / ("mlx-" + re.sub(r"[^\w.-]", "_", repo))
        if (out / "weights.safetensors").exists():
            return str(out)
        try:
            from huggingface_hub import snapshot_download

            log(f"Downloading {repo} and preparing it for the Apple GPU (once, a few minutes) …")
            src = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"]))
            convert_hf_whisper(src, out)
            return str(out)
        except Exception as e:
            errors.append(f"{repo}: {e}")
    raise RuntimeError("; ".join(errors))


# ---------------------------------------------------------------- transcriber
class MLXTranscriber:
    """Same interface as asr.Transcriber, running Whisper on the Apple GPU."""

    device = "mlx"
    split_utterances = False

    def __init__(self, model: str, device: str = "auto", compute_type: str | None = None,
                 beam_size: int | None = None, hint: str | None = None, log: Log = print,
                 prompt: str | None = None):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        self.hint = hint or None
        self.prompt = prompt
        self.log = log
        self.beam_size = 1  # mlx_whisper decodes greedily; kept for the shared interface
        self.path = resolve_mlx_model(model, log)
        try:
            self.model = load_model(self.path, dtype=mx.float16)
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), live=True)  # warm-up
        except Exception:
            if mx.default_device() == mx.cpu:
                raise
            mx.set_default_device(mx.cpu)  # no usable Metal device (e.g. a virtual machine)
            self.model = load_model(self.path, dtype=mx.float16)
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), live=True)
        started = time.perf_counter()
        self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), live=True)
        self.pass_seconds = time.perf_counter() - started
        # Every utterance on its own (see asr.utterance_clips) - affordable when a pass is fast.
        self.split_utterances = self.pass_seconds < 0.6
        where = "Apple GPU" if mx.default_device() == mx.gpu else "processor (MLX)"
        log(f"Speech model ready on the {where} ({self.pass_seconds:.2f} s per pass).")

    def _prompt(self) -> str | None:
        parts = [p for p in (self.prompt, self.hint) if p]
        return " ".join(parts) or None

    def _run(self, audio, live: bool, clips: list[tuple[int, int]] | None = None) -> dict:
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        options = dict(
            path_or_hf_repo=self.path,
            language="de",  # no Swiss German token; the output is Standard German
            task="transcribe",
            temperature=0.0 if live else (0.0, 0.2, 0.4, 0.6),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            initial_prompt=self._prompt(),
            fp16=True,
            verbose=None,
        )
        if clips:
            options["clip_timestamps"] = [t / SAMPLE_RATE for clip in clips for t in clip]
        with _MLX_LOCK:
            ModelHolder.model, ModelHolder.model_path = self.model, self.path
            return mlx_whisper.transcribe(audio, **options)

    def transcribe(self, audio: np.ndarray) -> str:
        from .asr import _echoes_prompt, clean_text, is_hallucination, utterance_clips

        clips = utterance_clips(audio) if self.split_utterances else []
        result = self._run(np.asarray(audio, dtype=np.float32), live=True,
                           clips=clips if len(clips) > 1 else None)
        parts = [clean_text(s["text"]) for s in result.get("segments", [])
                 if s.get("no_speech_prob", 0.0) < 0.8]
        text = clean_text(" ".join(p for p in parts if p and not is_hallucination(p)))
        if is_hallucination(text) or (self.prompt and _echoes_prompt(text, self.prompt)):
            return ""
        return text

    def transcribe_file(self, path: str, beam_size: int = 5) -> Iterator:
        from .asr import TextSegment, clean_text, is_hallucination

        from faster_whisper import decode_audio  # bundled decoder; mlx_whisper needs ffmpeg

        self.log("Decoding the audio …")
        result = self._run(decode_audio(path, sampling_rate=SAMPLE_RATE), live=False)
        for s in result.get("segments", []):
            text = clean_text(s["text"])
            if s.get("no_speech_prob", 0.0) < 0.8 and not is_hallucination(text):
                yield TextSegment(s["start"], s["end"], text)


def prepare(models=("swiss-turbo", "turbo"), log: Log = print) -> None:
    """Download / convert the MLX models now (the installer runs this)."""
    for name in models:
        try:
            log(f"{name}: {resolve_mlx_model(name, log)}")
        except Exception as e:
            log(f"{name}: not available for the Apple GPU ({e}); the processor will be used.")


if __name__ == "__main__":
    prepare()
