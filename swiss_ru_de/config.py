"""Shared constants: model presets, dialect names, audio format."""

SAMPLE_RATE = 16_000  # Whisper works on 16 kHz mono

# Speech recognition models (faster-whisper / CTranslate2 format).
# The Swiss German fine-tunes were trained on STT4SG-350, SDS-200, SwissDial and the
# Swiss Parliaments corpus, which together cover all major dialect regions
# (Bern, Basel, Zurich, Valais, Grisons, Central and Eastern Switzerland).
# They turn spoken dialect directly into Standard German text.
MODEL_PRESETS = {
    # Default: Whisper large-v3-turbo fine-tuned on Swiss German. Fast, good accuracy.
    "swiss-turbo": "jayr23/whisper-large-v3-turbo-swiss-german-ct2",
    # Same idea, int8 quantised conversion: smaller download, good for CPU-only PCs.
    "swiss-turbo-int8": "Benjam1nP/whisper-large-v3-turbo-swiss-german-ct2-int8",
    # Whisper large-v3 + Swiss German LoRA (MIT). Slower, often more accurate.
    "swiss-large": "gcoli/whisper-large-v3-swiss-german-mit",
    # Plain OpenAI Whisper (no Swiss fine-tune), used as automatic fallback.
    "turbo": "large-v3-turbo",
    "large": "large-v3",
}
DEFAULT_MODEL = "swiss-turbo"
FALLBACK_MODEL = "large-v3-turbo"

# Second model for the "as spoken" dialect line. The Swiss fine-tunes always write
# Standard German, so plain Whisper is used with a dialect-style prompt instead.
DEFAULT_DIALECT_MODEL = "turbo"

# Offline (no API key) German -> Russian translation model.
DEFAULT_OFFLINE_MODEL = "facebook/nllb-200-distilled-600M"

# Optional dialect hint. Recognition works for all dialects automatically; the hint
# is passed to the Claude translator so it can resolve leftover dialect words.
DIALECTS = {
    "auto": None,
    "bern": "Bernese German (Bärndütsch, city and Mittelland)",
    "berner-oberland": "Bernese Oberland dialect (Grindelwald, Interlaken, Highest Alemannic)",
    "basel": "Basel German (Baseldytsch)",
    "zurich": "Zurich German (Züritüütsch)",
    "wallis": "Valais German (Wallisertiitsch, Highest Alemannic)",
    "luzern": "Central Swiss German (Lucerne, Zug, Schwyz, Uri)",
    "ostschweiz": "Eastern Swiss German (St. Gallen, Thurgau, Appenzell)",
    "graubuenden": "Grisons German (Bündnerdeutsch)",
    "aargau": "Aargau German",
    "solothurn": "Solothurn German",
}
