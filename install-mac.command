#!/bin/bash
# =====================================================================
#  swiss-ru-de for Mac - complete installation (double-click in Finder)
#
#   1. uv + Python 3.12 (private to this folder, nothing system-wide)
#   2. swiss-ru-de with MLX (speech recognition on the Apple GPU) and the translators
#   3. audiotap: records the Mac's system audio (Teams, Zoom, YouTube, ...)
#   4. Local AI: Ollama + a translation model sized to the Mac's memory
#   5. Speech models: downloaded and prepared for the Apple GPU
#   6. "Swiss RU-DE" app in your Applications folder, then starts it
#
#  Safe to run again: it updates what is already there.
#  SWISS_SETUP_CI=1 (automated test): no Ollama, no model downloads, no app start.
# =====================================================================
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
ARCH="$(uname -m)"
export PATH="$HOME/.local/bin:$PATH"

step() { echo; echo "=== $*"; }
finish() {
    if [ -z "$SWISS_SETUP_CI" ]; then read -r -p "Press Enter to close this window." _; fi
    exit "$1"
}
fail() { echo; echo "Installation failed: $*"; echo "Fix the problem above, then run install-mac.command again."; finish 1; }

echo
echo " swiss-ru-de for Mac - takes 5 to 20 minutes, downloads up to about 12 GB."
echo " Keep this window open until it says \"Done\"."
[ "$(uname -s)" = "Darwin" ] || fail "this installer is for macOS"
[ "$ARCH" = "arm64" ] || echo " Intel Mac: speech recognition runs on the processor (the Apple GPU needs an M chip)."

# ------------------------------------------------------------------ 1. Python
step "1/6 Python 3.12 (uv)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv (internet connection?)"
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)"; then
    echo "The existing .venv uses an unsupported Python - rebuilding it."
    rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
    uv venv -p 3.12 .venv || fail "could not create the Python environment"
fi

# -------------------------------------------------------------- 2. packages
step "2/6 swiss-ru-de and its packages"
EXTRAS="claude,deepl,llm,offline"
[ "$ARCH" = "arm64" ] && EXTRAS="mac,$EXTRAS"
uv pip install -p .venv/bin/python -e ".[$EXTRAS]" || fail "could not install the packages"

# --------------------------------------------------------- 3. system audio
step "3/6 System audio helper (audiotap)"
if xcrun --find swiftc >/dev/null 2>&1; then
    mkdir -p bin
    if xcrun swiftc -O -target "$ARCH-apple-macos13.0" mac/audiotap.swift -o bin/audiotap; then
        echo "Built bin/audiotap."
    else
        echo "Could not build the system audio helper (see above). Without it, use a microphone or BlackHole."
    fi
else
    echo "Apple's command line tools are needed to build it (free). A window opens now:"
    echo "click Install, wait until it is done, then run install-mac.command again."
    xcode-select --install 2>/dev/null
fi

# ---------------------------------------------------------------- 4. local AI
step "4/6 Local AI (Ollama + translation model)"
if [ -z "$SWISS_SETUP_CI" ]; then
    ./install-local-ai.command nopause || echo "Local AI is not available - choose another translator in Settings."
fi

# ---------------------------------------------------------- 5. speech models
step "5/6 Speech models (about 3 GB, once)"
if [ -z "$SWISS_SETUP_CI" ]; then
    .venv/bin/python -m swiss_ru_de.mlx_asr || echo "Not prepared now - the app does it at its first start."
fi

# --------------------------------------------------------------------- 6. app
step "6/6 App"
./tools/make-app.sh "$ROOT" || echo "Could not create the app - use start-mac.command instead."

echo
echo "Done! Start swiss-ru-de with \"Swiss RU-DE\" in your Applications folder (or Launchpad)."
echo "The first time, macOS asks to allow \"Screen & System Audio Recording\" - allow it,"
echo "then start the app again."
if [ -z "$SWISS_SETUP_CI" ]; then
    open "$HOME/Applications/Swiss RU-DE.app" 2>/dev/null || ./start-mac.command &
fi
finish 0
