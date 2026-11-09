#!/bin/bash
# Installs Ollama (local AI on this Mac) and a translation model sized to the memory, then
# switches swiss-ru-de to "Local AI". Safe to run again.
cd "$(dirname "$0")" || exit 1
NOPAUSE="$1"
done_() { [ "$NOPAUSE" = "nopause" ] || read -r -p "Press Enter to close this window." _; exit "$1"; }

find_ollama() {
    for c in "$(command -v ollama 2>/dev/null)" /opt/homebrew/bin/ollama /usr/local/bin/ollama \
             /Applications/Ollama.app/Contents/Resources/ollama \
             "$HOME/Applications/Ollama.app/Contents/Resources/ollama"; do
        if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
    done
    return 1
}

OLLAMA="$(find_ollama)"
if [ -z "$OLLAMA" ]; then
    echo "=== Installing Ollama ..."
    TMP="$(mktemp -d)"
    curl -L --fail -o "$TMP/Ollama.zip" https://ollama.com/download/Ollama-darwin.zip || { echo "Download failed."; done_ 1; }
    DEST=/Applications
    [ -w "$DEST" ] || DEST="$HOME/Applications"
    mkdir -p "$DEST"
    ditto -x -k "$TMP/Ollama.zip" "$DEST" || { echo "Could not unpack Ollama."; done_ 1; }
    rm -rf "$TMP"
    OLLAMA="$(find_ollama)" || { echo "Ollama was not found after installing."; done_ 1; }
fi
echo "=== Ollama: $OLLAMA"

# start the server (the menu-bar app, or the plain server)
if ! curl -s http://localhost:11434/api/version >/dev/null; then
    open -a Ollama 2>/dev/null || ("$OLLAMA" serve >/dev/null 2>&1 &)
    for _ in $(seq 1 30); do
        curl -s http://localhost:11434/api/version >/dev/null && break
        sleep 1
    done
fi

# Model by memory (a Max has 36 GB or more).
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
# TranslateGemma = Google's translation model: better translations than a general model of
# the same size. From 24 GB (every Max) the 12B version.
MODEL="translategemma:4b"; FALLBACK="gemma3:4b"
if [ "$MEM_GB" -ge 24 ]; then MODEL="translategemma:12b"; FALLBACK="gemma3:12b"; fi
echo "=== Memory: $MEM_GB GB - downloading $MODEL (once) ..."
if ! "$OLLAMA" pull "$MODEL"; then
    echo "$MODEL is not available - using $FALLBACK instead."
    MODEL="$FALLBACK"
    "$OLLAMA" pull "$MODEL" || { echo "Model download failed."; done_ 1; }
fi

if [ -x .venv/bin/python ]; then
    .venv/bin/python -c "from swiss_ru_de.settings import load_settings, save_settings; s = load_settings(); s.translator = 'local'; s.local_model = '$MODEL'; s.local_url = 'http://localhost:11434/v1'; save_settings(s); print('Settings: Translator = Local AI, model = $MODEL')"
fi
echo "Done. Local AI is ready."
done_ 0
