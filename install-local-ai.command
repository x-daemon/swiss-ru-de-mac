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

# Model by memory. A Max has 36 GB or more: gemma3:12b is precise and still fast (about
# 1-2 s per sentence); gemma3:27b (64 GB+) is even more precise but slower.
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
MODEL="gemma3:4b"
[ "$MEM_GB" -ge 24 ] && MODEL="gemma3:12b"
echo "=== Memory: $MEM_GB GB - downloading $MODEL (once) ..."
"$OLLAMA" pull "$MODEL" || { echo "Model download failed."; done_ 1; }

if [ -x .venv/bin/python ]; then
    .venv/bin/python -c "from swiss_ru_de.settings import load_settings, save_settings; s = load_settings(); s.translator = 'local'; s.local_model = '$MODEL'; s.local_url = 'http://localhost:11434/v1'; save_settings(s); print('Settings: Translator = Local AI, model = $MODEL')"
fi
echo "Done. Local AI is ready."
done_ 0
