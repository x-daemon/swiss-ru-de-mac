#!/bin/bash
# Removes what install-mac.command set up. Models, local AI and settings only if you say so.
cd "$(dirname "$0")" || exit 1
ask() { read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[YyJj] ]]; }

echo "This removes the Swiss RU-DE app, the Python environment (.venv) and the audio helper."
ask "Continue?" || exit 0
rm -rf "$HOME/Applications/Swiss RU-DE.app" .venv bin
echo "Removed."

if ask "Also delete the downloaded speech models (about 5 GB)?"; then
    rm -rf "$HOME/.cache/huggingface/hub/"models--*whisper* "$HOME/.cache/huggingface/hub/"models--*swiss*
    rm -rf "$HOME/Library/Application Support/swiss-ru-de/models"
    echo "Speech models deleted."
fi
if command -v ollama >/dev/null 2>&1 && ask "Also delete the local AI models (gemma3)?"; then
    ollama rm gemma3:12b gemma3:27b gemma3:4b >/dev/null 2>&1
    echo "Deleted. Ollama itself: drag Ollama from Applications to the Trash."
fi
if ask "Also delete your settings?"; then
    rm -rf "$HOME/Library/Application Support/swiss-ru-de"
    echo "Settings deleted. API keys: Keychain Access, entries \"swiss-ru-de\"."
fi
echo "Done. You can now delete this folder."
read -r -p "Press Enter to close." _
