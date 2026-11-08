#!/bin/bash
# Starts swiss-ru-de from Terminal (shows its messages - useful when something goes wrong).
# Installs everything first if needed.
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || ./install-mac.command
exec .venv/bin/python -m swiss_ru_de --gui
