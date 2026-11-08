#!/bin/bash
# Creates "Swiss RU-DE.app" in ~/Applications: a small app bundle that starts swiss-ru-de
# from this folder. macOS then asks for the audio permissions for "Swiss RU-DE".
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
APP="$HOME/Applications/Swiss RU-DE.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$HOME/Library/Logs" || exit 1

cat > "$APP/Contents/MacOS/swiss-ru-de" <<EOF
#!/bin/bash
# Not exec: this process stays the app, so macOS attributes the permissions to it.
cd "$ROOT" || exit 1
"$ROOT/.venv/bin/python" -m swiss_ru_de --gui >> "\$HOME/Library/Logs/swiss-ru-de.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/swiss-ru-de"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Swiss RU-DE</string>
    <key>CFBundleDisplayName</key><string>Swiss RU-DE</string>
    <key>CFBundleIdentifier</key><string>ch.swiss-ru-de.mac</string>
    <key>CFBundleExecutable</key><string>swiss-ru-de</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>swiss-ru-de listens to speech to show subtitles.</string>
    <key>NSAudioCaptureUsageDescription</key><string>swiss-ru-de listens to the system audio to show subtitles.</string>
</dict>
</plist>
EOF

# icon from swiss_ru_de/icon.png
ICONSET="$(mktemp -d)/icon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z $size $size "$ROOT/swiss_ru_de/icon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null 2>&1
    double=$((size * 2))
    sips -z $double $double "$ROOT/swiss_ru_de/icon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns" 2>/dev/null
rm -rf "$(dirname "$ICONSET")"

codesign --force --deep -s - "$APP" >/dev/null 2>&1  # local (ad-hoc) signature
touch "$APP"
echo "App created: $APP"
