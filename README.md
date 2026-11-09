# swiss-ru-de for Mac

Live subtitles for **Swiss German** speech on a Mac (Teams, Zoom, YouTube, any app). It shows
the dialect as spoken on top and **Russian** (or Standard German) below.

This edition is built for **Apple Silicon**, especially the **M-series Max** chips of the MacBook Pro:

* **Speech recognition runs on the Apple GPU** with Apple's [MLX](https://github.com/ml-explore/mlx).
  This uses a Swiss German Whisper model (large-v3-turbo, converted for MLX automatically).
* **Translation runs on the same Mac** with a local AI model through [Ollama](https://ollama.com).
  A Max has enough memory for `translategemma:12b`, Google's translation model, which is much
  more precise than the small models a typical PC graphics card can hold.
* **System audio is recorded directly** with Apple's ScreenCaptureKit. You don't need a virtual
  audio device, and the sound keeps playing normally.

Windows and Linux version: [x-daemon/swiss-ru-de](https://github.com/x-daemon/swiss-ru-de).

## Install (one double-click)

1. Download this repository (*Code → Download ZIP*) and unzip it, e.g. into your home folder.
2. Double-click **`install-mac.command`**. If macOS says it is from an unidentified developer,
   right-click it → *Open* → *Open*. The installer sets up:
   1. Python 3.12 via [uv](https://docs.astral.sh/uv/), private to this folder;
   2. the app with MLX and all translators;
   3. `audiotap`, the system-audio helper. It is built from `mac/audiotap.swift` and needs
      Apple's free command line tools; the installer offers to install them;
   4. **Ollama** and a translation model sized to the memory: `translategemma:12b` (Google's
      translation model) from 24 GB, else `translategemma:4b`. The settings are switched to
      *Local AI*;
   5. the speech models (about 3 GB), prepared for the Apple GPU;
   6. **Swiss RU-DE** in your *Applications* folder, which is then started.
3. On the first start, macOS asks to allow **Screen & System Audio Recording** for Swiss
   RU-DE. Allow it (*System Settings → Privacy & Security*), then start the app again.

Running `install-mac.command` again updates everything. `start-mac.command` starts the app
from Terminal and shows its messages, which is useful when something goes wrong.
`uninstall-mac.command` removes the app and, if you choose, the models and settings.

## Use

* **Top pane:** Swiss German as spoken (dialect spelling). **Bottom pane:** Russian, or Standard
  German (*Settings*).
* *Settings → Audio source*: **PC audio** = everything the Mac plays; **mic** = a microphone.
* *Settings → Translator*: *Local AI* (default after installing, free, private), or Claude,
  ChatGPT, Gemini, Grok or DeepL with your own API key (stored in the macOS Keychain).
* *Settings → Run on*: *auto* / *mlx* = Apple GPU; *cpu* = processor (faster-whisper).
* *Settings → Check this PC…* shows the speed of each step.

## Speed on an M-series Max

| Step | Where | Time per sentence |
|---|---|---|
| Speech recognition (Swiss German model) | Apple GPU (MLX) | a fraction of a second |
| Dialect line (second model) | Apple GPU (MLX) | a fraction of a second |
| Translation, `translategemma:12b` | Apple GPU (Ollama, Metal) | about 1–2 s |

These are estimates; *Check this PC* shows the real numbers. TranslateGemma only translates.
To have the German corrected as well (top part = Standard German), use `gemma3:12b`, or
`gemma3:27b` with 48 GB of memory or more (`ollama pull <name>`, then *Settings → Model*).

## How it works

* `swiss_ru_de/mlx_asr.py`: Whisper on MLX.
  * `turbo` (the dialect line) is `mlx-community/whisper-large-v3-turbo`.
  * `swiss-turbo` is [Flurin17/whisper-large-v3-turbo-swiss-german](https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german),
    converted from Transformers format to MLX on first use. The result is kept in
    `~/Library/Application Support/swiss-ru-de/models`.
  * If MLX or a model is not available, recognition falls back to faster-whisper on the
    processor, using the same Swiss German model as the Windows version.
* `mac/audiotap.swift`: ScreenCaptureKit system-audio capture. It writes 16 kHz mono samples
  to standard output, and `swiss_ru_de/mac_audio.py` reads them. Without the helper, a
  [BlackHole](https://github.com/ExistentialAudio/BlackHole) input is used if installed.
* The rest is shared with the Windows version:
  * each line is split at pauses into utterances, so no short answer is skipped;
  * Swiss words are normalised (Gipfeli → croissant, Zmorge → breakfast);
  * sentences are numbered for the AI, so none is dropped;
  * the delay is measured and reported.

## Command line

```bash
.venv/bin/swiss-ru-de --to ru                       # live, Mac system audio
.venv/bin/swiss-ru-de --to ru --translator local --translator-model gemma3:12b
.venv/bin/swiss-ru-de --file interview.mp3 --to ru --save interview.srt
.venv/bin/swiss-ru-de --benchmark                   # speed of the speech model here
```

## Troubleshooting

* **No subtitles, status "macOS did not allow recording the system audio":** allow Swiss RU-DE,
  or Terminal if you use `start-mac.command`, in *System Settings → Privacy & Security →
  Screen & System Audio Recording*. Then quit and start it again.
* **"The system-audio helper is not built":** install Apple's command line tools
  (`xcode-select --install`), then run `install-mac.command` again.
* **Translation slow:** *Check this PC* shows the time per step. Use a 12B rather than a 27B
  model, and close other big apps.
* **Log file:** `~/Library/Logs/swiss-ru-de.log`.

## Кратко по-русски

Программа показывает субтитры для швейцарского немецкого: всё, что звучит на Mac (Teams,
Zoom, YouTube). Сверху диалект, снизу русский перевод. Версия рассчитана на MacBook Pro с
чипом M Max:

* распознавание речи идёт на графическом процессоре Apple (MLX);
* перевод делает локальный ИИ `translategemma:12b` (переводческая модель Google) прямо на
  Mac, бесплатно и без облака;
* звук записывается напрямую, без виртуальных аудиоустройств.

Установка: распакуйте архив и дважды щёлкните **`install-mac.command`**. Если macOS
предупреждает о неизвестном разработчике, щёлкните правой кнопкой → «Открыть». При первом
запуске разрешите **«Запись экрана и системного звука»** для Swiss RU-DE (Системные настройки →
Конфиденциальность и безопасность) и запустите программу ещё раз.

## License

MIT. The speech and translation models have their own licences; check them on their model pages
before commercial use.
