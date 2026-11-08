"""Command line entry point: ``swiss-ru-de``."""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import __version__
from .config import (DEFAULT_DIALECT_MODEL, DEFAULT_MODEL, DEFAULT_OFFLINE_MODEL, DIALECTS,
                     MODEL_PRESETS)

EXAMPLES = """\
examples:
  swiss-ru-de --gui                   window: dialect as spoken on top, Russian below
  swiss-ru-de                         live: everything playing on the PC -> Russian
  swiss-ru-de --to de --overlay       live German subtitles in a floating window
  swiss-ru-de --to both               German and Russian
  swiss-ru-de --device Headset        capture a specific speaker/headset (Teams)
  swiss-ru-de --url https://www.youtube.com/watch?v=...   whole video, faster than real time
  swiss-ru-de --file interview.mp4 --save interview.srt
  swiss-ru-de --list-devices
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swiss-ru-de",
        description="Swiss German speech (PC audio, Teams, YouTube) -> Russian or German text.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("input (default: live PC audio)")
    one = src.add_mutually_exclusive_group()
    one.add_argument("--url", help="YouTube or other video URL (downloads the audio track)")
    one.add_argument("--file", help="audio or video file")
    one.add_argument("--mic", action="store_true",
                     help="use a microphone / virtual input instead of the PC speaker output")
    src.add_argument("--device", help="speaker (or with --mic: input) name or part of it")
    src.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    src.add_argument("--benchmark", action="store_true",
                     help="measure the speech model on this PC, show hardware and advice, exit")

    out = p.add_argument_group("output")
    out.add_argument("--gui", action="store_true",
                     help="two-pane window (top: dialect as spoken, bottom: translation); "
                          "uses its own Settings dialog instead of the options below")
    out.add_argument("--to", choices=["ru", "de", "both"], default="ru",
                     help="output language (default: ru)")
    out.add_argument("--dialect-text", action="store_true",
                     help="also show the Swiss German as spoken (dialect spelling, 2nd model)")
    out.add_argument("--overlay", action="store_true", help="floating always-on-top subtitle window")
    out.add_argument("--font-size", type=int, default=22, help="overlay font size")
    out.add_argument("--save", metavar="FILE", help="also write a transcript (.srt or .txt)")

    tr = p.add_argument_group("translation")
    tr.add_argument("--translator", default="auto",
                    choices=["auto", "claude", "openai", "gemini", "grok", "local", "deepl", "offline"],
                    help="auto = the first with a key: Claude (ANTHROPIC_API_KEY), ChatGPT "
                         "(OPENAI_API_KEY), Gemini (GEMINI_API_KEY), Grok (XAI_API_KEY), DeepL "
                         "(DEEPL_AUTH_KEY), else offline")
    tr.add_argument("--claude-model", default=None, help="Claude model id (default claude-opus-5-5)")
    tr.add_argument("--local-url", default="http://localhost:11434/v1",
                    help="--translator local: Ollama (default) or LM Studio (http://localhost:1234/v1)")
    tr.add_argument("--translator-model", default=None,
                    help="model id for --translator openai/gemini/grok (default: the provider's "
                         "fast model, e.g. gpt-6-luna, gemini-3.8-flash, grok-4.7)")
    tr.add_argument("--offline-model", default=DEFAULT_OFFLINE_MODEL,
                    help=f"offline translation model (default {DEFAULT_OFFLINE_MODEL})")
    tr.add_argument("--dialect", choices=list(DIALECTS), default="auto",
                    help="optional dialect hint: improves the dialect spelling (--dialect-text) "
                         "and the Claude translation; recognition handles all dialects anyway")

    asr = p.add_argument_group("speech recognition")
    asr.add_argument("--speed", choices=["auto", "fast", "precise"], default="auto",
                     help="auto (default): tuned to the measured speed of this PC; fast: no "
                          "dialect line, longer segments; precise: dialect line, wider search")
    asr.add_argument("--recognizer", choices=["local", "azure"], default="local",
                     help="local = Whisper on this PC (default); azure = Azure AI Speech de-CH "
                          "in the cloud (fast, needs AZURE_SPEECH_KEY; live audio only)")
    asr.add_argument("--azure-region", default="switzerlandnorth",
                     help="region of your Azure Speech resource (default switzerlandnorth)")
    asr.add_argument("--model", default=DEFAULT_MODEL,
                     help=f"preset ({', '.join(MODEL_PRESETS)}), Hugging Face repo or local folder "
                          f"(default {DEFAULT_MODEL})")
    asr.add_argument("--dialect-model", default=DEFAULT_DIALECT_MODEL,
                     help=f"model for --dialect-text (default {DEFAULT_DIALECT_MODEL})")
    asr.add_argument("--hint", help="names/terms that occur in the audio, e.g. 'Grindelwald, Eiger'")
    dev = asr.add_mutually_exclusive_group()
    dev.add_argument("--cpu", action="store_true", help="force CPU")
    dev.add_argument("--gpu", action="store_true", help="force NVIDIA GPU (CUDA)")
    asr.add_argument("--compute-type", default=None, help="CTranslate2 type, e.g. int8, float16")
    asr.add_argument("--beam", type=int, default=None,
                     help="beam size: higher = a bit more accurate, slower (default live 1, file 5)")
    asr.add_argument("--pause-ms", type=int, default=600,
                     help="pause that ends a sentence in live mode (default 600)")
    asr.add_argument("--max-seconds", type=float, default=15.0,
                     help="longest live segment before it is cut (default 15)")
    asr.add_argument("--min-seconds", type=float, default=None,
                     help="join shorter utterances with the next one: more context, more "
                          "precise and fewer passes (default: from --speed, 0 = off)")

    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def log(msg: str) -> None:
    print(f"» {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .output import ConsolePrinter, TranscriptWriter, setup_console

    setup_console()

    if args.list_devices:
        from .audio import list_devices

        try:
            print(list_devices())
        except RuntimeError as e:
            log(str(e))
            return 1
        return 0

    if args.gui:
        from .gui import run_app

        return run_app(file=args.file, url=args.url)

    if args.benchmark:
        return _benchmark(args)

    from .translate import TranslationError, TranslatorHub

    hub = TranslatorHub(args.translator, args.claude_model, args.offline_model, args.dialect, log=log,
                        local_url=args.local_url,
                        **({f"{args.translator}_model": args.translator_model}
                           if args.translator in ("openai", "gemini", "grok", "local") else {}))
    if args.to != "de":
        try:
            log(f"Translation: {hub.get(args.to).name}")  # fail early, e.g. missing API key
        except TranslationError as e:
            log(str(e))
            return 2

    from .asr import dialect_prompt, make_transcriber as Transcriber

    device = "cpu" if args.cpu else "cuda" if args.gpu else "auto"
    live = not (args.url or args.file)
    cloud = live and args.recognizer == "azure"
    beam = args.beam or (None if live else 5)
    transcriber = None
    tuning = None
    if not cloud:
        transcriber = Transcriber(args.model, device=device, compute_type=args.compute_type,
                                  beam_size=beam, hint=args.hint, log=log)
        if live:
            from .tuning import choose

            tuning = choose(args.speed, transcriber.pass_seconds, transcriber.device)
            if args.beam is None:
                transcriber.beam_size = tuning.beam
            if args.min_seconds is None:
                args.min_seconds = tuning.min_segment_s
            log(tuning.describe(transcriber.pass_seconds))
            if args.dialect_text and not tuning.dialect:
                log("--dialect-text is paused: too slow on this PC (use --speed precise to force it).")
                args.dialect_text = False
    if args.min_seconds is None:
        args.min_seconds = 3.0
    dialect_tr = None
    if args.dialect_text:
        log("Loading dialect model …")
        dialect_tr = Transcriber(args.dialect_model, device=device, compute_type=args.compute_type,
                                 beam_size=beam, hint=args.hint, log=log,
                                 prompt=dialect_prompt(args.dialect))

    sinks = [ConsolePrinter(args.to)]
    writer = TranscriptWriter(args.save, args.to) if args.save else None
    if writer:
        sinks.append(writer)
    overlay = None
    if args.overlay:
        from .overlay import Overlay

        overlay = Overlay(args.to, font_size=args.font_size)
        sinks.append(overlay.push)

    def on_line(line) -> None:
        for sink in sinks:
            sink(line)

    def on_status(msg: str) -> None:
        log(msg)
        if overlay:
            overlay.status(msg)

    from .pipeline import LineBook, Options

    book = LineBook(on_line=on_line)
    # Only ask AI translators for the corrected German when it is printed: half the answer.
    options = Options(target=args.to, dialect=dialect_tr is not None,
                      correct_german=args.to in ("de", "both"))
    try:
        if cloud:
            return _run_cloud(args, dialect_tr, hub, book, options, on_status, overlay)
        if live:
            return _run_live(args, transcriber, dialect_tr, hub, book, options, on_status, overlay)
        return _run_file(args, transcriber, dialect_tr, hub, book, options, on_status, overlay, beam)
    finally:
        if writer:
            writer.close()
            log(f"Transcript saved to {args.save}")


def _run_live(args, transcriber, dialect_tr, hub, book, options, on_status, overlay) -> int:
    from .audio import SoundcardSource
    from .pipeline import LivePipeline
    from .segmenter import Segmenter

    if sys.platform == "darwin":  # system audio via ScreenCaptureKit (or BlackHole)
        from .mac_audio import make_source

        source = make_source(loopback=not args.mic, device=args.device)
    else:
        source = SoundcardSource(loopback=not args.mic, device=args.device)
    segmenter = Segmenter(silence_ms=args.pause_ms, max_segment_s=args.max_seconds,
                          min_segment_s=args.min_seconds)
    pipeline = LivePipeline(source, transcriber, hub.get, book, on_status, options=options,
                            segmenter=segmenter, dialect_transcriber=dialect_tr).start()
    log("Press Ctrl+C to stop." if not overlay else "Esc or double-click the subtitles to stop.")
    try:
        if overlay:
            overlay.run(should_stop=pipeline.stop_event.is_set)
        else:
            while not pipeline.wait(0.5):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 1 if pipeline.error else 0


def _benchmark(args) -> int:
    from .asr import make_transcriber as Transcriber
    from .tuning import choose, report

    device = "cpu" if args.cpu else "cuda" if args.gpu else "auto"
    log("Loading the speech model and timing it …")
    tr = Transcriber(args.model, device=device, compute_type=args.compute_type, log=log)
    for line in report(tr.pass_seconds, tr.device):
        print(line)
    print("Live mode (--speed auto): " + choose("auto", tr.pass_seconds, tr.device).describe(tr.pass_seconds))
    return 0


def _run_cloud(args, dialect_tr, hub, book, options, on_status, overlay) -> int:
    from . import keys
    from .audio import SoundcardSource
    from .cloud import CloudPipeline

    key = keys.get_key("azure")
    if not key:
        log("Set AZURE_SPEECH_KEY (or save the key in the window's Settings).")
        return 2
    phrases = [p.strip() for p in (args.hint or "").split(",") if p.strip()]
    pipeline = CloudPipeline(SoundcardSource(loopback=not args.mic, device=args.device), hub.get,
                             book, on_status, key=key, region=args.azure_region, options=options,
                             dialect_transcriber=dialect_tr, phrases=phrases).start()
    log("Press Ctrl+C to stop." if not overlay else "Esc or double-click the subtitles to stop.")
    try:
        if overlay:
            overlay.run(should_stop=pipeline.stop_event.is_set)
        else:
            while not pipeline.wait(0.5):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 1 if pipeline.error else 0


def _run_file(args, transcriber, dialect_tr, hub, book, options, on_status, overlay, beam) -> int:
    from .pipeline import run_file

    with tempfile.TemporaryDirectory(prefix="swiss-ru-de-") as tmp:
        if args.url:
            from .audio import download_audio

            on_status("Downloading audio …")
            path, title = download_audio(args.url, Path(tmp))
            on_status(f"Transcribing: {title}")
        else:
            path = Path(args.file)
            if not path.exists():
                log(f"File not found: {path}")
                return 2

        stop = threading.Event()
        started = time.monotonic()

        def work() -> None:
            try:
                run_file(transcriber, hub.get, str(path), book, on_status, options=options,
                         beam_size=beam, stop=stop, dialect_transcriber=dialect_tr)
            except Exception as e:
                on_status(f"Failed: {e}")
            finally:
                stop.set()

        if overlay:
            t = threading.Thread(target=work, daemon=True)
            t.start()
            overlay.run()  # keeps the last subtitle visible until closed
            stop.set()
            t.join()
        else:
            try:
                work()
            except KeyboardInterrupt:
                stop.set()
        log(f"Done in {time.monotonic() - started:.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
