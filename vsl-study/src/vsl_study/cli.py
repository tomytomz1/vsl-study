"""Command-line interface for VSL Study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vsl_study",
        description="Turn a local video into a timestamped transcript, screenshots, and an evidence package.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check FFmpeg, Whisper, and optional OCR")

    p_proc = sub.add_parser("process", help="Inspect, transcribe, screenshot, and export")
    p_proc.add_argument("--input", required=True, help="Path to a local MP4/MOV/MKV/WebM file")
    p_proc.add_argument("--output", required=True, help="Job output directory")
    p_proc.add_argument("--model", default="small.en")
    p_proc.add_argument("--language", default="en", help="Original language. Default transcribes; does not translate.")
    p_proc.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p_proc.add_argument("--detector", default="adaptive", choices=["adaptive", "content"])
    p_proc.add_argument("--interval", type=float, default=5.0, help="Backup screenshot interval in seconds")
    p_proc.add_argument("--scene-start-offset", type=float, default=0.25)
    p_proc.add_argument("--max-width", type=int, default=1280)
    p_proc.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read on-screen text from screenshots (default: on). Use --no-ocr to skip.",
    )
    p_proc.add_argument("--no-compact", action="store_true")
    p_proc.add_argument("--include-media", action="store_true")
    p_proc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_proc.add_argument("--context-window", type=float, default=2.0)

    p_frames = sub.add_parser("frames", help="Extract extra frames from an existing job without retranscribing")
    p_frames.add_argument("--job", required=True, help="Existing job directory")
    p_frames.add_argument(
        "--at",
        nargs="+",
        required=True,
        help="Timestamps: 00:02:15.500 or seconds",
    )

    p_ui = sub.add_parser("ui", help="Launch the optional Streamlit browser interface")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab.",
    )
    sub.add_parser("app", help="Open the Windows desktop app (file/folder pickers)")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        from vsl_study.doctor import collect_checks, format_report, required_ok

        print(format_report(collect_checks()), end="")
        ok, _ = required_ok()
        return 0 if ok else 1
    if args.command == "app":
        from vsl_study.desktop import main as desktop_main

        return desktop_main()
    if args.command == "ui":
        return _launch_ui(args.port, open_browser=not args.no_browser)
    if args.command == "process":
        return _cmd_process(args)
    if args.command == "frames":
        return _cmd_frames(args)
    parser.error("unknown command")
    return 2


def _print_progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def _cmd_process(args: argparse.Namespace) -> int:
    from vsl_study.doctor import format_report, required_ok
    from vsl_study.models import ProcessSettings
    from vsl_study.pipeline import PipelineError, process_video
    from vsl_study.transcribe import SettingsError

    ok, checks = required_ok()
    if not ok:
        print(format_report(checks), end="")
        print("Required dependencies are missing. Fix them before process.", file=sys.stderr)
        return 1
    settings = ProcessSettings(
        model=args.model,
        language=args.language,
        task=args.task,
        detector=args.detector,
        interval=args.interval,
        scene_start_offset=args.scene_start_offset,
        max_width=args.max_width,
        ocr=args.ocr,
        context_window_s=args.context_window,
        compact_view=not args.no_compact,
        include_media=args.include_media,
        device=args.device,
    )
    try:
        result = process_video(
            args.input,
            args.output,
            settings=settings,
            progress=_print_progress,
        )
    except (SettingsError, FileNotFoundError, ValueError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"job: {result['job']}")
    print(f"manifest: {result['manifest']}")
    print(f"zip: {result['zip']}")
    print(f"transcript: {result['transcript_status']}")
    print(f"scenes: {result['scene_count']}  screenshots: {result['screenshot_count']}")
    if result["transcript_status"] != "complete":
        print("Transcription was not complete. Visual outputs were still written.")
    return 0


def _cmd_frames(args: argparse.Namespace) -> int:
    from vsl_study.pipeline import PipelineError, add_frames_at
    from vsl_study.timeutil import parse_timecode

    try:
        times = [parse_timecode(value) for value in args.at]
        result = add_frames_at(args.job, times, progress=_print_progress)
    except (ValueError, PipelineError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"job: {result['job']}")
    print(f"screenshots: {result['screenshot_count']}")
    return 0


def _launch_ui(port: int, open_browser: bool = True) -> int:
    app = Path(__file__).with_name("app.py")
    from streamlit.web import cli as stcli

    # Headless skips Streamlit's first-run email prompt; credentials.toml also covers that.
    # When open_browser is True we still start headless and open the tab ourselves so a
    # double-click shortcut does not hang on stdin.
    import webbrowser
    import threading

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
