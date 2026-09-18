"""ffprobe / ffmpeg checks for a finished tab recording. Does not invent duration from FPS."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vsl_study import ffcmd
from vsl_study.ffcmd import CommandError


class RecordingInvalid(RuntimeError):
    pass


def _probe(path: Path) -> dict[str, Any]:
    try:
        result = ffcmd.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
    except CommandError as exc:
        raise RecordingInvalid(f"ffprobe could not read the recording: {exc.stderr[-400:]}") from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecordingInvalid("ffprobe returned invalid JSON for the recording.") from exc


def _duration_from_probe(probe: dict[str, Any]) -> float | None:
    fmt = probe.get("format") or {}
    raw = fmt.get("duration")
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value
    for stream in probe.get("streams") or []:
        try:
            stream_dur = float(stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        if stream_dur > 0:
            return stream_dur
    return None


def _decode_sample(path: Path, duration_s: float | None) -> None:
    try:
        ffcmd.run(["ffmpeg", "-v", "error", "-t", "2", "-i", str(path), "-f", "null", "-"], timeout=60)
    except CommandError as exc:
        raise RecordingInvalid(f"The recording could not be decoded: {exc.stderr[-500:]}") from exc
    if duration_s and duration_s > 5:
        try:
            ffcmd.run(
                ["ffmpeg", "-v", "error", "-sseof", "-2", "-i", str(path), "-f", "null", "-"],
                timeout=60,
            )
        except CommandError as exc:
            raise RecordingInvalid(
                f"The last seconds of the recording could not be decoded: {exc.stderr[-500:]}"
            ) from exc


def remux_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffcmd.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c",
            "copy",
            "-fflags",
            "+genpts",
            str(dest),
        ]
    )


def validate_recording(path: Path, *, require_audio: bool = True) -> dict[str, Any]:
    """Return media facts. Remux if duration/seek metadata is missing. Keep the original bytes."""
    src = path.expanduser().resolve()
    if not src.exists() or src.stat().st_size <= 0:
        raise RecordingInvalid("The recording file is missing or empty.")
    problems: list[str] = []
    used = src
    original = src
    probe = _probe(src)
    streams = probe.get("streams") or []
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_video:
        raise RecordingInvalid("The recording has no video stream.")
    duration = _duration_from_probe(probe)
    if duration is None:
        fixed = src.with_name(src.stem + ".fixed" + src.suffix)
        try:
            remux_copy(src, fixed)
            probe = _probe(fixed)
            duration = _duration_from_probe(probe)
            streams = probe.get("streams") or []
            has_video = any(s.get("codec_type") == "video" for s in streams)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            if duration is None:
                problems.append("Duration is still unknown after remux. FPS was not used to invent a length.")
            else:
                used = fixed
                problems.append("Container duration was missing; a copy remux restored seek metadata.")
                orig = src.with_name(src.stem + ".orig" + src.suffix)
                if not orig.exists():
                    shutil.copy2(src, orig)
                original = orig
        except CommandError as exc:
            problems.append(f"Remux failed: {exc.stderr[-300:]}")
            duration = None
    if require_audio and not has_audio:
        raise RecordingInvalid(
            "The recording has no audio track. Choose the tab again and turn on sharing that tab's sound."
        )
    _decode_sample(used, duration)
    mime = str((probe.get("format") or {}).get("format_name") or src.suffix.lstrip("."))
    audio_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
    video_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "video"), None)
    return {
        "path": str(used),
        "original_path": str(original),
        "duration_s": duration,
        "has_video": has_video,
        "has_audio": has_audio,
        "mime_type": mime,
        "audio_codec": audio_codec,
        "video_codec": video_codec,
        "problems": problems,
    }
