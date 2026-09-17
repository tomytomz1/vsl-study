"""Inspect local video files with ffprobe and extract timeline-aligned audio."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from vsl_study import ffcmd
from vsl_study.cache import fingerprint_file
from vsl_study.models import VideoInfo

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
ProgressCb = Callable[[str, str], None]


def _parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _stream_start(stream: dict[str, Any]) -> float:
    raw = stream.get("start_time")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") or {}
    rotate = tags.get("rotate")
    if rotate is not None:
        try:
            return int(float(rotate)) % 360
        except (TypeError, ValueError):
            pass
    for item in stream.get("side_data_list") or []:
        if not isinstance(item, dict):
            continue
        if item.get("side_data_type") in {"Display Matrix", "DisplayMatrix"}:
            rot = item.get("rotation")
            if rot is not None:
                try:
                    return int(round(float(rot))) % 360
                except (TypeError, ValueError):
                    pass
    return 0


def _is_vfr(stream: dict[str, Any]) -> bool:
    avg = stream.get("avg_frame_rate")
    r = stream.get("r_frame_rate")
    if not avg or not r or avg in {"0/0"} or r in {"0/0"}:
        return False
    a = _parse_fraction(avg)
    b = _parse_fraction(r)
    if a is None or b is None or b == 0:
        return False
    return abs(a - b) / max(b, 1e-6) > 0.02


def inspect_video(path: Path, progress: ProgressCb | None = None) -> VideoInfo:
    if progress:
        progress("inspect", "Reading container streams with ffprobe")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported extension {suffix}. Accepted: {', '.join(sorted(SUPPORTED_SUFFIXES))} "
            "(the actual streams are still validated with ffprobe)."
        )
    result = ffcmd.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(resolved),
        ]
    )
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = probe.get("streams") or []
    fmt = probe.get("format") or {}
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        raise RuntimeError("No video stream found. The file is not a usable input.")

    v = video_streams[0]
    a = audio_streams[0] if audio_streams else None
    try:
        format_start = float(fmt.get("start_time") or 0.0)
    except (TypeError, ValueError):
        format_start = 0.0
    try:
        duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        nb = v.get("nb_frames")
        fps = _parse_fraction(v.get("avg_frame_rate") or v.get("r_frame_rate"))
        try:
            duration = float(nb) / fps if nb and fps else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            duration = 0.0
    if duration <= 0:
        raise RuntimeError("Could not determine video duration from ffprobe.")

    width = int(v["width"]) if v.get("width") else None
    height = int(v["height"]) if v.get("height") else None
    notes: list[str] = []
    vfr = _is_vfr(v)
    if vfr:
        notes.append(
            "Variable frame rate detected. Screenshot times use presentation timestamps "
            "from ffmpeg, not frame_index/average_fps. Scene boundary seconds from "
            "PySceneDetect still use the decoder time base and may differ slightly on VFR."
        )
    video_start = _stream_start(v)
    audio_start = _stream_start(a) if a else None
    if audio_start is not None and abs(audio_start - video_start) > 0.001:
        notes.append(
            f"Audio start_time={audio_start:.3f}s vs video start_time={video_start:.3f}s. "
            "Extracted WAV is silence-padded so Whisper times stay on the video timeline."
        )
    rotation = _rotation(v)
    if rotation:
        notes.append(f"Rotation metadata: {rotation}°. ffmpeg autorotate is used for captures.")

    stat = resolved.stat()
    info = VideoInfo(
        path=str(path),
        resolved_path=str(resolved),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        fingerprint=fingerprint_file(resolved),
        duration_s=duration,
        width=width,
        height=height,
        fps_avg=_parse_fraction(v.get("avg_frame_rate")),
        fps_r=v.get("r_frame_rate"),
        time_base=v.get("time_base"),
        video_start_s=video_start,
        audio_start_s=audio_start,
        format_start_s=format_start,
        rotation=rotation,
        has_audio=bool(a),
        has_video=True,
        vfr=vfr,
        format_name=str(fmt.get("format_name") or ""),
        audio_codec=(a or {}).get("codec_name"),
        video_codec=v.get("codec_name"),
        notes=notes,
        probe=probe,
    )
    return info


def extract_aligned_audio(
    info: VideoInfo,
    wav_path: Path,
    progress: ProgressCb | None = None,
) -> Path:
    if not info.has_audio:
        raise RuntimeError("No audio stream to extract.")
    if progress:
        progress("audio", "Extracting 16 kHz mono WAV aligned to the video timeline")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    delay_s = max(0.0, (info.audio_start_s or 0.0) - info.video_start_s)
    delay_ms = int(round(delay_s * 1000.0))
    filters = ["aresample=16000", "aformat=sample_fmts=s16:channel_layouts=mono"]
    if delay_ms > 0:
        filters.insert(0, f"adelay={delay_ms}:all=1")
    args = [
        "ffmpeg",
        "-y",
        "-i",
        info.resolved_path,
        "-vn",
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    ffcmd.run(args, timeout=None)
    if not wav_path.exists() or wav_path.stat().st_size < 64:
        raise RuntimeError("Audio extraction produced an empty file.")
    return wav_path
