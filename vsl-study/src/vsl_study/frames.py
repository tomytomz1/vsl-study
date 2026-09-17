"""Capture actual video frames with ffmpeg presentation timestamps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont

from vsl_study import ffcmd
from vsl_study.models import Scene, ScreenshotRecord, VideoInfo
from vsl_study.timeutil import clamp_time, format_filename_time, format_timecode

ProgressCb = Callable[[str, str], None]
SHOWINFO_PTS = re.compile(r"pts_time:(?P<t>-?\d+(?:\.\d+)?)")


@dataclass
class CaptureCandidate:
    requested_time: float
    reason: str
    scene_id: str | None


def last_safe_time(info: VideoInfo) -> float:
    """Stay at least ~1.5 frames before duration so ffmpeg can decode a real last frame."""
    fps = info.fps_avg if info.fps_avg and info.fps_avg > 1 else 25.0
    margin = max(1.5 / fps, 0.08)
    return max(0.0, info.duration_s - margin)


def scene_for_time(scenes: list[Scene], t: float) -> Scene | None:
    for scene in scenes:
        if scene.start <= t < scene.end or (t == scene.end and scene is scenes[-1]):
            return scene
    if scenes:
        if t < scenes[0].start:
            return scenes[0]
        return scenes[-1]
    return None


def build_candidates(
    info: VideoInfo,
    scenes: list[Scene],
    interval: float,
    scene_start_offset: float,
    extra_times: Iterable[float] = (),
) -> list[CaptureCandidate]:
    duration = info.duration_s
    items: list[CaptureCandidate] = []
    end_cap = last_safe_time(info)
    for scene in scenes:
        span = max(0.0, scene.end - scene.start)
        offset = min(scene_start_offset, max(0.0, span * 0.5))
        t = clamp_time(scene.start + offset, scene.start, max(scene.start, min(scene.end, end_cap)))
        items.append(CaptureCandidate(t, "scene_start", scene.id))
    if interval and interval > 0:
        t = 0.0
        while t < duration - 1e-6:
            scene = scene_for_time(scenes, t)
            items.append(CaptureCandidate(min(t, end_cap), "interval", scene.id if scene else None))
            t += interval
        scene = scene_for_time(scenes, end_cap)
        items.append(CaptureCandidate(end_cap, "interval", scene.id if scene else None))
    for raw in extra_times:
        t = clamp_time(float(raw), 0.0, end_cap)
        scene = scene_for_time(scenes, t)
        items.append(CaptureCandidate(t, "user", scene.id if scene else None))

    # Preserve chronological order; drop exact duplicate times (ms). Keep first reason,
    # but prefer scene_start over interval over user? Spec: preserve chronological
    # candidate list and avoid exact duplicate captures at the same time.
    # Prefer more specific reasons when times collide: user > scene_start > interval
    rank = {"user": 0, "scene_start": 1, "interval": 2}
    by_ms: dict[int, CaptureCandidate] = {}
    for item in items:
        key = int(round(item.requested_time * 1000))
        prev = by_ms.get(key)
        if prev is None or rank.get(item.reason, 9) < rank.get(prev.reason, 9):
            by_ms[key] = item
    ordered = [by_ms[k] for k in sorted(by_ms)]
    return ordered


def _parse_actual_time(stderr: str, requested: float) -> float:
    times = [float(m.group("t")) for m in SHOWINFO_PTS.finditer(stderr or "")]
    times = [t for t in times if t >= 0]
    if not times:
        return requested
    # First decoded/selected frame after the select filter.
    return times[0]


def capture_frame(
    info: VideoInfo,
    requested: float,
    dest: Path,
    max_width: int,
) -> tuple[float, list[str]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    seek_at = clamp_time(requested, 0.0, last_safe_time(info))
    if abs(seek_at - requested) > 0.001:
        notes.append(
            f"clamped requested {requested:.3f}s to {seek_at:.3f}s to stay inside the last decodable frame"
        )

    actual, ok = _try_select_capture(info, seek_at, dest, max_width)
    if not ok:
        notes.append("select-filter capture produced no frame; used -ss fallback")
        actual, ok = _try_ss_capture(info, seek_at, dest)
    if not ok:
        earlier = max(0.0, last_safe_time(info) - 0.2)
        notes.append(f"retrying capture at {earlier:.3f}s")
        actual, ok = _try_select_capture(info, earlier, dest, max_width)
        if not ok:
            actual, ok = _try_ss_capture(info, earlier, dest)
    if not ok or not dest.exists():
        raise RuntimeError(f"Could not capture a frame at {requested:.3f}s from {info.resolved_path}")
    if abs(actual - requested) > 0.25:
        notes.append(f"actual capture time {actual:.3f}s differs from requested {requested:.3f}s")
    if dest.exists() and max_width:
        _ensure_max_width(dest, max_width)
    return actual, notes


def _scale_filter(info: VideoInfo, max_width: int) -> str | None:
    if max_width and info.width and info.width > max_width:
        return f"scale={max_width}:-2:flags=lanczos"
    if max_width and not info.width:
        return f"scale=min({max_width}\\,iw):-2:flags=lanczos"
    return None


def _try_select_capture(
    info: VideoInfo, seek_at: float, dest: Path, max_width: int
) -> tuple[float, bool]:
    vf_parts = [f"select=gte(t\\,{seek_at:.3f})", "showinfo"]
    scale = _scale_filter(info, max_width)
    if scale:
        vf_parts.append(scale)
    result = ffcmd.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            info.resolved_path,
            "-an",
            "-vf",
            ",".join(vf_parts),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-q:v",
            "2",
            str(dest),
        ],
        check=False,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 32:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return seek_at, False
    return _parse_actual_time(result.stderr, seek_at), True


def _try_ss_capture(info: VideoInfo, seek_at: float, dest: Path) -> tuple[float, bool]:
    result = ffcmd.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seek_at:.3f}",
            "-i",
            info.resolved_path,
            "-frames:v",
            "1",
            "-update",
            "1",
            "-q:v",
            "2",
            str(dest),
        ],
        check=False,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 32:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return seek_at, False
    return seek_at, True


def _ensure_max_width(path: Path, max_width: int) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width <= max_width:
            if path.suffix.lower() != ".jpg":
                image.save(path, format="JPEG", quality=90)
            return
        height = max(1, round(image.height * max_width / image.width))
        resized = image.resize((max_width, height), Image.Resampling.LANCZOS)
        resized.save(path, format="JPEG", quality=90)


def ahash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    avg = sum(pixels) / max(len(pixels), 1)
    return "".join("1" if p >= avg else "0" for p in pixels)


def hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) + abs(len(a) - len(b))


def apply_sequential_compact(
    records: list[ScreenshotRecord],
    frames_dir: Path,
    threshold: int = 6,
) -> None:
    """Deduplicate only against the last retained image so later repeats stay."""
    last_id: str | None = None
    last_hash: str | None = None
    for record in records:
        path = frames_dir.parent / record.relative_path
        if not path.exists():
            record.compact_retained = True
            continue
        digest = ahash(path)
        if last_hash is not None and hamming(digest, last_hash) <= threshold:
            record.compact_retained = False
            record.compact_refers_to = last_id
        else:
            record.compact_retained = True
            record.compact_refers_to = None
            last_id = record.id
            last_hash = digest


def write_contact_sheet(
    records: list[ScreenshotRecord],
    job_root: Path,
    dest: Path,
    columns: int = 4,
    thumb_w: int = 320,
) -> Path:
    """Contact sheet with IDs/timestamps in margins, not over the screenshot pixels."""
    items: list[tuple[ScreenshotRecord, Image.Image]] = []
    for record in records:
        path = job_root / record.relative_path
        if not path.exists():
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            ratio = thumb_w / max(rgb.width, 1)
            thumb = rgb.resize((thumb_w, max(1, round(rgb.height * ratio))), Image.Resampling.LANCZOS)
        items.append((record, thumb))
    if not items:
        dest.parent.mkdir(parents=True, exist_ok=True)
        empty = Image.new("RGB", (thumb_w, 80), (245, 245, 245))
        empty.save(dest, format="JPEG", quality=90)
        return dest

    caption_h = 36
    pad = 8
    rows = (len(items) + columns - 1) // columns
    cell_h = max(im.height for _, im in items) + caption_h
    width = columns * (thumb_w + pad) + pad
    height = rows * (cell_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for index, (record, thumb) in enumerate(items):
        row, col = divmod(index, columns)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (cell_h + pad)
        sheet.paste(thumb, (x, y))
        caption = f"{record.id}  {format_timecode(record.actual_time)}"
        draw.rectangle((x, y + thumb.height, x + thumb_w, y + thumb.height + caption_h), fill=(255, 255, 255))
        draw.text((x + 4, y + thumb.height + 8), caption, fill=(20, 20, 20), font=font)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=90)
    return dest


def capture_candidates(
    info: VideoInfo,
    scenes: list[Scene],
    job_frames: Path,
    max_width: int,
    candidates: list[CaptureCandidate],
    existing: dict[str, ScreenshotRecord] | None = None,
    progress: ProgressCb | None = None,
) -> list[ScreenshotRecord]:
    existing = existing or {}
    records: list[ScreenshotRecord] = []
    for index, cand in enumerate(candidates, start=1):
        frame_id = f"frame_{index:04d}"
        # Reuse by requested ms if a previous capture exists on disk.
        reuse_key = f"{cand.reason}:{cand.requested_time:.3f}"
        prior = existing.get(reuse_key)
        filename = f"{frame_id}_t{format_filename_time(cand.requested_time)}.jpg"
        dest = job_frames / filename
        if progress and (index == 1 or index % 10 == 0 or index == len(candidates)):
            progress("frames", f"Capturing {index}/{len(candidates)}: {format_timecode(cand.requested_time)}")
        if prior and (job_frames.parent / prior.relative_path).exists():
            src = job_frames.parent / prior.relative_path
            if src != dest:
                dest.write_bytes(src.read_bytes())
            actual = prior.actual_time
            notes = list(prior.notes) + ["reused prior capture"]
        else:
            actual, notes = capture_frame(info, cand.requested_time, dest, max_width)
        rel = dest.relative_to(job_frames.parent).as_posix()
        records.append(
            ScreenshotRecord(
                id=frame_id,
                requested_time=cand.requested_time,
                actual_time=actual,
                scene_id=cand.scene_id,
                relative_path=rel,
                capture_reason=cand.reason,
                notes=notes,
            )
        )
    return records
