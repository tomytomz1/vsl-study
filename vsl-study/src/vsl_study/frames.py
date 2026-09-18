"""Capture actual video frames with ffmpeg presentation timestamps."""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from PIL import Image, ImageDraw, ImageFont

from vsl_study import ffcmd
from vsl_study.models import Scene, ScreenshotRecord, VideoInfo
from vsl_study.timeutil import clamp_time, format_filename_time, format_timecode

ProgressCb = Callable[[str, str], None]
SHOWINFO_PTS = re.compile(r"pts_time:(?P<t>-?\d+(?:\.\d+)?)")
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


@dataclass
class CaptureCandidate:
    requested_time: float
    reason: str
    scene_id: str | None


def image_end_time(info: VideoInfo) -> float:
    """Last time at which a real video image can exist.

    Container/audio duration is not used as the image boundary when a video
    stream duration is known. Untrusted 1000-FPS metadata is not used as a
    frame-step margin.
    """
    video_end = info.video_duration_s
    if video_end is not None and video_end > 0:
        return min(float(video_end), info.duration_s)
    if info.fps_trusted:
        fps = info.fps_avg if info.fps_avg and info.fps_avg > 1 else 25.0
        margin = max(1.5 / fps, 0.08)
        return max(0.0, info.duration_s - margin)
    return max(0.0, info.duration_s)


def last_safe_time(info: VideoInfo) -> float:
    return image_end_time(info)


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
    duration = image_end_time(info)
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


def _parse_actual_time(stderr: str, requested: float) -> float | None:
    times = [float(m.group("t")) for m in SHOWINFO_PTS.finditer(stderr or "")]
    times = [t for t in times if t >= 0]
    if not times:
        return None
    return times[0]


def capture_frame(
    info: VideoInfo,
    requested: float,
    dest: Path,
    max_width: int,
) -> tuple[float, list[str]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    end_cap = last_safe_time(info)
    seek_at = clamp_time(requested, 0.0, end_cap)
    if abs(seek_at - requested) > 0.001:
        notes.append(
            f"clamped requested {requested:.3f}s to {seek_at:.3f}s to stay inside the last video frame"
        )

    actual, ok = _try_bounded_capture(info, seek_at, dest, max_width)
    if not ok:
        notes.append("bounded seek produced no frame; decoding from the last known video end")
        actual, ok = _try_bounded_capture(info, end_cap, dest, max_width)
    if not ok or not dest.exists():
        raise RuntimeError(f"Could not capture a frame at {requested:.3f}s from {info.resolved_path}")
    if actual is None:
        notes.append(
            f"ffmpeg did not report a presentation timestamp; not labeling requested {seek_at:.3f}s as observed"
        )
        actual = seek_at
        notes.append("actual_time falls back to the clamped request because PTS was unavailable")
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


def _try_bounded_capture(
    info: VideoInfo, seek_at: float, dest: Path, max_width: int
) -> tuple[float | None, bool]:
    """Seek near the target, then decode forward. Does not restart from t=0 unless the target is near 0."""
    preroll = min(seek_at, 3.0)
    ss = max(0.0, seek_at - preroll)
    vf_parts = [f"select=gte(t\\,{seek_at:.3f})", "showinfo"]
    scale = _scale_filter(info, max_width)
    if scale:
        vf_parts.append(scale)
    args = ["ffmpeg", "-y"]
    if ss > 0.001:
        args.extend(["-ss", f"{ss:.3f}"])
    args.extend(
        [
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
        ]
    )
    result = ffcmd.run(args, check=False)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 32:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return None, False
    return _parse_actual_time(result.stderr, seek_at), True


def _try_select_capture(
    info: VideoInfo, seek_at: float, dest: Path, max_width: int
) -> tuple[float | None, bool]:
    return _try_bounded_capture(info, seek_at, dest, max_width)


class _JpegPipe:
    def __init__(self, stream):
        self.stream = stream
        self.buf = b""

    def read_one(self) -> bytes | None:
        while True:
            start = self.buf.find(JPEG_SOI)
            if start >= 0:
                end = self.buf.find(JPEG_EOI, start + 2)
                if end >= 0:
                    jpeg = self.buf[start : end + 2]
                    self.buf = self.buf[end + 2 :]
                    return jpeg
                if start > 0:
                    self.buf = self.buf[start:]
            chunk = self.stream.read(65536)
            if not chunk:
                return None
            self.buf += chunk


def _select_expr(timestamps: list[float]) -> str:
    parts = []
    for index, t in enumerate(timestamps):
        parts.append(f"gte(t\\,{t:.3f})*eq(selected_n\\,{index})")
    return "+".join(parts)


def iter_decoded_frames(
    info: VideoInfo,
    max_width: int,
    timestamps: list[float] | None = None,
    start_s: float | None = None,
) -> Iterator[tuple[float, bytes]]:
    """One decode pass. If timestamps are given, only those frames are encoded.

    start_s, when set, is an input seek. -copyts keeps presentation timestamps
    on the original recording timeline.
    """
    vf_parts: list[str] = []
    if timestamps:
        vf_parts.append(f"select={_select_expr(timestamps)}")
    vf_parts.append("showinfo")
    scale = _scale_filter(info, max_width)
    if scale:
        vf_parts.append(scale)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
    ]
    if start_s is not None and start_s > 0:
        args.extend(["-ss", f"{start_s:.3f}", "-copyts"])
    args.extend(
        [
            "-i",
            info.resolved_path,
            "-an",
            "-vsync",
            "0",
            "-vf",
            ",".join(vf_parts),
        ]
    )
    if timestamps:
        args.extend(["-frames:v", str(len(timestamps))])
    args.extend(
        [
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "pipe:1",
        ]
    )
    proc = ffcmd.popen(args)
    pts_q: queue.Queue[float | object] = queue.Queue()
    sentinel = object()

    def read_stderr() -> None:
        assert proc.stderr is not None
        leftover = ""
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            leftover += chunk.decode("utf-8", "replace")
            while "\n" in leftover:
                line, leftover = leftover.split("\n", 1)
                match = SHOWINFO_PTS.search(line)
                if match:
                    t = float(match.group("t"))
                    if t >= 0:
                        pts_q.put(t)
        pts_q.put(sentinel)

    thread = threading.Thread(target=read_stderr, daemon=True, name="ffmpeg-showinfo")
    thread.start()
    try:
        assert proc.stdout is not None
        reader = _JpegPipe(proc.stdout)
        while True:
            jpeg = reader.read_one()
            if jpeg is None:
                break
            pts = pts_q.get(timeout=120)
            if pts is sentinel:
                break
            yield float(pts), jpeg
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=8)
        except Exception:
            pass
        thread.join(timeout=8)


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
    frame_source: Callable[[VideoInfo, int], Iterable[tuple[float, bytes]]] | None = None,
) -> list[ScreenshotRecord]:
    existing = existing or {}
    job_frames.mkdir(parents=True, exist_ok=True)
    records: list[ScreenshotRecord | None] = [None] * len(candidates)
    pending: list[int] = []
    source = frame_source or iter_decoded_frames

    for index, cand in enumerate(candidates):
        reuse_key = f"{cand.reason}:{cand.requested_time:.3f}"
        prior = existing.get(reuse_key)
        frame_id = f"frame_{index + 1:04d}"
        filename = f"{frame_id}_t{format_filename_time(cand.requested_time)}.jpg"
        dest = job_frames / filename
        if prior and (job_frames.parent / prior.relative_path).exists():
            src = job_frames.parent / prior.relative_path
            if src != dest:
                dest.write_bytes(src.read_bytes())
            rel = dest.relative_to(job_frames.parent).as_posix()
            records[index] = ScreenshotRecord(
                id=frame_id,
                requested_time=cand.requested_time,
                actual_time=prior.actual_time,
                scene_id=cand.scene_id,
                relative_path=rel,
                capture_reason=cand.reason,
                notes=list(prior.notes) + ["reused prior capture"],
            )
        else:
            pending.append(index)

    if pending:
        if progress:
            progress("frames", f"Decoding video once for {len(pending)} screenshots")
        _fill_from_sequential_decode(
            info,
            candidates,
            pending,
            records,
            job_frames,
            max_width,
            source,
            progress,
        )

    finished = [rec for rec in records if rec is not None]
    if len(finished) != len(candidates):
        missing = [i for i, rec in enumerate(records) if rec is None]
        raise RuntimeError(
            f"Could not capture {len(missing)} requested screenshot(s); decoded no usable video frame."
        )
    return finished


def _fill_from_sequential_decode(
    info: VideoInfo,
    candidates: list[CaptureCandidate],
    pending: list[int],
    records: list[ScreenshotRecord | None],
    job_frames: Path,
    max_width: int,
    source: Callable[[VideoInfo, int], Iterable[tuple[float, bytes]]],
    progress: ProgressCb | None,
) -> None:
    pending_i = 0
    last_jpeg: bytes | None = None
    last_pts: float | None = None
    decoded = 0
    times = [candidates[i].requested_time for i in pending]
    if source is iter_decoded_frames:
        stream = iter_decoded_frames(info, max_width, times)
    else:
        stream = source(info, max_width)
    for pts, jpeg in stream:
        decoded += 1
        last_jpeg, last_pts = jpeg, pts
        while pending_i < len(pending) and pts + 1e-4 >= candidates[pending[pending_i]].requested_time:
            idx = pending[pending_i]
            _write_shot(
                candidates[idx],
                idx,
                pts,
                jpeg,
                job_frames,
                records,
                notes=[],
            )
            if progress:
                done = sum(1 for rec in records if rec is not None)
                progress(
                    "frames",
                    f"Captured {done}/{len(candidates)} at {format_timecode(pts)}",
                )
            pending_i += 1
        if pending_i >= len(pending):
            break
    if pending_i < len(pending) and source is iter_decoded_frames:
        last_jpeg, last_pts, extra = _scan_true_last_frame(info, max_width, last_jpeg, last_pts)
        decoded += extra
        if extra and progress:
            progress("frames", f"Last available video frame is {format_timecode(last_pts or 0.0)}")
    while pending_i < len(pending):
        if last_jpeg is None or last_pts is None:
            break
        idx = pending[pending_i]
        cand = candidates[idx]
        notes = [
            f"requested {cand.requested_time:.3f}s is after the last available video frame at {last_pts:.3f}s; "
            "used that last frame"
        ]
        _write_shot(cand, idx, last_pts, last_jpeg, job_frames, records, notes)
        if progress:
            done = sum(1 for rec in records if rec is not None)
            progress("frames", f"Captured {done}/{len(candidates)} (last video frame)")
        pending_i += 1
    if decoded == 0 and pending:
        raise RuntimeError(f"Could not decode any video frames from {info.resolved_path}")


def _scan_true_last_frame(
    info: VideoInfo,
    max_width: int,
    last_jpeg: bytes | None,
    last_pts: float | None,
) -> tuple[bytes | None, float | None, int]:
    """Select-filter leftover is the last *selected* frame, not the file's last frame."""
    start = 0.0 if last_pts is None else max(0.0, last_pts - 0.5)
    extra = 0
    for pts, jpeg in iter_decoded_frames(info, max_width, start_s=start):
        extra += 1
        if last_pts is None or pts + 1e-4 >= last_pts:
            last_jpeg, last_pts = jpeg, pts
    return last_jpeg, last_pts, extra


def _write_shot(
    cand: CaptureCandidate,
    index: int,
    actual: float,
    jpeg: bytes,
    job_frames: Path,
    records: list[ScreenshotRecord | None],
    notes: list[str],
) -> None:
    frame_id = f"frame_{index + 1:04d}"
    dest = job_frames / f"{frame_id}_t{format_filename_time(cand.requested_time)}.jpg"
    dest.write_bytes(jpeg)
    if abs(actual - cand.requested_time) > 0.25:
        notes = list(notes) + [
            f"actual capture time {actual:.3f}s differs from requested {cand.requested_time:.3f}s"
        ]
    rel = dest.relative_to(job_frames.parent).as_posix()
    records[index] = ScreenshotRecord(
        id=frame_id,
        requested_time=cand.requested_time,
        actual_time=actual,
        scene_id=cand.scene_id,
        relative_path=rel,
        capture_reason=cand.reason,
        notes=notes,
    )
