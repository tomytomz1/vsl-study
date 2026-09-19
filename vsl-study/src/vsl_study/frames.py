"""Capture actual video frames with ffmpeg presentation timestamps."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from PIL import Image, ImageDraw, ImageFont

from vsl_study import ffcmd
from vsl_study.models import Scene, ScreenshotRecord, VideoInfo
from vsl_study.timeutil import clamp_time, format_filename_time, format_timecode

ProgressCb = Callable[[str, str], None]
SHOWINFO_PTS = re.compile(r"pts_time:(?P<t>-?\d+(?:\.\d+)?)")
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
IO_TIMEOUT_S = 120.0
STDERR_TAIL_CHARS = 8000
WIN32_CMDLINE_LIMIT = 32767
# Recorder UI (`recorder/index.html`) allows at most 180 minutes.
RECORDER_MAX_DURATION_S = 180 * 60
_JPEG_END = object()
_PTS_END = object()


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


class DecodeError(RuntimeError):
    """ffmpeg exited unsuccessfully or produced unusable frame data."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr_tail: str = ""):
        self.returncode = returncode
        self.stderr_tail = stderr_tail or ""
        parts = [message]
        if returncode not in (None, 0):
            parts.append(f"ffmpeg exit {returncode}")
        tail = self.stderr_tail.strip()
        if tail:
            parts.append(tail[-1500:])
        super().__init__("\n".join(parts))


class _JpegPipe:
    def __init__(self, stream):
        self.stream = stream
        self.buf = b""

    @property
    def incomplete(self) -> bool:
        start = self.buf.find(JPEG_SOI)
        if start < 0:
            return False
        return self.buf.find(JPEG_EOI, start + 2) < 0

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


def _filter_graph(info: VideoInfo, max_width: int, timestamps: list[float] | None) -> str:
    vf_parts: list[str] = []
    if timestamps:
        vf_parts.append(f"select={_select_expr(timestamps)}")
    vf_parts.append("showinfo")
    scale = _scale_filter(info, max_width)
    if scale:
        vf_parts.append(scale)
    return ",".join(vf_parts)


def write_filter_script(graph: str, directory: str | Path | None = None) -> Path:
    fd, name = tempfile.mkstemp(prefix="vsl-study-vf-", suffix=".txt", dir=directory, text=True)
    path = Path(name)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(graph)
    return path


def serialized_command_length(args: Sequence[str]) -> int:
    """Length of the command as Windows CreateProcess would see it."""
    if hasattr(subprocess, "list2cmdline"):
        return len(subprocess.list2cmdline(list(args)))
    return sum(len(str(part)) + 3 for part in args)


def build_decode_command(
    info: VideoInfo,
    max_width: int,
    timestamps: list[float] | None = None,
    start_s: float | None = None,
    filter_script: str | Path | None = None,
) -> list[str]:
    """Argument vector for one sequential decode. Filter graphs go in a file."""
    graph = _filter_graph(info, max_width, timestamps)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
    ]
    if start_s is not None and start_s > 0:
        args.extend(["-ss", f"{start_s:.3f}", "-copyts"])
    args.extend(["-i", info.resolved_path, "-an", "-vsync", "0"])
    if filter_script is None:
        args.extend(["-vf", graph])
    else:
        args.extend(["-filter_script:v", os.fspath(filter_script)])
    if timestamps:
        args.extend(["-frames:v", str(len(timestamps))])
    args.extend(["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "pipe:1"])
    return args


class FrameDecoder:
    """One ffmpeg process. Distinguishes EOF, intentional stop, and decoder failure."""

    def __init__(self, args: Sequence[str]):
        self.args = list(args)
        self._proc = ffcmd.popen(self.args)
        self._jpeg_q: queue.Queue[object] = queue.Queue()
        self._pts_q: queue.Queue[object] = queue.Queue()
        self._stderr_tail = ""
        self._truncated = False
        self._thread_error: BaseException | None = None
        self._intentional_stop = False
        self._closed = False
        self._out_thread = threading.Thread(target=self._read_stdout, daemon=True, name="ffmpeg-jpeg")
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True, name="ffmpeg-showinfo")
        self._out_thread.start()
        self._err_thread.start()

    def request_stop(self) -> None:
        """Caller cancelled decoding on purpose (generator close / job cancel)."""
        self._intentional_stop = True
        self._kill()

    def frames(self) -> Iterator[tuple[float, bytes]]:
        while True:
            jpeg = self._next_jpeg()
            if jpeg is None:
                return
            pts = self._next_pts()
            if pts is None:
                raise self._error("decoded a JPEG without a presentation timestamp")
            yield pts, jpeg

    def finish(self) -> None:
        """Wait for ffmpeg after stdout ended. Requested JPEGs are not proof of success."""
        self._join_io()
        rc = self._wait_proc()
        if self._thread_error is not None:
            raise self._error(f"decoder reader failed: {self._thread_error}") from self._thread_error
        if self._truncated:
            raise self._error("truncated JPEG output from ffmpeg")
        if self._intentional_stop:
            return
        if rc not in (0, None):
            raise self._error("ffmpeg decoder failed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._kill()
        self._close_pipes()
        self._join_io()
        self._wait_proc()

    def _next_jpeg(self) -> bytes | None:
        try:
            item = self._jpeg_q.get(timeout=IO_TIMEOUT_S)
        except queue.Empty as exc:
            self._kill()
            raise self._error("timed out waiting for decoded JPEG data") from exc
        if isinstance(item, BaseException):
            self._thread_error = item
            raise self._error(f"JPEG reader failed: {item}") from item
        if item is _JPEG_END:
            return None
        return item  # type: ignore[return-value]

    def _next_pts(self) -> float | None:
        try:
            item = self._pts_q.get(timeout=IO_TIMEOUT_S)
        except queue.Empty as exc:
            self._kill()
            raise self._error("timed out waiting for a presentation timestamp") from exc
        if isinstance(item, BaseException):
            self._thread_error = item
            raise self._error(f"stderr reader failed: {item}") from item
        if item is _PTS_END:
            return None
        return float(item)

    def _read_stdout(self) -> None:
        try:
            assert self._proc.stdout is not None
            reader = _JpegPipe(self._proc.stdout)
            while True:
                jpeg = reader.read_one()
                if jpeg is None:
                    break
                self._jpeg_q.put(jpeg)
            self._truncated = reader.incomplete
            self._jpeg_q.put(_JPEG_END)
        except Exception as exc:  # noqa: BLE001
            self._jpeg_q.put(exc)

    def _read_stderr(self) -> None:
        try:
            assert self._proc.stderr is not None
            leftover = ""
            while True:
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", "replace")
                self._stderr_tail = (self._stderr_tail + text)[-STDERR_TAIL_CHARS:]
                leftover += text
                while "\n" in leftover:
                    line, leftover = leftover.split("\n", 1)
                    self._offer_pts(line)
            self._offer_pts(leftover)
            self._pts_q.put(_PTS_END)
        except Exception as exc:  # noqa: BLE001
            self._pts_q.put(exc)

    def _offer_pts(self, line: str) -> None:
        match = SHOWINFO_PTS.search(line)
        if not match:
            return
        t = float(match.group("t"))
        if t >= 0:
            self._pts_q.put(t)

    def _join_io(self) -> None:
        self._out_thread.join(timeout=8)
        self._err_thread.join(timeout=8)

    def _kill(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            proc.kill()

    def _close_pipes(self) -> None:
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass

    def _wait_proc(self) -> int | None:
        try:
            return int(self._proc.wait(timeout=8))
        except Exception:
            self._kill()
            try:
                return int(self._proc.wait(timeout=3))
            except Exception:
                return self._proc.poll()

    def _error(self, message: str) -> DecodeError:
        return DecodeError(
            message,
            returncode=self._proc.poll(),
            stderr_tail=self._stderr_tail,
        )


def iter_decoded_frames(
    info: VideoInfo,
    max_width: int,
    timestamps: list[float] | None = None,
    start_s: float | None = None,
) -> Iterator[tuple[float, bytes]]:
    """One decode pass. If timestamps are given, only those frames are encoded.

    start_s, when set, is an input seek. -copyts keeps presentation timestamps
    on the original recording timeline. Filter graphs are written to a temp file
    so long recordings stay under the Windows command-line limit.
    """
    script = write_filter_script(_filter_graph(info, max_width, timestamps))
    decoder: FrameDecoder | None = None
    try:
        args = build_decode_command(
            info,
            max_width,
            timestamps=timestamps,
            start_s=start_s,
            filter_script=script,
        )
        decoder = FrameDecoder(args)
        try:
            for pts, jpeg in decoder.frames():
                yield pts, jpeg
            decoder.finish()
        except GeneratorExit:
            if decoder is not None:
                decoder.request_stop()
            raise
    finally:
        if decoder is not None:
            decoder.close()
        try:
            script.unlink(missing_ok=True)
        except OSError:
            pass


# Pillow 11 Image.new / JPEG encode refuse a side longer than 65500 and raise
# OSError: broken data stream when writing image file. Long VSLs with ~1600
# widescreen thumbs exceed that in a single sheet.
MAX_CONTACT_SHEET_DIMENSION = 65000


def _ensure_max_width(path: Path, max_width: int) -> None:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.load()
            if image.width <= max_width:
                if path.suffix.lower() != ".jpg":
                    image.save(path, format="JPEG", quality=90)
                return
            height = max(1, round(image.height * max_width / image.width))
            resized = image.resize((max_width, height), Image.Resampling.LANCZOS)
            resized.save(path, format="JPEG", quality=90)
    except OSError:
        # Leave the original bytes. A truncated JPEG must not abort capture.
        return


def ahash(path: Path, size: int = 8) -> str | None:
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            pixels = list(gray.getdata())
    except OSError:
        return None
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
        if digest is None:
            record.compact_retained = True
            record.notes.append("compact hash skipped: unreadable screenshot")
            continue
        if last_hash is not None and hamming(digest, last_hash) <= threshold:
            record.compact_retained = False
            record.compact_refers_to = last_id
        else:
            record.compact_retained = True
            record.compact_refers_to = None
            last_id = record.id
            last_hash = digest


def _thumbnail_for_contact(path: Path, thumb_w: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        ratio = thumb_w / max(rgb.width, 1)
        return rgb.resize((thumb_w, max(1, round(rgb.height * ratio))), Image.Resampling.LANCZOS)


def _contact_page_capacity(
    columns: int,
    thumb_w: int,
    cell_h: int,
    pad: int,
    max_dimension: int,
) -> tuple[int, int]:
    """Return (columns, items_per_page) that fit Pillow/JPEG dimension limits."""
    columns = max(1, columns)
    thumb_w = max(1, thumb_w)
    cell_h = max(1, cell_h)
    pad = max(0, pad)
    max_dimension = max(1, max_dimension)
    while columns > 1 and columns * (thumb_w + pad) + pad > max_dimension:
        columns -= 1
    usable_h = max(1, max_dimension - pad)
    row_h = cell_h + pad
    rows = max(1, usable_h // row_h)
    return columns, max(1, rows * columns)


def _contact_sheet_page_path(dest: Path, page: int) -> Path:
    if page <= 1:
        return dest
    return dest.with_name(f"{dest.stem}-{page:02d}{dest.suffix}")


def _remove_stale_contact_pages(dest: Path) -> None:
    parent = dest.parent
    if not parent.is_dir():
        return
    prefix = dest.stem + "-"
    suffix = dest.suffix.lower()
    for path in parent.glob(f"{dest.stem}-*{dest.suffix}"):
        rest = path.stem[len(prefix) :] if path.stem.startswith(prefix) else ""
        if rest.isdigit() and path.suffix.lower() == suffix:
            try:
                path.unlink()
            except OSError:
                pass


def _render_contact_page(
    items: list[tuple[ScreenshotRecord, Image.Image]],
    dest: Path,
    columns: int,
    thumb_w: int,
    caption_h: int,
    pad: int,
) -> None:
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


def write_contact_sheet(
    records: list[ScreenshotRecord],
    job_root: Path,
    dest: Path,
    columns: int = 4,
    thumb_w: int = 320,
    max_dimension: int = MAX_CONTACT_SHEET_DIMENSION,
) -> tuple[list[Path], list[str]]:
    """Write one or more JPEG pages. Unreadable frames are skipped, not fatal."""
    items: list[tuple[ScreenshotRecord, Image.Image]] = []
    skipped: list[str] = []
    for record in records:
        path = job_root / record.relative_path
        if not path.exists():
            continue
        try:
            thumb = _thumbnail_for_contact(path, thumb_w)
        except Exception as exc:  # noqa: BLE001
            record.notes.append(f"contact sheet skipped unreadable screenshot: {exc}")
            skipped.append(record.id)
            continue
        items.append((record, thumb))
    dest.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_contact_pages(dest)
    if not items:
        empty = Image.new("RGB", (thumb_w, 80), (245, 245, 245))
        empty.save(dest, format="JPEG", quality=90)
        return [dest], skipped

    caption_h = 36
    pad = 8
    cell_h = max(im.height for _, im in items) + caption_h
    page_columns, per_page = _contact_page_capacity(columns, thumb_w, cell_h, pad, max_dimension)
    written: list[Path] = []
    page = 1
    for offset in range(0, len(items), per_page):
        chunk = items[offset : offset + per_page]
        path = _contact_sheet_page_path(dest, page)
        _render_contact_page(chunk, path, page_columns, thumb_w, caption_h, pad)
        written.append(path)
        page += 1
    return written, skipped


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
    try:
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
            # Do not stop consuming here. Filling every screenshot is not
            # proof that ffmpeg completed; iter_decoded_frames() validates
            # the exit status only after this generator is exhausted.
    except DecodeError:
        raise
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
