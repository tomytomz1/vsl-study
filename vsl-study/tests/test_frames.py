from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from conftest import requires_ffmpeg, write_audio_longer_than_video, write_color_video, write_cut_video, write_vfr_video
from vsl_study.frames import (
    CaptureCandidate,
    apply_sequential_compact,
    build_candidates,
    capture_candidates,
    image_end_time,
    last_safe_time,
    write_contact_sheet,
)
from vsl_study.media import inspect_video
from vsl_study.models import Scene, ScreenshotRecord, VideoInfo
from vsl_study.scenes import detect_scenes


def _jpeg_bytes(color: tuple[int, int, int] = (20, 80, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _info(path: Path, **overrides) -> VideoInfo:
    data = dict(
        path=str(path),
        resolved_path=str(path),
        size_bytes=1,
        mtime_ns=1,
        fingerprint="abc",
        duration_s=2.5,
        width=320,
        height=240,
        fps_avg=25.0,
        fps_r="25/1",
        time_base="1/25",
        video_start_s=0.0,
        audio_start_s=0.0,
        format_start_s=0.0,
        rotation=0,
        has_audio=True,
        has_video=True,
        vfr=False,
        format_name="mp4",
        audio_codec="aac",
        video_codec="h264",
        video_duration_s=1.0,
        fps_trusted=True,
    )
    data.update(overrides)
    return VideoInfo(**data)


def test_image_end_ignores_untrusted_1000_fps():
    info = _info(
        Path("x.webm"),
        duration_s=120.062,
        fps_avg=1000.0,
        fps_r="1000/1",
        time_base="1/1000",
        video_duration_s=119.377,
        fps_trusted=False,
    )
    assert last_safe_time(info) == pytest.approx(119.377)
    assert image_end_time(info) < 119.982


def test_interval_candidates_stop_at_sampling_end():
    info = _info(
        Path("x.webm"),
        duration_s=12134.0,
        video_duration_s=12132.0,
        fps_avg=1000.0,
        fps_trusted=False,
    )
    scenes = [Scene("scene_0001", 0.0, info.duration_s)]
    items = build_candidates(
        info,
        scenes,
        interval=15.0,
        scene_start_offset=0.25,
        sampling_end=4278.0,
        extra_times=[12000.0],
    )
    autos = [item for item in items if item.reason != "user"]
    assert max(item.requested_time for item in autos) == pytest.approx(4278.0)
    assert all(item.requested_time <= 4278.0 + 1e-6 for item in autos)
    assert any(item.reason == "user" and item.requested_time == pytest.approx(12000.0) for item in items)
    assert not any(item.reason != "user" and item.requested_time > 4300 for item in items)


def test_efficient_extraction_decodes_once(tmp_path: Path):
    calls = []

    def source(info, max_width):
        calls.append(max_width)
        yield 0.00, _jpeg_bytes((255, 0, 0))
        yield 0.40, _jpeg_bytes((0, 255, 0))
        yield 0.90, _jpeg_bytes((0, 0, 255))

    candidates = [
        CaptureCandidate(0.0, "interval", "scene_0001"),
        CaptureCandidate(0.35, "scene_start", "scene_0001"),
        CaptureCandidate(0.80, "interval", "scene_0001"),
    ]
    records = capture_candidates(
        _info(tmp_path / "n.mp4"),
        [],
        tmp_path / "frames",
        320,
        candidates,
        frame_source=source,
    )
    assert len(calls) == 1
    assert [round(r.actual_time, 2) for r in records] == [0.0, 0.4, 0.9]
    assert all((tmp_path / r.relative_path).exists() for r in records)


def test_request_after_last_frame_uses_last_pts(tmp_path: Path):
    def source(info, max_width):
        yield 0.10, _jpeg_bytes()
        yield 0.95, _jpeg_bytes((9, 9, 9))

    records = capture_candidates(
        _info(tmp_path / "n.mp4", video_duration_s=1.0, duration_s=2.5),
        [],
        tmp_path / "frames",
        320,
        [CaptureCandidate(2.4, "interval", "scene_0001")],
        frame_source=source,
    )
    assert records[0].actual_time == pytest.approx(0.95)
    assert any("last available video frame" in note for note in records[0].notes)


def test_no_decoded_frame_is_a_hard_failure(tmp_path: Path):
    def source(info, max_width):
        if False:
            yield 0.0, b""
        return
        yield

    with pytest.raises(RuntimeError, match="Could not decode any video frames"):
        capture_candidates(
            _info(tmp_path / "n.mp4"),
            [],
            tmp_path / "frames",
            320,
            [CaptureCandidate(0.0, "interval", None)],
            frame_source=source,
        )


@requires_ffmpeg
def test_vfr_actual_timestamps_come_from_pts(tmp_path: Path):
    video = write_vfr_video(tmp_path / "vfr.mp4")
    info = inspect_video(video)
    candidates = build_candidates(info, [Scene("scene_0001", 0.0, info.duration_s)], interval=0.4, scene_start_offset=0.1)
    records = capture_candidates(info, [], tmp_path / "frames", 320, candidates)
    assert records
    for rec in records:
        assert rec.actual_time >= 0
        assert (tmp_path / rec.relative_path).stat().st_size > 32
        if rec.requested_time > 0:
            assert rec.actual_time != pytest.approx(rec.requested_time * 1000.0)


@requires_ffmpeg
def test_audio_longer_than_video_last_frame(tmp_path: Path):
    video = write_audio_longer_than_video(tmp_path / "tail.mkv")
    info = inspect_video(video)
    assert info.duration_s == pytest.approx(2.5, abs=0.3)
    if info.video_duration_s:
        assert info.video_duration_s < info.duration_s - 0.3
    end = image_end_time(info)
    assert end < info.duration_s - 0.3 or end <= 1.2
    candidates = [
        CaptureCandidate(0.2, "interval", "scene_0001"),
        CaptureCandidate(max(info.duration_s - 0.05, end + 0.5), "interval", "scene_0001"),
    ]
    records = capture_candidates(info, [], tmp_path / "frames", 320, candidates)
    assert records[-1].actual_time <= end + 0.15
    assert records[-1].actual_time < info.duration_s - 0.2
    # Must be the real last decoded frame, not an earlier selected interval.
    assert records[-1].actual_time >= end - 0.25
    assert any("last available" in n or abs(records[-1].actual_time - records[-1].requested_time) > 0.2 for n in (records[-1].notes or [""]))


@requires_ffmpeg
def test_scene_seconds_are_ordered_and_frame_indexes_are_honest(tmp_path: Path):
    video = write_cut_video(tmp_path / "cuts.mp4")
    info = inspect_video(video)
    scenes = detect_scenes(info, "content")
    assert scenes
    assert scenes[0].start == pytest.approx(0.0, abs=0.05)
    assert scenes[-1].end == pytest.approx(info.duration_s, abs=0.2)
    prev = -1.0
    for scene in scenes:
        assert scene.end >= scene.start
        assert scene.start >= prev - 1e-6
        prev = scene.end
        if scene.start_frame is not None and scene.end_frame is not None:
            assert scene.end_frame >= scene.start_frame
    untrusted = VideoInfo(**{**info.to_dict(), "fps_trusted": False, "fps_avg": 1000.0, "time_base": "1/1000"})
    from vsl_study.scenes import _scene_frame_indices

    class _TC:
        def __init__(self, seconds, frame_num):
            self.seconds = seconds
            self.frame_num = frame_num

    start_f, end_f = _scene_frame_indices(untrusted, _TC(117.874, 117874), _TC(120.062, 1807))
    assert start_f is None and end_f is None


@requires_ffmpeg
def test_one_ffmpeg_decode_for_many_screenshots(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "solid.mp4", duration=1.2, fps=15)
    info = inspect_video(video)
    popen_calls = []
    real_popen = __import__("vsl_study.ffcmd", fromlist=["popen"]).popen

    def wrapped(args, **kwargs):
        popen_calls.append(list(args))
        return real_popen(args, **kwargs)

    monkeypatch.setattr("vsl_study.ffcmd.popen", wrapped)
    monkeypatch.setattr("vsl_study.frames.ffcmd.popen", wrapped)
    candidates = build_candidates(info, [Scene("scene_0001", 0.0, info.duration_s)], interval=0.3, scene_start_offset=0.1)
    records = capture_candidates(info, [], tmp_path / "frames", 320, candidates)
    assert records
    decode_from_start = [
        call
        for call in popen_calls
        if "-i" in call and "pipe:1" in call
    ]
    assert 1 <= len(decode_from_start) <= 2
    # Must not spawn a from-zero select capture per screenshot.
    assert len(popen_calls) <= 2
    assert len(popen_calls) < len(candidates)


def _write_jpeg(path: Path, size: tuple[int, int] = (80, 45), color: tuple[int, int, int] = (20, 80, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=85)
    return path


def _sheet_shot(frame_id: str, rel: str, t: float) -> ScreenshotRecord:
    return ScreenshotRecord(
        id=frame_id,
        requested_time=t,
        actual_time=t,
        scene_id="scene_0001",
        relative_path=rel,
        capture_reason="interval",
    )


def test_contact_sheet_skips_truncated_jpeg_and_continues(tmp_path: Path):
    frames = tmp_path / "frames"
    _write_jpeg(frames / "frame_0001.jpg", color=(200, 0, 0))
    buf = io.BytesIO()
    Image.new("RGB", (80, 45), (0, 200, 0)).save(buf, format="JPEG", quality=85)
    (frames / "frame_0002.jpg").write_bytes(buf.getvalue()[:28])
    _write_jpeg(frames / "frame_0003.jpg", color=(0, 0, 200))
    records = [
        _sheet_shot("frame_0001", "frames/frame_0001.jpg", 1.0),
        _sheet_shot("frame_0002", "frames/frame_0002.jpg", 2.0),
        _sheet_shot("frame_0003", "frames/frame_0003.jpg", 3.0),
    ]
    dest = tmp_path / "contact_sheets" / "all.jpg"
    paths, skipped = write_contact_sheet(records, tmp_path, dest, columns=2, thumb_w=40)
    assert dest.is_file()
    assert dest.stat().st_size > 32
    assert [p.name for p in paths] == ["all.jpg"]
    assert skipped == ["frame_0002"]
    assert any("unreadable screenshot" in n for n in records[1].notes)
    with Image.open(dest) as sheet:
        sheet.verify()
    apply_sequential_compact(records, frames)
    assert records[1].compact_retained is True


def test_contact_sheet_paginates_before_jpeg_dimension_limit(tmp_path: Path):
    frames = tmp_path / "frames"
    records = []
    for i in range(1, 25):
        rel = f"frames/frame_{i:04d}.jpg"
        _write_jpeg(tmp_path / rel, size=(64, 48))
        records.append(_sheet_shot(f"frame_{i:04d}", rel, float(i)))
    dest = tmp_path / "contact_sheets" / "all.jpg"
    paths, skipped = write_contact_sheet(
        records,
        tmp_path,
        dest,
        columns=4,
        thumb_w=40,
        max_dimension=220,
    )
    assert not skipped
    assert len(paths) >= 2
    assert paths[0] == dest
    assert paths[1].name == "all-02.jpg"
    for path in paths:
        with Image.open(path) as sheet:
            assert sheet.size[0] <= 220
            assert sheet.size[1] <= 220
            sheet.verify()
