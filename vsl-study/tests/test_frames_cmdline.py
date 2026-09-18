from __future__ import annotations

from pathlib import Path

from conftest import ROOT, requires_ffmpeg, write_color_video
from vsl_study.frames import (
    WIN32_CMDLINE_LIMIT,
    CaptureCandidate,
    RECORDER_MAX_DURATION_S,
    build_candidates,
    build_decode_command,
    capture_candidates,
    serialized_command_length,
    write_filter_script,
)
from vsl_study.media import inspect_video
from vsl_study.models import Scene, VideoInfo


def _info(path: Path, duration_s: float) -> VideoInfo:
    return VideoInfo(
        path=str(path),
        resolved_path=str(path),
        size_bytes=1,
        mtime_ns=1,
        fingerprint="abc",
        duration_s=duration_s,
        width=3840,
        height=1730,
        fps_avg=1000.0,
        fps_r="1000/1",
        time_base="1/1000",
        video_start_s=0.0,
        audio_start_s=0.0,
        format_start_s=0.0,
        rotation=0,
        has_audio=True,
        has_video=True,
        vfr=True,
        format_name="webm",
        audio_codec="opus",
        video_codec="vp9",
        video_duration_s=duration_s,
        fps_trusted=False,
    )


def _long_candidates(duration_s: float) -> tuple[VideoInfo, list[CaptureCandidate]]:
    info = _info(Path(r"C:\Users\Tomas\Videos\my recording.webm"), duration_s)
    scenes = []
    t = 0.0
    index = 1
    while t < duration_s:
        end = min(duration_s, t + 12.0)
        scenes.append(Scene(f"scene_{index:04d}", t, end))
        t = end
        index += 1
    candidates = build_candidates(info, scenes, interval=5.0, scene_start_offset=0.25)
    return info, candidates


def test_90_minute_filter_uses_script_and_stays_under_windows_limit():
    info, candidates = _long_candidates(90 * 60)
    times = [c.requested_time for c in candidates]
    assert len(times) > 1000
    script = Path(r"C:\Users\Tomas\AppData\Local\Temp\vsl-study-vf-demo.txt")
    args = build_decode_command(info, 1280, times, filter_script=script)
    length = serialized_command_length(args)
    assert "-filter_script:v" in args
    assert "-vf" not in args
    assert length < 12_000
    assert length < WIN32_CMDLINE_LIMIT
    embedded = build_decode_command(info, 1280, times, filter_script=None)
    assert serialized_command_length(embedded) > WIN32_CMDLINE_LIMIT


def test_recorder_max_duration_command_stays_under_windows_limit():
    html = (ROOT / "src" / "vsl_study" / "recorder" / "index.html").read_text(encoding="utf-8")
    assert 'id="limit"' in html
    assert 'max="180"' in html
    assert RECORDER_MAX_DURATION_S == 180 * 60
    info, candidates = _long_candidates(RECORDER_MAX_DURATION_S)
    times = [c.requested_time for c in candidates]
    args = build_decode_command(
        info,
        1280,
        times,
        filter_script=Path(r"C:\Users\Tomas\AppData\Local\Temp\vsl-study-vf-max.txt"),
    )
    assert serialized_command_length(args) < 12_000
    assert serialized_command_length(args) < WIN32_CMDLINE_LIMIT


def test_filter_scripts_are_isolated_and_cleaned_up(tmp_path: Path):
    first = write_filter_script("showinfo", directory=tmp_path)
    second = write_filter_script("showinfo", directory=tmp_path)
    assert first != second
    assert first.exists() and second.exists()
    first.unlink()
    second.unlink()
    assert list(tmp_path.glob("vsl-study-vf-*")) == []


@requires_ffmpeg
def test_filter_script_works_with_spaces_and_unicode(tmp_path: Path, monkeypatch):
    folder = tmp_path / "estudio grabación"
    folder.mkdir()
    video = write_color_video(folder / "clip 01.mp4", duration=1.0, fps=12)

    original = __import__("vsl_study.frames", fromlist=["write_filter_script"]).write_filter_script

    def in_tmp(graph, directory=None):
        return original(graph, directory=folder)

    monkeypatch.setattr("vsl_study.frames.write_filter_script", in_tmp)
    info = inspect_video(video)
    records = capture_candidates(
        info,
        [],
        folder / "frames",
        320,
        [CaptureCandidate(0.0, "interval", None), CaptureCandidate(0.4, "interval", None)],
    )
    assert len(records) == 2
    leftover = list(folder.glob("vsl-study-vf-*"))
    assert leftover == []
    for rec in records:
        assert (folder / rec.relative_path).exists()
