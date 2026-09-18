from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_color_video, write_cut_video, write_vfr_video
from vsl_study.media import inspect_video


@requires_ffmpeg
def test_inspect_basic_and_no_audio(tmp_path: Path):
    video = write_color_video(tmp_path / "solid.mp4", duration=2.0, audio=False)
    info = inspect_video(video)
    assert info.has_video
    assert info.has_audio is False
    assert info.duration_s == pytest.approx(2.0, abs=0.15)
    assert info.width == 640
    assert info.height == 360
    assert info.fingerprint


@requires_ffmpeg
def test_inspect_path_with_spaces(tmp_path: Path):
    video = write_color_video(tmp_path / "my sample video.mp4", duration=1.0)
    info = inspect_video(video)
    assert info.duration_s > 0
    assert " " in info.resolved_path


@requires_ffmpeg
def test_inspect_cut_video_duration(tmp_path: Path):
    video = write_cut_video(tmp_path / "cuts.mp4")
    info = inspect_video(video)
    assert info.duration_s == pytest.approx(6.0, abs=0.2)


@requires_ffmpeg
def test_vfr_or_notes(tmp_path: Path):
    video = write_vfr_video(tmp_path / "vfr.mp4")
    info = inspect_video(video)
    assert info.duration_s > 0
    assert info.has_video


@requires_ffmpeg
def test_audio_stream_can_outlast_video(tmp_path: Path):
    from conftest import write_audio_longer_than_video

    video = write_audio_longer_than_video(tmp_path / "tail.mkv")
    info = inspect_video(video)
    assert info.has_audio and info.has_video
    if info.video_duration_s:
        assert info.video_duration_s <= info.duration_s + 0.05
