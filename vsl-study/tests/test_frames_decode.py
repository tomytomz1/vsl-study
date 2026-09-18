from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest
from PIL import Image

from conftest import requires_ffmpeg, write_audio_longer_than_video, write_color_video
from vsl_study.frames import (
    CaptureCandidate,
    DecodeError,
    capture_candidates,
    iter_decoded_frames,
)
from vsl_study.media import inspect_video
from vsl_study.models import ProcessSettings, VideoInfo
from vsl_study.pipeline import PipelineError, process_video


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
        duration_s=10.0,
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
        video_duration_s=9.5,
        fps_trusted=True,
    )
    data.update(overrides)
    return VideoInfo(**data)


class FakePopen:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self._rc = returncode
        self._done = False
        self._killed = False

    def poll(self):
        return self._rc if self._done else None

    def wait(self, timeout=None):
        self._done = True
        return self._rc

    def kill(self):
        self._killed = True
        self._done = True
        self._rc = 1


class BlockingStream:
    def __init__(self):
        self._evt = threading.Event()

    def read(self, n=-1):
        self._evt.wait(timeout=30)
        return b""

    def close(self):
        self._evt.set()


class BlockingPopen(FakePopen):
    def __init__(self):
        self.stdout = BlockingStream()
        self.stderr = BlockingStream()
        self._rc = 1
        self._done = False
        self._killed = False


def _install_script(monkeypatch, steps: list[dict]):
    calls: list[list[str]] = []

    def popen(args, **kwargs):
        calls.append(list(args))
        step = steps.pop(0)
        return FakePopen(step["stdout"], step["stderr"], step["returncode"])

    monkeypatch.setattr("vsl_study.frames.ffcmd.popen", popen)
    return calls


def test_nonzero_exit_after_valid_frame_is_decoder_failure(tmp_path: Path, monkeypatch):
    jpeg = _jpeg_bytes()
    stderr = b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\nError while decoding stream #0:0: Invalid data found\n"
    _install_script(
        monkeypatch,
        [{"stdout": jpeg, "stderr": stderr, "returncode": 23}],
    )
    dest = tmp_path / "frames"
    with pytest.raises(DecodeError, match="ffmpeg exit 23") as caught:
        capture_candidates(
            _info(tmp_path / "broken.mp4"),
            [],
            dest,
            320,
            [CaptureCandidate(9.0, "interval", "scene_0001")],
        )
    assert "Invalid data" in str(caught.value)
    notes = []
    for path in dest.glob("*.jpg"):
        notes.append(path.name)
    assert all("last available" not in p for p in notes)
    leftover = list(dest.glob("*.jpg"))
    for jpeg_path in leftover:
        # A frame at t=0 must not be reported as the 9s capture.
        assert "t00-00-09" not in jpeg_path.name


def test_tail_scan_failure_is_not_last_frame_success(tmp_path: Path, monkeypatch):
    jpeg = _jpeg_bytes()
    ok_err = b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\n"
    bad_err = b"Conversion failed!\n"
    _install_script(
        monkeypatch,
        [
            {"stdout": jpeg, "stderr": ok_err, "returncode": 0},
            {"stdout": b"", "stderr": bad_err, "returncode": 23},
        ],
    )
    with pytest.raises(DecodeError, match="ffmpeg decoder failed"):
        capture_candidates(
            _info(tmp_path / "n.mp4"),
            [],
            tmp_path / "frames",
            320,
            [CaptureCandidate(9.0, "interval", None)],
        )


def test_intentional_early_stop_is_not_a_decoder_failure(tmp_path: Path, monkeypatch):
    jpeg = _jpeg_bytes((1, 2, 3))
    extra = _jpeg_bytes((9, 9, 9))
    stderr = (
        b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\n"
        b"[Parsed_showinfo_0 @ 0] n:1 pts:1 pts_time:0.400 pos:0\n"
        b"[Parsed_showinfo_0 @ 0] n:2 pts:2 pts_time:0.800 pos:0\n"
    )
    _install_script(
        monkeypatch,
        [{"stdout": jpeg + extra + extra, "stderr": stderr, "returncode": 0}],
    )
    records = capture_candidates(
        _info(tmp_path / "n.mp4", duration_s=1.0, video_duration_s=1.0),
        [],
        tmp_path / "frames",
        320,
        [CaptureCandidate(0.0, "interval", None)],
    )
    assert len(records) == 1
    assert records[0].actual_time == pytest.approx(0.0)
    assert records[0].notes == [] or all("last available" not in n for n in records[0].notes)


def test_missing_pts_is_decoder_failure(tmp_path: Path, monkeypatch):
    _install_script(
        monkeypatch,
        [{"stdout": _jpeg_bytes(), "stderr": b"frame=1 fps=1\n", "returncode": 0}],
    )
    with pytest.raises(DecodeError, match="presentation timestamp"):
        list(
            iter_decoded_frames(
                _info(tmp_path / "n.mp4"),
                320,
                timestamps=[0.0],
            )
        )


def test_truncated_jpeg_is_decoder_failure(tmp_path: Path, monkeypatch):
    _install_script(
        monkeypatch,
        [{"stdout": b"\xff\xd8\x00\x01\x02incomplete", "stderr": b"", "returncode": 0}],
    )
    with pytest.raises(DecodeError, match="truncated JPEG"):
        list(iter_decoded_frames(_info(tmp_path / "n.mp4"), 320, timestamps=[0.0]))


def test_stalled_decoder_times_out(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("vsl_study.frames.IO_TIMEOUT_S", 0.2)
    monkeypatch.setattr("vsl_study.frames.ffcmd.popen", lambda args, **kwargs: BlockingPopen())
    with pytest.raises(DecodeError, match="timed out"):
        list(iter_decoded_frames(_info(tmp_path / "n.mp4"), 320, timestamps=[0.0]))


def test_timeout_is_decoder_failure_not_cancellation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("vsl_study.frames.IO_TIMEOUT_S", 0.2)
    monkeypatch.setattr("vsl_study.frames.ffcmd.popen", lambda args, **kwargs: BlockingPopen())
    dest = tmp_path / "frames"
    with pytest.raises(DecodeError, match="timed out"):
        capture_candidates(
            _info(tmp_path / "n.mp4"),
            [],
            dest,
            320,
            [CaptureCandidate(0.0, "interval", None)],
        )
    assert list(dest.glob("*.jpg")) == []


def test_all_requested_frames_then_nonzero_exit_is_decoder_failure(tmp_path: Path, monkeypatch):
    jpeg = _jpeg_bytes()
    stderr = (
        b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\n"
        b"Error while decoding stream #0:0: Invalid data found\n"
    )
    _install_script(
        monkeypatch,
        [{"stdout": jpeg, "stderr": stderr, "returncode": 23}],
    )
    dest = tmp_path / "frames"
    with pytest.raises(DecodeError, match="ffmpeg exit 23") as caught:
        capture_candidates(
            _info(tmp_path / "clip.mp4", duration_s=1.0, video_duration_s=1.0),
            [],
            dest,
            320,
            [CaptureCandidate(0.0, "interval", None)],
        )
    assert "Invalid data" in str(caught.value)
    saved = list(dest.glob("*.jpg"))
    assert saved
    assert any("t00-00-00" in path.name for path in saved)
    assert all("last available" not in path.name for path in saved)


def test_explicit_generator_close_is_cancellation(tmp_path: Path, monkeypatch):
    jpeg = _jpeg_bytes()
    extra = _jpeg_bytes((9, 9, 9))
    stderr = (
        b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\n"
        b"[Parsed_showinfo_0 @ 0] n:1 pts:1 pts_time:0.400 pos:0\n"
    )
    _install_script(
        monkeypatch,
        [{"stdout": jpeg + extra, "stderr": stderr, "returncode": 1}],
    )
    gen = iter_decoded_frames(
        _info(tmp_path / "n.mp4", duration_s=1.0, video_duration_s=1.0),
        320,
        timestamps=[0.0, 0.4],
    )
    first = next(gen)
    assert first[0] == pytest.approx(0.0)
    gen.close()


@requires_ffmpeg
def test_successful_eof_with_audio_outlasting_video(tmp_path: Path):
    video = write_audio_longer_than_video(tmp_path / "tail.mkv")
    info = inspect_video(video)
    records = capture_candidates(
        info,
        [],
        tmp_path / "frames",
        320,
        [
            CaptureCandidate(0.1, "interval", None),
            CaptureCandidate(info.duration_s - 0.05, "interval", None),
        ],
    )
    assert records[0].actual_time >= 0
    assert records[-1].actual_time < info.duration_s - 0.2
    assert any("last available" in n for n in records[-1].notes)


@requires_ffmpeg
def test_successful_completion_after_all_requested_screenshots(tmp_path: Path):
    video = write_color_video(tmp_path / "solid.mp4", duration=1.0, fps=15)
    info = inspect_video(video)
    records = capture_candidates(
        info,
        [],
        tmp_path / "frames",
        320,
        [
            CaptureCandidate(0.0, "interval", None),
            CaptureCandidate(0.4, "interval", None),
        ],
    )
    assert len(records) == 2
    assert all((tmp_path / rec.relative_path).exists() for rec in records)


@requires_ffmpeg
def test_pipeline_marks_frames_failed_when_decoder_exits_nonzero(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "clip.mp4", duration=1.0, audio=False)
    jpeg = _jpeg_bytes()
    stderr = b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\nError while decoding stream\n"
    _install_script(
        monkeypatch,
        [{"stdout": jpeg, "stderr": stderr, "returncode": 23}],
    )
    out = tmp_path / "job"
    with pytest.raises(PipelineError, match="Screenshot extraction failed"):
        process_video(
            video,
            out,
            settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=False),
        )
    job = __import__("json").loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["frames"]["status"] == "failed"
    assert job["stages"]["frames"]["status"] != "running"
    cache = __import__("json").loads((out / "cache" / "frames.json").read_text(encoding="utf-8"))
    assert cache["status"] == "failed"
    assert "ffmpeg exit 23" in cache["error"] or "decoder failed" in cache["error"]
    assert cache["completed"] == len(list((out / "frames").glob("frame_*.jpg")))


@requires_ffmpeg
def test_pipeline_marks_frames_failed_after_all_jpegs_then_exit_23(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "clip.mp4", duration=1.0, audio=False)
    jpeg = _jpeg_bytes()
    stderr = (
        b"[Parsed_showinfo_0 @ 0] n:0 pts:0 pts_time:0.000 pos:0\n"
        b"[Parsed_showinfo_0 @ 0] n:1 pts:1 pts_time:0.250 pos:0\n"
        b"[Parsed_showinfo_0 @ 0] n:2 pts:2 pts_time:1.000 pos:0\n"
        b"Error while decoding stream #0:0: Invalid data found\n"
    )
    _install_script(
        monkeypatch,
        [{"stdout": jpeg * 3, "stderr": stderr, "returncode": 23}],
    )
    out = tmp_path / "job"
    with pytest.raises(PipelineError, match="Screenshot extraction failed"):
        process_video(
            video,
            out,
            settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=False),
        )
    job = __import__("json").loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["frames"]["status"] == "failed"
    cache = __import__("json").loads((out / "cache" / "frames.json").read_text(encoding="utf-8"))
    assert cache["status"] == "failed"
    assert "ffmpeg exit 23" in cache["error"] or "decoder failed" in cache["error"]
    saved = list((out / "frames").glob("frame_*.jpg"))
    assert saved
    assert cache["completed"] == len(saved)
    assert cache["completed"] >= 1
