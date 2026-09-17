import json
from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_color_video
from vsl_study.cache import JobConflictError, JobDir, fingerprint_file
from vsl_study.models import ProcessSettings
from vsl_study.pipeline import process_video


def test_atomic_json_and_incomplete_not_reused(tmp_path: Path):
    job = JobDir(tmp_path / "job")
    job.ensure()
    job.write_stage("frames", "key-a", "running", {"screenshots": []})
    assert job.read_complete_stage("frames", "key-a") is None
    job.write_stage("frames", "key-a", "complete", {"screenshots": [{"id": "frame_0001"}]})
    cached = job.read_complete_stage("frames", "key-a")
    assert cached is not None
    assert job.read_complete_stage("frames", "other-key") is None


def test_fingerprint_changes_with_file(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    first = fingerprint_file(p)
    p.write_bytes(b"hello world!")
    second = fingerprint_file(p)
    assert first != second


@requires_ffmpeg
def test_job_refuses_different_source(tmp_path: Path):
    a = write_color_video(tmp_path / "a.mp4", duration=1.0, color="red")
    b = write_color_video(tmp_path / "b.mp4", duration=1.2, color="blue")
    out = tmp_path / "out"
    settings = ProcessSettings(model="tiny.en", interval=2.0, compact_view=False, ocr=False)
    process_video(a, out, settings=settings)
    with pytest.raises(JobConflictError):
        process_video(b, out, settings=settings)


@requires_ffmpeg
def test_interval_change_does_not_require_new_inspect(tmp_path: Path):
    video = write_color_video(tmp_path / "solid.mp4", duration=4.0)
    out = tmp_path / "out"
    process_video(video, out, settings=ProcessSettings(interval=2.0, compact_view=False, ocr=False))
    inspect_path = out / "cache" / "inspect.json"
    inspect_before = json.loads(inspect_path.read_text(encoding="utf-8"))
    process_video(video, out, settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False))
    inspect_after = json.loads(inspect_path.read_text(encoding="utf-8"))
    assert inspect_before["cache_key"] == inspect_after["cache_key"]
    frames = json.loads((out / "cache" / "frames.json").read_text(encoding="utf-8"))
    assert "1.000" in frames["cache_key"]
