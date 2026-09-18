"""Regression cases reproduced during the independent capture/export audit."""

import json
import zipfile
from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_color_video
from vsl_study import pipeline
from vsl_study.cache import JobDir, identity_from_info
from vsl_study.capture_meta import (
    INVALID_NOTE,
    build_capture_record,
    clear_portable_capture,
    public_capture_record,
    resolve_capture_for_source,
    write_portable_capture,
)
from vsl_study.export import write_zip
from vsl_study.media import inspect_video
from vsl_study.models import ProcessSettings, TranscriptResult


@pytest.mark.parametrize("field,value", [
    ("source_url_user_supplied", "https://example.com:invalid/vsl"),
    ("source_url_user_supplied", "https://example.com:99999/vsl"),
    ("source_url_user_supplied", "https://[broken/vsl"),
    ("media_duration_s", 10**400),
])
def test_malformed_values_are_unavailable(tmp_path, field, value):
    video = tmp_path / "recording.webm"
    record = build_capture_record(recording_id="audit01", recording_path=str(video))
    record[field] = value
    assert public_capture_record(record) is None
    (tmp_path / "capture.json").write_text(json.dumps(record), encoding="utf-8")
    assert resolve_capture_for_source(video) == (None, [INVALID_NOTE])
    assert resolve_capture_for_source(video, provided=record) == (None, [INVALID_NOTE])


@requires_ffmpeg
@pytest.mark.parametrize("previous_zip", [False, True])
def test_locked_rejected_metadata_blocks_export(tmp_path, monkeypatch, previous_zip):
    video = write_color_video(tmp_path / "recording.mp4", duration=0.4)
    info = inspect_video(video)
    record = build_capture_record(
        recording_id="audit01", recording_path=str(video), complete=True,
    )
    job = JobDir(tmp_path / "job")
    job.ensure()
    job.bind_source(identity_from_info(info), ProcessSettings(capture=record))
    write_portable_capture(job.root, record)
    job.write_stage("export", "previous", "complete", {})
    (job.root / "report.html").write_text("previous report", encoding="utf-8")
    zip_path = job.root / "vsl_study_evidence.zip"
    old_bytes = write_zip(job, False).read_bytes() if previous_zip else None

    record["complete"] = "false"
    (video.parent / "capture.json").write_text(json.dumps(record), encoding="utf-8")
    stale = job.root / "capture-session" / "capture.json"
    real_unlink = Path.unlink

    def locked_unlink(path, *args, **kwargs):
        if path == stale:
            raise PermissionError("capture file locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)

    def must_not_run(*args, **kwargs):
        pytest.fail("Processing/export continued after provenance cleanup failed")

    monkeypatch.setattr(pipeline, "_stage_transcribe", must_not_run)
    monkeypatch.setattr(pipeline, "write_zip", must_not_run)
    with pytest.raises(pipeline.PipelineError, match="no new evidence ZIP was published"):
        pipeline.process_video(video, job.root, settings=ProcessSettings(ocr=False))
    meta = job.load_job()
    assert meta["stages"]["export"]["status"] == "failed"
    assert "capture" not in meta
    if previous_zip:
        assert zip_path.read_bytes() == old_bytes
    else:
        assert not zip_path.exists()
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


def test_absent_portable_metadata_cleanup_is_idempotent(tmp_path):
    clear_portable_capture(tmp_path)
    clear_portable_capture(tmp_path)


@requires_ffmpeg
def test_invalid_url_pipeline_drops_cached_capture(tmp_path, monkeypatch):
    # Exercise real metadata, job binding, reporting and ZIP paths without loading
    # speech/scene models. This is not an end-to-end transcription test.
    video = write_color_video(tmp_path / "recording.mp4", duration=0.4)
    record = build_capture_record(recording_id="audit01", recording_path=str(video), complete=True)
    job = JobDir(tmp_path / "job")
    job.ensure()
    job.bind_source(identity_from_info(inspect_video(video)), ProcessSettings(capture=record))
    write_portable_capture(job.root, record)
    record["source_url_user_supplied"] = "https://example.com:invalid/vsl"
    (video.parent / "capture.json").write_text(json.dumps(record), encoding="utf-8")
    transcript = TranscriptResult(
        status="unavailable", model="none", language=None, language_source="n/a",
        device="none", device_note="test stub",
    )
    monkeypatch.setattr(pipeline, "_stage_transcribe", lambda *a, **kw: transcript)
    monkeypatch.setattr(pipeline, "_stage_scenes", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "_stage_frames", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "_stage_ocr", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "dependency_versions", lambda: {})
    pipeline.process_video(video, job.root, settings=ProcessSettings(ocr=False, compact_view=False))
    assert "capture" not in job.load_job()
    assert not (job.root / "capture-session" / "capture.json").exists()
    with zipfile.ZipFile(job.root / "vsl_study_evidence.zip") as zf:
        assert "capture-session/capture.json" not in zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
        assert "capture" not in manifest
        assert any(gap.get("detail") == INVALID_NOTE for gap in manifest["gaps"])
