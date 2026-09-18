from __future__ import annotations

import json
from pathlib import Path

from vsl_study.cache import JobDir, atomic_write_json
from vsl_study.capture_meta import (
    INVALID_NOTE,
    build_capture_record,
    clear_portable_capture,
    load_capture_json,
    public_capture_record,
    resolve_capture_for_source,
    write_portable_capture,
)
from vsl_study.export import _capture_markdown_lines
from vsl_study.models import ProcessSettings, SourceIdentity


def _base_record(**overrides) -> dict:
    raw = build_capture_record(
        recording_id="meta0001",
        source_url="https://user:secret@example.com/vsl?token=abc",
        title="Demo",
        mime_type="video/webm",
        audio_track=True,
        audio_detected=True,
        stop_reason="user_stop",
        complete=True,
        media_duration_s=12.5,
        recording_path=r"C:\captures\meta0001\recording.webm",
        problems=["remux restored seek metadata"],
    )
    raw.update(overrides)
    return raw


def test_string_false_complete_is_not_coerced_to_true():
    raw = _base_record(complete="false")
    assert public_capture_record(raw) is None


def test_string_false_audio_flags_are_not_coerced_to_true():
    raw = _base_record(audio_track="false", audio_detected="false")
    assert public_capture_record(raw) is None


def test_problems_non_list_is_invalid_not_typeerror(tmp_path: Path):
    raw = _base_record(problems=42)
    assert public_capture_record(raw) is None
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_capture_json(path) is None


def test_explicit_false_booleans_are_preserved():
    raw = _base_record(complete=False, audio_track=False, audio_detected=False)
    cleaned = public_capture_record(raw)
    assert cleaned is not None
    assert cleaned["complete"] is False
    assert cleaned["audio_track"] is False
    assert cleaned["audio_detected"] is False
    text = "\n".join(_capture_markdown_lines(cleaned))
    assert "complete: False" in text
    assert "claimed=False" in text
    assert "detected=False" in text
    assert "may cover only part of the video" in text


def test_missing_optional_fields_are_not_false():
    raw = {
        "recording_id": "meta0002",
        "input_type": "browser_tab_recording",
        "recording_path": r"C:\captures\meta0002\recording.webm",
    }
    cleaned = public_capture_record(raw)
    assert cleaned is not None
    assert cleaned["complete"] is None
    assert cleaned["audio_track"] is None
    assert cleaned["audio_detected"] is None
    assert cleaned["media_duration_s"] is None
    assert cleaned["problems"] == []
    text = "\n".join(_capture_markdown_lines(cleaned))
    assert "complete: unspecified" in text
    assert "complete: True" not in text
    assert "complete: False" not in text
    assert "may cover only part of the video" not in text
    assert "claimed=" not in text


def test_valid_existing_record_still_round_trips():
    raw = _base_record()
    cleaned = public_capture_record(raw)
    assert cleaned is not None
    again = public_capture_record(cleaned)
    assert again == cleaned
    assert cleaned["complete"] is True
    assert cleaned["audio_track"] is True
    assert cleaned["audio_detected"] is True
    assert cleaned["media_duration_s"] == 12.5
    assert "secret" not in json.dumps(cleaned)
    assert "token=abc" not in json.dumps(cleaned)
    assert "auth_token" not in cleaned


def test_invalid_duration_rejects_record():
    for value in (float("nan"), float("inf"), float("-inf"), -0.01, True, "12.5"):
        assert public_capture_record(_base_record(media_duration_s=value)) is None
    cleaned = public_capture_record(_base_record(media_duration_s=0))
    assert cleaned is not None
    assert cleaned["media_duration_s"] == 0.0


def test_resolve_malformed_capture_json_is_unavailable(tmp_path: Path):
    rec_id = "badjson01"
    folder = tmp_path / rec_id
    folder.mkdir()
    video = folder / "recording.webm"
    video.write_bytes(b"webm")
    raw = build_capture_record(
        recording_id=rec_id,
        recording_path=str(video),
        complete=True,
        audio_track=True,
        audio_detected=True,
        stop_reason="user_stop",
    )
    raw["complete"] = "false"
    raw["audio_track"] = "false"
    raw["audio_detected"] = "false"
    raw["problems"] = 42
    (folder / "capture.json").write_text(json.dumps(raw), encoding="utf-8")
    cap, notes = resolve_capture_for_source(video)
    assert cap is None
    assert notes == [INVALID_NOTE]


def test_desktop_file_picker_path_drops_invalid_capture(tmp_path: Path):
    rec_id = "pickerbad1"
    folder = tmp_path / rec_id
    folder.mkdir()
    video = folder / "recording.webm"
    video.write_bytes(b"webm")
    raw = build_capture_record(
        recording_id=rec_id,
        recording_path=str(video),
        complete=True,
        stop_reason="user_stop",
    )
    raw["complete"] = "false"
    (folder / "capture.json").write_text(json.dumps(raw), encoding="utf-8")
    capture, notes = resolve_capture_for_source(video)
    cleaned = public_capture_record(capture) if capture else None
    settings = ProcessSettings(capture=dict(cleaned) if cleaned else None)
    assert settings.capture is None
    assert notes == [INVALID_NOTE]


def test_rejected_capture_does_not_stay_on_job_or_portable_file(tmp_path: Path):
    rec_id = "recid0001"
    video = tmp_path / rec_id / "recording.webm"
    video.parent.mkdir()
    video.write_bytes(b"abc")
    identity = SourceIdentity(
        path=str(video),
        resolved_path=str(video.resolve()),
        size_bytes=3,
        mtime_ns=1,
        fingerprint="ff",
        duration_s=1.0,
    )
    job = JobDir(tmp_path / "job")
    job.ensure()
    good = build_capture_record(
        recording_id=rec_id,
        recording_path=str(video),
        complete=True,
        stop_reason="user_stop",
    )
    job.bind_source(identity, ProcessSettings(capture=good, ocr=False))
    write_portable_capture(job.root, good)
    assert job.load_job()["capture"]["complete"] is True
    assert (job.root / "capture-session" / "capture.json").is_file()

    job.bind_source(identity, ProcessSettings(capture=None, ocr=False), capture_rejected=True)
    clear_portable_capture(job.root)
    assert "capture" not in job.load_job()
    assert not (job.root / "capture-session" / "capture.json").exists()


def test_invalid_job_capture_cannot_reappear_when_not_provided(tmp_path: Path):
    rec_id = "recid0002"
    video = tmp_path / rec_id / "recording.webm"
    video.parent.mkdir()
    video.write_bytes(b"abc")
    identity = SourceIdentity(
        path=str(video),
        resolved_path=str(video.resolve()),
        size_bytes=3,
        mtime_ns=1,
        fingerprint="ee",
        duration_s=1.0,
    )
    job = JobDir(tmp_path / "job")
    job.ensure()
    good = build_capture_record(
        recording_id=rec_id,
        recording_path=str(video),
        complete=True,
        stop_reason="user_stop",
    )
    job.bind_source(identity, ProcessSettings(capture=good, ocr=False))
    stored = job.load_job()
    stored["capture"]["complete"] = "false"
    stored["capture"]["audio_track"] = "false"
    stored["capture"]["problems"] = 42
    atomic_write_json(job.job_file, stored)
    job.bind_source(identity, ProcessSettings(capture=None, ocr=False), capture_rejected=False)
    assert "capture" not in (job.load_job() or {})
