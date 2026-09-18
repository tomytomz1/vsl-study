from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_color_video
from vsl_study.capture_meta import (
    INVALID_NOTE,
    MISMATCH_NOTE,
    build_capture_record,
    public_capture_record,
    resolve_capture_for_source,
)
from vsl_study.models import ProcessSettings
from vsl_study.pipeline import PipelineError, add_frames_at, process_video


def _write_capture_dir(
    root: Path,
    rec_id: str,
    video_name: str = "recording.fixed.mp4",
    extra_names: tuple[str, ...] = (),
    *,
    duration: float = 1.2,
    raw: dict | None = None,
) -> tuple[Path, Path]:
    folder = root / rec_id
    folder.mkdir(parents=True, exist_ok=True)
    video = write_color_video(folder / video_name, duration=duration, audio=True)
    for name in extra_names:
        target = folder / name
        if not target.exists():
            target.write_bytes(video.read_bytes())
    record = raw or build_capture_record(
        recording_id=rec_id,
        source_url="https://user:secret@example.com/vsl?token=abc",
        title="Demo capture",
        mime_type="video/webm",
        audio_track=True,
        audio_detected=True,
        stop_reason="max_duration",
        complete=True,
        media_duration_s=duration,
        started_at="2026-09-18T03:08:57Z",
        ended_at="2026-09-18T03:11:01Z",
        recording_path=str(video),
        problems=["Container duration was missing; a copy remux restored seek metadata."],
    )
    (folder / "capture.json").write_text(json.dumps(record), encoding="utf-8")
    return folder, video


@requires_ffmpeg
def test_file_picker_reprocess_preserves_capture_through_export(tmp_path: Path):
    _folder, video = _write_capture_dir(tmp_path, "abc12345def")
    out = tmp_path / "job"
    process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False),
    )
    job = json.loads((out / "job.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    portable = json.loads((out / "capture-session" / "capture.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    html = (out / "report.html").read_text(encoding="utf-8")
    assert manifest["capture"]["recording_id"] == "abc12345def"
    assert job["capture"]["recording_id"] == "abc12345def"
    assert portable["recording_id"] == "abc12345def"
    assert portable["stop_reason"] == "max_duration"
    assert portable["complete"] is True
    assert portable["audio_track"] is True
    assert "secret" not in json.dumps(portable)
    assert "token=abc" not in json.dumps(portable)
    assert "not proof of the selected tab" in report.lower() or "not proof of the selected tab" in html.lower()
    assert "Address the user typed" in report
    with zipfile.ZipFile(out / "vsl_study_evidence.zip") as zf:
        zipped = json.loads(zf.read("manifest.json"))
        session = json.loads(zf.read("capture-session/capture.json"))
    assert zipped["capture"]["recording_id"] == "abc12345def"
    assert session["recording_id"] == "abc12345def"
    assert "meta.json" not in zipfile.ZipFile(out / "vsl_study_evidence.zip").namelist()


@requires_ffmpeg
def test_original_and_remuxed_filenames_share_capture_record(tmp_path: Path):
    folder, fixed = _write_capture_dir(
        tmp_path,
        "rec0000001",
        extra_names=("recording.mp4", "recording.orig.mp4"),
    )
    orig = folder / "recording.mp4"
    cap, notes = resolve_capture_for_source(orig)
    assert notes == []
    assert cap is not None
    assert cap["recording_id"] == "rec0000001"
    cap2, notes2 = resolve_capture_for_source(fixed)
    assert notes2 == []
    assert cap2 is not None
    assert cap2["recording_id"] == "rec0000001"


def test_missing_malformed_and_unrelated_capture_json(tmp_path: Path):
    plain = tmp_path / "clip.mp4"
    plain.write_bytes(b"not-a-video")
    cap, notes = resolve_capture_for_source(plain)
    assert cap is None and notes == []

    (tmp_path / "capture.json").write_text("{not json", encoding="utf-8")
    cap, notes = resolve_capture_for_source(plain)
    assert cap is None
    assert notes == [INVALID_NOTE]

    other = tmp_path / "other"
    other.mkdir()
    video = other / "holiday.mp4"
    video.write_bytes(b"abc")
    record = build_capture_record(
        recording_id="zzzzzzzz",
        recording_path=str(tmp_path / "somewhere" / "recording.webm"),
        complete=True,
        stop_reason="user_stop",
    )
    (other / "capture.json").write_text(json.dumps(record), encoding="utf-8")
    cap, notes = resolve_capture_for_source(video)
    assert cap is None
    assert notes == [MISMATCH_NOTE]


def test_public_record_drops_tokens_and_session_state():
    raw = build_capture_record(
        recording_id="tokenless1",
        source_url="https://user:pass@example.com/v?token=1",
        recording_path=r"C:\captures\tokenless1\recording.webm",
        complete=True,
        stop_reason="user_stop",
    )
    raw["auth_token"] = "super-secret"
    raw["session"] = {"cookie": "abc"}
    cleaned = public_capture_record(raw)
    assert cleaned is not None
    blob = json.dumps(cleaned)
    assert "super-secret" not in blob
    assert "auth_token" not in cleaned
    assert "session" not in cleaned
    assert "pass" not in blob
    assert cleaned["source_url_user_supplied"] == "https://example.com/v"


@requires_ffmpeg
def test_add_frames_keeps_capture_provenance(tmp_path: Path):
    _folder, video = _write_capture_dir(tmp_path, "keepcap01")
    out = tmp_path / "job"
    process_video(video, out, settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False))
    add_frames_at(out, [0.4])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture"]["recording_id"] == "keepcap01"
    portable = json.loads((out / "capture-session" / "capture.json").read_text(encoding="utf-8"))
    assert portable["recording_id"] == "keepcap01"
    with zipfile.ZipFile(out / "vsl_study_evidence.zip") as zf:
        assert json.loads(zf.read("manifest.json"))["capture"]["recording_id"] == "keepcap01"


@requires_ffmpeg
def test_include_media_puts_source_in_zip_with_matching_hash(tmp_path: Path):
    _folder, video = _write_capture_dir(tmp_path, "mediainc1")
    out = tmp_path / "job"
    process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False, include_media=True),
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rel = manifest["packaged_media"]["relative_path"]
    assert rel.startswith("media/")
    expected = __import__("hashlib").sha256(video.read_bytes()).hexdigest()
    assert manifest["packaged_media"]["sha256"] == expected
    assert (out / rel).read_bytes() == video.read_bytes()
    with zipfile.ZipFile(out / "vsl_study_evidence.zip") as zf:
        assert rel in zf.namelist()
        assert __import__("hashlib").sha256(zf.read(rel)).hexdigest() == expected
        extract = tmp_path / "extracted"
        zf.extractall(extract)
    assert (extract / rel).is_file()
    html = (extract / "report.html").read_text(encoding="utf-8")
    md = (extract / "report.md").read_text(encoding="utf-8")
    assert rel in html
    assert rel in md
    assert "not proof of the selected tab" in md.lower() or "not proof of the selected tab" in html.lower()


@requires_ffmpeg
def test_include_media_disabled_omits_recording(tmp_path: Path):
    _folder, video = _write_capture_dir(tmp_path, "medianone")
    out = tmp_path / "job"
    process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False, include_media=False),
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "packaged_media" not in manifest
    with zipfile.ZipFile(out / "vsl_study_evidence.zip") as zf:
        names = zf.namelist()
    assert not any(n.startswith("media/") for n in names)
    assert not any(Path(n).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"} and not n.startswith("frames/") for n in names)


@requires_ffmpeg
def test_media_copy_failure_is_not_complete_export(tmp_path: Path, monkeypatch):
    _folder, video = _write_capture_dir(tmp_path, "mediafail")
    out = tmp_path / "job"

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("vsl_study.pipeline.package_source_media", boom)
    with pytest.raises(PipelineError, match="Could not include the source recording"):
        process_video(
            video,
            out,
            settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False, include_media=True),
        )
    job = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["export"]["status"] == "failed"
    assert job["stages"]["export"]["status"] != "complete"
    assert not (out / "vsl_study_evidence.zip").exists() or job["stages"]["export"]["status"] == "failed"
