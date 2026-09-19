import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_color_video
from vsl_study.cache import JobDir, atomic_write_text
from vsl_study.export import write_five_minute_folders, write_reports, write_zip
from vsl_study.models import ProcessSettings, Scene, ScreenshotRecord, TranscriptResult, TranscriptSegment, VideoInfo
from vsl_study.pipeline import process_video


def _info(path: str, duration: float = 700.0) -> VideoInfo:
    return VideoInfo(
        path=path,
        resolved_path=path,
        size_bytes=1,
        mtime_ns=1,
        fingerprint="abc",
        duration_s=duration,
        width=640,
        height=360,
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
    )


def test_five_minute_folders_preserve_boundary_ids(tmp_path: Path):
    job = JobDir(tmp_path / "job")
    job.ensure()
    img = job.frames / "frame_0001_t00-04-50-000.jpg"
    img.write_bytes(b"not-a-real-jpeg")
    transcript = TranscriptResult(
        status="complete",
        model="tiny.en",
        language="en",
        language_source="configured",
        device="cpu",
        device_note="test",
        segments=[
            TranscriptSegment(id="seg_0007", start=290.0, end=310.0, text="crossing the five minute mark"),
            TranscriptSegment(id="seg_0008", start=400.0, end=410.0, text="later folder only"),
        ],
        text="crossing the five minute mark later folder only",
    )
    shots = [
        ScreenshotRecord(
            id="frame_0001",
            requested_time=290.0,
            actual_time=290.0,
            scene_id="scene_0001",
            relative_path="frames/frame_0001_t00-04-50-000.jpg",
            capture_reason="interval",
            ocr_status="ok",
            ocr_engine="rapidocr",
            ocr_text="SALE $39",
        )
    ]
    names = write_five_minute_folders(job, _info("x.mp4", 700), transcript, shots)
    assert "0000-0005" in names
    assert "0005-0010" in names
    first = json.loads((job.clips / "0000-0005" / "frames_manifest.json").read_text(encoding="utf-8"))
    second = json.loads((job.clips / "0005-0010" / "frames_manifest.json").read_text(encoding="utf-8"))
    assert "seg_0007" in first["transcript_segment_ids"]
    assert "seg_0007" in second["transcript_segment_ids"]
    assert "seg_0008" in second["transcript_segment_ids"]
    assert "seg_0008" not in first["transcript_segment_ids"]
    text = (job.clips / "0000-0005" / "transcript_timestamped.txt").read_text(encoding="utf-8")
    assert "seg_0007" in text
    assert "spans_folder_boundary" in text
    onscreen = (job.clips / "0000-0005" / "onscreen.txt").read_text(encoding="utf-8")
    assert "SALE $39" in onscreen
    assert "frame_0001" in onscreen


def test_zip_excludes_audio_and_preserves_relative_html(tmp_path: Path):
    job = JobDir(tmp_path / "job")
    job.ensure()
    atomic_write_text(
        job.root / "report.html",
        '<html><img src="frames/frame_0001.jpg"></html>',
    )
    (job.frames / "frame_0001.jpg").write_bytes(b"jpeg")
    (job.cache / "audio.wav").write_bytes(b"RIFF")
    (job.work / "tmp.bin").write_bytes(b"x")
    zpath = write_zip(job, include_media=False)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert "report.html" in names
    assert "frames/frame_0001.jpg" in names
    assert "cache/audio.wav" not in names
    assert not any(n.startswith("work/") for n in names)
    extract = tmp_path / "extracted"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract)
    html = (extract / "report.html").read_text(encoding="utf-8")
    assert 'src="frames/frame_0001.jpg"' in html
    assert (extract / "frames" / "frame_0001.jpg").exists()


def test_zip_includes_packaged_media_when_enabled(tmp_path: Path):
    from vsl_study.export import package_source_media, write_zip

    job = JobDir(tmp_path / "job")
    job.ensure()
    source = tmp_path / "recording.fixed.mp4"
    source.write_bytes(b"videobytes" * 1000)
    packaged = package_source_media(job, source)
    (job.root / "report.html").write_text('<html><img src="frames/a.jpg"></html>', encoding="utf-8")
    (job.frames / "a.jpg").write_bytes(b"jpeg")
    (job.cache / "audio.wav").write_bytes(b"RIFF")
    zpath = write_zip(job, include_media=True, packaged_media=packaged)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert packaged["relative_path"] in names
        assert zf.read(packaged["relative_path"]) == source.read_bytes()
        assert "cache/audio.wav" in names
        extract = tmp_path / "out"
        zf.extractall(extract)
    assert (extract / packaged["relative_path"]).read_bytes() == source.read_bytes()


def test_zip_without_requested_media_cannot_succeed(tmp_path: Path):
    job = JobDir(tmp_path / "job")
    job.ensure()
    atomic_write_text(job.root / "report.html", "<html></html>")
    with pytest.raises(RuntimeError, match="source recording was not copied"):
        write_zip(job, include_media=True, packaged_media=None)


def _report_job(tmp_path: Path, html: str = "<html>v1</html>") -> JobDir:
    job = JobDir(tmp_path / "job")
    job.ensure()
    atomic_write_text(job.root / "report.html", html)
    (job.frames / "frame.jpg").write_bytes(b"jpeg")
    return job


def test_zip_successful_replacement(tmp_path: Path):
    job = _report_job(tmp_path, "<html>v1</html>")
    first = write_zip(job, include_media=False)
    with zipfile.ZipFile(first) as zf:
        assert zf.read("report.html") == b"<html>v1</html>"
    atomic_write_text(job.root / "report.html", "<html>v2</html>")
    second = write_zip(job, include_media=False)
    assert second == first
    assert second.name == "vsl_study_evidence.zip"
    with zipfile.ZipFile(second) as zf:
        assert zf.read("report.html") == b"<html>v2</html>"
        names = zf.namelist()
    assert "vsl_study_evidence.zip" not in names
    assert "vsl_study_evidence.zip.part" not in names
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


def test_zip_write_failure_preserves_existing_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = _report_job(tmp_path)
    zpath = write_zip(job, include_media=False)
    original = zpath.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    atomic_write_text(job.root / "report.html", "<html>changed</html>")

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", boom)
    with pytest.raises(OSError, match="disk full"):
        write_zip(job, include_media=False)
    assert zpath.is_file()
    assert zpath.read_bytes() == original
    assert hashlib.sha256(zpath.read_bytes()).hexdigest() == digest
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


def test_zip_validation_failure_preserves_existing_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = _report_job(tmp_path)
    zpath = write_zip(job, include_media=False)
    original = zpath.read_bytes()
    atomic_write_text(job.root / "report.html", "<html>changed</html>")

    def bad_testzip(self):
        return "report.html"

    monkeypatch.setattr(zipfile.ZipFile, "testzip", bad_testzip)
    with pytest.raises(RuntimeError, match="integrity check"):
        write_zip(job, include_media=False)
    assert zpath.read_bytes() == original
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


def test_zip_write_failure_without_previous_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = _report_job(tmp_path)
    zpath = job.root / "vsl_study_evidence.zip"
    assert not zpath.exists()

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", boom)
    with pytest.raises(OSError, match="disk full"):
        write_zip(job, include_media=False)
    assert not zpath.exists()
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


def test_zip_does_not_include_temporary_archive(tmp_path: Path):
    job = _report_job(tmp_path)
    leftover = job.root / "vsl_study_evidence.zip.part"
    leftover.write_bytes(b"not-a-zip-should-not-be-archived")
    zpath = write_zip(job, include_media=False)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        blob = b"".join(zf.read(name) for name in names)
    assert "vsl_study_evidence.zip" not in names
    assert "vsl_study_evidence.zip.part" not in names
    assert not any(name.endswith(".part") for name in names)
    assert b"not-a-zip-should-not-be-archived" not in blob
    assert not leftover.exists()


def test_zip_media_validation_failure_preserves_existing_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from vsl_study.export import package_source_media

    job = _report_job(tmp_path)
    zpath = write_zip(job, include_media=False)
    original = zpath.read_bytes()
    source = tmp_path / "recording.fixed.mp4"
    source.write_bytes(b"videobytes" * 1000)
    packaged = package_source_media(job, source)
    real_namelist = zipfile.ZipFile.namelist

    def hidden_media(self):
        return [name for name in real_namelist(self) if not str(name).replace("\\", "/").startswith("media/")]

    monkeypatch.setattr(zipfile.ZipFile, "namelist", hidden_media)
    with pytest.raises(RuntimeError, match="without the source recording"):
        write_zip(job, include_media=True, packaged_media=packaged)
    assert zpath.read_bytes() == original
    assert not (job.root / "vsl_study_evidence.zip.part").exists()


@requires_ffmpeg
def test_failed_pipeline_rebuild_preserves_previous_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = write_color_video(tmp_path / "src.mp4", duration=1.2, audio=True)
    out = tmp_path / "job"
    process_video(video, out, settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False))
    zpath = out / "vsl_study_evidence.zip"
    original = zpath.read_bytes()
    digest = hashlib.sha256(original).hexdigest()

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("vsl_study.pipeline.write_zip", boom)
    result = process_video(video, out, settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False))
    assert result["package_status"] == "failed"
    assert result["report_ready"] is True
    assert result.get("zip") is None
    assert "disk full" in str(result.get("package_error") or "")
    assert zpath.read_bytes() == original
    assert hashlib.sha256(zpath.read_bytes()).hexdigest() == digest
    job = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["export"]["status"] == "failed"
    assert job["stages"]["export"]["status"] != "complete"
    assert (out / "report.html").exists()
    assert job["stages"]["reports"]["status"] == "complete"


def test_write_reports_survives_contact_sheet_pillow_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = JobDir(tmp_path / "job")
    job.ensure()
    (job.frames / "frame_0001.jpg").write_bytes(b"not-a-jpeg")
    shot = ScreenshotRecord(
        id="frame_0001",
        requested_time=1.0,
        actual_time=1.0,
        scene_id="scene_0001",
        relative_path="frames/frame_0001.jpg",
        capture_reason="interval",
    )
    transcript = TranscriptResult(
        status="complete",
        model="tiny.en",
        language="en",
        language_source="configured",
        device="cpu",
        device_note="test",
        segments=[TranscriptSegment(id="seg_0001", start=0.0, end=1.0, text="hello")],
        text="hello",
    )
    gaps: list[dict] = []

    def boom(*_args, **_kwargs):
        raise OSError("broken data stream when writing image file")

    monkeypatch.setattr("vsl_study.export.write_contact_sheet", boom)
    write_reports(
        job,
        _info("x.mp4", 12.0),
        ProcessSettings(compact_view=True, ocr=False),
        transcript,
        [Scene("scene_0001", 0.0, 12.0)],
        [shot],
        gaps,
        {"pillow": "test"},
        {"frames": {"status": "complete"}},
    )
    assert (job.root / "report.md").is_file()
    assert (job.root / "report.html").is_file()
    assert any("broken data stream when writing image file" in str(g.get("detail")) for g in gaps)
    md = (job.root / "report.md").read_text(encoding="utf-8")
    assert "broken data stream when writing image file" in md


@requires_ffmpeg
def test_pipeline_survives_contact_sheet_pillow_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = write_color_video(tmp_path / "src.mp4", duration=1.2, audio=True)

    def boom(*_args, **_kwargs):
        raise OSError("broken data stream when writing image file")

    monkeypatch.setattr("vsl_study.export.write_contact_sheet", boom)
    out = tmp_path / "job"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False),
    )
    assert (out / "vsl_study_evidence.zip").is_file()
    assert (out / "report.html").is_file()
    job = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["export"]["status"] == "complete"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "broken data stream when writing image file" in report
    assert result["screenshot_count"] >= 1

