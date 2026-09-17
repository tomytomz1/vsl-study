import json
import zipfile
from pathlib import Path

from vsl_study.cache import JobDir, atomic_write_text
from vsl_study.export import write_five_minute_folders, write_zip
from vsl_study.models import ScreenshotRecord, TranscriptResult, TranscriptSegment, VideoInfo


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
