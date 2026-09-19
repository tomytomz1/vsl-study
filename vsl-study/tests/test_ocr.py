import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from conftest import requires_ffmpeg, run_ffmpeg
from vsl_study.models import ProcessSettings, ScreenshotRecord
from vsl_study.ocr import (
    any_engine_available,
    apply_ocr,
    format_onscreen_text,
    ocr_image,
    write_ocr_outputs,
)
from vsl_study.pipeline import process_video


def _headline_font(size: int = 48):
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _write_slide(path: Path, lines: list[str]) -> Path:
    image = Image.new("RGB", (960, 540), "white")
    draw = ImageDraw.Draw(image)
    font = _headline_font(56)
    y = 140
    for line in lines:
        draw.text((48, y), line, fill="black", font=font)
        y += 90
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=95)
    return path


def test_ocr_reads_vsl_slide_text(tmp_path: Path):
    path = _write_slide(tmp_path / "slide.jpg", ["YU SLEEP", "BUY NOW 39"])
    text, status, err, engine = ocr_image(path)
    assert status == "ok"
    assert err is None
    assert text
    folded = text.casefold().replace(" ", "")
    assert "sleep" in folded or "buy" in folded or "39" in folded
    assert engine in {"rapidocr", "tesseract", "rapidocr+tesseract"}


def test_apply_ocr_skipped_leaves_transcript_untouched(tmp_path: Path):
    path = _write_slide(tmp_path / "frames" / "frame_0001.jpg", ["OFFER"])
    record = ScreenshotRecord(
        id="frame_0001",
        requested_time=1.0,
        actual_time=1.0,
        scene_id="scene_0001",
        relative_path="frames/frame_0001.jpg",
        capture_reason="interval",
    )
    note = apply_ocr([record], tmp_path, enabled=False)
    assert note is None
    assert record.ocr_status == "skipped"
    assert record.ocr_text is None
    write_ocr_outputs(tmp_path, [record], enabled=False, note=None)
    onscreen = (tmp_path / "onscreen.txt").read_text(encoding="utf-8")
    assert "turned off" in onscreen.lower()
    assert (tmp_path / "ocr.json").exists()


def test_format_onscreen_collapses_duplicate_frames():
    records = [
        ScreenshotRecord(
            id="frame_0001",
            requested_time=1.0,
            actual_time=1.0,
            scene_id="scene_0001",
            relative_path="frames/a.jpg",
            capture_reason="interval",
            ocr_status="ok",
            ocr_engine="rapidocr",
            ocr_text="BUY NOW\n$39",
        ),
        ScreenshotRecord(
            id="frame_0002",
            requested_time=6.0,
            actual_time=6.0,
            scene_id="scene_0001",
            relative_path="frames/b.jpg",
            capture_reason="interval",
            ocr_status="ok",
            ocr_engine="rapidocr",
            ocr_text="BUY NOW\n$39",
        ),
    ]
    body = format_onscreen_text(records)
    assert body.count("BUY NOW") == 1
    assert "same as previous frame" in body
    assert "frame_0002" in body


def test_process_settings_ocr_defaults_on():
    assert ProcessSettings().ocr is True


def _truncated_jpeg(path: Path, keep: int = 36) -> Path:
    buf = io.BytesIO()
    Image.new("RGB", (96, 54), "white").save(buf, format="JPEG", quality=90)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue()[:keep])
    return path


def _shot(frame_id: str, rel: str, t: float = 1.0) -> ScreenshotRecord:
    return ScreenshotRecord(
        id=frame_id,
        requested_time=t,
        actual_time=t,
        scene_id="scene_0001",
        relative_path=rel,
        capture_reason="interval",
    )


def test_ocr_image_truncated_jpeg_is_failed_not_ok(tmp_path: Path):
    if not any_engine_available():
        pytest.skip("OCR engine required to reach image decode")
    path = _truncated_jpeg(tmp_path / "trunc.jpg")
    text, status, err, engine = ocr_image(path)
    assert status == "failed"
    assert text is None
    assert err
    assert engine is None


def test_apply_ocr_continues_after_truncated_and_pillow_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not any_engine_available():
        pytest.skip("OCR engine required to reach image decode")
    from vsl_study import ocr as ocr_mod

    _write_slide(tmp_path / "frames" / "frame_0001.jpg", ["YU SLEEP"])
    _truncated_jpeg(tmp_path / "frames" / "frame_0002.jpg")
    _write_slide(tmp_path / "frames" / "frame_0003.jpg", ["BUY NOW 39"])
    _write_slide(tmp_path / "frames" / "frame_0004.jpg", ["LIMITED"])

    real_preprocess = ocr_mod.preprocess_for_ocr

    def wrapped(image):
        filename = str(getattr(image, "filename", "") or "")
        if "frame_0004" in filename.replace("\\", "/"):
            raise OSError("broken data stream when writing image file")
        return real_preprocess(image)

    monkeypatch.setattr(ocr_mod, "preprocess_for_ocr", wrapped)
    records = [
        _shot("frame_0001", "frames/frame_0001.jpg", 1.0),
        _shot("frame_0002", "frames/frame_0002.jpg", 2.0),
        _shot("frame_0003", "frames/frame_0003.jpg", 3.0),
        _shot("frame_0004", "frames/frame_0004.jpg", 4.0),
    ]
    note = apply_ocr(records, tmp_path, enabled=True)
    assert records[0].ocr_status == "ok"
    assert records[0].ocr_text
    assert records[1].ocr_status == "failed"
    assert records[1].ocr_text is None
    assert records[2].ocr_status == "ok"
    assert records[2].ocr_text
    assert records[3].ocr_status == "failed"
    assert records[3].ocr_text is None
    assert note
    assert "broken data stream when writing image file" in note
    assert {r.ocr_status for r in records} != {"ok"}


@requires_ffmpeg
def test_pipeline_ocr_reads_burned_in_text(tmp_path: Path):
    slide = _write_slide(tmp_path / "slide.jpg", ["LIMITED OFFER", "39"])
    video = tmp_path / "offer.mp4"
    run_ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            str(slide),
            "-t",
            "2",
            "-r",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(video),
        ]
    )
    out = tmp_path / "ocr-job"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=True),
    )
    assert result["screenshot_count"] >= 1
    onscreen = (out / "onscreen.txt").read_text(encoding="utf-8")
    folded = onscreen.casefold()
    assert "offer" in folded or "39" in folded or "limited" in folded
    payload = (out / "ocr.json").read_text(encoding="utf-8")
    assert "frames" in payload
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "On-screen text" in html
