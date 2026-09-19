from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from vsl_study.models import ScreenshotRecord
from vsl_study.ocr import any_engine_available, apply_ocr, format_onscreen_text, images_equivalent


def _font(size: int = 48):
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _slide(path: Path, lines: list[str]) -> Path:
    image = Image.new("RGB", (960, 540), "white")
    draw = ImageDraw.Draw(image)
    font = _font(56)
    y = 140
    for line in lines:
        draw.text((48, y), line, fill="black", font=font)
        y += 90
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=95)
    return path


def _shot(frame_id: str, rel: str, t: float, reason: str = "interval") -> ScreenshotRecord:
    return ScreenshotRecord(
        id=frame_id,
        requested_time=t,
        actual_time=t,
        scene_id="scene_0001",
        relative_path=rel,
        capture_reason=reason,
    )


def test_changed_price_is_not_treated_as_duplicate(tmp_path: Path):
    a = _slide(tmp_path / "a.jpg", ["YU SLEEP", "BUY NOW 39"])
    b = _slide(tmp_path / "b.jpg", ["YU SLEEP", "BUY NOW 29"])
    digest: dict[str, str] = {}
    preview: dict[str, object] = {}
    assert images_equivalent(a, a) is True
    assert images_equivalent(a, b) is False
    assert images_equivalent(a, b, digest_cache=digest, preview_cache=preview) is False
    assert images_equivalent(a, b, digest_cache=digest, preview_cache=preview) is False
    assert images_equivalent(a, a, digest_cache=digest, preview_cache=preview) is True
    identical = tmp_path / "a_copy.jpg"
    identical.write_bytes(a.read_bytes())
    assert images_equivalent(a, identical, digest_cache=digest, preview_cache=preview) is True


def test_sampling_skip_is_not_reported_as_no_text(tmp_path: Path):
    if not any_engine_available():
        pytest.skip("OCR engine required")
    _slide(tmp_path / "frames" / "frame_0001.jpg", ["OFFER"])
    _slide(tmp_path / "frames" / "frame_0002.jpg", ["PRICE 39"])
    records = [
        _shot("frame_0001", "frames/frame_0001.jpg", 1.0),
        _shot("frame_0002", "frames/frame_0002.jpg", 2.0),
    ]
    apply_ocr(records, tmp_path, enabled=True, selected_ids={"frame_0001"})
    assert records[0].ocr_status in {"ok", "failed", "unavailable"}
    assert records[1].ocr_status == "skipped"
    assert records[1].ocr_skip_reason == "sampling_policy"
    body = format_onscreen_text(records)
    assert "skipped by sampling policy" in body
    assert "OCR was turned off" not in body
    if records[0].ocr_status == "ok" and not records[0].ocr_text:
        assert "were not checked for text" in body


def test_ocr_checkpoint_resume_skips_completed(tmp_path: Path, monkeypatch):
    if not any_engine_available():
        pytest.skip("OCR engine required")
    from vsl_study import ocr as ocr_mod
    from vsl_study.cache import atomic_write_json

    _slide(tmp_path / "frames" / "frame_0001.jpg", ["ONE"])
    _slide(tmp_path / "frames" / "frame_0002.jpg", ["TWO"])
    checkpoint = tmp_path / "cache" / "ocr-checkpoint.json"
    atomic_write_json(
        checkpoint,
        {
            "screenshots": [
                {
                    "id": "frame_0001",
                    "ocr_text": "ONE",
                    "ocr_status": "ok",
                    "ocr_engine": "rapidocr",
                    "ocr_skip_reason": "",
                    "ocr_reused_from": "",
                    "selected_for_ocr": True,
                }
            ]
        },
    )
    calls: list[str] = []
    real = ocr_mod.ocr_image

    def wrapped(path, **kwargs):
        calls.append(Path(path).name)
        return real(path, **kwargs)

    monkeypatch.setattr(ocr_mod, "ocr_image", wrapped)
    records = [
        _shot("frame_0001", "frames/frame_0001.jpg", 1.0),
        _shot("frame_0002", "frames/frame_0002.jpg", 2.0),
    ]
    apply_ocr(
        records,
        tmp_path,
        enabled=True,
        selected_ids={"frame_0001", "frame_0002"},
        checkpoint_path=checkpoint,
    )
    assert records[0].ocr_status == "ok"
    assert records[0].ocr_text == "ONE"
    assert "frame_0001.jpg" not in calls
    assert records[1].ocr_status in {"ok", "failed", "unavailable", "reused"}


def test_apply_ocr_does_not_reuse_changed_price(tmp_path: Path):
    if not any_engine_available():
        pytest.skip("OCR engine required")
    _slide(tmp_path / "frames" / "frame_0001.jpg", ["YU SLEEP", "BUY NOW 39"])
    _slide(tmp_path / "frames" / "frame_0002.jpg", ["YU SLEEP", "BUY NOW 29"])
    records = [
        _shot("frame_0001", "frames/frame_0001.jpg", 1.0),
        _shot("frame_0002", "frames/frame_0002.jpg", 2.0),
    ]
    apply_ocr(records, tmp_path, enabled=True, selected_ids={"frame_0001", "frame_0002"})
    assert records[1].ocr_status != "reused"
    assert records[1].ocr_reused_from == ""
    if records[0].ocr_status == "ok" and records[1].ocr_status == "ok":
        assert (records[0].ocr_text or "") != (records[1].ocr_text or "")
