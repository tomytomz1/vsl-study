"""On-screen OCR for VSL slides, prices, and captions.

RapidOCR (bundled ONNX models) is the default engine so the app does not depend
on a PATH install. Tesseract is used when it is present and RapidOCR is weak.
Missing OCR must not block transcription or screenshots.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

from vsl_study.cache import atomic_write_json, atomic_write_text
from vsl_study.models import ScreenshotRecord
from vsl_study.timeutil import format_timecode

OCR_CACHE_VERSION = "v2"
MIN_SCORE = 0.35
ProgressCb = Callable[[str, str], None]

_rapid: Any = None
_rapid_error: str | None = None
_rapid_tried = False
_tesseract_configured = False


def find_tesseract() -> Path | None:
    env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    which = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )
    try:
        import pytesseract

        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        if cmd:
            candidates.append(Path(cmd))
    except Exception:
        pass
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            return path
    return None


def configure_tesseract() -> Path | None:
    global _tesseract_configured
    exe = find_tesseract()
    if exe is None:
        return None
    if _tesseract_configured:
        return exe
    try:
        import subprocess

        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = str(exe)
        original = pytesseract.pytesseract.subprocess_args

        def _hidden_subprocess_args(include_stdout=True):  # noqa: ANN001
            kwargs = original(include_stdout)
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            return kwargs

        pytesseract.pytesseract.subprocess_args = _hidden_subprocess_args
    except Exception:
        return exe
    tessdata = exe.parent / "tessdata"
    if tessdata.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
    bindir = str(exe.parent)
    path = os.environ.get("PATH", "")
    if bindir.lower() not in path.lower():
        os.environ["PATH"] = bindir + os.pathsep + path
    _tesseract_configured = True
    return exe


def tesseract_status() -> tuple[bool, str]:
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract is not installed"
    exe = configure_tesseract()
    if exe is None:
        return False, "Tesseract binary not found (optional fallback)"
    try:
        from vsl_study.ffcmd import run

        result = run([str(exe), "--version"], check=False, timeout=8)
        first = (result.stdout or result.stderr or "").splitlines()
        label = first[0].strip() if first else "Tesseract"
        if result.returncode != 0 and not first:
            return False, f"Tesseract binary not usable (exit {result.returncode})"
        return True, f"{label} ({exe})"
    except Exception as exc:  # noqa: BLE001
        return False, f"Tesseract binary not usable ({exc.__class__.__name__})"


def rapidocr_status() -> tuple[bool, str]:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False, "rapidocr-onnxruntime is not installed"
    except Exception as exc:  # noqa: BLE001
        return False, f"RapidOCR unavailable ({type(exc).__name__}: {exc})"
    return True, "RapidOCR onnxruntime (default on-screen engine)"


def engine_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    ok, detail = rapidocr_status()
    versions["rapidocr"] = detail if ok else f"unavailable ({detail})"
    ok, detail = tesseract_status()
    versions["tesseract"] = detail if ok else f"unavailable ({detail})"
    return versions


def any_engine_available() -> bool:
    ok, _ = rapidocr_status()
    if ok:
        return True
    ok, _ = tesseract_status()
    return ok


def _get_rapidocr() -> Any:
    global _rapid, _rapid_error, _rapid_tried
    if _rapid_tried:
        return _rapid
    _rapid_tried = True
    try:
        from rapidocr_onnxruntime import RapidOCR

        _rapid = RapidOCR()
    except Exception as exc:  # noqa: BLE001
        _rapid_error = f"{type(exc).__name__}: {exc}"
        _rapid = None
    return _rapid


def preprocess_for_ocr(image: Any) -> Any:
    from PIL import Image, ImageFilter, ImageOps, ImageStat

    rgb = image.convert("RGB")
    width, height = rgb.size
    shortest = min(width, height)
    if shortest < 720:
        scale = 720 / max(shortest, 1)
        new_w = min(1920, int(round(width * scale)))
        new_h = min(1920, int(round(height * scale)))
        rgb = rgb.resize((max(new_w, 1), max(new_h, 1)), Image.Resampling.LANCZOS)
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=2))
    gray = rgb.convert("L")
    if ImageStat.Stat(gray).mean[0] < 85:
        rgb = ImageOps.invert(rgb)
    return rgb


def _clean_line(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def _join_lines(lines: list[str]) -> str:
    cleaned = [_clean_line(line) for line in lines]
    return "\n".join(line for line in cleaned if line)


def _alnum_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _items_to_text(items: list[Any] | None, min_score: float = MIN_SCORE) -> tuple[str, float]:
    parsed: list[tuple[float, float, str, float]] = []
    for item in items or []:
        if not item or len(item) < 2:
            continue
        text = _clean_line(str(item[1]))
        try:
            score = float(item[2]) if len(item) > 2 else 1.0
        except (TypeError, ValueError):
            score = 1.0
        if not text or score < min_score:
            continue
        box = item[0]
        try:
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            y = min(ys)
            x = min(xs)
        except Exception:
            y, x = float(len(parsed)), 0.0
        parsed.append((y, x, text, score))
    if not parsed:
        return "", 0.0
    parsed.sort()
    lines: list[str] = []
    current: list[tuple[float, str]] = []
    current_y: float | None = None
    scores: list[float] = []
    y_slack = 16.0
    for y, x, text, score in parsed:
        scores.append(score)
        if current_y is None or abs(y - current_y) <= y_slack:
            current.append((x, text))
            current_y = y if current_y is None else (current_y * 0.6 + y * 0.4)
        else:
            current.sort()
            lines.append(" ".join(part for _, part in current))
            current = [(x, text)]
            current_y = y
    if current:
        current.sort()
        lines.append(" ".join(part for _, part in current))
    mean = sum(scores) / len(scores)
    return _join_lines(lines), mean


def _rapidocr_image(image: Any) -> tuple[str, float, str | None]:
    engine = _get_rapidocr()
    if engine is None:
        return "", 0.0, _rapid_error or "RapidOCR unavailable"
    try:
        import numpy as np

        result, _elapse = engine(np.array(image))
        text, score = _items_to_text(result)
        return text, score, None
    except Exception as exc:  # noqa: BLE001
        return "", 0.0, f"{type(exc).__name__}: {exc}"


def _tesseract_image(image: Any) -> tuple[str, str | None]:
    ok, detail = tesseract_status()
    if not ok:
        return "", detail
    try:
        import pytesseract
        from PIL import ImageOps

        gray = ImageOps.grayscale(image)
        configs = ["--oem 3 --psm 6", "--oem 3 --psm 11"]
        best = ""
        for config in configs:
            raw = pytesseract.image_to_string(gray, config=config) or ""
            text = _join_lines(raw.splitlines())
            if _alnum_count(text) > _alnum_count(best):
                best = text
        return best, None
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"


def _merge_texts(primary: str, secondary: str) -> str:
    if not secondary:
        return primary
    if not primary:
        return secondary
    if _alnum_count(secondary) <= _alnum_count(primary) * 0.5:
        prim_fold = primary.casefold()
        extras = [
            line
            for line in secondary.splitlines()
            if line and line.casefold() not in prim_fold
        ]
        if not extras:
            return primary
        return _join_lines(primary.splitlines() + extras)
    if _alnum_count(primary) >= _alnum_count(secondary):
        return primary
    return secondary


def ocr_image(path: Path) -> tuple[str | None, str, str | None, str | None]:
    """Return (text, status, error, engine)."""
    if not any_engine_available():
        _tesseract_ok, tess_detail = tesseract_status()
        _rapid_ok, rapid_detail = rapidocr_status()
        return None, "unavailable", f"{rapid_detail}; {tess_detail}", None
    try:
        from PIL import Image

        with Image.open(path) as raw:
            prepared = preprocess_for_ocr(raw)
            rapid_text, _score, _rapid_err = _rapidocr_image(prepared)
            engine = "rapidocr" if rapid_text else None
            text = rapid_text
            if _alnum_count(rapid_text) < 12:
                tess_text, _tess_err = _tesseract_image(prepared)
                if tess_text:
                    merged = _merge_texts(rapid_text, tess_text)
                    if merged != rapid_text:
                        engine = "rapidocr+tesseract" if rapid_text else "tesseract"
                    elif not engine and merged:
                        engine = "tesseract"
                    text = merged
            cleaned = text.strip() if text else ""
            if not cleaned:
                return None, "ok", None, None
            return cleaned, "ok", None, engine or "rapidocr"
    except Exception as exc:  # noqa: BLE001
        return None, "failed", str(exc), None


def apply_ocr(
    records: list[ScreenshotRecord],
    job_root: Path,
    enabled: bool,
    progress: ProgressCb | None = None,
) -> str | None:
    if not enabled:
        for record in records:
            record.ocr_status = "skipped"
            record.ocr_text = None
            record.ocr_engine = None
        return None
    if not any_engine_available():
        _ok_r, rapid_detail = rapidocr_status()
        _ok_t, tess_detail = tesseract_status()
        detail = f"{rapid_detail}; {tess_detail}"
        for record in records:
            record.ocr_status = "unavailable"
            record.ocr_text = None
            record.ocr_engine = None
        return detail
    if progress:
        if not _rapid_tried:
            progress("ocr", "Loading OCR models (first run may take a minute)")
        else:
            progress("ocr", "Reading on-screen text from screenshots")
    _get_rapidocr()
    last_error = None
    total = len(records)
    for index, record in enumerate(records):
        path = job_root / record.relative_path
        if not path.is_file():
            record.ocr_status = "failed"
            record.ocr_text = None
            record.ocr_engine = None
            last_error = f"missing image {record.relative_path}"
            continue
        report = total <= 30 or index == 0 or index + 1 == total or (index + 1) % 10 == 0
        if progress and report:
            progress("ocr", f"{index + 1}/{total} {record.id}")
        text, status, err, engine = ocr_image(path)
        record.ocr_status = status
        record.ocr_text = text
        record.ocr_engine = engine
        if err:
            last_error = err
    return last_error


def format_onscreen_text(records: list[ScreenshotRecord]) -> str:
    lines = [
        "# On-screen text from screenshots (OCR).",
        "# This is separate from the spoken Whisper transcript. It is not calibrated accuracy.",
        "",
    ]
    previous = None
    any_text = False
    for record in records:
        stamp = format_timecode(record.actual_time)
        engine = record.ocr_engine or record.ocr_status
        if record.ocr_status == "ok" and record.ocr_text:
            any_text = True
            same = previous is not None and record.ocr_text == previous
            header = f"{record.id}  {stamp}  [{engine}]"
            if same:
                lines.append(f"{header}  (same as previous frame)")
            else:
                lines.append(header)
                lines.append(record.ocr_text)
            lines.append("")
            previous = record.ocr_text
        elif record.ocr_status in {"failed", "unavailable"}:
            lines.append(f"{record.id}  {stamp}  [{record.ocr_status}]")
            lines.append("")
        else:
            previous = None
    if not any_text:
        statuses = {r.ocr_status for r in records} or {"skipped"}
        if statuses <= {"skipped"}:
            lines.append("OCR was turned off for this job.")
        elif statuses <= {"unavailable"}:
            lines.append("OCR engines were unavailable.")
        else:
            lines.append("No on-screen text was detected in the captured frames.")
        lines.append("")
    return "\n".join(lines)


def write_ocr_outputs(root: Path, records: list[ScreenshotRecord], enabled: bool, note: str | None) -> None:
    engines = engine_versions()
    payload = {
        "schema": 1,
        "enabled": enabled,
        "note": (
            "On-screen text is stored separately from spoken transcript. "
            "OCR is not a calibrated accuracy percentage."
        ),
        "engine_note": note,
        "engines": engines,
        "frames": [
            {
                "id": record.id,
                "requested_time": record.requested_time,
                "actual_time": record.actual_time,
                "timecode": format_timecode(record.actual_time),
                "relative_path": record.relative_path,
                "status": record.ocr_status,
                "engine": record.ocr_engine,
                "text": record.ocr_text,
            }
            for record in records
        ],
    }
    atomic_write_json(root / "ocr.json", payload)
    atomic_write_text(root / "onscreen.txt", format_onscreen_text(records))
