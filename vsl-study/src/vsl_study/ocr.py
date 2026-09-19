"""On-screen OCR for VSL slides, prices, and captions.

RapidOCR (bundled ONNX models) is the default engine so the app does not depend
on a PATH install. Tesseract is used when it is present and RapidOCR is weak.
Missing OCR must not block transcription or screenshots.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from vsl_study.cache import atomic_write_json, atomic_write_text
from vsl_study.models import ScreenshotRecord
from vsl_study.timeutil import format_timecode

OCR_CACHE_VERSION = "v3-select"
MIN_SCORE = 0.35
EQUIV_MAX_DIFF = 18
EQUIV_MEAN_DIFF = 2.0
CHECKPOINT_EVERY = 10
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
    """Standard path keeps native pixels. No unconditional 720px upscale."""
    from PIL import ImageOps

    return ImageOps.autocontrast(image.convert("RGB"), cutoff=0)


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


def _tesseract_image(image: Any, *, tess_ok: bool | None = None) -> tuple[str, str | None]:
    ok = tess_ok if tess_ok is not None else tesseract_status()[0]
    if not ok:
        detail = tesseract_status()[1] if tess_ok is None else "Tesseract unavailable"
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


def probe_ocr_engines() -> tuple[bool, bool, str]:
    rapid_ok, rapid_detail = rapidocr_status()
    tess_ok, tess_detail = tesseract_status()
    return rapid_ok, tess_ok, f"{rapid_detail}; {tess_detail}"


def _gray_preview(path: Path, width: int = 320):
    from PIL import Image

    with Image.open(path) as raw:
        rgb = raw.convert("RGB")
        if rgb.size[0] > width:
            height = max(1, int(round(rgb.size[1] * (width / rgb.size[0]))))
            rgb = rgb.resize((width, height))
        return rgb.convert("L")


def _file_digest(path: Path, cache: dict[str, str] | None = None) -> str:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache is not None:
        cache[key] = digest
    return digest


def images_equivalent(
    path_a: Path,
    path_b: Path,
    *,
    digest_cache: dict[str, str] | None = None,
    preview_cache: dict[str, Any] | None = None,
) -> bool:
    """Conservative pixel match. Layout-similar slides with changed text must not match."""
    if not path_a.is_file() or not path_b.is_file():
        return False
    try:
        if path_a.resolve() == path_b.resolve():
            return True
    except OSError:
        pass
    if _file_digest(path_a, digest_cache) == _file_digest(path_b, digest_cache):
        return True
    try:
        from PIL import ImageChops, ImageStat

        def preview(path: Path):
            key = str(path)
            if preview_cache is not None and key in preview_cache:
                return preview_cache[key]
            image = _gray_preview(path)
            if preview_cache is not None:
                preview_cache[key] = image
            return image

        left = preview(path_a)
        right = preview(path_b)
        if left.size != right.size:
            right = right.resize(left.size)
        diff = ImageChops.difference(left, right)
        extrema = diff.getextrema()
        max_diff = extrema[1] if isinstance(extrema, tuple) else 255
        mean_diff = float(ImageStat.Stat(diff).mean[0])
        return max_diff <= EQUIV_MAX_DIFF and mean_diff <= EQUIV_MEAN_DIFF
    except Exception:
        return False


def ocr_image(
    path: Path,
    *,
    rapid_ok: bool | None = None,
    tess_ok: bool | None = None,
) -> tuple[str | None, str, str | None, str | None]:
    """Return (text, status, error, engine)."""
    engine_detail = None
    if rapid_ok is None or tess_ok is None:
        probed_rapid, probed_tess, engine_detail = probe_ocr_engines()
        rapid_ok = probed_rapid if rapid_ok is None else rapid_ok
        tess_ok = probed_tess if tess_ok is None else tess_ok
    if not rapid_ok and not tess_ok:
        if engine_detail is None:
            _rapid_ok, rapid_detail = rapidocr_status()
            _tess_ok, tess_detail = tesseract_status()
            engine_detail = f"{rapid_detail}; {tess_detail}"
        return None, "unavailable", engine_detail, None
    try:
        from PIL import Image

        with Image.open(path) as raw:
            prepared = preprocess_for_ocr(raw)
            rapid_text, _score, _rapid_err = _rapidocr_image(prepared) if rapid_ok else ("", 0.0, None)
            engine = "rapidocr" if rapid_text else None
            text = rapid_text
            if tess_ok and _alnum_count(rapid_text) < 12:
                tess_text, _tess_err = _tesseract_image(prepared, tess_ok=True)
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
            return cleaned, "ok", None, engine or ("rapidocr" if rapid_ok else "tesseract")
    except Exception as exc:  # noqa: BLE001
        return None, "failed", str(exc), None


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        from vsl_study.cache import read_json

        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_checkpoint_row(record: ScreenshotRecord, row: dict[str, Any]) -> None:
    record.ocr_text = row.get("ocr_text")
    record.ocr_status = str(row.get("ocr_status") or "skipped")
    record.ocr_engine = row.get("ocr_engine")
    record.ocr_skip_reason = str(row.get("ocr_skip_reason") or "")
    record.ocr_reused_from = str(row.get("ocr_reused_from") or "")
    record.selected_for_ocr = bool(row.get("selected_for_ocr", True))


def _checkpoint_row(record: ScreenshotRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "ocr_text": record.ocr_text,
        "ocr_status": record.ocr_status,
        "ocr_engine": record.ocr_engine,
        "ocr_skip_reason": record.ocr_skip_reason,
        "ocr_reused_from": record.ocr_reused_from,
        "selected_for_ocr": record.selected_for_ocr,
    }


def apply_ocr(
    records: list[ScreenshotRecord],
    job_root: Path,
    enabled: bool,
    progress: ProgressCb | None = None,
    *,
    selected_ids: set[str] | None = None,
    checkpoint_path: Path | None = None,
) -> str | None:
    if not enabled:
        for record in records:
            record.ocr_status = "skipped"
            record.ocr_text = None
            record.ocr_engine = None
            record.ocr_skip_reason = "disabled"
            record.ocr_reused_from = ""
            record.selected_for_ocr = False
        return None
    rapid_ok, tess_ok, detail = probe_ocr_engines()
    if not rapid_ok and not tess_ok:
        for record in records:
            record.ocr_status = "unavailable"
            record.ocr_text = None
            record.ocr_engine = None
            record.ocr_skip_reason = ""
            record.ocr_reused_from = ""
            record.selected_for_ocr = False
        return detail
    targets = set(selected_ids) if selected_ids is not None else {record.id for record in records}
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    saved_rows = {str(row.get("id")): row for row in checkpoint.get("screenshots") or [] if row.get("id")}
    for record in records:
        row = saved_rows.get(record.id)
        if row:
            _apply_checkpoint_row(record, row)

    if progress:
        if not _rapid_tried:
            progress("ocr", "Loading OCR models (first run may take a minute)")
        else:
            progress("ocr", "Reading on-screen text from screenshots")
    if rapid_ok:
        _get_rapidocr()
    last_error = None
    completed_ok = [record for record in records if record.ocr_status in {"ok", "reused"}]
    pending = [
        record
        for record in records
        if record.id in targets and record.ocr_status not in {"ok", "reused", "failed", "unavailable"}
    ]
    total = max(len(targets), 1)
    done = sum(
        1
        for record in records
        if record.id in targets and record.ocr_status in {"ok", "reused", "failed", "unavailable"}
    )
    for record in records:
        if record.id in targets:
            record.selected_for_ocr = True
            continue
        if record.ocr_status in {"ok", "reused", "failed", "unavailable"}:
            continue
        record.ocr_status = "skipped"
        record.ocr_text = None
        record.ocr_engine = None
        record.ocr_skip_reason = "sampling_policy"
        record.ocr_reused_from = ""
        record.selected_for_ocr = False

    writes = 0
    digest_cache: dict[str, str] = {}
    preview_cache: dict[str, Any] = {}
    for record in pending:
        path = job_root / record.relative_path
        done += 1
        report = total <= 30 or done == 1 or done == total or done % 10 == 0
        if progress and report:
            progress("ocr", f"{done}/{total} {record.id}")
        if not path.is_file():
            record.ocr_status = "failed"
            record.ocr_text = None
            record.ocr_engine = None
            record.ocr_skip_reason = ""
            last_error = f"missing image {record.relative_path}"
        else:
            reused = None
            for prior in completed_ok:
                prior_path = job_root / prior.relative_path
                if prior.ocr_status in {"ok", "reused"} and images_equivalent(
                    path,
                    prior_path,
                    digest_cache=digest_cache,
                    preview_cache=preview_cache,
                ):
                    reused = prior
                    break
            if reused is not None:
                record.ocr_status = "reused"
                record.ocr_text = reused.ocr_text
                record.ocr_engine = reused.ocr_engine
                record.ocr_skip_reason = ""
                record.ocr_reused_from = reused.id
            else:
                text, status, err, engine = ocr_image(path, rapid_ok=rapid_ok, tess_ok=tess_ok)
                record.ocr_status = status
                record.ocr_text = text
                record.ocr_engine = engine
                record.ocr_skip_reason = ""
                record.ocr_reused_from = ""
                if err:
                    last_error = err
                if status == "ok":
                    completed_ok.append(record)
        writes += 1
        if checkpoint_path and (writes % CHECKPOINT_EVERY == 0 or done >= total):
            atomic_write_json(
                checkpoint_path,
                {"screenshots": [_checkpoint_row(item) for item in records]},
            )
    if checkpoint_path:
        atomic_write_json(checkpoint_path, {"screenshots": [_checkpoint_row(item) for item in records]})
    return last_error


def format_onscreen_text(records: list[ScreenshotRecord]) -> str:
    lines = [
        "# On-screen text from screenshots (OCR).",
        "# This is separate from the spoken Whisper transcript. It is not calibrated accuracy.",
        "# Sampled OCR is not an exhaustive check of every screenshot.",
        "",
    ]
    previous = None
    any_text = False
    sampled_skip = 0
    for record in records:
        stamp = format_timecode(record.actual_time)
        engine = record.ocr_engine or record.ocr_status
        if record.ocr_status in {"ok", "reused"} and record.ocr_text:
            any_text = True
            same = previous is not None and record.ocr_text == previous
            header = f"{record.id}  {stamp}  [{engine}]"
            if record.ocr_status == "reused" and record.ocr_reused_from:
                header += f"  (reused from {record.ocr_reused_from})"
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
        elif record.ocr_status == "skipped" and record.ocr_skip_reason == "sampling_policy":
            sampled_skip += 1
            lines.append(f"{record.id}  {stamp}  [skipped by sampling policy]")
            lines.append("")
            previous = None
        else:
            previous = None
    if not any_text:
        statuses = {r.ocr_status for r in records} or {"skipped"}
        skip_reasons = {r.ocr_skip_reason for r in records}
        if statuses <= {"skipped"} and skip_reasons <= {"disabled", ""}:
            lines.append("OCR was turned off for this job.")
        elif statuses <= {"unavailable"}:
            lines.append("OCR engines were unavailable.")
        elif sampled_skip:
            lines.append(
                "No on-screen text was detected in the OCR-selected frames. "
                "Other screenshots were skipped by sampling policy and were not checked for text."
            )
        else:
            lines.append("No on-screen text was detected in the captured frames.")
        lines.append("")
    elif sampled_skip:
        lines.append(
            f"# {sampled_skip} screenshot(s) were skipped by sampling policy and were not OCR-checked."
        )
        lines.append("")
    return "\n".join(lines)


def write_ocr_outputs(root: Path, records: list[ScreenshotRecord], enabled: bool, note: str | None) -> None:
    engines = engine_versions()
    selected = sum(1 for record in records if record.selected_for_ocr)
    reused = sum(1 for record in records if record.ocr_status == "reused")
    skipped = sum(1 for record in records if record.ocr_skip_reason == "sampling_policy")
    payload = {
        "schema": 1,
        "enabled": enabled,
        "note": (
            "On-screen text is stored separately from spoken transcript. "
            "OCR is not a calibrated accuracy percentage. "
            "Skipped-by-policy frames were not checked for text."
        ),
        "engine_note": note,
        "engines": engines,
        "coverage": {
            "screenshots": len(records),
            "selected": selected,
            "reused": reused,
            "skipped_by_policy": skipped,
        },
        "frames": [
            {
                "id": record.id,
                "requested_time": record.requested_time,
                "actual_time": record.actual_time,
                "timecode": format_timecode(record.actual_time),
                "relative_path": record.relative_path,
                "status": record.ocr_status,
                "engine": record.ocr_engine,
                "skip_reason": record.ocr_skip_reason or None,
                "reused_from": record.ocr_reused_from or None,
                "selected_for_ocr": record.selected_for_ocr,
                "text": record.ocr_text,
            }
            for record in records
        ],
    }
    atomic_write_json(root / "ocr.json", payload)
    atomic_write_text(root / "onscreen.txt", format_onscreen_text(records))
