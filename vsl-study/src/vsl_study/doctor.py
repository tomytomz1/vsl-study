"""Dependency checks before a long job."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass

from vsl_study.ocr import rapidocr_status, tesseract_status


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def collect_checks(*, include_streamlit: bool = True) -> list[Check]:
    checks: list[Check] = []
    py = sys.version_info
    py_ok = py.major == 3 and py.minor == 11
    checks.append(
        Check(
            "python",
            py_ok,
            f"{sys.executable} ({py.major}.{py.minor}.{py.micro}); 3.11 required",
        )
    )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    checks.append(Check("ffmpeg", bool(ffmpeg), ffmpeg or "not found on PATH"))
    checks.append(Check("ffprobe", bool(ffprobe), ffprobe or "not found on PATH"))

    try:
        import whisper  # noqa: F401

        checks.append(Check("openai-whisper", True, "import whisper ok"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("openai-whisper", False, f"{type(exc).__name__}: {exc}"))

    try:
        import torch

        cuda = torch.cuda.is_available()
        checks.append(
            Check(
                "torch",
                True,
                f"{torch.__version__}; cuda={'yes' if cuda else 'no (CPU fp32)'}",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("torch", False, str(exc)))

    try:
        import importlib.metadata

        checks.append(Check("scenedetect", True, importlib.metadata.version("scenedetect")))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("scenedetect", False, str(exc)))

    try:
        from PIL import Image  # noqa: F401

        checks.append(Check("pillow", True, "ok"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("pillow", False, str(exc)))

    if include_streamlit:
        try:
            import streamlit

            checks.append(Check("streamlit", True, getattr(streamlit, "__version__", "ok"), required=False))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("streamlit", False, str(exc), required=False))

    ok, detail = rapidocr_status()
    checks.append(Check("rapidocr", ok, detail + " (optional, default OCR)", required=False))
    ok, detail = tesseract_status()
    checks.append(Check("tesseract", ok, detail + " (optional fallback)", required=False))
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["VSL Study doctor", ""]
    for check in checks:
        mark = "OK" if check.ok else ("MISSING" if check.required else "optional-missing")
        lines.append(f"[{mark}] {check.name}: {check.detail}")
    return "\n".join(lines) + "\n"


def required_ok(checks: list[Check] | None = None) -> tuple[bool, list[Check]]:
    checks = checks or collect_checks()
    return all(c.ok for c in checks if c.required), checks
