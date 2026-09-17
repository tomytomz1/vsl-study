"""Timestamp helpers. All public times are seconds on the source playback timeline."""

from __future__ import annotations

import math
import re
from typing import Iterable

_HMS = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$"
)


def parse_timecode(value: str | float | int) -> float:
    """Parse seconds, or HH:MM:SS.mmm / MM:SS.mmm, into float seconds."""
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid timestamp: {value!r}")
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")
    try:
        seconds = float(text)
        if seconds < 0:
            raise ValueError(f"Invalid timestamp: {value!r}")
        return seconds
    except ValueError:
        pass
    match = _HMS.match(text)
    if not match:
        raise ValueError(f"Invalid timestamp: {value!r}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    secs = float(match.group("s"))
    if minutes >= 60 or secs >= 60:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + secs


def format_timecode(seconds: float, millis: bool = True) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_filename_time(seconds: float) -> str:
    """Windows-safe timestamp for filenames."""
    return format_timecode(seconds).replace(":", "-")


def clamp_time(value: float, start: float, end: float) -> float:
    if end < start:
        return start
    return min(max(value, start), end)


def windows_5min(duration_s: float) -> list[tuple[float, float, str]]:
    """Return (start, end, folder_name) covering [0, duration)."""
    if duration_s <= 0:
        return [(0.0, 0.0, "0000-0005")]
    windows: list[tuple[float, float, str]] = []
    start = 0.0
    while start < duration_s - 1e-9:
        end = min(start + 300.0, duration_s)
        start_m = int(start // 60)
        end_m = int(math.ceil(end / 60.0 - 1e-9)) if end < duration_s else int(math.ceil(duration_s / 60.0 - 1e-9))
        # Folder names use original-video minutes: 0000-0005, 0005-0010, ...
        folder_start = int(start // 300) * 5
        folder_end = folder_start + 5
        name = f"{folder_start:04d}-{folder_end:04d}"
        windows.append((start, end, name))
        start += 300.0
    return windows


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def unique_sorted(times: Iterable[float], *, digits: int = 3) -> list[float]:
    seen: set[int] = set()
    out: list[float] = []
    scale = 10 ** digits
    for t in sorted(times):
        key = int(round(t * scale))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
