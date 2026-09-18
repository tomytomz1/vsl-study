"""Capture provenance helpers. The entered URL is user-supplied context, not proof."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

INTENTIONAL_STOPS = {"user_stop", "max_duration"}
PROCESSABLE_STOPS = INTENTIONAL_STOPS


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_source_url(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    if parts.scheme not in {"http", "https"}:
        return ""
    host = parts.hostname or ""
    if not host:
        return ""
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def is_http_url(raw: str | None) -> bool:
    return bool(sanitize_source_url(raw))


def build_capture_record(
    *,
    recording_id: str,
    source_url: str = "",
    title: str = "",
    mime_type: str = "",
    audio_track: bool = False,
    audio_detected: bool = False,
    stop_reason: str = "",
    complete: bool = False,
    media_duration_s: float | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    browser: dict[str, Any] | None = None,
    problems: list[str] | None = None,
    recording_path: str = "",
) -> dict[str, Any]:
    return {
        "input_type": "browser_tab_recording",
        "recording_id": recording_id,
        "source_url_user_supplied": sanitize_source_url(source_url),
        "title": (title or "").strip()[:200],
        "started_at": started_at,
        "ended_at": ended_at or utc_now(),
        "media_duration_s": media_duration_s,
        "browser": browser or {},
        "mime_type": mime_type,
        "audio_track": bool(audio_track),
        "audio_detected": bool(audio_detected),
        "stop_reason": stop_reason,
        "complete": bool(complete),
        "problems": list(problems or []),
        "recording_path": recording_path,
        "timeline_note": (
            "Timestamps are relative to this recording, not verified times in the original video. "
            "Pauses, buffering, lead-in, and seeking can make the two timelines differ. "
            "The URL is what the user typed; it is not proof that the selected tab matched."
        ),
    }


def should_process(stop_reason: str, complete: bool, audio_ok: bool) -> bool:
    return bool(complete and audio_ok and stop_reason in PROCESSABLE_STOPS)
