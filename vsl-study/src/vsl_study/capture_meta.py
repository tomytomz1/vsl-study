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


def is_current_capture(
    *,
    active_id: str | None,
    active_generation: int | None,
    event_id: str | None = None,
    event_generation: int | None = None,
) -> bool:
    """True when a capture event belongs to the desktop's live recorder session."""
    if event_generation is not None and active_generation is not None:
        if int(event_generation) != int(active_generation):
            return False
    if event_id and active_id and str(event_id) != str(active_id):
        return False
    return True


def apply_desktop_capture_event(state: dict[str, Any], kind: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Pure capture-gate update for the desktop. Widget code stays on the Tk thread."""
    payload = data or {}
    rec_id = str(payload.get("id") or "")
    gen = payload.get("generation")
    event_generation = int(gen) if gen is not None else None
    out = dict(state)
    out["ignored"] = False
    out["process"] = False
    processed = set(state.get("processed") or [])
    out["processed"] = processed

    if not is_current_capture(
        active_id=state.get("session_id"),
        active_generation=state.get("generation"),
        event_id=rec_id or None,
        event_generation=event_generation,
    ):
        out["ignored"] = True
        return out

    if kind == "capture_created":
        if rec_id:
            out["session_id"] = rec_id
        out["capturing"] = True
        return out

    if kind == "capture_abandoned":
        if not state.get("capturing"):
            out["ignored"] = True
            return out
        out["capturing"] = False
        out["session_id"] = None
        out["status"] = "abandoned"
        return out

    if kind in {"capture_ready", "capture_incomplete"}:
        if not state.get("capturing"):
            out["ignored"] = True
            return out
        if rec_id and state.get("session_id") and rec_id != state.get("session_id"):
            out["ignored"] = True
            return out

    if kind == "capture_incomplete":
        out["capturing"] = False
        out["session_id"] = None
        out["status"] = "incomplete"
        return out

    if kind == "capture_ready":
        if rec_id and rec_id in processed:
            out["ignored"] = True
            return out
        if rec_id:
            processed.add(rec_id)
        out["processed"] = processed
        out["capturing"] = False
        out["process"] = True
        out["status"] = "ready"
        return out

    return out
