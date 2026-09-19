"""Capture provenance helpers. The entered URL is user-supplied context, not proof."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

INTENTIONAL_STOPS = {"user_stop", "max_duration"}
PROCESSABLE_STOPS = INTENTIONAL_STOPS
CAPTURE_INPUT_TYPE = "browser_tab_recording"
PUBLIC_CAPTURE_FIELDS = (
    "input_type",
    "recording_id",
    "source_url_user_supplied",
    "title",
    "started_at",
    "ended_at",
    "media_duration_s",
    "browser",
    "mime_type",
    "audio_track",
    "audio_detected",
    "stop_reason",
    "complete",
    "problems",
    "recording_path",
    "timeline_note",
    "capture_profile",
    "capture_requested",
    "capture_observed",
    "capture_matches_profile",
    "capture_constraint_error",
    "capture_constraint_applied",
)
RECORDING_FILENAMES = {
    "recording.webm",
    "recording.mp4",
    "recording.fixed.webm",
    "recording.fixed.mp4",
    "recording.orig.webm",
    "recording.orig.mp4",
}
STANDARD_TIMELINE_NOTE = (
    "Timestamps are relative to this recording, not verified times in the original video. "
    "Pauses, buffering, lead-in, and seeking can make the two timelines differ. "
    "The URL is what the user typed; it is not proof that the selected tab matched."
)
MISMATCH_NOTE = (
    "A capture.json file was next to this video but it does not describe this recording. "
    "The file is treated as an ordinary imported video, not a verified browser-tab capture."
)
INVALID_NOTE = (
    "A capture.json file was next to this video but it is not valid VSL Study capture metadata. "
    "The file is treated as an ordinary imported video, not a verified browser-tab capture."
)


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
    capture_profile: str = "",
    capture_requested: dict[str, Any] | None = None,
    capture_observed: dict[str, Any] | None = None,
    capture_matches_profile: bool | None = None,
    capture_constraint_error: str = "",
    capture_constraint_applied: bool | None = None,
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
        "timeline_note": STANDARD_TIMELINE_NOTE,
        "capture_profile": (capture_profile or "").strip(),
        "capture_requested": dict(capture_requested or {}),
        "capture_observed": dict(capture_observed or {}),
        "capture_matches_profile": capture_matches_profile,
        "capture_constraint_error": (capture_constraint_error or "").strip()[:500],
        "capture_constraint_applied": capture_constraint_applied,
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


def is_recording_filename(path: str | Path) -> bool:
    return Path(path).name.casefold() in RECORDING_FILENAMES


def valid_recording_id(value: Any) -> bool:
    text = str(value or "")
    return bool(text) and text.isalnum() and len(text) <= 64


def _optional_json_bool(raw: dict[str, Any], key: str) -> bool | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if type(value) is bool:
        return value
    raise ValueError(f"{key} must be a JSON boolean")


def _optional_settings_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, item in list(value.items())[:24]:
        name = str(key)[:64]
        if isinstance(item, bool):
            cleaned[name] = item
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            cleaned[name] = item
        elif isinstance(item, str):
            cleaned[name] = item[:200]
    return cleaned


def _optional_duration(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError("media_duration_s must be a finite nonnegative number")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("media_duration_s must be a finite nonnegative number")
    return duration


def _problems_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("problems must be a list")
    problems: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            problems.append(item.strip()[:500])
    return problems


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def public_capture_record(raw: Any) -> dict[str, Any] | None:
    """Return portable capture provenance, or None if this is not a capture record.

    Drops tokens, session state, and unknown keys. Does not invent capture facts.
    """
    if not isinstance(raw, dict):
        return None
    rec_id = raw.get("recording_id")
    if not valid_recording_id(rec_id):
        return None
    input_type = str(raw.get("input_type") or "")
    if input_type and input_type != CAPTURE_INPUT_TYPE:
        return None
    browser_in = raw.get("browser") if isinstance(raw.get("browser"), dict) else {}
    browser: dict[str, str] = {}
    if isinstance(browser_in.get("userAgent"), str):
        browser["userAgent"] = browser_in["userAgent"][:1000]
    if isinstance(browser_in.get("vendor"), str):
        browser["vendor"] = browser_in["vendor"][:200]
    try:
        problems = _problems_list(raw.get("problems"))
        duration = _optional_duration(raw.get("media_duration_s"))
        audio_track = _optional_json_bool(raw, "audio_track")
        audio_detected = _optional_json_bool(raw, "audio_detected")
        complete = _optional_json_bool(raw, "complete")
        capture_matches_profile = _optional_json_bool(raw, "capture_matches_profile")
        capture_constraint_applied = _optional_json_bool(raw, "capture_constraint_applied")
        url = sanitize_source_url(str(raw.get("source_url_user_supplied") or raw.get("source_url") or ""))
    except (ValueError, OverflowError):
        return None
    timeline = raw.get("timeline_note")
    if not isinstance(timeline, str) or not timeline.strip():
        timeline = STANDARD_TIMELINE_NOTE
    record = {
        "input_type": CAPTURE_INPUT_TYPE,
        "recording_id": str(rec_id),
        "source_url_user_supplied": url,
        "title": str(raw.get("title") or "").strip()[:200],
        "started_at": str(raw.get("started_at") or "") or None,
        "ended_at": str(raw.get("ended_at") or "") or None,
        "media_duration_s": duration,
        "browser": browser,
        "mime_type": str(raw.get("mime_type") or ""),
        "audio_track": audio_track,
        "audio_detected": audio_detected,
        "stop_reason": str(raw.get("stop_reason") or ""),
        "complete": complete,
        "problems": problems,
        "recording_path": str(raw.get("recording_path") or ""),
        "timeline_note": timeline.strip(),
        "capture_profile": str(raw.get("capture_profile") or ""),
        "capture_requested": _optional_settings_dict(raw.get("capture_requested")),
        "capture_observed": _optional_settings_dict(raw.get("capture_observed")),
        "capture_matches_profile": capture_matches_profile,
        "capture_constraint_error": str(raw.get("capture_constraint_error") or "")[:500],
        "capture_constraint_applied": capture_constraint_applied,
    }
    return {key: record[key] for key in PUBLIC_CAPTURE_FIELDS}


def capture_associates_with_video(record: dict[str, Any], video: str | Path) -> bool:
    """True when this capture record describes the selected recording or its remux sibling."""
    if not record or not valid_recording_id(record.get("recording_id")):
        return False
    video_path = Path(video)
    try:
        video_path = video_path.resolve()
    except OSError:
        video_path = Path(video)
    recorded = str(record.get("recording_path") or "").strip()
    rec_id = str(record.get("recording_id"))

    if recorded:
        recorded_path = Path(recorded)
        try:
            resolved_recorded = recorded_path.expanduser().resolve()
        except OSError:
            resolved_recorded = recorded_path
        if _same_path(resolved_recorded, video_path):
            return True
        try:
            same_dir = resolved_recorded.parent == video_path.parent
        except OSError:
            same_dir = False
        if same_dir and is_recording_filename(resolved_recorded) and is_recording_filename(video_path):
            return True

    if video_path.parent.name == rec_id and is_recording_filename(video_path):
        return True
    return False


def load_capture_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return public_capture_record(data)


def resolve_capture_for_source(
    source: str | Path,
    provided: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return capture provenance for this video, plus notes when association fails.

    Ordinary imported videos yield (None, []). A nearby capture.json is used only
    when the record matches this file's path or recording id.
    """
    notes: list[str] = []
    video = Path(source)
    if provided is not None:
        cleaned = public_capture_record(provided)
        if cleaned and capture_associates_with_video(cleaned, video):
            return cleaned, notes
        if cleaned:
            notes.append(MISMATCH_NOTE)
            return None, notes
        notes.append(INVALID_NOTE)
        return None, notes

    candidate = video.parent / "capture.json"
    if not candidate.is_file():
        return None, notes
    cleaned = load_capture_json(candidate)
    if not cleaned:
        notes.append(INVALID_NOTE)
        return None, notes
    if not capture_associates_with_video(cleaned, video):
        notes.append(MISMATCH_NOTE)
        return None, notes
    return cleaned, notes


def write_portable_capture(job_root: str | Path, capture: dict[str, Any]) -> Path | None:
    cleaned = public_capture_record(capture)
    if not cleaned:
        return None
    from vsl_study.cache import atomic_write_json

    folder = Path(job_root) / "capture-session"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "capture.json"
    atomic_write_json(dest, cleaned)
    return dest


def clear_portable_capture(job_root: str | Path) -> None:
    """Remove rejected provenance; other failures must block a new export."""
    dest = Path(job_root) / "capture-session" / "capture.json"
    try:
        dest.unlink()
    except FileNotFoundError:
        return

