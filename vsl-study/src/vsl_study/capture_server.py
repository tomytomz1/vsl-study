"""Loopback HTTP service for the browser-tab recorder. No transcription here."""

from __future__ import annotations

import json
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from vsl_study.cache import atomic_write_json
from vsl_study.capture_meta import (
    build_capture_record,
    sanitize_source_url,
    should_process,
    utc_now,
)
from vsl_study.capture_store import CaptureError, CaptureRegistry, sha256_hex
from vsl_study.capture_validate import RecordingInvalid, validate_recording

EventCb = Callable[[str, dict[str, Any]], None]
MAX_JSON = 256_000
MAX_CHUNK = 16 * 1024 * 1024


def recorder_bytes(name: str) -> bytes:
    try:
        return files("vsl_study").joinpath("recorder", name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        path = Path(__file__).resolve().parent / "recorder" / name
        return path.read_bytes()


class CaptureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, token: str, registry: CaptureRegistry, on_event: EventCb | None) -> None:
        super().__init__(("127.0.0.1", 0), CaptureHandler)
        self.token = token
        self.registry = registry
        self.on_event = on_event
        self.cancelled = False
        self.started_at = utc_now()


class CaptureHandler(BaseHTTPRequestHandler):
    server: CaptureHTTPServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin") or ""
        host = f"http://127.0.0.1:{self.server.server_address[1]}"
        alt = f"http://localhost:{self.server.server_address[1]}"
        if not origin:
            return True
        return origin.rstrip("/") in {host, alt}

    def _token_ok(self) -> bool:
        header = self.headers.get("X-VSL-Token") or ""
        if header == self.server.token:
            return True
        auth = self.headers.get("Authorization") or ""
        if auth == f"Bearer {self.server.token}":
            return True
        query = parse_qs(urlparse(self.path).query)
        return (query.get("token") or [""])[0] == self.server.token

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self, limit: int) -> bytes:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise CaptureError("length", "Content-Length is required.")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise CaptureError("length", "Invalid Content-Length.") from exc
        if length < 0 or length > limit:
            raise CaptureError("too_large", "Request body is too large.")
        return self.rfile.read(length)

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        cb = self.server.on_event
        if cb:
            cb(kind, payload)

    def _error(self, exc: CaptureError) -> None:
        status = 400
        if exc.code in {"gap", "conflict", "finalized", "cancelled", "backpressure", "missing_final"}:
            status = 409
        elif exc.code == "too_large":
            status = 413
        elif exc.code == "not_found":
            status = 404
        elif exc.code == "unauthorized":
            status = 401
        self._json(status, {"error": exc.code, "message": exc.message})

    def do_GET(self) -> None:  # noqa: N802
        if not self._origin_ok():
            self._json(403, {"error": "forbidden_origin"})
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/health":
                self._json(200, {"ok": True, "cancelled": self.server.cancelled})
                return
            if path == "/api/session":
                if not self._token_ok():
                    raise CaptureError("unauthorized", "Missing capture token.")
                self._json(200, {"ok": True, "cancelled": self.server.cancelled})
                return
            if path in {"/recorder", "/"}:
                if not self._token_ok():
                    raise CaptureError("unauthorized", "Missing capture token.")
                self._bytes(200, recorder_bytes("index.html"), "text/html; charset=utf-8")
                return
            if path == "/recorder.js":
                self._bytes(200, recorder_bytes("recorder.js"), "application/javascript; charset=utf-8")
                return
            if path == "/recorder.css":
                self._bytes(200, recorder_bytes("recorder.css"), "text/css; charset=utf-8")
                return
            parts = [p for p in path.split("/") if p]
            if parts[:2] == ["api", "recordings"] and len(parts) == 3:
                if not self._token_ok():
                    raise CaptureError("unauthorized", "Missing capture token.")
                session = self.server.registry.get(parts[2])
                self._json(200, session.snapshot())
                return
            self._json(404, {"error": "not_found"})
        except CaptureError as exc:
            self._error(exc)
        except FileNotFoundError:
            self._json(500, {"error": "recorder_missing"})

    def do_POST(self) -> None:  # noqa: N802
        self._mutate()

    def do_PUT(self) -> None:  # noqa: N802
        self._mutate()

    def _mutate(self) -> None:
        if not self._origin_ok():
            self._json(403, {"error": "forbidden_origin"})
            return
        try:
            if not self._token_ok():
                raise CaptureError("unauthorized", "Missing capture token.")
            if self.server.cancelled:
                raise CaptureError("cancelled", "The desktop app is no longer capturing.")
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            if self.command == "POST" and parts == ["api", "recordings"]:
                self._create()
                return
            if self.command == "PUT" and len(parts) == 5 and parts[:2] == ["api", "recordings"] and parts[3] == "chunks":
                self._chunk(parts[2], parts[4])
                return
            if self.command == "POST" and len(parts) == 4 and parts[:2] == ["api", "recordings"] and parts[3] == "finalize":
                self._finalize(parts[2])
                return
            if self.command == "POST" and len(parts) == 4 and parts[:2] == ["api", "recordings"] and parts[3] == "cancel":
                self._cancel(parts[2])
                return
            self._json(404, {"error": "not_found"})
        except CaptureError as exc:
            self._error(exc)
        except json.JSONDecodeError:
            self._json(400, {"error": "json", "message": "Request was not valid JSON."})

    def _create(self) -> None:
        raw = self._read_body(MAX_JSON)
        body = json.loads(raw or b"{}")
        rec_id = uuid.uuid4().hex
        session = self.server.registry.create(rec_id)
        max_dur = body.get("max_duration_s")
        try:
            max_duration_s = float(max_dur) if max_dur not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            max_duration_s = None
        session.configure(
            title=str(body.get("title") or ""),
            source_url=sanitize_source_url(str(body.get("source_url") or "")),
            mime_type=str(body.get("mime_type") or ""),
            max_duration_s=max_duration_s,
        )
        self._emit("capture_created", {"id": rec_id})
        self._json(200, {"id": rec_id, "created_at": session.created_at})

    def _chunk(self, rec_id: str, seq_s: str) -> None:
        try:
            seq = int(seq_s)
        except ValueError as exc:
            raise CaptureError("bad_seq", "Chunk sequence must be an integer.") from exc
        data = self._read_body(MAX_CHUNK)
        session = self.server.registry.get(rec_id)
        checksum = (self.headers.get("X-Content-SHA256") or "").strip() or sha256_hex(data)
        last = (self.headers.get("X-Last-Chunk") or "").lower() in {"1", "true", "yes"}
        result = session.write_chunk(seq, data, checksum, last=last)
        self._json(200, result)

    def _cancel(self, rec_id: str) -> None:
        session = self.server.registry.get(rec_id)
        session.cancel("client_cancel")
        self._emit("capture_incomplete", {"id": rec_id, "stop_reason": "client_cancel"})
        self._json(200, {"ok": True})

    def _finalize(self, rec_id: str) -> None:
        session = self.server.registry.get(rec_id)
        body = json.loads(self._read_body(MAX_JSON) or b"{}")
        stop_reason = str(body.get("stop_reason") or "user_stop")
        capture_file = session.dir / "capture.json"
        if session.finalized and capture_file.exists():
            existing = json.loads(capture_file.read_text(encoding="utf-8"))
            process = should_process(
                str(existing.get("stop_reason") or stop_reason),
                bool(existing.get("complete")),
                bool(existing.get("audio_track")),
            )
            if process:
                session.try_begin_processing()
                self._emit("capture_ready", {"id": rec_id, "path": existing.get("recording_path"), "capture": existing, "duplicate": True})
            self._json(200, {"ok": True, "complete": existing.get("complete"), "process": process, "duplicate": True, "capture": existing})
            return
        if body.get("audio_track") is False:
            session.cancel("no_audio_track")
            message = "No audio track. Choose the tab again and turn on sharing that tab's sound."
            self._emit("capture_incomplete", {"id": rec_id, "stop_reason": "no_audio_track", "message": message})
            self._json(409, {"error": "no_audio_track", "message": message, "complete": False})
            return
        info = session.begin_finalize(stop_reason)
        stream = session.stream_path
        mime = str(body.get("mime_type") or session.mime_type or "video/webm")
        suffix = ".mp4" if "mp4" in mime.lower() else ".webm"
        dest = session.dir / f"recording{suffix}"
        dest.write_bytes(stream.read_bytes())
        problems: list[str] = []
        duration = None
        used_path = dest
        audio_ok = bool(body.get("audio_track"))
        try:
            checked = validate_recording(dest, require_audio=True)
            used_path = Path(checked["path"])
            duration = checked.get("duration_s")
            audio_ok = bool(checked.get("has_audio"))
            problems.extend(checked.get("problems") or [])
        except RecordingInvalid as exc:
            problems.append(str(exc))
            audio_ok = False
        complete = stop_reason in {"user_stop", "max_duration"} and audio_ok and not session.cancelled
        record = build_capture_record(
            recording_id=rec_id,
            source_url=session.source_url,
            title=session.title or str(body.get("title") or ""),
            mime_type=mime,
            audio_track=bool(body.get("audio_track")),
            audio_detected=bool(body.get("audio_detected")),
            stop_reason=stop_reason,
            complete=complete,
            media_duration_s=duration,
            started_at=session.created_at,
            ended_at=utc_now(),
            browser=body.get("browser") if isinstance(body.get("browser"), dict) else {},
            problems=problems,
            recording_path=str(used_path),
        )
        atomic_write_json(capture_file, record)
        process = should_process(stop_reason, complete, audio_ok)
        payload = {"id": rec_id, "path": str(used_path), "capture": record, "duplicate": bool(info.get("already"))}
        if process:
            session.try_begin_processing()
            self._emit("capture_ready", payload)
        else:
            self._emit("capture_incomplete", payload)
        self._json(200, {"ok": True, "complete": complete, "process": process, "duplicate": bool(info.get("already")), "path": str(used_path), "capture": record})


class CaptureServer:
    def __init__(self, root: Path, on_event: EventCb | None = None) -> None:
        self.root = root
        self.on_event = on_event
        self.token = secrets.token_urlsafe(32)
        self.registry = CaptureRegistry(root)
        self.httpd: CaptureHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if not self.httpd:
            raise RuntimeError("Capture server is not running.")
        return int(self.httpd.server_address[1])

    @property
    def recorder_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/recorder?token={self.token}"

    def alive(self) -> bool:
        return bool(self.httpd) and self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.alive():
            return
        self.httpd = CaptureHTTPServer(self.token, self.registry, self.on_event)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vsl-capture-http", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.cancelled = True
            self.registry.cancel_all("app_closed")
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=3)
        self.httpd = None
        self.thread = None
