"""Loopback HTTP service for the browser-tab recorder. No transcription here."""

from __future__ import annotations

import json
import secrets
import threading
import time
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
from vsl_study.capture_store import CaptureError, CaptureRegistry, publish_recording, sha256_hex
from vsl_study.capture_validate import RecordingInvalid, validate_recording

EventCb = Callable[[str, dict[str, Any]], None]
MAX_JSON = 256_000
MAX_CHUNK = 16 * 1024 * 1024
# Chrome can throttle background timers to about once a minute. A 180s lease
# avoids cancelling an ordinary recording while the user watches another tab.
DEFAULT_LEASE_SECONDS = 180.0
DEFAULT_LEASE_CHECK_INTERVAL = 1.0
STALE_PAGE_MESSAGE = "This recording session has ended. Open a new recorder from VSL Study."


def recorder_bytes(name: str) -> bytes:
    try:
        return files("vsl_study").joinpath("recorder", name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        path = Path(__file__).resolve().parent / "recorder" / name
        return path.read_bytes()


class CaptureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        token: str,
        registry: CaptureRegistry,
        on_event: EventCb | None,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        lease_check_interval: float = DEFAULT_LEASE_CHECK_INTERVAL,
        publish: Callable[[Path, Path], None] = publish_recording,
    ) -> None:
        super().__init__(("127.0.0.1", 0), CaptureHandler)
        self.token = token
        self.registry = registry
        self.on_event = on_event
        self.cancelled = False
        self.started_at = utc_now()
        self.lease_seconds = float(lease_seconds)
        self.lease_check_interval = float(lease_check_interval)
        self.publish = publish
        self.life_lock = threading.RLock()
        self.last_beat = time.monotonic()
        self.page_expected = False
        self.page_generation = 0
        self.abandon_emitted = False
        self._test_pause: Callable[[str], None] | None = None

    def beat(self) -> None:
        with self.life_lock:
            self.last_beat = time.monotonic()

    def _stale(self) -> CaptureError:
        return CaptureError("stale_generation", STALE_PAGE_MESSAGE)

    def require_live_generation(self, generation: int | None, *, beat: bool = False) -> int:
        if generation is None:
            raise self._stale()
        with self.life_lock:
            if self._test_pause:
                self._test_pause("require_live")
            if self.cancelled:
                raise CaptureError("cancelled", "The desktop app is no longer capturing.")
            if not self.page_expected or int(generation) != int(self.page_generation):
                raise self._stale()
            if beat:
                self.last_beat = time.monotonic()
            return int(generation)

    def note_expected_client(self) -> int:
        prev: int | None = None
        with self.life_lock:
            if self._test_pause:
                self._test_pause("note_locked")
            if self.page_expected and self.page_generation:
                prev = int(self.page_generation)
            self.page_generation += 1
            self.page_expected = True
            self.abandon_emitted = False
            self.last_beat = time.monotonic()
            new_gen = int(self.page_generation)
        if prev is not None:
            self.registry.cancel_generation(prev, "superseded")
            cb = self.on_event
            if cb:
                cb("capture_abandoned", {"reason": "superseded", "generation": prev})
        return new_gen

    def abandon_page(self, reason: str, generation: int | None = None) -> bool:
        with self.life_lock:
            if self._test_pause:
                self._test_pause("abandon_locked")
            if generation is None:
                return False
            if int(generation) != int(self.page_generation):
                return False
            if self.abandon_emitted and not self.page_expected:
                return False
            if not self.page_expected:
                return False
            target = int(self.page_generation)
            self.page_expected = False
            self.abandon_emitted = True
        self.registry.cancel_generation(target, reason)
        cb = self.on_event
        if cb:
            cb("capture_abandoned", {"reason": reason, "generation": target})
        return True

    def check_lease(self, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else now
        with self.life_lock:
            if not self.page_expected or self.cancelled:
                return False
            if stamp - self.last_beat <= self.lease_seconds:
                return False
            generation = int(self.page_generation)
        return self.abandon_page("abandoned", generation=generation)


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

    def _emit(self, kind: str, payload: dict[str, Any], *, generation: int) -> None:
        body = dict(payload)
        body["generation"] = int(generation)
        cb = self.server.on_event
        if cb:
            cb(kind, body)

    def _error(self, exc: CaptureError) -> None:
        status = 400
        if exc.code in {
            "gap",
            "conflict",
            "finalized",
            "cancelled",
            "backpressure",
            "missing_final",
            "stale_generation",
        }:
            status = 409
        elif exc.code == "too_large":
            status = 413
        elif exc.code == "not_found":
            status = 404
        elif exc.code == "unauthorized":
            status = 401
        elif exc.code == "storage_failed":
            status = 500
        payload: dict[str, Any] = {"error": exc.code, "message": exc.message, "complete": False, "process": False}
        if exc.code == "stale_generation":
            payload["stale"] = True
        self._json(status, payload)

    def _client_generation(self, body: dict[str, Any] | None = None) -> int | None:
        header = (self.headers.get("X-VSL-Generation") or "").strip()
        raw = header
        if not raw:
            query = parse_qs(urlparse(self.path).query)
            raw = (query.get("g") or [""])[0]
        if not raw and body is not None and body.get("generation") is not None:
            raw = str(body.get("generation"))
        if not raw:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise CaptureError("stale_generation", STALE_PAGE_MESSAGE) from exc
        if value < 1:
            raise CaptureError("stale_generation", STALE_PAGE_MESSAGE)
        return value

    def _require_live(self, body: dict[str, Any] | None = None, *, beat: bool = False) -> int:
        return self.server.require_live_generation(self._client_generation(body), beat=beat)

    def _require_session(self, rec_id: str, body: dict[str, Any] | None = None, *, beat: bool = False):
        generation = self._require_live(body, beat=False)
        session = self.server.registry.get(rec_id)
        if session.generation is None or int(session.generation) != int(generation):
            raise CaptureError("stale_generation", STALE_PAGE_MESSAGE)
        if beat:
            self.server.beat()
        return session, generation

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
                generation = self._client_generation()
                stale = True
                try:
                    live = self.server.require_live_generation(generation, beat=True)
                    stale = False
                except CaptureError as exc:
                    if exc.code not in {"stale_generation", "cancelled"}:
                        raise
                    live = generation
                self._json(
                    200,
                    {
                        "ok": True,
                        "cancelled": self.server.cancelled,
                        "page_expected": self.server.page_expected and not stale,
                        "lease_seconds": self.server.lease_seconds,
                        "generation": generation,
                        "current_generation": self.server.page_generation,
                        "stale": stale or self.server.cancelled,
                        "message": STALE_PAGE_MESSAGE if stale or self.server.cancelled else "",
                    },
                )
                return
            if path in {"/recorder", "/"}:
                if not self._token_ok():
                    raise CaptureError("unauthorized", "Missing capture token.")
                generation = self._client_generation()
                if generation is not None:
                    try:
                        self.server.require_live_generation(generation, beat=True)
                    except CaptureError as exc:
                        if exc.code not in {"stale_generation", "cancelled"}:
                            raise
                self._bytes(200, recorder_bytes("index.html"), "text/html; charset=utf-8")
                return
            if path == "/recorder.js":
                self._bytes(200, recorder_bytes("recorder.js"), "application/javascript; charset=utf-8")
                return
            if path == "/recorder-core.js":
                self._bytes(200, recorder_bytes("recorder-core.js"), "application/javascript; charset=utf-8")
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
            if self.command == "POST" and parts == ["api", "heartbeat"]:
                self._heartbeat()
                return
            if self.command == "POST" and parts == ["api", "page", "cancel"]:
                self._cancel_page()
                return
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

    def _heartbeat(self) -> None:
        raw = self._read_body(MAX_JSON)
        body = json.loads(raw or b"{}") if raw else {}
        if not isinstance(body, dict):
            body = {}
        self._require_live(body, beat=True)
        self._json(200, {"ok": True, "lease_seconds": self.server.lease_seconds})

    def _cancel_page(self) -> None:
        raw = self._read_body(MAX_JSON)
        body = json.loads(raw or b"{}") if raw else {}
        if not isinstance(body, dict):
            body = {}
        generation = self._client_generation(body)
        if generation is None:
            raise CaptureError("stale_generation", STALE_PAGE_MESSAGE)
        emitted = self.server.abandon_page("client_cancel", generation=generation)
        if not emitted:
            raise CaptureError("stale_generation", STALE_PAGE_MESSAGE)
        self._json(200, {"ok": True, "emitted": True, "generation": generation})

    def _create(self) -> None:
        raw = self._read_body(MAX_JSON)
        body = json.loads(raw or b"{}")
        if not isinstance(body, dict):
            body = {}
        generation = self._require_live(body, beat=True)
        rec_id = uuid.uuid4().hex
        session = self.server.registry.create(rec_id, generation=generation)
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
        self._emit("capture_created", {"id": rec_id}, generation=generation)
        self._json(200, {"id": rec_id, "created_at": session.created_at, "generation": generation})

    def _chunk(self, rec_id: str, seq_s: str) -> None:
        try:
            seq = int(seq_s)
        except ValueError as exc:
            raise CaptureError("bad_seq", "Chunk sequence must be an integer.") from exc
        data = self._read_body(MAX_CHUNK)
        session, generation = self._require_session(rec_id, beat=True)
        checksum = (self.headers.get("X-Content-SHA256") or "").strip() or sha256_hex(data)
        last = (self.headers.get("X-Last-Chunk") or "").lower() in {"1", "true", "yes"}
        result = session.write_chunk(seq, data, checksum, last=last)
        result["generation"] = generation
        self._json(200, result)

    def _cancel(self, rec_id: str) -> None:
        raw = self._read_body(MAX_JSON)
        body = json.loads(raw or b"{}") if raw else {}
        if not isinstance(body, dict):
            body = {}
        session, generation = self._require_session(rec_id, body)
        session.cancel("client_cancel")
        if session.try_notify_incomplete():
            self._emit("capture_incomplete", {"id": rec_id, "stop_reason": "client_cancel"}, generation=generation)
        self._json(200, {"ok": True, "generation": generation})

    def _finalize(self, rec_id: str) -> None:
        body = json.loads(self._read_body(MAX_JSON) or b"{}")
        if not isinstance(body, dict):
            body = {}
        session, generation = self._require_session(rec_id, body)
        stop_reason = str(body.get("stop_reason") or "user_stop")
        capture_file = session.dir / "capture.json"
        with session.assemble_lock:
            if session.finalized and capture_file.exists():
                existing = json.loads(capture_file.read_text(encoding="utf-8"))
                process = should_process(
                    str(existing.get("stop_reason") or stop_reason),
                    bool(existing.get("complete")),
                    bool(existing.get("audio_track")),
                )
                ready = False
                if process and session.try_begin_processing():
                    ready = True
                    self._emit(
                        "capture_ready",
                        {
                            "id": rec_id,
                            "path": existing.get("recording_path"),
                            "capture": existing,
                            "duplicate": True,
                        },
                        generation=generation,
                    )
                self._json(
                    200,
                    {
                        "ok": True,
                        "complete": existing.get("complete"),
                        "process": bool(process and (ready or session.processing_started)),
                        "duplicate": True,
                        "capture": existing,
                    },
                )
                return
            if body.get("audio_track") is False:
                session.cancel("no_audio_track")
                message = "No audio track. Choose the tab again and turn on sharing that tab's sound."
                if session.try_notify_incomplete():
                    self._emit(
                        "capture_incomplete",
                        {"id": rec_id, "stop_reason": "no_audio_track", "message": message},
                        generation=generation,
                    )
                self._json(409, {"error": "no_audio_track", "message": message, "complete": False, "process": False})
                return
            info = session.begin_finalize(stop_reason)
            stream = session.stream_path
            mime = str(body.get("mime_type") or session.mime_type or "video/webm")
            suffix = ".mp4" if "mp4" in mime.lower() else ".webm"
            dest = session.dir / f"recording{suffix}"
            if not dest.exists():
                try:
                    self.server.publish(stream, dest)
                except OSError as exc:
                    message = "Could not finish saving the recording. Partial media already on disk was kept."
                    if session.try_notify_incomplete():
                        self._emit(
                            "capture_incomplete",
                            {"id": rec_id, "stop_reason": "storage_failed", "message": message},
                            generation=generation,
                        )
                    self._json(
                        500,
                        {
                            "error": "storage_failed",
                            "message": f"{message} {exc}",
                            "complete": False,
                            "process": False,
                        },
                    )
                    return
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
                capture_profile=str(body.get("capture_profile") or ""),
                capture_requested=body.get("capture_requested") if isinstance(body.get("capture_requested"), dict) else {},
                capture_observed=body.get("capture_observed") if isinstance(body.get("capture_observed"), dict) else {},
                capture_matches_profile=body.get("capture_matches_profile") if isinstance(body.get("capture_matches_profile"), bool) else None,
                capture_constraint_error=str(body.get("capture_constraint_error") or ""),
                capture_constraint_applied=body.get("capture_constraint_applied") if isinstance(body.get("capture_constraint_applied"), bool) else None,
            )
            atomic_write_json(capture_file, record)
            process = should_process(stop_reason, complete, audio_ok)
            payload = {"id": rec_id, "path": str(used_path), "capture": record, "duplicate": bool(info.get("already"))}
            emit_generation = int(session.generation) if session.generation is not None else generation
            if process:
                if session.try_begin_processing():
                    self._emit("capture_ready", payload, generation=emit_generation)
                else:
                    process = False
            elif session.try_notify_incomplete():
                self._emit("capture_incomplete", payload, generation=emit_generation)
            self._json(
                200,
                {
                    "ok": True,
                    "complete": complete,
                    "process": process,
                    "duplicate": bool(info.get("already")),
                    "path": str(used_path),
                    "capture": record,
                },
            )


class CaptureServer:
    def __init__(
        self,
        root: Path,
        on_event: EventCb | None = None,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        lease_check_interval: float = DEFAULT_LEASE_CHECK_INTERVAL,
        publish: Callable[[Path, Path], None] | None = None,
    ) -> None:
        self.root = root
        self.on_event = on_event
        self.token = secrets.token_urlsafe(32)
        self.registry = CaptureRegistry(root)
        self.lease_seconds = lease_seconds
        self.lease_check_interval = lease_check_interval
        self.publish = publish or publish_recording
        self.httpd: CaptureHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self._watch_alive = False
        self._watch_thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if not self.httpd:
            raise RuntimeError("Capture server is not running.")
        return int(self.httpd.server_address[1])

    @property
    def recorder_url(self) -> str:
        generation = int(self.httpd.page_generation) if self.httpd else 0
        return f"http://127.0.0.1:{self.port}/recorder?token={self.token}&g={generation}"

    @property
    def page_expected(self) -> bool:
        return bool(self.httpd and self.httpd.page_expected)

    def alive(self) -> bool:
        return bool(self.httpd) and self.thread is not None and self.thread.is_alive()

    def note_expected_client(self) -> int:
        if not self.httpd:
            raise RuntimeError("Capture server is not running.")
        return self.httpd.note_expected_client()

    def beat(self) -> None:
        if self.httpd:
            self.httpd.beat()

    def check_lease(self, now: float | None = None) -> bool:
        if not self.httpd:
            return False
        return self.httpd.check_lease(now)

    def abandon_page(self, reason: str, generation: int | None = None) -> bool:
        if not self.httpd:
            return False
        return self.httpd.abandon_page(reason, generation=generation)

    def start(self) -> None:
        if self.alive():
            return
        self.httpd = CaptureHTTPServer(
            self.token,
            self.registry,
            self.on_event,
            lease_seconds=self.lease_seconds,
            lease_check_interval=self.lease_check_interval,
            publish=self.publish,
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vsl-capture-http", daemon=True)
        self.thread.start()
        self._watch_alive = True
        self._watch_thread = threading.Thread(target=self._watch_lease, name="vsl-capture-lease", daemon=True)
        self._watch_thread.start()

    def _watch_lease(self) -> None:
        while self._watch_alive:
            try:
                self.check_lease()
            except Exception:
                pass
            interval = self.lease_check_interval if self.httpd else 0.2
            time.sleep(max(0.05, interval))

    def stop(self) -> None:
        self._watch_alive = False
        if self.httpd:
            self.httpd.cancelled = True
            self.httpd.page_expected = False
            self.registry.cancel_all("app_closed")
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=3)
        if self._watch_thread:
            self._watch_thread.join(timeout=3)
        self.httpd = None
        self.thread = None
        self._watch_thread = None
