import hashlib
import json
import threading
import time
from pathlib import Path

from vsl_study.capture_meta import apply_desktop_capture_event, is_current_capture
from vsl_study.capture_server import CaptureServer
from vsl_study.capture_store import publish_recording


def _request(server: CaptureServer, method: str, path: str, body: bytes = b"", extra=None, token=None, generation=None):
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{server.port}{path}"
    headers = {"X-VSL-Token": token if token is not None else server.token}
    if generation is not None:
        headers["X-VSL-Generation"] = str(generation)
    if extra:
        headers.update(extra)
    req = urllib.request.Request(url, data=body if method in {"POST", "PUT"} else None, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        return exc.code, payload


def _live_generation(server: CaptureServer) -> int:
    if not server.page_expected:
        return server.note_expected_client()
    return int(server.httpd.page_generation)


def _create_and_write(server: CaptureServer, data: bytes = b"not-a-real-webm", last: bool = True, generation: int | None = None) -> str:
    generation = _live_generation(server) if generation is None else generation
    status, created = _request(
        server,
        "POST",
        "/api/recordings",
        body=b"{}",
        extra={"Content-Type": "application/json"},
        generation=generation,
    )
    assert status == 200, created
    rec_id = created["id"]
    digest = hashlib.sha256(data).hexdigest()
    status, _ack = _request(
        server,
        "PUT",
        f"/api/recordings/{rec_id}/chunks/0",
        body=data,
        extra={"X-Content-SHA256": digest, "X-Last-Chunk": "1" if last else "0"},
        generation=generation,
    )
    assert status == 200, _ack
    if not last:
        empty = b""
        status, _marker = _request(
            server,
            "PUT",
            f"/api/recordings/{rec_id}/chunks/1",
            body=empty,
            extra={"X-Content-SHA256": hashlib.sha256(empty).hexdigest(), "X-Last-Chunk": "1"},
            generation=generation,
        )
        assert status == 200
    return rec_id


def test_abandon_before_recording_starts(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), lease_seconds=0.2, lease_check_interval=30)
    server.start()
    try:
        generation = server.note_expected_client()
        assert generation == 1
        assert server.page_expected is True
        timed_out = server.check_lease(now=time.monotonic() + 10)
        assert timed_out is True
        assert server.page_expected is False
        assert any(k == "capture_abandoned" for k, _p in events)
        assert events[-1][1]["reason"] == "abandoned"
        assert events[-1][1]["generation"] == 1
    finally:
        server.stop()


def test_heartbeat_expiration_during_recording(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), lease_seconds=0.2, lease_check_interval=30)
    server.start()
    try:
        server.note_expected_client()
        rec_id = _create_and_write(server, last=True)
        session = server.registry.get(rec_id)
        assert session.bytes_written > 0
        assert session.cancelled is False
        status, _beat = _request(server, "POST", "/api/heartbeat", body=b"{}", extra={"Content-Type": "application/json"}, generation=_live_generation(server))
        assert status == 200
        expired = server.check_lease(now=time.monotonic() + 10)
        assert expired is True
        assert session.cancelled is True
        assert any(k == "capture_abandoned" for k, _p in events)
        assert not any(k == "capture_ready" for k, _p in events)
        assert (tmp_path / rec_id / "stream.bin").read_bytes() == b"not-a-real-webm"
    finally:
        server.stop()


def test_explicit_cancellation_restores_desktop_availability():
    state = {
        "capturing": True,
        "running": False,
        "session_id": "abc",
        "generation": 1,
        "processed": set(),
    }
    nxt = apply_desktop_capture_event(state, "capture_abandoned", {"reason": "desktop_cancel", "generation": 1})
    assert nxt["ignored"] is False
    assert nxt["capturing"] is False
    assert nxt["session_id"] is None
    assert not nxt["capturing"] and not nxt["running"]


def test_late_events_from_old_session_are_ignored():
    assert is_current_capture(active_id="b", active_generation=2, event_id="a", event_generation=1) is False
    state = {
        "capturing": True,
        "running": False,
        "session_id": "newid",
        "generation": 2,
        "processed": set(),
    }
    stale_ready = apply_desktop_capture_event(
        state,
        "capture_ready",
        {"id": "oldid", "generation": 1, "path": "x", "capture": {}},
    )
    assert stale_ready["ignored"] is True
    assert stale_ready["capturing"] is True
    assert stale_ready["process"] is False
    live = apply_desktop_capture_event(state, "capture_created", {"id": "newid", "generation": 2})
    assert live["ignored"] is False
    assert live["session_id"] == "newid"


def test_concurrent_finalize_emits_processing_once(tmp_path: Path, monkeypatch):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)))
    server.start()
    try:
        rec_id = _create_and_write(server, last=True)

        def fake_validate(path, require_audio=True):
            return {"path": str(path), "duration_s": 1.0, "has_audio": True, "problems": []}

        monkeypatch.setattr("vsl_study.capture_server.validate_recording", fake_validate)
        results = []

        def worker():
            status, payload = _request(
                server,
                "POST",
                f"/api/recordings/{rec_id}/finalize",
                body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
                extra={"Content-Type": "application/json"},
                generation=_live_generation(server),
            )
            results.append((status, payload))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        ready = [item for item in events if item[0] == "capture_ready"]
        assert len(ready) == 1
        assert all(status == 200 for status, _payload in results)
        assert server.registry.get(rec_id).processing_started is True
        assert server.registry.get(rec_id).try_begin_processing() is False
    finally:
        server.stop()


def test_finalize_does_not_read_whole_file(tmp_path: Path, monkeypatch):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)))
    server.start()
    try:
        rec_id = _create_and_write(server, data=b"x" * 50_000, last=True)
        original = Path.read_bytes

        def guarded(self):
            if self.name == "stream.bin":
                raise AssertionError("whole-file read of recording")
            return original(self)

        monkeypatch.setattr(Path, "read_bytes", guarded)
        status, payload = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
            extra={"Content-Type": "application/json"},
            generation=_live_generation(server),
        )
        assert status == 200
        assert payload["process"] is False
        dest = tmp_path / rec_id / "recording.webm"
        assert dest.exists()
        assert dest.stat().st_size == 50_000
    finally:
        server.stop()


def test_copy_failure_keeps_partial_and_reports_incomplete(tmp_path: Path):
    events = []

    def boom(_src, _dest, chunk_size=1024 * 1024):  # noqa: ARG001
        raise OSError(28, "No space left on device")

    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), publish=boom)
    server.start()
    try:
        rec_id = _create_and_write(server, last=True)
        stream = tmp_path / rec_id / "stream.bin"
        assert stream.read_bytes() == b"not-a-real-webm"
        status, payload = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
            extra={"Content-Type": "application/json"},
            generation=_live_generation(server),
        )
        assert status == 500
        assert payload["error"] == "storage_failed"
        assert payload["process"] is False
        assert stream.read_bytes() == b"not-a-real-webm"
        assert not (tmp_path / rec_id / "recording.webm").exists()
        assert any(k == "capture_incomplete" for k, _p in events)
        assert not any(k == "capture_ready" for k, _p in events)
    finally:
        server.stop()


def test_page_cancel_is_idempotent(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)))
    server.start()
    try:
        gen = server.note_expected_client()
        status, first = _request(
            server,
            "POST",
            "/api/page/cancel",
            body=b"{}",
            extra={"Content-Type": "application/json"},
            generation=gen,
        )
        assert status == 200
        assert first["emitted"] is True
        status, second = _request(
            server,
            "POST",
            "/api/page/cancel",
            body=b"{}",
            extra={"Content-Type": "application/json"},
            generation=gen,
        )
        assert status == 409
        assert second["error"] == "stale_generation"
        assert len([item for item in events if item[0] == "capture_abandoned"]) == 1
    finally:
        server.stop()


def test_publish_helper_is_used_by_server(tmp_path: Path):
    copied = []

    def spy(src, dest, chunk_size=1024 * 1024):
        copied.append((src, dest, chunk_size))
        publish_recording(src, dest, chunk_size=chunk_size)

    server = CaptureServer(tmp_path, publish=spy)
    server.start()
    try:
        rec_id = _create_and_write(server, last=True)
        _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
            extra={"Content-Type": "application/json"},
            generation=_live_generation(server),
        )
        assert copied
    finally:
        server.stop()
