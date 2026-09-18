import hashlib
import json
import threading
from pathlib import Path

from vsl_study.capture_meta import apply_desktop_capture_event
from vsl_study.capture_server import CaptureServer, STALE_PAGE_MESSAGE
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


def _create_and_write(server: CaptureServer, generation: int, data: bytes = b"not-a-real-webm") -> str:
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
    status, ack = _request(
        server,
        "PUT",
        f"/api/recordings/{rec_id}/chunks/0",
        body=data,
        extra={"X-Content-SHA256": digest, "X-Last-Chunk": "1"},
        generation=generation,
    )
    assert status == 200, ack
    return rec_id


def test_recorder_url_binds_generation(tmp_path: Path):
    server = CaptureServer(tmp_path)
    server.start()
    try:
        gen = server.note_expected_client()
        assert f"g={gen}" in server.recorder_url
        assert "token=" in server.recorder_url
    finally:
        server.stop()


def test_old_page_cannot_cancel_new_recording(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), lease_check_interval=30)
    server.start()
    try:
        gen_a = server.note_expected_client()
        rec_a = _create_and_write(server, gen_a)
        assert server.abandon_page("desktop_cancel", generation=gen_a) is True
        assert server.registry.get(rec_a).cancelled is True
        gen_b = server.note_expected_client()
        rec_b = _create_and_write(server, gen_b, data=b"bbbb")
        status, payload = _request(
            server,
            "POST",
            "/api/page/cancel",
            body=json.dumps({"generation": gen_a}).encode(),
            extra={"Content-Type": "application/json"},
            generation=gen_a,
        )
        assert status == 409
        assert payload["error"] == "stale_generation"
        assert payload["stale"] is True
        assert STALE_PAGE_MESSAGE in payload["message"]
        assert server.registry.get(rec_b).cancelled is False
        assert server.page_expected is True
        assert int(server.httpd.page_generation) == gen_b
        assert not any(
            kind == "capture_abandoned" and item.get("generation") == gen_b for kind, item in events
        )
    finally:
        server.stop()


def test_old_heartbeat_and_session_poll_do_not_extend_new_lease(tmp_path: Path):
    server = CaptureServer(tmp_path, lease_check_interval=30)
    server.start()
    try:
        gen_a = server.note_expected_client()
        _create_and_write(server, gen_a)
        server.abandon_page("desktop_cancel", generation=gen_a)
        gen_b = server.note_expected_client()
        _create_and_write(server, gen_b, data=b"bbbb")
        last_beat = server.httpd.last_beat
        status, payload = _request(
            server,
            "POST",
            "/api/heartbeat",
            body=json.dumps({"generation": gen_a}).encode(),
            extra={"Content-Type": "application/json"},
            generation=gen_a,
        )
        assert status == 409
        assert payload["error"] == "stale_generation"
        assert server.httpd.last_beat == last_beat
        status, session = _request(server, "GET", "/api/session", generation=gen_a)
        assert status == 200
        assert session["stale"] is True
        assert session.get("current_generation") == gen_b
        assert server.httpd.last_beat == last_beat
        status, live = _request(server, "GET", "/api/session", generation=gen_b)
        assert status == 200
        assert live["stale"] is False
        assert server.httpd.last_beat >= last_beat
    finally:
        server.stop()


def test_old_page_cannot_create_a_recording(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)))
    server.start()
    try:
        gen_a = server.note_expected_client()
        server.abandon_page("desktop_cancel", generation=gen_a)
        gen_b = server.note_expected_client()
        rec_b = _create_and_write(server, gen_b, data=b"bbbb")
        before = set(server.registry._sessions)
        status, payload = _request(
            server,
            "POST",
            "/api/recordings",
            body=json.dumps({"title": "stale", "generation": gen_a}).encode(),
            extra={"Content-Type": "application/json"},
            generation=gen_a,
        )
        assert status == 409
        assert payload["error"] == "stale_generation"
        assert set(server.registry._sessions) == before
        assert rec_b in server.registry._sessions
        assert not any(kind == "capture_created" and item.get("generation") == gen_a for kind, item in events)
        desktop = {
            "capturing": True,
            "running": False,
            "session_id": rec_b,
            "generation": gen_b,
            "processed": set(),
        }
        created = apply_desktop_capture_event(desktop, "capture_created", {"id": "should-not-exist", "generation": gen_a})
        assert created["ignored"] is True
        assert created["session_id"] == rec_b
    finally:
        server.stop()


def test_recording_ownership_is_enforced(tmp_path: Path):
    server = CaptureServer(tmp_path)
    server.start()
    try:
        gen_a = server.note_expected_client()
        rec_a = _create_and_write(server, gen_a, data=b"AAAA")
        original = (tmp_path / rec_a / "stream.bin").read_bytes()
        server.abandon_page("desktop_cancel", generation=gen_a)
        gen_b = server.note_expected_client()
        rec_b = _create_and_write(server, gen_b, data=b"BBBB")
        b_session = server.registry.get(rec_b)
        assert b_session.cancelled is False
        assert b_session.processing_started is False
        status, payload = _request(
            server,
            "PUT",
            f"/api/recordings/{rec_b}/chunks/1",
            body=b"XXXX",
            extra={"X-Content-SHA256": hashlib.sha256(b"XXXX").hexdigest()},
            generation=gen_a,
        )
        assert status == 409
        assert payload["error"] == "stale_generation"
        assert b_session.bytes_written == 4
        assert (tmp_path / rec_b / "stream.bin").read_bytes() == b"BBBB"
        status, payload = _request(
            server,
            "POST",
            f"/api/recordings/{rec_b}/cancel",
            body=b"{}",
            extra={"Content-Type": "application/json"},
            generation=gen_a,
        )
        assert status == 409
        assert b_session.cancelled is False
        status, payload = _request(
            server,
            "POST",
            f"/api/recordings/{rec_b}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True}).encode(),
            extra={"Content-Type": "application/json"},
            generation=gen_a,
        )
        assert status == 409
        assert b_session.processing_started is False
        status, payload = _request(
            server,
            "PUT",
            f"/api/recordings/{rec_a}/chunks/1",
            body=b"YYYY",
            extra={"X-Content-SHA256": hashlib.sha256(b"YYYY").hexdigest()},
            generation=gen_b,
        )
        assert status == 409
        assert (tmp_path / rec_a / "stream.bin").read_bytes() == original
    finally:
        server.stop()


def test_delayed_operation_keeps_original_generation(tmp_path: Path):
    events = []
    started = threading.Event()
    release = threading.Event()

    def slow_publish(src, dest, chunk_size=1024 * 1024):
        started.set()
        assert release.wait(3)
        publish_recording(src, dest, chunk_size=chunk_size)

    server = CaptureServer(
        tmp_path,
        on_event=lambda k, p: events.append((k, p)),
        publish=slow_publish,
        lease_check_interval=30,
    )
    server.start()
    try:
        gen_a = server.note_expected_client()
        rec_a = _create_and_write(server, gen_a)
        result = {}

        def finalize_a():
            result["status"], result["payload"] = _request(
                server,
                "POST",
                f"/api/recordings/{rec_a}/finalize",
                body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
                extra={"Content-Type": "application/json"},
                generation=gen_a,
            )

        worker = threading.Thread(target=finalize_a)
        worker.start()
        assert started.wait(3)
        gen_b = server.note_expected_client()
        rec_b = _create_and_write(server, gen_b, data=b"BBBB")
        release.set()
        worker.join(5)
        assert result["status"] == 200
        a_events = [item for kind, item in events if item.get("id") == rec_a]
        assert a_events
        assert all(item.get("generation") == gen_a for item in a_events)
        assert server.registry.get(rec_b).cancelled is False
        desktop = {
            "capturing": True,
            "running": False,
            "session_id": rec_b,
            "generation": gen_b,
            "processed": set(),
        }
        for item in a_events:
            kind = "capture_ready" if item.get("path") and item.get("capture", {}).get("complete") else "capture_incomplete"
            if "stop_reason" in item or item.get("capture"):
                nxt = apply_desktop_capture_event(desktop, kind, item)
                assert nxt["ignored"] is True
                assert nxt["capturing"] is True
                assert nxt["session_id"] == rec_b
    finally:
        release.set()
        server.stop()


def test_cancel_and_new_session_are_race_safe(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), lease_check_interval=30)
    server.start()
    try:
        gen_a = server.note_expected_client()
        rec_a = _create_and_write(server, gen_a)
        entered = threading.Event()
        release = threading.Event()

        def pause(where: str) -> None:
            if where == "abandon_locked":
                entered.set()
                assert release.wait(3)

        server.httpd._test_pause = pause
        cancel_result = {}

        def cancel_a():
            cancel_result["status"], cancel_result["payload"] = _request(
                server,
                "POST",
                "/api/page/cancel",
                body=json.dumps({"generation": gen_a}).encode(),
                extra={"Content-Type": "application/json"},
                generation=gen_a,
            )

        cancel_thread = threading.Thread(target=cancel_a)
        cancel_thread.start()
        assert entered.wait(3)
        gen_b_holder: dict[str, int] = {}

        def open_b():
            gen_b_holder["g"] = server.note_expected_client()

        open_thread = threading.Thread(target=open_b)
        open_thread.start()
        release.set()
        cancel_thread.join(5)
        open_thread.join(5)
        gen_b = gen_b_holder["g"]
        assert gen_b != gen_a
        rec_b = _create_and_write(server, gen_b, data=b"BBBB")
        assert server.registry.get(rec_a).cancelled is True
        assert server.registry.get(rec_b).cancelled is False
        assert int(server.httpd.page_generation) == gen_b
        assert not any(kind == "capture_abandoned" and item.get("generation") == gen_b for kind, item in events)
        assert cancel_result["status"] in {200, 409}
    finally:
        server.stop()


def test_current_generation_can_still_capture_and_recover(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)), lease_check_interval=30)
    server.start()
    try:
        gen = server.note_expected_client()
        rec_id = _create_and_write(server, gen)
        status, beat = _request(
            server,
            "POST",
            "/api/heartbeat",
            body=b"{}",
            extra={"Content-Type": "application/json"},
            generation=gen,
        )
        assert status == 200
        status, fin = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
            extra={"Content-Type": "application/json"},
            generation=gen,
        )
        assert status == 200
        assert fin["process"] is False
        assert any(kind == "capture_incomplete" and item.get("generation") == gen for kind, item in events)
        gen2 = server.note_expected_client()
        rec2 = _create_and_write(server, gen2, data=b"next")
        assert server.abandon_page("desktop_cancel", generation=gen2) is True
        assert server.registry.get(rec2).cancelled is True
        desktop = {
            "capturing": True,
            "running": False,
            "session_id": rec2,
            "generation": gen2,
            "processed": set(),
        }
        nxt = apply_desktop_capture_event(desktop, "capture_abandoned", {"reason": "desktop_cancel", "generation": gen2})
        assert nxt["capturing"] is False
        assert nxt["ignored"] is False
    finally:
        server.stop()
