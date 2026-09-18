import json
from pathlib import Path

from vsl_study.capture_server import CaptureServer, recorder_bytes


def test_recorder_package_files_exist():
    html = recorder_bytes("index.html").decode("utf-8")
    js = recorder_bytes("recorder.js").decode("utf-8")
    css = recorder_bytes("recorder.css").decode("utf-8")
    assert "Choose tab and audio" in html
    assert "Stop and process" in html
    assert "Cancel recording" in html
    assert "getDisplayMedia" in js
    assert "audio" in js
    core = recorder_bytes("recorder-core.js").decode("utf-8")
    assert "CapturePipeline" in core
    assert "--accent" in css


def _request(server: CaptureServer, method: str, path: str, body: bytes = b"", extra=None, token=None):
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{server.port}{path}"
    headers = {"X-VSL-Token": token if token is not None else server.token}
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


def test_token_and_origin_and_no_audio(tmp_path: Path):
    events = []
    server = CaptureServer(tmp_path, on_event=lambda k, p: events.append((k, p)))
    server.start()
    try:
        status, _ = _request(server, "GET", "/api/session", token="nope")
        assert status == 401
        status, payload = _request(server, "GET", "/health", token="nope")
        assert status == 200
        assert payload["ok"] is True
        status, created = _request(
            server,
            "POST",
            "/api/recordings",
            body=json.dumps({"title": "demo"}).encode(),
            extra={"Content-Type": "application/json"},
        )
        assert status == 200
        rec_id = created["id"]
        status, fail = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": False}).encode(),
            extra={"Content-Type": "application/json"},
        )
        assert status == 409
        assert fail["error"] == "no_audio_track"
        assert any(k == "capture_incomplete" for k, _p in events)
    finally:
        server.stop()


def test_chunk_then_duplicate_finalize_without_ffmpeg_gap(tmp_path: Path):
    server = CaptureServer(tmp_path, on_event=lambda *_a: None)
    server.start()
    try:
        status, created = _request(
            server,
            "POST",
            "/api/recordings",
            body=b"{}",
            extra={"Content-Type": "application/json"},
        )
        rec_id = created["id"]
        data = b"not-a-real-webm"
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        status, ack = _request(
            server,
            "PUT",
            f"/api/recordings/{rec_id}/chunks/0",
            body=data,
            extra={"X-Content-SHA256": digest, "X-Last-Chunk": "1"},
        )
        assert status == 200
        status, fin = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True, "mime_type": "video/webm"}).encode(),
            extra={"Content-Type": "application/json"},
        )
        assert status == 200
        assert fin["complete"] is False
        assert fin["process"] is False
        status, fin2 = _request(
            server,
            "POST",
            f"/api/recordings/{rec_id}/finalize",
            body=json.dumps({"stop_reason": "user_stop", "audio_track": True}).encode(),
            extra={"Content-Type": "application/json"},
        )
        assert status == 200
        assert fin2.get("duplicate") is True
    finally:
        server.stop()


def test_stop_server_ends_health_and_cancels_session(tmp_path: Path):
    import hashlib
    import urllib.error
    import urllib.request

    server = CaptureServer(tmp_path)
    server.start()
    rec = server.registry.create("abc123")
    rec.write_chunk(0, b"AAA", hashlib.sha256(b"AAA").hexdigest(), last=True)
    url = f"http://127.0.0.1:{server.port}/health"
    server.stop()
    try:
        urllib.request.urlopen(url, timeout=2)
        raise AssertionError("server should be down")
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
    assert rec.cancelled is True


def test_recorder_core_script_is_served(tmp_path: Path):
    import urllib.request

    server = CaptureServer(tmp_path)
    server.start()
    try:
        url = f"http://127.0.0.1:{server.port}/recorder-core.js"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert "CapturePipeline" in body
        html_url = f"http://127.0.0.1:{server.port}/recorder?token={server.token}"
        with urllib.request.urlopen(html_url, timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert "recorder-core.js" in html
        assert "Cancel recording" in html
    finally:
        server.stop()
