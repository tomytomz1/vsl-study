import hashlib
import threading
import time
from pathlib import Path

import pytest

from vsl_study.capture_meta import sanitize_source_url, should_process
from vsl_study.capture_store import CaptureError, ChunkSession, publish_recording, sha256_hex


def _chunk(session: ChunkSession, seq: int, data: bytes, last: bool = False):
    return session.write_chunk(seq, data, sha256_hex(data), last=last)


def test_sanitize_source_url_strips_secrets():
    assert sanitize_source_url("https://user:token@example.com/vsl?key=secret#frag") == "https://example.com/vsl"
    assert sanitize_source_url("javascript:alert(1)") == ""
    assert sanitize_source_url("ftp://files.example") == ""


def test_chunk_order_retry_and_conflict(tmp_path: Path):
    session = ChunkSession(tmp_path, "abc123")
    _chunk(session, 0, b"AAA")
    _chunk(session, 1, b"BBB")
    again = _chunk(session, 1, b"BBB")
    assert again["duplicate"] is True
    with pytest.raises(CaptureError) as gap:
        _chunk(session, 3, b"DDD")
    assert gap.value.code == "gap"
    with pytest.raises(CaptureError) as conflict:
        _chunk(session, 1, b"XXX")
    assert conflict.value.code == "conflict"
    _chunk(session, 2, b"CCC", last=True)
    info = session.begin_finalize("user_stop")
    assert info["already"] is False
    assert session.stream_path.read_bytes() == b"AAABBBCCC"
    again_fin = session.begin_finalize("user_stop")
    assert again_fin["already"] is True


def test_missing_final_chunk(tmp_path: Path):
    session = ChunkSession(tmp_path, "abc123")
    _chunk(session, 0, b"AAA")
    with pytest.raises(CaptureError) as exc:
        session.begin_finalize("user_stop")
    assert exc.value.code == "missing_final"


def test_empty_last_marker(tmp_path: Path):
    session = ChunkSession(tmp_path, "abc123")
    _chunk(session, 0, b"AAA")
    session.write_chunk(1, b"", hashlib.sha256(b"").hexdigest(), last=True)
    info = session.begin_finalize("user_stop")
    assert session.stream_path.read_bytes() == b"AAA"
    assert info["already"] is False


def test_backpressure_and_slow_write(tmp_path: Path):
    session = ChunkSession(tmp_path, "abc123", max_chunk_bytes=100, max_pending_bytes=20)
    session.pending_bytes = 18
    with pytest.raises(CaptureError) as exc:
        session.write_chunk(0, b"123456", sha256_hex(b"123456"))
    assert exc.value.code == "backpressure"
    session.pending_bytes = 0

    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        time.sleep(0.05)
        release.wait(1)

    t = threading.Thread(
        target=lambda: session.write_chunk(
            0, b"okdata", sha256_hex(b"okdata"), write_hook=slow
        )
    )
    t.start()
    started.wait(1)
    release.set()
    t.join(2)
    assert session.bytes_written == 6


def test_cancel_and_processing_once(tmp_path: Path):
    session = ChunkSession(tmp_path, "abc123")
    _chunk(session, 0, b"AAA", last=True)
    session.cancel("app_closed")
    with pytest.raises(CaptureError) as exc:
        session.begin_finalize("stop_sharing")
    assert exc.value.code == "cancelled"
    session2 = ChunkSession(tmp_path, "def456")
    _chunk(session2, 0, b"AAA", last=True)
    session2.begin_finalize("user_stop")
    assert session2.try_begin_processing() is True
    assert session2.try_begin_processing() is False


def test_should_process_only_intentional_complete():
    assert should_process("user_stop", True, True) is True
    assert should_process("max_duration", True, True) is True
    assert should_process("trailing_silence", True, True) is True
    assert should_process("stop_sharing", True, True) is False
    assert should_process("user_stop", False, True) is False
    assert should_process("user_stop", True, False) is False


def test_publish_recording_copies_in_chunks(tmp_path: Path):
    src = tmp_path / "stream.bin"
    dest = tmp_path / "recording.webm"
    payload = b"abcdef" * 40_000
    src.write_bytes(payload)
    publish_recording(src, dest, chunk_size=1024)
    assert dest.read_bytes() == payload
    assert src.read_bytes() == payload
    assert not (tmp_path / "recording.webm.part").exists()
