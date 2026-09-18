"""Ordered chunk assembly for one MediaRecorder session."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from vsl_study.capture_meta import INTENTIONAL_STOPS, utc_now
from vsl_study.cache import atomic_write_json

WriteHook = Callable[[], None]


class CaptureError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ChunkSession:
    """One uninterrupted recorder session. Chunks are appended in sequence order."""

    def __init__(
        self,
        root: Path,
        recording_id: str,
        *,
        max_chunk_bytes: int = 16 * 1024 * 1024,
        max_pending_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not recording_id.isalnum() or len(recording_id) > 64:
            raise CaptureError("bad_id", "Invalid recording id.")
        self.id = recording_id
        self.dir = root / recording_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stream_path = self.dir / "stream.bin"
        self.meta_path = self.dir / "meta.json"
        self.max_chunk_bytes = max_chunk_bytes
        self.max_pending_bytes = max_pending_bytes
        self.lock = threading.RLock()
        self.pending_bytes = 0
        self.expected_seq = 0
        self.checksums: dict[int, str] = {}
        self.bytes_written = 0
        self.last_seq: int | None = None
        self.created_at = utc_now()
        self.finalized = False
        self.processing_started = False
        self.cancelled = False
        self.mime_type = ""
        self.title = ""
        self.source_url = ""
        self.max_duration_s: float | None = None
        self._load()

    def _load(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.expected_seq = int(data.get("expected_seq") or 0)
        self.checksums = {int(k): str(v) for k, v in (data.get("checksums") or {}).items()}
        self.bytes_written = int(data.get("bytes_written") or 0)
        last = data.get("last_seq")
        self.last_seq = int(last) if last is not None else None
        self.finalized = bool(data.get("finalized"))
        self.processing_started = bool(data.get("processing_started"))
        self.cancelled = bool(data.get("cancelled"))
        self.mime_type = str(data.get("mime_type") or "")
        self.title = str(data.get("title") or "")
        self.source_url = str(data.get("source_url") or "")
        max_dur = data.get("max_duration_s")
        self.max_duration_s = float(max_dur) if max_dur is not None else None
        self.created_at = str(data.get("created_at") or self.created_at)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "expected_seq": self.expected_seq,
                "bytes_written": self.bytes_written,
                "last_seq": self.last_seq,
                "finalized": self.finalized,
                "processing_started": self.processing_started,
                "cancelled": self.cancelled,
                "pending_bytes": self.pending_bytes,
                "mime_type": self.mime_type,
                "title": self.title,
                "created_at": self.created_at,
            }

    def _save(self, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "id": self.id,
            "expected_seq": self.expected_seq,
            "checksums": {str(k): v for k, v in self.checksums.items()},
            "bytes_written": self.bytes_written,
            "last_seq": self.last_seq,
            "finalized": self.finalized,
            "processing_started": self.processing_started,
            "cancelled": self.cancelled,
            "mime_type": self.mime_type,
            "title": self.title,
            "source_url": self.source_url,
            "max_duration_s": self.max_duration_s,
            "created_at": self.created_at,
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.meta_path, payload)

    def configure(self, *, title: str = "", source_url: str = "", mime_type: str = "", max_duration_s: float | None = None) -> None:
        with self.lock:
            if title:
                self.title = title[:200]
            if source_url:
                self.source_url = source_url
            if mime_type:
                self.mime_type = mime_type
            if max_duration_s is not None:
                self.max_duration_s = max_duration_s
            self._save()

    def write_chunk(
        self,
        seq: int,
        data: bytes,
        checksum: str,
        *,
        last: bool = False,
        write_hook: WriteHook | None = None,
    ) -> dict[str, Any]:
        digest = (checksum or "").strip().lower()
        actual = sha256_hex(data)
        if digest and digest != actual:
            raise CaptureError("checksum", "Chunk checksum did not match the bytes received.")
        digest = actual
        if seq < 0 or not isinstance(seq, int):
            raise CaptureError("bad_seq", "Chunk sequence must be 0, 1, 2, …")
        if len(data) > self.max_chunk_bytes:
            raise CaptureError("too_large", f"Chunk is larger than {self.max_chunk_bytes} bytes.")

        with self.lock:
            if self.cancelled:
                raise CaptureError("cancelled", "This recording was cancelled.")
            if self.finalized:
                stored = self.checksums.get(seq)
                if stored == digest:
                    return {"ok": True, "duplicate": True, "expected_seq": self.expected_seq}
                raise CaptureError("finalized", "This recording is already finished.")
            stored = self.checksums.get(seq)
            if stored is not None:
                if stored == digest:
                    if last:
                        self.last_seq = seq
                        self._save()
                    return {"ok": True, "duplicate": True, "expected_seq": self.expected_seq}
                raise CaptureError("conflict", f"Chunk {seq} was already saved with different bytes.")
            if seq > self.expected_seq:
                raise CaptureError(
                    "gap",
                    f"Chunk {seq} arrived before {self.expected_seq}. Upload in order and retry.",
                )
            if len(data) == 0:
                if last and self.expected_seq > 0:
                    self.last_seq = self.expected_seq - 1
                    self._save()
                    return {"ok": True, "last": True, "expected_seq": self.expected_seq, "empty_last": True}
                if last:
                    raise CaptureError("empty", "No media was saved.")
                return {"ok": True, "ignored": True, "expected_seq": self.expected_seq}
            if self.pending_bytes + len(data) > self.max_pending_bytes:
                raise CaptureError(
                    "backpressure",
                    "Saving cannot keep up. Stop the recording; unsaved audio was not dropped silently.",
                )
            self.pending_bytes += len(data)
            try:
                if write_hook:
                    write_hook()
                with self.stream_path.open("ab") as handle:
                    handle.write(data)
                    handle.flush()
                self.checksums[seq] = digest
                self.expected_seq = seq + 1
                self.bytes_written += len(data)
                if last:
                    self.last_seq = seq
                self._save()
            finally:
                self.pending_bytes -= len(data)
            return {
                "ok": True,
                "duplicate": False,
                "expected_seq": self.expected_seq,
                "bytes_written": self.bytes_written,
                "last": last,
            }

    def mark_last(self, seq: int) -> None:
        with self.lock:
            self.last_seq = seq
            self._save()

    def cancel(self, reason: str = "cancelled") -> None:
        with self.lock:
            self.cancelled = True
            self._save({"cancel_reason": reason, "ended_at": utc_now()})

    def ready_to_finalize(self) -> bool:
        with self.lock:
            return self.last_seq is not None and self.expected_seq == self.last_seq + 1

    def begin_finalize(self, stop_reason: str) -> dict[str, Any]:
        """Mark finalized so a second click cannot assemble twice. Caller then validates."""
        with self.lock:
            if self.cancelled and stop_reason not in INTENTIONAL_STOPS:
                raise CaptureError("cancelled", "This recording was cancelled.")
            if self.finalized:
                return {"already": True, **self.snapshot()}
            if self.last_seq is None:
                raise CaptureError("missing_final", "The last media chunk never arrived. Partial file was kept.")
            if self.expected_seq != self.last_seq + 1:
                raise CaptureError(
                    "gap",
                    f"Missing chunks before the end (have {self.expected_seq}, last is {self.last_seq}).",
                )
            if self.bytes_written <= 0:
                raise CaptureError("empty", "No media was saved.")
            self.finalized = True
            self._save({"stop_reason": stop_reason, "ended_at": utc_now()})
            return {"already": False, **self.snapshot(), "stream_path": str(self.stream_path)}

    def try_begin_processing(self) -> bool:
        with self.lock:
            if self.processing_started:
                return False
            self.processing_started = True
            self._save()
            return True


class CaptureRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sessions: dict[str, ChunkSession] = {}

    def create(self, recording_id: str) -> ChunkSession:
        with self._lock:
            session = ChunkSession(self.root, recording_id)
            self._sessions[recording_id] = session
            return session

    def get(self, recording_id: str) -> ChunkSession:
        with self._lock:
            session = self._sessions.get(recording_id)
            if session:
                return session
            path = self.root / recording_id
            if path.is_dir():
                session = ChunkSession(self.root, recording_id)
                self._sessions[recording_id] = session
                return session
            raise CaptureError("not_found", "Unknown recording.")

    def cancel_all(self, reason: str = "app_closed") -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.cancel(reason)
