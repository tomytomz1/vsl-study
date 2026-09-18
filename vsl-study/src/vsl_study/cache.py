"""Atomic JSON/file writes and job-directory cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vsl_study.models import ProcessSettings, SourceIdentity, VideoInfo


SCHEMA = 1


class JobConflictError(RuntimeError):
    pass


class IncompleteStageError(RuntimeError):
    pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fingerprint_file(path: Path) -> str:
    stat = path.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(b"|")
    chunk = 262144
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size > chunk * 2:
            handle.seek(size - chunk)
            digest.update(handle.read(chunk))
        elif size > chunk:
            digest.update(handle.read())
    return digest.hexdigest()


class JobDir:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.cache = self.root / "cache"
        self.frames = self.root / "frames"
        self.contact_sheets = self.root / "contact_sheets"
        self.work = self.root / "work"
        self.clips = self.root / "evidence_5min"
        self.job_file = self.root / "job.json"

    def ensure(self) -> None:
        for folder in (self.root, self.cache, self.frames, self.contact_sheets, self.work, self.clips):
            folder.mkdir(parents=True, exist_ok=True)

    def load_job(self) -> dict[str, Any] | None:
        if not self.job_file.exists():
            return None
        data = read_json(self.job_file)
        if not isinstance(data, dict):
            raise JobConflictError("job.json is not a JSON object")
        return data

    def bind_source(
        self,
        identity: SourceIdentity,
        settings: ProcessSettings,
        *,
        capture_rejected: bool = False,
    ) -> dict[str, Any]:
        existing = self.load_job()
        if existing:
            prev = (existing.get("source") or {}).get("fingerprint")
            if prev and prev != identity.fingerprint:
                raise JobConflictError(
                    "Output directory already belongs to a different source video "
                    f"(fingerprint {prev[:12]}… vs {identity.fingerprint[:12]}…). "
                    "Use a different --output folder."
                )
        payload = existing or {"schema": SCHEMA, "stages": {}}
        payload["schema"] = SCHEMA
        payload["source"] = {
            "path": identity.path,
            "resolved_path": identity.resolved_path,
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "fingerprint": identity.fingerprint,
            "duration_s": identity.duration_s,
        }
        payload["settings"] = {
            "model": settings.model,
            "language": settings.language,
            "task": settings.task,
            "detector": settings.detector,
            "interval": settings.interval,
            "scene_start_offset": settings.scene_start_offset,
            "max_width": settings.max_width,
            "ocr": settings.ocr,
            "context_window_s": settings.context_window_s,
            "compact_view": settings.compact_view,
            "include_media": settings.include_media,
            "device": settings.device,
        }
        from vsl_study.capture_meta import capture_associates_with_video, public_capture_record

        incoming = public_capture_record(settings.capture) if settings.capture else None
        rejected = capture_rejected or (settings.capture is not None and incoming is None)
        if incoming:
            payload["capture"] = incoming
        elif rejected:
            payload.pop("capture", None)
        elif existing and existing.get("capture"):
            cleaned = public_capture_record(existing.get("capture"))
            video_path = identity.resolved_path or identity.path
            if cleaned and capture_associates_with_video(cleaned, video_path):
                payload["capture"] = cleaned
            else:
                payload.pop("capture", None)
        else:
            payload.pop("capture", None)
        payload.setdefault("stages", {})
        atomic_write_json(self.job_file, payload)
        return payload

    def stage_path(self, name: str) -> Path:
        return self.cache / f"{name}.json"

    def read_complete_stage(self, name: str, expected_key: str) -> dict[str, Any] | None:
        path = self.stage_path(name)
        if not path.exists():
            return None
        data = read_json(path)
        if not isinstance(data, dict):
            return None
        if data.get("status") != "complete":
            return None
        if data.get("cache_key") != expected_key:
            return None
        return data

    def write_stage(self, name: str, cache_key: str, status: str, payload: dict[str, Any]) -> None:
        body = {"status": status, "cache_key": cache_key, **payload}
        atomic_write_json(self.stage_path(name), body)
        job = self.load_job() or {"schema": SCHEMA, "stages": {}}
        job.setdefault("stages", {})[name] = {"status": status, "cache_key": cache_key}
        atomic_write_json(self.job_file, job)

    def audio_wav(self) -> Path:
        return self.cache / "audio.wav"


def identity_from_info(info: VideoInfo) -> SourceIdentity:
    return SourceIdentity(
        path=info.path,
        resolved_path=info.resolved_path,
        size_bytes=info.size_bytes,
        mtime_ns=info.mtime_ns,
        fingerprint=info.fingerprint,
        duration_s=info.duration_s,
    )
