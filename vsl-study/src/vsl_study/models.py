"""Typed records for a VSL Study job."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


StageStatus = Literal["pending", "running", "complete", "skipped", "failed"]
CaptureReason = Literal["scene_start", "interval", "user"]
OcrStatus = Literal["skipped", "ok", "failed", "unavailable"]


def _from_dict(cls, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class SourceIdentity:
    path: str
    resolved_path: str
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    duration_s: float


@dataclass
class ProcessSettings:
    model: str = "small.en"
    language: str = "en"
    task: str = "transcribe"
    detector: str = "adaptive"
    interval: float = 5.0
    scene_start_offset: float = 0.25
    max_width: int = 1280
    ocr: bool = True
    context_window_s: float = 2.0
    compact_view: bool = True
    include_media: bool = False
    device: str = "auto"  # auto | cpu | cuda
    capture: dict[str, Any] | None = None

    def transcribe_key(self, fingerprint: str) -> str:
        return f"{fingerprint}|{self.model}|{self.language}|{self.task}"

    def scenes_key(self, fingerprint: str) -> str:
        return f"{fingerprint}|{self.detector}"

    def frames_key(self, fingerprint: str, extra_times: list[float]) -> str:
        extras = ",".join(f"{t:.3f}" for t in extra_times)
        return (
            f"{fingerprint}|{self.detector}|{self.interval:.3f}|"
            f"{self.scene_start_offset:.3f}|{self.max_width}|{extras}"
        )


@dataclass
class VideoInfo:
    path: str
    resolved_path: str
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    duration_s: float
    width: int | None
    height: int | None
    fps_avg: float | None
    fps_r: str | None
    time_base: str | None
    video_start_s: float
    audio_start_s: float | None
    format_start_s: float
    rotation: int
    has_audio: bool
    has_video: bool
    vfr: bool
    format_name: str
    audio_codec: str | None
    video_codec: str | None
    notes: list[str] = field(default_factory=list)
    probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoInfo":
        return _from_dict(cls, data)


@dataclass
class TranscriptSegment:
    id: str
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return _from_dict(cls, data)


@dataclass
class TranscriptResult:
    status: StageStatus
    model: str
    language: str | None
    language_source: str  # configured | detected | n/a
    device: str
    device_note: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    text: str = ""
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    diagnostics_note: str = (
        "avg_logprob, no_speech_prob, and compression_ratio are raw decoder "
        "diagnostics, not calibrated accuracy percentages."
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptResult":
        segs = [TranscriptSegment.from_dict(s) for s in data.get("segments") or []]
        payload = dict(data)
        payload["segments"] = segs
        return _from_dict(cls, payload)


@dataclass
class Scene:
    id: str
    start: float
    end: float
    start_frame: int | None = None
    end_frame: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return _from_dict(cls, data)


@dataclass
class ScreenshotRecord:
    id: str
    requested_time: float
    actual_time: float
    scene_id: str | None
    relative_path: str
    capture_reason: str
    nearby_segment_ids: list[str] = field(default_factory=list)
    ocr_text: str | None = None
    ocr_status: str = "skipped"
    ocr_engine: str | None = None
    compact_retained: bool = True
    compact_refers_to: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreenshotRecord":
        return _from_dict(cls, data)
