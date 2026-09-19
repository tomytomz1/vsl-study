"""Typed records for a VSL Study job."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


StageStatus = Literal["pending", "running", "complete", "skipped", "failed"]
CaptureReason = Literal["scene_start", "interval", "user"]
OcrStatus = Literal["skipped", "ok", "failed", "unavailable", "reused"]


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
    interval: float = 15.0
    scene_start_offset: float = 0.25
    max_width: int = 1280
    ocr: bool = True
    context_window_s: float = 2.0
    compact_view: bool = True
    include_media: bool = False
    device: str = "auto"  # auto | cpu | cuda
    capture: dict[str, Any] | None = None
    sampling_policy: str = "standard-v1"
    max_auto_screenshots: int | None = 600
    ocr_budget: int | None = 300
    extra_ocr_ids: list[str] = field(default_factory=list)
    transcribe_backend: str = "faster-whisper"
    transcribe_compute_type: str = "int8"
    transcribe_beam_size: int = 5
    transcribe_cpu_threads: int = 0

    def uses_legacy_screenshot_policy(self) -> bool:
        return self.sampling_policy in {"", "legacy"} or self.max_auto_screenshots is None

    def transcribe_key(self, fingerprint: str) -> str:
        base = f"{fingerprint}|{self.model}|{self.language}|{self.task}"
        backend = (self.transcribe_backend or "").strip() or "openai-whisper"
        if backend == "openai-whisper":
            return base
        threads = self.transcribe_cpu_threads or "auto"
        return (
            f"{base}|{backend}|{self.transcribe_compute_type}|"
            f"beam={int(self.transcribe_beam_size)}|threads={threads}"
        )

    def scenes_key(self, fingerprint: str) -> str:
        return f"{fingerprint}|{self.detector}|seconds-1"

    def frames_key(self, fingerprint: str, extra_times: list[float]) -> str:
        extras = ",".join(f"{t:.3f}" for t in extra_times)
        base = (
            f"{fingerprint}|{self.detector}|{self.interval:.3f}|"
            f"{self.scene_start_offset:.3f}|{self.max_width}|{extras}|seq-pts-4"
        )
        if self.uses_legacy_screenshot_policy():
            return base
        cap = "none" if self.max_auto_screenshots is None else int(self.max_auto_screenshots)
        return f"{base}|{self.sampling_policy}|max={cap}"

    def stored_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "language": self.language,
            "task": self.task,
            "detector": self.detector,
            "interval": self.interval,
            "scene_start_offset": self.scene_start_offset,
            "max_width": self.max_width,
            "ocr": self.ocr,
            "context_window_s": self.context_window_s,
            "compact_view": self.compact_view,
            "include_media": self.include_media,
            "device": self.device,
            "sampling_policy": self.sampling_policy,
            "max_auto_screenshots": self.max_auto_screenshots,
            "ocr_budget": self.ocr_budget,
            "extra_ocr_ids": list(self.extra_ocr_ids or []),
            "transcribe_backend": self.transcribe_backend,
            "transcribe_compute_type": self.transcribe_compute_type,
            "transcribe_beam_size": self.transcribe_beam_size,
            "transcribe_cpu_threads": self.transcribe_cpu_threads,
        }


def settings_from_stored(raw: dict[str, Any] | None, capture: dict[str, Any] | None = None) -> ProcessSettings:
    """Rebuild settings for an existing job without silently applying new defaults."""
    data = dict(raw or {})
    legacy = "sampling_policy" not in data and "max_auto_screenshots" not in data
    if legacy:
        policy = "legacy"
        max_auto = None
        ocr_budget = None
        interval = float(data.get("interval", 5.0))
        backend = str(data.get("transcribe_backend") or "openai-whisper")
        compute = str(data.get("transcribe_compute_type") or "")
    else:
        policy = str(data.get("sampling_policy") or "standard-v1")
        max_auto = data.get("max_auto_screenshots")
        if max_auto is not None:
            max_auto = int(max_auto)
        ocr_budget = data.get("ocr_budget")
        if ocr_budget is not None:
            ocr_budget = int(ocr_budget)
        interval = float(data.get("interval", 15.0))
        backend = str(data.get("transcribe_backend") or "faster-whisper")
        compute = str(data.get("transcribe_compute_type") or ("int8" if backend == "faster-whisper" else ""))
    return ProcessSettings(
        model=str(data.get("model") or "small.en"),
        language=str(data.get("language") or "en"),
        task=str(data.get("task") or "transcribe"),
        detector=str(data.get("detector") or "adaptive"),
        interval=interval,
        scene_start_offset=float(data.get("scene_start_offset", 0.25)),
        max_width=int(data.get("max_width", 1280)),
        ocr=bool(data.get("ocr", True)),
        context_window_s=float(data.get("context_window_s", 2.0)),
        compact_view=bool(data.get("compact_view", True)),
        include_media=bool(data.get("include_media", False)),
        device=str(data.get("device") or "auto"),
        capture=capture,
        sampling_policy=policy,
        max_auto_screenshots=max_auto,
        ocr_budget=ocr_budget,
        extra_ocr_ids=[str(item) for item in (data.get("extra_ocr_ids") or []) if item],
        transcribe_backend=backend,
        transcribe_compute_type=compute,
        transcribe_beam_size=int(data.get("transcribe_beam_size") or 5),
        transcribe_cpu_threads=int(data.get("transcribe_cpu_threads") or 0),
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
    video_duration_s: float | None = None
    fps_trusted: bool = True
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
    backend: str = ""
    compute_type: str = ""
    beam_size: int | None = None
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
    ocr_skip_reason: str = ""
    ocr_reused_from: str = ""
    selected_for_ocr: bool = True
    compact_retained: bool = True
    compact_refers_to: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreenshotRecord":
        return _from_dict(cls, data)
