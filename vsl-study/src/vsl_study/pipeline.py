"""Orchestrate inspect → transcribe → scenes → frames → align → export with caching."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from vsl_study.align import CONTEXT_WINDOW_NOTE, match_screenshots
from vsl_study.cache import JobConflictError, JobDir, identity_from_info
from vsl_study.export import (
    write_ai_prompt,
    write_five_minute_folders,
    write_reports,
    write_scenes_csv,
    write_transcripts,
    write_zip,
)
from vsl_study.frames import (
    apply_sequential_compact,
    build_candidates,
    capture_candidates,
)
from vsl_study.media import extract_aligned_audio, inspect_video
from vsl_study.models import (
    ProcessSettings,
    Scene,
    ScreenshotRecord,
    TranscriptResult,
    VideoInfo,
)
from vsl_study.ocr import (
    OCR_CACHE_VERSION,
    apply_ocr,
    engine_versions,
    write_ocr_outputs,
)
from vsl_study.scenes import detect_scenes
from vsl_study.transcribe import (
    failed_transcript,
    select_device,
    transcribe_wav,
    unavailable_transcript,
    validate_model_language,
)

ProgressCb = Callable[[str, str], None]


class PipelineError(RuntimeError):
    pass


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import whisper

        versions["openai-whisper"] = getattr(whisper, "__version__", "unknown")
    except Exception:
        versions["openai-whisper"] = "unavailable"
    try:
        import importlib.metadata

        versions["scenedetect"] = importlib.metadata.version("scenedetect")
    except Exception:
        versions["scenedetect"] = "unavailable"
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda"] = "yes" if torch.cuda.is_available() else "no"
    except Exception:
        versions["torch"] = "unavailable"
    try:
        import streamlit

        versions["streamlit"] = streamlit.__version__
    except Exception:
        versions["streamlit"] = "unavailable"
    try:
        from PIL import Image

        versions["pillow"] = getattr(Image, "__version__", "ok")
    except Exception:
        versions["pillow"] = "unavailable"
    versions.update(engine_versions())
    return versions


def _progress(cb: ProgressCb | None, stage: str, message: str) -> None:
    if cb:
        cb(stage, message)


def process_video(
    input_path: str | Path,
    output_dir: str | Path,
    settings: ProcessSettings | None = None,
    extra_times: list[float] | None = None,
    progress: ProgressCb | None = None,
    frames_only: bool = False,
) -> dict[str, Any]:
    settings = settings or ProcessSettings()
    extra_times = list(extra_times or [])
    validate_model_language(settings.model, settings.language, settings.task)

    src = Path(input_path)
    job = JobDir(Path(output_dir))
    job.ensure()
    _progress(progress, "inspect", "Validating input and inspecting streams")
    info = inspect_video(src, progress=progress)
    identity = identity_from_info(info)
    try:
        job.bind_source(identity, settings)
    except JobConflictError:
        raise

    inspect_key = info.fingerprint
    cached_inspect = job.read_complete_stage("inspect", inspect_key)
    if cached_inspect is None:
        job.write_stage("inspect", inspect_key, "complete", {"info": info.to_dict()})
    else:
        info = VideoInfo.from_dict(cached_inspect["info"])

    transcript = _stage_transcribe(job, info, settings, progress, skip=frames_only)
    scenes = _stage_scenes(job, info, settings, progress)
    screenshots = _stage_frames(job, info, settings, scenes, extra_times, progress)
    ocr_note = _stage_ocr(job, screenshots, settings, progress)
    write_ocr_outputs(job.root, screenshots, settings.ocr, ocr_note)
    gaps = match_screenshots(screenshots, transcript.segments, settings.context_window_s)
    if settings.compact_view:
        apply_sequential_compact(screenshots, job.frames)

    _progress(progress, "export", "Writing transcripts, reports, and evidence folders")
    write_transcripts(job, transcript)
    write_scenes_csv(job, scenes)
    write_ai_prompt(job)
    write_five_minute_folders(job, info, transcript, screenshots)
    versions = dependency_versions()
    stage_status = (job.load_job() or {}).get("stages", {})
    processing_gaps = list(gaps)
    if ocr_note and settings.ocr:
        processing_gaps.append({"type": "ocr", "status": "unavailable_or_failed", "detail": ocr_note})
    if transcript.status != "complete":
        processing_gaps.append(
            {
                "type": "transcription",
                "status": transcript.status,
                "detail": transcript.error,
            }
        )
    for note in info.notes:
        processing_gaps.append({"type": "media_note", "detail": note})
    capture = settings.capture or (job.load_job() or {}).get("capture")
    if capture and not capture.get("complete"):
        processing_gaps.append(
            {
                "type": "capture_partial",
                "detail": capture.get("timeline_note"),
                "stop_reason": capture.get("stop_reason"),
            }
        )

    manifest = {
        "schema": 1,
        "source": {
            "path": info.path,
            "resolved_path": info.resolved_path,
            "fingerprint": info.fingerprint,
            "size_bytes": info.size_bytes,
            "duration_s": info.duration_s,
            "width": info.width,
            "height": info.height,
            "fps_avg": info.fps_avg,
            "fps_r": info.fps_r,
            "time_base": info.time_base,
            "video_start_s": info.video_start_s,
            "audio_start_s": info.audio_start_s,
            "rotation": info.rotation,
            "has_audio": info.has_audio,
            "vfr": info.vfr,
            "format_name": info.format_name,
        },
        "settings": {
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
        },
        "alignment_note": CONTEXT_WINDOW_NOTE,
        "dependency_versions": versions,
        "stage_status": stage_status,
        "transcript": transcript.to_dict(),
        "scenes": [s.to_dict() for s in scenes],
        "screenshots": [s.to_dict() for s in screenshots],
        "gaps": processing_gaps,
    }
    capture = settings.capture or (job.load_job() or {}).get("capture")
    if capture:
        manifest["capture"] = capture
    from vsl_study.cache import atomic_write_json

    atomic_write_json(job.root / "manifest.json", manifest)
    write_reports(
        job,
        info,
        settings,
        transcript,
        scenes,
        screenshots,
        processing_gaps,
        versions,
        stage_status,
    )
    zip_path = write_zip(job, settings.include_media)
    job.write_stage(
        "export",
        f"{info.fingerprint}|reports|{settings.interval:.3f}|ocr={settings.ocr}",
        "complete",
        {"zip": zip_path.name},
    )
    _progress(progress, "done", f"Wrote evidence package to {job.root}")
    return {
        "job": str(job.root),
        "manifest": str(job.root / "manifest.json"),
        "zip": str(zip_path),
        "transcript_status": transcript.status,
        "screenshot_count": len(screenshots),
        "scene_count": len(scenes),
    }


def add_frames_at(job_dir: str | Path, timestamps: list[float], progress: ProgressCb | None = None) -> dict[str, Any]:
    job = JobDir(Path(job_dir))
    meta = job.load_job()
    if not meta or not meta.get("source"):
        raise PipelineError(f"Not a VSL Study job directory: {job.root}")
    inspect_data = job.read_complete_stage("inspect", meta["source"]["fingerprint"])
    if not inspect_data:
        raise PipelineError("Job is missing a complete inspect stage.")
    info = VideoInfo.from_dict(inspect_data["info"])
    raw_settings = meta.get("settings") or {}
    settings = ProcessSettings(
        model=raw_settings.get("model", "small.en"),
        language=raw_settings.get("language", "en"),
        task=raw_settings.get("task", "transcribe"),
        detector=raw_settings.get("detector", "adaptive"),
        interval=float(raw_settings.get("interval", 5)),
        scene_start_offset=float(raw_settings.get("scene_start_offset", 0.25)),
        max_width=int(raw_settings.get("max_width", 1280)),
        ocr=bool(raw_settings.get("ocr", True)),
        context_window_s=float(raw_settings.get("context_window_s", 2.0)),
        compact_view=bool(raw_settings.get("compact_view", True)),
        include_media=bool(raw_settings.get("include_media", False)),
        device=raw_settings.get("device", "auto"),
        capture=meta.get("capture"),
    )
    existing_extra = list((meta.get("extra_times") or []))
    merged = sorted(set(float(t) for t in existing_extra + list(timestamps)))
    meta["extra_times"] = merged
    from vsl_study.cache import atomic_write_json

    atomic_write_json(job.job_file, meta)
    return process_video(
        info.resolved_path,
        job.root,
        settings=settings,
        extra_times=merged,
        progress=progress,
        frames_only=True,
    )


def _stage_transcribe(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    progress: ProgressCb | None,
    skip: bool,
) -> TranscriptResult:
    key = settings.transcribe_key(info.fingerprint)
    cached = job.read_complete_stage("transcribe", key)
    if cached:
        _progress(progress, "transcribe", "Using cached transcript")
        return TranscriptResult.from_dict(cached["transcript"])
    if skip:
        # frames-only still needs whatever transcript already exists, even if settings changed.
        path = job.stage_path("transcribe")
        if path.exists():
            from vsl_study.cache import read_json

            data = read_json(path)
            if data.get("status") == "complete":
                return TranscriptResult.from_dict(data["transcript"])
        return unavailable_transcript("No cached transcript in this job.", settings)

    if not info.has_audio:
        result = unavailable_transcript(
            "No audio stream in the source video. Visual outputs were still produced.",
            settings,
        )
        job.write_stage("transcribe", key, "skipped", {"transcript": result.to_dict()})
        return result

    job.write_stage("transcribe", key, "running", {})
    device, device_note, _ = select_device(settings.device)
    try:
        wav = job.audio_wav()
        audio_key = info.fingerprint
        audio_meta = job.read_complete_stage("audio", audio_key)
        if audio_meta is None or not wav.exists():
            extract_aligned_audio(info, wav, progress=progress)
            job.write_stage("audio", audio_key, "complete", {"wav": wav.name})
        result = transcribe_wav(str(wav), settings, progress=progress)
        job.write_stage("transcribe", key, "complete", {"transcript": result.to_dict()})
        return result
    except Exception as exc:  # noqa: BLE001
        result = failed_transcript(str(exc), settings, device, device_note)
        job.write_stage(
            "transcribe",
            key,
            "failed",
            {"transcript": result.to_dict(), "traceback": traceback.format_exc()},
        )
        return result


def _stage_scenes(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    progress: ProgressCb | None,
) -> list[Scene]:
    key = settings.scenes_key(info.fingerprint)
    cached = job.read_complete_stage("scenes", key)
    if cached:
        _progress(progress, "scenes", "Using cached scene list")
        return [Scene.from_dict(s) for s in cached["scenes"]]
    job.write_stage("scenes", key, "running", {})
    scenes = detect_scenes(info, settings.detector, progress=progress)
    job.write_stage("scenes", key, "complete", {"scenes": [s.to_dict() for s in scenes]})
    return scenes


def _stage_frames(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    scenes: list[Scene],
    extra_times: list[float],
    progress: ProgressCb | None,
) -> list[ScreenshotRecord]:
    key = settings.frames_key(info.fingerprint, extra_times)
    cached = job.read_complete_stage("frames", key)
    if cached:
        _progress(progress, "frames", "Using cached screenshots")
        return [ScreenshotRecord.from_dict(s) for s in cached["screenshots"]]

    existing: dict[str, ScreenshotRecord] = {}
    old = job.stage_path("frames")
    if old.exists():
        from vsl_study.cache import read_json

        try:
            prev = read_json(old)
            if prev.get("status") == "complete":
                for rec in prev.get("screenshots") or []:
                    shot = ScreenshotRecord.from_dict(rec)
                    existing[f"{shot.capture_reason}:{shot.requested_time:.3f}"] = shot
        except Exception:
            existing = {}

    candidates = build_candidates(
        info,
        scenes,
        interval=settings.interval,
        scene_start_offset=settings.scene_start_offset,
        extra_times=extra_times,
    )
    job.write_stage("frames", key, "running", {"count": len(candidates)})
    records = capture_candidates(
        info,
        scenes,
        job.frames,
        settings.max_width,
        candidates,
        existing=existing,
        progress=progress,
    )
    job.write_stage(
        "frames",
        key,
        "complete",
        {"screenshots": [s.to_dict() for s in records]},
    )
    return records


def _ocr_cache_key(settings: ProcessSettings, screenshots: list[ScreenshotRecord]) -> str:
    import hashlib

    blob = "|".join(f"{s.id}:{s.relative_path}:{s.actual_time:.3f}" for s in screenshots)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return f"{OCR_CACHE_VERSION}|ocr={settings.ocr}|{digest}"


def _stage_ocr(
    job: JobDir,
    screenshots: list[ScreenshotRecord],
    settings: ProcessSettings,
    progress: ProgressCb | None,
) -> str | None:
    key = _ocr_cache_key(settings, screenshots)
    cached = job.read_complete_stage("ocr", key)
    if cached:
        _progress(progress, "ocr", "Using cached on-screen text")
        by_id = {row["id"]: row for row in cached.get("screenshots") or [] if row.get("id")}
        for record in screenshots:
            row = by_id.get(record.id)
            if not row:
                continue
            record.ocr_text = row.get("ocr_text")
            record.ocr_status = row.get("ocr_status") or "skipped"
            record.ocr_engine = row.get("ocr_engine")
        return cached.get("note")
    job.write_stage("ocr", key, "running", {"count": len(screenshots)})
    note = apply_ocr(screenshots, job.root, settings.ocr, progress=progress)
    job.write_stage(
        "ocr",
        key,
        "complete",
        {
            "note": note,
            "screenshots": [
                {
                    "id": record.id,
                    "ocr_text": record.ocr_text,
                    "ocr_status": record.ocr_status,
                    "ocr_engine": record.ocr_engine,
                }
                for record in screenshots
            ],
        },
    )
    return note
