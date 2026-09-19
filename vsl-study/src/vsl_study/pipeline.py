"""Orchestrate inspect → transcribe → scenes → frames → align → export with caching."""

from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from vsl_study.align import CONTEXT_WINDOW_NOTE, match_screenshots
from vsl_study.cache import JobConflictError, JobDir, atomic_write_json, identity_from_info
from vsl_study.capture_meta import (
    clear_portable_capture,
    resolve_capture_for_source,
    write_portable_capture,
)
from vsl_study.export import (
    package_source_media,
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
    settings_from_stored,
)
from vsl_study.ocr import (
    OCR_CACHE_VERSION,
    apply_ocr,
    engine_versions,
    write_ocr_outputs,
)
from vsl_study.sampling import coverage_note, select_ocr_ids, select_screenshots
from vsl_study.scenes import detect_scenes
from vsl_study.timing import StageClock
from vsl_study.transcribe import (
    failed_transcript,
    select_device,
    transcribe_wav,
    unavailable_transcript,
    validate_model_language,
)

ProgressCb = Callable[[str, str], None]
INSPECT_CACHE_VERSION = "media-v2"


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

        versions["faster-whisper"] = importlib.metadata.version("faster-whisper")
    except Exception:
        versions["faster-whisper"] = "unavailable"
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
    clock = StageClock()

    src = Path(input_path)
    job = JobDir(Path(output_dir))
    job.ensure()
    capture, capture_notes = resolve_capture_for_source(src, settings.capture)
    settings = replace(settings, capture=capture)
    _progress(progress, "inspect", "Opening the video and checking that it can be read.")
    with clock.span("inspect"):
        info = inspect_video(src, progress=progress)
    identity = identity_from_info(info)
    try:
        payload = job.bind_source(
            identity,
            settings,
            capture_rejected=capture is None and bool(capture_notes),
        )
    except JobConflictError:
        raise
    capture = payload.get("capture")
    settings = replace(settings, capture=capture)
    export_key = (
        f"{info.fingerprint}|reports|{settings.interval:.3f}|ocr={settings.ocr}|"
        f"media={int(settings.include_media)}|{settings.sampling_policy}|"
        f"max={settings.max_auto_screenshots}|ocrb={settings.ocr_budget}"
    )
    try:
        if capture:
            write_portable_capture(job.root, capture)
        else:
            clear_portable_capture(job.root)
    except OSError as exc:
        job.write_stage(
            "export",
            export_key,
            "failed",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        raise PipelineError(
            f"Could not update capture metadata; no new evidence ZIP was published: {exc}"
        ) from exc

    inspect_key = f"{info.fingerprint}|{INSPECT_CACHE_VERSION}"
    cached_inspect = job.read_complete_stage("inspect", inspect_key)
    if cached_inspect is None:
        job.write_stage("inspect", inspect_key, "complete", {"info": info.to_dict()})
        if clock.stages:
            clock.stages[-1]["cache_hit"] = False
    else:
        info = VideoInfo.from_dict(cached_inspect["info"])
        if clock.stages:
            clock.stages[-1]["cache_hit"] = True
    clock.note("source_width", info.width)
    clock.note("source_height", info.height)
    clock.note("source_duration_s", info.duration_s)
    clock.note("source_bytes", info.size_bytes)
    clock.note("source_video_codec", info.video_codec)
    clock.note("source_fps_avg", info.fps_avg)
    clock.note("source_fps_trusted", info.fps_trusted)
    clock.note("sampling_policy", settings.sampling_policy)
    clock.note("screenshot_interval_s", settings.interval)
    clock.note("max_auto_screenshots", settings.max_auto_screenshots)
    clock.note("ocr_budget", settings.ocr_budget)
    clock.note("transcribe_backend", settings.transcribe_backend)
    clock.note("transcribe_model", settings.model)
    clock.note("transcribe_compute_type", settings.transcribe_compute_type)

    transcript = _stage_transcribe(job, info, settings, progress, skip=frames_only, clock=clock)
    with clock.span("transcript_write"):
        write_transcripts(job, transcript)
    scenes = _stage_scenes(job, info, settings, progress, clock=clock)
    screenshots = _stage_frames(job, info, settings, scenes, extra_times, progress, clock=clock)
    ocr_note = _stage_ocr(job, screenshots, settings, progress, clock=clock)
    write_ocr_outputs(job.root, screenshots, settings.ocr, ocr_note)
    gaps = match_screenshots(screenshots, transcript.segments, settings.context_window_s)
    compact_error: str | None = None
    if settings.compact_view:
        try:
            apply_sequential_compact(screenshots, job.frames)
        except Exception as exc:  # noqa: BLE001
            compact_error = str(exc)

    _progress(progress, "export", "Writing transcripts, reports, and evidence folders")
    write_transcripts(job, transcript)
    write_scenes_csv(job, scenes)
    write_ai_prompt(job)
    write_five_minute_folders(job, info, transcript, screenshots)
    versions = dependency_versions()
    stage_status = (job.load_job() or {}).get("stages", {})
    processing_gaps = list(gaps)
    if compact_error:
        processing_gaps.append({"type": "compact", "status": "failed", "detail": compact_error})
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
    for note in capture_notes:
        processing_gaps.append({"type": "capture_metadata", "detail": note})
    processing_gaps.append(
        {
            "type": "sampling_coverage",
            "detail": coverage_note(
                policy=settings.sampling_policy,
                interval=settings.interval,
                max_auto=settings.max_auto_screenshots,
                ocr_budget=settings.ocr_budget,
            ),
        }
    )
    capture = settings.capture
    if capture and capture.get("complete") is False:
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
            "fps_trusted": info.fps_trusted,
            "time_base": info.time_base,
            "video_duration_s": info.video_duration_s,
            "video_start_s": info.video_start_s,
            "audio_start_s": info.audio_start_s,
            "rotation": info.rotation,
            "has_audio": info.has_audio,
            "vfr": info.vfr,
            "format_name": info.format_name,
        },
        "settings": settings.stored_dict(),
        "alignment_note": CONTEXT_WINDOW_NOTE,
        "dependency_versions": versions,
        "stage_status": stage_status,
        "transcript": transcript.to_dict(),
        "scenes": [s.to_dict() for s in scenes],
        "screenshots": [s.to_dict() for s in screenshots],
        "gaps": processing_gaps,
    }
    if capture:
        manifest["capture"] = capture
    packaged_media = None
    if settings.include_media:
        _progress(progress, "export", "Copying the source recording into the evidence package")
        try:
            with clock.span("media_copy"):
                packaged_media = package_source_media(job, info.resolved_path)
        except Exception as exc:  # noqa: BLE001
            job.write_stage(
                "export",
                export_key,
                "failed",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
            _write_timings(job, clock)
            raise PipelineError(
                f"Could not include the source recording in the evidence package: {exc}"
            ) from exc
        manifest["packaged_media"] = packaged_media
    atomic_write_json(job.root / "manifest.json", manifest)
    with clock.span("reports"):
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
            packaged_media=packaged_media,
        )
    job.write_stage(
        "reports",
        export_key,
        "complete",
        {"report": "report.html", "package_status": "packaging"},
    )
    _progress(progress, "export", "Report ready; packaging in progress")
    _progress(progress, "report_ready", str(job.root))
    try:
        with clock.span("zip"):
            zip_path = write_zip(job, settings.include_media, packaged_media=packaged_media)
    except Exception as exc:  # noqa: BLE001
        job.write_stage(
            "export",
            export_key,
            "failed",
            {"error": str(exc), "traceback": traceback.format_exc(), "report_ready": True},
        )
        _write_timings(job, clock)
        _progress(progress, "export", "Report is ready. Packaging failed.")
        return {
            "job": str(job.root),
            "manifest": str(job.root / "manifest.json"),
            "zip": None,
            "transcript_status": transcript.status,
            "screenshot_count": len(screenshots),
            "scene_count": len(scenes),
            "report_ready": True,
            "package_status": "failed",
            "package_error": str(exc),
            "timings": clock.to_dict(),
        }
    job.write_stage(
        "export",
        export_key,
        "complete",
        {"zip": zip_path.name, "packaged_media": packaged_media, "report_ready": True},
    )
    _write_timings(job, clock)
    _progress(progress, "done", f"Wrote evidence package to {job.root}")
    return {
        "job": str(job.root),
        "manifest": str(job.root / "manifest.json"),
        "zip": str(zip_path),
        "transcript_status": transcript.status,
        "screenshot_count": len(screenshots),
        "scene_count": len(scenes),
        "report_ready": True,
        "package_status": "complete",
        "timings": clock.to_dict(),
    }


def _write_timings(job: JobDir, clock: StageClock) -> None:
    payload = clock.to_dict()
    atomic_write_json(job.root / "timings.json", payload)
    meta = job.load_job() or {}
    meta["timings"] = payload
    atomic_write_json(job.job_file, meta)


def add_frames_at(job_dir: str | Path, timestamps: list[float], progress: ProgressCb | None = None) -> dict[str, Any]:
    job = JobDir(Path(job_dir))
    meta = job.load_job()
    if not meta or not meta.get("source"):
        raise PipelineError(f"Not a VSL Study job directory: {job.root}")
    inspect_data = job.read_complete_stage(
        "inspect", f"{meta['source']['fingerprint']}|{INSPECT_CACHE_VERSION}"
    )
    if not inspect_data:
        inspect_data = job.read_complete_stage("inspect", meta["source"]["fingerprint"])
    if not inspect_data:
        raise PipelineError("Job is missing a complete inspect stage.")
    info = VideoInfo.from_dict(inspect_data["info"])
    settings = settings_from_stored(meta.get("settings") or {}, capture=meta.get("capture"))
    existing_extra = list((meta.get("extra_times") or []))
    merged = sorted(set(float(t) for t in existing_extra + list(timestamps)))
    meta["extra_times"] = merged
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
    clock: StageClock | None = None,
) -> TranscriptResult:
    key = settings.transcribe_key(info.fingerprint)
    cached = job.read_complete_stage("transcribe", key)
    if cached:
        if clock:
            with clock.span("transcribe", cache_hit=True):
                _progress(progress, "transcribe", "Using cached transcript")
        else:
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
            if clock:
                with clock.span("audio_extract", cache_hit=False):
                    extract_aligned_audio(info, wav, progress=progress)
            else:
                extract_aligned_audio(info, wav, progress=progress)
            job.write_stage("audio", audio_key, "complete", {"wav": wav.name})
        elif clock:
            with clock.span("audio_extract", cache_hit=True):
                pass
        if clock:
            with clock.span("transcribe", cache_hit=False):
                result = transcribe_wav(str(wav), settings, progress=progress)
        else:
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
    clock: StageClock | None = None,
) -> list[Scene]:
    key = settings.scenes_key(info.fingerprint)
    cached = job.read_complete_stage("scenes", key)
    if cached:
        if clock:
            with clock.span("scenes", cache_hit=True):
                _progress(progress, "scenes", "Using cached scene list")
        else:
            _progress(progress, "scenes", "Using cached scene list")
        return [Scene.from_dict(s) for s in cached["scenes"]]
    job.write_stage("scenes", key, "running", {})
    if clock:
        with clock.span("scenes", cache_hit=False):
            scenes = detect_scenes(info, settings.detector, progress=progress)
    else:
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
    clock: StageClock | None = None,
) -> list[ScreenshotRecord]:
    key = settings.frames_key(info.fingerprint, extra_times)
    cached = job.read_complete_stage("frames", key)
    if cached:
        if clock:
            with clock.span("frames", cache_hit=True):
                _progress(progress, "frames", "Using cached screenshots")
                clock.note("screenshot_completed", len(cached.get("screenshots") or []))
        else:
            _progress(progress, "frames", "Using cached screenshots")
        return [ScreenshotRecord.from_dict(s) for s in cached["screenshots"]]

    existing: dict[str, ScreenshotRecord] = {}
    old = job.stage_path("frames")
    if old.exists():
        from vsl_study.cache import read_json

        try:
            prev = read_json(old)
            if prev.get("status") == "complete" and prev.get("cache_key") == key:
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
    candidate_count = len(candidates)
    if not settings.uses_legacy_screenshot_policy():
        candidates = select_screenshots(
            candidates,
            duration_s=float(info.duration_s or 0.0),
            max_auto=settings.max_auto_screenshots,
        )
    if clock:
        clock.note("screenshot_candidates", candidate_count)
        clock.note("screenshot_selected", len(candidates))
    _progress(
        progress,
        "frames",
        f"Capturing {len(candidates)} screenshots"
        + (f" (from {candidate_count} candidates)" if candidate_count != len(candidates) else ""),
    )
    job.write_stage(
        "frames",
        key,
        "running",
        {"scheduled": len(candidates), "completed": 0},
    )

    def frames_progress(stage: str, message: str) -> None:
        _progress(progress, stage, message)
        if stage != "frames":
            return
        done = sum(1 for path in job.frames.glob("frame_*.jpg") if path.is_file())
        job.write_stage(
            "frames",
            key,
            "running",
            {"scheduled": len(candidates), "completed": done},
        )

    try:
        if clock:
            with clock.span("frames", cache_hit=False):
                records = capture_candidates(
                    info,
                    scenes,
                    job.frames,
                    settings.max_width,
                    candidates,
                    existing=existing,
                    progress=frames_progress,
                )
        else:
            records = capture_candidates(
                info,
                scenes,
                job.frames,
                settings.max_width,
                candidates,
                existing=existing,
                progress=frames_progress,
            )
    except Exception as exc:  # noqa: BLE001
        done = sum(1 for path in job.frames.glob("frame_*.jpg") if path.is_file())
        job.write_stage(
            "frames",
            key,
            "failed",
            {
                "scheduled": len(candidates),
                "completed": done,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise PipelineError(f"Screenshot extraction failed: {exc}") from exc
    job.write_stage(
        "frames",
        key,
        "complete",
        {
            "scheduled": len(candidates),
            "completed": len(records),
            "candidates": candidate_count,
            "screenshots": [s.to_dict() for s in records],
        },
    )
    if clock:
        clock.note("screenshot_completed", len(records))
    return records


def _ocr_cache_key(settings: ProcessSettings, screenshots: list[ScreenshotRecord]) -> str:
    import hashlib

    blob = "|".join(f"{s.id}:{s.relative_path}:{s.actual_time:.3f}" for s in screenshots)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    budget = "all" if settings.ocr_budget is None else str(int(settings.ocr_budget))
    extras = ",".join(sorted(settings.extra_ocr_ids or []))
    return (
        f"{OCR_CACHE_VERSION}|ocr={settings.ocr}|budget={budget}|"
        f"policy={settings.sampling_policy}|extra={extras}|{digest}"
    )


def _stage_ocr(
    job: JobDir,
    screenshots: list[ScreenshotRecord],
    settings: ProcessSettings,
    progress: ProgressCb | None,
    clock: StageClock | None = None,
) -> str | None:
    key = _ocr_cache_key(settings, screenshots)
    cached = job.read_complete_stage("ocr", key)
    if cached:
        if clock:
            with clock.span("ocr", cache_hit=True):
                _progress(progress, "ocr", "Using cached on-screen text")
        else:
            _progress(progress, "ocr", "Using cached on-screen text")
        by_id = {row["id"]: row for row in cached.get("screenshots") or [] if row.get("id")}
        for record in screenshots:
            row = by_id.get(record.id)
            if not row:
                continue
            record.ocr_text = row.get("ocr_text")
            record.ocr_status = row.get("ocr_status") or "skipped"
            record.ocr_engine = row.get("ocr_engine")
            record.ocr_skip_reason = str(row.get("ocr_skip_reason") or "")
            record.ocr_reused_from = str(row.get("ocr_reused_from") or "")
            record.selected_for_ocr = bool(row.get("selected_for_ocr", record.ocr_status != "skipped"))
        if clock:
            clock.note("ocr_completed", sum(1 for row in screenshots if row.ocr_status in {"ok", "reused"}))
        return cached.get("note")
    selected_ids = select_ocr_ids(
        screenshots,
        budget=settings.ocr_budget if settings.ocr else 0,
        extra_ids=settings.extra_ocr_ids,
    )
    if not settings.ocr:
        selected_ids = set()
    if clock:
        clock.note("ocr_candidates", len(screenshots))
        clock.note("ocr_selected", len(selected_ids))
    job.write_stage(
        "ocr",
        key,
        "running",
        {"count": len(screenshots), "selected": len(selected_ids)},
    )
    checkpoint = job.cache / "ocr-checkpoint.json"
    if clock:
        with clock.span("ocr", cache_hit=False):
            note = apply_ocr(
                screenshots,
                job.root,
                settings.ocr,
                progress=progress,
                selected_ids=selected_ids,
                checkpoint_path=checkpoint,
            )
    else:
        note = apply_ocr(
            screenshots,
            job.root,
            settings.ocr,
            progress=progress,
            selected_ids=selected_ids,
            checkpoint_path=checkpoint,
        )
    reused = sum(1 for record in screenshots if record.ocr_status == "reused")
    completed = sum(1 for record in screenshots if record.ocr_status in {"ok", "reused", "failed", "unavailable"})
    if clock:
        clock.note("ocr_reused", reused)
        clock.note("ocr_completed", completed)
    job.write_stage(
        "ocr",
        key,
        "complete",
        {
            "note": note,
            "selected": len(selected_ids),
            "reused": reused,
            "completed": completed,
            "screenshots": [
                {
                    "id": record.id,
                    "ocr_text": record.ocr_text,
                    "ocr_status": record.ocr_status,
                    "ocr_engine": record.ocr_engine,
                    "ocr_skip_reason": record.ocr_skip_reason,
                    "ocr_reused_from": record.ocr_reused_from,
                    "selected_for_ocr": record.selected_for_ocr,
                }
                for record in screenshots
            ],
        },
    )
    return note
