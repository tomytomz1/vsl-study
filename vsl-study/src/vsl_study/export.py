"""Write transcripts, reports, 5-minute evidence folders, and ZIP."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import shutil
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from vsl_study.cache import JobDir, atomic_write_json, atomic_write_text
from vsl_study.frames import write_contact_sheet
from vsl_study.models import (
    ProcessSettings,
    Scene,
    ScreenshotRecord,
    TranscriptResult,
    TranscriptSegment,
    VideoInfo,
)
from vsl_study.ocr import format_onscreen_text
from vsl_study.timeutil import format_timecode, windows_5min

PACKAGED_MEDIA_DIR = "media"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
COPY_CHUNK = 1024 * 1024


def sha256_file(path: Path, chunk_size: int = COPY_CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def package_source_media(job: JobDir, source: str | Path) -> dict[str, Any]:
    """Copy the processed source video into the job folder without loading it all at once."""
    src = Path(source)
    if not src.is_file():
        raise RuntimeError(f"Source recording is missing: {src}")
    name = src.name
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        name = f"source{src.suffix.lower() or '.bin'}"
    dest_dir = job.root / PACKAGED_MEDIA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    expected = src.stat().st_size
    try:
        if tmp.exists():
            tmp.unlink()
        with src.open("rb") as inf, tmp.open("wb") as out:
            while True:
                block = inf.read(COPY_CHUNK)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
        if size != expected:
            raise RuntimeError(
                f"Copied {size} bytes but the source recording is {expected} bytes"
            )
        os.replace(tmp, dest)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    return {
        "relative_path": f"{PACKAGED_MEDIA_DIR}/{name}".replace("\\", "/"),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "original_name": src.name,
    }


def _capture_bool_label(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "unspecified"


def _capture_markdown_lines(capture: dict[str, Any]) -> list[str]:
    url = capture.get("source_url_user_supplied") or ""
    complete = capture.get("complete")
    lines = [
        "- Input: browser tab recording. Timestamps are relative to this recording, not the original video.",
        f"- Recording id: `{capture.get('recording_id')}`",
        f"- Stop reason: {capture.get('stop_reason')} · complete: {_capture_bool_label(complete)}",
    ]
    if url:
        lines.append(f"- Address the user typed (not proof of the selected tab): `{url}`")
    if capture.get("title"):
        lines.append(f"- Title: {capture.get('title')}")
    audio_track = capture.get("audio_track")
    audio_detected = capture.get("audio_detected")
    if audio_track is not None or audio_detected is not None:
        lines.append(
            "- Audio: "
            + (
                f"share-tab audio claimed={_capture_bool_label(audio_track)}, "
                f"detected={_capture_bool_label(audio_detected)}"
            )
        )
    if capture.get("started_at") or capture.get("ended_at"):
        lines.append(
            f"- Capture window: {capture.get('started_at') or '?'} → {capture.get('ended_at') or '?'}"
        )
    if capture.get("media_duration_s") is not None:
        lines.append(f"- Recorded media duration: {capture.get('media_duration_s')}s")
    requested = capture.get("capture_requested") if isinstance(capture.get("capture_requested"), dict) else {}
    observed = capture.get("capture_observed") if isinstance(capture.get("capture_observed"), dict) else {}
    if requested or observed or capture.get("capture_profile"):
        match = capture.get("capture_matches_profile")
        match_label = _capture_bool_label(match)
        lines.append(
            "- Capture profile: "
            f"{capture.get('capture_profile') or 'unspecified'} · matches requested study size: {match_label}"
        )
        if requested:
            lines.append(
                "- Requested capture: "
                f"width={requested.get('width')} frameRate={requested.get('frameRate')} "
                f"videoBitsPerSecond={requested.get('videoBitsPerSecond')}"
            )
        if observed:
            lines.append(
                "- Observed capture track: "
                f"width={observed.get('width')} height={observed.get('height')} "
                f"frameRate={observed.get('frameRate')}"
            )
        if match is False:
            lines.append("- This file is not labeled as an optimized study recording.")
        if capture.get("capture_constraint_error"):
            lines.append(f"- Capture constraint note: {capture.get('capture_constraint_error')}")
    if complete is False:
        lines.append("- This recording may cover only part of the video.")
    if capture.get("timeline_note"):
        lines.append(f"- {capture.get('timeline_note')}")
    problems = capture.get("problems")
    if isinstance(problems, list):
        for problem in problems:
            if isinstance(problem, str) and problem.strip():
                lines.append(f"- Capture note: {problem}")
    return lines


def _capture_html_items(capture: dict[str, Any]) -> str:
    return "".join(f"<li>{html.escape(line.lstrip('- ').strip())}</li>" for line in _capture_markdown_lines(capture))


def write_transcripts(job: JobDir, transcript: TranscriptResult) -> None:
    if transcript.status not in {"complete"}:
        note = transcript.error or "Transcription unavailable."
        atomic_write_text(
            job.root / "transcript.txt",
            f"[transcription {transcript.status}] {note}\n",
        )
        atomic_write_text(
            job.root / "transcript_timestamped.txt",
            f"[transcription {transcript.status}] {note}\n",
        )
        atomic_write_text(job.root / "transcript.srt", "")
        atomic_write_json(
            job.root / "transcript.json",
            {
                "status": transcript.status,
                "error": transcript.error,
                "model": transcript.model,
                "language": transcript.language,
                "segments": [],
                "flags": transcript.flags,
                "diagnostics_note": transcript.diagnostics_note,
            },
        )
        return

    atomic_write_text(job.root / "transcript.txt", (transcript.text + "\n") if transcript.text else "")
    lines = [
        f"{seg.id}  {format_timecode(seg.start)} --> {format_timecode(seg.end)}  {seg.text}"
        + (f"  [{', '.join(seg.flags)}]" if seg.flags else "")
        for seg in transcript.segments
    ]
    atomic_write_text(job.root / "transcript_timestamped.txt", "\n".join(lines) + ("\n" if lines else ""))
    atomic_write_text(job.root / "transcript.srt", _to_srt(transcript.segments))
    atomic_write_json(
        job.root / "transcript.json",
        {
            "status": transcript.status,
            "model": transcript.model,
            "language": transcript.language,
            "language_source": transcript.language_source,
            "device": transcript.device,
            "device_note": transcript.device_note,
            "diagnostics_note": transcript.diagnostics_note,
            "flags": transcript.flags,
            "text": transcript.text,
            "segments": [s.to_dict() for s in transcript.segments],
        },
    )


def _to_srt(segments: list[TranscriptSegment]) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = format_timecode(seg.start).replace(".", ",")
        end = format_timecode(seg.end).replace(".", ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
    return "\n".join(blocks)


def write_scenes_csv(job: JobDir, scenes: list[Scene]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["scene_id", "start", "end", "start_timecode", "end_timecode", "start_frame", "end_frame"])
    for scene in scenes:
        writer.writerow(
            [
                scene.id,
                f"{scene.start:.3f}",
                f"{scene.end:.3f}",
                format_timecode(scene.start),
                format_timecode(scene.end),
                scene.start_frame if scene.start_frame is not None else "",
                scene.end_frame if scene.end_frame is not None else "",
            ]
        )
    atomic_write_text(job.root / "scenes.csv", buffer.getvalue())


def write_ai_prompt(job: JobDir) -> None:
    text = files("vsl_study").joinpath("ai_study_prompt.md").read_text(encoding="utf-8")
    atomic_write_text(job.root / "ai_study_prompt.md", text)


def write_five_minute_folders(
    job: JobDir,
    info: VideoInfo,
    transcript: TranscriptResult,
    screenshots: list[ScreenshotRecord],
) -> list[str]:
    if job.clips.exists():
        shutil.rmtree(job.clips)
    job.clips.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for start, end, name in windows_5min(info.duration_s):
        folder = job.clips / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frames").mkdir(exist_ok=True)
        segs = [
            s
            for s in transcript.segments
            if s.start < end and s.end > start
        ]
        shots = [s for s in screenshots if start <= s.actual_time < end or (end == info.duration_s and s.actual_time == end)]
        # Boundary-spanning transcript is included with the same IDs.
        stamp_lines = [
            f"{seg.id}  {format_timecode(seg.start)} --> {format_timecode(seg.end)}  {seg.text}"
            + ("  [spans_folder_boundary]" if seg.start < start or seg.end > end else "")
            for seg in segs
        ]
        text_body = " ".join(s.text for s in segs).strip()
        atomic_write_text(folder / "transcript.txt", (text_body + "\n") if text_body else "")
        atomic_write_text(
            folder / "transcript_timestamped.txt",
            "\n".join(stamp_lines) + ("\n" if stamp_lines else ""),
        )
        atomic_write_text(folder / "onscreen.txt", format_onscreen_text(shots))
        copied: list[dict[str, Any]] = []
        for shot in shots:
            src = job.root / shot.relative_path
            dest = folder / "frames" / Path(shot.relative_path).name
            if src.exists():
                dest.write_bytes(src.read_bytes())
            copied.append(
                {
                    **shot.to_dict(),
                    "relative_path": dest.relative_to(folder).as_posix(),
                }
            )
        atomic_write_json(
            folder / "frames_manifest.json",
            {
                "folder": name,
                "start": start,
                "end": end,
                "transcript_segment_ids": [s.id for s in segs],
                "screenshots": copied,
                "note": (
                    "Transcript segments that cross this folder's start or end are included "
                    "with their original IDs so duplicated boundary context is identifiable."
                ),
            },
        )
        created.append(name)
    return created


def _write_contact_sheet_safely(
    records: list[ScreenshotRecord],
    job: JobDir,
    dest: Path,
    gaps: list[dict[str, Any]],
    label: str,
) -> None:
    try:
        _paths, skipped = write_contact_sheet(records, job.root, dest)
    except Exception as exc:  # noqa: BLE001
        gaps.append(
            {
                "type": "contact_sheet",
                "status": "failed",
                "detail": f"{label}: {exc}",
            }
        )
        return
    if skipped:
        gaps.append(
            {
                "type": "contact_sheet",
                "status": "partial",
                "detail": (
                    f"Skipped {len(skipped)} unreadable screenshot(s) while writing {label}: "
                    + ", ".join(skipped[:24])
                ),
                "ids": skipped,
            }
        )


def write_reports(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    transcript: TranscriptResult,
    scenes: list[Scene],
    screenshots: list[ScreenshotRecord],
    gaps: list[dict[str, Any]],
    versions: dict[str, str],
    stage_status: dict[str, Any],
    packaged_media: dict[str, Any] | None = None,
) -> None:
    _write_contact_sheet_safely(
        screenshots,
        job,
        job.contact_sheets / "all.jpg",
        gaps,
        label="contact_sheets/all.jpg",
    )
    retained = [s for s in screenshots if s.compact_retained]
    if settings.compact_view:
        _write_contact_sheet_safely(
            retained,
            job,
            job.contact_sheets / "compact.jpg",
            gaps,
            label="contact_sheets/compact.jpg",
        )
    _write_markdown(
        job, info, settings, transcript, scenes, screenshots, gaps, versions, stage_status, packaged_media
    )
    _write_html(
        job, info, settings, transcript, scenes, screenshots, gaps, versions, stage_status, packaged_media
    )


def _is_zip_output_path(path: Path, zip_path: Path, tmp_path: Path) -> bool:
    if path.name in {zip_path.name, tmp_path.name}:
        return True
    try:
        resolved = path.resolve()
        return resolved == zip_path.resolve() or resolved == tmp_path.resolve()
    except OSError:
        return False


def _validate_evidence_zip(tmp_path: Path, *, include_media: bool, packaged_rel: str) -> None:
    with zipfile.ZipFile(tmp_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"The rebuilt evidence ZIP failed an integrity check at {bad}.")
        names = zf.namelist()
    nested = {tmp_path.name, "vsl_study_evidence.zip"}
    if any(Path(name).name in nested for name in names):
        raise RuntimeError("The rebuilt evidence ZIP includes a nested copy of the archive itself.")
    if include_media and packaged_rel not in names:
        raise RuntimeError(
            "The evidence ZIP was written without the source recording. The package is incomplete."
        )


def write_zip(
    job: JobDir,
    include_media: bool,
    packaged_media: dict[str, Any] | None = None,
) -> Path:
    zip_path = job.root / "vsl_study_evidence.zip"
    tmp_path = zip_path.with_name(zip_path.name + ".part")
    packaged_rel = ""
    if include_media:
        if not packaged_media or not packaged_media.get("relative_path"):
            raise RuntimeError(
                "Include media was requested, but the source recording was not copied into the package."
            )
        packaged_rel = str(packaged_media["relative_path"]).replace("\\", "/")
        media_file = job.root / packaged_rel
        if not media_file.is_file():
            raise RuntimeError(
                f"Include media was requested, but {packaged_rel} is missing from the job folder."
            )
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in job.root.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(job.root)
                parts = rel.parts
                posix = rel.as_posix()
                if parts[0] == "work":
                    continue
                if _is_zip_output_path(path, zip_path, tmp_path):
                    continue
                if posix == "cache/audio.wav" and not include_media:
                    continue
                if parts[0] == PACKAGED_MEDIA_DIR and not include_media:
                    continue
                suffix = path.suffix.lower()
                if suffix in VIDEO_SUFFIXES and "frames" not in parts:
                    if not include_media or posix != packaged_rel:
                        continue
                compress = zipfile.ZIP_STORED if suffix in VIDEO_SUFFIXES else zipfile.ZIP_DEFLATED
                zf.write(path, posix, compress_type=compress)
        _validate_evidence_zip(tmp_path, include_media=include_media, packaged_rel=packaged_rel)
        os.replace(tmp_path, zip_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return zip_path


def _write_markdown(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    transcript: TranscriptResult,
    scenes: list[Scene],
    screenshots: list[ScreenshotRecord],
    gaps: list[dict[str, Any]],
    versions: dict[str, str],
    stage_status: dict[str, Any],
    packaged_media: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = [
        "# VSL Study report",
        "",
        f"- Source: `{info.resolved_path}`",
        f"- Duration: {format_timecode(info.duration_s)} ({info.duration_s:.3f}s container/audio)",
        f"- Dimensions: {info.width}x{info.height}",
        f"- Audio: {'yes' if info.has_audio else 'no'}",
        f"- Model: {transcript.model} / language: {transcript.language} / device: {transcript.device}",
        f"- Detector: {settings.detector} / interval: {settings.interval}s",
        f"- Sampling: {settings.sampling_policy} / max automatic screenshots: {settings.max_auto_screenshots}",
        f"- OCR budget: {settings.ocr_budget}",
        f"- Transcription backend: {transcript.backend or settings.transcribe_backend} / {transcript.compute_type or settings.transcribe_compute_type}",
        "",
        "## Processing notes",
        "",
    ]
    if packaged_media and packaged_media.get("relative_path"):
        lines.append(
            f"- Recording inside this package: `{packaged_media['relative_path']}` "
            "(use this relative path after unzipping; it does not depend on the original computer path)"
        )
        if packaged_media.get("sha256"):
            lines.append(f"- Packaged recording SHA-256: `{packaged_media['sha256']}`")
    if info.video_duration_s:
        lines.append(f"- Video stream duration: {info.video_duration_s:.3f}s (screenshots use this bound, not audio length)")
    if not info.fps_trusted:
        lines.append("- Container FPS metadata is not treated as a measured frame rate.")
    if transcript.status != "complete":
        lines.append(f"- Transcription {transcript.status}: {transcript.error}")
    if transcript.device_note:
        lines.append(f"- Device: {transcript.device_note}")
    capture = getattr(settings, "capture", None)
    if capture:
        lines.extend(_capture_markdown_lines(capture))
    lines.append("- Timestamp labels are in captions, not burned into screenshot pixels.")
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    if not gaps:
        lines.append("- None recorded.")
    for gap in gaps:
        lines.append(f"- `{gap}`")
    lines.append("")
    lines.append("## Scenes")
    lines.append("")
    for scene in scenes:
        lines.append(f"- {scene.id}: {format_timecode(scene.start)} – {format_timecode(scene.end)}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    if transcript.status != "complete":
        lines.append(f"Transcription is **{transcript.status}**. {transcript.error or ''}")
    for seg in transcript.segments:
        flag = f" _{', '.join(seg.flags)}_" if seg.flags else ""
        lines.append(f"- `{seg.id}` {format_timecode(seg.start)}–{format_timecode(seg.end)}: {seg.text}{flag}")
    lines.append("")
    lines.append("## On-screen text (OCR)")
    lines.append("")
    lines.append("This layer is separate from spoken transcript. OCR is not calibrated accuracy.")
    lines.append("")
    ocr_hits = [s for s in screenshots if s.ocr_status in {"ok", "reused"} and s.ocr_text]
    skipped_policy = [s for s in screenshots if s.ocr_skip_reason == "sampling_policy"]
    if any(s.ocr_status == "skipped" and s.ocr_skip_reason == "disabled" for s in screenshots) and not ocr_hits:
        lines.append("OCR was turned off for this job. Re-run with OCR enabled to capture slides, prices, and captions.")
    elif any(s.ocr_status == "unavailable" for s in screenshots) and not ocr_hits:
        lines.append("OCR engines were unavailable. Install RapidOCR (`rapidocr-onnxruntime`) or Tesseract.")
    elif not ocr_hits and skipped_policy:
        lines.append(
            "No on-screen text was detected in the OCR-selected frames. "
            "Other screenshots were skipped by sampling policy and were not checked for text."
        )
    elif not ocr_hits:
        lines.append("No on-screen text was detected in the captured frames.")
    else:
        lines.append(
            f"OCR ran on {sum(1 for s in screenshots if s.selected_for_ocr)} of {len(screenshots)} screenshots. "
            "This is sampled coverage, not an exhaustive still-by-still read."
        )
    for shot in ocr_hits:
        engine = f" _{shot.ocr_engine}_" if shot.ocr_engine else ""
        block = shot.ocr_text.replace("\n", " / ")
        lines.append(f"- `{shot.id}` {format_timecode(shot.actual_time)}{engine}: {block}")
    lines.append("")
    lines.append("## Screenshots")
    lines.append("")
    lines.append("A still does not illustrate an entire long spoken passage. Nearby segment IDs are time matches only.")
    lines.append("")
    for shot in screenshots:
        rel = shot.relative_path
        ocr = ""
        if shot.ocr_status == "ok" and shot.ocr_text:
            ocr = f"\n  - On-screen text (OCR): {shot.ocr_text}"
        elif shot.ocr_status in {"failed", "unavailable"}:
            ocr = f"\n  - On-screen text (OCR): {shot.ocr_status}"
        compact = ""
        if not shot.compact_retained:
            compact = f" (compact view refers to `{shot.compact_refers_to}`)"
        lines.append(
            f"- `{shot.id}` {format_timecode(shot.actual_time)} ({shot.capture_reason}, {shot.scene_id})"
            f"{compact}"
        )
        lines.append(f"  - requested {format_timecode(shot.requested_time)}; nearby {shot.nearby_segment_ids}")
        lines.append(f"  - ![{shot.id}]({rel}){ocr}")
    lines.append("")
    lines.append("## Versions")
    lines.append("")
    for key, value in versions.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Stage status")
    lines.append("")
    lines.append(f"```json\n{json.dumps(stage_status, indent=2)}\n```")
    atomic_write_text(job.root / "report.md", "\n".join(lines) + "\n")


def _write_html(
    job: JobDir,
    info: VideoInfo,
    settings: ProcessSettings,
    transcript: TranscriptResult,
    scenes: list[Scene],
    screenshots: list[ScreenshotRecord],
    gaps: list[dict[str, Any]],
    versions: dict[str, str],
    stage_status: dict[str, Any],
    packaged_media: dict[str, Any] | None = None,
) -> None:
    shot_by_time = screenshots
    events: list[str] = []
    if transcript.status != "complete":
        events.append(
            _event(
                0.0,
                "gap",
                f"Transcription {html.escape(transcript.status)}",
                html.escape(transcript.error or ""),
            )
        )
    for seg in transcript.segments:
        nearby_imgs = [
            s
            for s in shot_by_time
            if s.id in {i for i in s.nearby_segment_ids} or seg.id in s.nearby_segment_ids
        ]
        imgs = "".join(
            f'<figure><img src="{html.escape(s.relative_path)}" alt="{html.escape(s.id)}">'
            f"<figcaption>{html.escape(s.id)} {html.escape(format_timecode(s.actual_time))}</figcaption></figure>"
            for s in nearby_imgs[:4]
        )
        flags = ", ".join(seg.flags)
        events.append(
            _event(
                seg.start,
                "speech",
                f"{seg.id} {format_timecode(seg.start)}–{format_timecode(seg.end)}",
                f"<p>{html.escape(seg.text)}</p>"
                + (f"<p class='flags'>flags: {html.escape(flags)}</p>" if flags else "")
                + imgs,
            )
        )
    for shot in screenshots:
        ocr_html = ""
        if shot.ocr_status in {"ok", "reused"} and shot.ocr_text:
            reused = (
                f" (reused from {html.escape(shot.ocr_reused_from)})"
                if shot.ocr_status == "reused" and shot.ocr_reused_from
                else ""
            )
            ocr_html = (
                f"<p class='ocr'><strong>On-screen text (OCR)</strong>: "
                f"{html.escape(shot.ocr_text)}{reused}</p>"
            )
        elif shot.ocr_status in {"failed", "unavailable"}:
            ocr_html = f"<p class='ocr'>OCR {html.escape(shot.ocr_status)}</p>"
        elif shot.ocr_skip_reason == "sampling_policy":
            ocr_html = "<p class='ocr'>OCR skipped by sampling policy</p>"
        events.append(
            _event(
                shot.actual_time,
                "frame",
                f"{shot.id} {format_timecode(shot.actual_time)} ({shot.capture_reason})",
                f'<figure><img src="{html.escape(shot.relative_path)}" alt="{html.escape(shot.id)}">'
                f"<figcaption>requested {html.escape(format_timecode(shot.requested_time))} "
                f"scene {html.escape(shot.scene_id or 'n/a')} nearby {html.escape(', '.join(shot.nearby_segment_ids) or 'none')}</figcaption></figure>"
                + ocr_html,
            )
        )
    for gap in gaps:
        events.append(
            _event(
                float(gap.get("start") or gap.get("actual_time") or 0.0),
                "gap",
                html.escape(str(gap.get("type"))),
                f"<pre>{html.escape(json.dumps(gap, indent=2))}</pre>",
            )
        )

    notes = "".join(f"<li>{html.escape(n)}</li>" for n in info.notes)
    if packaged_media and packaged_media.get("relative_path"):
        notes += (
            f"<li>Recording inside this package: "
            f"<code>{html.escape(str(packaged_media['relative_path']))}</code> "
            "(use this relative path after unzipping)</li>"
        )
        if packaged_media.get("sha256"):
            notes += (
                f"<li>Packaged recording SHA-256: "
                f"<code>{html.escape(str(packaged_media['sha256']))}</code></li>"
            )
    capture = getattr(settings, "capture", None)
    if capture:
        notes += _capture_html_items(capture)
    gap_note = "".join(f"<li><pre>{html.escape(json.dumps(g))}</pre></li>" for g in gaps)
    versions_html = "".join(f"<li>{html.escape(k)}: {html.escape(v)}</li>" for k, v in versions.items())
    ocr_hits = [s for s in screenshots if s.ocr_status in {"ok", "reused"} and s.ocr_text]
    skipped_policy = any(s.ocr_skip_reason == "sampling_policy" for s in screenshots)
    if ocr_hits:
        ocr_list = "".join(
            f"<li><code>{html.escape(s.id)}</code> {html.escape(format_timecode(s.actual_time))}"
            f" — {html.escape(s.ocr_text.replace(chr(10), ' / '))}</li>"
            for s in ocr_hits
        )
        coverage = (
            f"<p>OCR ran on {sum(1 for s in screenshots if s.selected_for_ocr)} of "
            f"{len(screenshots)} screenshots. This is sampled coverage, not exhaustive.</p>"
        )
        ocr_section = f"<h2>On-screen text (OCR)</h2>{coverage}<ul>{ocr_list}</ul>"
    elif any(s.ocr_status == "skipped" and s.ocr_skip_reason == "disabled" for s in screenshots):
        ocr_section = "<h2>On-screen text (OCR)</h2><p>OCR was turned off for this job.</p>"
    elif any(s.ocr_status == "unavailable" for s in screenshots):
        ocr_section = "<h2>On-screen text (OCR)</h2><p>OCR engines were unavailable.</p>"
    elif skipped_policy:
        ocr_section = (
            "<h2>On-screen text (OCR)</h2>"
            "<p>No on-screen text was detected in the OCR-selected frames. "
            "Other screenshots were skipped by sampling policy and were not checked for text.</p>"
        )
    else:
        ocr_section = "<h2>On-screen text (OCR)</h2><p>No on-screen text was detected in the captured frames.</p>"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VSL Study report</title>
<style>
body {{ font-family: Georgia, serif; max-width: 960px; margin: 1.5rem auto; padding: 0 1rem; color: #111; }}
input#q {{ width: 100%; padding: 0.5rem; margin: 1rem 0; font-size: 1rem; }}
.event {{ border-top: 1px solid #ddd; padding: 0.75rem 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
figcaption {{ font-size: 0.85rem; color: #333; }}
.flags, .ocr {{ font-size: 0.9rem; color: #444; }}
.kind-gap {{ background: #fff6e8; }}
code, pre {{ font-family: Consolas, monospace; }}
</style>
</head>
<body>
<h1>VSL Study report</h1>
<p>Offline timeline. Images are relative files next to this HTML. No remote scripts.</p>
<p>Processed from <code>{html.escape(info.resolved_path)}</code><br>
{("Recording in this package: <code>" + html.escape(str(packaged_media.get("relative_path"))) + "</code><br>") if packaged_media and packaged_media.get("relative_path") else ""}
Duration {html.escape(format_timecode(info.duration_s))} · {info.width}x{info.height} ·
audio {"yes" if info.has_audio else "no"} · model {html.escape(str(transcript.model))} ·
device {html.escape(str(transcript.device))}</p>
<p>{html.escape(transcript.device_note or "")}</p>
<h2>Processing notes</h2>
<ul>{notes or "<li>None</li>"}</ul>
<h2>Gaps</h2>
<ul>{gap_note or "<li>None recorded</li>"}</ul>
{ocr_section}
<p>Search filters this page locally:</p>
<input id="q" type="search" placeholder="Search transcript, OCR, frame IDs">
<div id="timeline">
{''.join(events)}
</div>
<h2>Versions</h2>
<ul>{versions_html}</ul>
<pre>{html.escape(json.dumps(stage_status, indent=2))}</pre>
<script>
document.getElementById('q').addEventListener('input', function () {{
  var q = this.value.toLowerCase();
  document.querySelectorAll('.event').forEach(function (el) {{
    el.style.display = !q || el.innerText.toLowerCase().indexOf(q) !== -1 ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""
    atomic_write_text(job.root / "report.html", page)


def _event(when: float, kind: str, title: str, body: str) -> str:
    return (
        f'<article class="event kind-{html.escape(kind)}" data-t="{when:.3f}">'
        f"<h3>{title}</h3>{body}</article>"
    )
