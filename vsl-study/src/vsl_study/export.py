"""Write transcripts, reports, 5-minute evidence folders, and ZIP."""

from __future__ import annotations

import csv
import html
import io
import json
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


def _capture_markdown_lines(capture: dict[str, Any]) -> list[str]:
    url = capture.get("source_url_user_supplied") or ""
    lines = [
        "- Input: browser tab recording. Timestamps are relative to this recording, not the original video.",
        f"- Recording id: `{capture.get('recording_id')}`",
        f"- Stop reason: {capture.get('stop_reason')} · complete: {capture.get('complete')}",
    ]
    if url:
        lines.append(f"- Address the user typed (not proof of the selected tab): `{url}`")
    if capture.get("title"):
        lines.append(f"- Title: {capture.get('title')}")
    if capture.get("media_duration_s") is not None:
        lines.append(f"- Recorded media duration: {capture.get('media_duration_s')}s")
    if not capture.get("complete"):
        lines.append("- This recording may cover only part of the video.")
    if capture.get("timeline_note"):
        lines.append(f"- {capture.get('timeline_note')}")
    for problem in capture.get("problems") or []:
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
) -> None:
    write_contact_sheet(screenshots, job.root, job.contact_sheets / "all.jpg")
    retained = [s for s in screenshots if s.compact_retained]
    if settings.compact_view:
        write_contact_sheet(retained, job.root, job.contact_sheets / "compact.jpg")
    _write_markdown(
        job, info, settings, transcript, scenes, screenshots, gaps, versions, stage_status
    )
    _write_html(
        job, info, settings, transcript, scenes, screenshots, gaps, versions, stage_status
    )


def write_zip(job: JobDir, include_media: bool) -> Path:
    zip_path = job.root / "vsl_study_evidence.zip"
    skip_names = {zip_path.name}
    skip_dirs = {"work"}
    if not include_media:
        skip_dirs.add("work")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in job.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(job.root)
            parts = rel.parts
            if parts[0] in skip_dirs:
                continue
            if rel.name in skip_names:
                continue
            if not include_media and rel.as_posix() == "cache/audio.wav":
                continue
            if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} and "frames" not in parts:
                continue
            zf.write(path, rel.as_posix())
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
) -> None:
    lines: list[str] = [
        "# VSL Study report",
        "",
        f"- Source: `{info.resolved_path}`",
        f"- Duration: {format_timecode(info.duration_s)} ({info.duration_s:.3f}s)",
        f"- Dimensions: {info.width}x{info.height}",
        f"- Audio: {'yes' if info.has_audio else 'no'}",
        f"- Model: {transcript.model} / language: {transcript.language} / device: {transcript.device}",
        f"- Detector: {settings.detector} / interval: {settings.interval}s",
        "",
        "## Processing notes",
        "",
    ]
    for note in info.notes:
        lines.append(f"- {note}")
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
    ocr_hits = [s for s in screenshots if s.ocr_status == "ok" and s.ocr_text]
    if any(s.ocr_status == "skipped" for s in screenshots) and not ocr_hits:
        lines.append("OCR was turned off for this job. Re-run with OCR enabled to capture slides, prices, and captions.")
    elif any(s.ocr_status == "unavailable" for s in screenshots) and not ocr_hits:
        lines.append("OCR engines were unavailable. Install RapidOCR (`rapidocr-onnxruntime`) or Tesseract.")
    elif not ocr_hits:
        lines.append("No on-screen text was detected in the captured frames.")
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
        if shot.ocr_status == "ok" and shot.ocr_text:
            ocr_html = f"<p class='ocr'><strong>On-screen text (OCR)</strong>: {html.escape(shot.ocr_text)}</p>"
        elif shot.ocr_status in {"failed", "unavailable"}:
            ocr_html = f"<p class='ocr'>OCR {html.escape(shot.ocr_status)}</p>"
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
    capture = getattr(settings, "capture", None)
    if capture:
        notes += _capture_html_items(capture)
    gap_note = "".join(f"<li><pre>{html.escape(json.dumps(g))}</pre></li>" for g in gaps)
    versions_html = "".join(f"<li>{html.escape(k)}: {html.escape(v)}</li>" for k, v in versions.items())
    ocr_hits = [s for s in screenshots if s.ocr_status == "ok" and s.ocr_text]
    if ocr_hits:
        ocr_list = "".join(
            f"<li><code>{html.escape(s.id)}</code> {html.escape(format_timecode(s.actual_time))}"
            f" — {html.escape(s.ocr_text.replace(chr(10), ' / '))}</li>"
            for s in ocr_hits
        )
        ocr_section = f"<h2>On-screen text (OCR)</h2><ul>{ocr_list}</ul>"
    elif any(s.ocr_status == "skipped" for s in screenshots):
        ocr_section = "<h2>On-screen text (OCR)</h2><p>OCR was turned off for this job.</p>"
    elif any(s.ocr_status == "unavailable" for s in screenshots):
        ocr_section = "<h2>On-screen text (OCR)</h2><p>OCR engines were unavailable.</p>"
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
<p>Source <code>{html.escape(info.resolved_path)}</code><br>
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
