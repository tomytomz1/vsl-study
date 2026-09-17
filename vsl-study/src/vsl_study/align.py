"""Align screenshots with transcript segments by time overlap or a context window."""

from __future__ import annotations

from vsl_study.models import ScreenshotRecord, TranscriptSegment
from vsl_study.timeutil import overlaps


CONTEXT_WINDOW_NOTE = (
    "A screenshot is matched to a transcript segment when its actual timestamp overlaps "
    "the segment [start, end), or falls within the configured context window of a segment "
    "boundary. A still does not illustrate an entire long spoken passage."
)


def match_screenshots(
    screenshots: list[ScreenshotRecord],
    segments: list[TranscriptSegment],
    context_window_s: float = 2.0,
) -> list[dict]:
    """Mutate screenshot nearby_segment_ids. Return unmatched / silent period records."""
    for shot in screenshots:
        nearby: list[str] = []
        t = shot.actual_time
        for seg in segments:
            if overlaps(t, t + 1e-3, seg.start, seg.end):
                nearby.append(seg.id)
            elif min(abs(t - seg.start), abs(t - seg.end)) <= context_window_s:
                if seg.id not in nearby:
                    nearby.append(seg.id)
        shot.nearby_segment_ids = nearby

    unmatched_shots = [s for s in screenshots if not s.nearby_segment_ids]
    silent_gaps = _silent_gaps(segments, screenshots)
    return [
        {
            "type": "unmatched_screenshot",
            "screenshot_id": s.id,
            "actual_time": s.actual_time,
        }
        for s in unmatched_shots
    ] + silent_gaps


def _silent_gaps(
    segments: list[TranscriptSegment],
    screenshots: list[ScreenshotRecord],
    min_gap: float = 3.0,
) -> list[dict]:
    if not segments:
        if screenshots:
            return [
                {
                    "type": "no_transcript",
                    "start": 0.0,
                    "end": max(s.actual_time for s in screenshots),
                    "note": "No spoken transcript segments are available for this span.",
                }
            ]
        return []
    ordered = sorted(segments, key=lambda s: s.start)
    gaps: list[dict] = []
    cursor = 0.0
    for seg in ordered:
        if seg.start - cursor >= min_gap:
            shot_ids = [
                s.id for s in screenshots if cursor <= s.actual_time < seg.start
            ]
            gaps.append(
                {
                    "type": "unmatched_period",
                    "start": cursor,
                    "end": seg.start,
                    "screenshot_ids": shot_ids,
                    "note": "No transcript segment overlaps this span.",
                }
            )
        cursor = max(cursor, seg.end)
    return gaps
