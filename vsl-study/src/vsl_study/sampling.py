"""Screenshot and OCR sampling policies for new jobs.

These caps are initial tuning values, not proven optimal settings.
Sampled evidence is not exhaustive.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, TypeVar

from vsl_study.models import ScreenshotRecord

STANDARD_POLICY = "standard-v1"
DENSE_POLICY = "dense-v1"
LEGACY_POLICY = "legacy"

STANDARD_INTERVAL_S = 15.0
DENSE_INTERVAL_S = 5.0
STANDARD_MAX_AUTO_SCREENSHOTS = 600
DENSE_MAX_AUTO_SCREENSHOTS = 2000
STANDARD_OCR_BUDGET = 300
DENSE_OCR_BUDGET = 800
FASTER_MAX_AUTO_SCREENSHOTS = 400
FASTER_OCR_BUDGET = 200

NEAR_S = 0.25
OCR_NEAR_S = 0.05
END_ZONE_FRACTION = 0.88
MIN_TRAILING_SILENCE_S = 90.0
END_CARD_PAD_S = 30.0
MIN_SPEECH_CHARS = 8
ISLAND_GAP_S = 300.0

PRIORITY = {"user": 0, "scene_start": 1, "interval": 2}

T = TypeVar("T")


class TimedCandidate(Protocol):
    requested_time: float
    reason: str


def sampling_for_quality(quality: str) -> dict[str, float | int | str | None]:
    name = (quality or "recommended").strip().lower()
    if name == "accurate":
        return {
            "interval": DENSE_INTERVAL_S,
            "max_auto_screenshots": DENSE_MAX_AUTO_SCREENSHOTS,
            "ocr_budget": DENSE_OCR_BUDGET,
            "sampling_policy": DENSE_POLICY,
        }
    if name == "faster":
        return {
            "interval": STANDARD_INTERVAL_S,
            "max_auto_screenshots": FASTER_MAX_AUTO_SCREENSHOTS,
            "ocr_budget": FASTER_OCR_BUDGET,
            "sampling_policy": STANDARD_POLICY,
        }
    return {
        "interval": STANDARD_INTERVAL_S,
        "max_auto_screenshots": STANDARD_MAX_AUTO_SCREENSHOTS,
        "ocr_budget": STANDARD_OCR_BUDGET,
        "sampling_policy": STANDARD_POLICY,
    }


def _prefer(existing: TimedCandidate, incoming: TimedCandidate) -> TimedCandidate:
    if PRIORITY.get(incoming.reason, 9) < PRIORITY.get(existing.reason, 9):
        return incoming
    return existing


def _near(time_s: float, others: Iterable[float], window: float) -> bool:
    return any(abs(time_s - other) <= window for other in others)


def _segment_text(segment: object) -> str:
    raw = getattr(segment, "text", None)
    if raw is None and isinstance(segment, dict):
        raw = segment.get("text")
    return " ".join(str(raw or "").split())


def _segment_end(segment: object) -> float:
    raw = getattr(segment, "end", None)
    if raw is None and isinstance(segment, dict):
        raw = segment.get("end")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def last_substantial_speech_end(segments: Sequence[object] | None) -> float | None:
    """Last real spoken end time. Ignores tiny leftover words after a long gap."""
    ends: list[float] = []
    for segment in segments or []:
        compact = "".join(ch for ch in _segment_text(segment) if ch.isalnum())
        if len(compact) < MIN_SPEECH_CHARS:
            continue
        ends.append(_segment_end(segment))
    if not ends:
        return None
    if len(ends) >= 2 and (ends[-1] - ends[-2]) >= ISLAND_GAP_S:
        return ends[-2]
    return ends[-1]


def sampling_horizon(duration_s: float, speech_end_s: float | None) -> tuple[float, float | None]:
    """Screenshot end time, plus leftover seconds if the recording continued after speech."""
    duration = max(float(duration_s or 0.0), 0.0)
    if speech_end_s is None:
        return duration, None
    speech_end = max(0.0, float(speech_end_s))
    tail = duration - speech_end
    if tail < MIN_TRAILING_SILENCE_S:
        return duration, None
    horizon = min(duration, speech_end + END_CARD_PAD_S)
    leftover = max(0.0, duration - horizon)
    return horizon, leftover if leftover >= 1.0 else None


def select_screenshots(
    candidates: Sequence[TimedCandidate],
    *,
    duration_s: float,
    max_auto: int | None,
) -> list[TimedCandidate]:
    """Keep opening/ending coverage, spread the automatic budget, and leave manuals outside it."""
    manuals = [item for item in candidates if item.reason == "user"]
    autos = [item for item in candidates if item.reason != "user"]
    duration = max(float(duration_s or 0.0), 0.001)
    autos = [item for item in autos if item.requested_time <= duration + NEAR_S]
    if max_auto is None or max_auto <= 0 or len(autos) <= max_auto:
        return _merge_unique(list(autos) + list(manuals))

    autos_sorted = sorted(autos, key=lambda item: (item.requested_time, PRIORITY.get(item.reason, 9)))
    selected: list[TimedCandidate] = []

    def selected_times() -> list[float]:
        return [item.requested_time for item in selected]

    def take(item: TimedCandidate) -> None:
        for index, existing in enumerate(selected):
            if abs(item.requested_time - existing.requested_time) <= NEAR_S:
                selected[index] = _prefer(existing, item)
                return
        selected.append(item)

    take(autos_sorted[0])
    take(autos_sorted[-1])

    duration = max(float(duration_s or 0.0), autos_sorted[-1].requested_time, 0.001)
    end_zone = duration * END_ZONE_FRACTION
    pool = [item for item in autos_sorted if not _near(item.requested_time, selected_times(), NEAR_S)]

    while len(selected) < max_auto and pool:
        best: TimedCandidate | None = None
        best_score = -1.0
        for item in pool:
            distance = min(abs(item.requested_time - other.requested_time) for other in selected)
            score = distance
            if item.reason == "scene_start":
                score += 0.05
            if item.requested_time >= end_zone:
                score += 0.12
            if score > best_score:
                best_score = score
                best = item
        if best is None:
            break
        take(best)
        pool = [item for item in pool if not _near(item.requested_time, selected_times(), NEAR_S)]

    return _merge_unique(selected + manuals)


def _merge_unique(candidates: Sequence[TimedCandidate]) -> list[TimedCandidate]:
    ordered: list[TimedCandidate] = []
    for item in sorted(candidates, key=lambda row: (row.requested_time, PRIORITY.get(row.reason, 9))):
        matched = False
        for index, existing in enumerate(ordered):
            if abs(item.requested_time - existing.requested_time) <= NEAR_S:
                ordered[index] = _prefer(existing, item)
                matched = True
                break
        if not matched:
            ordered.append(item)
    return ordered


def select_ocr_ids(
    records: Sequence[ScreenshotRecord],
    *,
    budget: int | None,
    extra_ids: Sequence[str] | None = None,
) -> set[str]:
    """Pick OCR targets across the timeline. User frames and extra_ids are outside the automatic budget."""
    extras = {item for item in (extra_ids or []) if item}
    always = {record.id for record in records if record.capture_reason == "user"} | extras
    if budget is None or budget <= 0:
        return {record.id for record in records}

    ordered = sorted(records, key=lambda record: (record.actual_time, record.id))
    if len(ordered) <= budget:
        return {record.id for record in ordered} | always

    chosen: list[ScreenshotRecord] = []

    def chosen_times() -> list[float]:
        return [item.actual_time for item in chosen]

    def take(record: ScreenshotRecord) -> None:
        if any(item.id == record.id for item in chosen):
            return
        if _near(record.actual_time, chosen_times(), OCR_NEAR_S):
            return
        chosen.append(record)

    take(ordered[0])
    take(ordered[-1])
    duration = max(ordered[-1].actual_time, 0.001)
    end_zone = duration * END_ZONE_FRACTION
    pool = [item for item in ordered if item.id not in {row.id for row in chosen}]

    while len(chosen) < budget and pool:
        best: ScreenshotRecord | None = None
        best_score = -1.0
        for item in pool:
            distance = min(abs(item.actual_time - other.actual_time) for other in chosen)
            score = distance
            if item.capture_reason == "scene_start":
                score += 0.05
            if item.actual_time >= end_zone:
                score += 0.12
            if score > best_score:
                best_score = score
                best = item
        if best is None:
            break
        take(best)
        pool = [item for item in pool if item.id not in {row.id for row in chosen}]

    return {item.id for item in chosen} | always


def coverage_note(
    *,
    policy: str,
    interval: float,
    max_auto: int | None,
    ocr_budget: int | None,
    leftover_s: float | None = None,
    horizon_s: float | None = None,
) -> str:
    if policy == LEGACY_POLICY or max_auto is None:
        note = (
            f"Screenshots use the stored job policy (interval {interval:g}s, no automatic cap). "
            "This is not a new bounded-sampling run."
        )
    else:
        ocr = "all selected screenshots" if ocr_budget is None else f"up to {int(ocr_budget)} automatic images"
        note = (
            f"Screenshots are sampled ({policy}: about every {interval:g}s plus scene changes, "
            f"max {int(max_auto)} automatic stills). OCR covers {ocr}. "
            "This is not an exhaustive frame-by-frame record."
        )
    if leftover_s and horizon_s is not None:
        note += (
            f" Spoken content ended around {horizon_s:.0f}s; automatic screenshots stop there "
            f"instead of covering {leftover_s:.0f}s of ended-video leftover."
        )
    return note
