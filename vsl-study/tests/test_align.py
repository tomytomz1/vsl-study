from vsl_study.align import match_screenshots
from vsl_study.models import ScreenshotRecord, TranscriptSegment
from vsl_study.timeutil import windows_5min


def test_overlap_and_context_window_match():
    segments = [
        TranscriptSegment(id="seg_0001", start=1.0, end=3.0, text="hello"),
        TranscriptSegment(id="seg_0002", start=10.0, end=12.0, text="later"),
    ]
    shots = [
        ScreenshotRecord(
            id="frame_0001",
            requested_time=1.5,
            actual_time=1.5,
            scene_id="scene_0001",
            relative_path="frames/a.jpg",
            capture_reason="interval",
        ),
        ScreenshotRecord(
            id="frame_0002",
            requested_time=6.5,
            actual_time=6.5,
            scene_id="scene_0001",
            relative_path="frames/b.jpg",
            capture_reason="interval",
        ),
        ScreenshotRecord(
            id="frame_0003",
            requested_time=9.5,
            actual_time=9.5,
            scene_id="scene_0002",
            relative_path="frames/c.jpg",
            capture_reason="interval",
        ),
    ]
    gaps = match_screenshots(shots, segments, context_window_s=2.0)
    assert shots[0].nearby_segment_ids == ["seg_0001"]
    assert shots[1].nearby_segment_ids == []
    assert "seg_0002" in shots[2].nearby_segment_ids
    types = {g["type"] for g in gaps}
    assert "unmatched_screenshot" in types
    assert "unmatched_period" in types


def test_boundary_segment_belongs_to_both_windows():
    windows = windows_5min(600)
    assert len(windows) == 2
    first = windows[0]
    second = windows[1]
    # Segment spanning 290-310 must overlap both [0,300) and [300,600]
    seg_start, seg_end = 290.0, 310.0
    assert seg_start < first[1] and seg_end > first[0]
    assert seg_start < second[1] and seg_end > second[0]
