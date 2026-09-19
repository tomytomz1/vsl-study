from vsl_study.frames import CaptureCandidate
from vsl_study.models import ProcessSettings, ScreenshotRecord, settings_from_stored
from vsl_study.sampling import select_ocr_ids, select_screenshots


def _shot(i: int, t: float, reason: str = "interval") -> ScreenshotRecord:
    return ScreenshotRecord(
        id=f"frame_{i:04d}",
        requested_time=t,
        actual_time=t,
        scene_id="scene_0001",
        relative_path=f"frames/frame_{i:04d}.jpg",
        capture_reason=reason,
    )


def test_screenshot_budget_keeps_opening_ending_and_manuals():
    duration = 72 * 60
    autos = [CaptureCandidate(float(t), "interval", "scene_0001") for t in range(0, int(duration) + 1, 5)]
    autos.append(CaptureCandidate(100.0, "scene_start", "scene_0002"))
    autos.append(CaptureCandidate(duration - 8.0, "scene_start", "scene_0009"))
    manuals = [CaptureCandidate(333.3, "user", "scene_0001"), CaptureCandidate(4000.0, "user", "scene_0001")]
    selected = select_screenshots(autos + manuals, duration_s=duration, max_auto=40)
    auto_selected = [item for item in selected if item.reason != "user"]
    assert len(auto_selected) <= 40
    times = [item.requested_time for item in selected]
    assert min(times) == 0.0
    assert max(item.requested_time for item in auto_selected) >= duration - 6
    assert any(item.reason == "user" and abs(item.requested_time - 333.3) < 0.01 for item in selected)
    assert any(item.reason == "user" and abs(item.requested_time - 4000.0) < 0.01 for item in selected)
    assert len(selected) == len(auto_selected) + 2


def test_screenshot_budget_spreads_across_timeline():
    duration = 3600.0
    autos = [CaptureCandidate(float(t), "interval", None) for t in range(0, 3601, 5)]
    selected = select_screenshots(autos, duration_s=duration, max_auto=30)
    times = [item.requested_time for item in selected]
    assert times[0] == 0.0
    assert times[-1] >= 3590
    early = sum(1 for t in times if t < 600)
    late = sum(1 for t in times if t >= 3000)
    assert early < 20
    assert late >= 3


def test_ocr_selection_covers_open_end_and_keeps_manuals():
    records = [_shot(i, float(i * 15)) for i in range(80)]
    records[10].capture_reason = "user"
    records[-5].capture_reason = "scene_start"
    chosen = select_ocr_ids(records, budget=12)
    assert records[0].id in chosen
    assert records[-1].id in chosen
    assert records[10].id in chosen
    assert len(chosen) >= 12
    assert len(chosen) <= 13


def test_legacy_job_settings_do_not_apply_new_defaults():
    stored = {
        "model": "small.en",
        "language": "en",
        "task": "transcribe",
        "interval": 5.0,
        "ocr": True,
        "detector": "adaptive",
    }
    settings = settings_from_stored(stored)
    assert settings.sampling_policy == "legacy"
    assert settings.max_auto_screenshots is None
    assert settings.ocr_budget is None
    assert settings.interval == 5.0
    assert settings.transcribe_backend == "openai-whisper"
    assert settings.transcribe_key("fp") == "fp|small.en|en|transcribe"
    assert settings.frames_key("fp", []) == "fp|adaptive|5.000|0.250|1280||seq-pts-4"


def test_new_settings_version_cache_keys_and_ocr_does_not_touch_transcript():
    a = ProcessSettings()
    b = ProcessSettings(ocr_budget=50)
    assert a.interval == 15.0
    assert "faster-whisper" in a.transcribe_key("fp")
    assert "max=600" in a.frames_key("fp", [])
    assert a.transcribe_key("fp") == b.transcribe_key("fp")
    assert a.frames_key("fp", []) == b.frames_key("fp", [])
    from vsl_study.pipeline import _ocr_cache_key

    shots = [_shot(1, 0.0)]
    assert _ocr_cache_key(a, shots) != _ocr_cache_key(b, shots)
