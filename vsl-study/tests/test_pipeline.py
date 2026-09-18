import json
import zipfile
from pathlib import Path

import pytest

from conftest import (
    mux_offset_speech,
    mux_speech_video,
    requires_ffmpeg,
    synthesize_speech_wav,
    write_color_video,
    write_cut_video,
    write_vfr_video,
)
from vsl_study.models import ProcessSettings
from vsl_study.pipeline import add_frames_at, process_video
from vsl_study.timeutil import parse_timecode


@requires_ffmpeg
def test_no_audio_is_not_fake_success(tmp_path: Path):
    video = write_color_video(tmp_path / "silent.mp4", duration=3.0, audio=False)
    out = tmp_path / "out"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.5, detector="content", compact_view=False, ocr=False),
    )
    assert result["transcript_status"] == "skipped"
    payload = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["segments"] == []
    text = (out / "transcript.txt").read_text(encoding="utf-8")
    assert "transcription skipped" in text.lower() or "unavailable" in text.lower()
    assert (out / "report.html").exists()
    assert list((out / "frames").glob("*.jpg"))
    assert result["scene_count"] == 1


@requires_ffmpeg
def test_single_scene_and_spaces_and_report_links(tmp_path: Path):
    video = write_color_video(tmp_path / "file with spaces.mp4", duration=2.5, color="white")
    out = tmp_path / "job spaces"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=5.0, detector="content", compact_view=True, max_width=320, ocr=False),
    )
    assert result["scene_count"] == 1
    html = (out / "report.html").read_text(encoding="utf-8")
    md = (out / "report.md").read_text(encoding="utf-8")
    frames = list((out / "frames").glob("*.jpg"))
    assert frames
    rel = frames[0].relative_to(out).as_posix()
    assert rel in html
    assert rel in md
    # Moving the folder keeps relative links.
    moved = tmp_path / "moved-job"
    out.rename(moved)
    html2 = (moved / "report.html").read_text(encoding="utf-8")
    assert rel in html2
    assert (moved / Path(rel)).exists()
    zpath = moved / "vsl_study_evidence.zip"
    extract = tmp_path / "unzipped"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract)
    assert (extract / Path(rel)).exists()
    assert 'src="' in (extract / "report.html").read_text(encoding="utf-8")


@requires_ffmpeg
def test_cut_video_captures_and_user_frames(tmp_path: Path):
    video = write_cut_video(tmp_path / "cuts.mp4")
    out = tmp_path / "cuts-out"
    process_video(
        video,
        out,
        settings=ProcessSettings(interval=2.0, detector="content", compact_view=False, ocr=False),
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scenes"]
    shots = manifest["screenshots"]
    assert len(shots) >= 3
    for shot in shots:
        assert (out / shot["relative_path"]).exists()
        assert shot["actual_time"] >= 0
        # Capture uses a real file; requested vs actual are both recorded.
        assert "requested_time" in shot
    before = {s["id"] for s in shots}
    add_frames_at(out, [parse_timecode("00:00:01.000")])
    after = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    reasons = {s["capture_reason"] for s in after["screenshots"]}
    assert "user" in reasons
    assert len(after["screenshots"]) >= len(before)


@requires_ffmpeg
def test_vfr_capture_records_actual_pts(tmp_path: Path):
    video = write_vfr_video(tmp_path / "vfr.mp4")
    out = tmp_path / "vfr-out"
    process_video(
        video,
        out,
        settings=ProcessSettings(interval=0.5, detector="content", compact_view=False, ocr=False),
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["screenshots"]
    for shot in manifest["screenshots"]:
        assert (out / shot["relative_path"]).stat().st_size > 0
        assert isinstance(shot["actual_time"], float)


@pytest.mark.slow
@requires_ffmpeg
def test_real_whisper_fixture(tmp_path: Path):
    wav = synthesize_speech_wav(tmp_path / "speech.wav")
    if wav is None:
        pytest.skip("Windows SAPI speech synthesis unavailable")
    video = mux_speech_video(tmp_path / "spoken.mp4", wav, duration=5.0)
    result = process_video(
        video,
        tmp_path / "spoken-out",
        settings=ProcessSettings(
            model="tiny.en",
            language="en",
            interval=2.0,
            detector="content",
            compact_view=False,
            device="cpu",
            ocr=False,
        ),
    )
    assert result["transcript_status"] in {"complete", "failed"}
    if result["transcript_status"] != "complete":
        pytest.fail("Whisper ran but did not complete; see job cache for the error.")
    data = json.loads((tmp_path / "spoken-out" / "transcript.json").read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["model"] == "tiny.en"
    assert data["device"] == "cpu"
    # Do not require exact wording; require some spoken text and timestamps.
    assert data["segments"]
    joined = " ".join(s["text"] for s in data["segments"]).strip()
    assert joined
    for seg in data["segments"]:
        assert seg["end"] >= seg["start"]
        assert "avg_logprob" in seg
    html = (tmp_path / "spoken-out" / "report.html").read_text(encoding="utf-8")
    assert "frame_" in html or "frames/" in html


@requires_ffmpeg
def test_offset_audio_fixture_inspectable(tmp_path: Path):
    wav = synthesize_speech_wav(tmp_path / "speech.wav", "Hello there.")
    if wav is None:
        pytest.skip("Windows SAPI speech synthesis unavailable")
    video = mux_offset_speech(tmp_path / "offset.mp4", wav)
    from vsl_study.media import inspect_video

    info = inspect_video(video)
    assert info.has_audio
    # Offset is recorded so extraction can pad; exact container start times vary by muxer.
    assert info.audio_start_s is not None


@requires_ffmpeg
def test_report_and_zip_after_transcription_failure(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "talk.mp4", duration=2.0, audio=True)
    from vsl_study import pipeline as pipeline_mod

    def boom(wav_path, settings, progress=None):
        raise RuntimeError("tqdm console exploded")

    monkeypatch.setattr(pipeline_mod, "transcribe_wav", boom)
    out = tmp_path / "failed-speech"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, detector="content", compact_view=False, ocr=False),
    )
    assert result["transcript_status"] == "failed"
    html = (out / "report.html").read_text(encoding="utf-8")
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "failed" in html.lower()
    assert "tqdm console exploded" in html
    assert "no speech detected" not in html.lower()
    assert "Transcription failed" in md or "transcription failed" in md.lower()
    assert (out / "vsl_study_evidence.zip").exists()
    assert list((out / "frames").glob("*.jpg"))
    job = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["transcribe"]["status"] == "failed"
    assert job["stages"]["frames"]["status"] == "complete"
    cache = json.loads((out / "cache" / "transcribe.json").read_text(encoding="utf-8"))
    assert cache["status"] == "failed"
    frames_cache = json.loads((out / "cache" / "frames.json").read_text(encoding="utf-8"))
    assert frames_cache["scheduled"] == frames_cache["completed"]
    assert frames_cache["status"] == "complete"


@requires_ffmpeg
def test_failed_transcription_is_retryable(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "retry.mp4", duration=1.5, audio=True)
    from vsl_study import pipeline as pipeline_mod
    from vsl_study.models import TranscriptResult, TranscriptSegment

    calls = {"n": 0}

    def flaky(wav_path, settings, progress=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt failed")
        return TranscriptResult(
            status="complete",
            model=settings.model,
            language="en",
            language_source="configured",
            device="cpu",
            device_note="test",
            segments=[TranscriptSegment(id="seg_0001", start=0.0, end=0.4, text="hello")],
            text="hello",
        )

    monkeypatch.setattr(pipeline_mod, "transcribe_wav", flaky)
    out = tmp_path / "retry-job"
    first = process_video(
        video,
        out,
        settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=False),
    )
    assert first["transcript_status"] == "failed"
    second = process_video(
        video,
        out,
        settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=False),
    )
    assert second["transcript_status"] == "complete"
    payload = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["text"] == "hello"


@requires_ffmpeg
def test_frame_extraction_failure_is_terminal(tmp_path: Path, monkeypatch):
    video = write_color_video(tmp_path / "broken-frames.mp4", duration=1.0, audio=False)
    from vsl_study import pipeline as pipeline_mod
    from vsl_study.pipeline import PipelineError

    def fail_capture(*args, **kwargs):
        raise RuntimeError("no pixels")

    monkeypatch.setattr(pipeline_mod, "capture_candidates", fail_capture)
    with pytest.raises(PipelineError, match="Screenshot extraction failed"):
        process_video(
            video,
            tmp_path / "no-frames",
            settings=ProcessSettings(interval=5.0, detector="content", compact_view=False, ocr=False),
        )
    job = json.loads((tmp_path / "no-frames" / "job.json").read_text(encoding="utf-8"))
    assert job["stages"]["frames"]["status"] == "failed"
    assert job["stages"]["frames"]["status"] != "running"


@requires_ffmpeg
def test_stale_frame_cache_is_not_reused_across_key_versions(tmp_path: Path):
    video = write_color_video(tmp_path / "solid.mp4", duration=1.5, fps=15)
    out = tmp_path / "job"
    settings = ProcessSettings(interval=0.7, detector="content", compact_view=False, ocr=False)
    process_video(video, out, settings=settings)
    frames_path = out / "cache" / "frames.json"
    payload = json.loads(frames_path.read_text(encoding="utf-8"))
    for shot in payload["screenshots"]:
        shot["notes"] = ["poisoned prior capture"]
        shot["actual_time"] = 0.0
    payload["cache_key"] = payload["cache_key"] + "|stale-version"
    frames_path.write_text(json.dumps(payload), encoding="utf-8")
    process_video(video, out, settings=settings)
    refreshed = json.loads(frames_path.read_text(encoding="utf-8"))
    assert "|stale-version" not in refreshed["cache_key"]
    assert refreshed["screenshots"]
    assert all("poisoned prior capture" not in (shot.get("notes") or []) for shot in refreshed["screenshots"])
    later = [shot for shot in refreshed["screenshots"] if shot["requested_time"] > 0.2]
    assert later
    assert any(shot["actual_time"] > 0.15 for shot in later)
