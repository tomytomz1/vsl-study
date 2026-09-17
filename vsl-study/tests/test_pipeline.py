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
