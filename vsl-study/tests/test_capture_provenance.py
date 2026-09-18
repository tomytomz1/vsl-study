import json
import zipfile
from pathlib import Path

from conftest import requires_ffmpeg, write_color_video
from vsl_study.capture_meta import build_capture_record
from vsl_study.models import ProcessSettings
from vsl_study.pipeline import add_frames_at, process_video


@requires_ffmpeg
def test_capture_metadata_survives_manifest_and_zip(tmp_path: Path):
    video = write_color_video(tmp_path / "src.mp4", duration=2.0, audio=True)
    capture = build_capture_record(
        recording_id="recabc",
        source_url="https://user:secret@example.com/vsl?token=abc",
        title="Demo",
        mime_type="video/mp4",
        audio_track=True,
        audio_detected=True,
        stop_reason="user_stop",
        complete=True,
        media_duration_s=2.0,
        recording_path=str(video),
    )
    assert "secret" not in json.dumps(capture)
    assert "token=abc" not in json.dumps(capture)
    out = tmp_path / "job"
    result = process_video(
        video,
        out,
        settings=ProcessSettings(interval=1.0, compact_view=False, ocr=False, capture=capture),
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture"]["recording_id"] == "recabc"
    assert manifest["capture"]["input_type"] == "browser_tab_recording"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "browser tab recording" in report.lower()
    add_frames_at(out, [1.2])
    manifest2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2["capture"]["recording_id"] == "recabc"
    with zipfile.ZipFile(out / "vsl_study_evidence.zip") as zf:
        zipped = json.loads(zf.read("manifest.json"))
    assert zipped["capture"]["recording_id"] == "recabc"
    assert result["job"] == str(out)
