import json

from vsl_study.desktop import _whisper_model, format_job_completion, job_done_payload, unique_job_dir


def test_whisper_model_quality_language_and_translate():
    for quality, english, multi in (
        ("recommended", "small.en", "small"),
        ("faster", "base.en", "base"),
        ("accurate", "medium.en", "medium"),
    ):
        assert _whisper_model(quality, "en") == english
        assert _whisper_model(quality, "auto") == multi
        assert _whisper_model(quality, "es") == multi
        assert _whisper_model(quality, "en", "translate") == multi
        assert _whisper_model(quality, "auto", "translate") == multi


def test_format_job_completion_distinguishes_package_and_report_ready():
    done, log = format_job_completion(
        {"job": "/out", "transcript_status": "complete", "package_status": "complete", "report_ready": True}
    )
    assert "package complete" in done.lower()
    failed, flog = format_job_completion(
        {"job": "/out", "transcript_status": "complete", "package_status": "failed", "report_ready": True}
    )
    assert "report is ready" in failed.lower()
    assert "packaging failed" in failed.lower()
    assert "/out" in flog
    payload = job_done_payload(
        {
            "job": "/out",
            "transcript_status": "complete",
            "package_status": "failed",
            "report_ready": True,
            "package_error": "disk full",
            "zip": None,
        }
    )
    assert payload["package_status"] == "failed"
    assert payload["report_ready"] is True
    assert "screenshot_count" not in payload
    done, log = format_job_completion({"job": "/out", "transcript_status": "complete"})
    assert "ready" in done.lower()
    assert "/out" in log
    failed, flog = format_job_completion({"job": "/out", "transcript_status": "failed"})
    assert "could not be written down" in failed.lower()
    assert "failed" in flog.lower()
    skipped, _slog = format_job_completion({"job": "/out", "transcript_status": "skipped"})
    assert "no speech" in skipped.lower()


def test_unique_job_dir_only_when_bound(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    assert unique_job_dir(dest, "abc") == dest
    (dest / "job.json").write_text("{}", encoding="utf-8")
    other = unique_job_dir(dest, "abc")
    assert other != dest
    assert other.name.endswith("-abc")


def test_unique_job_dir_reuses_same_recording(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    rec = "3590c01ea35543008dc08a13f5670e65"
    (dest / "job.json").write_text(
        json.dumps({"capture": {"recording_id": rec}}),
        encoding="utf-8",
    )
    assert unique_job_dir(dest, rec[:8]) == dest
