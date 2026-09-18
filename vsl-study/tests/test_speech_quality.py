from vsl_study.desktop import _whisper_model, format_job_completion, unique_job_dir


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


def test_format_job_completion_distinguishes_transcript_failure():
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
