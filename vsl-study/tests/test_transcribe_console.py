from __future__ import annotations

import io
import sys
import wave
from pathlib import Path
from unittest import mock

import pytest

from vsl_study.models import ProcessSettings
from vsl_study.transcribe import transcribe_wav
from vsl_study.whisper_io import ProgressWriter, console_streams_writable, safe_tqdm, stream_is_writable


def _silent_wav(path: Path, duration_s: float = 0.3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(16000 * duration_s)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def test_progress_writer_survives_missing_streams():
    assert stream_is_writable(None) is False
    messages: list[str] = []
    writer = ProgressWriter(lambda stage, msg: messages.append(msg), "transcribe")
    writer.write("hello\r")
    writer.write("world\n")
    writer.flush()
    assert messages
    assert writer.writes >= 1


def test_tqdm_write_path_with_stdout_and_stderr_none():
    import tqdm

    old_out, old_err = sys.stdout, sys.stderr
    messages: list[tuple[str, str]] = []
    try:
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]
        assert console_streams_writable() is False
        with safe_tqdm(lambda stage, msg: messages.append((stage, msg)), "transcribe") as sink:
            # Same disable rule as installed Whisper when verbose=False.
            with tqdm.tqdm(total=8, unit="frames", disable=False) as bar:
                bar.update(8)
            sink.flush()
        assert sink.writes >= 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_whisper_download_progress_without_console_streams():
    import tqdm

    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]
        with safe_tqdm(None, "transcribe") as sink:
            with tqdm.tqdm(
                total=1024,
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                loop.update(512)
            sink.flush()
        assert sink.writes >= 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class _FakeModel:
    def transcribe(self, wav_path, **decode):
        import tqdm

        verbose = decode.get("verbose")
        # Reproduce Whisper's bar: enabled when verbose is False.
        with tqdm.tqdm(total=4, unit="frames", disable=verbose is not False) as bar:
            bar.update(4)
        # Also force the crashy configuration explicitly.
        with tqdm.tqdm(total=4, unit="frames", disable=False) as bar:
            bar.update(4)
        return {
            "language": "en",
            "text": "hello there",
            "segments": [{"start": 0.0, "end": 0.2, "text": " hello there"}],
        }


def test_transcribe_wav_with_missing_console_streams(tmp_path: Path, monkeypatch):
    wav = _silent_wav(tmp_path / "silent.wav")
    old_out, old_err = sys.stdout, sys.stderr
    monkeypatch.setattr("whisper.load_model", lambda *args, **kwargs: _FakeModel())
    try:
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]
        result = transcribe_wav(
            str(wav),
            ProcessSettings(model="tiny.en", language="en", device="cpu", transcribe_backend="openai-whisper"),
        )
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    assert result.status == "complete"
    assert "hello" in result.text
    assert result.segments


@pytest.mark.skipif(
    not Path(sys.executable).with_name("pythonw.exe").exists(),
    reason="pythonw.exe is not next to this interpreter",
)
def test_pythonw_natural_missing_streams(tmp_path: Path):
    """Real pythonw.exe with inherited stdio, not capture_output and not child-forced None."""
    import json
    import subprocess
    import textwrap

    result_path = tmp_path / "pythonw_result.json"
    script = tmp_path / "pythonw_tqdm.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import traceback
            import sys
            from pathlib import Path

            payload = {{
                "stdout_is_none": sys.stdout is None,
                "stderr_is_none": sys.stderr is None,
                "stdout_type": type(sys.stdout).__name__,
                "stderr_type": type(sys.stderr).__name__,
            }}
            try:
                sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
                from vsl_study.whisper_io import safe_tqdm
                import tqdm
                with safe_tqdm(None, "transcribe") as sink:
                    with tqdm.tqdm(total=3, unit="frames", disable=False) as bar:
                        bar.update(3)
                    sink.flush()
                payload["writes"] = sink.writes
                payload["ok"] = True
            except Exception:
                payload["ok"] = False
                payload["error"] = traceback.format_exc()
            Path({str(result_path)!r}).write_text(json.dumps(payload), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    completed = subprocess.run(
        [str(pythonw), str(script)],
        timeout=30,
        check=False,
        close_fds=True,
    )
    if not result_path.exists():
        pytest.fail(
            f"pythonw child did not write {result_path} (exit {completed.returncode})"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not payload.get("stdout_is_none") or not payload.get("stderr_is_none"):
        pytest.skip(
            "pythonw did not have naturally absent stdout/stderr in this environment "
            f"(stdout={payload.get('stdout_type')}, stderr={payload.get('stderr_type')})"
        )
    assert completed.returncode == 0
    assert payload.get("ok") is True, payload.get("error")
    assert int(payload.get("writes") or 0) >= 1
