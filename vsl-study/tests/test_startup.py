import os
import subprocess
import sys
from pathlib import Path

from vsl_study.doctor import collect_checks
from vsl_study.ffcmd import _hidden_kwargs


SRC = Path(__file__).resolve().parents[1] / "src"


def test_collect_checks_can_skip_streamlit():
    names = [check.name for check in collect_checks(include_streamlit=False)]
    assert "streamlit" not in names
    assert "ffmpeg" in names
    assert "torch" in names
    assert "scenedetect" in names


def test_desktop_import_does_not_load_heavy_libraries():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import vsl_study.desktop\n"
        "heavy = [name for name in ('torch', 'whisper', 'streamlit', 'scenedetect') if name in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_doctor_does_not_load_scenedetect_video_probe():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from vsl_study.doctor import collect_checks\n"
        "collect_checks(include_streamlit=False)\n"
        "assert 'scenedetect.output.video' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hidden_subprocess_kwargs_on_windows():
    if os.name != "nt":
        return
    kwargs = _hidden_kwargs()
    assert "creationflags" in kwargs
    assert "startupinfo" in kwargs


def test_capture_server_import_does_not_load_heavy_libraries():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import vsl_study.capture_server\n"
        "heavy = [name for name in ('torch', 'whisper', 'streamlit', 'scenedetect') if name in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
