import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_TEST = ROOT / "tests" / "js" / "test_recorder_core.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to run recorder state-machine tests")
def test_recorder_core_javascript_state_machine():
    result = subprocess.run(
        ["node", "--test", str(JS_TEST)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
