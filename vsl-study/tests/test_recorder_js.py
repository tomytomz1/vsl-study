import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_TESTS = sorted((ROOT / "tests" / "js").glob("test_*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to run recorder state-machine tests")
def test_recorder_javascript():
    assert JS_TESTS, "expected JavaScript tests under tests/js"
    result = subprocess.run(
        ["node", "--test", *[str(path) for path in JS_TESTS]],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
