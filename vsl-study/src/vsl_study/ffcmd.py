"""Safe subprocess helpers. Never use shell=True; paths may contain spaces or Unicode."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class CommandError(RuntimeError):
    def __init__(self, message: str, *, cmd: Sequence[str], returncode: int, stderr: str):
        super().__init__(message)
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr


def which(name: str) -> str | None:
    return shutil.which(name)


def _hidden_kwargs() -> dict[str, object]:
    """Keep console tools from flashing a window when the app is launched with pythonw."""
    if os.name != "nt":
        return {}
    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    kwargs["creationflags"] = create_no_window
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    kwargs["startupinfo"] = startupinfo
    return kwargs


def run(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        shell=False,
        **_hidden_kwargs(),
    )
    if check and result.returncode != 0:
        preview = " ".join(args[:4])
        raise CommandError(
            f"Command failed ({result.returncode}): {preview}",
            cmd=args,
            returncode=result.returncode,
            stderr=(result.stderr or "")[-4000:],
        )
    return result


def popen(
    args: Sequence[str],
    *,
    stdout: int | None = subprocess.PIPE,
    stderr: int | None = subprocess.PIPE,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen(
        list(args),
        stdout=stdout,
        stderr=stderr,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        shell=False,
        bufsize=0,
        **_hidden_kwargs(),
    )
