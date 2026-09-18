"""Keep Whisper/tqdm from writing to missing pythonw console streams."""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator

ProgressCb = Callable[[str, str], None]


def stream_is_writable(stream: Any) -> bool:
    if stream is None:
        return False
    write = getattr(stream, "write", None)
    if not callable(write):
        return False
    try:
        write("")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        return False
    return True


def console_streams_writable() -> bool:
    return stream_is_writable(getattr(sys, "stdout", None)) and stream_is_writable(
        getattr(sys, "stderr", None)
    )


class ProgressWriter(io.TextIOBase):
    """File-like object tqdm can write to when stdout/stderr are missing."""

    def __init__(self, progress: ProgressCb | None, stage: str = "transcribe"):
        self.progress = progress
        self.stage = stage
        self._buf = ""
        self.writes = 0

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        self.writes += 1
        self._buf += str(s).replace("\r", "\n")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = " ".join(line.split())
            if line and self.progress:
                self.progress(self.stage, line[:200])
        return len(str(s))

    def flush(self) -> None:
        if self._buf.strip() and self.progress:
            self.progress(self.stage, " ".join(self._buf.split())[:200])
            self._buf = ""

    def isatty(self) -> bool:
        return False


def whisper_transcribe_verbose() -> None:
    """Installed Whisper turns on its terminal bar when verbose is False.

    verbose=None disables that bar. Production still wraps tqdm so download
    progress (which ignores verbose) cannot touch a missing stream, and tests
    can force the False/bar path through the same wrapper.
    """
    return None


@contextmanager
def safe_tqdm(progress: ProgressCb | None = None, stage: str = "transcribe") -> Iterator[ProgressWriter]:
    """Patch tqdm used by Whisper so write() never hits a None console stream."""
    import tqdm as tqdm_mod
    import whisper
    import whisper.transcribe as whisper_transcribe

    original = tqdm_mod.tqdm
    sink = ProgressWriter(progress, stage)

    def factory(*args: Any, **kwargs: Any):
        kwargs = dict(kwargs)
        if not stream_is_writable(kwargs.get("file")):
            kwargs["file"] = sink
        return original(*args, **kwargs)

    tqdm_mod.tqdm = factory  # type: ignore[misc]
    whisper.tqdm = factory
    whisper_transcribe.tqdm = factory
    try:
        yield sink
    finally:
        tqdm_mod.tqdm = original
        whisper.tqdm = original
        whisper_transcribe.tqdm = original
