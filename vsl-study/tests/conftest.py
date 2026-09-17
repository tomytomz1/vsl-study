from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


requires_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe required")


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def write_color_video(
    dest: Path,
    *,
    duration: float = 3.0,
    color: str = "red",
    fps: int = 25,
    size: str = "640x360",
    audio: bool = False,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d={duration}:r={fps}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-movflags",
        "+faststart",
    ]
    if audio:
        args = [
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:d={duration}:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
        ]
    run_ffmpeg([*args, str(dest)])
    return dest


def write_cut_video(dest: Path, *, audio: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=640x360:d=2:r=25",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=640x360:d=2:r=25",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=640x360:d=2:r=25",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
    ]
    if audio:
        args = [
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=640x360:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=640x360:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=640x360:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
        ]
    run_ffmpeg([*args, str(dest)])
    return dest


def write_vfr_video(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:r=30:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(dest),
        ]
    )
    return dest


def synthesize_speech_wav(dest: Path, text: str = "The sample product is called test sleep.") -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        return None
    escaped = str(dest).replace("'", "''")
    spoken = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{escaped}'); "
        f"$s.Speak('{spoken}'); "
        "$s.Dispose();"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        return None
    return dest


def mux_speech_video(dest: Path, wav: Path, duration: float = 4.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=navy:s=640x360:d={duration}:r=25",
            "-i",
            str(wav),
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(dest),
        ]
    )
    return dest


def mux_offset_speech(dest: Path, wav: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:d=4:r=25",
            "-itsoffset",
            "0.5",
            "-i",
            str(wav),
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(dest),
        ]
    )
    return dest
