"""Scene detection with PySceneDetect 0.7.1 APIs only."""

from __future__ import annotations

from typing import Callable

from vsl_study.models import Scene, VideoInfo

ProgressCb = Callable[[str, str], None]


def prepare_scenedetect() -> None:
    """Stop PySceneDetect from launching ffmpeg/mkvmerge just to see if they exist."""
    import shutil

    import scenedetect.platform as platform

    if getattr(platform, "_vsl_study_no_probe", False):
        return

    def _which(name: str) -> str | None:
        return shutil.which(name) or shutil.which(f"{name}.exe")

    platform.get_ffmpeg_path = lambda: _which("ffmpeg")
    platform.get_mkvmerge_path = lambda: _which("mkvmerge")
    platform._vsl_study_no_probe = True  # type: ignore[attr-defined]


def detect_scenes(
    info: VideoInfo,
    detector_name: str = "adaptive",
    progress: ProgressCb | None = None,
) -> list[Scene]:
    if progress:
        progress("scenes", f"Detecting scenes with PySceneDetect {detector_name}")
    prepare_scenedetect()
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    name = detector_name.lower().strip()
    if name not in {"adaptive", "content"}:
        raise ValueError("detector must be 'adaptive' or 'content'")

    video = open_video(info.resolved_path)
    manager = SceneManager()
    if name == "adaptive":
        manager.add_detector(AdaptiveDetector())
    else:
        manager.add_detector(ContentDetector())
    manager.detect_scenes(video)
    pairs = manager.get_scene_list(start_in_scene=True)
    scenes: list[Scene] = []
    if not pairs:
        scenes.append(
            Scene(
                id="scene_0001",
                start=0.0,
                end=info.duration_s,
                start_frame=0,
                end_frame=None,
            )
        )
        return scenes
    for index, (start_tc, end_tc) in enumerate(pairs, start=1):
        start_s = float(start_tc.seconds)
        end_s = float(end_tc.seconds)
        if index == len(pairs):
            end_s = max(end_s, info.duration_s)
        scenes.append(
            Scene(
                id=f"scene_{index:04d}",
                start=max(0.0, start_s),
                end=min(info.duration_s, max(end_s, start_s)),
                start_frame=int(start_tc.frame_num),
                end_frame=int(end_tc.frame_num),
            )
        )
    if scenes and scenes[-1].end < info.duration_s:
        scenes[-1].end = info.duration_s
    if scenes and scenes[0].start > 0:
        scenes[0].start = 0.0
    return scenes
