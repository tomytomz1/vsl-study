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
                start_frame=None,
                end_frame=None,
            )
        )
        return scenes
    video_end = info.video_duration_s if info.video_duration_s and info.video_duration_s > 0 else None
    for index, (start_tc, end_tc) in enumerate(pairs, start=1):
        start_s = max(0.0, float(start_tc.seconds))
        end_s = max(start_s, float(end_tc.seconds))
        if index == len(pairs):
            end_s = max(end_s, info.duration_s)
            if video_end is not None:
                # Keep the last scene covering the recording timeline, including
                # any audio-only tail after the final video frame.
                end_s = max(end_s, info.duration_s)
        start_frame, end_frame = _scene_frame_indices(info, start_tc, end_tc)
        scenes.append(
            Scene(
                id=f"scene_{index:04d}",
                start=start_s,
                end=min(info.duration_s, max(end_s, start_s)),
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    if scenes and scenes[-1].end < info.duration_s:
        scenes[-1].end = info.duration_s
    if scenes and scenes[0].start > 0:
        scenes[0].start = 0.0
        scenes[0].start_frame = None
    return scenes


def _scene_frame_indices(info: VideoInfo, start_tc, end_tc) -> tuple[int | None, int | None]:
    if not info.fps_trusted:
        return None, None
    try:
        start_f = int(start_tc.frame_num)
        end_f = int(end_tc.frame_num)
    except (TypeError, ValueError, AttributeError):
        return None, None
    if end_f < start_f:
        return None, None
    if start_f < 0 or end_f < 0:
        return None, None
    return start_f, end_f
