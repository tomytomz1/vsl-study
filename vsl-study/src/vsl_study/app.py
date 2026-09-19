"""Thin local Streamlit UI that calls the same pipeline as the CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from vsl_study.doctor import collect_checks, format_report, required_ok
from vsl_study.models import ProcessSettings
from vsl_study.pipeline import process_video
from vsl_study.transcribe import SettingsError

UPLOAD_LIMIT_MB = 100


def _init_state() -> None:
    st.session_state.setdefault("job_running", False)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("log_lines", [])
    st.session_state.setdefault("stage", "")


def main() -> None:
    st.set_page_config(page_title="VSL Study", layout="wide")
    _init_state()
    st.title("VSL Study")
    st.caption(
        "Local evidence extractor. Input is a video file you already have. "
        "Nothing is uploaded to a cloud API. To record a browser tab, use the desktop app "
        "(`python -m vsl_study app` or Launch VSL Study.cmd)."
    )

    with st.expander("Dependency check", expanded=False):
        st.code(format_report(collect_checks()))

    source_mode = st.radio(
        "Source",
        ["Path on this computer (recommended for large recordings)", "Upload a small file"],
        index=0,
    )

    deps_ok, _checks = required_ok()
    if not deps_ok:
        st.error("Required dependencies are missing. See Dependency check.")

    with st.form("run_job"):
        input_path = ""
        if source_mode.startswith("Path"):
            input_path = st.text_input(
                "Video path on the computer running this app",
                help="Use a full path to the MP4/MOV/MKV/WebM file. Large screen recordings should use this field, not upload.",
            )
        else:
            st.info(
                f"Upload is limited to {UPLOAD_LIMIT_MB} MB because Streamlit holds the file in memory. "
                "For longer VSLs, use the path field instead."
            )
            uploaded = st.file_uploader(
                "Small video file",
                type=["mp4", "mov", "mkv", "webm"],
                max_upload_size=UPLOAD_LIMIT_MB,
            )
            if uploaded is not None:
                tmp = Path(tempfile.gettempdir()) / "vsl_study_uploads"
                tmp.mkdir(parents=True, exist_ok=True)
                dest = tmp / Path(uploaded.name).name
                dest.write_bytes(uploaded.getbuffer())
                input_path = str(dest)
        output_dir = st.text_input("Output folder", value=str(Path.cwd() / "output" / "job"))
        col1, col2, col3 = st.columns(3)
        with col1:
            model = st.selectbox(
                "Whisper model",
                ["small.en", "base.en", "tiny.en", "tiny", "base", "small", "medium", "medium.en", "large", "turbo"],
                index=0,
            )
            language = st.text_input("Language", value="en")
            task = st.selectbox("Task", ["transcribe", "translate"], index=0)
        with col2:
            detector = st.selectbox("Scene detector", ["adaptive", "content"], index=0)
            interval = st.number_input("Screenshot interval (seconds)", min_value=0.5, value=5.0, step=0.5)
            max_width = st.number_input("Max screenshot width", min_value=320, value=1280, step=64)
        with col3:
            ocr = st.checkbox("OCR on-screen text (slides, prices, captions)", value=True)
            compact = st.checkbox("Compact similar-frame view", value=True)
            device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        st.caption("Progress is by stage name only. No completion percentage is estimated.")
        run = st.form_submit_button(
            "Run",
            type="primary",
            disabled=st.session_state.job_running or not deps_ok,
        )

    log_box = st.empty()
    status_box = st.empty()

    if run:
        if st.session_state.job_running:
            st.warning("A job is already running in this session.")
        elif not input_path.strip():
            st.error("Provide a video path or upload a small file.")
        else:
            st.session_state.job_running = True
            st.session_state.log_lines = []
            try:
                settings = ProcessSettings(
                    model=model,
                    language=language,
                    task=task,
                    detector=detector,
                    interval=float(interval),
                    max_width=int(max_width),
                    ocr=ocr,
                    compact_view=compact,
                    device=device,
                )

                def on_progress(stage: str, message: str) -> None:
                    st.session_state.stage = stage
                    st.session_state.log_lines.append(f"[{stage}] {message}")
                    status_box.info(f"Stage: {stage} — {message}")
                    log_box.code("\n".join(st.session_state.log_lines[-40:]))

                result = process_video(input_path.strip(), output_dir.strip(), settings=settings, progress=on_progress)
                st.session_state.last_result = result
                if result.get("package_status") == "failed":
                    status_box.warning(
                        f"Report is ready in {result['job']}. Packaging failed: {result.get('package_error') or 'ZIP incomplete'}."
                    )
                else:
                    status_box.success(f"Done. Package: {result['job']}")
            except SettingsError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
            finally:
                st.session_state.job_running = False

    if st.session_state.log_lines:
        log_box.code("\n".join(st.session_state.log_lines[-40:]))

    result = st.session_state.last_result
    if result:
        st.subheader("Result")
        st.json(
            {
                "job": result["job"],
                "transcript": result["transcript_status"],
                "scenes": result["scene_count"],
                "screenshots": result["screenshot_count"],
                "zip": result["zip"],
            }
        )
        job = Path(result["job"])
        preview_txt = job / "transcript_timestamped.txt"
        if preview_txt.exists():
            st.text_area("Transcript preview", preview_txt.read_text(encoding="utf-8")[:4000], height=200)
        frames = sorted((job / "frames").glob("*.jpg"))[:6]
        if frames:
            st.image([str(p) for p in frames], caption=[p.name for p in frames], width=240)
        report = job / "report.html"
        zip_path = job / "vsl_study_evidence.zip"
        if report.exists():
            st.download_button("Download report.html", report.read_bytes(), file_name="report.html", mime="text/html")
        if zip_path.exists():
            st.download_button(
                "Download evidence ZIP",
                zip_path.read_bytes(),
                file_name="vsl_study_evidence.zip",
                mime="application/zip",
            )
        st.caption(
            "A ZIP or Markdown image path does not mean another AI has inspected the pixels. "
            "Attach the relevant images with the transcript."
        )


if __name__ == "__main__":
    main()
