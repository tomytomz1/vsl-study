# Processing performance (post-Stop)

This note describes the current defaults after **Stop and process**. Recording in real time is expected. The goal is to shorten the wait between Stop and a usable study package.

These sampling numbers are **initial tuning values**, not proven optimal settings. Sampled screenshots and OCR are **not exhaustive**.

## New defaults (new jobs only)

- Browser capture profile: max width 1280, 10 fps, tab audio, no crop/stretch/upscale. Constraints are requested in `getDisplayMedia` and applied with `applyConstraints` after the picker. Actual `getSettings()` values are stored. A larger fallback is not labeled as an optimized study recording.
- Target video bitrate for that profile: 1,200,000 bps.
- CPU transcription: **faster-whisper** INT8, model still **small.en** for English. Automatic language uses the matching multilingual model. `base.en` remains the Faster quality option.
- Screenshots: about every 15 seconds plus scene changes, max 600 automatic stills, opening and last available frame kept. Manual timestamps are outside that budget.
- OCR: up to 300 automatic images across the timeline. Duplicates are reused only when pixels are conservatively equivalent. The coarse 8×8 compact hash is not used as OCR identity. Standard OCR does not upscale to a 720-pixel shortest side.
- Progress names the stage. Report HTML/Markdown can be opened when those files exist; ZIP packaging is a later, separate status.
- `timings.json` records monotonic stage durations and cache hits.
- If spoken content ends and the recording keeps going (Replay / ended screen), automatic screenshots stop about 30 seconds after the last real speech. The recorder also stops after about 3 minutes of silence once tab audio has been heard.

Older `job.json` files keep their stored interval, screenshot policy, and transcription backend. Loading them does not silently adopt these defaults.

## What this does not do

- Live transcription or OCR during recording
- A canvas/`requestAnimationFrame` capture downscale (background recorder tabs can lose those callbacks)
- Cloud transcription
- A claimed 10–15 minute post-Stop time from a short fixture or cached rerun

## Recovery of the existing 72-minute recording

Located locally as `C:\Users\Tomas\Desktop\Yu Sleep Test`, capture id `3590c01ea35543008dc08a13f5670e65`.

- Source: 3840×1730 VP9 WebM, 1,076,706,024 bytes, 4296.766s. Container FPS metadata is untrusted (1000).
- Transcript, scenes, and 1,599 screenshots were already complete. No process was still writing the job.
- Original OCR artifacts were copied to `Yu Sleep Test-ocr-v2-backup` (not a substitute for the original video).
- Recovery reused the transcript and screenshots, applied the new OCR budget (300 of 1,599), then wrote reports and the ZIP. The original recording was packaged, not replaced.
- Measured recovery on this Windows machine: **2945s (~49 min)** total. OCR was 2700s; reports 86s; ZIP 100s; media copy 5s; inspect/transcribe/scenes/frames were cache hits.
- That OCR time includes naive duplicate scanning that re-decoded prior frames. Signature caching was added afterward so later jobs do not repeat that cost. The cached-signature OCR path has not been re-timed on this 72-minute job.
- This recovery is **not** evidence of a new 1280×10fps capture, and it does **not** meet the 10–15 minute post-Stop target for an uncached equivalent run.

## Measurement machine

- CPU: 11th Gen Intel Core i7-1165G7 @ 2.80GHz
- RAM: 15.6 GB
- OS: Windows 10/11 (build 26200)
- Python 3.11.9
- faster-whisper 1.2.1
- FFmpeg 2025-06-02 (gyan.dev full build)
- torch 2.14.0+cpu is installed but is not required for the default CPU faster-whisper path

First faster-whisper model download is separate from processing time and was not part of the recovery timing (transcript was already complete).

## Live checks still required on the user’s machine

- Chrome/Edge tab capture at the study profile, including background-tab continuity, A/V sync, small-text readability, and verifying the saved file dimensions/cadence independently of WebM 1000-FPS metadata
- A full-length (~72 minute) uncached run with OCR and include-media enabled, on the same Windows computer, with first model download timed separately
- Re-timing OCR duplicate detection after signature caching on a comparable screenshot set

## Automated tests (this machine)

`pytest tests -m "not slow"`: **162 passed**, **0 failed**, **0 skipped**, **1 deselected** (`test_real_whisper_fixture`, marked slow). Recorder JavaScript tests ran through Node. Real FFmpeg and OCR engines were used where those tests require them. Faster-whisper adaptation tests are mocked except where an engine is actually imported.
