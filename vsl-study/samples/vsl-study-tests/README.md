# Live Chrome tab-recording sample

Snapshot of `C:\Users\Tomas\Desktop\VSL STUDY TESTS` after a successful browser-tab capture on `cursor/stale-recorder-generation-isolation`.

- Recording id: `010fac1ba67146b7a8f7c8e615dd9ca2`
- Title: `My first test ever`
- Duration: 120.062s (`stop_reason`: `max_duration`)
- Media: VP9 + Opus, `audio_track` and `audio_detected` both true
- Capture metadata: `capture-session/` (copied from `%LOCALAPPDATA%\VSL Study\captures\...`)
- The source `.webm` (~24MB) is not in git

This copy was taken while screenshot extraction was still running. Whisper transcription failed in this run because tqdm tried to write to a missing stdout under `pythonw.exe` (`AttributeError: 'NoneType' object has no attribute 'write'`). Inspect `job.json` and `cache/transcribe.json` for that error.
