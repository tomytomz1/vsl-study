# Completed reprocess sample

This folder is the **successful** reprocess of the same local recording that produced the failed snapshot in `samples/vsl-study-tests/`. That failed snapshot was not rewritten.

## Source

- Capture id: `010fac1ba67146b7a8f7c8e615dd9ca2`
- Local recording (not in git): `%LOCALAPPDATA%\VSL Study\captures\010fac1ba67146b7a8f7c8e615dd9ca2\recording.fixed.webm` (24,341,269 bytes)
- Stop reason: `max_duration`, complete capture of this recording (not a claim that the original VSL was fully watched)
- Processing code: `cf2e599c416d8780789ce6e53ec1c76774ee39cd` on `cursor/stale-recorder-generation-isolation`

## Settings

- Model `small.en`, language `en`, device auto → CPU fp32
- Detector `adaptive`, interval 5s, scene-start offset 0.25s, max width 1280
- OCR off, compact view on
- Capture provenance copied from the original `capture.json`

## Result

- Transcript status: `complete` (real speech, not “no speech detected”)
- Screenshots: 80 scheduled, 80 completed
- Scenes: 55, ordered seconds, empty frame-index columns (1000 FPS metadata is untrusted)
- Final image: requested 119.377s, actual PTS **119.376s**, with an explicit last-frame fallback note
- Local job folder: `C:\Users\Tomas\Desktop\VSL STUDY TESTS-reprocess-cf2e599`

## ZIP (local only)

The evidence ZIP was **not** committed. It duplicates the frames already in this sample.

See `zip_inventory.json`:

- Path: `C:\Users\Tomas\Desktop\VSL STUDY TESTS-reprocess-cf2e599\vsl_study_evidence.zip`
- Size: 12,218,681 bytes
- Entries: 184
- JPEGs inside ZIP: 162
- SHA-256: `037dcccb83a8f58f1c0add163cb77a30738676c781ef396c7ca029764abd8678`
- `ZipFile.testzip()`: no CRC errors

## Measured timings

Method: UTC timestamps written by the processing callback, plus `elapsed_s` around `process_video`. Same machine as the earlier from-zero subset. These are wall times for this recording, not extrapolated.

| Stage | Wall time | How it was measured |
| --- | --- | --- |
| Full `process_video` | **91.1s** | `elapsed_s` in the worker result |
| Inspect + audio extract | ~3s | 04:54:45Z–04:54:48Z |
| Whisper `small.en` CPU | ~39s | 04:54:48Z–04:55:27Z |
| Scene detect | ~36s | 04:55:27Z–04:56:03Z |
| Screenshot stage (80 frames, including last-frame tail) | **~10s** | 04:56:03Z–04:56:13Z |
| Export / ZIP | ~3s | 04:56:14Z–04:56:17Z |

Earlier representative subset on the **old** from-zero decoder (3 timestamps at 10s, 40s, 80s only): **15.32s**. That subset was not scaled into an 80-frame estimate.

## Verification limits

- Processing was invoked with `pythonw.exe` calling `process_video` directly. It was **not** started from the Tkinter “Create study folder” flow, so desktop status-line text for this job was not verified.
- Audio was not played through speakers. Transcript text matches on-screen captions in early/mid/final stills.
- The original failed sample in `samples/vsl-study-tests/` is unchanged.
