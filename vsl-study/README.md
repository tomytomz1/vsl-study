# VSL Study

Local Python tool that turns a **video sales letter you already have** into a timestamped transcript, real screenshots, and a portable evidence package. It prepares evidence. It does not generate unsupported marketing analysis, download videos, or call a cloud AI.

Default English model: **`small.en`**. Faster CPU option: **`base.en`**.

## What you get

Each job folder contains:

- `transcript.txt`, `transcript_timestamped.txt`, `transcript.srt`, `transcript.json`
- `onscreen.txt`, `ocr.json` — on-screen text from screenshots, kept separate from speech
- `scenes.csv`, `manifest.json`
- `frames/` — actual JPEG captures (IDs and times in the filename)
- `contact_sheets/` — navigation thumbnails; labels sit in the **margin**, not on the screenshot
- `report.html` — searchable offline timeline (relative images, no remote scripts)
- `report.md` — portable Markdown with relative image paths
- `ai_study_prompt.md` — reusable analysis instructions (no API call)
- `evidence_5min/` — chronological 5-minute slices for models with image/context limits
- `vsl_study_evidence.zip` — the package without the original video or temp audio (unless you ask)

**A Markdown image path or a ZIP upload does not mean an AI inspected the images.** Attach the relevant `frames/` files with the matching transcript, or use a workflow that actually opens the archive and pixels. Upload limits differ by product; this tool does not claim compatibility with any particular limit.

## Requirements

- **Python 3.11** (Whisper’s supported range is 3.8–3.11; 3.12+ is not used here)
- **FFmpeg and ffprobe** on `PATH`
- Optional: **Tesseract** as an extra OCR fallback. RapidOCR ships with the Python package and does not need a PATH install. Missing OCR does not block transcription or screenshots.

CPU transcription uses **faster-whisper** with CPU INT8 by default. The English model remains **`small.en`** unless you pick Faster (`base.en`) or More accurate (`medium.en`). This is still not real-time. CUDA/openai-whisper remains available when a GPU is present. There is no NVIDIA GPU requirement.

New browser recordings request a **study capture profile**: maximum width 1280, 10 fps, tab audio, no crop or stretch. Actual track settings are stored with the job. A 4K fallback is not labeled as an optimized recording.

New jobs sample screenshots about every **15 seconds** plus scene changes, with a **600** automatic still cap (opening and last frame kept; manual timestamps are extra). If the recording continues after spoken content ends, automatic screenshots stop shortly after that end instead of covering hours of Replay. OCR covers up to **300** automatic images and does not treat layout-similar slides as identical. Sampled evidence is not exhaustive. More accurate speech quality uses a denser 5-second preset.

The report can be opened when writing finishes even if ZIP packaging is still running or later fails. `timings.json` records stage durations and cache hits.

Older jobs keep the settings stored in `job.json`. Loading them does not silently switch to the new sampling or transcription backend.

### FFmpeg

- Windows: [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/), Chocolatey `choco install ffmpeg`, or Scoop `scoop install ffmpeg`
- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`

Confirm:

```bash
ffmpeg -version
ffprobe -version
```

### On-screen OCR

VSLs put prices, headlines, and CTAs on the video. RapidOCR (bundled models) can read each screenshot, with optional Tesseract fallback. Results go in `onscreen.txt` and `ocr.json`, and are copied into each `evidence_5min/` folder. They stay separate from the Whisper transcript.

**Defaults differ by entry point:**

- **Desktop app:** Copy on-screen text starts **unchecked** for a new preference profile. After you change it, that choice is remembered.
- **CLI:** OCR is **on** unless you pass `--no-ocr`.
- **Streamlit UI:** the checkbox starts on.

Tesseract is optional. If you already have it, the app finds `tesseract.exe` even when it is not on PATH (typical Windows install: `C:\Program Files\Tesseract-OCR\`).

- Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) or `winget install --id UB-Mannheim.TesseractOCR -e`
- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt install tesseract-ocr`

Use `--no-ocr` (CLI) to skip. In the desktop app, leave **Copy on-screen text** unchecked to skip.

## Setup

### Windows PowerShell

```powershell
cd "vsl-study"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Optional: CUDA / openai-whisper GPU path
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m vsl_study doctor
```

If `py -3.11` is missing: `winget install --id Python.Python.3.11 -e`

### macOS / Linux

```bash
cd vsl-study
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Optional: CUDA / openai-whisper GPU path
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m vsl_study doctor
```

The first transcription run **downloads model weights** (faster-whisper uses a Hugging Face cache; openai-whisper often uses `~/.cache/whisper`). After that, core processing does not need the network.

## Use it like a normal Windows app

Local-file processing stays in the desktop window. Recording a browser tab opens one companion page in Chrome or Edge; transcription still runs in the desktop app. You do not need Cursor or a Streamlit server.

1. Double-click **Launch VSL Study.cmd** (or `python -m vsl_study app`).
2. Choose a **save folder**.
3. Either **Choose file** for a video you already have, or **Record a browser tab**.
4. Click **Create study folder** (file input) or **Stop and process** in the recorder (tab capture).
5. When it finishes, use **Open save folder** or **Open report**.

### Record a browser tab

Tested target: **current Chrome or Edge on Windows**. Tab audio depends on the browser picker; this app cannot silently select a tab from a URL. A microphone is not used.

Recordings are saved under `%LOCALAPPDATA%\VSL Study\captures\` (unique folder per recording). The study folder you picked receives the transcript and screenshots. If that folder already belongs to another video, a new sibling folder is used.

If the recorder tab crashes, the browser loses the unsaved buffer. Chunks already acknowledged on disk are kept. Incomplete recordings are **not** treated as a full VSL and are not analyzed automatically.

The desktop watches a **3-minute heartbeat lease**. The recorder page pings the local service about every 15 seconds, and also when you return to that tab. Chrome may delay timers in a background tab to about once a minute, so the lease is longer than that on purpose — watching the VSL in another tab should not cancel the recording. If the page is closed or the browser disappears, the lease expires, saved chunks are kept, and **Cancel recording** (or waiting out the lease) returns the desktop to a usable state without restarting the app. Unload beacons are not relied on.

Capture takes **real playback time**. Keep the computer awake. The app cannot press play or detect when a cross-origin video ends. A minutes field, if you fill it, is only a time limit.

After you click **Record a browser tab**:

1. **Open page** — optional address; or open the VSL yourself.
2. **Select tab with audio** — choose that browser tab and turn on sharing its sound.
3. **Start and play** — start recording, then play the video from the beginning if you can. Lead-in is part of the recording timeline.
4. **Stop and process** — wait until saving finishes; the desktop window then transcribes and takes screenshots. **Cancel recording** (on this page or in VSL Study) gives up without analyzing. Canceling the browser’s tab picker only lets you choose again.
5. **Open evidence** — **Open save folder** / **Open report** in VSL Study.

Timestamps in the report are relative to **this recording**, not verified times in the original video.

See `docs/processing-performance.md` for post-Stop defaults, cache versions, and what still needs a live benchmark.

## Commands

```bash
python -m vsl_study doctor
python -m vsl_study process --input "C:/Videos/yu-sleep.mp4" --output "./output/yu-sleep" --model small.en --language en
python -m vsl_study frames --job "./output/yu-sleep" --at 00:02:15.500 00:17:40
python -m vsl_study ui
```

`ui` binds Streamlit to **127.0.0.1**. Prefer a **file path on the computer running the app** for large screen recordings. Upload is optional and capped at **100 MB**.

If Streamlit asks for an email on first launch, leave it blank or use the project `.streamlit/credentials.toml` already in this repo. `python -m vsl_study ui` starts headless so that prompt is skipped.

### First Yu Sleep VSL

Put your local recording on disk, then:

```powershell
python -m vsl_study process --input "PATH\TO\yu-sleep.mp4" --output ".\output\yu-sleep" --model small.en --language en
```

Nothing in the code is hardcoded to Yu Sleep. Any local MP4/MOV/MKV/WebM with decodable streams works.

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--model base.en` | Faster, lower-quality English CPU run |
| `--model small` / `--language es` | Multilingual model + original language (not `.en`) |
| `--task translate` | English translation; requires a multilingual model (not `.en`, not `turbo`) |
| `--detector content` | Fixed-threshold cuts instead of adaptive |
| `--ocr` / `--no-ocr` | On-screen text from screenshots (default on). Stored separately from speech |
| `--include-media` | Put extracted WAV in the ZIP |
| `--device cpu` | Force CPU even if CUDA exists |

## How timestamps stay honest

- The original file is never overwritten.
- Audio is extracted to 16 kHz mono WAV and **silence-padded** when the audio stream starts after the video, so Whisper seconds match playback time.
- Screenshots are real decoded frames. Requested vs actual capture times are both stored. Labels are captions, not burned-in text.
- Scene detection uses PySceneDetect 0.7.1 (`AdaptiveDetector` by default, `get_scene_list(start_in_scene=True)` so a no-cut video is one scene).
- Screenshot times use ffmpeg presentation timestamps, not `frame_index / average_fps`. On **variable frame rate**, scene *boundary* seconds from PySceneDetect still follow the decoder time base and can differ slightly; capture actual times remain PTS-based. That VFR boundary mapping is a documented limitation.

Sampling is **one frame shortly after each scene start** plus a periodic backup (default 5s). Short text flashes can be missed. Use `frames --at` for extra times without retranscribing.

## Cache

Stages are written atomically under `cache/`. Transcription is reused when the source fingerprint, model, language, and task match. Changing screenshot interval or report options does **not** rerun Whisper. Incomplete stage files are not treated as success. An output folder already bound to a **different** source is refused.

## OCR

On-screen text is stored separately from the spoken transcript (`onscreen.txt`, `ocr.json`). RapidOCR runs inside the venv. Tesseract is an optional extra engine. If both are missing, the job continues and OCR is recorded as unavailable.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `doctor` fails on Python | Use 3.11, not 3.13 |
| `ffmpeg` / `ffprobe` missing | Install FFmpeg and reopen the terminal |
| First run hangs on network | Model download; wait, or pre-copy weights into the Whisper cache |
| Empty transcript on a talking video | Confirm the file has an audio stream (`doctor` after a failed job: see `manifest.json`) |
| Video with no audio | Visual outputs still write; transcription is marked unavailable (not a fake empty success) |
| CUDA requested but CPU used | No supported GPU; the UI/CLI will say so |
| OCR never runs / empty `onscreen.txt` | Desktop: check **Copy on-screen text**. CLI: OCR is on by default. RapidOCR is the default engine. Use `--no-ocr` only to skip |
| Streamlit rerun starts nothing | Use Run once; a session lock blocks overlapping jobs |
| Another AI “didn’t see” images | Attach `frames/*.jpg` plus transcript; don’t rely on ZIP/Markdown paths |

## Tests

```powershell
python -m pytest
```

The suite generates short ffmpeg fixtures (including a filename with spaces, no-audio, single-scene, and an offset/VFR case). A real Whisper pass uses `tiny.en` on a short spoken clip when weights can be loaded.
