"""Local Whisper transcription. No speaker diarization, no invented identities."""

from __future__ import annotations

from typing import Any, Callable

from vsl_study.models import ProcessSettings, TranscriptResult, TranscriptSegment

ENGLISH_ONLY_MODELS = {"tiny.en", "base.en", "small.en", "medium.en"}
MULTILINGUAL_MODELS = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "turbo"}
KNOWN_MODELS = ENGLISH_ONLY_MODELS | MULTILINGUAL_MODELS
ENGLISH_ALIASES = {"en", "english"}
ProgressCb = Callable[[str, str], None]


class SettingsError(ValueError):
    pass


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text or text in {"auto", "detect"}:
        return None
    return text


def validate_model_language(model: str, language: str | None, task: str) -> tuple[str, str | None, str]:
    if model not in KNOWN_MODELS:
        raise SettingsError(
            f"Unknown model {model!r}. Known: {', '.join(sorted(KNOWN_MODELS))}"
        )
    if task not in {"transcribe", "translate"}:
        raise SettingsError("task must be 'transcribe' or 'translate'")
    lang = normalize_language(language)
    if model in ENGLISH_ONLY_MODELS:
        if lang and lang not in ENGLISH_ALIASES:
            raise SettingsError(
                f"Model {model} is English-only. Use language 'en' or a multilingual model "
                f"(tiny/base/small/medium/large/turbo) for {lang}."
            )
        lang = "en"
    if task == "translate" and model in ENGLISH_ONLY_MODELS:
        raise SettingsError(
            "Translation requires a multilingual model. The .en models cannot translate."
        )
    if task == "translate" and model == "turbo":
        raise SettingsError(
            "Whisper turbo is not trained for translation. Use small/medium/large instead, "
            "or keep task=transcribe."
        )
    return model, lang, task


def select_device(requested: str = "auto") -> tuple[str, str, bool]:
    import torch

    requested = (requested or "auto").lower()
    cuda_ok = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_ok:
            return "cpu", "CUDA was requested but is not available; using CPU (fp32).", False
        return "cuda", "Using CUDA.", True
    if requested == "cpu":
        return "cpu", "CPU selected (fp32). CUDA is not used.", False
    if cuda_ok:
        return "cuda", "CUDA available; using GPU.", True
    return "cpu", "CUDA not available or not supported; using CPU (fp32). This is not real-time.", False


def _flag_segments(segments: list[TranscriptSegment]) -> list[str]:
    global_flags: list[str] = []
    prev_text = ""
    repeat_run = 0
    for seg in segments:
        flags: list[str] = []
        if seg.no_speech_prob is not None and seg.no_speech_prob >= 0.6 and seg.text.strip():
            flags.append("high_no_speech_prob")
        if seg.compression_ratio is not None and seg.compression_ratio >= 2.4:
            flags.append("high_compression_ratio")
        if seg.avg_logprob is not None and seg.avg_logprob <= -1.0:
            flags.append("low_avg_logprob")
        compact = " ".join(seg.text.split())
        if compact and compact == prev_text:
            repeat_run += 1
            flags.append("repeated_text")
        else:
            repeat_run = 0
        if repeat_run >= 2:
            flags.append("silence_or_repetition_loop")
            global_flags.append(f"{seg.id}: possible silence-related repetition")
        prev_text = compact
        seg.flags = flags
    return global_flags


def transcribe_wav(
    wav_path: str,
    settings: ProcessSettings,
    progress: ProgressCb | None = None,
) -> TranscriptResult:
    model_name, language, task = validate_model_language(
        settings.model, settings.language, settings.task
    )
    device, device_note, use_fp16 = select_device(settings.device)
    if progress:
        progress(
            "transcribe",
            f"Loading Whisper {model_name} on {device}. First use downloads model weights.",
        )
    import whisper

    model = whisper.load_model(model_name, device=device)
    if progress:
        progress("transcribe", f"Transcribing with {model_name} ({device_note})")
    decode: dict[str, Any] = {
        "task": task,
        "fp16": use_fp16,
        "verbose": False,
    }
    language_source = "configured"
    if language:
        decode["language"] = language
    elif model_name in ENGLISH_ONLY_MODELS:
        decode["language"] = "en"
        language_source = "configured"
    else:
        language_source = "detected"

    result = model.transcribe(wav_path, **decode)
    raw_segments = result.get("segments") or []
    segments: list[TranscriptSegment] = []
    for index, raw in enumerate(raw_segments, start=1):
        text = str(raw.get("text") or "").strip()
        segments.append(
            TranscriptSegment(
                id=f"seg_{index:04d}",
                start=float(raw.get("start") or 0.0),
                end=float(raw.get("end") or 0.0),
                text=text,
                avg_logprob=_opt_float(raw.get("avg_logprob")),
                no_speech_prob=_opt_float(raw.get("no_speech_prob")),
                compression_ratio=_opt_float(raw.get("compression_ratio")),
                temperature=_opt_float(raw.get("temperature")),
            )
        )
    flags = _flag_segments(segments)
    detected_lang = result.get("language") or language or ("en" if model_name in ENGLISH_ONLY_MODELS else None)
    return TranscriptResult(
        status="complete",
        model=model_name,
        language=str(detected_lang) if detected_lang else None,
        language_source=language_source,
        device=device,
        device_note=device_note,
        segments=segments,
        text=str(result.get("text") or "").strip(),
        flags=flags,
    )


def unavailable_transcript(reason: str, settings: ProcessSettings, device_note: str = "") -> TranscriptResult:
    return TranscriptResult(
        status="skipped",
        model=settings.model,
        language=settings.language,
        language_source="n/a",
        device="n/a",
        device_note=device_note or reason,
        segments=[],
        text="",
        flags=["transcription_unavailable"],
        error=reason,
    )


def failed_transcript(reason: str, settings: ProcessSettings, device: str, device_note: str) -> TranscriptResult:
    return TranscriptResult(
        status="failed",
        model=settings.model,
        language=settings.language,
        language_source="n/a",
        device=device,
        device_note=device_note,
        segments=[],
        text="",
        flags=["transcription_failed"],
        error=reason,
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
