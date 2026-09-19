"""Local Whisper transcription. No speaker diarization, no invented identities."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator

from vsl_study.models import ProcessSettings, TranscriptResult, TranscriptSegment
from vsl_study.whisper_io import safe_tqdm, whisper_transcribe_verbose

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


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cpu_thread_count(requested: int = 0) -> int:
    ncpu = os.cpu_count() or 4
    if requested and requested > 0:
        return max(1, min(int(requested), ncpu))
    return max(1, min(8, ncpu))


def select_device(requested: str = "auto") -> tuple[str, str, bool]:
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", "CPU selected. CUDA is not used.", False
    cuda_ok = cuda_available()
    if requested == "cuda":
        if not cuda_ok:
            return "cpu", "CUDA was requested but is not available; using CPU.", False
        return "cuda", "Using CUDA.", True
    if cuda_ok:
        return "cuda", "CUDA available; using GPU.", True
    return "cpu", "CUDA not available; using CPU. This is not real-time.", False


def resolve_transcribe_backend(settings: ProcessSettings, device: str) -> str:
    requested = (settings.transcribe_backend or "faster-whisper").strip().lower()
    if requested in {"", "auto"}:
        requested = "faster-whisper" if device == "cpu" else "openai-whisper"
    if device == "cpu" and requested == "openai-whisper":
        return "openai-whisper"
    if device == "cuda" and requested == "faster-whisper":
        return "faster-whisper"
    if device == "cuda":
        return requested if requested in {"faster-whisper", "openai-whisper"} else "openai-whisper"
    return requested


def _consume_segments(segments: Iterator[Any] | list[Any] | tuple[Any, ...]) -> list[Any]:
    """Fully consume a segment generator before declaring completion."""
    return list(segments)


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
    backend = resolve_transcribe_backend(settings, device)
    if backend == "faster-whisper":
        return _transcribe_faster_whisper(
            wav_path,
            settings=settings,
            model_name=model_name,
            language=language,
            task=task,
            device=device,
            device_note=device_note,
            progress=progress,
        )
    if backend != "openai-whisper":
        raise RuntimeError(f"Unknown transcription backend {backend!r}.")
    return _transcribe_openai_whisper(
        wav_path,
        model_name=model_name,
        language=language,
        task=task,
        device=device,
        device_note=device_note,
        use_fp16=use_fp16,
        progress=progress,
    )


def _language_source(model_name: str, language: str | None) -> str:
    if language:
        return "configured"
    if model_name in ENGLISH_ONLY_MODELS:
        return "configured"
    return "detected"


def _transcribe_openai_whisper(
    wav_path: str,
    *,
    model_name: str,
    language: str | None,
    task: str,
    device: str,
    device_note: str,
    use_fp16: bool,
    progress: ProgressCb | None,
) -> TranscriptResult:
    if progress:
        progress(
            "transcribe",
            f"Loading openai-whisper {model_name} on {device}. First use downloads model weights.",
        )
    import whisper

    decode: dict[str, Any] = {
        "task": task,
        "fp16": use_fp16,
        "verbose": whisper_transcribe_verbose(),
    }
    language_source = _language_source(model_name, language)
    if language:
        decode["language"] = language
    elif model_name in ENGLISH_ONLY_MODELS:
        decode["language"] = "en"
        language_source = "configured"

    with safe_tqdm(progress, "transcribe"):
        model = whisper.load_model(model_name, device=device)
        if progress:
            progress("transcribe", f"Transcribing with openai-whisper {model_name} ({device_note})")
        result = model.transcribe(wav_path, **decode)
    segments = _segments_from_openai(result.get("segments") or [])
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
        backend="openai-whisper",
        compute_type="fp16" if use_fp16 else "fp32",
        beam_size=None,
    )


def _transcribe_faster_whisper(
    wav_path: str,
    *,
    settings: ProcessSettings,
    model_name: str,
    language: str | None,
    task: str,
    device: str,
    device_note: str,
    progress: ProgressCb | None,
) -> TranscriptResult:
    compute = (settings.transcribe_compute_type or "int8").strip() or "int8"
    if device == "cuda" and compute == "int8":
        compute = "float16"
    threads = cpu_thread_count(settings.transcribe_cpu_threads)
    beam = int(settings.transcribe_beam_size or 5)
    if progress:
        progress(
            "transcribe",
            f"Loading faster-whisper {model_name} ({compute}) on {device}. First use downloads model weights.",
        )
    try:
        import huggingface_hub.file_download  # noqa: F401
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "faster-whisper is required for CPU transcription and is not available. "
            f"Install it with the VSL Study package. ({exc})"
        ) from exc
    language_source = _language_source(model_name, language)
    decode_language = language
    if not decode_language and model_name in ENGLISH_ONLY_MODELS:
        decode_language = "en"
        language_source = "configured"
    with safe_tqdm(progress, "transcribe"):
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
            cpu_threads=threads if device == "cpu" else 0,
        )
        if progress:
            progress(
                "transcribe",
                f"Transcribing with faster-whisper {model_name} {compute} "
                f"(beam={beam}, threads={threads if device == 'cpu' else 'n/a'}).",
            )
        segment_iter, info = model.transcribe(
            wav_path,
            language=decode_language,
            task=task,
            beam_size=beam,
            vad_filter=False,
        )
        raw_segments = _consume_segments(segment_iter)
    segments = _segments_from_faster(raw_segments)
    flags = _flag_segments(segments)
    detected = getattr(info, "language", None) or decode_language
    text = " ".join(seg.text for seg in segments).strip()
    return TranscriptResult(
        status="complete",
        model=model_name,
        language=str(detected) if detected else None,
        language_source=language_source,
        device=device,
        device_note=device_note,
        segments=segments,
        text=text,
        flags=flags,
        backend="faster-whisper",
        compute_type=compute,
        beam_size=beam,
    )


def _segments_from_openai(raw_segments: list[Any]) -> list[TranscriptSegment]:
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
    return segments


def _segments_from_faster(raw_segments: list[Any]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for index, raw in enumerate(raw_segments, start=1):
        if isinstance(raw, dict):
            start = raw.get("start")
            end = raw.get("end")
            text = raw.get("text")
            avg_logprob = raw.get("avg_logprob")
            no_speech_prob = raw.get("no_speech_prob")
            compression_ratio = raw.get("compression_ratio")
            temperature = raw.get("temperature")
        else:
            start = getattr(raw, "start", 0.0)
            end = getattr(raw, "end", 0.0)
            text = getattr(raw, "text", "")
            avg_logprob = getattr(raw, "avg_logprob", None)
            no_speech_prob = getattr(raw, "no_speech_prob", None)
            compression_ratio = getattr(raw, "compression_ratio", None)
            temperature = getattr(raw, "temperature", None)
        segments.append(
            TranscriptSegment(
                id=f"seg_{index:04d}",
                start=float(start or 0.0),
                end=float(end or 0.0),
                text=str(text or "").strip(),
                avg_logprob=_opt_float(avg_logprob),
                no_speech_prob=_opt_float(no_speech_prob),
                compression_ratio=_opt_float(compression_ratio),
                temperature=_opt_float(temperature),
            )
        )
    return segments


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
        backend=settings.transcribe_backend,
        compute_type=settings.transcribe_compute_type,
        beam_size=settings.transcribe_beam_size,
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
        backend=settings.transcribe_backend,
        compute_type=settings.transcribe_compute_type,
        beam_size=settings.transcribe_beam_size,
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
