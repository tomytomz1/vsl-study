import sys
import types
from pathlib import Path
from types import SimpleNamespace

from vsl_study.models import ProcessSettings
from vsl_study.transcribe import _consume_segments, transcribe_wav
from vsl_study.whisper_io import safe_tqdm


class TrackingGen:
    def __init__(self, items):
        self.items = list(items)
        self.index = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            self.exhausted = True
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


def test_consume_segments_exhausts_generator():
    gen = TrackingGen([1, 2, 3])
    assert _consume_segments(gen) == [1, 2, 3]
    assert gen.exhausted is True


def test_faster_whisper_adapts_segments_and_keeps_final_item(monkeypatch, tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    gen = TrackingGen(
        [
            SimpleNamespace(
                start=0.0,
                end=1.2,
                text=" hello",
                avg_logprob=-0.2,
                no_speech_prob=0.1,
                compression_ratio=1.1,
                temperature=0.0,
            ),
            SimpleNamespace(
                start=1.2,
                end=4.0,
                text=" last words",
                avg_logprob=-0.3,
                no_speech_prob=0.05,
                compression_ratio=1.2,
                temperature=0.0,
            ),
        ]
    )

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def transcribe(self, path, **kwargs):
            assert path == str(wav)
            assert kwargs["language"] == "en"
            assert kwargs["vad_filter"] is False
            return gen, SimpleNamespace(language="en")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr("vsl_study.transcribe.cuda_available", lambda: False)

    result = transcribe_wav(
        str(wav),
        ProcessSettings(model="small.en", language="en", transcribe_backend="faster-whisper"),
    )
    assert gen.exhausted is True
    assert result.status == "complete"
    assert result.backend == "faster-whisper"
    assert result.compute_type == "int8"
    assert result.beam_size == 5
    assert len(result.segments) == 2
    assert result.segments[-1].text == "last words"
    assert result.segments[-1].end == 4.0
    assert "hello" in result.text and "last words" in result.text


def test_faster_whisper_load_does_not_require_stdio(monkeypatch, tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    gen = TrackingGen(
        [
            SimpleNamespace(
                start=0.0,
                end=1.0,
                text="ok",
                avg_logprob=None,
                no_speech_prob=None,
                compression_ratio=None,
                temperature=None,
            )
        ]
    )

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            return gen, SimpleNamespace(language="en")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr("vsl_study.transcribe.cuda_available", lambda: False)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    with safe_tqdm(None, "transcribe"):
        result = transcribe_wav(str(wav), ProcessSettings(model="tiny.en", transcribe_backend="faster-whisper"))
    assert result.status == "complete"
    assert result.segments[0].text == "ok"
