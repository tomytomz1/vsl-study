from vsl_study.timeutil import format_timecode, parse_timecode, unique_sorted, windows_5min
from vsl_study.transcribe import SettingsError, validate_model_language
import pytest


def test_parse_timecode_variants():
    assert parse_timecode("00:02:15.500") == pytest.approx(135.5)
    assert parse_timecode("2:15.5") == pytest.approx(135.5)
    assert parse_timecode("12.25") == pytest.approx(12.25)
    assert parse_timecode(3) == 3.0
    with pytest.raises(ValueError):
        parse_timecode("nope")


def test_format_roundtrip():
    assert format_timecode(135.5) == "00:02:15.500"


def test_unique_sorted_ms():
    assert unique_sorted([1.0, 1.0004, 2.0]) == [1.0, 2.0]


def test_five_minute_windows_cover_duration():
    windows = windows_5min(700)
    assert windows[0][2] == "0000-0005"
    assert windows[1][0] == pytest.approx(300)
    assert windows[-1][1] == pytest.approx(700)
    assert windows[-1][2] == "0010-0015"


def test_reject_english_only_with_other_language():
    with pytest.raises(SettingsError):
        validate_model_language("small.en", "es", "transcribe")


def test_reject_translate_on_en_and_turbo():
    with pytest.raises(SettingsError):
        validate_model_language("small.en", "en", "translate")
    with pytest.raises(SettingsError):
        validate_model_language("turbo", "en", "translate")


def test_multilingual_ok():
    model, lang, task = validate_model_language("small", "es", "transcribe")
    assert model == "small"
    assert lang == "es"
    assert task == "transcribe"
