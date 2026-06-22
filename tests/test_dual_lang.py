"""Unit tests for dual-language subtitle generation in _apply_translation."""
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_sub import _apply_translation  # noqa: E402


def _state(video):
    return {"processed": {str(video): {}}, "errors": {}, "skipped": {}}


def test_swedish_source_gets_english(tmp_path):
    video = tmp_path / "film.mkv"
    srt = tmp_path / "film.sv.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhej\n")
    state = _state(video)
    with patch("translate.translator.translate_srt") as m:
        m.return_value = tmp_path / "film.en.srt"
        _apply_translation(video, srt, ["sv", "en"], "openai", state, tmp_path / "state.json")
    # Called exactly once, for 'en' — 'sv' is the source language and is skipped.
    assert m.call_count == 1
    assert m.call_args.args[1] == "en"
    assert state["processed"][str(video)]["translated"] is True


def test_english_source_gets_swedish(tmp_path):
    video = tmp_path / "movie.mkv"
    srt = tmp_path / "movie.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    state = _state(video)
    with patch("translate.translator.translate_srt") as m:
        m.return_value = tmp_path / "movie.sv.srt"
        _apply_translation(video, srt, ["sv", "en"], "openai", state, tmp_path / "state.json")
    assert m.call_count == 1
    assert m.call_args.args[1] == "sv"


def test_existing_target_skipped(tmp_path):
    video = tmp_path / "film.mkv"
    srt = tmp_path / "film.sv.srt"
    srt.write_text("x")
    (tmp_path / "film.en.srt").write_text("already there")  # en already exists
    state = _state(video)
    with patch("translate.translator.translate_srt") as m:
        _apply_translation(video, srt, ["sv", "en"], "openai", state, tmp_path / "state.json")
    assert m.call_count == 0


def test_single_language_is_source_noop(tmp_path):
    video = tmp_path / "film.mkv"
    srt = tmp_path / "film.sv.srt"
    srt.write_text("x")
    state = _state(video)
    with patch("translate.translator.translate_srt") as m:
        _apply_translation(video, srt, ["sv"], "openai", state, tmp_path / "state.json")
    assert m.call_count == 0
