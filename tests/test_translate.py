"""Unit tests for the translation pipeline."""
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure project root is importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Provider YAML loading
# ---------------------------------------------------------------------------


def test_provider_yaml_all_files_load():
    providers_dir = Path(__file__).parent.parent / "translate" / "providers"
    provider_files = list(providers_dir.glob("*.yml"))
    assert len(provider_files) == 5, f"Expected 5 provider files, found {len(provider_files)}"

    for provider_file in provider_files:
        with provider_file.open() as fh:
            data = yaml.safe_load(fh)
        assert "name" in data, f"{provider_file.name}: missing 'name'"
        assert "driver" in data, f"{provider_file.name}: missing 'driver'"
        assert "model" in data, f"{provider_file.name}: missing 'model'"
        assert "endpoint" in data, f"{provider_file.name}: missing 'endpoint'"
        assert "api_key_env" in data, f"{provider_file.name}: missing 'api_key_env'"
        assert "free_tier" in data, f"{provider_file.name}: missing 'free_tier'"
        assert isinstance(data["free_tier"], dict), f"{provider_file.name}: 'free_tier' must be a dict"


def test_provider_gemini_lite_free_tier():
    providers_dir = Path(__file__).parent.parent / "translate" / "providers"
    with (providers_dir / "gemini-lite.yml").open() as fh:
        data = yaml.safe_load(fh)
    assert data["free_tier"]["enabled"] is True
    assert data["free_tier"]["max_requests_per_day"] == 1500
    assert data["free_tier"]["safety_margin"] == 100


def test_provider_paid_tiers_not_free():
    providers_dir = Path(__file__).parent.parent / "translate" / "providers"
    for name in ("gemini", "openai", "deepseek", "anthropic"):
        with (providers_dir / f"{name}.yml").open() as fh:
            data = yaml.safe_load(fh)
        assert data["free_tier"]["enabled"] is False, f"{name}: free_tier.enabled should be false"


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------


def _make_tracker(tmp_path, initial_data=None):
    from translate.usage_tracker import UsageTracker
    usage_file = tmp_path / "usage.json"
    tracker = UsageTracker(usage_file)
    if initial_data is not None:
        tracker._data = initial_data
        tracker._save()
        tracker = UsageTracker(usage_file)
    return tracker


def test_usage_tracker_increment_writes_today(tmp_path):
    tracker = _make_tracker(tmp_path)
    tracker.increment("gemini-lite")
    tracker.increment("gemini-lite")
    data = json.loads((tmp_path / "usage.json").read_text())
    assert data["requests"] == 2
    assert data["date"] == str(date.today())
    assert data["provider"] == "gemini-lite"


def test_usage_tracker_date_rollover_resets_counter(tmp_path):
    yesterday = str(date.today() - timedelta(days=1))
    tracker = _make_tracker(tmp_path, {"date": yesterday, "requests": 1400, "provider": "gemini-lite"})
    free_tier_cfg = {
        "free_tier": {"enabled": True, "max_requests_per_day": 1500, "safety_margin": 100}
    }
    assert tracker.can_send(free_tier_cfg) is True


def test_usage_tracker_can_send_below_effective_limit(tmp_path):
    tracker = _make_tracker(tmp_path, {"date": str(date.today()), "requests": 1399, "provider": "gemini-lite"})
    cfg = {"free_tier": {"enabled": True, "max_requests_per_day": 1500, "safety_margin": 100}}
    assert tracker.can_send(cfg) is True


def test_usage_tracker_can_send_at_effective_limit(tmp_path):
    tracker = _make_tracker(tmp_path, {"date": str(date.today()), "requests": 1400, "provider": "gemini-lite"})
    cfg = {"free_tier": {"enabled": True, "max_requests_per_day": 1500, "safety_margin": 100}}
    assert tracker.can_send(cfg) is False


def test_usage_tracker_paid_provider_always_allowed(tmp_path):
    tracker = _make_tracker(tmp_path, {"date": str(date.today()), "requests": 9999, "provider": "openai"})
    cfg = {"free_tier": {"enabled": False}}
    assert tracker.can_send(cfg) is True


def test_usage_tracker_increment_before_api_call(tmp_path):
    """Counter must be written before the API call (safe over-count direction)."""
    usage_file = tmp_path / "usage.json"
    from translate.usage_tracker import UsageTracker
    tracker = UsageTracker(usage_file)
    tracker.increment("gemini-lite")
    # File must exist and show count=1 immediately after increment
    data = json.loads(usage_file.read_text())
    assert data["requests"] == 1


# ---------------------------------------------------------------------------
# SRT validation
# ---------------------------------------------------------------------------


def test_validate_srt_valid():
    from translate.translator import _validate_srt
    srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nBye\n"
    assert _validate_srt(srt) is True


def test_validate_srt_invalid_no_timestamps():
    from translate.translator import _validate_srt
    assert _validate_srt("This is not an SRT file") is False


def test_validate_srt_empty():
    from translate.translator import _validate_srt
    assert _validate_srt("") is False


# ---------------------------------------------------------------------------
# translate_srt — missing API key
# ---------------------------------------------------------------------------


def test_translate_srt_missing_api_key(tmp_path):
    from translate.translator import translate_srt

    srt_path = tmp_path / "film.en.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        result = translate_srt(
            srt_path, "sv", "gemini-lite", {},
            usage_file=tmp_path / "usage.json",
        )

    assert result is None


def test_translate_srt_unknown_provider(tmp_path):
    from translate.translator import translate_srt

    srt_path = tmp_path / "film.en.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

    result = translate_srt(
        srt_path, "sv", "nonexistent-provider", {},
        usage_file=tmp_path / "usage.json",
    )
    assert result is None


# ---------------------------------------------------------------------------
# translate_srt — Lite mode refuses at limit
# ---------------------------------------------------------------------------


def test_translate_srt_lite_mode_refuses_at_limit(tmp_path):
    from translate.translator import translate_srt
    from translate.usage_tracker import UsageTracker

    usage_file = tmp_path / "usage.json"
    tracker = UsageTracker(usage_file)
    tracker._data = {"date": str(date.today()), "requests": 1400, "provider": "gemini-lite"}
    tracker._save()

    srt_path = tmp_path / "film.en.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        result = translate_srt(
            srt_path, "sv", "gemini-lite", {},
            usage_file=usage_file,
        )

    assert result is None
    # Counter must NOT have been incremented (limit refused before increment)
    data = json.loads(usage_file.read_text())
    assert data["requests"] == 1400


# ---------------------------------------------------------------------------
# translate_srt — malformed API response is rejected
# ---------------------------------------------------------------------------


def test_translate_srt_rejects_malformed_response(tmp_path):
    from translate.translator import translate_srt

    srt_path = tmp_path / "film.en.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

    # Patch the gemini driver to return plain text without timestamps
    mock_driver = MagicMock()
    mock_driver.translate.return_value = "Hej världen\nDetta är inte SRT"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        with patch("importlib.import_module", return_value=mock_driver):
            result = translate_srt(
                srt_path, "sv", "gemini-lite", {},
                usage_file=tmp_path / "usage.json",
            )

    assert result is None
    # No translated file should have been written
    assert not (tmp_path / "film.sv.srt").exists()


# ---------------------------------------------------------------------------
# translate_srt — successful translation writes correct output path
# ---------------------------------------------------------------------------


def test_translate_srt_writes_correct_output_path(tmp_path):
    from translate.translator import translate_srt

    srt_path = tmp_path / "film.en.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

    valid_srt = "1\n00:00:01,000 --> 00:00:03,000\nHej\n"
    mock_driver = MagicMock()
    mock_driver.translate.return_value = valid_srt

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        with patch("importlib.import_module", return_value=mock_driver):
            result = translate_srt(
                srt_path, "sv", "gemini-lite", {},
                usage_file=tmp_path / "usage.json",
            )

    assert result is not None
    assert result == tmp_path / "film.sv.srt"
    assert result.exists()
    assert "Hej" in result.read_text()
