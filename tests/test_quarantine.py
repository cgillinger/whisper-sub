"""Unit tests for error quarantine in build_queue (smart hardening)."""
import sys
from pathlib import Path

# Ensure project root is importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_sub import (  # noqa: E402
    DEFAULT_MAX_ERROR_ATTEMPTS,
    _is_quarantined,
    build_queue,
)


def _state(errors=None, processed=None, skipped=None):
    return {
        "processed": processed or {},
        "errors": errors or {},
        "skipped": skipped or {},
    }


def test_permanent_error_quarantined_after_first_failure(tmp_path):
    v = tmp_path / "broken.mkv"
    state = _state(errors={str(v): {"error": "ffmpeg fallback failed for broken.mkv: ..."}})
    assert build_queue([v], state, force=False) == []


def test_permanent_signature_without_attempts_field(tmp_path):
    # Existing backlog entries pre-date the attempts counter — still quarantined.
    v = tmp_path / "old.avi"
    state = _state(errors={str(v): {"error": "moov atom not found"}})
    assert build_queue([v], state, force=False) == []


def test_transient_error_retried_below_threshold(tmp_path):
    v = tmp_path / "flaky.mp4"
    state = _state(errors={str(v): {"error": "some random glitch", "attempts": 1}})
    assert build_queue([v], state, force=False) == [v]


def test_transient_error_quarantined_at_threshold(tmp_path):
    v = tmp_path / "flaky.mp4"
    state = _state(errors={str(v): {
        "error": "some random glitch", "attempts": DEFAULT_MAX_ERROR_ATTEMPTS,
    }})
    assert build_queue([v], state, force=False) == []


def test_thermal_abort_never_quarantines(tmp_path):
    # Thermal aborts are stored without an 'attempts' count and no signature.
    v = tmp_path / "fine.mkv"
    state = _state(errors={str(v): {"error": "thermal pause exceeded 30 min"}})
    assert build_queue([v], state, force=False) == [v]


def test_force_requeues_quarantined(tmp_path):
    v = tmp_path / "broken.mkv"
    state = _state(errors={str(v): {"error": "ffmpeg fallback failed"}})
    assert build_queue([v], state, force=True) == [v]


def test_custom_max_error_attempts(tmp_path):
    v = tmp_path / "flaky.mp4"
    state = _state(errors={str(v): {"error": "glitch", "attempts": 2}})
    assert build_queue([v], state, force=False, max_error_attempts=5) == [v]
    assert build_queue([v], state, force=False, max_error_attempts=2) == []


def test_is_quarantined_helper():
    assert _is_quarantined({"error": "ffmpeg fallback failed"}, 2) is True
    assert _is_quarantined({"error": "x", "attempts": 2}, 2) is True
    assert _is_quarantined({"error": "x", "attempts": 1}, 2) is False
    assert _is_quarantined({"error": "x"}, 2) is False
