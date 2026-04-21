import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_USAGE_FILE = Path.home() / ".translate_usage.json"


class UsageTracker:
    """Tracks daily API request counts to enforce free-tier limits.

    The JSON file format is: {"date": "YYYY-MM-DD", "requests": N, "provider": "name"}
    The counter is reset automatically when the stored date differs from today.
    """

    def __init__(self, usage_file: Path = DEFAULT_USAGE_FILE) -> None:
        self._file = usage_file
        self._data = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"date": "", "requests": 0, "provider": ""}

    def _save(self) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self._file)

    def increment(self, provider: str) -> None:
        """Increment the request counter, resetting it if the date has changed.

        Written BEFORE the API call so a crash after increment over-counts —
        the safe direction for a rate-limit guard.
        """
        today = str(date.today())
        if self._data.get("date") != today:
            self._data = {"date": today, "requests": 0, "provider": provider}
        self._data["requests"] += 1
        self._data["provider"] = provider
        self._save()

    def can_send(self, provider_config: dict) -> bool:
        """Return False if the free-tier daily limit has been reached.

        Logs a clear WARNING when the limit is hit. Returns True for providers
        with free_tier.enabled = false.
        """
        free_tier = provider_config.get("free_tier", {})
        if not free_tier.get("enabled", False):
            return True

        today = str(date.today())
        if self._data.get("date") != today:
            return True

        limit = free_tier.get("max_requests_per_day", 1500)
        margin = free_tier.get("safety_margin", 100)
        effective_limit = limit - margin
        current = self._data.get("requests", 0)

        if current >= effective_limit:
            logger.warning(
                "Lite mode: daily limit reached (%d/%d). Stopping translation. Will reset tomorrow.",
                current,
                limit,
            )
            return False

        return True
