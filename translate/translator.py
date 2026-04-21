import importlib
import logging
import os
from pathlib import Path
from typing import Optional

from translate import TranslationError
from translate.usage_tracker import DEFAULT_USAGE_FILE, UsageTracker

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent / "providers"

_LANG_NAMES: dict[str, str] = {
    "sv": "Swedish",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pl": "Polish",
    "tr": "Turkish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
}


def _load_provider(provider_name: str) -> dict:
    import yaml

    provider_file = _PROVIDERS_DIR / f"{provider_name}.yml"
    with provider_file.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _validate_srt(text: str) -> bool:
    """Return True if text contains at least one SRT timestamp line."""
    return "-->" in text


def translate_srt(
    srt_path: Path,
    target_lang: str,
    provider_name: str,
    state: dict,
    *,
    usage_file: Optional[Path] = None,
) -> Optional[Path]:
    """Translate *srt_path* to *target_lang* using *provider_name*.

    Returns the path of the written translated SRT, or None if skipped/failed.
    Errors are logged; this function never raises.
    """
    if usage_file is None:
        usage_file = DEFAULT_USAGE_FILE

    try:
        provider = _load_provider(provider_name)
    except FileNotFoundError:
        logger.error(
            "Unknown provider '%s' — no translate/providers/%s.yml found",
            provider_name,
            provider_name,
        )
        return None
    except Exception as exc:
        logger.error("Failed to load provider config for '%s': %s", provider_name, exc)
        return None

    api_key_env = provider.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        logger.error(
            "Missing API key: set the %s environment variable for provider '%s'",
            api_key_env,
            provider_name,
        )
        return None

    tracker = UsageTracker(usage_file)
    free_tier = provider.get("free_tier", {})

    if free_tier.get("enabled", False) and not tracker.can_send(provider):
        return None

    try:
        srt_content = srt_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read SRT file %s: %s", srt_path, exc)
        return None

    if free_tier.get("enabled", False):
        tracker.increment(provider_name)

    driver_name = provider.get("driver", "")
    try:
        driver = importlib.import_module(f"translate.drivers.{driver_name}")
    except ImportError as exc:
        logger.error("Cannot load driver '%s': %s", driver_name, exc)
        return None

    lang_name = _LANG_NAMES.get(target_lang.lower(), target_lang)

    try:
        translated = driver.translate(
            text=srt_content,
            target_lang=lang_name,
            model=provider["model"],
            endpoint=provider["endpoint"],
            api_key=api_key,
        )
    except TranslationError as exc:
        logger.error("Translation failed for %s: %s", srt_path.name, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error translating %s: %s", srt_path.name, exc)
        return None

    if not _validate_srt(translated):
        logger.warning(
            "Response for %s has no '-->' timestamp lines — looks malformed, skipping write",
            srt_path.name,
        )
        return None

    stem = srt_path.stem
    base = stem.rsplit(".", 1)[0] if "." in stem else stem
    output_path = srt_path.parent / f"{base}.{target_lang}.srt"

    try:
        output_path.write_text(translated, encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot write translated SRT %s: %s", output_path, exc)
        return None

    logger.info("Translated %s → %s", srt_path.name, output_path.name)
    return output_path
