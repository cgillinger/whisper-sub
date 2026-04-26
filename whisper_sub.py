#!/usr/bin/env python3
"""whisper_sub.py — Generate .srt subtitle files from video using faster-whisper.

Usage:
  Single file:   python whisper_sub.py <video_path> [options]
  Transcribe:    python whisper_sub.py transcribe <video_path> [options]
  Batch scan:    python whisper_sub.py scan <directory> [options]
  Config-driven: python whisper_sub.py scan --config /app/config.yml

Language logic:
  - Swedish (above confidence threshold) → transcribe → <filename>.sv.srt
  - Everything else → translate to English → <filename>.en.srt

KB-Whisper:
  When swedish_model differs from default_model the scan command uses a
  per-file pipeline that keeps only one large model in memory at a time:
    1. Detect language with the currently loaded model
    2. If Swedish → load KB-Whisper (if not already loaded) → transcribe
    3. If other  → load default model (if not already loaded) → translate
  The model is kept in memory between consecutive files of the same
  language; it is only swapped when the required model changes.

Home-video adaptations (Phase 6):
  Per-path settings in config scan_paths override global defaults:
    vad_filter        — skip silence, reduce hallucinations
    min_confidence    — confidence threshold below which default_language is used
    default_language  — fallback language when detection confidence is low
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".m4v", ".mov", ".wmv", ".ts"}
)
DEFAULT_STATE_FILE = Path.home() / ".emby-whisper-state.json"

# ---------------------------------------------------------------------------
# Global shutdown flag — set by SIGTERM / SIGINT handler
# ---------------------------------------------------------------------------

_shutdown_requested = False

# Set to False after the first OpenVINO load failure so that subsequent
# load_model() calls skip OpenVINO entirely and only one warning is logged.
_openvino_available: bool = True


def _handle_shutdown(signum: int, frame) -> None:  # noqa: ANN001
    """Signal handler: request a graceful shutdown after the current file."""
    global _shutdown_requested
    logger.warning(
        "Signal %d received — will stop after the current file and save state.",
        signum,
    )
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    """Convert faster-whisper segments to SRT-formatted string."""
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thermal monitoring
# ---------------------------------------------------------------------------


class ThermalAbortError(Exception):
    """Raised when a thermal pause exceeds the configured maximum duration."""


class NoAudioStreamError(RuntimeError):
    """Raised when a video file has no audio track."""


def read_cpu_temp() -> Optional[float]:
    """Read CPU package temperature from sysfs thermal zones.

    Searches for a zone typed 'x86_pkg_temp' or 'coretemp'.
    Returns temperature in °C, or None if unavailable.
    """
    base = Path("/sys/class/thermal")
    if not base.exists():
        return None
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            zone_type = (zone / "type").read_text().strip()
            if zone_type in ("x86_pkg_temp", "coretemp"):
                return int((zone / "temp").read_text().strip()) / 1000.0
        except OSError:
            continue
    return None


def read_nvme_temp() -> Optional[float]:
    """Read NVMe temperature from sysfs hwmon nodes.

    Identifies the NVMe hwmon by checking that hwmon*/name contains 'nvme'.
    Returns temperature in °C, or None if unavailable.
    """
    base = Path("/sys/class/hwmon")
    if not base.exists():
        return None
    for hwmon in sorted(base.glob("hwmon*")):
        try:
            name = (hwmon / "name").read_text().strip()
            if "nvme" in name.lower():
                # Prefer Sensor 1 (temp2) over Composite (temp1)
                for sensor in ("temp2_input", "temp1_input"):
                    temp_file = hwmon / sensor
                    if temp_file.exists():
                        return int(temp_file.read_text().strip()) / 1000.0
        except OSError:
            continue
    return None


class ThermalMonitor:
    """Background thread that monitors CPU and NVMe temperatures.

    Behaviour:
    - Checks temperatures every ``check_interval`` seconds.
    - When a threshold is exceeded the first time, logs a Thermal pause message
      and records the pause start time.
    - When all temperatures fall below ``resume_temp``, logs a Thermal resume
      message.
    - If temperatures stay above threshold for longer than ``max_pause_minutes``,
      sets the abort flag; callers should raise ThermalAbortError.

    Public interface for the scan loop:
    - ``start()`` / ``stop()`` — lifecycle management.
    - ``wait_until_cool()`` — blocking call before starting each file.
    - ``check_abort()`` — non-blocking; called between transcription segments.
    - ``reset_for_next_file()`` — clears abort flag after handling an aborted file.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._thread = threading.Thread(target=self._run, daemon=True, name="thermal-monitor")
        self._stop = threading.Event()
        self._abort = threading.Event()
        self._pause_start: Optional[float] = None
        self._currently_paused = False

    def start(self) -> None:
        """Start the background monitoring thread."""
        logger.debug("ThermalMonitor starting (check_interval=%ds)", self._cfg["check_interval"])
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread and wait for it to exit."""
        self._stop.set()
        self._thread.join(timeout=5)

    def check_abort(self) -> None:
        """Raise ThermalAbortError if the abort flag has been set.

        Call this between transcription segments.
        """
        if self._abort.is_set():
            raise ThermalAbortError(
                f"Thermal pause exceeded {self._cfg['max_pause_minutes']} minutes — aborting file"
            )

    def wait_until_cool(self) -> None:
        """Block until all temperatures are below the resume threshold.

        Called before starting each file in the queue.  Logs pause/resume
        messages on the first and last iteration respectively.
        """
        first = True
        while True:
            cpu = read_cpu_temp()
            nvme = read_nvme_temp()

            cpu_hot = cpu is not None and cpu >= self._cfg["cpu_pause_temp"]
            nvme_hot = nvme is not None and nvme >= self._cfg["nvme_pause_temp"]

            if cpu_hot or nvme_hot:
                if first:
                    if cpu_hot:
                        logger.warning(
                            "Thermal pause: CPU %.0f°C (threshold %d°C)",
                            cpu,
                            self._cfg["cpu_pause_temp"],
                        )
                    if nvme_hot:
                        logger.warning(
                            "Thermal pause: NVMe %.0f°C (threshold %d°C)",
                            nvme,
                            self._cfg["nvme_pause_temp"],
                        )
                    first = False
                time.sleep(self._cfg["check_interval"])
            else:
                if not first:
                    logger.info(
                        "Thermal resume: CPU %.0f°C, NVMe %.0f°C",
                        cpu if cpu is not None else 0.0,
                        nvme if nvme is not None else 0.0,
                    )
                return

    def reset_for_next_file(self) -> None:
        """Clear the abort flag and reset pause tracking.

        Call after handling a ThermalAbortError, before starting the next file.
        """
        self._abort.clear()
        self._pause_start = None
        self._currently_paused = False

    def _run(self) -> None:
        """Background thread body: checks temperatures at regular intervals."""
        while not self._stop.wait(timeout=self._cfg["check_interval"]):
            cpu = read_cpu_temp()
            nvme = read_nvme_temp()

            cpu_hot = cpu is not None and cpu >= self._cfg["cpu_pause_temp"]
            nvme_hot = nvme is not None and nvme >= self._cfg["nvme_pause_temp"]

            if cpu_hot or nvme_hot:
                if not self._currently_paused:
                    # First interval above threshold
                    self._currently_paused = True
                    self._pause_start = time.monotonic()
                    if cpu_hot:
                        logger.warning(
                            "Thermal pause: CPU %.0f°C (threshold %d°C)",
                            cpu,
                            self._cfg["cpu_pause_temp"],
                        )
                    if nvme_hot:
                        logger.warning(
                            "Thermal pause: NVMe %.0f°C (threshold %d°C)",
                            nvme,
                            self._cfg["nvme_pause_temp"],
                        )
                else:
                    # Still hot — check if we've exceeded the patience limit
                    assert self._pause_start is not None
                    paused_min = (time.monotonic() - self._pause_start) / 60.0
                    if paused_min >= self._cfg["max_pause_minutes"] and not self._abort.is_set():
                        logger.error(
                            "Thermal pause has exceeded %.0f minutes — aborting current file",
                            self._cfg["max_pause_minutes"],
                        )
                        self._abort.set()
            else:
                if self._currently_paused:
                    self._currently_paused = False
                    self._pause_start = None
                    logger.info(
                        "Thermal resume: CPU %.0f°C, NVMe %.0f°C",
                        cpu if cpu is not None else 0.0,
                        nvme if nvme is not None else 0.0,
                    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

# Default thermal config (mirrors config.yml defaults)
_DEFAULT_THERMAL: dict = {
    "enabled": False,
    "cpu_pause_temp": 72,
    "nvme_pause_temp": 70,
    "resume_temp": 65,
    "check_interval": 30,
    "max_pause_minutes": 30,
}


def load_config(config_path: Path) -> dict:
    """Load and return the YAML configuration file."""
    import yaml  # lazy import — only needed for config mode

    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Per-path configuration (Phase 6 — home-video adaptations)
# ---------------------------------------------------------------------------


@dataclass
class PathConfig:
    """Per-scan-path settings for VAD, confidence threshold, and language override.

    When ``min_confidence`` is None the global ``swedish_threshold`` is used.
    """

    path: Path
    min_confidence: Optional[float] = None
    vad_filter: bool = False
    min_silence_duration: float = 0.5   # seconds; converted to ms for faster-whisper
    default_language: Optional[str] = None  # language code used when confidence is low


def parse_scan_paths(raw: list) -> list[tuple[Path, "PathConfig"]]:
    """Parse the *scan_paths* list from config into (Path, PathConfig) pairs.

    Supports both the legacy string format and the Phase-6 dict format:

    Legacy (string)::

        scan_paths:
          - /media/Emby

    Per-path dict::

        scan_paths:
          - path: /media/Emby
            min_confidence: 0.7
          - path: /media/EgenFilm1
            min_confidence: 0.5
            vad_filter: true
            default_language: sv
    """
    result: list[tuple[Path, PathConfig]] = []
    for entry in raw:
        if isinstance(entry, str):
            p = Path(entry)
            result.append((p, PathConfig(path=p)))
        elif isinstance(entry, dict):
            p = Path(entry["path"])
            cfg = PathConfig(
                path=p,
                min_confidence=entry.get("min_confidence"),
                vad_filter=bool(entry.get("vad_filter", False)),
                min_silence_duration=float(entry.get("min_silence_duration", 0.5)),
                default_language=entry.get("default_language"),
            )
            result.append((p, cfg))
        else:
            logger.warning("Ignoring unrecognised scan_paths entry: %r", entry)
    return result


# ---------------------------------------------------------------------------
# Whisper helpers
# ---------------------------------------------------------------------------


def has_audio_stream(video_path: Path) -> bool:
    """Return True if *video_path* contains at least one audio stream.

    Uses ffprobe to inspect the file. Returns True (permissive) when ffprobe
    is unavailable or fails, so that existing decode-error handling stays in
    effect for ambiguous cases.
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            check=True,
        )
        return "audio" in result.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return True
    except subprocess.CalledProcessError:
        return True


def decode_audio_robust(video_path: Path, sampling_rate: int = 16000) -> "np.ndarray":
    """Decode a video's audio track to a 16 kHz mono float32 numpy array.

    First tries ``faster_whisper.audio.decode_audio`` (fast, in-process PyAV).
    On PyAV decode errors — typically caused by a single corrupt audio
    packet somewhere in the file — falls back to the ffmpeg CLI, which is
    tolerant of corrupt packets and simply logs + skips them.

    Returns a float32 numpy array of audio samples.
    Raises RuntimeError if both paths fail.
    """
    import numpy as np
    from faster_whisper.audio import decode_audio

    try:
        return decode_audio(str(video_path), sampling_rate=sampling_rate)
    except Exception as exc:
        logger.warning(
            "PyAV decode failed for %s (%s) — falling back to ffmpeg CLI.",
            video_path.name, exc,
        )

    import subprocess
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", str(video_path),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sampling_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg fallback failed for {video_path.name}: "
            f"{exc.stderr.decode('utf-8', errors='replace')[:500]}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary not found on PATH — required for robust audio decode."
        ) from exc

    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def detect_language(model, video_path: Path) -> tuple[str, float]:
    """Run language detection on the first ~30 seconds of the video.

    Returns a (language_code, probability) tuple.
    Raises NoAudioStreamError if the file has no audio track.
    """
    if not has_audio_stream(video_path):
        raise NoAudioStreamError(
            f"No audio stream found in {video_path.name}"
            " — skipping (not a bug, file has no audio track)"
        )
    logger.info("Detecting language from first 30 seconds of %s", video_path.name)
    audio = decode_audio_robust(video_path, sampling_rate=16000)
    clip = audio[: 16000 * 30]

    _, info = model.transcribe(
        clip,
        task="transcribe",
        language=None,
        beam_size=1,
        best_of=1,
        without_timestamps=True,
        max_new_tokens=1,
    )
    return info.language, info.language_probability


def detect_file_language(
    model,
    video_path: Path,
    swedish_threshold: float = 0.0,
    path_config: Optional[PathConfig] = None,
) -> dict:
    """Detect language for a single file and determine the appropriate task.

    Per-path settings (Phase 6):
    - ``path_config.min_confidence`` overrides *swedish_threshold* for this file.
    - When detection confidence is below the threshold and
      ``path_config.default_language`` is set, the default language is used
      instead of the detected one (useful for home videos where audio quality
      is poor and Whisper may misidentify the language).

    Returns a dict with keys: language, probability, task, suffix.
    """
    language, probability = detect_language(model, video_path)
    logger.info("Detected language: %s (probability=%.2f)", language, probability)

    # Per-path threshold takes precedence over the global one
    threshold = (
        path_config.min_confidence
        if path_config is not None and path_config.min_confidence is not None
        else swedish_threshold
    )

    # Apply default_language fallback when confidence is too low
    effective_language = language
    if path_config is not None and path_config.default_language is not None:
        if probability < threshold:
            effective_language = path_config.default_language
            logger.info(
                "Low confidence (%.2f < %.2f) for %s — using default_language=%s",
                probability, threshold, video_path.name, effective_language,
            )

    if effective_language == "sv" and probability >= threshold:
        task = "transcribe"
        suffix = ".sv.srt"
    else:
        task = "translate"
        suffix = ".en.srt"

    return {
        "language": effective_language,
        "probability": probability,
        "task": task,
        "suffix": suffix,
    }


def transcribe_file(
    video_path: Path,
    model,
    language: str,
    task: str,
    thermal: Optional[ThermalMonitor] = None,
    vad_filter: bool = False,
    min_silence_duration: float = 0.5,
) -> list:
    """Transcribe or translate the entire video file.

    Iterates the faster-whisper segment generator one segment at a time so
    that *thermal* abort requests are honoured between segments.

    Phase 6 additions:
    - *vad_filter*: enable Voice Activity Detection to skip silence and reduce
      hallucinations — recommended for home videos with background noise.
    - *min_silence_duration*: minimum silence length in seconds used by VAD to
      split audio segments (passed to faster-whisper as milliseconds).

    Returns a list of segment objects from faster-whisper.
    Raises ThermalAbortError if thermal monitoring requests an abort.
    """
    logger.info(
        "Running task=%s language=%s vad=%s on %s",
        task, language, vad_filter, video_path.name,
    )

    transcribe_kwargs: dict = {
        "task": task,
        "language": language if task == "transcribe" else None,
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": vad_filter,
        "hallucination_silence_threshold": 1.0,
        "no_speech_threshold": 0.4,
    }
    if vad_filter:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": int(min_silence_duration * 1000),
        }

    audio = decode_audio_robust(video_path, sampling_rate=16000)
    segments_gen, _info = model.transcribe(audio, **transcribe_kwargs)
    segments: list = []
    for segment in segments_gen:
        segments.append(segment)
        if thermal is not None:
            thermal.check_abort()
    return segments


def load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper WhisperModel.

    When *device* is ``"openvino"`` the first call attempts OpenVINO.  If
    that fails the module-level flag ``_openvino_available`` is set to False
    and a single warning is logged; all subsequent calls skip OpenVINO and
    use CPU directly without further warnings.
    """
    global _openvino_available
    from faster_whisper import WhisperModel

    logger.info(
        "Loading model=%s device=%s compute_type=%s",
        model_name,
        device,
        compute_type,
    )

    if device == "openvino":
        if _openvino_available:
            try:
                return WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception as exc:
                logger.warning(
                    "OpenVINO unavailable for model=%s (%s) — "
                    "falling back to CPU for this session.",
                    model_name,
                    exc,
                )
                _openvino_available = False
                return WhisperModel(model_name, device="cpu", compute_type=compute_type)
        # OpenVINO already known to be unavailable — use CPU without re-logging.
        return WhisperModel(model_name, device="cpu", compute_type=compute_type)

    return WhisperModel(model_name, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Core per-file processing (shared between single-file and batch modes)
# ---------------------------------------------------------------------------


def _transcribe_and_write(
    video_path: Path,
    model,
    language: str,
    task: str,
    suffix: str,
    force: bool,
    thermal: Optional[ThermalMonitor] = None,
    vad_filter: bool = False,
    min_silence_duration: float = 0.5,
) -> dict:
    """Transcribe/translate *video_path* and write the SRT file.

    Assumes language detection has already been performed by the caller.
    Returns a dict with keys: output, skipped.
    Raises on errors (including ThermalAbortError).
    """
    output_path = video_path.parent / f"{video_path.stem}{suffix}"

    if output_path.exists() and not force:
        logger.info(
            "Skipping %s: %s already exists (use --force to overwrite)",
            video_path.name,
            output_path.name,
        )
        return {"output": str(output_path), "skipped": True}

    segments = transcribe_file(
        video_path, model, language, task,
        thermal=thermal,
        vad_filter=vad_filter,
        min_silence_duration=min_silence_duration,
    )
    segments = split_segments_on_pause(segments)
    segments = clamp_segment_durations(segments)

    if not segments:
        logger.warning("No segments produced for %s", video_path.name)

    srt_content = segments_to_srt(segments)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info("Wrote %d segments to %s", len(segments), output_path)

    return {"output": str(output_path), "skipped": False}


def _process_one(
    video_path: Path,
    model,
    dry_run: bool,
    force: bool,
    thermal: Optional[ThermalMonitor] = None,
    swedish_threshold: float = 0.0,
    path_config: Optional[PathConfig] = None,
    vad_filter: bool = False,
    min_silence_duration: float = 0.5,
) -> dict:
    """Detect language and process a single video with a pre-loaded model.

    Used by single-model mode (single-file command and non-KB-Whisper scan).
    Per-path VAD and confidence settings from *path_config* override the
    global *vad_filter* / *swedish_threshold* values when provided.
    Returns a result dict with keys: language, task, output, skipped.
    Raises on errors so callers can decide how to handle them.
    """
    # Per-path config overrides CLI / global settings
    effective_vad = path_config.vad_filter if path_config is not None else vad_filter
    effective_silence = (
        path_config.min_silence_duration if path_config is not None else min_silence_duration
    )

    det = detect_file_language(
        model, video_path, swedish_threshold, path_config=path_config
    )

    output_path = video_path.parent / f"{video_path.stem}{det['suffix']}"

    if dry_run:
        print(
            f"[dry-run] {video_path.name}: language={det['language']} "
            f"({det['probability']:.2f}) → task={det['task']} "
            f"vad={effective_vad} → {output_path.name}"
        )
        return {
            "language": det["language"],
            "task": det["task"],
            "output": str(output_path),
            "skipped": False,
        }

    result = _transcribe_and_write(
        video_path, model,
        det["language"], det["task"], det["suffix"],
        force=force, thermal=thermal,
        vad_filter=effective_vad, min_silence_duration=effective_silence,
    )
    return {"language": det["language"], "task": det["task"], **result}


# ---------------------------------------------------------------------------
# State file management
# ---------------------------------------------------------------------------


def load_state(state_file: Path) -> dict:
    """Load processing state from a JSON file, or return a fresh state dict."""
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            data.setdefault("processed", {})
            data.setdefault("errors", {})
            data.setdefault("skipped", {})
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read state file %s: %s — starting fresh.", state_file, exc
            )
    return {"processed": {}, "errors": {}, "skipped": {}}


def save_state(state: dict, state_file: Path) -> None:
    """Persist processing state to a JSON file atomically (write + rename)."""
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)


# ---------------------------------------------------------------------------
# Video discovery and queue building
# ---------------------------------------------------------------------------


def find_videos(directory: Path, extensions: Optional[frozenset] = None) -> list[Path]:
    """Recursively find all video files under *directory*, sorted by path."""
    exts = extensions if extensions is not None else VIDEO_EXTENSIONS
    videos: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            videos.append(path)
    return videos


def has_subtitle(video_path: Path) -> bool:
    """Return True if a .sv.srt or .en.srt file exists alongside the video."""
    stem = video_path.stem
    parent = video_path.parent
    return (parent / f"{stem}.sv.srt").exists() or (parent / f"{stem}.en.srt").exists()


def build_queue(videos: list[Path], state: dict, force: bool) -> list[Path]:
    """Filter *videos* down to those that actually need processing.

    Skips files that (unless *force* is True):
    - Are already recorded as successfully processed in *state*.
    - Already have a .sv.srt or .en.srt file alongside them.
    """
    processed = state.get("processed", {})
    skipped = state.get("skipped", {})
    queue: list[Path] = []
    for video in videos:
        if not force:
            if str(video) in processed:
                logger.debug("Skipping (already processed): %s", video.name)
                continue
            if str(video) in skipped:
                logger.debug("Skipping (no audio stream): %s", video.name)
                continue
            if has_subtitle(video):
                logger.debug("Skipping (subtitle exists): %s", video.name)
                continue
        queue.append(video)
    return queue


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle single-file transcription (``transcribe`` subcommand or legacy positional mode).

    When ``--config`` is provided, model/device/compute-type/threshold/VAD
    defaults are read from the config file; CLI arguments take precedence.
    When ``--swedish-model`` (via CLI or config) differs from the default
    model, KB-Whisper two-model logic is used.
    """
    video_path: Path = args.path
    if not video_path.exists():
        logger.error("File not found: %s", video_path)
        return 1

    # Load optional config (only present for the ``transcribe`` subcommand;
    # absent in legacy positional mode — use getattr to avoid AttributeError).
    cfg: dict = {}
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = load_config(Path(config_path))

    # Resolve settings: CLI > config > hard-coded defaults.
    default_model_name: str = (
        args.model
        or cfg.get("default_model")
        or cfg.get("model", "large-v3")
    )
    swedish_model_name: Optional[str] = args.swedish_model or cfg.get("swedish_model")
    device: str = args.device or cfg.get("device", "cpu")
    compute_type: str = args.compute_type or cfg.get("compute_type", "int8")
    swedish_threshold: float = float(cfg.get("swedish_threshold", 0.0))
    use_kb = bool(swedish_model_name and swedish_model_name != default_model_name)

    vad: bool = args.vad_filter
    silence: float = args.min_silence_duration

    try:
        if not use_kb:
            model = load_model(default_model_name, device, compute_type)
            _process_one(
                video_path, model,
                dry_run=args.dry_run, force=args.force,
                swedish_threshold=swedish_threshold,
                vad_filter=vad, min_silence_duration=silence,
            )
        else:
            # KB-Whisper: detect with default model, swap to appropriate model for transcription.
            det_model = load_model(default_model_name, device, compute_type)
            det = detect_file_language(det_model, video_path, swedish_threshold)

            if args.dry_run:
                output_path = video_path.parent / f"{video_path.stem}{det['suffix']}"
                model_used = swedish_model_name if det["task"] == "transcribe" else default_model_name
                print(
                    f"[dry-run] {video_path.name}: language={det['language']} "
                    f"({det['probability']:.2f}) → task={det['task']} "
                    f"vad={vad} → model={model_used} → {output_path.name}"
                )
                return 0

            if det["task"] == "transcribe":
                # Swedish: swap to KB-Whisper model.
                del det_model
                sv_model = load_model(swedish_model_name, device, compute_type)
                _transcribe_and_write(
                    video_path, sv_model,
                    det["language"], det["task"], det["suffix"],
                    force=args.force,
                    vad_filter=vad, min_silence_duration=silence,
                )
            else:
                # Non-Swedish: reuse the already-loaded default model.
                _transcribe_and_write(
                    video_path, det_model,
                    det["language"], det["task"], det["suffix"],
                    force=args.force,
                    vad_filter=vad, min_silence_duration=silence,
                )
    except Exception as exc:
        logger.error("Failed to process %s: %s", video_path.name, exc)
        return 1
    return 0


def cmd_scan(args: argparse.Namespace) -> int:  # noqa: C901
    """Handle batch scan mode: find all videos and process them sequentially.

    When swedish_model differs from default_model (KB-Whisper mode) the
    command uses a per-file pipeline with in-memory model caching:
      For each file: detect language → select model → transcribe/translate
      The model is swapped only when the required model changes between files.
    """
    # ------------------------------------------------------------------
    # 1. Load config file (if provided)
    # ------------------------------------------------------------------
    cfg: dict = {}
    if args.config:
        cfg = load_config(Path(args.config))

    # ------------------------------------------------------------------
    # 2. Resolve settings: CLI args take precedence over config, then defaults
    # ------------------------------------------------------------------
    # default_model: used for language detection and non-Swedish transcription
    default_model_name: str = (
        args.model
        or cfg.get("default_model")
        or cfg.get("model", "large-v3")
    )
    # swedish_model: used for Swedish transcription (KB-Whisper)
    swedish_model_name: Optional[str] = (
        args.swedish_model or cfg.get("swedish_model")
    )
    use_kb_whisper = bool(
        swedish_model_name and swedish_model_name != default_model_name
    )

    device: str = args.device or cfg.get("device", "cpu")
    compute_type: str = args.compute_type or cfg.get("compute_type", "int8")
    limit: int = args.limit if args.limit != 0 else cfg.get("max_files_per_run", 0)
    force: bool = args.force
    swedish_threshold: float = float(cfg.get("swedish_threshold", 0.0))
    # CLI-level VAD settings (per-path config overrides these)
    cli_vad_filter: bool = getattr(args, "vad_filter", False)
    cli_min_silence: float = getattr(args, "min_silence_duration", 0.5)

    if use_kb_whisper:
        logger.info(
            "KB-Whisper mode: default_model=%s swedish_model=%s",
            default_model_name, swedish_model_name,
        )

    # Translation (optional post-processing step)
    translation_cfg: dict = cfg.get("translation", {})
    translation_enabled: bool = bool(translation_cfg.get("enabled", False))
    translation_provider: str = translation_cfg.get("provider", "gemini-lite")
    translation_target_lang: str = translation_cfg.get("target_language", "sv")

    # Video extensions (from config or global default)
    if "video_extensions" in cfg:
        video_exts = frozenset(
            e if e.startswith(".") else f".{e}" for e in cfg["video_extensions"]
        )
    else:
        video_exts = VIDEO_EXTENSIONS

    # ------------------------------------------------------------------
    # 3. Resolve scan paths with per-path config (Phase 6)
    #    CLI arg > config scan_paths (supports both string and dict entries)
    # ------------------------------------------------------------------
    if args.directory:
        # CLI directory: wrap in a PathConfig using CLI VAD settings
        cli_path_cfg = PathConfig(
            path=args.directory,
            vad_filter=cli_vad_filter,
            min_silence_duration=cli_min_silence,
        )
        scan_path_entries: list[tuple[Path, PathConfig]] = [
            (args.directory, cli_path_cfg)
        ]
    elif "scan_paths" in cfg:
        scan_path_entries = parse_scan_paths(cfg["scan_paths"])
    else:
        logger.error(
            "No directory specified. Provide a directory argument or "
            "--config with scan_paths."
        )
        return 1

    # ------------------------------------------------------------------
    # 4. State file
    # ------------------------------------------------------------------
    state_file = Path(args.state_file).expanduser()
    state = load_state(state_file)

    # ------------------------------------------------------------------
    # 5. Build queue across all scan directories, tracking per-path config
    # ------------------------------------------------------------------
    all_videos: list[Path] = []
    video_path_configs: dict[Path, PathConfig] = {}
    for d, path_cfg in scan_path_entries:
        if not d.is_dir():
            logger.warning("Scan path not found or not a directory: %s", d)
            continue
        logger.info(
            "Scanning %s (vad=%s min_confidence=%s default_language=%s)...",
            d, path_cfg.vad_filter, path_cfg.min_confidence, path_cfg.default_language,
        )
        found = find_videos(d, extensions=video_exts)
        logger.info("  %d video file(s) found in %s", len(found), d)
        all_videos.extend(found)
        for v in found:
            video_path_configs[v] = path_cfg

    logger.info("Total: %d video file(s) found.", len(all_videos))
    queue = build_queue(all_videos, state, force)
    logger.info("%d file(s) queued for processing.", len(queue))

    if args.dry_run:
        print(f"\nFiles that would be processed ({len(queue)}):")
        for video in queue:
            print(f"  {video}")
        return 0

    if not queue:
        logger.info("Nothing to do.")
        return 0

    # ------------------------------------------------------------------
    # 6. Thermal monitor (optional)
    # ------------------------------------------------------------------
    thermal_cfg: dict = {**_DEFAULT_THERMAL, **cfg.get("thermal", {})}
    thermal: Optional[ThermalMonitor] = None
    if thermal_cfg.get("enabled", False):
        thermal = ThermalMonitor(thermal_cfg)
        thermal.start()
        logger.info(
            "Thermal throttling enabled: CPU pause=%.0f°C, NVMe pause=%.0f°C, resume=%.0f°C",
            thermal_cfg["cpu_pause_temp"],
            thermal_cfg["nvme_pause_temp"],
            thermal_cfg["resume_temp"],
        )

    n_processed = 0
    n_errors = 0
    n_skipped = 0

    try:
        if not use_kb_whisper:
            # ----------------------------------------------------------
            # Single-model path (original behaviour)
            # ----------------------------------------------------------
            model = load_model(default_model_name, device, compute_type)

            for i, video in enumerate(queue, start=1):
                if _shutdown_requested:
                    logger.info(
                        "Shutdown requested — stopping after %d/%d file(s).",
                        i - 1, len(queue),
                    )
                    break
                if limit > 0 and n_processed >= limit:
                    logger.info("Limit of %d file(s) reached.", limit)
                    break

                if thermal is not None:
                    thermal.reset_for_next_file()
                    thermal.wait_until_cool()

                logger.info("[%d/%d] %s", i, len(queue), video)
                t0 = time.monotonic()

                try:
                    result = _process_one(
                        video, model,
                        dry_run=False, force=force,
                        thermal=thermal,
                        swedish_threshold=swedish_threshold,
                        path_config=video_path_configs.get(video),
                    )
                    elapsed = time.monotonic() - t0

                    if not result["skipped"]:
                        state["processed"][str(video)] = {
                            "language": result["language"],
                            "task": result["task"],
                            "output": result["output"],
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "duration_seconds": round(elapsed, 2),
                            "model": default_model_name,
                        }
                        n_processed += 1
                        logger.info(
                            "Done: %s → %s in %.1fs",
                            video.name, Path(result["output"]).name, elapsed,
                        )
                        if translation_enabled:
                            _apply_translation(
                                video, Path(result["output"]),
                                translation_target_lang, translation_provider,
                                state, state_file,
                            )

                except NoAudioStreamError:
                    logger.info("Skipping %s: no audio stream", video.name)
                    state.setdefault("skipped", {})[str(video)] = {
                        "reason": "no_audio_stream",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_skipped += 1

                except ThermalAbortError as exc:
                    elapsed = time.monotonic() - t0
                    logger.error("Thermal abort for %s after %.1fs: %s", video.name, elapsed, exc)
                    state.setdefault("errors", {})[str(video)] = {
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_errors += 1

                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    logger.error("Error processing %s (%.1fs): %s", video.name, elapsed, exc)
                    state.setdefault("errors", {})[str(video)] = {
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_errors += 1

                finally:
                    save_state(state, state_file)

        else:
            # ----------------------------------------------------------
            # KB-Whisper per-file pipeline (Uppgift 2)
            # Detect → select model → transcribe, one file at a time.
            # Model stays in memory; swaps only when the required model
            # changes (i.e. on a Swedish ↔ non-Swedish language boundary).
            # ----------------------------------------------------------
            current_model_obj = None
            current_model_name: Optional[str] = None

            for i, video in enumerate(queue, start=1):
                if _shutdown_requested:
                    logger.info(
                        "Shutdown requested — stopping after %d/%d file(s).",
                        i - 1, len(queue),
                    )
                    break
                if limit > 0 and n_processed >= limit:
                    logger.info("Limit of %d file(s) reached.", limit)
                    break

                if thermal is not None:
                    thermal.reset_for_next_file()
                    thermal.wait_until_cool()

                logger.info("[%d/%d] %s", i, len(queue), video.name)

                # Load the default model on the first iteration.
                if current_model_obj is None:
                    current_model_name = default_model_name
                    current_model_obj = load_model(default_model_name, device, compute_type)

                t0 = time.monotonic()
                try:
                    path_cfg: Optional[PathConfig] = video_path_configs.get(video)

                    # Step 1: detect language with the currently loaded model.
                    det = detect_file_language(
                        current_model_obj, video, swedish_threshold,
                        path_config=path_cfg,
                    )

                    # Step 2: determine the correct model for transcription.
                    needed_model_name = (
                        swedish_model_name if det["task"] == "transcribe" else default_model_name
                    )

                    # Step 3: swap model only when language changed.
                    if current_model_name != needed_model_name:
                        logger.info(
                            "Model swap: %s → %s (language=%s)",
                            current_model_name, needed_model_name, det["language"],
                        )
                        del current_model_obj
                        current_model_obj = load_model(needed_model_name, device, compute_type)
                        current_model_name = needed_model_name

                    # Step 4: transcribe / translate.
                    vad = path_cfg.vad_filter if path_cfg is not None else False
                    silence = path_cfg.min_silence_duration if path_cfg is not None else 0.5

                    result = _transcribe_and_write(
                        video, current_model_obj,
                        det["language"], det["task"], det["suffix"],
                        force=force, thermal=thermal,
                        vad_filter=vad, min_silence_duration=silence,
                    )
                    elapsed = time.monotonic() - t0

                    if not result["skipped"]:
                        state["processed"][str(video)] = {
                            "language": det["language"],
                            "task": det["task"],
                            "output": result["output"],
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "duration_seconds": round(elapsed, 2),
                            "model": current_model_name,
                        }
                        n_processed += 1
                        logger.info(
                            "Done: %s → %s in %.1fs",
                            video.name, Path(result["output"]).name, elapsed,
                        )
                        if translation_enabled:
                            _apply_translation(
                                video, Path(result["output"]),
                                translation_target_lang, translation_provider,
                                state, state_file,
                            )

                except NoAudioStreamError:
                    logger.info("Skipping %s: no audio stream", video.name)
                    state.setdefault("skipped", {})[str(video)] = {
                        "reason": "no_audio_stream",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_skipped += 1

                except ThermalAbortError as exc:
                    elapsed = time.monotonic() - t0
                    logger.error("Thermal abort for %s after %.1fs: %s", video.name, elapsed, exc)
                    state.setdefault("errors", {})[str(video)] = {
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_errors += 1

                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    logger.error("Error processing %s (%.1fs): %s", video.name, elapsed, exc)
                    state.setdefault("errors", {})[str(video)] = {
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    n_errors += 1

                finally:
                    save_state(state, state_file)

            if current_model_obj is not None:
                del current_model_obj

    finally:
        if thermal is not None:
            thermal.stop()

    logger.info(
        "Scan complete: %d processed, %d skipped (no audio), %d error(s). State: %s",
        n_processed, n_skipped, n_errors, state_file,
    )
    return 0 if n_errors == 0 else 1


def _apply_translation(
    video: Path,
    srt_path: Path,
    target_lang: str,
    provider_name: str,
    state: dict,
    state_file: Path,
) -> None:
    """Translate *srt_path* and record the result in *state* if successful."""
    lang_suffix = f".{target_lang}.srt"
    if srt_path.name.endswith(lang_suffix):
        return

    try:
        from translate.translator import translate_srt as _translate_srt
    except ImportError:
        logger.warning("translate module not available — skipping translation")
        return

    tr_path = _translate_srt(
        srt_path, target_lang, provider_name, state,
        usage_file=state_file.parent / "translate_usage.json",
    )
    if tr_path is not None:
        state["processed"][str(video)]["translated"] = True
        save_state(state, state_file)
        logger.info("Translation written: %s", tr_path.name)


def cmd_translate(args: argparse.Namespace) -> int:
    """Handle standalone translate subcommand for existing SRT files."""
    cfg: dict = {}
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = load_config(Path(config_path))

    provider_name: str = (
        args.provider
        or cfg.get("translation", {}).get("provider", "gemini-lite")
    )
    target_lang: str = (
        args.lang
        or cfg.get("translation", {}).get("target_language", "sv")
    )
    state_file = Path(args.state_file).expanduser()
    usage_file = state_file.parent / "translate_usage.json"

    target: Path = args.path
    if target.is_file():
        srt_files = [target]
    elif target.is_dir():
        srt_files = sorted(target.rglob("*.en.srt"))
    else:
        logger.error("Path not found: %s", target)
        return 1

    if not srt_files:
        logger.info("No .en.srt files found in %s", target)
        return 0

    state = load_state(state_file)
    n_ok = 0
    n_fail = 0

    try:
        from translate.translator import translate_srt
    except ImportError:
        logger.error("translate module not found — cannot run translation")
        return 1

    delay: float = getattr(args, "delay", 2.0)
    n_skipped = 0

    for i, srt_path in enumerate(srt_files):
        # Skip if translated file already exists
        translated_path = srt_path.with_name(
            srt_path.name.replace(".en.srt", f".{target_lang}.srt")
        )
        if translated_path.exists():
            n_skipped += 1
            continue

        if _shutdown_requested:
            logger.info("Shutdown requested — stopping translation.")
            break

        logger.info(
            "[%d/%d] Translating %s → %s (%s)",
            i + 1, len(srt_files), srt_path.name, target_lang, provider_name,
        )
        result = translate_srt(
            srt_path, target_lang, provider_name, state,
            usage_file=usage_file,
        )
        if result is not None:
            n_ok += 1
        else:
            n_fail += 1

        # Rate-limit delay (skip after the last file)
        if delay > 0 and i < len(srt_files) - 1:
            time.sleep(delay)

    logger.info(
        "Translation complete: %d translated, %d skipped (existing), %d failed.",
        n_ok, n_skipped, n_fail,
    )
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_model_args(parser: argparse.ArgumentParser, defaults: bool = True) -> None:
    """Add shared Whisper model arguments to *parser*.

    When *defaults* is False, model/device/compute-type default to None so
    that config-file values can take precedence in cmd_scan.
    """
    default_model = "large-v3" if defaults else None
    default_device = "cpu" if defaults else None
    default_ct = "int8" if defaults else None

    parser.add_argument(
        "--model",
        default=default_model,
        metavar="MODEL",
        help="Whisper model for language detection and non-Swedish transcription.",
    )
    parser.add_argument(
        "--swedish-model",
        default=None,
        dest="swedish_model",
        metavar="MODEL",
        help=(
            "Whisper model used for Swedish transcription (KB-Whisper). "
            "When set and different from --model, enables two-pass processing: "
            "detect all languages first, then transcribe Swedish with this model "
            "and translate everything else with --model. "
            "Example: KBLab/kb-whisper-large"
        ),
    )
    parser.add_argument(
        "--device",
        default=default_device,
        choices=["cpu", "cuda", "openvino", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--compute-type",
        default=default_ct,
        dest="compute_type",
        choices=["int8", "float16", "float32"],
        help="Quantisation type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .srt files / ignore state.",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        dest="vad_filter",
        help=(
            "Enable Voice Activity Detection to skip silence and reduce "
            "hallucinations. Recommended for home videos with background noise."
        ),
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.5,
        dest="min_silence_duration",
        metavar="SECONDS",
        help="Minimum silence duration in seconds used by VAD (default: 0.5).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SRT subtitles from video using faster-whisper.\n\n"
            "  Single file:   whisper_sub.py <video_path> [options]\n"
            "  Batch scan:    whisper_sub.py scan <directory> [options]\n"
            "  Config-driven: whisper_sub.py scan --config /app/config.yml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # scan subcommand
    # ------------------------------------------------------------------
    scan_p = subparsers.add_parser(
        "scan",
        help="Recursively scan a directory and transcribe all videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scan_p.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help=(
            "Directory to scan recursively. "
            "Omit when using --config with scan_paths defined."
        ),
    )
    scan_p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a config.yml file (Docker mode).",
    )
    scan_p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without doing any work.",
    )
    scan_p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Stop after processing N files (0 = unlimited).",
    )
    scan_p.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        metavar="PATH",
        help="JSON file used to track processed files across runs.",
    )
    _add_model_args(scan_p, defaults=False)

    # ------------------------------------------------------------------
    # transcribe subcommand — single file, optional config
    # ------------------------------------------------------------------
    transcribe_p = subparsers.add_parser(
        "transcribe",
        help="Detect language and transcribe (or translate) a single video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    transcribe_p.add_argument(
        "path",
        type=Path,
        help="Path to the video file.",
    )
    transcribe_p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Optional config.yml to load model/device/threshold/VAD defaults from. "
            "CLI arguments override config values."
        ),
    )
    transcribe_p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Detect language and print planned action; do not write any file.",
    )
    _add_model_args(transcribe_p, defaults=False)

    # ------------------------------------------------------------------
    # translate subcommand — translate existing .en.srt files
    # ------------------------------------------------------------------
    translate_p = subparsers.add_parser(
        "translate",
        help="Translate existing .en.srt files to another language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    translate_p.add_argument(
        "path",
        type=Path,
        help="Path to a .en.srt file, or a directory to scan for .en.srt files.",
    )
    translate_p.add_argument(
        "--provider",
        default=None,
        metavar="PROVIDER",
        help=(
            "Translation provider to use "
            "(gemini-lite, gemini, openai, deepseek, anthropic). "
            "Default: gemini-lite."
        ),
    )
    translate_p.add_argument(
        "--lang",
        default=None,
        metavar="LANG",
        help="Target language ISO code (e.g. sv for Swedish). Default: sv.",
    )
    translate_p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Optional config.yml to read translation defaults from.",
    )
    translate_p.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        dest="state_file",
        metavar="PATH",
        help="JSON state file (used to place translate_usage.json alongside it).",
    )
    translate_p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Delay in seconds between API calls to avoid rate limits (default: 2.0).",
    )

    # ------------------------------------------------------------------
    # Single-file mode (no subcommand) — args on the main parser
    # NOTE: no top-level "path" positional here; the legacy positional path
    # is handled by the sf_parser pre-detection in main() before argparse
    # ever sees it.  Adding a top-level "path" positional causes argparse to
    # overwrite the transcribe subcommand's own "path" with None.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect language and print planned action; do not write any file.",
    )
    _add_model_args(parser, defaults=True)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the whisper_sub CLI."""
    # Single-file mode detection: argparse greedily assigns the first
    # positional token to the subparsers action (raising "invalid choice")
    # even when subparsers are optional.  Pre-detect: if the first non-option
    # argument is not a known subcommand, parse with a dedicated single-file
    # parser that has no subparsers.
    _known_subcommands = {"scan", "transcribe", "translate"}
    argv = sys.argv[1:]
    first_positional = next((a for a in argv if not a.startswith("-")), None)

    if first_positional is not None and first_positional not in _known_subcommands:
        sf_parser = argparse.ArgumentParser(
            description="Generate SRT subtitles from a single video file."
        )
        sf_parser.add_argument("path", type=Path, help="Path to a video file.")
        sf_parser.add_argument(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            help="Detect language and print planned action; do not write any file.",
        )
        _add_model_args(sf_parser, defaults=True)
        args = sf_parser.parse_args(argv)
        sys.exit(cmd_transcribe(args))

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        sys.exit(cmd_scan(args))
    elif args.command == "transcribe":
        sys.exit(cmd_transcribe(args))
    elif args.command == "translate":
        sys.exit(cmd_translate(args))
    else:
        parser.print_help()
        sys.exit(1)


def split_segments_on_pause(
    segments: list,
    min_pause: float = 0.8,
) -> list:
    """Split segments that contain a long pause between words.

    When word_timestamps are available, any gap >= *min_pause* seconds
    between consecutive words triggers a split into two segments.
    Segments without word-level data are passed through unchanged.
    """
    from types import SimpleNamespace
    result = []
    for seg in segments:
        words = getattr(seg, "words", None)
        if not words or len(words) < 2:
            result.append(seg)
            continue
        # Find split points: gaps >= min_pause between consecutive words
        split_indices = []
        for i in range(1, len(words)):
            gap = words[i].start - words[i - 1].end
            if gap >= min_pause:
                split_indices.append(i)
        if not split_indices:
            result.append(seg)
            continue
        # Build sub-segments, but only keep splits where both sides
        # have at least 2 words (avoids orphaned single words)
        boundaries = [0] + split_indices + [len(words)]
        valid = [0]
        for j in range(1, len(boundaries) - 1):
            left_count = boundaries[j] - valid[-1]
            right_count = boundaries[j + 1] - boundaries[j]
            if left_count >= 2 and right_count >= 2:
                valid.append(boundaries[j])
        valid.append(len(words))
        for j in range(len(valid) - 1):
            chunk = words[valid[j]:valid[j + 1]]
            text = "".join(w.word for w in chunk).strip()
            if text:
                result.append(SimpleNamespace(
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=" " + text,
                    words=chunk,
                ))
    return result


def clamp_segment_durations(
    segments: list,
    max_duration: float = 7.0,
    min_chars_per_sec: float = 15.0,
) -> list:
    """Post-process segments to cap display duration.

    For each segment the duration is clamped to the shorter of:
      - max_duration (default 7 s)
      - len(text) / min_chars_per_sec (so short lines vanish quickly)
    Minimum duration is 2 s regardless of text length.
    """
    from types import SimpleNamespace

    clamped = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        text_dur = max(2.0, len(text) / min_chars_per_sec)
        ideal_end = seg.start + min(max_duration, text_dur)
        # Cap at next segment start to prevent overlap
        next_start = segments[i + 1].start if i + 1 < len(segments) else None
        # Minimum display time: 1s per 10 chars, floor 2s
        min_display = max(2.0, len(text) / 10.0)
        if ideal_end < seg.end:
            end = ideal_end
        elif seg.end < seg.start + min_display:
            end = seg.start + min_display
        else:
            end = seg.end
        if next_start is not None and end > next_start:
            end = next_start
        if end > seg.end or end < seg.end or end != seg.end:
            clamped.append(SimpleNamespace(start=seg.start, end=end, text=seg.text))
        else:
            clamped.append(seg)
    return clamped


if __name__ == "__main__":
    main()
