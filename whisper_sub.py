#!/usr/bin/env python3
"""whisper_sub.py — Generate .srt subtitle files from video using faster-whisper.

Usage:
  Single file:  python whisper_sub.py <video_path> [options]
  Batch scan:   python whisper_sub.py scan <directory> [options]
  Config-driven: python whisper_sub.py scan --config /app/config.yml

Language logic:
  - Swedish (above confidence threshold) → transcribe → <filename>.sv.srt
  - Everything else → translate to English → <filename>.en.srt
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
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
                temp_file = hwmon / "temp1_input"
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
# Whisper helpers
# ---------------------------------------------------------------------------


def detect_language(model, video_path: Path) -> tuple[str, float]:
    """Run language detection on the first ~30 seconds of the video.

    Returns a (language_code, probability) tuple.
    """
    from faster_whisper.audio import decode_audio

    logger.info("Detecting language from first 30 seconds of %s", video_path.name)
    audio = decode_audio(str(video_path), sampling_rate=16000)
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


def transcribe_file(
    video_path: Path,
    model,
    language: str,
    task: str,
    thermal: Optional[ThermalMonitor] = None,
) -> list:
    """Transcribe or translate the entire video file.

    Iterates the faster-whisper segment generator one segment at a time so
    that *thermal* abort requests are honoured between segments.

    Returns a list of segment objects from faster-whisper.
    Raises ThermalAbortError if thermal monitoring requests an abort.
    """
    logger.info(
        "Running task=%s language=%s on %s", task, language, video_path.name
    )
    segments_gen, _info = model.transcribe(
        str(video_path),
        task=task,
        language=language if task == "transcribe" else None,
        beam_size=5,
    )
    segments: list = []
    for segment in segments_gen:
        segments.append(segment)
        if thermal is not None:
            thermal.check_abort()
    return segments


def load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper WhisperModel."""
    from faster_whisper import WhisperModel

    logger.info(
        "Loading model=%s device=%s compute_type=%s",
        model_name,
        device,
        compute_type,
    )
    return WhisperModel(model_name, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Core per-file processing (shared between single-file and batch modes)
# ---------------------------------------------------------------------------


def _process_one(
    video_path: Path,
    model,
    dry_run: bool,
    force: bool,
    thermal: Optional[ThermalMonitor] = None,
    swedish_threshold: float = 0.0,
) -> dict:
    """Process a single video with a pre-loaded model.

    Returns a result dict with keys: language, task, output, skipped.
    Raises on transcription / IO errors (including ThermalAbortError) so
    callers can decide how to handle them.
    """
    language, probability = detect_language(model, video_path)
    logger.info(
        "Detected language: %s (probability=%.2f)", language, probability
    )

    if language == "sv" and probability >= swedish_threshold:
        task = "transcribe"
        suffix = ".sv.srt"
    else:
        task = "translate"
        suffix = ".en.srt"

    stem = video_path.stem
    output_path = video_path.parent / f"{stem}{suffix}"

    if dry_run:
        print(
            f"[dry-run] {video_path.name}: language={language} "
            f"({probability:.2f}) → task={task} → {output_path.name}"
        )
        return {"language": language, "task": task, "output": str(output_path), "skipped": False}

    if output_path.exists() and not force:
        logger.info(
            "Skipping %s: %s already exists (use --force to overwrite)",
            video_path.name,
            output_path.name,
        )
        return {"language": language, "task": task, "output": str(output_path), "skipped": True}

    segments = transcribe_file(video_path, model, language, task, thermal=thermal)

    if not segments:
        logger.warning("No segments produced for %s", video_path.name)

    srt_content = segments_to_srt(segments)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info("Wrote %d segments to %s", len(segments), output_path)

    return {"language": language, "task": task, "output": str(output_path), "skipped": False}


# ---------------------------------------------------------------------------
# State file management
# ---------------------------------------------------------------------------


def load_state(state_file: Path) -> dict:
    """Load processing state from a JSON file, or return a fresh state dict."""
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read state file %s: %s — starting fresh.", state_file, exc
            )
    return {"processed": {}, "errors": {}}


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
    queue: list[Path] = []
    for video in videos:
        if not force:
            if str(video) in processed:
                logger.debug("Skipping (already processed): %s", video.name)
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
    """Handle single-file transcription mode."""
    video_path: Path = args.path
    if not video_path.exists():
        logger.error("File not found: %s", video_path)
        return 1

    model = load_model(args.model, args.device, args.compute_type)
    try:
        _process_one(video_path, model, dry_run=args.dry_run, force=args.force)
    except Exception as exc:
        logger.error("Failed to process %s: %s", video_path.name, exc)
        return 1
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Handle batch scan mode: find all videos and process them sequentially."""
    # ------------------------------------------------------------------
    # 1. Load config file (if provided)
    # ------------------------------------------------------------------
    cfg: dict = {}
    if args.config:
        cfg = load_config(Path(args.config))

    # ------------------------------------------------------------------
    # 2. Resolve settings: CLI args take precedence over config, then defaults
    # ------------------------------------------------------------------
    model_name: str = args.model or cfg.get("model", "large-v3")
    device: str = args.device or cfg.get("device", "cpu")
    compute_type: str = args.compute_type or cfg.get("compute_type", "int8")
    limit: int = args.limit if args.limit != 0 else cfg.get("max_files_per_run", 0)
    force: bool = args.force
    swedish_threshold: float = float(cfg.get("swedish_threshold", 0.0))

    # Video extensions (from config or global default)
    if "video_extensions" in cfg:
        video_exts = frozenset(
            e if e.startswith(".") else f".{e}" for e in cfg["video_extensions"]
        )
    else:
        video_exts = VIDEO_EXTENSIONS

    # ------------------------------------------------------------------
    # 3. Resolve scan directories: CLI arg > config scan_paths > error
    # ------------------------------------------------------------------
    if args.directory:
        scan_dirs: list[Path] = [args.directory]
    elif "scan_paths" in cfg:
        scan_dirs = [Path(p) for p in cfg["scan_paths"]]
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
    # 5. Build queue across all scan directories
    # ------------------------------------------------------------------
    all_videos: list[Path] = []
    for d in scan_dirs:
        if not d.is_dir():
            logger.warning("Scan path not found or not a directory: %s", d)
            continue
        logger.info("Scanning %s for video files...", d)
        found = find_videos(d, extensions=video_exts)
        logger.info("  %d video file(s) found in %s", len(found), d)
        all_videos.extend(found)

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

    # ------------------------------------------------------------------
    # 7. Load model
    # ------------------------------------------------------------------
    model = load_model(model_name, device, compute_type)

    # ------------------------------------------------------------------
    # 8. Process queue
    # ------------------------------------------------------------------
    n_processed = 0
    n_errors = 0

    try:
        for i, video in enumerate(queue, start=1):
            if _shutdown_requested:
                logger.info(
                    "Shutdown requested — stopping after %d/%d file(s). State saved.",
                    i - 1,
                    len(queue),
                )
                break

            if limit > 0 and n_processed >= limit:
                logger.info("Limit of %d file(s) reached.", limit)
                break

            # Thermal check before each file (spec point 1)
            if thermal is not None:
                thermal.reset_for_next_file()
                thermal.wait_until_cool()

            logger.info("[%d/%d] %s", i, len(queue), video)
            t0 = time.monotonic()

            try:
                result = _process_one(
                    video,
                    model,
                    dry_run=False,
                    force=force,
                    thermal=thermal,
                    swedish_threshold=swedish_threshold,
                )
                elapsed = time.monotonic() - t0

                if not result["skipped"]:
                    state["processed"][str(video)] = {
                        "language": result["language"],
                        "task": result["task"],
                        "output": result["output"],
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "duration_seconds": round(elapsed, 2),
                        "model": model_name,
                    }
                    n_processed += 1
                    logger.info(
                        "Done: %s → %s in %.1fs",
                        video.name,
                        Path(result["output"]).name,
                        elapsed,
                    )

            except ThermalAbortError as exc:
                elapsed = time.monotonic() - t0
                logger.error(
                    "Thermal abort for %s after %.1fs: %s", video.name, elapsed, exc
                )
                state.setdefault("errors", {})[str(video)] = {
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                n_errors += 1

            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.error(
                    "Error processing %s (%.1fs): %s", video.name, elapsed, exc
                )
                state.setdefault("errors", {})[str(video)] = {
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                n_errors += 1

            finally:
                save_state(state, state_file)

    finally:
        if thermal is not None:
            thermal.stop()

    logger.info(
        "Scan complete: %d processed, %d error(s). State: %s",
        n_processed,
        n_errors,
        state_file,
    )
    return 0 if n_errors == 0 else 1


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
        help="Whisper model name (e.g. large-v3, medium, small).",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        choices=["cpu", "cuda", "auto"],
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
    # Single-file mode (no subcommand) — args on the main parser
    # ------------------------------------------------------------------
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to a video file (single-file mode).",
    )
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
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        sys.exit(cmd_scan(args))
    else:
        if args.path is None:
            parser.print_help()
            sys.exit(1)
        sys.exit(cmd_transcribe(args))


if __name__ == "__main__":
    main()
