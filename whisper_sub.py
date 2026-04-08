#!/usr/bin/env python3
"""whisper_sub.py — Generate .srt subtitle files from video using faster-whisper.

Usage:
  Single file:  python whisper_sub.py <video_path> [options]
  Batch scan:   python whisper_sub.py scan <directory> [options]

Language logic:
  - Swedish → transcribe → <filename>.sv.srt
  - Any other language → translate to English → <filename>.en.srt
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

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


def transcribe_file(video_path: Path, model, language: str, task: str) -> list:
    """Transcribe or translate the entire video file.

    Returns a list of segment objects from faster-whisper.
    """
    logger.info(
        "Running task=%s language=%s on %s", task, language, video_path.name
    )
    segments, _info = model.transcribe(
        str(video_path),
        task=task,
        language=language if task == "transcribe" else None,
        beam_size=5,
    )
    return list(segments)


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
) -> dict:
    """Process a single video with a pre-loaded model.

    Returns a result dict with keys: language, task, output, skipped.
    Raises on transcription / IO errors so callers can decide how to handle them.
    """
    language, probability = detect_language(model, video_path)
    logger.info(
        "Detected language: %s (probability=%.2f)", language, probability
    )

    if language == "sv":
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

    segments = transcribe_file(video_path, model, language, task)

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
            logger.warning("Could not read state file %s: %s — starting fresh.", state_file, exc)
    return {"processed": {}, "errors": {}}


def save_state(state: dict, state_file: Path) -> None:
    """Persist processing state to a JSON file atomically (write + rename)."""
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)


# ---------------------------------------------------------------------------
# Video discovery and queue building
# ---------------------------------------------------------------------------


def find_videos(directory: Path) -> list[Path]:
    """Recursively find all video files under *directory*, sorted by path."""
    videos: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
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
    """Handle batch scan mode: find all videos under a directory and process them."""
    directory: Path = args.directory
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return 1

    state_file = Path(args.state_file).expanduser()
    state = load_state(state_file)

    logger.info("Scanning %s for video files...", directory)
    videos = find_videos(directory)
    logger.info("Found %d video file(s) total.", len(videos))

    queue = build_queue(videos, state, args.force)
    logger.info("%d file(s) queued for processing.", len(queue))

    if args.dry_run:
        print(f"\nFiles that would be processed ({len(queue)}):")
        for video in queue:
            print(f"  {video}")
        return 0

    if not queue:
        logger.info("Nothing to do.")
        return 0

    model = load_model(args.model, args.device, args.compute_type)

    n_processed = 0
    n_errors = 0

    for i, video in enumerate(queue, start=1):
        if _shutdown_requested:
            logger.info(
                "Shutdown requested — stopping after %d/%d file(s). State saved.",
                i - 1,
                len(queue),
            )
            break

        if args.limit > 0 and n_processed >= args.limit:
            logger.info("Limit of %d file(s) reached.", args.limit)
            break

        logger.info("[%d/%d] %s", i, len(queue), video)
        t0 = time.monotonic()

        try:
            result = _process_one(video, model, dry_run=False, force=args.force)
            elapsed = time.monotonic() - t0

            if not result["skipped"]:
                state["processed"][str(video)] = {
                    "language": result["language"],
                    "task": result["task"],
                    "output": result["output"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": round(elapsed, 2),
                    "model": args.model,
                }
                n_processed += 1
                logger.info(
                    "Done: %s → %s in %.1fs",
                    video.name,
                    Path(result["output"]).name,
                    elapsed,
                )

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


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add shared Whisper model arguments to *parser*."""
    parser.add_argument(
        "--model",
        default="large-v3",
        metavar="MODEL",
        help="Whisper model name (e.g. large-v3, medium, small).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
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
            "  Single file:  whisper_sub.py <video_path> [options]\n"
            "  Batch scan:   whisper_sub.py scan <directory> [options]"
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
    scan_p.add_argument("directory", type=Path, help="Directory to scan recursively.")
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
    _add_model_args(scan_p)

    # ------------------------------------------------------------------
    # Single-file mode (no subcommand) — attach args to the main parser
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
    _add_model_args(parser)

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
        # Single-file mode
        if args.path is None:
            parser.print_help()
            sys.exit(1)
        sys.exit(cmd_transcribe(args))


if __name__ == "__main__":
    main()
