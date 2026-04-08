#!/usr/bin/env python3
"""whisper_sub.py — Generate .srt subtitle files from video using faster-whisper.

Language logic:
  - Swedish → transcribe → <filename>.sv.srt
  - Any other language → translate to English → <filename>.en.srt
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def detect_language(model, video_path: Path) -> tuple[str, float]:
    """Run language detection on the first ~30 seconds of the video.

    Returns a (language_code, probability) tuple.
    """
    import numpy as np

    logger.info("Detecting language from first 30 seconds of %s", video_path.name)

    # Use faster-whisper's decode_audio helper to load audio
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(video_path), sampling_rate=16000)
    # Clip to first 30 seconds (16 000 samples/s)
    clip = audio[: 16000 * 30]

    _, info = model.transcribe(
        clip,
        task="transcribe",
        language=None,  # auto-detect
        beam_size=1,
        best_of=1,
        without_timestamps=True,
        max_new_tokens=1,  # we only need detection metadata
    )
    return info.language, info.language_probability


def transcribe_file(
    video_path: Path,
    model,
    language: str,
    task: str,
) -> list:
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
    # Materialise the generator so any errors surface here
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


def process_video(
    video_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    dry_run: bool,
    force: bool,
) -> int:
    """Main processing pipeline for a single video file.

    Returns 0 on success, 1 on error.
    """
    if not video_path.exists():
        logger.error("File not found: %s", video_path)
        return 1

    model = load_model(model_name, device, compute_type)
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

    logger.info("Planned output: %s (task=%s)", output_path.name, task)

    if dry_run:
        print(
            f"[dry-run] language={language} probability={probability:.2f} "
            f"task={task} output={output_path}"
        )
        return 0

    if output_path.exists() and not force:
        logger.info(
            "Output file already exists: %s (use --force to overwrite)",
            output_path,
        )
        return 0

    segments = transcribe_file(video_path, model, language, task)

    if not segments:
        logger.warning("No segments produced for %s", video_path.name)

    srt_content = segments_to_srt(segments)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info("Wrote %d segments to %s", len(segments), output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from a video file using faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="Path to the video file.")
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
        "--dry-run",
        action="store_true",
        help="Detect language and print planned action; do not write any file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .srt file.",
    )
    return parser


def main() -> None:
    """Entry point for the whisper_sub CLI."""
    parser = build_parser()
    args = parser.parse_args()

    exit_code = process_video(
        video_path=args.path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        dry_run=args.dry_run,
        force=args.force,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
