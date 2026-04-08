# whisper-sub

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Automatic subtitle generation for Emby using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Transcribes Swedish audio to `.sv.srt` and translates everything else to `.en.srt`, dropping subtitle files next to the video so Emby picks them up without any plugin.

---

## Requirements

- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend)
- `ffmpeg` available on `PATH` (used by faster-whisper for audio decoding)

## Installation

```bash
git clone https://github.com/cgillinger/emby-whisper.git
cd emby-whisper
pip install -r requirements.txt
```

## Usage

```bash
# Transcribe / translate a single video
python whisper_sub.py /path/to/video.mp4

# Detect language only — no file written
python whisper_sub.py /path/to/video.mp4 --dry-run

# Use a smaller, faster model
python whisper_sub.py /path/to/video.mp4 --model small

# Force overwrite of an existing subtitle
python whisper_sub.py /path/to/video.mp4 --force
```

### Arguments

| Argument        | Default     | Description                                      |
|-----------------|-------------|--------------------------------------------------|
| `path`          | *(required)*| Path to the video file                           |
| `--model`       | `large-v3`  | Whisper model (`large-v3`, `medium`, `small`, …) |
| `--device`      | `cpu`       | Compute device (`cpu`, `cuda`, `auto`)           |
| `--compute-type`| `int8`      | Quantisation (`int8`, `float16`, `float32`)      |
| `--dry-run`     | false       | Print detected language and planned action only  |
| `--force`       | false       | Overwrite an existing `.srt` file                |

## Output

Files are placed in the same directory as the video:

```
/media/Emby/Movie (2020)/
├── Movie (2020).mkv
└── Movie (2020).en.srt   ← generated
```

Swedish audio produces `<name>.sv.srt`; all other languages produce `<name>.en.srt` via Whisper's built-in translation.

## Language logic

1. Detect language from the first ~30 seconds of audio.
2. **Swedish** → `task=transcribe`, output `<stem>.sv.srt`
3. **Everything else** → `task=translate` (Whisper translates to English), output `<stem>.en.srt`

## Emby integration

No plugin needed. Emby automatically discovers external subtitle files that follow the `<VideoName>.<lang>.srt` convention when placed in the same folder as the video.

> *Screenshot of Emby subtitle selector — placeholder*

## Hardware notes

Optimised for CPU-only inference with `int8` quantisation. A single `large-v3` model requires ~3–4 GB RAM. On an i3-12100 expect roughly 3–5× real-time speed.
