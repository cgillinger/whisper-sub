# whisper-sub

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Automatic subtitle generation for Emby and Jellyfin using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Detects the audio language, transcribes Swedish to `.sv.srt`, and translates everything else to `.en.srt` — files are placed alongside the video so your media server picks them up with no plugin required.

---

## Features

- **KB-Whisper support** — use [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) for superior Swedish transcription quality
- **Automatic language detection** — confidence-based decision with configurable per-path thresholds
- **Two-pass KB-Whisper pipeline** — detect with `large-v3`, swap to KB-Whisper for Swedish files; only one large model in memory at a time (~3–4 GB each)
- **Word-level timestamps** — accurate subtitle timing anchored to individual words
- **Pause-aware segment splitting** — splits subtitle blocks at natural speech pauses ≥ 0.8 s (minimum 2 words per side)
- **Smart display duration** — text-length-based clamping (max 7 s, floor 2 s) with overlap prevention between segments
- **VAD (Voice Activity Detection)** — skips silence and reduces hallucinations; recommended for home videos
- **Thermal throttling** — monitors CPU and NVMe temperatures via sysfs; pauses the queue when thresholds are exceeded and resumes automatically
- **Graceful shutdown** — SIGTERM/SIGINT saves state after the current file so the next run continues where it left off
- **Per-path configuration** — different VAD, confidence threshold, and fallback language per media directory
- **State tracking** — JSON state file prevents reprocessing already-completed files
- **Docker support** — PUID/PGID privilege drop, optional Intel iGPU (OpenVINO) passthrough, model-cache volume

---

## Requirements

- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ≥ 1.0.0
- `ffmpeg` on `PATH` (used by faster-whisper for audio decoding)
- `pyyaml` ≥ 6.0 (config-file mode)
- `openvino` ≥ 2024.0 (optional — Intel iGPU acceleration)

---

## Installation

### Bare metal

```bash
git clone https://github.com/cgillinger/whisper-sub.git
cd whisper-sub
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Docker

```bash
git clone https://github.com/cgillinger/whisper-sub.git
cd whisper-sub
# Edit docker-compose.yml to set your volume paths, then:
docker compose build
docker compose run --rm emby-whisper
```

---

## Usage

### Single file (legacy positional)

```bash
python whisper_sub.py /path/to/video.mkv
```

### Single file (`transcribe` subcommand)

```bash
python whisper_sub.py transcribe /path/to/video.mkv
python whisper_sub.py transcribe /path/to/video.mkv --dry-run
python whisper_sub.py transcribe /path/to/video.mkv --config config.yml
```

### Batch scan

```bash
# Scan a directory
python whisper_sub.py scan /media/Movies

# Config-driven (recommended for Docker / automation)
python whisper_sub.py scan --config config.yml

# Preview what would be processed
python whisper_sub.py scan /media/Movies --dry-run --limit 10
```

### Output

Subtitle files are written next to the video:

```
/media/Movies/Film (2021)/
├── Film (2021).mkv
├── Film (2021).sv.srt   ← Swedish audio
└── Film (2021).en.srt   ← all other languages (translated to English)
```

---

## Configuration

`config.yml` controls all settings when running in batch mode. Pass it with `--config`.

```yaml
# Directories to scan (container paths when using Docker).
# Per-path settings override global defaults for that directory.
scan_paths:
  - path: /media/Movies
    min_confidence: 0.7        # treat as Swedish only when Whisper is ≥ 70% confident
    vad_filter: true
  - path: /media/HomeVideos
    min_confidence: 0.5        # lower bar for home video audio quality
    vad_filter: true
    min_silence_duration: 0.5  # seconds of silence VAD treats as a break
    default_language: sv       # assume Swedish when confidence is below threshold

# Whisper models
# When swedish_model differs from default_model, KB-Whisper two-pass mode is used:
#   detect language with default_model → transcribe Swedish with swedish_model
#   → translate non-Swedish with default_model (reloaded as needed)
# Remove swedish_model or set it equal to default_model to disable KB-Whisper.
default_model: large-v3
swedish_model: KBLab/kb-whisper-large

# Compute device: cpu | cuda | openvino | auto
# openvino uses the Intel iGPU and falls back to CPU automatically if unavailable.
device: cpu
compute_type: int8             # int8 | float16 | float32

# Language decision
swedish_threshold: 0.7         # global confidence threshold for Swedish
default_task: translate        # action when audio is not Swedish: translate to English

# Video extensions to include
video_extensions:
  - .mp4
  - .mkv
  - .avi
  - .m4v
  - .mov
  - .wmv
  - .ts

# Behaviour
skip_if_subtitle_exists: true
max_files_per_run: 0           # 0 = unlimited

# Thermal throttling
thermal:
  enabled: true
  cpu_pause_temp: 72           # pause when CPU  ≥ this °C
  nvme_pause_temp: 70          # pause when NVMe ≥ this °C
  resume_temp: 65              # resume when BOTH sensors drop below this °C
  check_interval: 30           # seconds between temperature checks
  max_pause_minutes: 30        # abort current file if paused longer than this
```

---

## Translation (optional)

Translate the generated `.en.srt` subtitles to any language using an external LLM API. The recommended starting point is **Gemini Lite** — it is completely free, requires no credit card, and the code physically enforces the free-tier limit.

Translation runs automatically in `scan` mode when enabled in `config.yml`, and is also available as a standalone subcommand for existing SRT files.

---

### Getting a free API key (Gemini)

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key** in the left sidebar
4. Click **Create API key** → copy the key
5. Create a `.env` file in the project directory (next to `whisper_sub.py`):

```
GEMINI_API_KEY=your-key-here
```

**No credit card required.**

---

### Lite mode

| | |
|---|---|
| **Provider** | Gemini Flash-Lite (free tier) |
| **Daily limit** | 20 requests (reduced from 1,500 in 2025) |
| **Hard cap in code** | 18 requests (2 safety margin) |
| **One movie** | one request |
| **Cost** | $0.00. Always. |

The code hard-stops at 18 requests per day and resets automatically at midnight. There is **no flag, environment variable, or config override** to bypass this limit — the only way to go higher is to switch to a paid provider.

**Trade-off:** Google may use free-tier requests for model training. For translating commercial movie subtitles this is not a practical concern — the content is already public.

---

### Config example (Lite — free)

Add to `config.yml`:

```yaml
translation:
  enabled: true
  provider: gemini-lite
  target_language: sv
```

Add your API key to `.env` in the project directory, then run as normal:

```
GEMINI_API_KEY=your-key-here
```

```bash
python whisper_sub.py scan --config config.yml
```

Translated subtitles are written next to the video alongside the English subtitle:

```
Film (2021).mkv
Film (2021).en.srt   ← Whisper output (English)
Film (2021).sv.srt   ← translation output (Swedish)
```

---

### Standalone translate subcommand

Translate existing SRT files without re-running Whisper:

```bash
# Translate a single file
python whisper_sub.py translate /path/to/film.en.srt --provider gemini-lite --lang sv

# Translate all .en.srt files under a directory
python whisper_sub.py translate /media/Movies --provider gemini-lite --lang sv
```

---

### Paid providers (optional)

For higher quality, privacy requirements, or higher throughput, any of these providers can be used instead of Gemini Lite.

| Provider | Model | Cost/movie (approx.) | Sign up |
|---|---|---|---|
| Gemini (paid) | Flash 2.5 | ~$0.06 | [https://aistudio.google.com](https://aistudio.google.com) |
| OpenAI | GPT-4.1 Nano | ~$0.01 | [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| DeepSeek | V3 | ~$0.01 | [https://platform.deepseek.com](https://platform.deepseek.com) |
| Anthropic | Haiku 4.5 | ~$0.12 | [https://console.anthropic.com](https://console.anthropic.com) |

#### Gemini (paid)

1. Go to [https://aistudio.google.com](https://aistudio.google.com) and sign in
2. Click **Get API key** → **Create API key** → copy the key
3. Add to `.env`: `GEMINI_API_KEY=your-key-here`
4. Set `provider: gemini` in `config.yml`

#### OpenAI

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) and sign in
2. Click **Create new secret key** → copy the key
3. Add to `.env`: `OPENAI_API_KEY=your-key-here`
4. Set `provider: openai` in `config.yml`

#### DeepSeek

1. Go to [https://platform.deepseek.com](https://platform.deepseek.com) and create an account
2. Navigate to **API keys** → **Create API key** → copy the key
3. Add to `.env`: `DEEPSEEK_API_KEY=your-key-here`
4. Set `provider: deepseek` in `config.yml`

#### Anthropic

1. Go to [https://console.anthropic.com](https://console.anthropic.com) and create an account
2. Navigate to **API keys** → **Create key** → copy the key
3. Add to `.env`: `ANTHROPIC_API_KEY=your-key-here`
4. Set `provider: anthropic` in `config.yml`

---

### Cost safety

**Lite mode** is guaranteed free. The code refuses to make a request once the daily counter reaches 18. There is no override.

**Paid providers** charge per token. Typical costs per full-length movie:

| Provider | Cost/movie | 1 000 movies |
|---|---|---|
| OpenAI GPT-4.1 Nano | ~$0.01 | ~$10 |
| DeepSeek V3 | ~$0.01 | ~$10 |
| Gemini Flash (paid) | ~$0.06 | ~$60 |
| Anthropic Haiku 4.5 | ~$0.12 | ~$120 |

All providers offer spending limits or caps on their dashboards:

- **Google / Gemini:** [https://aistudio.google.com/usage](https://aistudio.google.com/usage) — set monthly budget alerts
- **OpenAI:** [https://platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits) — hard monthly spend limit
- **DeepSeek:** [https://platform.deepseek.com/usage](https://platform.deepseek.com/usage) — account balance model (prepaid)
- **Anthropic:** [https://console.anthropic.com/settings/limits](https://console.anthropic.com/settings/limits) — monthly spend limit

---

### Privacy

| Provider | Free tier | Paid tier |
|---|---|---|
| Gemini Lite | Data **may** be used for model training | N/A |
| Gemini (paid) | N/A | Data is **not** used for training |
| OpenAI | N/A | Data is **not** used for training |
| DeepSeek | N/A | Data is **not** used for training |
| Anthropic | N/A | Data is **never** used for training on any tier |

For translating commercial movie subtitles on the free tier, the content is already publicly distributed — this is not a practical concern.

---

## CLI Arguments

### Common arguments (all subcommands)

| Argument | Default | Description |
|---|---|---|
| `--model` | `large-v3` | Whisper model for language detection and non-Swedish transcription |
| `--swedish-model` | *(none)* | Model for Swedish transcription (enables KB-Whisper, e.g. `KBLab/kb-whisper-large`) |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, `openvino`, `auto` |
| `--compute-type` | `int8` | Quantisation: `int8`, `float16`, `float32` |
| `--force` | false | Overwrite existing `.srt` files and ignore state |
| `--vad-filter` | false | Enable Voice Activity Detection |
| `--min-silence-duration` | `0.5` | Minimum silence in seconds for VAD splitting |

### `transcribe` subcommand

```
python whisper_sub.py transcribe <path> [options]
```

| Argument | Description |
|---|---|
| `path` | Path to a video file *(required)* |
| `--config PATH` | Load model/device/threshold/VAD defaults from a config file |
| `--dry-run` | Detect language and print the planned action; write no files |

### `scan` subcommand

```
python whisper_sub.py scan [directory] [options]
```

| Argument | Default | Description |
|---|---|---|
| `directory` | *(none)* | Directory to scan recursively; omit when using `--config` with `scan_paths` |
| `--config PATH` | *(none)* | Path to `config.yml` |
| `--dry-run` | false | List files that would be processed without doing any work |
| `--limit N` | `0` | Stop after processing N files (0 = unlimited) |
| `--state-file PATH` | `~/.emby-whisper-state.json` | JSON file for tracking completed files |

---

## Language Logic

```
Audio
  └─ detect language (first ~30 s, default_model)
        ├─ Swedish AND confidence ≥ threshold
        │     └─ task=transcribe → <file>.sv.srt
        │           (KB-Whisper model used if configured)
        └─ anything else  OR  confidence < threshold
              ├─ default_language set? → use it
              └─ task=translate → <file>.en.srt
```

**Per-path overrides** let you tune the threshold and fallback language per directory. This is useful for home videos where audio quality is lower and Whisper may misidentify the language.

**KB-Whisper two-pass** (scan mode with `swedish_model` set):
1. Detect language for the current file using the loaded model.
2. If Swedish → load KB-Whisper (swap only if not already loaded) → transcribe.
3. If other language → load `default_model` (swap if needed) → translate.

Only one large model lives in memory at a time. The model is kept across consecutive files of the same language class and only swapped on a language boundary.

---

## Subtitle Post-processing

Every segment output by Whisper passes through two post-processing steps before the SRT file is written.

### 1. `split_segments_on_pause`

Splits a Whisper segment into two when word-level timestamps reveal a gap ≥ 0.8 s between consecutive words. The split is only applied when both resulting halves contain at least 2 words, preventing orphaned single-word segments.

### 2. `clamp_segment_durations`

Caps the display duration of each segment:

| Rule | Value |
|---|---|
| Text-length-based duration | `len(text) ÷ 15 chars/s` |
| Maximum duration | 7 s |
| Minimum duration | 2 s |
| Overlap prevention | End clamped to the start of the next segment |

Short subtitles vanish quickly; long ones never linger more than 7 s. Segments never overlap.

---

## Thermal Throttling

When `thermal.enabled: true` the scan loop reads temperatures from sysfs before starting each file:

| Sensor | sysfs location |
|---|---|
| CPU | `/sys/class/thermal/thermal_zone*` (zone type `x86_pkg_temp` or `coretemp`) |
| NVMe | `/sys/class/hwmon/hwmon*` (hwmon name contains `nvme`) |

**Behaviour:**

1. If CPU ≥ `cpu_pause_temp` or NVMe ≥ `nvme_pause_temp` → pause the queue and log a warning.
2. Poll every `check_interval` seconds until **both** sensors drop below `resume_temp`.
3. If the pause exceeds `max_pause_minutes` → abort the current file, record an error in the state file, and move to the next.

In Docker, bind-mount the host sysfs paths read-only (see Docker section below).

---

## Docker Deployment

### `docker-compose.yml`

```yaml
services:
  emby-whisper:
    build:
      context: .
    container_name: emby-whisper
    environment:
      - PUID=1000          # host user UID — written .srt files will be owned by this user
      - PGID=1000          # host group GID
      - TZ=Europe/Stockholm
    volumes:
      # Media (read-write — writes .srt files alongside videos)
      - /path/to/media:/media
      # Config and state
      - /path/to/appdata/emby-whisper/config.yml:/app/config.yml:ro
      - /path/to/appdata/emby-whisper/statedata:/app/statedata
      # Model cache — survives container rebuilds
      - /path/to/appdata/emby-whisper/models:/root/.cache/huggingface
      # Thermal sensors (read-only)
      - /sys/class/thermal:/sys/class/thermal:ro
      - /sys/class/hwmon:/sys/class/hwmon:ro
    # Intel iGPU passthrough for OpenVINO acceleration (optional)
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    group_add:
      - "video"
      - "render"
    deploy:
      resources:
        limits:
          cpus: "3.0"      # leave headroom for OS and other containers
    restart: "no"
```

### Running

```bash
# One-shot scan (reads config.yml automatically)
docker compose run --rm emby-whisper

# Pass custom arguments
docker compose run --rm emby-whisper transcribe /media/Movies/film.mkv
docker compose run --rm emby-whisper scan /media/Movies --dry-run
```

### Scheduling with cron

```cron
# Run every night at 02:00
0 2 * * * docker compose -f /path/to/whisper-sub/docker-compose.yml run --rm emby-whisper
```

### Notes

- `PUID`/`PGID` — the entrypoint creates a `whisper` user with matching IDs and drops privileges using `gosu`, so subtitle files are owned by your host user.
- **Model cache** — first run downloads models from Hugging Face (several GB). Subsequent runs reuse the cache.
- **OpenVINO** — set `device: openvino` in `config.yml` and pass through `/dev/dri/renderD128`. Falls back to CPU automatically if OpenVINO fails to load a model.

---

## Multi-Device Setup

You can split the workload across machines by pointing each instance at different scan paths and using a shared media volume:

| Machine | Config | Use case |
|---|---|---|
| GPU server (CUDA) | `device: cuda`, `default_model: large-v3` | Large batches of foreign-language content |
| CPU machine | `device: cpu`, `swedish_model: KBLab/kb-whisper-large` | Swedish content with KB-Whisper |
| NAS / server | `device: cpu`, cron schedule | Automatic nightly processing of new files |

Each instance uses its own state file so they don't collide. Point them at non-overlapping `scan_paths` or use the `--limit` flag to stagger load.

---

## Emby / Jellyfin Integration

No plugin required. Both Emby and Jellyfin automatically discover external subtitle files when they follow the `<VideoFilename>.<lang>.srt` naming convention and are placed in the same folder as the video.

**Language codes used:**

| Audio | Subtitle file | Language shown in player |
|---|---|---|
| Swedish (≥ threshold) | `<name>.sv.srt` | Swedish |
| Anything else | `<name>.en.srt` | English (translated) |

After a scan completes, trigger a library refresh in Emby/Jellyfin — subtitles will appear in the subtitle selector immediately.

---

## Hardware Notes

| Setup | RAM needed | Expected speed |
|---|---|---|
| `large-v3` (CPU, int8) | ~3–4 GB | ~3–5× real-time on a modern quad-core |
| `kb-whisper-large` (CPU, int8) | ~3–4 GB | Similar to `large-v3` |
| `large-v3` (CUDA) | 4–6 GB VRAM | 10–20× real-time on a mid-range GPU |
| `large-v3` (OpenVINO / iGPU) | shared system RAM | 2–4× real-time vs CPU |

`int8` quantisation (the default) halves memory usage versus `float16` with negligible quality loss for most content. VAD is especially recommended for home videos — it reduces hallucinated text during silence and can speed up transcription significantly.
