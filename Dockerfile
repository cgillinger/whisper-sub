FROM python:3.11-slim

# Install system dependencies:
#   ffmpeg       — audio decoding (required by faster-whisper)
#   gosu         — privilege-drop helper used by entrypoint.sh
#   libgl1, etc. — low-level libs sometimes needed by OpenVINO / numpy
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      gosu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cache friendly).
# OpenVINO is installed via pip (openvino package) rather than basing the
# image on openvino/ubuntu22_runtime, keeping the image simpler and giving
# us control over the Python version.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY whisper_sub.py .

# Copy and prepare entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model cache lives in a persistent volume mounted at /root/.cache/huggingface
# (see docker-compose.yml).  Set HF_HOME so faster-whisper / HuggingFace find it
# regardless of which user ultimately runs the process.
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["/entrypoint.sh"]
