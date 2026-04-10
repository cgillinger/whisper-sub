#!/bin/bash
# entrypoint.sh — drop privileges to PUID:PGID then run the whisper scan.
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "[entrypoint] Starting as PUID=${PUID} PGID=${PGID}"

# Create group/user matching the host PUID/PGID so that .srt files written
# to the media volume are owned by the correct host user.
if ! getent group whisper &>/dev/null; then
    groupadd -g "${PGID}" whisper
fi
if ! getent passwd whisper &>/dev/null; then
    useradd -u "${PUID}" -g "${PGID}" -M -s /sbin/nologin whisper
fi

# Ensure the model-cache directory (persistent volume) is writable by the
# whisper user before we drop privileges.
mkdir -p "${HF_HOME:-/root/.cache/huggingface}"
chown -R whisper:whisper "${HF_HOME:-/root/.cache/huggingface}"
chown whisper:whisper /app

# Drop privileges and run whisper_sub.py.
# gosu handles the exec so the Python process is PID 1's direct child and
# receives signals (SIGTERM from docker stop) correctly.
#
# No arguments → default scheduled scan with config + state file.
# Any arguments → passed through as-is (e.g. "transcribe /media/film.mkv",
#                 "scan /media/Movies --dry-run").
if [ $# -eq 0 ]; then
    exec gosu whisper python /app/whisper_sub.py scan \
        --config /app/config.yml \
        --state-file /app/statedata/state.json
else
    exec gosu whisper python /app/whisper_sub.py "$@"
fi
