#!/usr/bin/env bash
# Transcribe every recording in tests/samples/ using the built image, so no local
# ffmpeg, whisper.cpp or model weights are needed. Prints one transcript per file.
#
#   ./scripts/transcribe-samples.sh                     # image echoback:local, model small
#   MODEL=medium IMAGE=echoback:local ./scripts/transcribe-samples.sh
#   SAMPLES=/path/to/other/dir ./scripts/transcribe-samples.sh --json
#
# The image must have the requested model baked in:
#   docker build --build-arg MODELS="small medium" -t echoback:local .
set -euo pipefail

IMAGE=${IMAGE:-echoback:local}
MODEL=${MODEL:-small}
SAMPLES=${SAMPLES:-"$(cd "$(dirname "$0")/.." && pwd)/tests/samples"}

if [ ! -d "$SAMPLES" ]; then
  echo "no samples directory at $SAMPLES" >&2
  exit 1
fi
if [ -z "$(find "$SAMPLES" -maxdepth 1 -type f ! -name '.*' ! -name '*.md' -print -quit)" ]; then
  echo "no recordings in $SAMPLES — drop some voicemail files in first" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image $IMAGE not found — build it with:" >&2
  echo "  docker build --build-arg MODELS=\"$MODEL\" -t $IMAGE ." >&2
  exit 1
fi

exec docker run --rm \
  -v "$SAMPLES:/samples:ro" \
  -e MODEL_DEFAULT="$MODEL" \
  --entrypoint python \
  "$IMAGE" -m echoback.transcribe --model "$MODEL" "$@" /samples
