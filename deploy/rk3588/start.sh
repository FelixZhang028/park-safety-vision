#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${1:-${VIDEO_SOURCE:-}}"
SCENE="${2:-${SCENE_CONFIG:-}}"
CONFIG="${CONFIG_PATH:-$APP_DIR/config.yaml}"

if [[ -z "$SOURCE" ]]; then
  echo "VIDEO_SOURCE is not set."
  echo "Usage: $0 <video-path|camera-index|rtsp-url> [scene-yaml]"
  exit 2
fi

ARGS=(main.py --config "$CONFIG" --source "$SOURCE" --no-show)
if [[ -n "$SCENE" ]]; then
  ARGS+=(--scene "$SCENE")
fi

cd "$APP_DIR"
exec python3 "${ARGS[@]}"
