#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="$(cat "$SOURCE_DIR/VERSION")"
INSTALL_ROOT="${1:-/opt/park-safety}"
RELEASE_DIR="$INSTALL_ROOT/releases/$VERSION"

if [[ "$EUID" -ne 0 ]]; then
  echo "Run this installer as root."
  exit 1
fi
if [[ -e "$RELEASE_DIR" ]]; then
  echo "Release already exists: $RELEASE_DIR"
  exit 1
fi

mkdir -p "$RELEASE_DIR" /etc/park-safety /var/lib/park-safety/outputs
cp -a "$SOURCE_DIR/." "$RELEASE_DIR/"
chmod +x "$RELEASE_DIR/start.sh" "$RELEASE_DIR/stop.sh"
ln -sfn "$RELEASE_DIR" "$INSTALL_ROOT/current"
cp "$RELEASE_DIR/park-safety.service" /etc/systemd/system/park-safety.service

if [[ ! -f /etc/park-safety/park-safety.env ]]; then
  cat > /etc/park-safety/park-safety.env <<'EOF'
VIDEO_SOURCE=/root/test.mp4
SCENE_CONFIG=
CONFIG_PATH=/opt/park-safety/current/config.yaml
EOF
fi

python3 -c "import cv2, numpy, yaml; from rknnlite.api import RKNNLite; print('Runtime dependencies: OK')"
systemctl daemon-reload
systemctl enable park-safety.service

echo "Installed: $RELEASE_DIR"
echo "Edit /etc/park-safety/park-safety.env, then run:"
echo "  systemctl start park-safety"
echo "  journalctl -u park-safety -f"
