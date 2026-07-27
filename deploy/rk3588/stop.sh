#!/usr/bin/env bash
set -euo pipefail

if command -v systemctl >/dev/null 2>&1; then
  systemctl stop park-safety.service
else
  pkill -f "python3 main.py" || true
fi
