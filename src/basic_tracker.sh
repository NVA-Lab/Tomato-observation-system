#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# 비디오에 YOLO 검출 + ByteTrack/SORT 트래킹을 실행하는 런처.
# 실제 로직 → src/tracking/basic_tracker.py
#
# Usage:
#   scripts/basic_tracker.sh bytetrack
#   scripts/basic_tracker.sh sort --source input.mp4 --output out.mp4
# ──────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
python3 -m src.tracking.basic_tracker "$@"
