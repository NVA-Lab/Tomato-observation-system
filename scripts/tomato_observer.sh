#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# 토마토 관측 파이프라인(YOLO 검출 + 추적 + CSV 기록).
# 웹캠(정수) 또는 비디오 파일을 입력으로 받는다.
# 실제 로직 → src/tracking/tomato_observer.py
#
# Usage:
#   scripts/tomato_observer.sh 0 --show-window                # 웹캠 라이브
#   scripts/tomato_observer.sh clip.mp4 --write-video         # 비디오 파일
#   scripts/tomato_observer.sh clip.mp4 --conf 0.75 --tracker sort
#   scripts/tomato_observer.sh -h
# ──────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
python3 -m src.tracking.tomato_observer "$@"
