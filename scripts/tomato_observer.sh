#!/usr/bin/env bash
# 토마토 관측 파이프라인(YOLO + 추적 + CSV)을 라이브 웹캠 또는 비디오 파일에 실행한다.
# source 인자가 정수면 웹캠 인덱스(라이브), 그 외는 비디오 파일 경로로 처리한다.
# 웹앱과 동일한 파이프라인/설정을 Flask 없이 사용한다.
# 실제 로직: src/tracking/tomato_observer.py (이 파일은 얇은 런처)
#
# 실행 방법:
#   scripts/tomato_observer.sh <source> [옵션...]
#
# 예시:
#   scripts/tomato_observer.sh 0 --show-window            # 라이브 웹캠(인덱스 0)
#   scripts/tomato_observer.sh clip.mp4 --write-video     # 비디오 파일 처리
#   scripts/tomato_observer.sh clip.mp4 --conf 0.75 --tracker sort --mode sbs_left
#   scripts/tomato_observer.sh -h                         # 전체 옵션 보기

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
python3 -m src.tracking.tomato_observer "$@"
