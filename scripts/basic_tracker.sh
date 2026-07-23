#!/usr/bin/env bash
# YOLO + ByteTrack/SORT 기본 트래킹을 비디오에 실행한다.
# tracker 인자로 파이프라인을 선택한다 (bytetrack | sort).
# 공통 설정은 src/tracking/basic_tracker.py 의 CONFIG dict에서 수정한다.
# 실제 로직: src/tracking/basic_tracker.py (이 파일은 얇은 런처)
#
# 실행 방법:
#   scripts/basic_tracker.sh <bytetrack|sort> [--source 입력.mp4] [--output 출력.mp4]
#
# 예시:
#   scripts/basic_tracker.sh bytetrack
#   scripts/basic_tracker.sh sort --output tracking_result/my_sort.mp4
#   scripts/basic_tracker.sh -h

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
python3 -m src.tracking.basic_tracker "$@"
