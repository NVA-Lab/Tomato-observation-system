#!/usr/bin/env bash
# 녹화된 세션({stem}.svo2 + {stem}.csv)에서 RGB 이미지 + COCO bbox/클래스 어노테이션 +
# depth 맵을 추출해 detector 학습용 데이터셋을 만든다.
# 실제 로직: src/dataset/build_training_dataset.py (SVO 재생·프레임 추출·COCO 병합)
# 필요: ZED SDK + pyzed + NVIDIA GPU/CUDA (없으면 scripts/setup_env.sh 로 세팅)
#
# 실행 방법:
#   scripts/postprocess_dataset.sh [dataset_dir] [--out DIR] [--depth-mode MODE] [--force]
#
# 예시:
#   scripts/postprocess_dataset.sh                        # 기본 dataset/ 폴더 전체
#   scripts/postprocess_dataset.sh dataset --only-annotated   # 라벨 있는 프레임만
#   scripts/postprocess_dataset.sh dataset --depth-mode NEURAL
#   scripts/postprocess_dataset.sh dataset --force        # 캐시 무시하고 재추출

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${TOMATO_DATASET_DIR:-$REPO_ROOT/dataset}"

# First positional arg (if not starting with --) overrides the dataset dir.
if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET_DIR="$1"
    shift
fi

DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
echo "[postprocess_dataset] dataset_dir=$DATASET_DIR"

cd "$REPO_ROOT"
python3 -m src.dataset.build_training_dataset "$DATASET_DIR" "$@"
