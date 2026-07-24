#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SVO2 녹화 + CSV 라벨 → 학습용 데이터셋(RGB/Depth/Conf + COCO JSON) 추출.
# 실제 로직 → src/dataset/build_training_dataset.py
# 필요: ZED SDK + pyzed + NVIDIA GPU/CUDA
#
# Usage:
#   scripts/postprocess_dataset.sh                               # dataset/ 전체
#   scripts/postprocess_dataset.sh dataset --only-annotated      # 라벨 프레임만
#   scripts/postprocess_dataset.sh dataset --depth-mode NEURAL   # depth 모드 지정
#   scripts/postprocess_dataset.sh dataset --force               # 캐시 무시 재추출
# ──────────────────────────────────────────────────────────────

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
