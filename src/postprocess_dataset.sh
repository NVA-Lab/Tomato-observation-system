#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SVO2 녹화 → 학습 데이터셋(RGB/Depth/Conf + COCO JSON) 추출 런처.
# 실제 로직 → src/dataset/build_training_dataset.py
# 요구사항: ZED SDK + pyzed + NVIDIA GPU/CUDA
#
# Usage:
#   scripts/postprocess_dataset dataset/smart_farm_tomato_dataset_JUL
#   scripts/postprocess_dataset dataset --only-annotated
#   scripts/postprocess_dataset dataset --force
# ──────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
python3 -m src.dataset.build_training_dataset "$@"
