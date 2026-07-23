#!/usr/bin/env bash
# Build a detector-training dataset (RGB images + COCO bbox/class annotations + depth maps)
# from recorded data-collection sessions.
#
# Thin wrapper around scripts/build_training_dataset.py, which does the actual per-session
# SVO replay (frame extraction, depth maps, COCO annotation assembly, fragment caching).
#
# Usage:
#   scripts/postprocess_dataset.sh [dataset_dir] [--out DIR] [--depth-mode MODE] [--force]
#
# Examples:
#   scripts/postprocess_dataset.sh
#   scripts/postprocess_dataset.sh dataset --depth-mode NEURAL
#   scripts/postprocess_dataset.sh dataset --force

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

python3 "$REPO_ROOT/scripts/build_training_dataset.py" "$DATASET_DIR" "$@"
