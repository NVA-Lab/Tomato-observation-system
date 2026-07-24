:; # ─────────────────────────────────────────────────────────────
:; # Cross-platform 런처: 실행 시 OS 자동 감지.
:; #   Windows(cmd) : scripts\postprocess_dataset
:; #   Linux/Mac    : bash scripts/postprocess_dataset.cmd
:; # SVO2 녹화 -> 학습 데이터셋(RGB/Depth/Conf + COCO JSON) 추출.
:; # 요구사항: ZED SDK + pyzed + NVIDIA GPU/CUDA
:; #
:; # Usage:
:; #   scripts\postprocess_dataset dataset\smart_farm_tomato_dataset_JUL
:; #   scripts\postprocess_dataset dataset --only-annotated
:; #   scripts\postprocess_dataset dataset --force
:; # ─────────────────────────────────────────────────────────────
:; REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
:; exec bash "$REPO_ROOT/src/postprocess_dataset.sh" "$@"

@echo off
REM -- Windows path --
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
"%REPO_ROOT%\.venv\Scripts\python.exe" -m src.dataset.build_training_dataset %*
popd
