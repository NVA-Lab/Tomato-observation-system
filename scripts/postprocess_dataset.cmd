@echo off
REM Extract training dataset (RGB/Depth/Conf + COCO JSON) from SVO2 recordings.
REM Windows -> venv python / Linux -> src/postprocess_dataset.sh
REM Requires: ZED SDK + pyzed + NVIDIA GPU/CUDA
REM
REM Usage:
REM   scripts\postprocess_dataset dataset\smart_farm_tomato_dataset_JUL
REM   scripts\postprocess_dataset dataset --only-annotated
REM   scripts\postprocess_dataset dataset --force

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if defined OS (
    "%REPO_ROOT%\.venv\Scripts\python.exe" -m src.dataset.build_training_dataset %*
) else (
    bash "%REPO_ROOT%/src/postprocess_dataset.sh" %*
)

popd
