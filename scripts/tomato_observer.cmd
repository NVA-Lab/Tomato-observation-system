@echo off
REM Tomato observation pipeline (YOLO detection + tracking + CSV logging).
REM Windows -> venv python / Linux -> src/tomato_observer.sh
REM
REM Usage:
REM   scripts\tomato_observer 0 --show-window
REM   scripts\tomato_observer clip.mp4 --write-video
REM   scripts\tomato_observer clip.mp4 --conf 0.75 --tracker sort

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if defined OS (
    "%REPO_ROOT%\.venv\Scripts\python.exe" -m src.tracking.tomato_observer %*
) else (
    bash "%REPO_ROOT%/src/tomato_observer.sh" %*
)

popd
