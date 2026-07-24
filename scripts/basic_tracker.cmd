@echo off
REM Run YOLO detection + ByteTrack/SORT tracking on video.
REM Windows -> venv python / Linux -> src/basic_tracker.sh
REM
REM Usage:
REM   scripts\basic_tracker bytetrack
REM   scripts\basic_tracker sort --source input.mp4 --output out.mp4

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if defined OS (
    "%REPO_ROOT%\.venv\Scripts\python.exe" -m src.tracking.basic_tracker %*
) else (
    bash "%REPO_ROOT%/src/basic_tracker.sh" %*
)

popd
