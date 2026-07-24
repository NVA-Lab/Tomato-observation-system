:; # ─────────────────────────────────────────────────────────────
:; # Cross-platform 런처: 실행 시 OS 자동 감지.
:; #   Windows(cmd) : scripts\basic_tracker
:; #   Linux/Mac    : bash scripts/basic_tracker.cmd  (또는 ./scripts/basic_tracker.cmd)
:; # YOLO 검출 + ByteTrack/SORT 트래킹.
:; #
:; # Usage:
:; #   scripts\basic_tracker bytetrack
:; #   scripts\basic_tracker sort --source input.mp4 --output out.mp4
:; # ─────────────────────────────────────────────────────────────
:; REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
:; exec bash "$REPO_ROOT/src/basic_tracker.sh" "$@"

@echo off
REM -- Windows path --
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
"%REPO_ROOT%\.venv\Scripts\python.exe" -m src.tracking.basic_tracker %*
popd
