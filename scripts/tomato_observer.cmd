:; # ─────────────────────────────────────────────────────────────
:; # Cross-platform 런처: 실행 시 OS 자동 감지.
:; #   Windows(cmd) : scripts\tomato_observer
:; #   Linux/Mac    : bash scripts/tomato_observer.cmd
:; # 토마토 관측 파이프라인(YOLO 검출 + 추적 + CSV 기록).
:; #
:; # Usage:
:; #   scripts\tomato_observer 0 --show-window
:; #   scripts\tomato_observer clip.mp4 --write-video
:; #   scripts\tomato_observer clip.mp4 --conf 0.75 --tracker sort
:; # ─────────────────────────────────────────────────────────────
:; REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
:; exec bash "$REPO_ROOT/src/tomato_observer.sh" "$@"

@echo off
REM -- Windows path --
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
"%REPO_ROOT%\.venv\Scripts\python.exe" -m src.tracking.tomato_observer %*
popd
