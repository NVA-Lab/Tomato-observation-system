:; # ─────────────────────────────────────────────────────────────
:; # Cross-platform 런처: 실행 시 OS 자동 감지.
:; #   Windows(cmd) : scripts\setup   -> src\setup_env.ps1
:; #   Linux/Mac    : bash scripts/setup.cmd -> src/setup_env.sh
:; # 프로젝트 환경 세팅(venv + 의존성).
:; #
:; # Usage:
:; #   scripts\setup
:; # ─────────────────────────────────────────────────────────────
:; REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
:; echo "[setup] Linux/WSL -- running setup_env.sh"
:; exec bash "$REPO_ROOT/src/setup_env.sh" "$@"

@echo off
REM -- Windows path --
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
echo [setup] Windows -- running setup_env.ps1
powershell -ExecutionPolicy Bypass -File "%REPO_ROOT%\src\setup_env.ps1"
popd
