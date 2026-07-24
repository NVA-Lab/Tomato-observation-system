
@echo off
:: ──────────────────────────────────────────────────────────────
:: 프로젝트 환경 셋업 (venv + 의존성 설치).
:: Windows → setup_env.ps1 / Linux·WSL → setup_env.sh 자동 분기.
::
:: Usage:
::   scripts\setup.cmd
:: ──────────────────────────────────────────────────────────────
chcp 65001 >nul 2>&1

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if defined OS (
    echo [setup] Windows -- running setup_env.ps1
    powershell -ExecutionPolicy Bypass -File "%REPO_ROOT%\src\setup_env.ps1"
) else (
    echo [setup] Linux/WSL -- running setup_env.sh
    bash "%REPO_ROOT%/src/setup_env.sh"
)

popd
