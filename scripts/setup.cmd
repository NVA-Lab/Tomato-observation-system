@echo off
REM Project environment setup (venv + dependencies).
REM Windows -> setup_env.ps1 / Linux -> setup_env.sh
REM
REM Usage:
REM   scripts\setup

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if defined OS (
    echo [setup] Windows -- running setup_env.ps1
    powershell -ExecutionPolicy Bypass -File "%REPO_ROOT%\src\setup_env.ps1"
) else (
    echo [setup] Linux/WSL -- running setup_env.sh
    bash "%REPO_ROOT%\src\setup_env.sh"
)

popd
