#!/usr/bin/env bash
# 다른 로컬 컴에서 dataset postprocess(build_training_dataset.py) 실행 환경을 자동 세팅한다.
#
# 자동으로 하는 것:
#   1. NVIDIA GPU / 드라이버 확인
#   2. uv 설치 확인(없으면 설치)
#   3. uv sync 로 파이썬 의존성 설치 (torch cu128, opencv, ultralytics ...)
#   4. ZED SDK 감지 -> uv venv 안에 pyzed(Python API) 설치
#   5. import pyzed 검증
#
# 자동으로 못 하는 것(감지 후 안내만 함):
#   - ZED SDK 본체 설치 (Stereolabs 로그인/브라우저 다운로드 필요)
#   - NVIDIA 드라이버 / CUDA 설치
#
# 실행 방법:
#   scripts/setup.cmd                                # 래퍼 (OS 자동 감지)
#   ZED_INSTALL_DIR=/opt/zed scripts/setup.cmd       # ZED SDK 경로 직접 지정

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] WARN\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[setup] ERROR\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. NVIDIA GPU / 드라이버
# ---------------------------------------------------------------------------
log "1/5  NVIDIA GPU 확인"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/         GPU: /'
else
    warn "nvidia-smi 를 찾을 수 없습니다. NVIDIA 드라이버 + CUDA 가 설치돼야 SVO depth 재계산이 됩니다."
    warn "드라이버 설치 후 다시 실행하세요. (계속 진행은 하지만 postprocess 실행은 실패할 수 있음)"
fi

# ---------------------------------------------------------------------------
# 2. uv 설치
# ---------------------------------------------------------------------------
log "2/5  uv 확인"
if ! command -v uv >/dev/null 2>&1; then
    warn "uv 가 없어 설치합니다."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 설치 직후 PATH 반영
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { err "uv 설치 실패. 수동 설치 후 재시도하세요: https://astral.sh/uv"; exit 1; }
log "         uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. 파이썬 의존성 설치
# ---------------------------------------------------------------------------
log "3/5  uv sync (파이썬 의존성 설치)"
uv sync

# uv venv 의 파이썬 경로 (pyzed 를 여기에 설치해야 함)
VENV_PY="$(uv run python -c 'import sys; print(sys.executable)')"
log "         venv python: $VENV_PY"

# ---------------------------------------------------------------------------
# 4. ZED SDK 감지 + pyzed 설치
# ---------------------------------------------------------------------------
log "4/5  ZED SDK / pyzed 설치"

# 이미 venv 안에 pyzed 가 있으면 스킵
if uv run python -c 'import pyzed.sl' >/dev/null 2>&1; then
    log "         pyzed 이미 설치됨. 스킵."
else
    # ZED SDK 설치 경로에서 get_python_api.py 찾기
    CANDIDATES=(
        "${ZED_INSTALL_DIR:-}"
        "/usr/local/zed"
        "/opt/zed"
        "$HOME/zed"
        "/mnt/c/Program Files (x86)/ZED SDK"
        "/c/Program Files (x86)/ZED SDK"
    )
    GET_API=""
    for d in "${CANDIDATES[@]}"; do
        [[ -n "$d" && -f "$d/get_python_api.py" ]] && { GET_API="$d/get_python_api.py"; break; }
    done

    if [[ -z "$GET_API" ]]; then
        err "ZED SDK 를 찾을 수 없습니다. pyzed 를 설치할 수 없습니다."
        err "  1) https://www.stereolabs.com/developers/release 에서 ZED SDK 설치 후"
        err "  2) 다시 실행하거나, 경로를 지정: ZED_INSTALL_DIR=/설치/경로 scripts/setup.cmd"
        exit 1
    fi

    ZED_ROOT="$(dirname "$GET_API")"
    log "         발견: $GET_API"
    log "         ZED_SDK_ROOT_DIR=$ZED_ROOT"
    export ZED_SDK_ROOT_DIR="$ZED_ROOT"
    uv pip install cython numpy
    uv run python "$GET_API"
    if [ $? -ne 0 ]; then
        warn "get_python_api.py 의 자동 설치 실패 — .whl 직접 설치 시도"
        WHL="$(ls "$REPO_ROOT"/pyzed-*.whl 2>/dev/null | head -1)"
        if [[ -n "$WHL" ]]; then
            log "         .whl 발견: $(basename "$WHL")"
            uv pip install "$WHL"
            rm -f "$WHL"
        else
            err "pyzed 빌드 실패 (다운로드된 .whl 도 없음)"; exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. 검증
# ---------------------------------------------------------------------------
log "5/5  검증"
if uv run python -c 'import pyzed.sl as sl; print("         ZED SDK", sl.Camera().get_sdk_version())'; then
    log "환경 세팅 완료."
else
    err "pyzed import 검증 실패. ZED SDK 버전과 파이썬 버전을 확인하세요."
    exit 1
fi
