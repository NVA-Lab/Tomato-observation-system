#Requires -Version 5.1
<#
.SYNOPSIS
    Windows 환경 세팅 스크립트 (setup_env.sh 의 Windows 대응).
.DESCRIPTION
    1. NVIDIA GPU / 드라이버 확인
    2. uv 설치 확인 (없으면 설치)
    3. uv sync 로 파이썬 의존성 설치
    4. ZED SDK 감지 -> pyzed 설치
    5. import pyzed 검증
.NOTES
    실행:  scripts\setup.cmd  (래퍼, 권장)
    직접:  powershell -ExecutionPolicy Bypass -File src\setup_env.ps1
    ZED SDK 경로 지정:
           $env:ZED_INSTALL_DIR = 'D:\ZED SDK'; scripts\setup.cmd
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $RepoRoot
try {

function Log  { param([string]$msg) Write-Host "[setup] $msg" -ForegroundColor Green }
function Warn { param([string]$msg) Write-Host "[setup] WARN $msg" -ForegroundColor Yellow }
function Err  { param([string]$msg) Write-Host "[setup] ERROR $msg" -ForegroundColor Red }

# -------------------------------------------------------------------------
# 1. NVIDIA GPU / 드라이버
# -------------------------------------------------------------------------
Log "1/5  NVIDIA GPU 확인"
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    foreach ($line in $gpuInfo) { Log "         GPU: $line" }
} else {
    Warn "nvidia-smi 를 찾을 수 없습니다. NVIDIA 드라이버 + CUDA 가 설치돼야 SVO depth 재계산이 됩니다."
    Warn "드라이버 설치 후 다시 실행하세요. (계속 진행은 하지만 postprocess 실행은 실패할 수 있음)"
}

# -------------------------------------------------------------------------
# 2. uv 설치
# -------------------------------------------------------------------------
Log "2/5  uv 확인"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Warn "uv 가 없어 설치합니다."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Err "uv 자동 설치 실패. winget 으로 시도합니다."
        winget install --id astral-sh.uv --accept-source-agreements --accept-package-agreements
    }
    # 설치 직후 PATH 반영
    $uvPaths = @(
        "$env:USERPROFILE\.local\bin",
        "$env:USERPROFILE\.cargo\bin"
    )
    foreach ($p in $uvPaths) {
        if ((Test-Path $p) -and ($env:PATH -notlike "*$p*")) {
            $env:PATH = "$p;$env:PATH"
        }
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Err "uv 설치 실패. 수동 설치 후 재시도하세요: https://astral.sh/uv"
    exit 1
}
Log "         uv: $(uv --version)"

# -------------------------------------------------------------------------
# 3. 파이썬 의존성 설치
# -------------------------------------------------------------------------
Log "3/5  uv sync (파이썬 의존성 설치)"
uv sync
if ($LASTEXITCODE -ne 0) { Err "uv sync 실패"; exit 1 }

$VenvPy = (uv run python -c "import sys; print(sys.executable)") | Select-Object -Last 1
Log "         venv python: $VenvPy"

# -------------------------------------------------------------------------
# 4. ZED SDK 감지 + pyzed 설치
# -------------------------------------------------------------------------
Log "4/5  ZED SDK / pyzed 설치"

# ZED SDK 경로 찾기 (설치 + 검증 모두에서 필요)
$candidates = @(
    $env:ZED_INSTALL_DIR,
    $env:ZED_SDK_ROOT_DIR,
    "${env:ProgramFiles(x86)}\ZED SDK",
    "$env:ProgramFiles\ZED SDK",
    "C:\Program Files (x86)\ZED SDK",
    "C:\Program Files\ZED SDK"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

$ZedBinDir = $null
foreach ($d in $candidates) {
    $bin = Join-Path $d "bin"
    if (Test-Path $bin) { $ZedBinDir = $bin; $env:ZED_SDK_ROOT_DIR = $d; break }
}

if (-not $ZedBinDir) {
    Err "ZED SDK 를 찾을 수 없습니다. pyzed 를 설치할 수 없습니다."
    Err "  1) https://www.stereolabs.com/developers/release 에서 ZED SDK 설치 후"
    Err "  2) 다시 실행하거나, 경로를 지정:"
    Err '     $env:ZED_INSTALL_DIR = "D:\ZED SDK"; .\scripts\setup.cmd'
    exit 1
}
Log "         ZED SDK: $($env:ZED_SDK_ROOT_DIR)"

$pyzedOk = $false
try {
    uv run python -c "import os; os.add_dll_directory(r'$ZedBinDir'); import pyzed.sl" 2>$null
    if ($LASTEXITCODE -eq 0) { $pyzedOk = $true }
} catch {}

if ($pyzedOk) {
    Log "         pyzed 이미 설치됨. 스킵."
} else {
    $getApi = Join-Path $env:ZED_SDK_ROOT_DIR "get_python_api.py"
    if (-not (Test-Path $getApi)) {
        Err "get_python_api.py 를 찾을 수 없습니다: $getApi"; exit 1
    }

    Log "         get_python_api.py 실행"
    uv pip install cython numpy
    uv run python $getApi
    if ($LASTEXITCODE -ne 0) {
        Warn "get_python_api.py 의 자동 설치 실패 — .whl 직접 설치 시도"
        $whl = Get-ChildItem "$RepoRoot\pyzed-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($whl) {
            Log "         .whl 발견: $($whl.Name)"
            uv pip install $whl.FullName
            if ($LASTEXITCODE -ne 0) { Err "pyzed .whl 설치 실패"; exit 1 }
            Remove-Item $whl.FullName -Force
        } else {
            Err "pyzed 빌드 실패 (다운로드된 .whl 도 없음)"; exit 1
        }
    }
}

# -------------------------------------------------------------------------
# 5. 검증
# -------------------------------------------------------------------------
Log "5/5  검증"
$verifyCmd = "import os; os.add_dll_directory(r'$ZedBinDir'); import pyzed.sl as sl; print('ZED SDK', sl.Camera().get_sdk_version())"
$verifyOutput = uv run python -c $verifyCmd 2>&1
if ($LASTEXITCODE -eq 0) {
    Log "         $verifyOutput"
    Log "환경 세팅 완료."
} else {
    Err "pyzed import 검증 실패. ZED SDK 버전과 파이썬 버전을 확인하세요."
    Err "상세: $verifyOutput"
    exit 1
}

} finally { Pop-Location }
