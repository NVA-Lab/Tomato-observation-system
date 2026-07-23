## Tomato Observation System

YOLO 기반 토마토 관측 시스템입니다.  
실시간 영상에서 토마토를 검출하고(ripe/unripe), 객체 추적을 통해 개수를 집계하며, CLI 또는 Flask 웹 UI로 실행할 수 있습니다.

## 폴더 구조

```
tomato-observation-system/
├── configs/                    # 설정 파일용 디렉터리(선택)
├── models/                     # YOLO 가중치(.pt)
├── scripts/                    # 실행용 bash 런처(.sh)만 위치
│   ├── tomato_observer.sh      # 라이브 카메라·비디오 파일 통합 CLI 런처
│   ├── basic_tracker.sh        # ByteTrack/SORT 기본 트래킹 런처
│   ├── postprocess_dataset.sh  # SVO→학습 데이터셋 빌드 런처
│   └── setup_env.sh            # 다른 컴 환경 자동 세팅
├── src/
│   ├── dataset/                # 데이터셋 빌드 로직(build_training_dataset)
│   └── tracking/
│       ├── tomato_observer.py  # 라이브/비디오 CLI 로직
│       ├── basic_tracker.py    # ByteTrack/SORT CLI 로직(CONFIG)
│       ├── pipelines/          # ByteTrack/SORT·토마토 파이프라인
│       └── utils/              # ROI, 모션 등 유틸
├── templates/
│   └── index.html              # Flask 웹 UI 템플릿
├── tomato_observer_app.py      # 웹 UI 서버
├── pyproject.toml
├── uv.lock
├── README.md
├── .python-version
├── .gitignore
├── .project_root
└── .cursorrules
```

`uv sync` 후에는 프로젝트 루트에 `.venv`가 생기며, 실행 시 `__pycache__`가 생성될 수 있습니다.

## 주요 기능

- 실시간 입력 처리: 웹캠/비디오(OpenCV), ZED(옵션)
- 토마토 상태 분류: `ripe`, `unripe`
- 객체 추적: `ByteTrack` 또는 `SORT` 선택 가능
- ROI 기반 추론 영역 제한, 모션 보정/안정화 옵션 제공
- 웹 UI에서 시작/중지 및 임계값 조절
- 세션 통계 및 CSV 저장 지원

## 환경 요구사항

- Python 3.11 이상
- 주요 패키지:
  - `torch`, `torchvision`
  - `ultralytics`
  - `opencv-python`
  - `flask`
  - `supervision`, `trackers`, `vidstab`, `pyrootutils`

## 설치

이 프로젝트는 `uv` 기준으로 의존성을 관리합니다.

```bash
uv sync
```

필요 시 실행:

```bash
uv run python --version
```

## 데이터셋 후처리 (SVO → 학습 데이터)

`dataset/`의 `{stem}.svo2 + {stem}.csv` 세션에서 RGB/depth/confidence + COCO 어노테이션을
추출해 detector 학습용 데이터셋을 만듭니다. (`scripts/postprocess_dataset.sh` → `src/dataset/build_training_dataset.py`)

이 작업은 `pyproject.toml` 의존성 외에 **ZED SDK + pyzed(Python API) + NVIDIA GPU/CUDA**가 추가로 필요합니다.
`pyzed`는 pip/uv로 설치되지 않고 ZED SDK 설치 시 딸려오는 바인딩이라, `uv sync`만으로는 실행되지 않습니다.

**다른 로컬 컴 세팅 (자동 스크립트)**

`scripts/setup_env.sh`가 uv 설치 · `uv sync` · uv venv에 pyzed 설치 · 검증까지 자동 처리합니다.
단, **ZED SDK 본체**는 먼저 수동 설치해야 합니다(Stereolabs 로그인 필요).

```bash
# 1) ZED SDK 5.2.x 먼저 설치: https://www.stereolabs.com/developers/release
# 2) 환경 자동 세팅 (ZED SDK가 비표준 경로면 ZED_INSTALL_DIR로 지정)
scripts/setup_env.sh
#   ZED_INSTALL_DIR=/설치/경로 scripts/setup_env.sh

# 3) 후처리 실행
uv run scripts/postprocess_dataset.sh dataset --only-annotated
```

> 주의: `pyzed`는 `uv`가 쓰는 venv 파이썬에 설치돼야 합니다. `setup_env.sh`는 `uv run`으로 설치·실행하므로
> 파이썬 버전 불일치를 자동으로 피합니다. ZED SDK 버전은 녹화에 쓴 버전(현재 5.2.3)과 맞추는 것을 권장합니다.

## 실행 방법

### 1) CLI 실행

`source` 인자로 입력을 결정합니다 — 정수는 웹캠 인덱스(라이브), 그 외는 비디오 파일 경로.

```bash
scripts/tomato_observer.sh 0 --show-window            # 라이브 웹캠(인덱스 0)
scripts/tomato_observer.sh clip.mp4 --write-video     # 비디오 파일 처리
```

옵션(`--conf`, `--iou`, `--tracker bytetrack|sort`, `--mode full|sbs_left|sbs_right` 등)은
`scripts/tomato_observer.sh -h`로 확인할 수 있습니다.

### 2) 웹 UI 실행

```bash
python tomato_observer_app.py
```

실행 후 브라우저에서 Flask 서버 주소(기본 localhost)로 접속하여 카메라 시작/중지 및 임계값을 제어합니다.

## 설정 가이드

주요 설정 위치:

- `scripts/tomato_observer.sh -h`의 CLI 옵션(conf/iou/tracker/mode 등)
- `src/tracking/pipelines/tomato_observer_pipeline.py`의 `DEFAULT_CONFIG`

자주 조절하는 파라미터:

- `tracker_type`: `bytetrack` / `sort`
- `camera_backend`: `opencv` / `zed`
- `source`: 카메라 인덱스 또는 영상 파일 경로
- `model_path`: YOLO 모델 경로
- `device`: `cpu` 또는 CUDA 디바이스 인덱스
- `conf`: confidence threshold
- `nms_iou`: NMS IoU threshold
- `min_box_area`: 최소 박스 면적 필터

웹 UI 관련:

- `tomato_observer_app.py`에서 `camera_mode`, `conf_thres`, `iou_thres` 기본값을 관리합니다.

