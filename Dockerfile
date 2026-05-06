# Ubuntu + Python 3.11 (managed by uv) + project venv.
# GPU: 호스트에 NVIDIA 드라이버 및 nvidia-container-toolkit 설치 후
#      docker run --gpus all ...  (CUDA용 torch는 uv가 pytorch-cu128에서 설치함)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libavcodec-extra \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version .project_root README.md ./
COPY scripts ./scripts
COPY src ./src
COPY templates ./templates
COPY tomato_observer_app.py ./
COPY scripts/main_tomato_observer.py scripts/main_tomato_observer.py

RUN uv sync --frozen --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5000

# 웹 UI (Flask는 이미 0.0.0.0:5000 에 바인드)
CMD ["python", "tomato_observer_app.py"]
