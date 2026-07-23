#!/usr/bin/env python3
"""CLI: YOLO + ByteTrack/SORT 트래킹을 비디오에 실행한다.

tracker 인자로 파이프라인을 선택한다 (bytetrack | sort). 공통 설정은 아래 CONFIG dict에서
수정한다 (두 파이프라인 모두 config.get()으로 읽으므로 서로의 키는 무시된다).

실행 방법 (repo 루트에서 모듈로 실행):
  scripts/basic_tracker.sh bytetrack
  scripts/basic_tracker.sh sort --output tracking_result/my.mp4
  python3 -m src.tracking.basic_tracker bytetrack
"""

from __future__ import annotations

import argparse

from src.tracking.pipelines import bytetrack_pipeline, sort_pipeline

# ---------------------------------------------------------------------------
# 공통 설정 (여기서만 수정)
# ---------------------------------------------------------------------------
CONFIG = {
    "source": "notebook/rgb.mp4",
    "model_path": "runs/yolo26_custom_tomato/trained_yolo26_custom.pt",
    "show_window": True,
    "conf": 0.5,
    "iou": 0.3,
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.3,  # ByteTrack용
    "minimum_iou_threshold": 0.3,       # SORT용
    "high_conf_det_threshold": 0.25,    # ByteTrack용
    "minimum_consecutive_frames": 1,
    "frame_rate": 30,
    "motion_compensation": True,
    "motion_max_points": 900,
    "motion_min_distance": 6,
    "motion_block_size": 5,
    "motion_quality_level": 0.003,
    "motion_ransac_reproj_threshold": 2.5,
    "show_trace": False,
    "trace_length": 30,
}

# tracker 이름 -> (파이프라인 모듈, 기본 출력 경로)
_TRACKERS = {
    "bytetrack": (bytetrack_pipeline, "tracking_result/basic_bytetrack.mp4"),
    "sort": (sort_pipeline, "tracking_result/basic_sort.mp4"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + ByteTrack/SORT 기본 트래킹 러너")
    p.add_argument("tracker", choices=sorted(_TRACKERS), help="사용할 트래커")
    p.add_argument("--source", default=None, help="입력 영상 경로 (기본: CONFIG.source)")
    p.add_argument("--output", default=None, help="출력 영상 경로 (기본: 트래커별 기본값)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline, default_output = _TRACKERS[args.tracker]

    cfg = {**CONFIG}
    if args.source:
        cfg["source"] = args.source
    cfg["output_path"] = args.output or default_output

    result = pipeline.run(cfg)
    print(
        f"[결과] ripe {len(result['unique_ids'][0])}개 / "
        f"unripe {len(result['unique_ids'][1])}개 (누적 고유 ID 기준)"
    )


if __name__ == "__main__":
    main()
