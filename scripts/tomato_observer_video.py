#!/usr/bin/env python3
"""Run the same tomato observer pipeline (YOLO + tracking + CSV) on a video file.

Mirrors the web app (`app.py`) defaults and tunable settings, without Flask or a live camera.
Writes detection CSV (and optionally an annotated MP4) to the export directory.

Example:
  uv run python scripts/tomato_observer_video.py /path/to/input.mp4 --write-video
  uv run python scripts/tomato_observer_video.py clip.mp4 --conf 0.75 --tracker sort --mode sbs_left
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrootutils
import torch
from ultralytics import YOLO

pyrootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.tracking.pipelines.tomato_observer_pipeline import DEFAULT_CONFIG, run as run_observer

# Same columns as `app.py` CSV export.
CSV_FIELDNAMES = [
    "timestamp",
    "elapsed",
    "frame_index",
    "svo_frame_index",
    "id",
    "ripeness",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area",
    "confidence_threshold",
    "iou_threshold",
    "total_fps",
]


def resolve_export_dir() -> Path:
    raw = (os.environ.get("TOMATO_DATASET_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (REPO_ROOT / "dataset").resolve()


def format_elapsed(sec: float) -> str:
    sec = max(int(sec), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def camera_overrides(mode: str) -> dict:
    """Align with `app.py` camera modes; here `full` is plain video (no SBS crop)."""
    mode = str(mode).lower().strip()
    if mode == "full":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": False,
            "left_only": False,
            "camera_width": None,
            "camera_height": None,
            "opencv_use_mjpeg": False,
            "opencv_api": "default",
        }
    if mode == "sbs_left":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": True,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": False,
            "opencv_api": "default",
        }
    if mode == "sbs_right":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": False,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": False,
            "opencv_api": "default",
        }
    raise ValueError(f"Unknown --mode: {mode}")


def resolve_model_path(model_path: str) -> str:
    from src.tracking.pipelines import tomato_observer_pipeline as top

    return top._resolve_model_path(model_path)


def torch_device_str(device_cfg: object) -> str:
    from src.tracking.pipelines import tomato_observer_pipeline as top

    return top._torch_device_str(device_cfg)


def prime_model(model: YOLO, device_str: str) -> None:
    z = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(z, conf=0.5, iou=0.5, verbose=False, imgsz=640)
    if torch.cuda.is_available() and str(device_str).strip().lower() != "cpu":
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    d_conf = 0.8
    d_iou = 0.25
    p = argparse.ArgumentParser(
        description="Tomato observer on a video file (same pipeline as the web app).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Input video file path")
    p.add_argument("--conf", type=float, default=d_conf, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=d_iou, help="NMS IoU threshold")
    p.add_argument(
        "--tracker",
        choices=("bytetrack", "sort"),
        default="bytetrack",
        help="Multi-object tracker",
    )
    p.add_argument(
        "--stabilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Video stabilization (VidStab)",
    )
    p.add_argument(
        "--show-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw track traces on the annotated frames",
    )
    p.add_argument(
        "--raw-detections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include every YOLO detection even before the tracker confirms a stable id (id left blank for those)",
    )
    p.add_argument(
        "--mode",
        choices=("full", "sbs_left", "sbs_right"),
        default="full",
        help="full=regular video; sbs_*=side-by-side stereo crop (half frame)",
    )
    p.add_argument("--roi-width", type=float, default=0.7, help="Center ROI width ratio")
    p.add_argument("--roi-height", type=float, default=1.0, help="Center ROI height ratio")
    p.add_argument("--model", type=str, default="yolo26n_640.pt", help="YOLO weights path or name")
    p.add_argument("--device", type=str, default="0", help='Torch device: "cpu" or CUDA index like "0"')
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV / annotated video (default: dataset/ or TOMATO_DATASET_DIR)",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="",
        help="Output file basename without extension (default: input video stem)",
    )
    p.add_argument(
        "--write-video",
        action="store_true",
        help="Also save annotated MP4 next to the CSV",
    )
    p.add_argument(
        "--show-window",
        action="store_true",
        help="Preview annotated frames (press q or Esc to stop early)",
    )
    p.add_argument("--min-box-area", type=int, default=int(DEFAULT_CONFIG["min_box_area"]))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else resolve_export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (args.stem or "").strip() or video_path.stem

    source_str = str(video_path)
    overrides = camera_overrides(args.mode)

    resolved_model = resolve_model_path(args.model)
    device_str = torch_device_str(args.device)
    yolo_model = YOLO(resolved_model)
    yolo_model.to(device_str)
    prime_model(yolo_model, device_str)

    annotated_rel: str | None = None
    if args.write_video:
        annotated_rel = str((out_dir / f"{stem}_annotated.mp4").resolve())

    runtime_lock = threading.Lock()
    runtime: dict = {
        "lock": runtime_lock,
        "conf": float(args.conf),
        "nms_iou": float(args.iou),
        "show_trace": bool(args.show_trace),
        # When no output video, pipeline uses dynamic_output=True; keep recording on so CSV rows emit.
        "recording": annotated_rel is None,
        "recording_video_relpath": None,
    }

    csv_rows_out: list[dict] = []
    session_start = time.time()

    def on_frame(_annotated, meta: dict) -> None:
        rows = meta.get("csv_rows") or []
        if rows:
            csv_rows_out.extend(rows)

    cfg: dict = {
        **DEFAULT_CONFIG,
        **overrides,
        "model_path": args.model,
        "yolo_model": yolo_model,
        "device": args.device,
        "source": source_str,
        "tracker_type": args.tracker,
        "conf": float(args.conf),
        "nms_iou": float(args.iou),
        "use_stabilization": bool(args.stabilization),
        "show_trace": bool(args.show_trace),
        "use_center_roi": True,
        "use_raw_detections": bool(args.raw_detections),
        "roi_width_ratio": float(args.roi_width),
        "roi_height_ratio": float(args.roi_height),
        "min_box_area": int(args.min_box_area),
        "show_window": bool(args.show_window),
        "output_path": annotated_rel,
        "stop_event": threading.Event(),
        "runtime": runtime,
        "on_frame": on_frame,
        "session_elapsed_fn": lambda: format_elapsed(time.time() - session_start),
        "_exit_stats": {},
    }

    print(
        f"[tomato_observer_video] input={video_path}\n"
        f"  conf={args.conf} iou={args.iou} tracker={args.tracker} stabilization={args.stabilization}\n"
        f"  mode={args.mode} out_dir={out_dir}\n"
        f"  csv={out_dir / (stem + '.csv')}  video={'(none)' if not annotated_rel else annotated_rel}"
    )

    try:
        run_observer(cfg)
    except KeyboardInterrupt:
        print("\n[tomato_observer_video] interrupted", file=sys.stderr)
        sys.exit(130)

    csv_path = out_dir / f"{stem}.csv"
    if csv_rows_out:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, restval="")
            w.writeheader()
            w.writerows(csv_rows_out)
        print(f"[tomato_observer_video] wrote CSV ({len(csv_rows_out)} rows): {csv_path}")
    else:
        print("[tomato_observer_video] warning: no CSV rows (no tracked detections or empty video)", file=sys.stderr)

    if args.write_video and annotated_rel:
        ap = Path(annotated_rel)
        if ap.is_file() and ap.stat().st_size > 0:
            print(f"[tomato_observer_video] wrote video: {ap}")
        else:
            print(f"[tomato_observer_video] warning: expected video missing or empty: {ap}", file=sys.stderr)


if __name__ == "__main__":
    main()
