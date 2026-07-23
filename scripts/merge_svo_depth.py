#!/usr/bin/env python3
"""Replay a ZED SVO recording to compute per-detection depth and merge it into the detection CSV.

The live app records raw stereo frames to an SVO (depth off, for live speed) and writes
detection bboxes to a CSV with a `svo_frame_index` column that lines up 1:1 with the SVO's
own frame numbering. This script replays the SVO with depth enabled, samples the median
depth inside each bbox, and writes a new CSV with `depth` / `depth_valid_ratio` columns added.

Example:
  uv run python scripts/merge_svo_depth.py dataset/rec_20260720_120000.svo2
  uv run python scripts/merge_svo_depth.py rec.svo2 --csv rec.csv --out rec_depth.csv --depth-mode NEURAL --units METER
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _sample_bbox_depth(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[str, str]:
    h, w = depth_map.shape[:2]
    xi1, yi1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    xi2, yi2 = max(xi1 + 1, min(x2, w)), max(yi1 + 1, min(y2, h))
    patch = depth_map[yi1:yi2, xi1:xi2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return "", "0.0000"
    depth_val = float(np.median(valid))
    valid_ratio = valid.size / patch.size
    return f"{depth_val:.4f}", f"{valid_ratio:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svo", help="Path to the recorded .svo2 file")
    parser.add_argument("--csv", default=None, help="Detection CSV path (default: same stem as the SVO, .csv)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: {csv_stem}_depth.csv)")
    parser.add_argument(
        "--depth-mode",
        default="NEURAL",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS"],
        help="ZED depth computation mode used during replay (default: NEURAL)",
    )
    parser.add_argument(
        "--units",
        default="METER",
        choices=["MILLIMETER", "CENTIMETER", "METER", "INCH", "FOOT"],
        help="Depth unit (default: METER)",
    )
    args = parser.parse_args()

    try:
        import pyzed.sl as sl
    except ImportError as e:
        raise SystemExit(
            "This script requires the ZED SDK Python API (pyzed). "
            "Install from https://www.stereolabs.com/docs/app-development/python/install"
        ) from e

    svo_path = Path(args.svo).expanduser().resolve()
    if not svo_path.is_file():
        raise SystemExit(f"SVO file not found: {svo_path}")

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else svo_path.with_suffix(".csv")
    if not csv_path.is_file():
        raise SystemExit(f"Detection CSV not found: {csv_path} (pass --csv to point at the right file)")

    out_path = (
        Path(args.out).expanduser().resolve() if args.out else csv_path.with_name(f"{csv_path.stem}_depth.csv")
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "svo_frame_index" not in fieldnames:
        raise SystemExit(
            "CSV has no 'svo_frame_index' column - it was not recorded alongside an SVO, "
            "or predates this feature. Cannot align bboxes to SVO frames."
        )

    rows_by_frame: dict[int, list[dict]] = {}
    skipped = 0
    for row in rows:
        raw = (row.get("svo_frame_index") or "").strip()
        if raw == "":
            skipped += 1
            continue
        rows_by_frame.setdefault(int(raw), []).append(row)

    if not rows_by_frame:
        raise SystemExit("No CSV rows have a valid svo_frame_index; nothing to merge.")

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = getattr(sl.UNIT, args.units)

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Failed to open SVO: {err}")

    runtime_params = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    matched = 0

    try:
        while True:
            err = zed.grab(runtime_params)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[merge_svo_depth] grab error, stopping early: {err}")
                break

            pos = zed.get_svo_position()
            frame_rows = rows_by_frame.get(pos)
            if not frame_rows:
                continue

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_np = np.squeeze(depth_mat.get_data())

            for row in frame_rows:
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                depth_val, valid_ratio = _sample_bbox_depth(depth_np, x1, y1, x2, y2)
                row["depth"] = depth_val
                row["depth_valid_ratio"] = valid_ratio
                matched += 1
    finally:
        zed.close()

    out_fieldnames = fieldnames + [f for f in ("depth", "depth_valid_ratio") if f not in fieldnames]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[merge_svo_depth] matched {matched}/{len(rows) - skipped} alignable row(s) "
        f"({skipped} row(s) had no svo_frame_index and were left without depth) -> {out_path}"
    )


if __name__ == "__main__":
    main()
