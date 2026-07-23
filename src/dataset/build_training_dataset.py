#!/usr/bin/env python3
"""Build a detector-training dataset (RGB + aligned depth + confidence + intrinsics + COCO
bbox/class annotations + scene/clip grouping) from recorded ZED data-collection sessions.

For every {stem}.svo2 + {stem}.csv pair in the dataset directory, replays the SVO with depth
enabled and extracts, for every frame:
  - the raw left RGB image                    -> {out}/rgb/{stem}_{frame:06d}.jpg
  - a 16-bit depth map (mm, aligned to RGB)    -> {out}/depth/{stem}_{frame:06d}.png  (0=invalid)
  - the sensor confidence map (raw ZED values) -> {out}/conf/{stem}_{frame:06d}.png
  - bbox + class annotations for that frame, each with a bbox-median depth, an IoU-based
    occlusion estimate, and a border-touching truncation flag

Camera intrinsics/distortion/baseline are written once per session to {out}/calib/{stem}.json
(depth is already registered to the left RGB image by the ZED SDK, so no extra RGB-D extrinsic
is needed here). {out}/meta.csv lists every frame with its scene_id/clip_id/plant/row/lighting
group keys (for leakage-free splits) plus timestamp and sensor info; these group keys default to
the recording stem but can be overridden per-session via an optional
{dataset_dir}/session_meta.csv (columns: stem,scene_id,plant,row,lighting).

Per-session extraction results are cached as COCO fragments under {out}/_fragments/{stem}.json so
re-running only re-extracts sessions that are new (see --force). The final
{out}/annotations/annotations.json merges every session's fragment with sequential global ids.

Example:
  scripts/postprocess_dataset.sh dataset
  python3 -m src.dataset.build_training_dataset dataset --out dataset/training_dataset --force
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Must match CLASS_NAMES in src/tracking/pipelines/tomato_observer_pipeline.py
CATEGORIES = [
    {"id": 0, "name": "ripe"},
    {"id": 1, "name": "unripe"},
]
RIPENESS_TO_CATEGORY_ID = {c["name"]: c["id"] for c in CATEGORIES}

OCCLUSION_IOU_THRESHOLD = 0.1
TRUNCATION_MARGIN_PX = 2


def _sample_bbox_depth_mm(depth_mm: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[float | None, float]:
    h, w = depth_mm.shape[:2]
    xi1, yi1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    xi2, yi2 = max(xi1 + 1, min(x2, w)), max(yi1 + 1, min(y2, h))
    patch = depth_mm[yi1:yi2, xi1:xi2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None, 0.0
    return float(np.median(valid)), float(valid.size / patch.size)


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_rows_by_frame(csv_path: Path) -> tuple[dict[int, list[dict]], int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows_by_frame: dict[int, list[dict]] = {}
    skipped = 0
    for row in rows:
        raw = (row.get("svo_frame_index") or "").strip()
        if raw == "":
            skipped += 1
            continue
        rows_by_frame.setdefault(int(raw), []).append(row)
    return rows_by_frame, skipped


def _load_session_meta_overrides(dataset_dir: Path) -> dict[str, dict]:
    path = dataset_dir / "session_meta.csv"
    if not path.is_file():
        return {}
    overrides: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            stem = (row.get("stem") or "").strip()
            if not stem:
                continue
            overrides[stem] = {
                "scene_id": (row.get("scene_id") or "").strip(),
                "plant": (row.get("plant") or "").strip(),
                "row": (row.get("row") or "").strip(),
                "lighting": (row.get("lighting") or "").strip(),
            }
    return overrides


def _write_calib(zed, sl, stem: str, calib_dir: Path, depth_mode: str) -> None:
    info = zed.get_camera_information()
    cal = info.camera_configuration.calibration_parameters
    left = cal.left_cam
    calib = {
        "session": stem,
        "camera_model": str(info.camera_model),
        "serial_number": info.serial_number,
        "resolution": {"width": info.camera_configuration.resolution.width, "height": info.camera_configuration.resolution.height},
        "fps": info.camera_configuration.fps,
        "left_intrinsics": {
            "fx": left.fx,
            "fy": left.fy,
            "cx": left.cx,
            "cy": left.cy,
            "distortion": list(left.disto),
        },
        "baseline_mm": cal.get_camera_baseline(),
        "depth_units": "mm",
        "depth_aligned_to": "left_rgb",
        "depth_mode": depth_mode,
        "depth_clip_mm": 65535,
        "depth_clip_note": "distant background beyond ~65.5m saturates to 65535, not 0/invalid - only 0 means invalid",
        "confidence_units": "raw ZED sl.MEASURE.CONFIDENCE, range 0-100",
        "confidence_direction": "LOWER_IS_MORE_CONFIDENT",
        "confidence_direction_note": (
            "ZED convention: 0=most confident, 100=least confident. This is the OPPOSITE of "
            "gating pipelines that assume higher=more confident (e.g. DA3-style conf in [1,5]). "
            "Convert before use: conf_used = 1 - (conf_raw / 100)."
        ),
    }
    calib_dir.mkdir(parents=True, exist_ok=True)
    (calib_dir / f"{stem}.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")


def _build_session_fragment(
    sl,
    svo_path: Path,
    csv_path: Path,
    stem: str,
    out_dir: Path,
    depth_mode: str,
    jpg_quality: int,
    stride: int,
    only_annotated: bool,
) -> dict:
    rows_by_frame, skipped = _load_rows_by_frame(csv_path)

    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    conf_dir = out_dir / "conf"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    init.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO {svo_path}: {err}")

    _write_calib(zed, sl, stem, out_dir / "calib", depth_mode)

    runtime_params = sl.RuntimeParameters()
    left_mat = sl.Mat()
    depth_mat = sl.Mat()
    conf_mat = sl.Mat()

    images: list[dict] = []
    annotations: list[dict] = []
    local_image_id = 0
    local_ann_id = 0
    matched_rows = 0
    unknown_ripeness: dict[str, int] = {}
    frames_skipped_by_stride = 0

    try:
        while True:
            err = zed.grab(runtime_params)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[build_training_dataset] {stem}: grab error, stopping early: {err}")
                break

            pos = zed.get_svo_position()
            has_ann = bool(rows_by_frame.get(pos))

            if only_annotated:
                keep = has_ann
            else:
                keep = has_ann or stride <= 1 or (pos % stride == 0)
            if not keep:
                frames_skipped_by_stride += 1
                continue

            frame_name = f"{stem}_{pos:06d}"
            ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            bgra = left_mat.get_data()
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            h, w = bgr.shape[:2]
            cv2.imwrite(str(rgb_dir / f"{frame_name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_mm = np.squeeze(depth_mat.get_data())
            depth_mm_clean = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
            depth_u16 = np.clip(depth_mm_clean, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_u16)

            zed.retrieve_measure(conf_mat, sl.MEASURE.CONFIDENCE)
            conf_np = np.nan_to_num(np.squeeze(conf_mat.get_data()), nan=0.0, posinf=0.0, neginf=0.0)
            conf_u8 = np.clip(conf_np, 0, 255).astype(np.uint8)
            cv2.imwrite(str(conf_dir / f"{frame_name}.png"), conf_u8)

            images.append(
                {
                    "id": local_image_id,
                    "file_name": f"rgb/{frame_name}.jpg",
                    "depth_file": f"depth/{frame_name}.png",
                    "conf_file": f"conf/{frame_name}.png",
                    "width": int(w),
                    "height": int(h),
                    "session": stem,
                    "svo_frame_index": int(pos),
                    "timestamp_ns": int(ts_ns),
                }
            )

            frame_anns: list[dict] = []
            for row in rows_by_frame.get(pos, []):
                ripeness_raw = (row.get("ripeness") or "").strip().lower()
                cat_id = RIPENESS_TO_CATEGORY_ID.get(ripeness_raw)
                if cat_id is None:
                    unknown_ripeness[ripeness_raw] = unknown_ripeness.get(ripeness_raw, 0) + 1
                    continue
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                bw, bh = x2 - x1, y2 - y1
                depth_val, valid_ratio = _sample_bbox_depth_mm(depth_mm, x1, y1, x2, y2)
                truncated = x1 <= TRUNCATION_MARGIN_PX or y1 <= TRUNCATION_MARGIN_PX or x2 >= w - TRUNCATION_MARGIN_PX or y2 >= h - TRUNCATION_MARGIN_PX
                frame_anns.append(
                    {
                        "id": local_ann_id,
                        "image_id": local_image_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, bw, bh],
                        "_xyxy": (float(x1), float(y1), float(x2), float(y2)),
                        "area": int(bw * bh),
                        "iscrowd": 0,
                        "confidence": float(row.get("confidence") or 0.0),
                        "depth_mm": depth_val,
                        "depth_valid_ratio": valid_ratio,
                        "track_id": int(row["id"]) if (row.get("id") or "").strip() else None,
                        "truncated": bool(truncated),
                    }
                )
                local_ann_id += 1
                matched_rows += 1

            for i, ann in enumerate(frame_anns):
                max_iou = 0.0
                for j, other in enumerate(frame_anns):
                    if i == j:
                        continue
                    max_iou = max(max_iou, _iou_xyxy(ann["_xyxy"], other["_xyxy"]))
                ann["occlusion_iou"] = round(max_iou, 4)
                ann["occluded"] = bool(max_iou > OCCLUSION_IOU_THRESHOLD)
            for ann in frame_anns:
                del ann["_xyxy"]
                annotations.append(ann)

            local_image_id += 1
    finally:
        zed.close()

    total_rows = sum(len(v) for v in rows_by_frame.values())
    print(
        f"[build_training_dataset] {stem}: {local_image_id} frame(s), {matched_rows}/{total_rows} "
        f"annotation(s) matched ({skipped} CSV row(s) had no svo_frame_index"
        f"{f', {frames_skipped_by_stride} frame(s) skipped by stride/only-annotated' if frames_skipped_by_stride else ''})"
    )
    if unknown_ripeness:
        print(
            f"[build_training_dataset] WARNING {stem}: dropped {sum(unknown_ripeness.values())} row(s) "
            f"with unrecognized ripeness label(s) {unknown_ripeness} - check for typos vs. {list(RIPENESS_TO_CATEGORY_ID)}"
        )
    return {"images": images, "annotations": annotations}


def _merge_fragments(fragments: dict[str, dict], session_meta: dict[str, dict]) -> tuple[dict, list[dict]]:
    images: list[dict] = []
    annotations: list[dict] = []
    meta_rows: list[dict] = []
    next_image_id = 0
    next_ann_id = 0

    for stem in sorted(fragments):
        frag = fragments[stem]
        override = session_meta.get(stem, {})
        scene_id = override.get("scene_id") or stem
        plant = override.get("plant", "")
        row_ = override.get("row", "")
        lighting = override.get("lighting", "")

        id_map = {}
        for img in frag["images"]:
            new_img = dict(img)
            id_map[img["id"]] = next_image_id
            new_img["id"] = next_image_id
            images.append(new_img)

            ts_iso = datetime.fromtimestamp(img["timestamp_ns"] / 1e9, tz=timezone.utc).isoformat()
            meta_rows.append(
                {
                    "image_id": next_image_id,
                    "file_name": new_img["file_name"],
                    "depth_file": new_img["depth_file"],
                    "conf_file": new_img["conf_file"],
                    "session": stem,
                    "clip_id": stem,
                    "scene_id": scene_id,
                    "plant": plant,
                    "row": row_,
                    "lighting": lighting,
                    "svo_frame_index": img["svo_frame_index"],
                    "timestamp_ns": img["timestamp_ns"],
                    "timestamp_iso": ts_iso,
                    "width": img["width"],
                    "height": img["height"],
                }
            )
            next_image_id += 1

        for ann in frag["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            annotations.append(new_ann)
            next_ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    return coco, meta_rows


def _write_meta_csv(meta_rows: list[dict], out_dir: Path) -> None:
    fieldnames = [
        "image_id", "file_name", "depth_file", "conf_file",
        "session", "clip_id", "scene_id", "plant", "row", "lighting",
        "svo_frame_index", "timestamp_ns", "timestamp_iso", "width", "height",
    ]
    with (out_dir / "meta.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Directory containing {stem}.svo2 + {stem}.csv session pairs")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: {dataset_dir}/training_dataset)")
    parser.add_argument(
        "--depth-mode",
        default="NEURAL",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS"],
        help="ZED depth computation mode used during replay (default: NEURAL)",
    )
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality for extracted RGB frames (default: 95)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame that has no annotation (annotated frames are always kept regardless). Default: 1 (keep all)",
    )
    parser.add_argument(
        "--only-annotated",
        action="store_true",
        help="Only extract frames that have at least one bbox annotation (drop all negative/empty frames). Overrides --stride.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract sessions even if a cached fragment exists")
    args = parser.parse_args()

    try:
        import pyzed.sl as sl
    except ImportError as e:
        raise SystemExit(
            "This script requires the ZED SDK Python API (pyzed). "
            "Install from https://www.stereolabs.com/docs/app-development/python/install"
        ) from e

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    out_dir = args.out.expanduser().resolve() if args.out else dataset_dir / "training_dataset"
    fragments_dir = out_dir / "_fragments"
    annotations_dir = out_dir / "annotations"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    svo_files = sorted(dataset_dir.glob("*.svo2"))
    if not svo_files:
        raise SystemExit(f"No .svo2 files found in {dataset_dir}")

    fragments: dict[str, dict] = {}
    for svo_path in svo_files:
        stem = svo_path.stem
        csv_path = dataset_dir / f"{stem}.csv"
        if not csv_path.is_file():
            print(f"[build_training_dataset] skip {stem}: no matching CSV")
            continue

        extract_key = {
            "stride": args.stride,
            "only_annotated": args.only_annotated,
            "depth_mode": args.depth_mode,
            "jpg_quality": args.jpg_quality,
        }

        frag_path = fragments_dir / f"{stem}.json"
        if frag_path.is_file() and not args.force:
            cached = json.loads(frag_path.read_text(encoding="utf-8"))
            if cached.get("_extract_key") == extract_key:
                print(f"[build_training_dataset] {stem}: using cached fragment (use --force to re-extract)")
                fragments[stem] = cached
                continue
            print(f"[build_training_dataset] {stem}: extract params changed {cached.get('_extract_key')} -> {extract_key}, re-extracting")

        fragment = _build_session_fragment(
            sl, svo_path, csv_path, stem, out_dir, args.depth_mode, args.jpg_quality,
            args.stride, args.only_annotated,
        )
        fragment["_extract_key"] = extract_key
        frag_path.write_text(json.dumps(fragment), encoding="utf-8")
        fragments[stem] = fragment

    if not fragments:
        raise SystemExit("No sessions were processed; nothing to merge.")

    session_meta = _load_session_meta_overrides(dataset_dir)
    merged, meta_rows = _merge_fragments(fragments, session_meta)

    ann_path = annotations_dir / "annotations.json"
    ann_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _write_meta_csv(meta_rows, out_dir)

    print(
        f"[build_training_dataset] wrote {len(merged['images'])} image(s), "
        f"{len(merged['annotations'])} annotation(s) across {len(fragments)} session(s) -> {ann_path}"
    )
    missing_plant = [s for s in fragments if not session_meta.get(s, {}).get("plant")]
    if missing_plant:
        print(
            f"[build_training_dataset] WARNING: {len(missing_plant)} session(s) have no 'plant' group key "
            f"in session_meta.csv: {sorted(missing_plant)}. Splitting only by session/scene_id does NOT "
            "prevent the same physical plant/row from appearing in both train and test if it was recorded "
            "across multiple sessions - fill in session_meta.csv's plant/row columns before doing a split."
        )

    _write_readme(out_dir)


def _write_readme(out_dir: Path) -> None:
    readme = """# Tomato detector training dataset

Layout:
  rgb/{stem}_{frame}.jpg     - raw left RGB frame
  depth/{stem}_{frame}.png   - 16-bit depth map in mm, aligned to rgb/ (0 = invalid pixel)
  conf/{stem}_{frame}.png    - raw ZED sensor confidence map (see confidence direction below)
  calib/{stem}.json          - per-session intrinsics/distortion/baseline
  annotations/annotations.json - COCO bbox+class, with extra per-annotation fields:
                                  depth_mm, depth_valid_ratio, confidence, track_id,
                                  truncated, occlusion_iou, occluded
  meta.csv                   - per-image scene_id/clip_id/plant/row/lighting/timestamp

IMPORTANT - confidence direction:
  ZED's raw confidence convention is LOWER = MORE CONFIDENT (0=best, 100=worst), which is the
  OPPOSITE of gating pipelines that assume higher=more confident. Convert before use:
    conf_used = 1 - (conf_raw / 100)
  See calib/{stem}.json -> confidence_direction_note for the same warning in machine-readable form.

IMPORTANT - group-disjoint splitting:
  scene_id/clip_id default to the recording session name. That alone only guarantees no single
  recording clip is split across train/test - it does NOT guarantee the same physical plant/row
  stays on one side if you recorded it in more than one session. Fill in
  {dataset_dir}/session_meta.csv (columns: stem,scene_id,plant,row,lighting) and rerun before
  splitting, then split by `plant` (or `row`), not by session/scene_id alone.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
