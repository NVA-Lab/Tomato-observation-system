"""Tomato observer web UI — Flask on top of `tomato_observer_pipeline`.

Before the HTTP server starts: YOLO is loaded, GPU is warmed with one dummy `predict` (avoids
first-frame stutter). On Start, OpenCV capture is opened in the main thread, then the worker
reads immediately.
"""

from __future__ import annotations

from typing import Any

import csv
import io
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrootutils  # type: ignore[import-untyped]
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_export_dataset_dir() -> Path:
    """CSV/MP4 저장 디렉터리. 환경변수 TOMATO_DATASET_DIR이 있으면 우선."""
    raw = (os.environ.get("TOMATO_DATASET_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("/home/user/Tomato/Dataset0514").resolve()


EXPORT_DATASET_DIR = _resolve_export_dataset_dir()
pyrootutils.setup_root(str(_REPO_ROOT), indicator="pyproject.toml", pythonpath=True)

from src.tracking.pipelines import tomato_observer_pipeline as _top
from src.tracking.pipelines.tomato_observer_pipeline import DEFAULT_CONFIG, run as run_observer

torch.backends.cudnn.benchmark = True

DEFAULT_CONF_THRES = 0.8
DEFAULT_IOU_THRES = 0.25

# Single UI control → full pipeline camera-related settings.
VALID_CAMERA_MODES = frozenset({"webcam", "sbs_left", "sbs_right", "zed"})
VALID_TRACKER_TYPES = frozenset({"bytetrack", "sort"})


def _sanitize_export_basename(raw: object, *, fallback_stem: str) -> str:
    """Strip unsafe path chars; empty or invalid → fallback_stem (no extension)."""
    s = str(raw or "").strip()
    low = s.lower()
    for ext in (".mp4", ".csv", ".mov", ".avi", ".mkv", ".webm"):
        if low.endswith(ext):
            s = s[: -len(ext)].strip()
            low = s.lower()
            break
    if not s:
        return fallback_stem
    bad = '\\/:*?"<>|\n\r\t\x00'
    out = "".join("_" if (c in bad or ord(c) < 32) else c for c in s)
    out = out.strip(" .")
    if not out or out in (".", ".."):
        return fallback_stem
    if len(out) > 120:
        out = out[:120]
    return out


def _content_disposition_attachment(download_name: str) -> str:
    """Build Content-Disposition for downloads (ASCII → quoted filename, else RFC 5987)."""
    from urllib.parse import quote

    if not download_name:
        return 'attachment; filename="download"'
    safe_ascii = all(32 <= ord(c) <= 126 for c in download_name) and '"' not in download_name and "\\" not in download_name
    if safe_ascii:
        return f'attachment; filename="{download_name}"'
    return "attachment; filename*=UTF-8''" + quote(download_name)


_yolo_lock = threading.Lock()
_yolo_cached: YOLO | None = None
_yolo_cache_key: tuple[str, str] | None = None


def ensure_detector_ready(model_path: str, device: str) -> YOLO:
    """Load YOLO once (thread-safe); reused across sessions so Start does not wait on weights."""
    global _yolo_cached, _yolo_cache_key
    resolved = _top._resolve_model_path(str(model_path))
    dev = _top._torch_device_str(device)
    key = (resolved, dev)
    with _yolo_lock:
        if _yolo_cached is not None and _yolo_cache_key == key:
            return _yolo_cached
        m = YOLO(resolved)
        m.to(dev)
        _yolo_cached = m
        _yolo_cache_key = key
        return m


def _prime_yolo_warmup(model: YOLO, device_str: str) -> None:
    """One dummy run so the first real frame is not stuck on CUDA graph compile / kernels."""
    z = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(z, conf=0.5, iou=0.5, verbose=False, imgsz=640)
    if torch.cuda.is_available() and str(device_str).strip().lower() != "cpu":
        torch.cuda.synchronize()


def _camera_mode_label(mode: str) -> str:
    return {
        "webcam": "일반 웹캠",
        "sbs_left": "양안분할·왼쪽",
        "sbs_right": "양안분할·오른쪽",
        "zed": "ZED",
    }.get(mode, mode)


def _pipeline_camera_overrides(mode: str) -> dict[str, Any]:
    """Map UI `camera_mode` to tomato_observer_pipeline fields (no manual tuning in the browser)."""
    if mode == "zed":
        return {
            "camera_backend": "zed",
            "stereo_sbs": False,
            "left_only": False,
            "camera_width": None,
            "camera_height": None,
        }
    if mode == "webcam":
        o: dict[str, Any] = {
            "camera_backend": "opencv",
            "stereo_sbs": False,
            "left_only": False,
            "camera_width": None,
            "camera_height": None,
            "opencv_use_mjpeg": False,
            "opencv_api": "default",
        }
        if sys.platform == "win32":
            o["opencv_api"] = "dshow"
        return o
    if mode == "sbs_left":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": True,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": True,
            "opencv_api": "default",
        }
    if mode == "sbs_right":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": False,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": True,
            "opencv_api": "default",
        }
    raise ValueError(f"unsupported camera_mode: {mode}")


app = Flask(__name__)

frame_lock = threading.Lock()
control_lock = threading.Lock()
state_lock = threading.Lock()
stop_event = threading.Event()

worker_thread: threading.Thread | None = None
session_start_time: float | None = None
frame_index = 0
pipeline_runtime: dict | None = None

# Gated CSV / video: filled only while UI "녹화" is active; last_export_* is used for downloads.
recording_csv_buffer: list[dict] = []
last_export_csv_rows: list[dict] = []
last_export_video_relpath: str | None = None
active_recording_video_relpath: str | None = None
last_export_stem: str | None = None
active_recording_stem: str | None = None
ui_recording_active = False
last_export_csv_abspath: str | None = None
last_export_video_abspath: str | None = None

settings_state = {
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES,
    "use_stabilization": False,
    "tracker_type": "bytetrack",
    "show_trace": False,
    "camera_mode": "zed",
    "opencv_source": 0,
}


def _recording_status_snapshot() -> dict[str, Any]:
    with state_lock:
        ra = ui_recording_active
        csv_ok = bool(last_export_csv_rows)
        vid_ok = False
        if last_export_video_relpath:
            vr = last_export_video_relpath
            vp = Path(vr) if Path(vr).is_absolute() else (_REPO_ROOT / vr)
            try:
                vid_ok = vp.is_file() and vp.stat().st_size > 0
            except OSError:
                vid_ok = False
        export_any = csv_ok or vid_ok
        stem_s = last_export_stem or ""
        ds_dir = str(EXPORT_DATASET_DIR)
        csv_abs = last_export_csv_abspath or ""
        vid_abs = last_export_video_abspath or ""
    return {
        "recording_active": ra,
        "export_ready": export_any,
        "export_csv_ready": csv_ok,
        "export_video_ready": vid_ok,
        "export_stem": stem_s,
        "export_dataset_dir": ds_dir,
        "export_csv_path": csv_abs,
        "export_video_path": vid_abs,
    }


_CSV_EXPORT_FIELDNAMES = [
    "timestamp",
    "elapsed",
    "frame_index",
    "id",
    "ripeness",
    "score",
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


def _persist_dataset_csv(rows: list[dict], stem: str | None) -> Path | None:
    """Save detection CSV next to the recording under EXPORT_DATASET_DIR. Returns path or None."""
    if not rows:
        return None
    EXPORT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = _sanitize_export_basename(
        stem or "", fallback_stem=f"tomato_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_path = EXPORT_DATASET_DIR / f"{safe_stem}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_EXPORT_FIELDNAMES, restval="")
        w.writeheader()
        w.writerows(rows)
    return out_path


def _finalize_recording_segment() -> None:
    """Copy in-memory CSV buffer to last export and clear the active recording session."""
    global last_export_csv_rows, last_export_video_relpath, last_export_stem
    global active_recording_video_relpath, active_recording_stem, ui_recording_active, recording_csv_buffer
    global last_export_csv_abspath, last_export_video_abspath
    rows_for_disk: list[dict] = []
    stem_for_disk: str | None = None
    with state_lock:
        last_export_csv_rows = list(recording_csv_buffer)
        if active_recording_video_relpath:
            last_export_video_relpath = active_recording_video_relpath
        if active_recording_stem:
            last_export_stem = active_recording_stem
        elif active_recording_video_relpath:
            last_export_stem = Path(active_recording_video_relpath).stem
        active_recording_video_relpath = None
        active_recording_stem = None
        ui_recording_active = False
        recording_csv_buffer.clear()
        rows_for_disk = list(last_export_csv_rows)
        stem_for_disk = last_export_stem
    csv_path = _persist_dataset_csv(rows_for_disk, stem_for_disk)
    with state_lock:
        last_export_csv_abspath = str(csv_path.resolve()) if csv_path else None
        vr = last_export_video_relpath
        if vr:
            p = Path(vr)
            if not p.is_absolute():
                p = _REPO_ROOT / vr
            last_export_video_abspath = str(p.resolve())
        else:
            last_export_video_abspath = None


def format_elapsed(sec: float) -> str:
    sec = max(int(sec), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_placeholder(text: str) -> bytes:
    canvas = np.full((720, 1280, 3), 28, dtype=np.uint8)
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines:
        lines = ["Camera stopped", "Press the Start button"]
    y0 = 330
    dy = 56
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (290, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (220, 220, 220),
            3,
        )
    ok, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


latest_frame = make_placeholder("Camera stopped\nPress the Start button")
latest_status: dict = {
    "running": False,
    "connected": False,
    "total_fps": 0.0,
    "ripe_count": 0,
    "unripe_count": 0,
    "ripe_ids": [],
    "unripe_ids": [],
    "current_frame_count": 0,
    "total_count": 0,
    "elapsed": "00:00:00",
    "resolution": "-",
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES,
    "use_stabilization": bool(settings_state.get("use_stabilization", False)),
    "tracker_type": str(settings_state.get("tracker_type", "bytetrack")).lower(),
    "show_trace": bool(settings_state.get("show_trace", False)),
    "camera_mode": settings_state.get("camera_mode", "zed"),
    "camera_label": _camera_mode_label(settings_state.get("camera_mode", "zed")),
    "recording_active": False,
    "export_ready": False,
    "export_csv_ready": False,
    "export_video_ready": False,
    "export_stem": "",
    "export_dataset_dir": str(EXPORT_DATASET_DIR),
    "export_csv_path": "",
    "export_video_path": "",
}


def set_idle_state(text: str = "Camera stopped\nPress the Start button") -> None:
    global latest_frame, latest_status
    rec = _recording_status_snapshot()
    with state_lock:
        conf_thres = settings_state["conf_thres"]
        iou_thres = settings_state["iou_thres"]
        use_stabilization = bool(settings_state.get("use_stabilization", False))
        tracker_type = str(settings_state.get("tracker_type", "bytetrack")).lower()
        show_trace = bool(settings_state.get("show_trace", False))
        cam_mode = settings_state.get("camera_mode", "zed")

    with frame_lock:
        latest_frame = make_placeholder(text)
        latest_status = {
            "running": False,
            "connected": False,
            "total_fps": 0.0,
            "ripe_count": 0,
            "unripe_count": 0,
            "ripe_ids": [],
            "unripe_ids": [],
            "current_frame_count": 0,
            "total_count": 0,
            "elapsed": "00:00:00",
            "resolution": "-",
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "use_stabilization": use_stabilization,
            "tracker_type": tracker_type,
            "show_trace": show_trace,
            "camera_mode": cam_mode,
            "camera_label": _camera_mode_label(cam_mode),
            "recording_active": rec["recording_active"],
            "export_ready": rec["export_ready"],
            "export_csv_ready": rec["export_csv_ready"],
            "export_video_ready": rec["export_video_ready"],
            "export_stem": rec["export_stem"],
            "export_dataset_dir": rec["export_dataset_dir"],
            "export_csv_path": rec["export_csv_path"],
            "export_video_path": rec["export_video_path"],
        }


def _session_elapsed_str() -> str:
    global session_start_time
    if session_start_time is None:
        return "00:00:00"
    return format_elapsed(time.time() - session_start_time)


def _on_frame(annotated: np.ndarray, meta: dict) -> None:
    global latest_frame, latest_status, frame_index, recording_csv_buffer

    frame_index = int(meta.get("frame_idx", frame_index))

    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(annotated))
    if not ok:
        return

    elapsed = _session_elapsed_str()
    rec = _recording_status_snapshot()
    with frame_lock:
        latest_frame = buffer.tobytes()
        latest_status = {
            "running": True,
            "connected": True,
            "total_fps": round(float(meta.get("fps", 0.0)), 2),
            "ripe_count": int(meta.get("ripe_count", 0)),
            "unripe_count": int(meta.get("unripe_count", 0)),
            "ripe_ids": list(meta.get("ripe_ids", [])),
            "unripe_ids": list(meta.get("unripe_ids", [])),
            "current_frame_count": int(meta.get("current_frame_count", 0)),
            "total_count": int(meta.get("total_count", 0)),
            "elapsed": elapsed,
            "resolution": f'{int(meta.get("width", 0))} x {int(meta.get("height", 0))}',
            "conf_thres": settings_state["conf_thres"],
            "iou_thres": settings_state["iou_thres"],
            "use_stabilization": bool(settings_state.get("use_stabilization", False)),
            "tracker_type": str(settings_state.get("tracker_type", "bytetrack")).lower(),
            "show_trace": bool(settings_state.get("show_trace", False)),
            "camera_mode": settings_state.get("camera_mode", "zed"),
            "camera_label": _camera_mode_label(settings_state.get("camera_mode", "zed")),
            "recording_active": rec["recording_active"],
            "export_ready": rec["export_ready"],
            "export_csv_ready": rec["export_csv_ready"],
            "export_video_ready": rec["export_video_ready"],
            "export_stem": rec["export_stem"],
            "export_dataset_dir": rec["export_dataset_dir"],
            "export_csv_path": rec["export_csv_path"],
            "export_video_path": rec["export_video_path"],
        }

    rows = meta.get("csv_rows") or []
    with state_lock:
        if rows:
            recording_csv_buffer.extend(rows)


def _observer_worker(cfg: dict) -> None:
    stats = cfg.setdefault("_exit_stats", {})
    stats.clear()
    try:
        run_observer(cfg)
    except Exception as e:
        set_idle_state(f"Error: {e}")
    else:
        fi = int(stats.get("frame_idx", 0))
        user_stop = bool(stats.get("user_stop", False))
        if user_stop:
            set_idle_state("Camera stopped\nPress the Start button")
        elif fi == 0:
            set_idle_state("No frames (wrong camera index, device busy, or try DirectShow / resolution)")
        else:
            set_idle_state("Stream finished")


def start_camera() -> tuple[bool, str]:
    global worker_thread, session_start_time, frame_index, pipeline_runtime, stop_event, ui_recording_active
    global active_recording_video_relpath, active_recording_stem

    with control_lock:
        if worker_thread is not None and worker_thread.is_alive():
            return False, "already running"

        with state_lock:
            mode = str(settings_state.get("camera_mode", "zed")).lower().strip()
            if mode not in VALID_CAMERA_MODES:
                return False, f"camera_mode must be one of: {', '.join(sorted(VALID_CAMERA_MODES))}"
            source = int(settings_state.get("opencv_source", 0))
            conf_thres = float(settings_state["conf_thres"])
            iou_thres = float(settings_state["iou_thres"])
            use_stabilization = bool(settings_state.get("use_stabilization", False))
            tracker_type = str(settings_state.get("tracker_type", "bytetrack")).lower()
            if tracker_type not in VALID_TRACKER_TYPES:
                tracker_type = "bytetrack"
            show_trace = bool(settings_state.get("show_trace", False))

        stop_event.clear()
        session_start_time = time.time()
        frame_index = 0
        with state_lock:
            recording_csv_buffer.clear()
            ui_recording_active = False
            active_recording_video_relpath = None
            active_recording_stem = None

        pipeline_runtime = {
            "lock": threading.Lock(),
            "conf": conf_thres,
            "nms_iou": iou_thres,
            "show_trace": show_trace,
            "recording": False,
            "recording_video_relpath": None,
        }

        cfg: dict = {**DEFAULT_CONFIG}
        cfg.update(
            {
                "model_path": "yolo26n_640.pt",
                "output_path": None,
                "device": "0",
                "tracker_type": tracker_type,
                "source": source,
                "show_window": False,
                "stop_event": stop_event,
                "runtime": pipeline_runtime,
                "on_frame": _on_frame,
                "session_elapsed_fn": _session_elapsed_str,
                "conf": conf_thres,
                "nms_iou": iou_thres,
                "use_stabilization": use_stabilization,
                "show_trace": show_trace,
                "motion_compensation": True,
            }
        )
        cfg.update(_pipeline_camera_overrides(mode))
        start_message = "started"

        # Fail fast for ZED (or fallback) so API does not report started with zero frames.
        if str(cfg.get("camera_backend", "opencv")).lower() == "zed":
            try:
                zed_probe = _top._ZedCapture(cfg)
                zed_probe.release()
            except Exception as e:
                fallback_source: int | None = None
                for candidate in (2, 0):
                    try:
                        probe_cfg = {**cfg}
                        probe_cfg.update(
                            {
                                "camera_backend": "opencv",
                                "source": candidate,
                                "stereo_sbs": False,
                                "left_only": False,
                                "camera_width": None,
                                "camera_height": None,
                                "opencv_use_mjpeg": False,
                                "opencv_api": "default",
                            }
                        )
                        cap_probe = _top._open_source(probe_cfg)
                        _top._apply_capture_size(cap_probe, probe_cfg)
                        cap_probe.release()
                        cfg.update(probe_cfg)
                        fallback_source = candidate
                        break
                    except Exception:
                        continue

                if fallback_source is None:
                    return False, f"ZED 열기 실패: {e}"

                with state_lock:
                    settings_state["camera_mode"] = "webcam"
                    settings_state["opencv_source"] = fallback_source
                start_message = f"ZED 실패로 webcam({fallback_source}) fallback"

        cfg["yolo_model"] = ensure_detector_ready(str(cfg["model_path"]), str(cfg["device"]))

        if str(cfg.get("camera_backend", "opencv")).lower() == "opencv":
            try:
                cap = _top._open_source(cfg)
                _top._apply_capture_size(cap, cfg)
                cfg["video_capture"] = cap
            except Exception as e:
                return False, f"카메라를 열 수 없습니다: {e}"

        cfg["_exit_stats"] = {}

        worker_thread = threading.Thread(target=_observer_worker, args=(cfg,), daemon=True)
        worker_thread.start()
        return True, start_message


def stop_camera() -> tuple[bool, str]:
    global worker_thread, session_start_time, pipeline_runtime

    with control_lock:
        stop_event.set()
        pr = pipeline_runtime
        thread_ref = worker_thread

    if pr is not None:
        lk = pr.get("lock")
        if lk is not None:
            with lk:
                pr["recording"] = False
                pr["recording_video_relpath"] = None
        else:
            pr["recording"] = False
            pr["recording_video_relpath"] = None

    if thread_ref is not None:
        thread_ref.join(timeout=5.0)

    time.sleep(0.08)

    with state_lock:
        need_fin = ui_recording_active or (active_recording_video_relpath is not None)
    if need_fin:
        _finalize_recording_segment()

    with control_lock:
        worker_thread = None
        session_start_time = None
        pipeline_runtime = None

    set_idle_state("Camera stopped\nPress the Start button")
    return True, "stopped"


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("camera_mode")
    legacy = data.get("camera_backend")
    src = data.get("opencv_source")

    with state_lock:
        if mode is not None:
            m = str(mode).lower().strip()
            if m not in VALID_CAMERA_MODES:
                return jsonify({"ok": False, "message": f"camera_mode: one of {sorted(VALID_CAMERA_MODES)}"}), 400
            settings_state["camera_mode"] = m
        elif legacy is not None:
            b = str(legacy).lower().strip()
            settings_state["camera_mode"] = "zed" if b == "zed" else "webcam"
        if src is not None:
            try:
                settings_state["opencv_source"] = int(src)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "message": "opencv_source must be an integer"}), 400

    ok, message = start_camera()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok, message = stop_camera()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with control_lock:
        running = worker_thread is not None and worker_thread.is_alive()

    if running:
        stop_camera()
        ok, message = start_camera()
        if not ok:
            return jsonify({"ok": False, "message": f"리셋 실패: {message}"}), 500
        return jsonify({"ok": True, "message": "ID 누적값이 초기화되었습니다. (세션 재시작)"})

    with state_lock:
        recording_csv_buffer.clear()
    set_idle_state("ID counters reset")
    return jsonify({"ok": True, "message": "ID 누적값이 초기화되었습니다."})


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    global active_recording_video_relpath, active_recording_stem, ui_recording_active

    with control_lock:
        running = worker_thread is not None and worker_thread.is_alive()
    if not running:
        return jsonify({"ok": False, "message": "카메라를 먼저 시작해 주세요."}), 400

    pr = pipeline_runtime
    if pr is None:
        return jsonify({"ok": False, "message": "파이프라인이 준비되지 않았습니다."}), 500
    lk = pr.get("lock")
    if lk is None:
        return jsonify({"ok": False, "message": "내부 오류: runtime lock 없음."}), 500

    data = request.get_json(silent=True) or {}
    if not data.get("basename") and not data.get("filename") and request.form:
        data = {**data, **request.form.to_dict(flat=True)}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback = f"rec_{ts}"
    stem = _sanitize_export_basename(data.get("basename", data.get("filename", "")), fallback_stem=fallback)
    relpath = str(EXPORT_DATASET_DIR / f"{stem}.mp4")

    with lk:
        if pr.get("recording"):
            return jsonify({"ok": False, "message": "이미 녹화 중입니다."}), 400

    with state_lock:
        recording_csv_buffer.clear()
        active_recording_video_relpath = relpath
        active_recording_stem = stem
        ui_recording_active = True

    with lk:
        pr["recording_video_relpath"] = relpath
        pr["recording"] = True

    rec = _recording_status_snapshot()
    with frame_lock:
        latest_status["recording_active"] = rec["recording_active"]
        latest_status["export_ready"] = rec["export_ready"]
        latest_status["export_csv_ready"] = rec["export_csv_ready"]
        latest_status["export_video_ready"] = rec["export_video_ready"]
        latest_status["export_stem"] = rec["export_stem"]
        latest_status["export_dataset_dir"] = rec["export_dataset_dir"]
        latest_status["export_csv_path"] = rec["export_csv_path"]
        latest_status["export_video_path"] = rec["export_video_path"]

    return jsonify(
        {
            "ok": True,
            "message": f"녹화를 시작했습니다. (파일 베이스 이름: {stem})",
            "stem": stem,
            "video_path": relpath,
        }
    )


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    pr = pipeline_runtime
    if pr is None:
        return jsonify({"ok": False, "message": "카메라가 실행 중이 아닙니다."}), 400
    lk = pr.get("lock")
    if lk is None:
        return jsonify({"ok": False, "message": "내부 오류: runtime lock 없음."}), 500

    with lk:
        if not pr.get("recording"):
            return jsonify({"ok": False, "message": "녹화 중이 아닙니다."}), 400
        pr["recording"] = False
        pr["recording_video_relpath"] = None

    time.sleep(0.08)
    _finalize_recording_segment()

    rec = _recording_status_snapshot()
    with frame_lock:
        latest_status["recording_active"] = rec["recording_active"]
        latest_status["export_ready"] = rec["export_ready"]
        latest_status["export_csv_ready"] = rec["export_csv_ready"]
        latest_status["export_video_ready"] = rec["export_video_ready"]
        latest_status["export_stem"] = rec["export_stem"]
        latest_status["export_dataset_dir"] = rec["export_dataset_dir"]
        latest_status["export_csv_path"] = rec["export_csv_path"]
        latest_status["export_video_path"] = rec["export_video_path"]

    return jsonify(
        {
            "ok": True,
            "message": "녹화를 종료했습니다. CSV·MP4는 서버 폴더에 저장되었습니다. (브라우저 다운로드 아님)",
            "saved_csv_path": rec["export_csv_path"],
            "saved_video_path": rec["export_video_path"],
            "export_dataset_dir": rec["export_dataset_dir"],
        }
    )


@app.route("/api/status")
def api_status():
    with frame_lock:
        return jsonify(dict(latest_status))


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global pipeline_runtime

    data = request.get_json(silent=True) or {}
    try:
        conf_thres = float(data.get("conf_thres", DEFAULT_CONF_THRES))
        iou_thres = float(data.get("iou_thres", DEFAULT_IOU_THRES))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "숫자를 입력해주세요."}), 400

    if not (0 <= conf_thres <= 1):
        return jsonify({"ok": False, "message": "Confidence는 0~1 사이여야 합니다."}), 400
    if not (0 <= iou_thres <= 1):
        return jsonify({"ok": False, "message": "IOU는 0~1 사이여야 합니다."}), 400

    raw_stab = data.get("use_stabilization", settings_state.get("use_stabilization", False))
    if isinstance(raw_stab, bool):
        use_stabilization = raw_stab
    else:
        use_stabilization = str(raw_stab).strip().lower() in {"1", "true", "on", "yes"}

    raw_tracker = data.get("tracker_type", settings_state.get("tracker_type", "bytetrack"))
    tracker_type = str(raw_tracker).lower().strip()
    if tracker_type not in VALID_TRACKER_TYPES:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": f"tracker_type은 다음 중 하나여야 합니다: {', '.join(sorted(VALID_TRACKER_TYPES))}",
                }
            ),
            400,
        )

    raw_trace = data.get("show_trace", settings_state.get("show_trace", False))
    if isinstance(raw_trace, bool):
        show_trace = raw_trace
    else:
        show_trace = str(raw_trace).strip().lower() in {"1", "true", "on", "yes"}

    with state_lock:
        prev_use_stabilization = bool(settings_state.get("use_stabilization", False))
        prev_tracker_type = str(settings_state.get("tracker_type", "bytetrack")).lower()
        settings_state["conf_thres"] = round(conf_thres, 2)
        settings_state["iou_thres"] = round(iou_thres, 2)
        settings_state["use_stabilization"] = bool(use_stabilization)
        settings_state["tracker_type"] = tracker_type
        settings_state["show_trace"] = bool(show_trace)

    pr = pipeline_runtime
    if pr is not None:
        lock = pr.get("lock")
        if lock is not None:
            with lock:
                pr["conf"] = float(settings_state["conf_thres"])
                pr["nms_iou"] = float(settings_state["iou_thres"])
                pr["show_trace"] = bool(show_trace)
        else:
            pr["conf"] = float(settings_state["conf_thres"])
            pr["nms_iou"] = float(settings_state["iou_thres"])
            pr["show_trace"] = bool(show_trace)

    rec = _recording_status_snapshot()
    with frame_lock:
        latest_status["conf_thres"] = round(conf_thres, 2)
        latest_status["iou_thres"] = round(iou_thres, 2)
        latest_status["use_stabilization"] = bool(use_stabilization)
        latest_status["tracker_type"] = tracker_type
        latest_status["show_trace"] = bool(show_trace)
        latest_status["recording_active"] = rec["recording_active"]
        latest_status["export_ready"] = rec["export_ready"]
        latest_status["export_csv_ready"] = rec["export_csv_ready"]
        latest_status["export_video_ready"] = rec["export_video_ready"]
        latest_status["export_stem"] = rec["export_stem"]
        latest_status["export_dataset_dir"] = rec["export_dataset_dir"]
        latest_status["export_csv_path"] = rec["export_csv_path"]
        latest_status["export_video_path"] = rec["export_video_path"]

    need_restart = prev_use_stabilization != bool(use_stabilization) or prev_tracker_type != tracker_type
    restart_applied = False
    if need_restart:
        with control_lock:
            running = worker_thread is not None and worker_thread.is_alive()
        if running:
            stop_camera()
            ok, message = start_camera()
            if not ok:
                return jsonify({"ok": False, "message": f"설정 적용 재시작 실패: {message}"}), 500
            restart_applied = True

    restart_message = "설정 저장됨"
    if restart_applied:
        parts: list[str] = []
        if prev_use_stabilization != bool(use_stabilization):
            parts.append("Stabilization")
        if prev_tracker_type != tracker_type:
            parts.append("트래커")
        restart_message = f"{' · '.join(parts)} 변경 반영을 위해 재시작됨"

    return jsonify(
        {
            "ok": True,
            "conf_thres": round(conf_thres, 2),
            "iou_thres": round(iou_thres, 2),
            "use_stabilization": bool(use_stabilization),
            "tracker_type": tracker_type,
            "show_trace": bool(show_trace),
            "message": restart_message,
        }
    )


@app.route("/download_csv")
def download_csv():
    with state_lock:
        rows = list(last_export_csv_rows)
        stem = last_export_stem

    if not rows:
        return jsonify({"ok": False, "message": "CSV에 저장된 행이 없습니다. 녹화 시작 후 녹화 종료를 눌러 주세요."}), 404

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_EXPORT_FIELDNAMES, restval="")
    writer.writeheader()
    if rows:
        writer.writerows(rows)

    csv_text = output.getvalue()
    safe_stem = _sanitize_export_basename(stem or "", fallback_stem=f"tomato_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    filename = f"{safe_stem}.csv"

    return Response(
        csv_text.encode("utf-8-sig"),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": _content_disposition_attachment(filename)},
    )


@app.route("/download_video")
def download_video():
    with state_lock:
        rel = last_export_video_relpath
        stem = last_export_stem
    if not rel:
        return jsonify({"ok": False, "message": "저장된 영상 경로가 없습니다. 녹화를 시작했다가 종료해 주세요."}), 404

    video_path = Path(rel) if Path(rel).is_absolute() else (_REPO_ROOT / rel)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return jsonify({"ok": False, "message": "다운로드할 영상 파일이 없거나 비어 있습니다."}), 404

    safe_stem = _sanitize_export_basename(
        stem or Path(rel).stem, fallback_stem=f"tomato_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    filename = f"{safe_stem}.mp4"
    return send_file(
        str(video_path),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=filename,
    )


def _maybe_open_browser(port: int) -> None:
    if os.environ.get("TOMATO_NO_BROWSER", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    url = f"http://127.0.0.1:{port}/"

    def _open() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    _dev = str(DEFAULT_CONFIG["device"])
    _mp = str(DEFAULT_CONFIG["model_path"])
    _port_raw = (os.environ.get("PORT") or os.environ.get("TOMATO_PORT") or "5000").strip()
    try:
        _listen_port = max(1, min(65535, int(_port_raw)))
    except ValueError:
        _listen_port = 5000
        print(f"[tomato_observer_app] 잘못된 PORT 값 '{_port_raw}', 5000 사용.", flush=True)

    print("[tomato_observer_app] detector 로드 중(최초 1회)…", flush=True)
    _m = ensure_detector_ready(_mp, _dev)
    _prime_yolo_warmup(_m, _dev)
    print("[tomato_observer_app] detector 준비됨. 서버 시작…", flush=True)
    set_idle_state("Camera stopped\nPress the Start button")
    print("", flush=True)
    print(f"[tomato_observer_app] 웹 UI 주소 (이 컴퓨터에서 브라우저): http://127.0.0.1:{_listen_port}/", flush=True)
    print(f"[tomato_observer_app] 같은 Wi‑Fi의 다른 기기: http://<이_PC의_LAN_IP>:{_listen_port}/", flush=True)
    print(
        "[tomato_observer_app] Cursor/SSH로 원격 폴더를 열었다면: PC 브라우저의 localhost는 서버가 아닙니다. "
        "Cursor 'Ports' 탭에서 포트를 Forward 하거나, ssh -L "
        f"{_listen_port}:127.0.0.1:{_listen_port} user@서버 로 터널을 만드세요.",
        flush=True,
    )
    print(
        "[tomato_observer_app] 자동으로 브라우저를 엽니다. 끄려면 실행 전에 TOMATO_NO_BROWSER=1 을 설정하세요.",
        flush=True,
    )
    print("[tomato_observer_app] 서버 종료: 이 터미널에서 Ctrl+C", flush=True)
    print("", flush=True)
    _maybe_open_browser(_listen_port)
    try:
        app.run(host="0.0.0.0", port=_listen_port, threaded=True, use_reloader=False, debug=False)
    except OSError as e:
        print(f"[tomato_observer_app] 포트 {_listen_port} 에서 서버를 열 수 없습니다: {e}", flush=True)
        print(f"[tomato_observer_app] 다른 포트로 시도: PORT=5050 python3 app.py", flush=True)
        raise
