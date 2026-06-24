import base64
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

sys.path.insert(0, os.path.dirname(__file__))

from camera import Camera
from recommender import Recommender
from pose import PoseTracker
import tryon as tryon_mod

app = Flask(__name__)

class _NoPosePoll(logging.Filter):
    def filter(self, record):
        return 'pose_poll' not in record.getMessage()

logging.getLogger('werkzeug').addFilter(_NoPosePoll())

_camera          = Camera()
_recommender     = Recommender()
_pose_tracker    = PoseTracker()
_upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_person_url_future: concurrent.futures.Future | None = None
_tops_result_url: str | None = None   # reused as person image for bottoms try-on

_pose_lock      = threading.Lock()
_pose_state     = {"in_zone": False, "hold_pct": 0.0, "triggered": False, "disabled": False, "rw": None}
_pose_triggered = False   # consumed once by /pose_poll


def _pose_loop():
    global _pose_triggered
    while True:
        try:
            time.sleep(0.2)
            frame = _camera.get_frame()
            if frame is None:
                continue
            state = _pose_tracker.process(frame)
            with _pose_lock:
                _pose_state["in_zone"]   = state["in_zone"]
                _pose_state["hold_pct"]  = state["hold_pct"]
                _pose_state["triggered"] = state["triggered"]
                _pose_state["disabled"]  = state["disabled"]
                _pose_state["rw"]        = state["rw"]
                if state["triggered"]:
                    _pose_triggered = True
        except Exception as e:
            print(f"[pose] error: {e}", flush=True)
            time.sleep(0.1)


_pose_thread_started = False

@app.before_request
def _startup():
    global _pose_thread_started
    if not _camera._running:
        _camera.start()
    if not _pose_thread_started:
        _pose_thread_started = True
        threading.Thread(target=_pose_loop, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


_STREAM_INTERVAL = 1 / 25

def _gen_detection_stream():
    while True:
        t0 = time.time()
        frame = _camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        with _pose_lock:
            p_state = dict(_pose_state)
        _pose_tracker.draw_overlay(frame, p_state)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        leftover = _STREAM_INTERVAL - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)


@app.route("/detection_feed")
def detection_feed():
    return Response(
        _gen_detection_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/pose_poll")
def pose_poll():
    global _pose_triggered
    with _pose_lock:
        state     = dict(_pose_state)
        triggered = _pose_triggered
        _pose_triggered = False
    return jsonify({
        "in_zone":   state["in_zone"],
        "hold_pct":  round(state["hold_pct"], 2),
        "triggered": triggered,
        "disabled":  state["disabled"],
        "rw":        state.get("rw"),
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    body       = request.get_json(silent=True) or {}
    text_query = body.get("text_query", "")
    use_camera = body.get("use_camera", True)

    if use_camera:
        frame = _camera.get_frame()
        if frame is None:
            return jsonify({"error": "카메라 프레임을 읽을 수 없습니다."}), 503
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    global _person_url_future, _tops_result_url
    _tops_result_url = None
    _person_url_future = _upload_executor.submit(tryon_mod.upload_frame, frame.copy())

    t0     = time.time()
    result = _recommender.recommend(frame, text_query=text_query)
    elapsed_ms = int((time.time() - t0) * 1000)

    _, buf = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
    annotated_b64 = base64.b64encode(buf).decode()

    return jsonify({
        "detected":     result["detected"],
        "palette":      result["palette"],
        "results": [
            {
                "product_id":  r["product_id"],
                "category":    r["category"],
                "name":        r.get("name", ""),
                "url":         r["url"],
                "image_path":  r.get("image_path", ""),
                "final_score": round(r["final_score"], 4),
                "qr_b64":      r["qr_b64"],
            }
            for r in result["results"]
        ],
        "annotated_b64": annotated_b64,
        "elapsed_ms":    elapsed_ms,
    })


@app.route("/tryon", methods=["POST"])
def tryon():
    body        = request.get_json(silent=True) or {}
    top_path    = body.get("top_image_path") or None
    bottom_path = body.get("bottom_image_path") or None

    if not top_path and not bottom_path:
        return jsonify({"error": "top_image_path 또는 bottom_image_path 중 하나는 필요합니다."}), 400
    if _person_url_future is None:
        return jsonify({"error": "먼저 추천 버튼을 눌러 인물을 캡처해주세요."}), 400

    def _sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        global _tops_result_url
        # 하의만 요청 + 이전 상의 결과 있으면 → 상의 결과를 person으로 사용
        if not top_path and bottom_path and _tops_result_url:
            person_url = _tops_result_url
        else:
            try:
                person_url = _person_url_future.result(timeout=30)
            except Exception as e:
                yield _sse({"error": str(e)})
                return
        try:
            for step, url in tryon_mod.run_tryon_stream(person_url, top_path, bottom_path):
                if step == "tops":
                    _tops_result_url = url  # 다음 하의 try-on에서 재사용
                b64, mime = tryon_mod.fetch_b64(url)
                yield _sse({"step": step, "result_b64": b64, "mime": mime})
        except Exception as e:
            yield _sse({"error": str(e)})
            return
        yield _sse({"step": "done"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


DB_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "musinsa_db")


@app.route("/product_image/<path:rel_path>")
def product_image(rel_path):
    full    = os.path.realpath(os.path.join(DB_IMAGE_DIR, rel_path))
    allowed = os.path.realpath(DB_IMAGE_DIR)
    if not full.startswith(allowed):
        return "", 403
    return send_file(full)


@app.route("/pose_reset", methods=["POST"])
def pose_reset():
    global _pose_triggered
    _pose_tracker.reset()
    with _pose_lock:
        _pose_triggered = False
    return jsonify({"ok": True})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    _camera.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
