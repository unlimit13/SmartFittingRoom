import base64
import os
import sys
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

sys.path.insert(0, os.path.dirname(__file__))

from camera import Camera
from recommender import Recommender
import tryon as tryon_mod

app = Flask(__name__)

_camera = Camera()
_recommender = Recommender()


@app.before_request
def _startup():
    # Start camera background capture thread once
    if not _camera._running:
        _camera.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _camera.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _gen_detection_stream():
    while True:
        frame = _camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        result = _recommender.detector.detect(frame)
        annotated = result["annotated"]
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/detection_feed")
def detection_feed():
    return Response(
        _gen_detection_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    text_query = body.get("text_query", "")
    use_camera = body.get("use_camera", True)

    if use_camera:
        frame = _camera.get_frame()
        if frame is None:
            return jsonify({"error": "카메라 프레임을 읽을 수 없습니다."}), 503
    else:
        # Blank frame for text-only mode
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    result = _recommender.recommend(frame, text_query=text_query)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Encode annotated frame as base64 JPEG for UI overlay
    _, buf = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
    annotated_b64 = base64.b64encode(buf).decode()

    return jsonify({
        "detected": result["detected"],
        "palette": result["palette"],
        "results": [
            {
                "product_id": r["product_id"],
                "category": r["category"],
                "name": r.get("name", ""),
                "url": r["url"],
                "image_path": r.get("image_path", ""),
                "final_score": round(r["final_score"], 4),
                "qr_b64": r["qr_b64"],
            }
            for r in result["results"]
        ],
        "annotated_b64": annotated_b64,
        "elapsed_ms": elapsed_ms,
    })


@app.route("/tryon", methods=["POST"])
def tryon():
    body = request.get_json(silent=True) or {}
    top_path    = body.get("top_image_path") or None
    bottom_path = body.get("bottom_image_path") or None

    if not top_path and not bottom_path:
        return jsonify({"error": "top_image_path 또는 bottom_image_path 중 하나는 필요합니다."}), 400

    frame = _camera.get_frame()
    if frame is None:
        return jsonify({"error": "카메라 프레임을 읽을 수 없습니다."}), 503

    try:
        t0 = time.time()
        person_url = tryon_mod.upload_frame(frame)
        result_url = tryon_mod.run_tryon(person_url, top_path, bottom_path)
        result_b64 = tryon_mod.fetch_b64(result_url)
        elapsed_ms = int((time.time() - t0) * 1000)
        return jsonify({"result_b64": result_b64, "elapsed_ms": elapsed_ms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


DB_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "musinsa_db")


@app.route("/product_image/<path:rel_path>")
def product_image(rel_path):
    full = os.path.realpath(os.path.join(DB_IMAGE_DIR, rel_path))
    allowed = os.path.realpath(DB_IMAGE_DIR)
    if not full.startswith(allowed):
        return "", 403
    return send_file(full)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    _camera.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
