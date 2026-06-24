"""
R-07: /recommend API — Top-3 결과 포함 JSON 반환
R-08: QR코드 생성 — base64 PNG 포함
R-09: 전체 응답 시간 ≤ 2000ms
"""
import base64
import json
import time
import unittest.mock as mock

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_mock_recommender():
    """Return a recommender that returns a deterministic fake result."""
    fake_result = {
        "detected": True,
        "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "palette": ["#3D6B9F", "#FFFFFF", "#000000"],
        "results": [
            {
                "product_id": "musinsa_0001",
                "category": "tops",
                "name": "오버핏 린넨 셔츠",
                "url": "https://www.musinsa.com/products/1",
                "image_path": "tops/musinsa_0001.jpg",
                "final_score": 0.82,
                "qr_b64": base64.b64encode(b"fakepng").decode(),
            },
            {
                "product_id": "musinsa_0002",
                "category": "bottoms",
                "name": "슬림핏 청바지",
                "url": "https://www.musinsa.com/products/2",
                "image_path": "bottoms/musinsa_0002.jpg",
                "final_score": 0.75,
                "qr_b64": base64.b64encode(b"fakepng").decode(),
            },
            {
                "product_id": "musinsa_0003",
                "category": "shoes",
                "name": "화이트 스니커즈",
                "url": "https://www.musinsa.com/products/3",
                "image_path": "shoes/musinsa_0003.jpg",
                "final_score": 0.71,
                "qr_b64": base64.b64encode(b"fakepng").decode(),
            },
        ],
    }

    rec = mock.MagicMock()
    rec.recommend.return_value = fake_result
    return rec


@pytest.fixture
def client():
    """Flask test client with mocked camera and recommender."""
    import app as app_module

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera = mock.MagicMock()
    mock_camera.get_frame.return_value = fake_frame
    mock_camera.generate_frames.return_value = iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"])
    mock_camera._running = True

    mock_rec = _make_mock_recommender()

    with mock.patch.object(app_module, "_camera", mock_camera), \
         mock.patch.object(app_module, "_recommender", mock_rec):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_index_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_recommend_returns_200(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "데이트룩 추천", "use_camera": True}),
        content_type="application/json",
    )
    assert resp.status_code == 200


def test_recommend_result_structure(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "", "use_camera": True}),
        content_type="application/json",
    )
    data = resp.get_json()
    assert "results" in data
    assert "palette" in data
    assert "detected" in data
    assert "elapsed_ms" in data


def test_recommend_top3(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "", "use_camera": True}),
        content_type="application/json",
    )
    data = resp.get_json()
    assert len(data["results"]) == 3


def test_recommend_qr_present(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "", "use_camera": True}),
        content_type="application/json",
    )
    data = resp.get_json()
    for r in data["results"]:
        assert "qr_b64" in r
        assert len(r["qr_b64"]) > 0


def test_recommend_response_time(client):
    """R-09: response time ≤ 2000ms (mock only verifies structure, not actual model latency)."""
    t0 = time.time()
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "데이트룩", "use_camera": True}),
        content_type="application/json",
    )
    elapsed_ms = (time.time() - t0) * 1000
    assert resp.status_code == 200
    # With mocks this should be well under 2000ms
    assert elapsed_ms < 2000, f"Response took {elapsed_ms:.0f}ms"
