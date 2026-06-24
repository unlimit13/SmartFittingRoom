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
    fake_outfit = {
        "detected": True,
        "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "palette": ["#3D6B9F", "#FFFFFF", "#000000"],
        "outfits": [
            {
                "snap_id": "snap1", "anchor_score": 0.95,
                "tops":    [{"product_id": "musinsa_001", "name": "오버핏 린넨 셔츠", "url": "https://www.musinsa.com/products/1", "image_path": "tops/musinsa_001.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "bottoms": [{"product_id": "musinsa_002", "name": "슬림핏 청바지", "url": "https://www.musinsa.com/products/2", "image_path": "bottoms/musinsa_002.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "shoes":   [{"product_id": "musinsa_003", "name": "화이트 스니커즈", "url": "https://www.musinsa.com/products/3", "image_path": "shoes/musinsa_003.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
            },
            {
                "snap_id": "snap2", "anchor_score": 0.88,
                "tops":    [{"product_id": "musinsa_004", "name": "니트", "url": "https://www.musinsa.com/products/4", "image_path": "tops/musinsa_004.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "bottoms": [{"product_id": "musinsa_005", "name": "슬랙스", "url": "https://www.musinsa.com/products/5", "image_path": "bottoms/musinsa_005.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "shoes":   [{"product_id": "musinsa_006", "name": "로퍼", "url": "https://www.musinsa.com/products/6", "image_path": "shoes/musinsa_006.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
            },
            {
                "snap_id": "snap3", "anchor_score": 0.80,
                "tops":    [{"product_id": "musinsa_007", "name": "후드티", "url": "https://www.musinsa.com/products/7", "image_path": "tops/musinsa_007.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "bottoms": [{"product_id": "musinsa_008", "name": "조거팬츠", "url": "https://www.musinsa.com/products/8", "image_path": "bottoms/musinsa_008.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
                "shoes":   [{"product_id": "musinsa_009", "name": "운동화", "url": "https://www.musinsa.com/products/9", "image_path": "shoes/musinsa_009.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
            },
        ],
    }
    rec = mock.MagicMock()
    rec.recommend_outfit.return_value = fake_outfit
    return rec


@pytest.fixture
def client():
    """Flask test client with mocked camera and recommender."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera = mock.MagicMock()
    mock_camera.get_frame.return_value = fake_frame
    mock_camera.generate_frames.return_value = iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"])
    mock_camera._running = True

    mock_rec = _make_mock_recommender()

    # Patch Camera and Recommender constructors before app module-level code runs.
    # app.py instantiates _camera = Camera() and _recommender = Recommender() at import
    # time, so we must intercept both before import (or module reload).
    import importlib
    import sys

    # Remove cached module so we get a fresh import with our patches applied.
    sys.modules.pop("app", None)

    with mock.patch("camera.Camera", return_value=mock_camera), \
         mock.patch("recommender.Recommender", return_value=mock_rec):
        import app as app_module
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
    assert "outfits" in data
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
    assert len(data["outfits"]) == 3


def test_recommend_outfit_structure(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"text_query": "", "use_camera": True}),
        content_type="application/json",
    )
    data = resp.get_json()
    for outfit in data["outfits"]:
        assert "snap_id" in outfit
        assert "anchor_score" in outfit
        for slot in ("tops", "bottoms", "shoes"):
            assert slot in outfit
            for product in outfit[slot]:
                assert "qr_b64" in product
                assert len(product["qr_b64"]) > 0


def test_recommend_anchor_category_default(client):
    import app as app_module
    client.post(
        "/recommend",
        data=json.dumps({"text_query": "", "use_camera": True}),
        content_type="application/json",
    )
    call_args = app_module._recommender.recommend_outfit.call_args
    assert call_args.args[1] == "bottoms"


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
