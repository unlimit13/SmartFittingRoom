"""
R-07: /recommend API — 최대 3개의 코디 후보 포함 JSON 반환 (tops/bottoms/shoes)
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


def _make_outfit(i):
    return {
        "tops":    [{"product_id": f"musinsa_{i}01", "name": "오버핏 린넨 셔츠", "url": f"https://www.musinsa.com/products/{i}01", "image_path": f"tops/musinsa_{i}01.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
        "bottoms": [{"product_id": f"musinsa_{i}02", "name": "슬림핏 청바지", "url": f"https://www.musinsa.com/products/{i}02", "image_path": f"bottoms/musinsa_{i}02.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
        "shoes":   [{"product_id": f"musinsa_{i}03", "name": "화이트 스니커즈", "url": f"https://www.musinsa.com/products/{i}03", "image_path": f"shoes/musinsa_{i}03.jpg", "qr_b64": base64.b64encode(b"fakepng").decode()}],
    }


def _make_mock_recommender():
    fake_outfit = {
        "detected": True,
        "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "palette": ["#3D6B9F", "#FFFFFF", "#000000"],
        "outfits": [_make_outfit(1), _make_outfit(2), _make_outfit(3)],
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


def test_brand_logo_lg_endpoint(client):
    resp = client.get("/brand_logo/lg")
    assert resp.status_code == 200
    assert resp.content_type == "image/jpeg"


def test_brand_logo_unknown_returns_404(client):
    resp = client.get("/brand_logo/unknown")
    assert resp.status_code == 404


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


def test_recommend_returns_three_outfits(client):
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
    assert call_args.args[1] == "tops"


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
