"""
R-01: 웹캠 라이브 피드 — /video_feed HTTP 200, multipart content-type
"""
import numpy as np
import unittest.mock as mock
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def client():
    import app as app_module

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera = mock.MagicMock()
    mock_camera.get_frame.return_value = fake_frame
    mock_camera._running = True

    def _fake_generate():
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nFAKEJPEG\r\n"

    mock_camera.generate_frames.return_value = _fake_generate()

    mock_rec = mock.MagicMock()

    with mock.patch.object(app_module, "_camera", mock_camera), \
         mock.patch.object(app_module, "_recommender", mock_rec):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c


def test_video_feed_status_200(client):
    resp = client.get("/video_feed")
    assert resp.status_code == 200


def test_video_feed_content_type(client):
    resp = client.get("/video_feed")
    ct = resp.content_type
    assert "multipart/x-mixed-replace" in ct
    assert "boundary=frame" in ct


def test_video_feed_contains_frame_data(client):
    resp = client.get("/video_feed")
    data = resp.data
    assert b"--frame" in data
    assert b"Content-Type: image/jpeg" in data
