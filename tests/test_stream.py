"""
웹캠 라이브 피드 — /detection_feed HTTP 200, multipart content-type, MJPEG 프레임 청크.
(라이브 피드는 /detection_feed 로 서빙된다 — pose 오버레이가 그려진 감지 스트림.)
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

    with mock.patch.object(app_module, "_camera", mock_camera), \
         mock.patch.object(app_module, "_pose_tracker", mock.MagicMock()), \
         mock.patch.object(app_module, "_recommender", mock.MagicMock()):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c


def test_detection_feed_status_200(client):
    resp = client.get("/detection_feed")
    assert resp.status_code == 200
    resp.close()


def test_detection_feed_content_type(client):
    resp = client.get("/detection_feed")
    ct = resp.content_type
    assert "multipart/x-mixed-replace" in ct
    assert "boundary=frame" in ct
    resp.close()


def test_detection_feed_yields_frame_chunk(client):
    # The stream is an infinite generator; pull only the first chunk.
    resp = client.get("/detection_feed")
    chunk = next(resp.response)
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    resp.close()
