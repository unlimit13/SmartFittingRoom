"""
R-02: 의류 감지 (상/하/신발) — YOLOv8n ONNX
"""
import pytest
import numpy as np
from conftest import MODELS_READY


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_detector_returns_dict(dummy_frame):
    from detector import Detector
    det = Detector()
    result = det.detect(dummy_frame)
    assert "persons" in result
    assert "crops" in result
    assert "annotated" in result


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_detector_crops_shape(dummy_frame):
    from detector import Detector
    det = Detector()
    result = det.detect(dummy_frame)
    for key in ("tops", "bottoms", "shoes"):
        crop = result["crops"].get(key)
        if crop is not None:
            assert crop.ndim == 3
            assert crop.shape[2] == 3


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_detector_annotated_same_size(dummy_frame):
    from detector import Detector
    det = Detector()
    result = det.detect(dummy_frame)
    assert result["annotated"].shape == dummy_frame.shape


def test_detector_no_person_blank_crops():
    """With a solid-color frame, YOLO should return no/low-confidence persons."""
    if not MODELS_READY:
        pytest.skip("models not available")
    from detector import Detector
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = Detector().detect(black)
    # Either no persons or crops may be None — system should not crash
    assert isinstance(result["persons"], list)
