"""
Shared fixtures for all tests.
Tests are designed to run with or without real models/data.
"""
import os
import sys

import numpy as np
import pytest

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

MODELS_READY = (
    os.path.exists(os.path.join(MODELS_DIR, "clip_image_encoder.onnx"))
    and os.path.exists(os.path.join(MODELS_DIR, "clip_preprocessor", "visual_projection.npy"))
)
DATA_READY = os.path.exists(os.path.join(DATA_DIR, "faiss_index", "index.bin"))
KO_READY = any(
    os.path.exists(os.path.join(MODELS_DIR, "ko_sroberta", p))
    for p in ["model.onnx", os.path.join("onnx", "model.onnx")]
)


@pytest.fixture
def dummy_frame():
    """640×480 BGR dummy frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_crop():
    """200×200 BGR crop."""
    return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
