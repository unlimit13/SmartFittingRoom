"""
CLIP ViT-B/32 image embedder (ONNX).
Outputs L2-normalized 512-dim vectors.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CLIP_ONNX = os.path.join(MODELS_DIR, "clip_image_encoder.onnx")
PROJ_PATH = os.path.join(MODELS_DIR, "clip_preprocessor", "visual_projection.npy")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class Embedder:
    def __init__(self, model_path=CLIP_ONNX, proj_path=PROJ_PATH):
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._proj = np.load(proj_path)  # (512, 768)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - CLIP_MEAN) / CLIP_STD
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def embed(self, frame: np.ndarray) -> np.ndarray:
        """Returns L2-normalized 512-dim vector."""
        pixels = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: pixels})
        cls_token = outputs[0][:, 0, :]       # (1, 768)
        vec = cls_token @ self._proj.T         # (1, 512)
        vec = vec[0]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
