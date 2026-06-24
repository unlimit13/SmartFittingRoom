"""
ko-sroberta-multitask Korean text encoder (ONNX).
Outputs L2-normalized 768-dim sentence embeddings.

ONNX model outputs token-level hidden states; mean pooling + L2 norm
is applied here to produce sentence embeddings.
"""
import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
KO_MODEL_DIR = os.path.join(MODELS_DIR, "ko_sroberta")


def _find_onnx(model_dir):
    for candidate in ["model.onnx", os.path.join("onnx", "model.onnx")]:
        path = os.path.join(model_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"ko-sroberta ONNX not found in {model_dir}")


class TextEncoder:
    def __init__(self, model_dir=KO_MODEL_DIR):
        onnx_path = _find_onnx(model_dir)
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        count = mask.sum(axis=1).clip(min=1e-9)
        return summed / count

    def encode(self, text: str) -> np.ndarray:
        """Returns L2-normalized 768-dim sentence embedding."""
        enc = self._tokenizer(
            text, return_tensors="np", padding=True,
            truncation=True, max_length=128
        )
        outputs = self._session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )
        pooled = self._mean_pool(outputs[0], enc["attention_mask"])  # (1, 768)
        vec = pooled[0]
        return vec / (np.linalg.norm(vec) + 1e-8)
