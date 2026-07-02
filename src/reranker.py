"""
Two-stage reranker:
  1. CLIP image similarity score (from FAISS)
  2. ko-sroberta text similarity (optional)
  3. HSV color compatibility score

Final score = α×clip_sim + β×text_sim + γ×color_compat
  default (text provided):        α=0.4,  β=0.4,  γ=0.2
  fallback (no text input):       α=0.6,  β=0.0,  γ=0.4
  text_boost (text provided,
  text_priority flag enabled):    α=0.15, β=0.75, γ=0.10
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(ROOT, "data", "faiss_index")
META_PATH = os.path.join(ROOT, "data", "musinsa_db", "metadata.json")


def _hex_to_hsv(hex_color: str) -> np.ndarray:
    """Convert '#RRGGBB' → HSV (H in 0-180, S/V in 0-255) as float32."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    return hsv


def _color_compat(hsv_a: np.ndarray, hsv_b: np.ndarray) -> float:
    """HSV-based color compatibility in [0, 1]."""
    sa, va = hsv_a[1], hsv_a[2]
    sb, vb = hsv_b[1], hsv_b[2]

    # Achromatic: low saturation or very dark/light
    if sa < 30 or va < 30 or sb < 30 or vb < 30:
        return 0.8

    ha, hb = hsv_a[0] * 2.0, hsv_b[0] * 2.0  # convert to 0-360 degrees
    diff = abs(ha - hb)
    if diff > 180:
        diff = 360 - diff

    if diff <= 30:
        return 1.0   # analogous
    if 150 <= diff <= 210:
        return 0.6   # complementary
    return 0.4


def _extract_palette(frame: np.ndarray, k: int = 3) -> list[str]:
    """Extract k dominant colors as hex strings using K-means."""
    pixels = frame.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return ["#808080"] * k

    _, labels, centers = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    sorted_centers = centers[np.argsort(-counts)]
    result = []
    for c in sorted_centers:
        b, g, r = int(c[0]), int(c[1]), int(c[2])
        result.append(f"#{r:02X}{g:02X}{b:02X}")
    return result


class Reranker:
    def __init__(self):
        sv_path = os.path.join(INDEX_DIR, "style_vectors.npy")
        self._style_vectors = np.load(sv_path)  # (N, 768)

        with open(os.path.join(INDEX_DIR, "id_map.json"), "r", encoding="utf-8") as f:
            id_map = json.load(f)
        self._id_to_idx = {pid: i for i, pid in enumerate(id_map)}

    def extract_palette(self, frame: np.ndarray) -> list[str]:
        return _extract_palette(frame)

    def rerank(
        self,
        candidates: list[dict],
        text_vec: np.ndarray | None,
        color_palette: list[str],
        top_n: int = 3,
        text_boost: bool = False,
    ) -> list[dict]:
        """
        candidates: list of dicts from Searcher.search() with 'score' (clip_sim) and 'dominant_color'
        text_vec: (768,) L2-normalized vector or None
        color_palette: list of dominant hex colors from the detected clothing
        text_boost: if True and text_vec is not None, weight text_sim much more heavily
        """
        use_text = text_vec is not None
        if text_boost and use_text:
            alpha, beta, gamma = 0.3, 0.6, 0.10
        elif use_text:
            alpha, beta, gamma = 0.4, 0.4, 0.2
        else:
            alpha, beta, gamma = 0.6, 0.0, 0.4

        scored = []
        for item in candidates:
            clip_sim = item["score"]

            # Text similarity
            text_sim = 0.0
            if use_text:
                idx = self._id_to_idx.get(item["product_id"])
                if idx is not None:
                    sv = self._style_vectors[idx]
                    text_sim = float(np.dot(text_vec, sv))

            # Color compatibility: average over palette colors vs product dominant color
            try:
                prod_hsv = _hex_to_hsv(item.get("dominant_color", "#808080"))
            except Exception:
                prod_hsv = np.array([0.0, 0.0, 128.0])

            color_score = 0.0
            for hex_c in color_palette:
                try:
                    pal_hsv = _hex_to_hsv(hex_c)
                    color_score += _color_compat(pal_hsv, prod_hsv)
                except Exception:
                    color_score += 0.4
            color_score /= max(len(color_palette), 1)

            final = alpha * clip_sim + beta * text_sim + gamma * color_score
            scored.append({**item, "final_score": final, "text_sim": text_sim, "color_score": color_score})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_n]
