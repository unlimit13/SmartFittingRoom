"""
Integrates detector → embedder → searcher → reranker into a single pipeline call.
"""
import base64
import io
import os

import numpy as np
import qrcode
from PIL import Image

from detector import Detector
from embedder import Embedder
from searcher import Searcher
from reranker import Reranker
from text_encoder import TextEncoder


class Recommender:
    def __init__(self):
        self.detector = Detector()
        self.embedder = Embedder()
        self.searcher = Searcher()
        self.reranker = Reranker()
        self.text_encoder = TextEncoder()

    def _make_qr(self, url: str) -> str:
        """Return base64-encoded PNG QR code."""
        qr = qrcode.make(url)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def recommend(self, frame, text_query: str = "") -> dict:
        """
        Args:
            frame: BGR np.ndarray from camera
            text_query: optional Korean natural language string

        Returns dict:
          {
            'detected': bool,
            'annotated_frame': np.ndarray,
            'palette': list[str],         # hex colors
            'results': list[{
                product_id, category, url, name, image_path,
                final_score, qr_b64
            }]
          }
        """
        detection = self.detector.detect(frame)
        annotated = detection["annotated"]
        crops = detection["crops"]
        persons_found = bool(detection["persons"])

        # Color palette from tops crop (most visible region)
        primary_crop = next(
            (crops[c] for c in ("tops", "bottoms") if crops.get(c) is not None),
            frame,
        )
        palette = self.reranker.extract_palette(primary_crop)

        # Text embedding
        text_vec = self.text_encoder.encode(text_query) if text_query.strip() else None

        all_results = []
        for category in ("tops", "bottoms", "shoes"):
            crop = crops.get(category)
            if crop is None:
                continue
            try:
                img_vec = self.embedder.embed(crop)
                candidates = self.searcher.search(img_vec, category=category, top_k=50)
                top = self.reranker.rerank(candidates, text_vec, palette, top_n=1)
                all_results.extend(top)
            except Exception:
                continue

        # Sort all categories by final score and keep top 3
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        top3 = all_results[:3]

        for item in top3:
            item["qr_b64"] = self._make_qr(item["url"])

        return {
            "detected": persons_found,
            "annotated_frame": annotated,
            "palette": palette,
            "results": top3,
        }
