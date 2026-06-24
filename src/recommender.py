"""
Integrates detector → embedder → searcher into snap-based outfit recommendation.
"""
import base64
import io
import json
import os

import numpy as np
import qrcode

ROOT = os.path.dirname(os.path.dirname(__file__))
SNAP_OUTFITS_PATH = os.path.join(ROOT, "data", "musinsa_db", "snap_outfits.json")


class Recommender:
    def __init__(self):
        from detector import Detector
        from embedder import Embedder
        from searcher import Searcher
        from reranker import Reranker
        from text_encoder import TextEncoder

        self.detector = Detector()
        self.embedder = Embedder()
        self.searcher = Searcher()
        self.reranker = Reranker()
        self.text_encoder = TextEncoder()
        with open(SNAP_OUTFITS_PATH, encoding="utf-8") as f:
            self._snap_outfits = json.load(f)

    def _make_qr(self, url: str) -> str:
        qr = qrcode.make(url)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _resolve_products(self, product_ids: list) -> list:
        result = []
        for pid in product_ids:
            item = self.searcher._meta.get(pid)
            if item:
                result.append({
                    "product_id": pid,
                    "name": item.get("name", ""),
                    "url": item.get("url", ""),
                    "image_path": item.get("image_path", ""),
                    "qr_b64": self._make_qr(item.get("url", "")),
                })
        return result

    def recommend_outfit(self, frame, anchor_category: str, text_query: str = "") -> dict:
        """
        anchor_category: "tops" | "bottoms"
        Returns dict with detected, annotated_frame, palette, outfits (max 3, score desc).
        """
        detection = self.detector.detect(frame)
        annotated = detection["annotated"]
        crops = detection["crops"]
        persons_found = bool(detection["persons"])

        anchor_crop = crops.get(anchor_category)
        if anchor_crop is None:
            anchor_crop = frame

        palette = self.reranker.extract_palette(anchor_crop)
        img_vec = self.embedder.embed(anchor_crop)
        candidates = self.searcher.search(img_vec, category=anchor_category, top_k=20)

        snap_scores: dict = {}
        for item in candidates:
            sid = item.get("snap_id")
            if not sid:
                continue
            if sid not in snap_scores or item["score"] > snap_scores[sid]:
                snap_scores[sid] = item["score"]

        top_snap_ids = sorted(snap_scores, key=lambda s: snap_scores[s], reverse=True)[:3]

        outfits = []
        for sid in top_snap_ids:
            slot_products = self._snap_outfits.get(sid, {})
            outfits.append({
                "snap_id": sid,
                "anchor_score": round(snap_scores[sid], 4),
                "tops":    self._resolve_products(slot_products.get("tops", [])),
                "bottoms": self._resolve_products(slot_products.get("bottoms", [])),
                "shoes":   self._resolve_products(slot_products.get("shoes", [])),
            })

        return {
            "detected": persons_found,
            "annotated_frame": annotated,
            "palette": palette,
            "outfits": outfits,
        }
