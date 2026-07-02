"""
Integrates detector → embedder → searcher into snap-based outfit recommendation.
"""
import base64
import io
import json
import os
import random

import qrcode

ROOT = os.path.dirname(os.path.dirname(__file__))
SNAP_OUTFITS_PATH = os.path.join(ROOT, "data", "musinsa_db", "snap_outfits.json")
NUM_CANDIDATES = 3


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

    def _find_snap(self, product_ids: list, gender: str = "") -> dict | None:
        """Return the first curated snap outfit whose snap_id matches a product's own snap_id field."""
        gender_prefix = "men" if "남" in gender else ("women" if "여" in gender else "")
        for pid in product_ids:
            item = self.searcher._meta.get(pid)
            snap_id = item.get("snap_id") if item else None
            if not snap_id:
                continue
            if gender_prefix and not snap_id.startswith(gender_prefix):
                continue
            outfit = self._snap_outfits.get(snap_id)
            if outfit:
                return outfit
        return None

    def _shoes_from_snap(self, product_ids: list, gender: str = "") -> list:
        """Return shoes from the curated snap outfit tied to the given product(s)' own snap_id."""
        outfit = self._find_snap(product_ids, gender)
        return self._resolve_products(outfit.get("shoes", [])) if outfit else []

    def recommend_outfit(self, frame, anchor_category: str, text_query: str = "", gender: str = "", text_priority: bool = True) -> dict:
        detection = self.detector.detect(frame)
        annotated = detection["annotated"]
        crops = detection["crops"]
        persons_found = bool(detection["persons"])

        anchor_crop = crops.get(anchor_category)
        if anchor_crop is None:
            anchor_crop = frame

        palette = self.reranker.extract_palette(anchor_crop)
        img_vec = self.embedder.embed(anchor_crop)
        text_vec = self.text_encoder.encode(text_query) if text_query.strip() else None
        if text_query in ["남", "여"]:
            text_vec = None

        def get_items(category, n=1):
            candidates = self.searcher.search(img_vec, category=category, gender=gender or None, top_k=50)
            if text_priority and text_vec is None:
                picked = random.sample(candidates, min(n, len(candidates)))
            else:
                picked = self.reranker.rerank(candidates, text_vec, palette, top_n=n, text_boost=text_priority)

            return [
                {
                    "product_id": r["product_id"],
                    "name":       r.get("name", ""),
                    "url":        r.get("url", ""),
                    "image_path": r.get("image_path", ""),
                    "qr_b64":    self._make_qr(r.get("url", "")),
                }
                for r in picked
            ]

        show_anchor  = anchor_category != "bottoms"
        anchor_field = "bottoms" if anchor_category == "bottoms" else "tops"
        other_field  = "tops" if anchor_field == "bottoms" else "bottoms"

        anchor_items = get_items(anchor_field, NUM_CANDIDATES)

        other_pool = None
        shoes_pool = None
        outfits = []
        for i, anchor_item in enumerate(anchor_items):
            snap = self._find_snap([anchor_item["product_id"]], gender)
            if snap:
                other_items = self._resolve_products(snap.get(other_field, []))
                shoes_items = self._resolve_products(snap.get("shoes", []))
            else:
                if other_pool is None:
                    other_pool = get_items(other_field, NUM_CANDIDATES)
                if shoes_pool is None:
                    shoes_pool = get_items("shoes", NUM_CANDIDATES)
                other_items = other_pool[i:i + 1]
                shoes_items = shoes_pool[i:i + 1]

            outfit = {"tops": [], "bottoms": [], "shoes": shoes_items}
            outfit[anchor_field] = [anchor_item] if show_anchor else []
            outfit[other_field]  = other_items
            outfits.append(outfit)

        return {
            "detected": persons_found,
            "annotated_frame": annotated,
            "palette": palette,
            "outfits": outfits,
            "tops_crop": crops.get("tops"),
            "bottoms_crop": crops.get("bottoms") if anchor_category == "bottoms" else None,
        }
