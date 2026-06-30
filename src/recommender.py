"""
Integrates detector → embedder → searcher into snap-based outfit recommendation.
"""
import base64
import io
import json
import os

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

    def recommend_outfit(self, frame, anchor_category: str, text_query: str = "", gender: str = "") -> dict:
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

        def get_items(category, n=1):
            candidates = self.searcher.search(img_vec, category=category, gender=gender or None, top_k=20)
            ranked = self.reranker.rerank(candidates, text_vec, palette, top_n=n)
            return [
                {
                    "product_id": r["product_id"],
                    "name":       r.get("name", ""),
                    "url":        r.get("url", ""),
                    "image_path": r.get("image_path", ""),
                    "qr_b64":    self._make_qr(r.get("url", "")),
                }
                for r in ranked
            ]

        if anchor_category == "bottoms":
            anchor_items = get_items("bottoms")
            snap = self._find_snap([i["product_id"] for i in anchor_items], gender)
            if snap:
                tops_items  = self._resolve_products(snap.get("tops",  []))
                shoes_items = self._resolve_products(snap.get("shoes", []))
            else:
                tops_items  = get_items("tops")
                shoes_items = get_items("shoes")
            outfit_bottoms = []
        else:
            anchor_items = get_items("tops")
            tops_items   = anchor_items
            snap = self._find_snap([i["product_id"] for i in anchor_items], gender)
            if snap:
                outfit_bottoms = self._resolve_products(snap.get("bottoms", []))
                shoes_items    = self._resolve_products(snap.get("shoes",   []))
            else:
                outfit_bottoms = get_items("bottoms")
                shoes_items    = get_items("shoes")

        outfit = {
            "tops":    tops_items,
            "bottoms": outfit_bottoms,
            "shoes":   shoes_items,
        }

        return {
            "detected": persons_found,
            "annotated_frame": annotated,
            "palette": palette,
            "outfits": [outfit],
            "tops_crop": crops.get("tops"),
            "bottoms_crop": crops.get("bottoms") if anchor_category == "bottoms" else None,
        }
