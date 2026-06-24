"""
FAISS-based visual similarity search.
Searches CLIP image embedding index, optionally filtered by category.
"""
import json
import os

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(ROOT, "data", "faiss_index")
META_PATH = os.path.join(ROOT, "data", "musinsa_db", "metadata.json")


class Searcher:
    def __init__(self):
        self._index = faiss.read_index(os.path.join(INDEX_DIR, "index.bin"))

        with open(os.path.join(INDEX_DIR, "id_map.json"), "r", encoding="utf-8") as f:
            self._id_map = json.load(f)  # list of product_ids in index order

        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self._meta = {item["product_id"]: item for item in metadata}

    def search(self, query_vec: np.ndarray, category: str | None = None, top_k: int = 50) -> list[dict]:
        """
        Returns list of up to top_k dicts:
          {product_id, category, score (cosine similarity), url, image_path, name, style_text, dominant_color}
        """
        q = query_vec.astype(np.float32).reshape(1, -1)
        k = min(top_k * 3, self._index.ntotal)  # over-fetch for category filtering
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            pid = self._id_map[idx]
            item = self._meta.get(pid)
            if item is None:
                continue
            if category and item["category"] != category:
                continue
            results.append({**item, "score": float(score)})
            if len(results) >= top_k:
                break

        return results
