"""
R-07: 코디 추천 — Recommender.recommend_outfit()가 코디 세트 1개를 반환
      (outfits=[{tops, bottoms, shoes}], 각 슬롯은 product_id/name/url/image_path/qr_b64).
"""
import json
import os
import sys
import unittest.mock as mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SNAP_OUTFITS = {
    "snap1": {"tops": ["musinsa_001"], "bottoms": ["musinsa_002"], "shoes": ["musinsa_003"]},
    "snap2": {"tops": ["musinsa_004"], "bottoms": ["musinsa_005"], "shoes": ["musinsa_006"]},
    "snap3": {"tops": ["musinsa_007"], "bottoms": ["musinsa_008"], "shoes": ["musinsa_009"]},
}
META = {
    "musinsa_001": {"product_id": "musinsa_001", "name": "셔츠", "url": "https://musinsa.com/1", "image_path": "tops/musinsa_001.jpg"},
    "musinsa_002": {"product_id": "musinsa_002", "name": "청바지", "url": "https://musinsa.com/2", "image_path": "bottoms/musinsa_002.jpg"},
    "musinsa_003": {"product_id": "musinsa_003", "name": "스니커즈", "url": "https://musinsa.com/3", "image_path": "shoes/musinsa_003.jpg"},
    "musinsa_004": {"product_id": "musinsa_004", "name": "니트", "url": "https://musinsa.com/4", "image_path": "tops/musinsa_004.jpg"},
    "musinsa_005": {"product_id": "musinsa_005", "name": "슬랙스", "url": "https://musinsa.com/5", "image_path": "bottoms/musinsa_005.jpg"},
    "musinsa_006": {"product_id": "musinsa_006", "name": "로퍼", "url": "https://musinsa.com/6", "image_path": "shoes/musinsa_006.jpg"},
    "musinsa_007": {"product_id": "musinsa_007", "name": "후드티", "url": "https://musinsa.com/7", "image_path": "tops/musinsa_007.jpg"},
    "musinsa_008": {"product_id": "musinsa_008", "name": "조거팬츠", "url": "https://musinsa.com/8", "image_path": "bottoms/musinsa_008.jpg"},
    "musinsa_009": {"product_id": "musinsa_009", "name": "운동화", "url": "https://musinsa.com/9", "image_path": "shoes/musinsa_009.jpg"},
}
CANDIDATES = [
    {**META["musinsa_002"], "category": "bottoms", "snap_id": "snap1", "score": 0.95},
    {**META["musinsa_005"], "category": "bottoms", "snap_id": "snap2", "score": 0.88},
    {**META["musinsa_008"], "category": "bottoms", "snap_id": "snap3", "score": 0.80},
]
DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def rec():
    from recommender import Recommender
    r = object.__new__(Recommender)
    r._snap_outfits = SNAP_OUTFITS
    r.detector = mock.MagicMock()
    r.detector.detect.return_value = {
        "annotated": DUMMY_FRAME.copy(),
        "crops": {"bottoms": np.zeros((100, 100, 3), dtype=np.uint8)},
        "persons": [True],
    }
    r.embedder = mock.MagicMock()
    r.embedder.embed.return_value = np.zeros(512, dtype=np.float32)
    r.searcher = mock.MagicMock()
    r.searcher.search.return_value = CANDIDATES
    r.searcher._meta = META
    r.reranker = mock.MagicMock()
    r.reranker.extract_palette.return_value = ["#3D6B9F", "#FFFFFF", "#000000"]
    # rerank returns the top_n candidates so each outfit slot is actually populated.
    r.reranker.rerank.side_effect = (
        lambda candidates, text_vec, palette, top_n=1: candidates[:top_n]
    )
    r._make_qr = lambda url: "fake_qr"
    return r


def test_recommend_outfit_returns_outfits_key(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert "outfits" in result


def test_recommend_outfit_returns_one_set(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert len(result["outfits"]) == 1


def test_recommend_outfit_set_structure(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for outfit in result["outfits"]:
        assert "tops" in outfit
        assert "bottoms" in outfit
        assert "shoes" in outfit


def test_recommend_outfit_products_have_required_fields(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for outfit in result["outfits"]:
        for slot in ("tops", "bottoms", "shoes"):
            for product in outfit[slot]:
                assert "product_id" in product
                assert "name" in product
                assert "url" in product
                assert "image_path" in product
                assert "qr_b64" in product


def test_recommend_outfit_one_product_per_slot(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    outfit = result["outfits"][0]
    for slot in ("tops", "bottoms", "shoes"):
        assert len(outfit[slot]) == 1


def test_recommend_outfit_fallback_when_no_crop(rec):
    rec.detector.detect.return_value = {
        "annotated": DUMMY_FRAME.copy(),
        "crops": {},
        "persons": [],
    }
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert "outfits" in result
    rec.embedder.embed.assert_called_once_with(DUMMY_FRAME)
