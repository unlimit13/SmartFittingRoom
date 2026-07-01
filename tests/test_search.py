"""
FR-07: 색상 팔레트 추출 — 지배색 3개 반환
FR-05: FAISS 유사 검색 — Top-50 후보, 검색 ≤ 100ms
"""
import time

import numpy as np
import pytest
from conftest import DATA_READY, MODELS_READY


def test_palette_extraction(dummy_crop):
    from reranker import _extract_palette
    palette = _extract_palette(dummy_crop, k=3)
    assert len(palette) == 3
    for color in palette:
        assert color.startswith("#")
        assert len(color) == 7


def test_palette_single_color():
    """Solid blue image should yield a palette dominated by blue."""
    from reranker import _extract_palette
    blue_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_frame[:, :] = [255, 0, 0]  # BGR blue
    palette = _extract_palette(blue_frame, k=3)
    assert len(palette) == 3


@pytest.mark.skipif(not DATA_READY, reason="FAISS index not available")
def test_search_returns_list():
    from searcher import Searcher
    s = Searcher()
    query = np.random.rand(512).astype(np.float32)
    query /= np.linalg.norm(query)
    results = s.search(query, top_k=50)
    assert isinstance(results, list)
    assert len(results) <= 50


@pytest.mark.skipif(not DATA_READY, reason="FAISS index not available")
def test_search_result_fields():
    from searcher import Searcher
    s = Searcher()
    query = np.random.rand(512).astype(np.float32)
    query /= np.linalg.norm(query)
    results = s.search(query, top_k=10)
    for r in results:
        assert "product_id" in r
        assert "category" in r
        assert "score" in r
        assert "url" in r


@pytest.mark.skipif(not DATA_READY, reason="FAISS index not available")
def test_search_latency():
    from searcher import Searcher
    s = Searcher()
    query = np.random.rand(512).astype(np.float32)
    query /= np.linalg.norm(query)
    # Warm up
    s.search(query, top_k=50)
    t0 = time.time()
    s.search(query, top_k=50)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms <= 100, f"Search took {elapsed_ms:.0f}ms > 100ms"


@pytest.mark.skipif(not DATA_READY, reason="FAISS index not available")
def test_search_category_filter():
    from searcher import Searcher
    s = Searcher()
    query = np.random.rand(512).astype(np.float32)
    query /= np.linalg.norm(query)
    for cat in ("tops", "bottoms", "shoes"):
        results = s.search(query, category=cat, top_k=10)
        for r in results:
            assert r["category"] == cat, f"Expected {cat}, got {r['category']}"


@pytest.mark.skipif(not DATA_READY, reason="FAISS index not available")
def test_search_gender_filter():
    from searcher import Searcher
    s = Searcher()
    query = np.random.rand(512).astype(np.float32)
    query /= np.linalg.norm(query)
    for gender in ("남", "여"):
        results = s.search(query, gender=gender, top_k=10)
        for r in results:
            assert r.get("gender") == gender, f"Expected {gender}, got {r.get('gender')}"
