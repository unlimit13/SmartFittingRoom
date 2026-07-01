"""
FR-06: ko-sroberta 한국어 텍스트 임베딩 — 768-dim, L2 정규화
"""
import numpy as np
import pytest
from conftest import KO_READY


@pytest.mark.skipif(not KO_READY, reason="ko-sroberta model not available")
def test_encode_output_dim():
    from text_encoder import TextEncoder
    enc = TextEncoder()
    vec = enc.encode("맑은 날 여자친구와 데이트하려고 해. 추천해줘.")
    assert vec.shape == (768,)


@pytest.mark.skipif(not KO_READY, reason="ko-sroberta model not available")
def test_encode_l2_normalized():
    from text_encoder import TextEncoder
    vec = TextEncoder().encode("캐주얼 데이트룩")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"


@pytest.mark.skipif(not KO_READY, reason="ko-sroberta model not available")
def test_encode_different_texts_different_vecs():
    from text_encoder import TextEncoder
    enc = TextEncoder()
    v1 = enc.encode("여름 캐주얼 룩")
    v2 = enc.encode("겨울 포멀 정장")
    cosine = float(np.dot(v1, v2))
    # Different style descriptions should not be identical
    assert cosine < 0.99


@pytest.mark.skipif(not KO_READY, reason="ko-sroberta model not available")
def test_encode_similar_texts_high_cosine():
    from text_encoder import TextEncoder
    enc = TextEncoder()
    v1 = enc.encode("캐주얼 데이트룩 추천")
    v2 = enc.encode("캐주얼한 데이트 스타일")
    cosine = float(np.dot(v1, v2))
    assert cosine > 0.7, f"Similar texts should have high cosine: {cosine}"
