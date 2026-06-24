"""
R-03: CLIP 이미지 임베딩 — 512-dim 벡터, 추론 ≤ 500ms
"""
import time

import numpy as np
import pytest
from conftest import MODELS_READY


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_embed_output_dim(dummy_crop):
    from embedder import Embedder
    emb = Embedder()
    vec = emb.embed(dummy_crop)
    assert vec.shape == (512,)


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_embed_l2_normalized(dummy_crop):
    from embedder import Embedder
    vec = Embedder().embed(dummy_crop)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_embed_inference_time(dummy_crop):
    from embedder import Embedder
    emb = Embedder()
    # Warm up
    emb.embed(dummy_crop)
    t0 = time.time()
    emb.embed(dummy_crop)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms <= 500, f"Inference took {elapsed_ms:.0f}ms > 500ms"


@pytest.mark.skipif(not MODELS_READY, reason="models not available")
def test_embed_different_images_different_vecs(dummy_crop):
    from embedder import Embedder
    emb = Embedder()
    v1 = emb.embed(dummy_crop)
    other = np.zeros_like(dummy_crop)
    v2 = emb.embed(other)
    # Different images should yield different embeddings
    assert not np.allclose(v1, v2, atol=1e-3)
