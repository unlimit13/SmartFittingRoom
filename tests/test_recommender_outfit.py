"""
FR-08: 코디 추천 — Recommender.recommend_outfit()가 최대 3개의 코디 후보를 반환
      (outfits=[{tops, bottoms, shoes}, ...] 최대 NUM_CANDIDATES개,
       각 슬롯은 product_id/name/url/image_path/qr_b64). 검색 결과가 부족하면
       그보다 적은 수의 후보만 반환한다.

하의기준(anchor='bottoms') 계약:
  - outfit["bottoms"] == []  (DB 아이템 아닌 사용자 캡처)
  - result["bottoms_crop"] is not None
  - outfit["shoes"]는 anchor 하의의 snap_id로 먼저 조회, 없으면 유사도 검색 폴백
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
# 각 상품에 snap_id 필드 포함 (metadata.json 실제 구조와 동일)
META = {
    "musinsa_001": {"product_id": "musinsa_001", "gender": "남", "snap_id": "snap1", "name": "셔츠", "url": "https://musinsa.com/1", "image_path": "tops/musinsa_001.jpg"},
    "musinsa_002": {"product_id": "musinsa_002", "gender": "남", "snap_id": "snap1", "name": "청바지", "url": "https://musinsa.com/2", "image_path": "bottoms/musinsa_002.jpg"},
    "musinsa_003": {"product_id": "musinsa_003", "gender": "남", "snap_id": "snap1", "name": "스니커즈", "url": "https://musinsa.com/3", "image_path": "shoes/musinsa_003.jpg"},
    "musinsa_004": {"product_id": "musinsa_004", "gender": "남", "snap_id": "snap2", "name": "니트", "url": "https://musinsa.com/4", "image_path": "tops/musinsa_004.jpg"},
    "musinsa_005": {"product_id": "musinsa_005", "gender": "남", "snap_id": "snap2", "name": "슬랙스", "url": "https://musinsa.com/5", "image_path": "bottoms/musinsa_005.jpg"},
    "musinsa_006": {"product_id": "musinsa_006", "gender": "남", "snap_id": "snap2", "name": "로퍼", "url": "https://musinsa.com/6", "image_path": "shoes/musinsa_006.jpg"},
    "musinsa_007": {"product_id": "musinsa_007", "gender": "남", "snap_id": "snap3", "name": "후드티", "url": "https://musinsa.com/7", "image_path": "tops/musinsa_007.jpg"},
    "musinsa_008": {"product_id": "musinsa_008", "gender": "남", "snap_id": "snap3", "name": "조거팬츠", "url": "https://musinsa.com/8", "image_path": "bottoms/musinsa_008.jpg"},
    "musinsa_009": {"product_id": "musinsa_009", "gender": "남", "snap_id": "snap3", "name": "운동화", "url": "https://musinsa.com/9", "image_path": "shoes/musinsa_009.jpg"},
}
# searcher.search가 항상 반환하는 후보 (snap1/snap2/snap3에 각각 소속, 점수 내림차순)
CANDIDATES = [
    {**META["musinsa_001"], "category": "tops", "score": 0.95},
    {**META["musinsa_004"], "category": "tops", "score": 0.88},
    {**META["musinsa_007"], "category": "tops", "score": 0.80},
]
DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
DUMMY_CROP  = np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def rec():
    from recommender import Recommender
    r = object.__new__(Recommender)
    r._snap_outfits = SNAP_OUTFITS
    r.detector = mock.MagicMock()
    r.detector.detect.return_value = {
        "annotated": DUMMY_FRAME.copy(),
        "crops": {
            "tops":    DUMMY_CROP.copy(),
            "bottoms": DUMMY_CROP.copy(),
            "shoes":   DUMMY_CROP.copy(),
        },
        "persons": [True],
    }
    r.embedder = mock.MagicMock()
    r.embedder.embed.return_value = np.zeros(512, dtype=np.float32)
    r.searcher = mock.MagicMock()
    r.searcher.search.return_value = CANDIDATES
    r.searcher._meta = META
    r.reranker = mock.MagicMock()
    r.reranker.extract_palette.return_value = ["#3D6B9F", "#FFFFFF", "#000000"]
    r.reranker.rerank.side_effect = (
        lambda candidates, text_vec, palette, top_n=1: candidates[:top_n]
    )
    r._make_qr = lambda url: "fake_qr"
    return r


def test_recommend_outfit_returns_outfits_key(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert "outfits" in result


def test_recommend_outfit_returns_three_sets(rec):
    """검색 결과가 충분하면 NUM_CANDIDATES(3)개의 코디 후보를 반환한다."""
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert len(result["outfits"]) == 3


def test_recommend_outfit_returns_fewer_sets_when_candidates_scarce(rec):
    """검색 결과가 NUM_CANDIDATES보다 적으면 그 수만큼만 반환한다 (인덱스 에러 없음)."""
    rec.searcher.search.return_value = CANDIDATES[:1]
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert len(result["outfits"]) == 1


def test_recommend_outfit_set_structure(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for outfit in result["outfits"]:
        assert "tops" in outfit
        assert "bottoms" in outfit
        assert "shoes" in outfit


def test_recommend_outfit_products_have_required_fields(rec):
    """하의기준: tops·shoes 슬롯 아이템은 필수 필드를 가져야 한다 (bottoms는 사용자 캡처라 빈 리스트)."""
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for outfit in result["outfits"]:
        for slot in ("tops", "shoes"):
            for product in outfit[slot]:
                assert "product_id" in product
                assert "name" in product
                assert "url" in product
                assert "image_path" in product
                assert "qr_b64" in product


def test_recommend_outfit_bottoms_anchor_bottoms_slot_is_empty(rec):
    """하의기준: 모든 후보의 outfit['bottoms']는 빈 리스트여야 한다 (DB 아이템 표시 금지)."""
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for outfit in result["outfits"]:
        assert outfit["bottoms"] == []


def test_recommend_outfit_bottoms_anchor_tops_has_one_item(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert len(result["outfits"][0]["tops"]) == 1


def test_recommend_outfit_bottoms_anchor_shoes_has_one_item(rec):
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert len(result["outfits"][0]["shoes"]) == 1


def test_recommend_outfit_bottoms_anchor_returns_bottoms_crop(rec):
    """하의기준: result['bottoms_crop']에 캡처 이미지가 담겨야 한다."""
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert result.get("bottoms_crop") is not None


def test_recommend_outfit_tops_anchor_bottoms_crop_is_none(rec):
    """상의기준: result['bottoms_crop']은 None이어야 한다."""
    result = rec.recommend_outfit(DUMMY_FRAME, "tops")
    assert result.get("bottoms_crop") is None


def test_recommend_outfit_tops_anchor_bottoms_from_snap(rec):
    """상의기준: bottoms는 anchor top의 snap에서 온다 (snap1 → musinsa_002)."""
    result = rec.recommend_outfit(DUMMY_FRAME, "tops")
    bottoms = result["outfits"][0]["bottoms"]
    assert len(bottoms) == 1
    assert bottoms[0]["product_id"] == "musinsa_002"


def test_recommend_outfit_tops_anchor_shoes_from_snap(rec):
    """상의기준: shoes도 anchor top의 snap에서 온다 (snap1 → musinsa_003)."""
    result = rec.recommend_outfit(DUMMY_FRAME, "tops")
    shoes = result["outfits"][0]["shoes"]
    assert len(shoes) == 1
    assert shoes[0]["product_id"] == "musinsa_003"


def test_recommend_outfit_tops_anchor_candidates_use_own_snap(rec):
    """상의기준: 2·3번째 후보도 각자의 anchor top과 매칭되는 snap(snap2/snap3)을 사용해야 한다."""
    result = rec.recommend_outfit(DUMMY_FRAME, "tops")
    outfits = result["outfits"]
    assert outfits[1]["tops"][0]["product_id"] == "musinsa_004"
    assert outfits[1]["bottoms"][0]["product_id"] == "musinsa_005"
    assert outfits[1]["shoes"][0]["product_id"] == "musinsa_006"
    assert outfits[2]["tops"][0]["product_id"] == "musinsa_007"
    assert outfits[2]["bottoms"][0]["product_id"] == "musinsa_008"
    assert outfits[2]["shoes"][0]["product_id"] == "musinsa_009"


def test_recommend_outfit_tops_anchor_fallback_when_no_snap_match(rec):
    """상의기준: snap 없으면 bottoms·shoes 모두 유사도 검색 폴백으로 채워진다."""
    rec._snap_outfits = {}
    result = rec.recommend_outfit(DUMMY_FRAME, "tops")
    outfit = result["outfits"][0]
    assert len(outfit["bottoms"]) == 1
    assert len(outfit["shoes"]) == 1


def test_recommend_outfit_shoes_from_snap_lookup(rec):
    """하의기준: tops top-1이 snap1의 musinsa_001 → shoes는 snap1의 musinsa_003이어야 한다."""
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    shoes = result["outfits"][0]["shoes"]
    assert len(shoes) == 1
    assert shoes[0]["product_id"] == "musinsa_003"


def test_recommend_outfit_shoes_fallback_when_no_snap_match(rec):
    """anchor bottoms의 snap이 없으면 tops·shoes 모두 유사도 검색 폴백으로 채워져야 한다."""
    rec._snap_outfits = {}  # 비어있어 snap 매칭 불가
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    shoes = result["outfits"][0]["shoes"]
    assert len(shoes) == 1  # 폴백으로 get_items("shoes") 호출됨


def test_recommend_outfit_fallback_when_no_crop(rec):
    rec.detector.detect.return_value = {
        "annotated": DUMMY_FRAME.copy(),
        "crops": {},
        "persons": [],
    }
    result = rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    assert "outfits" in result
    rec.embedder.embed.assert_called_once_with(DUMMY_FRAME)


def test_recommend_outfit_passes_gender_to_searcher(rec):
    rec.recommend_outfit(DUMMY_FRAME, "bottoms", gender="남")
    for call in rec.searcher.search.call_args_list:
        assert call.kwargs.get("gender") == "남" or call.args[1:2] == ("남",), \
            f"gender='남' not passed to searcher.search: {call}"


def test_recommend_outfit_no_gender_passes_none_to_searcher(rec):
    rec.recommend_outfit(DUMMY_FRAME, "bottoms")
    for call in rec.searcher.search.call_args_list:
        assert call.kwargs.get("gender") is None


def test_shoes_from_snap_gender_filter(rec):
    """gender='남' 일 때 'men_' 접두사 snap만 반환하고, 'women_'는 제외한다."""
    rec._snap_outfits = {
        "men_snap1":   {"tops": ["musinsa_001"], "bottoms": [], "shoes": ["musinsa_003"]},
        "women_snap1": {"tops": ["musinsa_010"], "bottoms": [], "shoes": ["musinsa_006"]},
    }
    # musinsa_001의 snap_id를 men_snap1으로 직접 지정
    rec.searcher._meta["musinsa_001"] = {**rec.searcher._meta["musinsa_001"], "snap_id": "men_snap1"}

    shoes = rec._shoes_from_snap(["musinsa_001"], gender="남")
    assert len(shoes) == 1
    assert shoes[0]["product_id"] == "musinsa_003"

    # 여성 필터: men_snap1 접두사 불일치 → 빈 결과
    shoes_f = rec._shoes_from_snap(["musinsa_001"], gender="여")
    assert shoes_f == []


def test_shoes_from_snap_returns_empty_when_no_match(rec):
    shoes = rec._shoes_from_snap(["nonexistent_id"], gender="")
    assert shoes == []
