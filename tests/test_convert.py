"""
데이터 변환 (R-05/R-07 지원) — 무신사 크롤 결과 → metadata.json / snap_outfits.json /
이미지 복사. FAISS 검색(R-05)과 코디 추천(R-07)이 소비하는 DB 자원을 생성하는 단계를 검증.
남성(musinsa_out_men)과 여성(musinsa_out_women) 두 소스를 통합하고 gender 필드를 부여하는지 검증.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_fake_result_gendered(tmp_path):
    """남/여 각각 1개 snap을 가진 가짜 데이터 디렉토리를 생성한다."""
    for folder, goods_nos in [("musinsa_out_men", ["001", "002", "003"]),
                               ("musinsa_out_women", ["004", "005", "006"])]:
        nukki = tmp_path / folder / "nukki"
        for slot in ("상의", "하의", "신발"):
            (nukki / slot).mkdir(parents=True)
        snap_id = "snap_m" if folder == "musinsa_out_men" else "snap_w"
        slots = {"상의": goods_nos[0], "하의": goods_nos[1], "신발": goods_nos[2]}
        for slot, gno in slots.items():
            (nukki / slot / f"{snap_id}_{gno}_0.jpg").write_bytes(b"\xff\xd8\xff")
        result = [
            {
                "snap_id": snap_id,
                "description": "test",
                "hashtags": ["캐주얼"],
                "items_by_slot": {
                    slot: [{"goodsNo": gno, "name": f"상품{gno}",
                             "saved_path": f"nukki/{slot}/{snap_id}_{gno}_0.jpg"}]
                    for slot, gno in slots.items()
                },
            }
        ]
        (tmp_path / folder / "result.json").write_text(json.dumps(result), encoding="utf-8")


@pytest.fixture
def converted(tmp_path, monkeypatch):
    _make_fake_result_gendered(tmp_path)
    import scripts.convert_musinsa_out as conv
    monkeypatch.setattr(conv, "ROOT", str(tmp_path))
    monkeypatch.setattr(conv, "DST_DIR", str(tmp_path / "data" / "musinsa_db"))
    monkeypatch.setattr(conv, "SOURCES", [
        (str(tmp_path / "musinsa_out_men"), "남"),
        (str(tmp_path / "musinsa_out_women"), "여"),
    ])
    conv.main()
    return tmp_path / "data" / "musinsa_db"


def test_metadata_contains_all_products(converted):
    meta = json.loads((converted / "metadata.json").read_text())
    assert len(meta) == 6  # 3 men + 3 women


def test_metadata_has_gender_field(converted):
    meta = json.loads((converted / "metadata.json").read_text())
    for m in meta:
        assert "gender" in m
        assert m["gender"] in ("남", "여")


def test_metadata_gender_values_correct(converted):
    meta = json.loads((converted / "metadata.json").read_text())
    men = [m for m in meta if m["gender"] == "남"]
    women = [m for m in meta if m["gender"] == "여"]
    assert len(men) == 3
    assert len(women) == 3


def test_snap_outfits_json_created(converted):
    snap_outfits = json.loads((converted / "snap_outfits.json").read_text())
    assert "men_snap_m" in snap_outfits
    assert "women_snap_w" in snap_outfits
    assert snap_outfits["men_snap_m"]["tops"] == ["musinsa_001"]
    assert snap_outfits["women_snap_w"]["tops"] == ["musinsa_004"]


def test_images_copied(converted):
    assert (converted / "tops" / "musinsa_001.jpg").exists()
    assert (converted / "bottoms" / "musinsa_002.jpg").exists()
    assert (converted / "shoes" / "musinsa_003.jpg").exists()
    assert (converted / "tops" / "musinsa_004.jpg").exists()
    assert (converted / "bottoms" / "musinsa_005.jpg").exists()
    assert (converted / "shoes" / "musinsa_006.jpg").exists()
