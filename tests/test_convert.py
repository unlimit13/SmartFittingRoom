import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_fake_result(tmp_path):
    nukki = tmp_path / "musinsa_out" / "nukki"
    for slot in ("상의", "하의", "신발"):
        (nukki / slot).mkdir(parents=True)
    (nukki / "상의" / "snap1_001_0.jpg").write_bytes(b"\xff\xd8\xff")
    (nukki / "하의" / "snap1_002_0.jpg").write_bytes(b"\xff\xd8\xff")
    (nukki / "신발" / "snap1_003_0.jpg").write_bytes(b"\xff\xd8\xff")

    result = [
        {
            "snap_id": "snap1",
            "description": "test",
            "hashtags": ["캐주얼"],
            "items_by_slot": {
                "상의": [{"goodsNo": "001", "name": "셔츠", "saved_path": "nukki/상의/snap1_001_0.jpg"}],
                "하의": [{"goodsNo": "002", "name": "청바지", "saved_path": "nukki/하의/snap1_002_0.jpg"}],
                "신발": [{"goodsNo": "003", "name": "스니커즈", "saved_path": "nukki/신발/snap1_003_0.jpg"}],
            },
        }
    ]
    (tmp_path / "musinsa_out" / "result.json").write_text(json.dumps(result), encoding="utf-8")


@pytest.fixture
def converted(tmp_path, monkeypatch):
    _make_fake_result(tmp_path)
    import scripts.convert_musinsa_out as conv
    monkeypatch.setattr(conv, "ROOT", str(tmp_path))
    monkeypatch.setattr(conv, "SRC_DIR", str(tmp_path / "musinsa_out"))
    monkeypatch.setattr(conv, "DST_DIR", str(tmp_path / "data" / "musinsa_db"))
    monkeypatch.setattr(conv, "RESULT_JSON", str(tmp_path / "musinsa_out" / "result.json"))
    conv.main()
    return tmp_path / "data" / "musinsa_db"


def test_metadata_contains_snap_id(converted):
    meta = json.loads((converted / "metadata.json").read_text())
    assert len(meta) == 3
    assert all(m["snap_id"] == "snap1" for m in meta)


def test_snap_outfits_json_created(converted):
    snap_outfits = json.loads((converted / "snap_outfits.json").read_text())
    assert "snap1" in snap_outfits
    assert snap_outfits["snap1"]["tops"] == ["musinsa_001"]
    assert snap_outfits["snap1"]["bottoms"] == ["musinsa_002"]
    assert snap_outfits["snap1"]["shoes"] == ["musinsa_003"]


def test_images_copied(converted):
    assert (converted / "tops" / "musinsa_001.jpg").exists()
    assert (converted / "bottoms" / "musinsa_002.jpg").exists()
    assert (converted / "shoes" / "musinsa_003.jpg").exists()
