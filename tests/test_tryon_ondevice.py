"""
R-13: 가상 피팅 (온디바이스 백엔드) — Unit tests for src/tryon_ondevice.py.

The real backend shells out to a Raspberry Pi cluster (rsync + a spatial launcher),
so these tests exercise only the locally-verifiable glue logic and mock out anything
that touches the cluster (_build_single_data / _sync_to_peers / _run_cluster).
The cluster-dependent path itself is covered by the demo, not here.
"""
import base64
import json
import os

import numpy as np
import pytest
from PIL import Image

import tryon_ondevice as tod


# --- upload_frame --------------------------------------------------------------

def test_upload_frame_writes_jpeg(dummy_frame):
    path = tod.upload_frame(dummy_frame)
    try:
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.format == "JPEG"
        # PIL is (W, H); frame is (H, W, C).
        assert img.size == (dummy_frame.shape[1], dummy_frame.shape[0])
    finally:
        os.remove(path)


def test_upload_frame_converts_bgr_to_rgb():
    # Solid blue in BGR (B=255). After BGR->RGB the dominant channel must be R index 2 (blue).
    bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # blue channel in BGR
    path = tod.upload_frame(bgr)
    try:
        r, g, b = Image.open(path).convert("RGB").getpixel((16, 16))
        assert b > 200 and r < 60 and g < 60
    finally:
        os.remove(path)


# --- fetch_b64 -----------------------------------------------------------------

def test_fetch_b64_local_path_roundtrips(tmp_path):
    raw = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
    f = tmp_path / "result.jpg"
    f.write_bytes(raw)
    b64, mime = tod.fetch_b64(str(f))
    assert base64.b64decode(b64) == raw
    assert mime == "image/jpeg"


# --- _garment_description ------------------------------------------------------

def _write_meta(tmp_path, meta):
    db = tmp_path / "musinsa_db"
    db.mkdir()
    (db / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(db)


def test_garment_description_uses_style_text(tmp_path, monkeypatch):
    db = _write_meta(tmp_path, {
        "p1": {"image_path": "tops/musinsa_001.jpg", "style_text": "blue oxford shirt", "name": "셔츠"},
    })
    monkeypatch.setattr(tod, "DB_DIR", db)
    assert tod._garment_description("tops/musinsa_001.jpg") == "blue oxford shirt"


def test_garment_description_falls_back_to_name(tmp_path, monkeypatch):
    db = _write_meta(tmp_path, {
        "p1": {"image_path": "tops/musinsa_001.jpg", "name": "셔츠"},
    })
    monkeypatch.setattr(tod, "DB_DIR", db)
    assert tod._garment_description("tops/musinsa_001.jpg") == "셔츠"


def test_garment_description_default_when_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tod, "DB_DIR", str(tmp_path / "does_not_exist"))
    assert tod._garment_description("tops/x.jpg") == tod._DEFAULT_DESC


def test_garment_description_default_when_no_match(tmp_path, monkeypatch):
    db = _write_meta(tmp_path, {
        "p1": {"image_path": "tops/other.jpg", "style_text": "x"},
    })
    monkeypatch.setattr(tod, "DB_DIR", db)
    assert tod._garment_description("tops/musinsa_001.jpg") == tod._DEFAULT_DESC


# --- _build_single_data --------------------------------------------------------

def test_build_single_data_lays_out_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(tod, "RANK0_DIR", str(tmp_path))
    person = tmp_path / "person_in.jpg"
    cloth = tmp_path / "cloth_in.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(person, "JPEG")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(cloth, "JPEG")

    person_name, cloth_name = tod._build_single_data(str(person), str(cloth), "a red shirt")

    assert (person_name, cloth_name) == ("person.jpg", "cloth.jpg")
    root = tmp_path / "_vton_run" / "single_data"
    assert (root / "test" / "image" / "person.jpg").exists()
    assert (root / "test" / "cloth" / "cloth.jpg").exists()
    assert (root / "test_pairs.txt").read_text() == "person.jpg cloth.jpg\n"
    assert "a red shirt" in (root / "test" / "image_descriptions.txt").read_text()


# --- _tryon_one ----------------------------------------------------------------

def test_tryon_one_returns_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tod, "RANK0_DIR", str(tmp_path))
    monkeypatch.setattr(tod, "_garment_description", lambda rel: "desc")
    monkeypatch.setattr(tod, "_build_single_data", lambda *a: ("person.jpg", "cloth.jpg"))
    monkeypatch.setattr(tod, "_sync_to_peers", lambda: None)

    out_dir = tmp_path / "_vton_run" / "output"
    out_dir.mkdir(parents=True)
    expected = out_dir / "person_cloth.jpg"

    def fake_cluster():
        expected.write_bytes(b"img")
    monkeypatch.setattr(tod, "_run_cluster", fake_cluster)

    assert tod._tryon_one("/person/path.jpg", "tops/g.jpg") == str(expected)


def test_tryon_one_raises_when_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(tod, "RANK0_DIR", str(tmp_path))
    monkeypatch.setattr(tod, "_garment_description", lambda rel: "desc")
    monkeypatch.setattr(tod, "_build_single_data", lambda *a: ("person.jpg", "cloth.jpg"))
    monkeypatch.setattr(tod, "_sync_to_peers", lambda: None)
    monkeypatch.setattr(tod, "_run_cluster", lambda: None)  # produces nothing

    with pytest.raises(RuntimeError):
        tod._tryon_one("/person/path.jpg", "tops/g.jpg")


# --- run_tryon_stream (sequential strategy: tops result feeds bottoms) ---------

def test_run_tryon_stream_chains_top_then_bottom(monkeypatch):
    calls = []

    def fake_one(person, garment):
        calls.append((person, garment))
        return f"out::{garment}"
    monkeypatch.setattr(tod, "_tryon_one", fake_one)

    steps = list(tod.run_tryon_stream("PERSON", top_rel_path="t.jpg", bottom_rel_path="b.jpg"))

    assert steps == [("tops", "out::t.jpg"), ("bottoms", "out::b.jpg")]
    # bottoms must run on the tops result, not the original person.
    assert calls[0] == ("PERSON", "t.jpg")
    assert calls[1] == ("out::t.jpg", "b.jpg")


def test_run_tryon_stream_top_only(monkeypatch):
    monkeypatch.setattr(tod, "_tryon_one", lambda person, garment: f"out::{garment}")
    steps = list(tod.run_tryon_stream("PERSON", top_rel_path="t.jpg"))
    assert steps == [("tops", "out::t.jpg")]


def test_run_tryon_stream_no_garment_yields_nothing(monkeypatch):
    monkeypatch.setattr(tod, "_tryon_one", lambda *a: pytest.fail("should not be called"))
    assert list(tod.run_tryon_stream("PERSON")) == []
