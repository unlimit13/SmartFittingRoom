"""
fal-ai/fashn/tryon virtual try-on helper.

Requires FAL_KEY environment variable.
Sequential strategy: apply tops → use result as person → apply bottoms.
"""
import base64
import io
import os
import tempfile
import urllib.request

import cv2
import fal_client
import numpy as np
from PIL import Image

ENDPOINT = "fal-ai/fashn/tryon/v1.6"
MAX_SIDE = 1296
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "musinsa_db")


def _upload_path(path: str) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(img.size)
        img = img.resize(
            (round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, "JPEG", quality=92)
            url = fal_client.upload_file(tmp.name)
        os.unlink(tmp.name)
        return url
    return fal_client.upload_file(path)


def upload_frame(frame: np.ndarray) -> str:
    """BGR numpy frame → fal storage URL."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if max(img.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(img.size)
        img = img.resize(
            (round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS
        )
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG", quality=92)
        url = fal_client.upload_file(tmp.name)
    os.unlink(tmp.name)
    return url


def _call(person_url: str, garment_url: str, category: str) -> str:
    result = fal_client.subscribe(
        ENDPOINT,
        arguments={
            "model_image": person_url,
            "garment_image": garment_url,
            "category": category,
            "mode": "performance",
            "num_samples": 1,
            "output_format": "jpeg",
        },
    )
    return result["images"][0]["url"]


def run_tryon(person_url: str, top_rel_path: str = None, bottom_rel_path: str = None) -> str:
    """
    Sequential try-on. Returns final result image URL.
    top_rel_path / bottom_rel_path: relative to data/musinsa_db/
    """
    current = person_url

    if top_rel_path:
        full = os.path.join(DB_DIR, top_rel_path)
        garment_url = _upload_path(full)
        current = _call(current, garment_url, "tops")

    if bottom_rel_path:
        full = os.path.join(DB_DIR, bottom_rel_path)
        garment_url = _upload_path(full)
        current = _call(current, garment_url, "bottoms")

    return current


def fetch_b64(url: str) -> str:
    """Image URL → base64 JPEG string."""
    with urllib.request.urlopen(url) as res:
        data = res.read()
    return base64.b64encode(data).decode()
