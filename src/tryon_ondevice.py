"""
On-device virtual try-on backend (Mobile-VTON spatial-parallel over a 4-Pi cluster).

Drop-in replacement for `tryon.py` (the fal-ai cloud backend): exposes the same
three functions used by app.py — `upload_frame`, `run_tryon_stream`, `fetch_b64`.
Selected by setting `VTON_BACKEND=ondevice` (default backend is the fal-ai API).

How it works: instead of calling a cloud API, each garment is run through the
vendored `src/ondevice_vton/` pipeline, distributed across the Raspberry Pi cluster
via `parallel/run_sp_multi.sh`. A "person_url" here is just a LOCAL file path (no
upload). Sequential strategy mirrors the API path: apply tops → use result as the
person → apply bottoms.

Requires the 4-Pi (or 2-Pi) spatial cluster to be up — full-res single-Pi OOMs.
See parallel/PI_SETUP.md + RUN.md (on-device try-on section) for setup and env vars.
"""
import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image

# Vendored pipeline lives next to this file under src/ondevice_vton/.
_HERE = os.path.dirname(__file__)
RANK0_DIR = os.environ.get("VTON_RANK0_DIR", os.path.join(_HERE, "ondevice_vton"))
LAUNCHER = os.path.join(RANK0_DIR, "parallel", "run_sp_multi.sh")
DB_DIR = os.path.join(os.path.dirname(_HERE), "data", "musinsa_db")

# Cluster / run configuration (all overridable via env; see RUN.md).
PEERS = os.environ.get("VTON_PEERS", "192.168.100.2 192.168.100.3 192.168.100.4")
PEER_DIR = os.environ.get("VTON_PEER_DIR", RANK0_DIR)
STEPS = os.environ.get("VTON_STEPS", "6")
# download_ckpt.py drops the checkpoint into src/ondevice_vton/checkpoint.
CKPT = os.environ.get("VTON_CHECKPOINT_PATH", os.path.join(RANK0_DIR, "checkpoint"))
# Interpreter that has the vton deps (requirements_vton.txt). Default assumes a
# `.venv` inside the vendored dir; override for a shared env.
PY = os.environ.get("VTON_PYTHON", ".venv/bin/python")
PEER_PY = os.environ.get("VTON_PEER_PYTHON", PY)

# Relative (to RANK0_DIR/PEER_DIR) scratch dirs the launcher reads/writes.
_RUN_DATA = "_vton_run/single_data"
_RUN_OUT = "_vton_run/output"

# The cluster runs one image at a time; serialize concurrent /tryon requests.
_run_lock = threading.Lock()

_DEFAULT_DESC = "a clothing garment worn by the person"


def upload_frame(frame: np.ndarray) -> str:
    """BGR numpy frame → local JPEG path (no network; mirrors the API signature)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="vton_person_")
    os.close(fd)
    Image.fromarray(rgb).save(path, "JPEG", quality=92)
    return path


def _garment_description(rel_path: str) -> str:
    """Best-effort garment text from musinsa metadata; falls back to a neutral desc."""
    meta_path = os.path.join(DB_DIR, "metadata.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        items = meta.values() if isinstance(meta, dict) else meta
        for it in items:
            if isinstance(it, dict) and it.get("image_path", "").endswith(rel_path):
                return it.get("style_text") or it.get("name") or _DEFAULT_DESC
    except (OSError, ValueError):
        pass
    return _DEFAULT_DESC


def _build_single_data(person_path: str, garment_full: str, desc: str) -> tuple[str, str]:
    """Write a one-pair single_data dir under RANK0_DIR; return (person_name, cloth_name)."""
    root = os.path.join(RANK0_DIR, _RUN_DATA)
    img_dir = os.path.join(root, "test", "image")
    cloth_dir = os.path.join(root, "test", "cloth")
    if os.path.isdir(os.path.join(RANK0_DIR, "_vton_run")):
        shutil.rmtree(os.path.join(RANK0_DIR, "_vton_run"))
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(cloth_dir, exist_ok=True)

    # Fixed names so the output filename ({person_stem}_{cloth_name}) is predictable.
    person_name = "person.jpg"
    cloth_name = "cloth.jpg"
    Image.open(person_path).convert("RGB").save(os.path.join(img_dir, person_name), "JPEG")
    Image.open(garment_full).convert("RGB").save(os.path.join(cloth_dir, cloth_name), "JPEG")

    with open(os.path.join(root, "test", "image_descriptions.txt"), "w", encoding="utf-8") as f:
        f.write(f"{cloth_name}: {desc}\n")
    with open(os.path.join(root, "test_pairs.txt"), "w", encoding="utf-8") as f:
        f.write(f"{person_name} {cloth_name}\n")
    return person_name, cloth_name


def _sync_to_peers() -> None:
    """rsync the scratch single_data to each peer so its rank can read its own copy."""
    src = os.path.join(RANK0_DIR, _RUN_DATA) + "/"
    for ip in PEERS.split():
        dst = f"{ip}:{os.path.join(PEER_DIR, _RUN_DATA)}/"
        subprocess.run(
            ["rsync", "-a", "--delete", "-e", "ssh -o BatchMode=yes", src, dst],
            check=True,
        )


def _run_cluster() -> None:
    """Invoke the spatial launcher across the cluster (blocks until rank0 finishes)."""
    env = dict(os.environ)
    env.update(
        RANK0_DIR=RANK0_DIR,
        PEER_DIR=PEER_DIR,
        VTON_CHECKPOINT_PATH=CKPT,
        PEER_CHECKPOINT_PATH=os.environ.get("VTON_PEER_CHECKPOINT_PATH", CKPT),
        VTON_PYTHON=PY,
        PEER_PYTHON=PEER_PY,
    )
    subprocess.run(
        ["bash", LAUNCHER, STEPS, _RUN_OUT, _RUN_DATA, PEERS],
        cwd=RANK0_DIR,
        env=env,
        check=True,
    )


def _tryon_one(person_path: str, garment_rel_path: str) -> str:
    """Run one garment through the cluster; return the local result image path."""
    garment_full = os.path.join(DB_DIR, garment_rel_path)
    person_name, cloth_name = _build_single_data(
        person_path, garment_full, _garment_description(garment_rel_path)
    )
    _sync_to_peers()
    _run_cluster()
    out = os.path.join(RANK0_DIR, _RUN_OUT, f"{person_name[:-4]}_{cloth_name}")
    if not os.path.exists(out):
        raise RuntimeError(f"on-device try-on produced no output at {out}")
    return out


def run_tryon_stream(person_url: str, top_rel_path: str = None, bottom_rel_path: str = None):
    """Generator: yields (step, local_image_path) as each garment completes."""
    print(f"[tryon-ondevice] top={top_rel_path}  bottom={bottom_rel_path}", flush=True)
    with _run_lock:
        current = person_url
        if top_rel_path:
            current = _tryon_one(current, top_rel_path)
            yield "tops", current
        if bottom_rel_path:
            current = _tryon_one(current, bottom_rel_path)
            yield "bottoms", current


def fetch_b64(url: str) -> tuple[str, str]:
    """Image path-or-URL → (base64 string, mime_type). Local paths are read directly."""
    if os.path.exists(url):
        with open(url, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode(), "image/jpeg"
    with urllib.request.urlopen(url) as res:
        mime = res.headers.get_content_type() or "image/jpeg"
        data = res.read()
    return base64.b64encode(data).decode(), mime
