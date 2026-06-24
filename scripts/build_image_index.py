"""
Build FAISS image index from Musinsa product images using CLIP ViT-B/32.
Run on a machine with the models and data already set up.

Output:
  data/faiss_index/index.bin    - FAISS IndexFlatIP (inner product = cosine on L2-normalized vecs)
  data/faiss_index/id_map.json  - list of product_ids in index order
  data/musinsa_db/metadata.json - updated with dominant_color per product
"""
import json
import os

import cv2
import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "musinsa_db")
INDEX_DIR = os.path.join(ROOT, "data", "faiss_index")
MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(DB_DIR, "metadata.json")

CLIP_ONNX = os.path.join(MODELS_DIR, "clip_image_encoder.onnx")
PROJ_PATH = os.path.join(MODELS_DIR, "clip_preprocessor", "visual_projection.npy")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)


def embed(session, proj, pixel_values):
    outputs = session.run(None, {"pixel_values": pixel_values})
    cls_token = outputs[0][:, 0, :]  # (1, 768)
    vec = cls_token @ proj.T         # (1, 512)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    return vec[0]  # (512,)


def extract_dominant_color(img_path, k=1):
    img = cv2.imread(img_path)
    if img is None:
        return "#000000"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    dominant = centers[np.argmax(np.bincount(labels.flatten()))].astype(int)
    return "#{:02X}{:02X}{:02X}".format(*dominant)


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    session = ort.InferenceSession(CLIP_ONNX, providers=["CPUExecutionProvider"])
    proj = np.load(PROJ_PATH)  # (512, 768)

    vectors = []
    id_map = []

    for i, item in enumerate(metadata):
        img_path = os.path.join(DB_DIR, item["image_path"])
        if not os.path.exists(img_path):
            print(f"  skip (missing): {img_path}")
            continue

        try:
            pixels = preprocess(img_path)
            vec = embed(session, proj, pixels)
            vectors.append(vec)
            id_map.append(item["product_id"])

            # Update dominant color in metadata
            item["dominant_color"] = extract_dominant_color(img_path)

            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(metadata)}")
        except Exception as e:
            print(f"  error on {item['product_id']}: {e}")

    if not vectors:
        print("No vectors collected. Exiting.")
        return

    matrix = np.stack(vectors, axis=0).astype(np.float32)  # (N, 512)

    index = faiss.IndexFlatIP(512)
    index.add(matrix)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.bin"))

    with open(os.path.join(INDEX_DIR, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    # Save updated metadata (with dominant_color)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Index built: {len(vectors)} items. Saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
