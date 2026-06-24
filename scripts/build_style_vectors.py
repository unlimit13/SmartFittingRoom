"""
Build ko-sroberta style vectors from product style_text fields in metadata.json.

Output:
  data/faiss_index/style_vectors.npy  - shape (N, 768), L2-normalized
  data/faiss_index/id_map.json must already exist (same order as FAISS index)
"""
import json
import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "musinsa_db")
INDEX_DIR = os.path.join(ROOT, "data", "faiss_index")
KO_MODEL_DIR = os.path.join(ROOT, "models", "ko_sroberta")
META_PATH = os.path.join(DB_DIR, "metadata.json")


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    count = mask.sum(axis=1).clip(min=1e-9)
    return summed / count


def encode(session, tokenizer, text):
    enc = tokenizer(
        text, return_tensors="np", padding=True,
        truncation=True, max_length=128
    )
    outputs = session.run(
        None,
        {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )
    # outputs[0]: (1, seq_len, 768) — last hidden state
    pooled = mean_pool(outputs[0], enc["attention_mask"])  # (1, 768)
    vec = pooled[0]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def main():
    with open(os.path.join(INDEX_DIR, "id_map.json"), "r", encoding="utf-8") as f:
        id_map = json.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    meta_by_id = {item["product_id"]: item for item in metadata}

    # Find ONNX model file
    onnx_candidates = [
        os.path.join(KO_MODEL_DIR, "model.onnx"),
        os.path.join(KO_MODEL_DIR, "onnx", "model.onnx"),
    ]
    onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), None)
    if not onnx_path:
        raise FileNotFoundError(f"ko-sroberta ONNX not found in {KO_MODEL_DIR}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(KO_MODEL_DIR)

    vectors = []
    for i, pid in enumerate(id_map):
        item = meta_by_id.get(pid)
        text = item["style_text"] if item else pid
        vec = encode(session, tokenizer, text)
        vectors.append(vec)
        if (i + 1) % 100 == 0:
            print(f"  encoded {i + 1}/{len(id_map)}")

    matrix = np.stack(vectors, axis=0).astype(np.float32)
    out_path = os.path.join(INDEX_DIR, "style_vectors.npy")
    np.save(out_path, matrix)
    print(f"Style vectors saved: {matrix.shape} → {out_path}")


if __name__ == "__main__":
    main()
