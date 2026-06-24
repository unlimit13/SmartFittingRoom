"""
One-time script to download and convert all required ONNX models.
Run on a machine with internet access (local Mac recommended).
Total disk usage: ~800MB
"""
import os
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
CLIP_PREP_DIR = os.path.join(MODELS_DIR, "clip_preprocessor")
KO_SROBERTA_DIR = os.path.join(MODELS_DIR, "ko_sroberta")

os.makedirs(CLIP_PREP_DIR, exist_ok=True)
os.makedirs(KO_SROBERTA_DIR, exist_ok=True)


def setup_yolov8n():
    """Export yolov8n.pt → yolov8n.onnx (~13MB)"""
    out_path = os.path.join(MODELS_DIR, "yolov8n.onnx")
    if os.path.exists(out_path):
        print(f"[yolov8n] already exists: {out_path}")
        return

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    exported = model.export(format="onnx", opset=12, imgsz=640)
    shutil.move(str(exported), out_path)
    print(f"[yolov8n] saved to {out_path}")


def setup_clip():
    """Export CLIP ViT-B/32 image encoder → clip_image_encoder.onnx (~310MB)
    Also saves visual_projection.npy for 512-dim projection."""
    onnx_path = os.path.join(MODELS_DIR, "clip_image_encoder.onnx")
    proj_path = os.path.join(CLIP_PREP_DIR, "visual_projection.npy")

    if os.path.exists(onnx_path) and os.path.exists(proj_path):
        print(f"[CLIP] already exists: {onnx_path}")
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("[CLIP] loading openai/clip-vit-base-patch32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(CLIP_PREP_DIR)

    # Save projection weight (768 → 512)
    proj_w = model.visual_projection.weight.detach().numpy()  # (512, 768)
    np.save(proj_path, proj_w)
    print(f"[CLIP] visual_projection saved: {proj_path}")

    # Export vision model (outputs last_hidden_state)
    vision_model = model.vision_model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        vision_model,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        opset_version=14,
        dynamic_axes={"pixel_values": {0: "batch"}},
    )
    print(f"[CLIP] image encoder saved: {onnx_path}")


def setup_ko_sroberta():
    """Export jhgan/ko-sroberta-multitask → ONNX (~460MB)"""
    model_file = os.path.join(KO_SROBERTA_DIR, "model.onnx")
    if os.path.exists(model_file):
        print(f"[ko-sroberta] already exists: {model_file}")
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    print("[ko-sroberta] exporting jhgan/ko-sroberta-multitask to ONNX ...")
    tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
    model = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask").eval()

    dummy_ids = torch.ones(1, 64, dtype=torch.long)
    dummy_mask = torch.ones(1, 64, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            model_file,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )

    tokenizer.save_pretrained(KO_SROBERTA_DIR)
    print(f"[ko-sroberta] saved to {KO_SROBERTA_DIR}")


if __name__ == "__main__":
    print("=== Setting up models ===")
    setup_yolov8n()
    setup_clip()
    setup_ko_sroberta()
    print("=== Done ===")
