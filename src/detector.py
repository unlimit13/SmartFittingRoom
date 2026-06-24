"""
YOLOv8n ONNX person detection + rule-based clothing region split.

Standard YOLOv8n is trained on COCO (80 classes) which has no clothing classes.
Strategy: detect 'person' (class 0), then split bounding box vertically:
  tops    : y_top  ~ y_top + 45% of height
  bottoms : y_top + 40% ~ y_top + 80% of height
  shoes   : y_top + 75% ~ y_bottom
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
YOLO_ONNX = os.path.join(MODELS_DIR, "yolov8n.onnx")

PERSON_CLASS = 0
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 640

# Vertical split ratios within person bounding box
SPLIT = {"tops": (0.0, 0.45), "bottoms": (0.40, 0.80), "shoes": (0.75, 1.0)}


class Detector:
    def __init__(self, model_path=YOLO_ONNX):
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis], h, w

    def _postprocess(self, outputs, orig_h, orig_w):
        # YOLOv8 output shape: (1, 84, 8400) — [cx,cy,w,h, 80 class scores]
        pred = outputs[0][0].T  # (8400, 84)
        scores = pred[:, 4:].max(axis=1)
        class_ids = pred[:, 4:].argmax(axis=1)

        mask = (scores >= CONF_THRESHOLD) & (class_ids == PERSON_CLASS)
        pred = pred[mask]
        scores = scores[mask]

        if len(pred) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 (normalized to INPUT_SIZE)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - bw / 2) / INPUT_SIZE * orig_w
        y1 = (cy - bh / 2) / INPUT_SIZE * orig_h
        x2 = (cx + bw / 2) / INPUT_SIZE * orig_w
        y2 = (cy + bh / 2) / INPUT_SIZE * orig_h

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
        )
        if len(indices) == 0:
            return []

        results = []
        for i in indices.flatten():
            x1i, y1i, x2i, y2i = (
                int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
            )
            results.append({
                "bbox": [x1i, y1i, x2i, y2i],
                "confidence": float(scores[i]),
            })
        return results

    def detect(self, frame: np.ndarray) -> dict:
        """
        Returns dict with:
          'persons': list of {'bbox': [x1,y1,x2,y2], 'confidence': float}
          'crops': {'tops': ndarray|None, 'bottoms': ndarray|None, 'shoes': ndarray|None}
          'annotated': frame with bounding boxes drawn
        """
        inp, orig_h, orig_w = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: inp})
        persons = self._postprocess(outputs, orig_h, orig_w)

        annotated = frame.copy()
        crops = {"tops": None, "bottoms": None, "shoes": None}

        if not persons:
            return {"persons": [], "crops": crops, "annotated": annotated}

        # Use largest person box for cropping
        best = max(persons, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))
        x1, y1, x2, y2 = best["bbox"]
        box_h = y2 - y1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"person {best['confidence']:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        REGION_COLOR = {"tops": (255, 100, 0), "bottoms": (0, 100, 255), "shoes": (180, 0, 255)}
        for cat, (r0, r1) in SPLIT.items():
            cy1 = max(0, int(y1 + box_h * r0))
            cy2 = min(orig_h, int(y1 + box_h * r1))
            cx1, cx2 = max(0, x1), min(orig_w, x2)
            if cy2 > cy1 and cx2 > cx1:
                crops[cat] = frame[cy1:cy2, cx1:cx2].copy()
                color = REGION_COLOR[cat]
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 1)
                cv2.putText(annotated, cat, (cx1 + 4, cy1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return {"persons": persons, "crops": crops, "annotated": annotated}
