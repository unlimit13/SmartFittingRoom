"""
MediaPipe Pose person detection + rule-based clothing region split.

MediaPipe Pose detects 33 landmarks. Person bounding box is derived from
the extent of visible landmarks, then split vertically:
  tops    : y_top  ~ y_top + 45% of height
  bottoms : y_top + 40% ~ y_top + 80% of height
  shoes   : y_top + 75% ~ y_bottom
"""
import cv2
import mediapipe as mp
import numpy as np

CONF_THRESHOLD = 0.5

SPLIT = {"tops": (0.0, 0.45), "bottoms": (0.40, 0.80), "shoes": (0.75, 1.0)}

KEY_LANDMARKS = [0, 11, 12, 23, 24]  # nose, shoulders, hips


class Detector:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> dict:
        """
        Returns dict with:
          'persons': list of {'bbox': [x1,y1,x2,y2], 'confidence': float}
          'crops': {'tops': ndarray|None, 'bottoms': ndarray|None, 'shoes': ndarray|None}
          'annotated': frame with bounding boxes drawn
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        annotated = frame.copy()
        crops = {"tops": None, "bottoms": None, "shoes": None}

        if not results.pose_landmarks:
            return {"persons": [], "crops": crops, "annotated": annotated}

        lm = results.pose_landmarks.landmark

        visible = [
            (lm[i].x * w, lm[i].y * h)
            for i in range(len(lm))
            if lm[i].visibility >= CONF_THRESHOLD
        ]
        if not visible:
            return {"persons": [], "crops": crops, "annotated": annotated}

        x1 = max(0, int(min(x for x, _ in visible)))
        y1 = max(0, int(min(y for _, y in visible)))
        x2 = min(w, int(max(x for x, _ in visible)))
        y2 = min(h, int(max(y for _, y in visible)))

        confidence = float(
            sum(lm[i].visibility for i in KEY_LANDMARKS) / len(KEY_LANDMARKS)
        )
        persons = [{"bbox": [x1, y1, x2, y2], "confidence": confidence}]

        box_h = y2 - y1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"person {confidence:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        REGION_COLOR = {"tops": (255, 100, 0), "bottoms": (0, 100, 255), "shoes": (180, 0, 255)}
        for cat, (r0, r1) in SPLIT.items():
            cy1 = max(0, int(y1 + box_h * r0))
            cy2 = min(h, int(y1 + box_h * r1))
            cx1, cx2 = max(0, x1), min(w, x2)
            if cy2 > cy1 and cx2 > cx1:
                crops[cat] = frame[cy1:cy2, cx1:cx2].copy()
                color = REGION_COLOR[cat]
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 1)
                cv2.putText(annotated, cat, (cx1 + 4, cy1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return {"persons": persons, "crops": crops, "annotated": annotated}
