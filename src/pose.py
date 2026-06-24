"""
Human pose tracker for Smart Fitting Room.
Triggers recommend when required joints stay inside the zone for HOLD_SEC seconds.
After trigger, zone is disabled until reset() is called externally.
"""
import time

import cv2
import mediapipe as mp
import numpy as np

ZONE = (0.22, 0.03, 0.78, 0.97)
HOLD_SEC = 3.0

REQUIRED = [0, 11, 12, 23, 24]

CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

PROC_W, PROC_H = 320, 240


class PoseTracker:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._in_zone_since: float | None = None
        self.disabled = False   # set True after trigger; cleared by reset()

    def reset(self):
        """Re-enable zone counting (call when user presses cancel)."""
        self.disabled = False
        self._in_zone_since = None

    def process(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (PROC_W, PROC_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        joints = None
        in_zone = False
        rw = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            joints = [(int(p.x * w), int(p.y * h), float(p.visibility)) for p in lm]

            zx1, zy1, zx2, zy2 = ZONE
            visible_idx = [i for i in range(len(joints)) if joints[i][2] >= 0.5]
            key_present = all(joints[i][2] >= 0.5 for i in REQUIRED)
            all_in_zone = all(
                zx1 <= joints[i][0] / w <= zx2 and zy1 <= joints[i][1] / h <= zy2
                for i in REQUIRED if joints[i][2] >= 0.5
            )
            in_zone = key_present and all_in_zone

            rh = None
            for idx in [20, 18, 22, 16]:  # index → pinky → thumb → wrist
                if len(joints) > idx and joints[idx][2] >= 0.4:
                    rh = joints[idx][:2]
                    break
            rw = rh

        triggered = False
        hold_pct = 0.0

        if not self.disabled:
            now = time.time()
            if in_zone:
                if self._in_zone_since is None:
                    self._in_zone_since = now
            else:
                self._in_zone_since = None

            elapsed = (now - self._in_zone_since) if self._in_zone_since else 0.0
            hold_pct = min(1.0, elapsed / HOLD_SEC) if self._in_zone_since else 0.0
            triggered = hold_pct >= 1.0

            if triggered:
                self.disabled = True
                self._in_zone_since = None

        return {
            "joints":   joints,
            "in_zone":  in_zone and not self.disabled,
            "hold_pct": hold_pct,
            "triggered": triggered,
            "disabled": self.disabled,
            "rw":       list(rw) if rw else None,
        }

    def draw_overlay(self, frame: np.ndarray, state: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = ZONE
        px1, py1 = int(zx1 * w), int(zy1 * h)
        px2, py2 = int(zx2 * w), int(zy2 * h)

        hold_pct = state.get("hold_pct", 0.0)
        in_zone  = state.get("in_zone", False)
        disabled = state.get("disabled", False)

        if disabled:
            color = (60, 60, 60)
            thickness = 1
        elif hold_pct > 0:
            g = int(80 + 175 * hold_pct)
            r = int(180 * (1 - hold_pct))
            color = (r, g, 80)
            thickness = 2 + int(hold_pct * 3)
        elif in_zone:
            color = (0, 200, 80)
            thickness = 2
        else:
            color = (160, 160, 160)
            thickness = 2

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, thickness)

        if not disabled and hold_pct > 0:
            bar_w = int((px2 - px1) * hold_pct)
            cv2.rectangle(frame, (px1, py2 - 8), (px1 + bar_w, py2), color, -1)
            secs_left = max(0.0, HOLD_SEC * (1 - hold_pct))
            cv2.putText(frame, f"{secs_left:.1f}s",
                        (px1 + 6, py2 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rw = state.get("rw")
        if rw:
            cv2.circle(frame, tuple(rw), 12, (0, 255, 180), 2)
            cv2.circle(frame, tuple(rw), 4,  (0, 255, 180), -1)

        return frame
