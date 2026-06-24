import threading
import time
import cv2
import numpy as np


class Camera:
    def __init__(self, device_index=0):
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._fps = 0.0
        self._last_ts = time.time()

    def start(self):
        self._running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                now = time.time()
                elapsed = now - self._last_ts
                self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self._last_ts = now
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._cap.release()
