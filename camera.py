"""Camera acquisition (threaded: always returns the most recent frame)."""
import os
import threading
import time
import numpy as np
import cv2


class Camera:
    def __init__(self, cfg):
        idx = cfg.camera_index
        # DirectShow on Windows starts faster and honours the resolution request
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}")
        # optional lens-distortion correction (from calibrate_intrinsics.py)
        self.maps = None
        if os.path.exists(cfg.intrinsics_path):
            d = np.load(cfg.intrinsics_path)
            K, dist = d["K"], d["dist"]
            ok, f = self.cap.read()
            if ok:
                h, w = f.shape[:2]
                self.maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
        self._frame = None
        self._lock = threading.Lock()
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.maps is not None:
                f = cv2.remap(f, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
            with self._lock:
                self._frame = f

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._run = False
        self._t.join(timeout=1.0)
        self.cap.release()
