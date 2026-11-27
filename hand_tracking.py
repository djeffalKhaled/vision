"""Index-finger tracking with MediaPipe Hands (21 hand landmarks).

The pointer is landmark 8 = tip of the index finger.

MediaPipe is a *predefined* hand-landmark model: say so (and justify it) in
the report -- the rest of the pipeline (calibration, homography, filtering,
interface) is still our own code.

Two MediaPipe APIs exist; both are supported:
  * legacy  mp.solutions.hands   (older mediapipe versions, model bundled);
  * Tasks   HandLandmarker       (recent versions).  It needs the model file
            hand_landmarker.task, downloaded automatically on the first run
            into models/ (internet needed once).  If the download fails,
            download it manually from cfg.hand_model_url and put it at
            cfg.hand_model_path.
"""
import os
import time
import urllib.request

import numpy as np

INDEX_TIP = 8


def ensure_model(path, url):
    """Return the model path, downloading the model once if it is missing."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading the MediaPipe hand model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Could not download the hand model ({e}).\n"
            f"Download it manually:\n  {url}\nand save it as: {path}") from e
    return path


def build_backend(cfg):
    """Return (run, name); run(rgb uint8 HxWx3) -> list of hands, each a list of
    21 (x, y) landmarks normalised to [0, 1]."""
    import mediapipe as mp
    n = int(cfg.hand_max_hands)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=n,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

        def run(rgb):
            res = hands.process(rgb)
            return [[(l.x, l.y) for l in h.landmark] for h in (res.multi_hand_landmarks or [])]
        return run, "mediapipe legacy solutions"

    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    path = ensure_model(cfg.hand_model_path, cfg.hand_model_url)
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO, num_hands=n,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    lm = vision.HandLandmarker.create_from_options(opts)
    state = {"ts": 0}

    def run(rgb):
        state["ts"] = max(int(time.monotonic() * 1000), state["ts"] + 1)   # strictly increasing
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, state["ts"])
        return [[(l.x, l.y) for l in h] for h in res.hand_landmarks]
    return run, "mediapipe tasks HandLandmarker"


def point_in_convex_quad(p, q):
    sgn = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        sgn.append((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return all(s >= 0 for s in sgn) or all(s <= 0 for s in sgn)


class MediaPipeFingerDetector:
    """Same interface as MarkerDetector: detect(frame) -> (x, y) pixel or None."""

    last_mask = None            # no colour mask in this mode (debug view skips it)

    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        if backend is None:
            self._run, self.backend_name = build_backend(cfg)
        else:                                           # injected (tests)
            self._run, self.backend_name = backend, "injected"
        self.last_landmarks = None      # pixel coordinates of the 21 points (debug)
        self._roi = None

    def set_roi_quad(self, corners_img):
        self._roi = None if corners_img is None else np.asarray(corners_img, float)

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[..., ::-1])          # BGR -> RGB
        hands = self._run(rgb)
        self.last_landmarks = None
        best = None
        for lms in hands:
            pts = [(x * w, y * h) for x, y in lms]
            tip = pts[INDEX_TIP]
            inside = self._roi is None or point_in_convex_quad(tip, self._roi)
            if best is None or (inside and not best[2]):
                best = (pts, tip, inside)
        if best is None:
            return None
        self.last_landmarks = best[0]
        return best[1]
