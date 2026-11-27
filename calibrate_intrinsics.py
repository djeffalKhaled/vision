"""OPTIONAL: estimate the camera intrinsics / lens distortion (chessboard).

Print a chessboard (9x6 inner corners), run this script, press SPACE to grab
a view when the corners are detected (>= 10 views from different angles),
press q to finish.  Result: intrinsics.npz (loaded automatically by camera.py).
Uses cv2.findChessboardCorners / cv2.calibrateCamera (Zhang's method).
"""
import numpy as np
import cv2
from config import Config

PATTERN = (9, 6)
SQUARE = 1.0   # unit does not matter for K and distortion

cfg = Config.load()
cap = cv2.VideoCapture(cfg.camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
obj_pts, img_pts, size = [], [], None
while True:
    ok, f = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    size = g.shape[::-1]
    found, c = cv2.findChessboardCorners(g, PATTERN)
    vis = f.copy()
    if found:
        cv2.drawChessboardCorners(vis, PATTERN, c, found)
    cv2.putText(vis, f"views: {len(obj_pts)}  SPACE=grab  q=finish", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Intrinsics", vis)
    k = cv2.waitKey(30) & 0xFF
    if k == ord(" ") and found:
        obj_pts.append(objp)
        img_pts.append(c)
    if k == ord("q"):
        break
if len(obj_pts) >= 5:
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    np.savez(cfg.intrinsics_path, K=K, dist=dist)
    print("RMS reprojection error (px):", rms)
    print("K =\n", K, "\ndist =", dist.ravel())
else:
    print("Not enough views (need >= 5).")
