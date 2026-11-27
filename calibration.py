"""Calibration and image -> board coordinate mapping.

Coordinate systems
------------------
* image   : pixels of the camera frame, origin top-left, x right, y down.
* board   : pixels of the projected interface, origin = upper-left corner of
            the board, X right, Y down  (the "world" frame of the project).

The board is a plane, so image and board are related by a 3x3 homography H
(8 degrees of freedom):    s * [X, Y, 1]^T = H * [x, y, 1]^T
H is estimated from the 4 board corners clicked in the camera image with the
Direct Linear Transform (DLT) solved by SVD, with Hartley normalisation.
"""
import json
import numpy as np


# ----------------------------------------------------------------------------
# Homography (DLT)
# ----------------------------------------------------------------------------
def apply_homography(H, pts):
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def _normalisation(pts):
    """Similarity T: centroid -> origin, mean distance -> sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def compute_homography(src, dst):
    """Homography H such that dst ~ H @ src (needs >= 4 point pairs)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    assert len(src) >= 4 and len(src) == len(dst)
    Ts, Td = _normalisation(src), _normalisation(dst)
    s = apply_homography(Ts, src)
    d = apply_homography(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def order_corners(pts):
    """Sort 4 points as TL, TR, BR, BL (image axes: y points down)."""
    pts = np.asarray(pts, np.float64)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]])


def is_convex_quad(q):
    """True if the 4 ordered points form a convex, non-degenerate quadrilateral."""
    sign = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        sign.append(cross)
    sign = np.array(sign)
    return bool(np.all(sign > 1e-6) or np.all(sign < -1e-6))


class BoardMapper:
    """Maps camera-image points to board points (and back)."""

    def __init__(self, H_img2board, corners_img, board_w, board_h):
        self.H = np.asarray(H_img2board, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.corners_img = np.asarray(corners_img, np.float64)
        self.board_w, self.board_h = board_w, board_h

    @staticmethod
    def from_corners(corners_img, board_w, board_h):
        """corners_img: 4 clicked points in any order."""
        q = order_corners(corners_img)
        if not is_convex_quad(q):
            raise ValueError("The 4 points do not form a valid quadrilateral")
        board = np.array([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]], float)
        return BoardMapper(compute_homography(q, board), q, board_w, board_h)

    def to_board(self, pt):
        return apply_homography(self.H, pt)[0]

    def to_image(self, pt):
        return apply_homography(self.Hinv, pt)[0]

    def corner_error(self):
        """Fit residual of the 4 calibration corners (board pixels)."""
        board = np.array([[0, 0], [self.board_w, 0],
                          [self.board_w, self.board_h], [0, self.board_h]], float)
        e = apply_homography(self.H, self.corners_img) - board
        return float(np.sqrt((e ** 2).sum(axis=1)).max())

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"H": self.H.tolist(), "corners_img": self.corners_img.tolist(),
                       "board_w": self.board_w, "board_h": self.board_h}, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        return BoardMapper(np.array(d["H"]), np.array(d["corners_img"]),
                           d["board_w"], d["board_h"])


# ----------------------------------------------------------------------------
# Interactive calibration (OpenCV window used only for display and mouse)
# ----------------------------------------------------------------------------
def calibrate_interactive(get_frame, board_w, board_h, window="Calibration"):
    """Let the user click the 4 board corners on the live camera image.

    Keys: Enter = accept, u = undo last point, r = reset, Esc = cancel.
    Returns a BoardMapper or None.
    """
    import cv2
    from graphics import draw_line, fill_circle, put_text

    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    mapper, msg = None, "Click the 4 corners of the board (any order)"
    while True:
        frame = get_frame()
        if frame is None:
            if cv2.waitKey(30) == 27:
                break
            continue
        vis = frame.copy()
        for p in pts:
            fill_circle(vis, p[0], p[1], 5, (0, 0, 255))
        if len(pts) == 4:
            try:
                q = order_corners(pts)
                for i in range(4):
                    draw_line(vis, q[i], q[(i + 1) % 4], (0, 255, 255), 2)
                msg = "Enter = accept | u = undo | r = reset | Esc = cancel"
            except Exception:
                pass
        put_text(vis, msg, 10, 25, 0.6, (0, 255, 255), 2)
        put_text(vis, f"points: {len(pts)}/4", 10, 50, 0.6, (255, 255, 255), 1)
        cv2.imshow(window, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord("r"):
            pts.clear()
        if k == ord("u") and pts:
            pts.pop()
        if k in (13, 10) and len(pts) == 4:
            try:
                mapper = BoardMapper.from_corners(pts, board_w, board_h)
                break
            except ValueError as e:
                msg = str(e)
                pts.clear()
    cv2.destroyWindow(window)
    return mapper
