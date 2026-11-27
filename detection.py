"""Pointer detection: colour-marker segmentation, written with NumPy.

Pipeline (one camera frame -> one measurement):
  1. sub-sample the frame (stride) for speed;
  2. BGR -> HSV conversion (own implementation);
  3. thresholding on hue / saturation / value  -> binary mask;
  4. morphological opening (erosion then dilation, own implementation)
     to remove isolated noisy pixels;
  5. robust localisation: median of mask pixels, keep pixels close to it
     (rejects far-away false positives), centroid of the kept pixels;
  6. temporal filtering (PointerFilter) on board coordinates.

Why a colour marker?  A saturated coloured tip (green sticker / LED on the
remote) is far more reliable under a projector than skin colour, which is
confused with the projected content and with the user's arm/face.
Limits: strong ambient light, projected content with a similar hue, marker
hidden by the hand, motion blur.  The UI therefore avoids saturated green.
"""
import numpy as np


# ----------------------------------------------------------------------------
def bgr_to_hsv(img):
    """uint8 BGR (H,W,3) -> h in [0,360), s and v in [0,1] (float32)."""
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = v - mn
    s = np.where(v > 1e-6, d / np.maximum(v, 1e-6), 0.0)
    dd = np.maximum(d, 1e-6)
    h = np.zeros_like(v)
    is_r = (d > 0) & (v == r)
    is_g = (d > 0) & (v == g) & ~is_r
    is_b = (d > 0) & ~is_r & ~is_g
    h = np.where(is_r, (60.0 * ((g - b) / dd)) % 360.0, h)
    h = np.where(is_g, 60.0 * ((b - r) / dd + 2.0), h)
    h = np.where(is_b, 60.0 * ((r - g) / dd + 4.0), h)
    return h, s, v


def threshold_marker(img, h_min, h_max, s_min, v_min):
    """Binary mask of pixels whose HSV lies in the marker range."""
    h, s, v = bgr_to_hsv(img)
    h_min, h_max = h_min % 360.0, h_max % 360.0
    if h_min <= h_max:
        hue_ok = (h >= h_min) & (h <= h_max)
    else:                                   # range wraps around 0 (red)
        hue_ok = (h >= h_min) | (h <= h_max)
    return hue_ok & (s >= s_min) & (v >= v_min)


# ---- morphology (3x3 square structuring element, built from shifted views) --
def erode(mask, r=1):
    p = np.pad(mask, r, constant_values=False)
    out = np.ones_like(mask)
    H, W = mask.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out &= p[dy:dy + H, dx:dx + W]
    return out


def dilate(mask, r=1):
    p = np.pad(mask, r, constant_values=False)
    out = np.zeros_like(mask)
    H, W = mask.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + H, dx:dx + W]
    return out


def opening(mask, r=1):
    return dilate(erode(mask, r), r)


# ---- localisation -------------------------------------------------------------
def locate_blob(mask, min_area=12, radius=30.0, iters=2):
    """Robust centroid of the dominant blob.  Returns (x, y, area) or None."""
    ys, xs = np.nonzero(mask)
    if xs.size < min_area:
        return None
    cx, cy = float(np.median(xs)), float(np.median(ys))
    for _ in range(iters):
        keep = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        if keep.sum() < min_area:
            return None
        cx, cy = float(xs[keep].mean()), float(ys[keep].mean())
    return cx, cy, int(keep.sum())


class MarkerDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_mask = None

    def set_roi_quad(self, corners_img):    # same interface as FingerDetector
        pass

    def detect(self, frame):
        """frame: BGR image.  Returns (x, y) in full-resolution image pixels or None."""
        c = self.cfg
        st = max(1, int(c.detect_stride))
        small = frame[::st, ::st]
        mask = threshold_marker(small, c.h_min, c.h_max, c.s_min, c.v_min)
        mask = opening(mask, 1)
        self.last_mask = mask
        res = locate_blob(mask, c.min_area, radius=40.0 / st)
        if res is None:
            return None
        x, y, _ = res
        return (x * st, y * st)   # index i of frame[::st] is pixel i*st

    def mask_fullres(self, shape):
        """Last mask upsampled to the frame size (for the debug view)."""
        st = max(1, int(self.cfg.detect_stride))
        m = np.repeat(np.repeat(self.last_mask, st, 0), st, 1)
        out = np.zeros(shape[:2], bool)
        h, w = min(out.shape[0], m.shape[0]), min(out.shape[1], m.shape[1])
        out[:h, :w] = m[:h, :w]
        return out


# ----------------------------------------------------------------------------
class PointerFilter:
    """Speed-adaptive exponential smoothing + dead zone + loss tolerance.

    * slow motion  -> small alpha  : strong smoothing, no jitter;
    * fast motion  -> large alpha  : low latency;
    * |change| < dead zone -> position frozen (hand tremor);
    * detection lost for <= hold frames -> last position kept (flicker).
    """

    def __init__(self, min_alpha=0.25, max_alpha=0.85, speed_ref=120.0,
                 dead_zone=3.0, hold=8):
        self.min_a, self.max_a = min_alpha, max_alpha
        self.speed_ref, self.dead, self.hold = speed_ref, dead_zone, hold
        self.pos = None
        self.lost = 0

    def reset(self):
        self.pos, self.lost = None, 0

    def update(self, meas):
        """meas: (x, y) or None.  Returns (position or None, is_fresh)."""
        if meas is None:
            self.lost += 1
            if self.lost > self.hold:
                self.pos = None
            return self.pos, False
        self.lost = 0
        m = np.asarray(meas, float)
        if self.pos is None:
            self.pos = m
            return self.pos, True
        dist = float(np.hypot(*(m - self.pos)))
        if dist < self.dead:
            return self.pos, True
        a = self.min_a + (self.max_a - self.min_a) * min(dist / self.speed_ref, 1.0)
        self.pos = self.pos + a * (m - self.pos)
        return self.pos, True


# ============================================================================
# Index-finger tracking (skin segmentation + fingertip = farthest point of hand)
# ============================================================================
def bgr_to_ycrcb(img):
    """uint8 BGR -> (Y, Cr, Cb) float32 in 0..255 (ITU-R BT.601, own code)."""
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def skin_mask(img, cr_rng, cb_rng, y_min):
    y, cr, cb = bgr_to_ycrcb(img)
    return ((cr >= cr_rng[0]) & (cr <= cr_rng[1]) &
            (cb >= cb_rng[0]) & (cb <= cb_rng[1]) & (y >= y_min))


def connected_components(mask, min_area=1):
    """4-connected components by flood fill.  Returns a list of (ys, xs) arrays."""
    H, W = mask.shape
    flat = bytearray(mask.astype(np.uint8).ravel().tobytes())
    comps = []
    for start in np.flatnonzero(mask.ravel()).tolist():
        if not flat[start]:
            continue
        flat[start] = 0
        stack, comp = [start], []
        while stack:
            i = stack.pop()
            comp.append(i)
            x = i % W
            if x > 0 and flat[i - 1]:
                flat[i - 1] = 0; stack.append(i - 1)
            if x < W - 1 and flat[i + 1]:
                flat[i + 1] = 0; stack.append(i + 1)
            if i >= W and flat[i - W]:
                flat[i - W] = 0; stack.append(i - W)
            if i < (H - 1) * W and flat[i + W]:
                flat[i + W] = 0; stack.append(i + W)
        if len(comp) >= min_area:
            a = np.array(comp)
            comps.append((a // W, a % W))
    return comps


def quad_mask(corners, shape, stride=1):
    """Boolean mask (at 1/stride resolution) of the inside of a convex quadrilateral."""
    h, w = -(-shape[0] // stride), -(-shape[1] // stride)   # ceil: same size as frame[::stride, ::stride]
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs * stride, ys * stride
    sign = None
    ok = np.ones((h, w), bool)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        if sign is None:
            sign = np.sign(((b[0] - a[0]) * (corners[(i + 2) % 4][1] - a[1]) -
                            (b[1] - a[1]) * (corners[(i + 2) % 4][0] - a[0])))
        ok &= cross * sign >= 0
    return ok


class FingerDetector:
    """Index-finger tracker.

    1. skin segmentation: colour thresholds on Cr/Cb (+ Y floor), robust to
       brightness changes because chrominance is separated from luminance;
    2. opening to remove speckle; restriction to the board area (removes the
       user's face and background);
    3. flood-fill components -> keep the largest one (the hand/arm);
    4. arm base = where the blob meets the image / board-area border;
    5. fingertip = blob pixels farthest from the base (average of the extreme
       few pixels for stability).  Works when the hand points at the board
       with the index finger extended; other fingers should be folded.
    Limits: skin-like projected colours, skin tones outside the thresholds
    (tune in config), long sleeves (no arm base -> centroid fallback),
    several people in the view.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_mask = None
        self.stride = max(1, int(cfg.finger_stride))
        self._roi_corners = None
        self._roi = None
        self._roi_key = None

    def set_roi_quad(self, corners_img):
        self._roi_corners = None if corners_img is None else np.asarray(corners_img, float)
        self._roi_key = None

    def _get_roi(self, shape):
        if self._roi_corners is None:
            return None
        key = (shape[:2], self.stride)
        if self._roi_key != key:
            self._roi = quad_mask(self._roi_corners, shape, self.stride)
            self._roi_key = key
        return self._roi

    def detect(self, frame):
        c = self.cfg
        st = self.stride = max(1, int(c.finger_stride))
        small = frame[::st, ::st]
        mask = skin_mask(small, c.skin_cr, c.skin_cb, c.skin_y_min)
        roi = self._get_roi(frame.shape)
        if roi is not None and roi.shape == mask.shape:
            mask &= roi
        mask = opening(mask, 1)
        self.last_mask = mask
        comps = connected_components(mask, c.finger_min_area)
        if not comps:
            return None
        ys, xs = max(comps, key=lambda t: len(t[0]))
        H, W = mask.shape
        # ---- arm base: blob pixels on the image border or on the board-area border
        edge = np.zeros((H, W), bool)
        edge[:2, :] = edge[-2:, :] = True
        edge[:, :2] = edge[:, -2:] = True
        if roi is not None and roi.shape == mask.shape:
            edge |= roi & ~erode(roi, 2)
        on_edge = edge[ys, xs]
        if on_edge.any():
            bx, by = xs[on_edge].mean(), ys[on_edge].mean()
        else:
            bx, by = xs.mean(), ys.mean()
        d = np.hypot(xs - bx, ys - by)
        tip = d >= d.max() - 2.0
        tx, ty = xs[tip].mean(), ys[tip].mean()
        return (tx * st, ty * st)

    def mask_fullres(self, shape):
        st = self.stride
        m = np.repeat(np.repeat(self.last_mask, st, 0), st, 1)
        out = np.zeros(shape[:2], bool)
        h, w = min(out.shape[0], m.shape[0]), min(out.shape[1], m.shape[1])
        out[:h, :w] = m[:h, :w]
        return out
