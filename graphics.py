"""Drawing primitives implemented with NumPy (no cv2.line / cv2.circle ...).

All images are uint8 arrays of shape (H, W, 3) in BGR order.  The only OpenCV
calls in this module are text rendering (putText / getTextSize): writing a
vector font by hand is outside the scope of the project.
"""
import numpy as np
import cv2


def rgb(r, g, b):
    """Convenience: write colours as (R, G, B), return OpenCV's (B, G, R)."""
    return (b, g, r)


# ----------------------------------------------------------------------------
# Rectangles / circles
# ----------------------------------------------------------------------------
def fill_rect(img, x0, y0, x1, y1, color):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = color


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        reg = img[y0:y1, x0:x1].astype(np.float32)
        reg = reg * (1.0 - alpha) + np.array(color, np.float32) * alpha
        img[y0:y1, x0:x1] = reg.astype(np.uint8)


def rect_outline(img, x0, y0, x1, y1, color, t=2):
    fill_rect(img, x0, y0, x1, y0 + t, color)
    fill_rect(img, x0, y1 - t, x1, y1, color)
    fill_rect(img, x0, y0, x0 + t, y1, color)
    fill_rect(img, x1 - t, y0, x1, y1, color)


def _circle_mask(img, cx, cy, r):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return x0, x1, y0, y1, d2


def fill_circle(img, cx, cy, r, color):
    res = _circle_mask(img, cx, cy, r)
    if res is None:
        return
    x0, x1, y0, y1, d2 = res
    img[y0:y1, x0:x1][d2 <= r * r] = color


def circle_outline(img, cx, cy, r, color, t=2):
    res = _circle_mask(img, cx, cy, r)
    if res is None:
        return
    x0, x1, y0, y1, d2 = res
    r_in = max(r - t, 0)
    img[y0:y1, x0:x1][(d2 <= r * r) & (d2 >= r_in * r_in)] = color


# ----------------------------------------------------------------------------
# Lines (sampled DDA + disc stamping => thick lines with round caps)
# ----------------------------------------------------------------------------
_DISK_CACHE = {}


def disk_offsets(radius):
    """(dy, dx) offsets of all pixels inside a disc of the given radius."""
    radius = int(radius)
    if radius not in _DISK_CACHE:
        ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        m = xs ** 2 + ys ** 2 <= radius * radius + radius
        _DISK_CACHE[radius] = (ys[m], xs[m])
    return _DISK_CACHE[radius]


def draw_line(img, p0, p1, color, thickness=1, mask=None, erase=False):
    """Draw a segment p0->p1.

    img   : image to write to.
    mask  : optional boolean array (H, W); pixels touched are set to True
            (or to False when erase=True, in which case img is left untouched).
    """
    h, w = img.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, n)).astype(np.int64)
    oy, ox = disk_offsets(max(int(thickness) // 2, 0))
    Y = (ys[:, None] + oy[None, :]).ravel()
    X = (xs[:, None] + ox[None, :]).ravel()
    ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    X, Y = X[ok], Y[ok]
    if erase:
        if mask is not None:
            mask[Y, X] = False
        return
    img[Y, X] = color
    if mask is not None:
        mask[Y, X] = True


def draw_polyline(img, pts, color, thickness=1):
    for a, b in zip(pts[:-1], pts[1:]):
        draw_line(img, a, b, color, thickness)


# ----------------------------------------------------------------------------
# Resampling (nearest neighbour, written with index arrays)
# ----------------------------------------------------------------------------
def resize_nn(img, new_w, new_h):
    ih, iw = img.shape[:2]
    xs = np.minimum((np.arange(new_w) * iw / new_w).astype(int), iw - 1)
    ys = np.minimum((np.arange(new_h) * ih / new_h).astype(int), ih - 1)
    return img[ys[:, None], xs[None, :]]


def render_view(img, zoom, cx, cy, vw, vh, bg):
    """Render a zoomed/panned view of `img` into a (vh, vw) viewport.

    The image is first fitted in the viewport (scale s0) then multiplied by
    `zoom`; (cx, cy) is the image point shown at the viewport centre.
    Returns (view, scale) where scale = viewport pixels per image pixel.
    """
    ih, iw = img.shape[:2]
    s = min(vw / iw, vh / ih) * zoom
    xs = cx + (np.arange(vw) - vw / 2 + 0.5) / s
    ys = cy + (np.arange(vh) - vh / 2 + 0.5) / s
    xi = np.floor(xs).astype(int)
    yi = np.floor(ys).astype(int)
    inside = ((yi >= 0) & (yi < ih))[:, None] & ((xi >= 0) & (xi < iw))[None, :]
    sub = img[np.clip(yi, 0, ih - 1)[:, None], np.clip(xi, 0, iw - 1)[None, :]]
    out = np.empty((vh, vw, 3), np.uint8)
    out[:] = bg
    out[inside] = sub[inside]
    return out, s


# ----------------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------------
def hsv_to_bgr(h, s, v):
    """Vectorised HSV (h in degrees, s,v in 0..1) -> BGR uint8."""
    h = np.asarray(h, np.float64) % 360.0
    s = np.asarray(s, np.float64) * np.ones_like(h)
    v = np.asarray(v, np.float64) * np.ones_like(h)
    c = v * s
    hp = h / 60.0
    x = c * (1 - np.abs(hp % 2 - 1))
    m = v - c
    z = np.zeros_like(c)
    sec = hp.astype(int) % 6
    r = np.choose(sec, [c, x, z, z, x, c])
    g = np.choose(sec, [x, c, c, x, z, z])
    b = np.choose(sec, [z, z, x, c, c, x])
    out = np.stack([b + m, g + m, r + m], axis=-1)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Text (OpenCV Hershey font)
# ----------------------------------------------------------------------------
def text_size(text, scale=1.0, thick=2):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    return w, h


def put_text(img, text, x, y, scale=1.0, color=(30, 30, 30), thick=2, center=False):
    """Draw text with its baseline at y (x is the left edge, or the centre)."""
    if center:
        w, _ = text_size(text, scale, thick)
        x = x - w // 2
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)
