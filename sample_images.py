"""Procedurally generated images (NumPy only) + loader for images/ folder.

The application therefore works even without any image file; drop .jpg/.png
files in images/ to show your own pictures (decoded with cv2.imread).
"""
import os
import numpy as np
import cv2
from graphics import hsv_to_bgr, put_text


def colour_wheel(w=640, h=480):
    ys, xs = np.mgrid[0:h, 0:w]
    dx, dy = xs - w / 2, ys - h / 2
    r = np.sqrt(dx ** 2 + dy ** 2) / (min(w, h) / 2)
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    img = hsv_to_bgr(ang, np.clip(r, 0, 1), 1.0)
    img[r > 1.0] = 255
    put_text(img, "HSV colour wheel (hue = angle)", 150, 30, 0.7, (40, 40, 40), 2)
    return img


def mandelbrot(w=640, h=480, n=60):
    x = np.linspace(-2.2, 1.0, w)
    y = np.linspace(-1.2, 1.2, h)
    c = x[None, :] + 1j * y[:, None]
    z = np.zeros_like(c)
    it = np.zeros(c.shape, int)
    for _ in range(n):
        m = np.abs(z) <= 2
        z[m] = z[m] ** 2 + c[m]
        it += m
    return hsv_to_bgr(it * 6.0, 0.9, (it < n) * 1.0)


def test_pattern(w=640, h=480):
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 255, np.uint8)
    chk = ((xs // 40 + ys // 40) % 2).astype(bool)
    img[chk] = (225, 225, 225)
    img[::20, :] = (180, 180, 180)          # fine lines: visible when zooming
    img[:, ::20] = (180, 180, 180)
    img[h // 2 - 1:h // 2 + 1, :] = (0, 0, 200)
    img[:, w // 2 - 1:w // 2 + 1] = (0, 0, 200)
    put_text(img, "Zoom test pattern", w // 2, 40, 0.9, (150, 60, 0), 2, center=True)
    return img


def load_images(folder):
    """List of (name, loader) -- images are loaded lazily and cached by the app."""
    items = [("Colour wheel", colour_wheel), ("Mandelbrot", mandelbrot),
             ("Test pattern", test_pattern)]
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                path = os.path.join(folder, fn)
                items.append((os.path.splitext(fn)[0],
                              lambda p=path: cv2.imread(p)))
    return items[:6]
