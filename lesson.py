"""Content of the demo lesson: 5 slides drawn with the graphics primitives."""
import numpy as np
from graphics import (put_text, fill_rect, rect_outline, draw_line, fill_circle,
                      hsv_to_bgr, draw_polyline, rgb)

N_SLIDES = 5
INK = (40, 40, 40)
ACC = rgb(20, 70, 160)
RED = rgb(200, 40, 40)
GREY = rgb(120, 120, 120)

TITLES = ["Welcome", "1. Seeing the pointer", "2. From pixels to the board",
          "3. Stabilising the measurement", "Your turn"]


def _bullets(img, lines, x, y, scale=0.85, step=48):
    for i, t in enumerate(lines):
        fill_circle(img, x, y + i * step - 8, 5, ACC)
        put_text(img, t, x + 20, y + i * step, scale, INK, 2)


def draw_slide(img, idx, top):
    """Draw slide `idx` below the toolbar (y >= top)."""
    W = img.shape[1]
    put_text(img, TITLES[idx], 40, top + 60, 1.4, ACC, 3)
    draw_line(img, (40, top + 78), (W - 40, top + 78), ACC, 2)
    y0 = top + 140
    if idx == 0:
        put_text(img, "How does the board see your pointer?", 40, y0, 1.1, INK, 2)
        _bullets(img, ["A camera watches the board and finds the coloured marker",
                       "A homography converts image pixels into board coordinates",
                       "Remote button NEXT     = left click  (select / draw)",
                       "Remote button PREVIOUS = right click (back / pen up)"],
                 60, y0 + 70)
    elif idx == 1:
        _bullets(img, ["Convert each pixel from BGR to HSV",
                       "Keep the pixels whose hue, saturation, value fit the marker",
                       "Clean the mask (opening), then take a robust centroid"], 60, y0)
        hue = np.arange(360)
        bar = hsv_to_bgr(hue, 1.0, 1.0)                     # (360, 3)
        bx, by, bh = 100, y0 + 230, 70
        for i in range(360):
            fill_rect(img, bx + i * 3, by, bx + i * 3 + 3, by + bh, bar[i])
        rect_outline(img, bx + 85 * 3, by - 8, bx + 160 * 3, by + bh + 8, INK, 3)
        put_text(img, "marker range (hue 85-160 deg)", bx + 85 * 3 - 20, by + bh + 45, 0.7, INK, 2)
        put_text(img, "hue 0", bx, by - 18, 0.6, GREY, 1)
        put_text(img, "360", bx + 340 * 3, by - 18, 0.6, GREY, 1)
    elif idx == 2:
        quad = [(120, y0 + 40), (480, y0 + 10), (540, y0 + 270), (90, y0 + 300)]
        for i in range(4):
            draw_line(img, quad[i], quad[(i + 1) % 4], RED, 4)
        put_text(img, "camera image", 190, y0 + 345, 0.8, RED, 2)
        rx0, ry0, rx1, ry1 = 760, y0 + 30, 1160, y0 + 300
        rect_outline(img, rx0, ry0, rx1, ry1, ACC, 4)
        put_text(img, "board (0,0) -> (W,H)", 790, y0 + 345, 0.8, ACC, 2)
        draw_line(img, (570, y0 + 160), (730, y0 + 160), INK, 4)
        draw_line(img, (730, y0 + 160), (705, y0 + 145), INK, 4)
        draw_line(img, (730, y0 + 160), (705, y0 + 175), INK, 4)
        put_text(img, "H (3x3)", 590, y0 + 140, 0.8, INK, 2)
        put_text(img, "s [X Y 1]^T = H [x y 1]^T   (4 corners -> DLT + SVD)",
                 120, y0 + 400, 0.85, INK, 2)
    elif idx == 3:
        _bullets(img, ["Raw detection jitters (noise, hand tremor, flicker)",
                       "Adaptive smoothing: slow = smooth, fast = reactive",
                       "Dead zone + a few frames of memory when detection drops"], 60, y0)
        rs = np.random.RandomState(1)
        t = np.arange(0, 60)
        true = 330 + 90 * np.sin(t / 9.0)
        noisy = true + rs.randn(60) * 14
        filt = np.zeros(60)
        filt[0] = noisy[0]
        for i in range(1, 60):
            a = 0.3
            filt[i] = filt[i - 1] + a * (noisy[i] - filt[i - 1])
        ox, oy = 100, y0 + 40
        pn = [(ox + int(i * 17), int(oy + (noisy[i] - 200))) for i in range(60)]
        pf = [(ox + int(i * 17), int(oy + (filt[i] - 200))) for i in range(60)]
        draw_polyline(img, pn, rgb(230, 130, 130), 2)
        draw_polyline(img, pf, ACC, 4)
        put_text(img, "raw", 1130, oy + 180, 0.7, rgb(200, 90, 90), 2)
        put_text(img, "filtered", 1100, oy + 215, 0.7, ACC, 2)
    else:
        _bullets(img, ["Press Annotate, then draw on this slide with NEXT",
                       "Use Draw (Home menu) for a free canvas",
                       "Use Images to explore pictures with zoom",
                       "Use Objects to drag and drop shapes",
                       "PREVIOUS (right click) always goes back"], 60, y0)
