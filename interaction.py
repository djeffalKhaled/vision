"""Interface and interaction logic of the whiteboard (no camera, no OpenCV window).

The class App is a state machine:
    inputs  : set_pointer(position or None)   -- continuous, from the tracker
              click('left' | 'right')         -- from the remote buttons
    output  : render() -> BGR image to project on the board

Click semantics (same everywhere):
    LEFT  click (Next button)     : press the button under the pointer;
                                    on a canvas: pen down / pen up toggle;
                                    on an object: grab / drop.
    RIGHT click (Previous button) : pen up if drawing, drop if dragging,
                                    otherwise go back one menu level.

Page tree:
    main --+-- lesson (5 slides + annotation)
           +-- draw   (colours, thicknesses, eraser, undo, clear)
           +-- images (gallery) -- viewer (zoom + / - / reset, click to centre)
           +-- objects (drag & drop)
           +-- settings -- validate (accuracy test of the calibration)
           +-- quit
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import lesson
from graphics import (rgb, fill_rect, blend_rect, rect_outline, fill_circle,
                      circle_outline, draw_line, put_text, text_size, resize_nn,
                      render_view)
from sample_images import load_images

# ---- palette (BGR) ------------------------------------------------------------
BG = (252, 252, 250)
INK = (40, 40, 40)
GREY = (140, 140, 140)
C_BAR = rgb(235, 238, 245)
C_BTN = rgb(60, 90, 170)
C_ACC = rgb(20, 70, 160)
CARD_COLORS = [rgb(40, 90, 170), rgb(200, 90, 30), rgb(150, 50, 140),
               rgb(30, 130, 150), rgb(90, 90, 110), rgb(170, 40, 40)]
PEN_COLORS = [("Black", rgb(25, 25, 25)), ("Red", rgb(220, 30, 30)),
              ("Blue", rgb(30, 60, 220)), ("Orange", rgb(240, 130, 0))]
PEN_THICK = [("Thin", 3), ("Medium", 8), ("Thick", 16)]
ERASER_SIZE = 40
TOP = 80                      # height of the toolbar zone


# ----------------------------------------------------------------------------
@dataclass
class Btn:
    rect: tuple
    label: str
    fn: Callable
    selected: bool = False
    color: tuple = C_BTN
    thumb: Optional[str] = None      # image name to draw inside the button
    swatch: Optional[tuple] = None   # colour disc instead of text

    def contains(self, p):
        x0, y0, x1, y1 = self.rect
        return x0 <= p[0] < x1 and y0 <= p[1] < y1


def row_rects(n, x0, x1, y0, y1, gap=8):
    w = (x1 - x0 - gap * (n - 1)) / n
    return [(int(x0 + i * (w + gap)), y0, int(x0 + i * (w + gap) + w), y1) for i in range(n)]


def grid_rects(cols, rows, x0, y0, x1, y1, gap=24):
    w = (x1 - x0 - gap * (cols - 1)) / cols
    h = (y1 - y0 - gap * (rows - 1)) / rows
    return [(int(x0 + c * (w + gap)), int(y0 + r * (h + gap)),
             int(x0 + c * (w + gap) + w), int(y0 + r * (h + gap) + h))
            for r in range(rows) for c in range(cols)]


# ----------------------------------------------------------------------------
class DrawingLayer:
    """A transparent drawing surface (RGB + boolean mask) with undo."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), bool)
        self.strokes = []
        self.last = None

    def begin(self, color, thick, erase):
        self.strokes.append({"color": color, "thick": thick, "erase": erase, "paths": []})
        self.last = None

    def break_(self):
        self.last = None

    def add_point(self, p):
        s = self.strokes[-1]
        p = (float(p[0]), float(p[1]))
        self._seg(s, self.last, p)
        if self.last is None:
            s["paths"].append([p])
        else:
            s["paths"][-1].append(p)
        self.last = p

    def _seg(self, s, a, b):
        draw_line(self.rgb, a if a is not None else b, b, s["color"], s["thick"],
                  self.mask, s["erase"])

    def _redraw(self):
        self.mask[:] = False
        self.rgb[:] = 0
        for s in self.strokes:
            for path in s["paths"]:
                prev = None
                for p in path:
                    self._seg(s, prev, p)
                    prev = p

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.last = None
            self._redraw()

    def clear(self):
        self.strokes.clear()
        self.last = None
        self._redraw()

    def composite(self, img):
        img[self.mask] = self.rgb[self.mask]

    def empty(self):
        return not self.mask.any()


# ----------------------------------------------------------------------------
class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.H = cfg.board_w, cfg.board_h
        self.flags = {"show_pointer": True, "show_status": True, "debug": False,
                      "tracking": cfg.tracking}
        self.requests = []                    # 'calibrate' | 'quit' | 'debug'
        self.image_items = load_images(cfg.images_dir)
        self._img_cache = {}
        self._thumb_cache = {}
        self.pointer = None
        self.pointer_fresh = False
        self.flash = None
        self.last_event = "-"
        self.reset()

    # ---- state ---------------------------------------------------------------
    def reset(self):
        self.page, self.history = "main", []
        self.canvas = DrawingLayer(self.W, self.H)
        self.annot = {}
        self.pen_color, self.pen_thick, self.eraser = 0, 1, False
        self.pen_down = False
        self.lesson_idx, self.annotate = 0, False
        self.cur_image, self.zoom, self.vc = None, 1.0, (0.0, 0.0)
        self.view_scale = 1.0
        self._init_objects()
        self.grabbed = None
        self.val = None

    def _init_objects(self):
        self.objects = [
            {"kind": "square", "x": 250, "y": 330, "s": 90, "c": rgb(40, 90, 170)},
            {"kind": "circle", "x": 620, "y": 380, "s": 55, "c": rgb(220, 120, 20)},
            {"kind": "bar", "x": 980, "y": 330, "s": 120, "c": rgb(150, 50, 140)},
        ]
        self.home_objects = [dict(o) for o in self.objects]

    def goto(self, page):
        self._lift_pen()
        self.grabbed = None
        self.history.append(self.page)
        self.page = page

    def back(self):
        self._lift_pen()
        self.grabbed = None
        if self.history:
            self.page = self.history.pop()

    def home(self):
        self._lift_pen()
        self.grabbed = None
        self.page, self.history = "main", []

    def _lift_pen(self):
        self.pen_down = False
        layer = self.active_layer()
        if layer is not None:
            layer.break_()

    # ---- layers --------------------------------------------------------------
    def active_layer(self):
        if self.page == "draw":
            return self.canvas
        if self.page == "lesson" and self.annotate:
            if self.lesson_idx not in self.annot:
                self.annot[self.lesson_idx] = DrawingLayer(self.W, self.H)
            return self.annot[self.lesson_idx]
        return None

    # ---- inputs --------------------------------------------------------------
    def set_pointer(self, pos, fresh=True):
        """pos: (x, y) in board pixels, or None when the pointer is lost."""
        self.pointer = None if pos is None else (float(pos[0]), float(pos[1]))
        self.pointer_fresh = fresh
        p = self.pointer
        layer = self.active_layer()
        if self.pen_down and layer is not None:
            if p is not None and p[1] > TOP:
                layer.add_point(p)
            else:
                layer.break_()
        if self.grabbed is not None and p is not None and self.page == "objects":
            o = self.objects[self.grabbed["i"]]
            o["x"] = min(max(p[0] + self.grabbed["dx"], 0), self.W)
            o["y"] = min(max(p[1] + self.grabbed["dy"], TOP), self.H)

    def click(self, kind):
        self.last_event = kind
        if self.pointer is None:
            return
        p = self.pointer
        self.flash = (p[0], p[1], kind, time.time())
        if kind == "right":
            self._right_click()
            return
        for b in self.buttons():
            if b.contains(p):
                b.fn()
                return
        self._canvas_click(p)

    def _right_click(self):
        if self.pen_down:
            self._lift_pen()
        elif self.grabbed is not None:
            self.grabbed = None
        elif self.page != "main":
            self.back()

    def _canvas_click(self, p):
        if p[1] <= TOP:
            return
        layer = self.active_layer()
        if layer is not None:                                  # draw / annotate
            if self.pen_down:
                self._lift_pen()
            else:
                self.pen_down = True
                thick = ERASER_SIZE if self.eraser else PEN_THICK[self.pen_thick][1]
                layer.begin(PEN_COLORS[self.pen_color][1], thick, self.eraser)
                layer.add_point(p)
        elif self.page == "viewer" and self.cur_image is not None:
            ih, iw = self.cur_image.shape[:2]
            s = self.view_scale
            cx = self.vc[0] + (p[0] - self.W / 2) / s
            cy = self.vc[1] + (p[1] - TOP - (self.H - TOP) / 2) / s
            self.vc = (min(max(cx, 0), iw), min(max(cy, 0), ih))
        elif self.page == "objects":
            if self.grabbed is not None:
                self.grabbed = None
            else:
                for i in range(len(self.objects) - 1, -1, -1):
                    o = self.objects[i]
                    if self._hit_object(o, p):
                        self.grabbed = {"i": i, "dx": o["x"] - p[0], "dy": o["y"] - p[1]}
                        self.objects.append(self.objects.pop(i))  # bring to front
                        self.grabbed["i"] = len(self.objects) - 1
                        break
        elif self.page == "validate":
            self._validate_click(p)

    @staticmethod
    def _hit_object(o, p):
        if o["kind"] == "circle":
            return np.hypot(p[0] - o["x"], p[1] - o["y"]) <= o["s"]
        w = o["s"] * (2.2 if o["kind"] == "bar" else 1.0)
        return abs(p[0] - o["x"]) <= w and abs(p[1] - o["y"]) <= (o["s"] * (0.35 if o["kind"] == "bar" else 1.0))

    # ---- actions ---------------------------------------------------------------
    def _open_image(self, i):
        name, loader = self.image_items[i]
        if i not in self._img_cache:
            img = loader()
            self._img_cache[i] = img if img is not None else np.full((480, 640, 3), 200, np.uint8)
        self.cur_image = self._img_cache[i]
        ih, iw = self.cur_image.shape[:2]
        self.zoom, self.vc = 1.0, (iw / 2.0, ih / 2.0)
        self.goto("viewer")

    def _zoom_by(self, f):
        self.zoom = min(max(self.zoom * f, 1.0), 16.0)

    def _zoom_reset(self):
        self.zoom = 1.0
        ih, iw = self.cur_image.shape[:2]
        self.vc = (iw / 2.0, ih / 2.0)

    def _set_pen(self, i):
        self.pen_color, self.eraser = i, False

    def _set_thick(self, i):
        self.pen_thick, self.eraser = i, False

    def _toggle(self, key):
        self.flags[key] = not self.flags[key]
        if key == "debug":
            self.requests.append("debug")

    def toggle_tracking(self):
        modes = ["marker", "finger", "skin"]
        self.flags["tracking"] = modes[(modes.index(self.flags["tracking"]) + 1) % len(modes)]
        self.cfg.tracking = self.flags["tracking"]

    def _start_validate(self):
        fr = [0.1, 0.5, 0.9]
        # targets stay below the toolbar so that a click can never hit a button
        targets = [(self.W * fx, TOP + (self.H - TOP) * fy) for fy in fr for fx in fr]
        self.val = {"targets": targets, "idx": 0, "errs": []}
        self.goto("validate")

    def _validate_click(self, p):
        v = self.val
        if v is None or v["idx"] >= len(v["targets"]):
            return
        t = v["targets"][v["idx"]]
        v["errs"].append(float(np.hypot(p[0] - t[0], p[1] - t[1])))
        v["idx"] += 1

    # ---- buttons per page -------------------------------------------------------
    def _bar(self, items, x0=10):
        """Toolbar from a list of (label, fn, kwargs); Home button is added first."""
        items = [("Home", self.home, {})] + items
        rects = row_rects(len(items), x0, self.W - 10, 10, TOP - 10)
        return [Btn(r, lab, fn, **kw) for r, (lab, fn, kw) in zip(rects, items)]

    def buttons(self):
        pg = self.page
        if pg == "main":
            labels = [("Lesson", lambda: self.goto("lesson")),
                      ("Draw", lambda: self.goto("draw")),
                      ("Images", lambda: self.goto("images")),
                      ("Objects", lambda: self.goto("objects")),
                      ("Settings", lambda: self.goto("settings")),
                      ("Quit", lambda: self.requests.append("quit"))]
            rects = grid_rects(3, 2, 80, 190, self.W - 80, self.H - 60)
            return [Btn(r, l, f, color=CARD_COLORS[i]) for i, (r, (l, f)) in enumerate(zip(rects, labels))]
        if pg == "lesson":
            n = lesson.N_SLIDES
            return self._bar([
                ("< Prev", lambda: setattr(self, "lesson_idx", max(0, self.lesson_idx - 1)), {}),
                ("Next >", lambda: setattr(self, "lesson_idx", min(n - 1, self.lesson_idx + 1)), {}),
                ("Annotate", lambda: (self._lift_pen(), setattr(self, "annotate", not self.annotate)),
                 {"selected": self.annotate}),
                ("Undo", lambda: self._layer_op("undo"), {}),
                ("Clear", lambda: self._layer_op("clear"), {}),
                ("Color", lambda: setattr(self, "pen_color", (self.pen_color + 1) % len(PEN_COLORS)),
                 {"color": PEN_COLORS[self.pen_color][1]})])
        if pg == "draw":
            items = []
            for i, (nm, col) in enumerate(PEN_COLORS):
                items.append(("", lambda i=i: self._set_pen(i),
                              {"swatch": col, "selected": (self.pen_color == i and not self.eraser)}))
            for i, (nm, t) in enumerate(PEN_THICK):
                items.append((nm, lambda i=i: self._set_thick(i),
                              {"selected": (self.pen_thick == i and not self.eraser)}))
            items += [("Eraser", lambda: setattr(self, "eraser", not self.eraser), {"selected": self.eraser}),
                      ("Undo", self.canvas.undo, {}),
                      ("Clear", self.canvas.clear, {})]
            return self._bar(items)
        if pg == "images":
            rects = grid_rects(3, 2, 60, TOP + 30, self.W - 60, self.H - 50)
            bs = self._bar([])
            for i, r in enumerate(rects[:len(self.image_items)]):
                bs.append(Btn(r, self.image_items[i][0], lambda i=i: self._open_image(i),
                              color=C_BAR, thumb=self.image_items[i][0]))
            return bs
        if pg == "viewer":
            return self._bar([("Gallery", self.back, {}),
                              ("Zoom +", lambda: self._zoom_by(1.5), {}),
                              ("Zoom -", lambda: self._zoom_by(1 / 1.5), {}),
                              ("Reset", self._zoom_reset, {})])
        if pg == "objects":
            return self._bar([("Reset objects", lambda: (setattr(self, "objects", [dict(o) for o in self.home_objects]),
                                                         setattr(self, "grabbed", None)), {})])
        if pg == "settings":
            f = self.flags
            labels = [(f"Tracking: {f['tracking']}", self.toggle_tracking),
                      (f"Pointer: {'ON' if f['show_pointer'] else 'OFF'}", lambda: self._toggle("show_pointer")),
                      (f"Status bar: {'ON' if f['show_status'] else 'OFF'}", lambda: self._toggle("show_status")),
                      (f"Camera view: {'ON' if f['debug'] else 'OFF'}", lambda: self._toggle("debug")),
                      ("Calibrate camera", lambda: self.requests.append("calibrate")),
                      ("Check accuracy", self._start_validate),
                      ("Reset application", self.reset)]
            rects = grid_rects(4, 2, 50, 190, self.W - 50, self.H - 60)
            bs = self._bar([])
            return bs + [Btn(r, l, fn, color=CARD_COLORS[i % 6]) for i, (r, (l, fn)) in enumerate(zip(rects, labels))]
        if pg == "validate":
            done = self.val is not None and self.val["idx"] >= len(self.val["targets"])
            items = [("Retry", self._start_validate_again, {})] if done else []
            return self._bar(items)
        return self._bar([])

    def _start_validate_again(self):
        self.history.pop() if self.history else None
        self.page = "settings"
        self._start_validate()

    def _layer_op(self, op):
        layer = self.active_layer()
        if layer is None and self.page == "lesson":
            layer = self.annot.setdefault(self.lesson_idx, DrawingLayer(self.W, self.H))
        if layer is not None:
            getattr(layer, op)()

    # ---- rendering -----------------------------------------------------------------
    def _thumb(self, name, w, h):
        key = (name, w, h)
        if key not in self._thumb_cache:
            i = [n for n, _ in self.image_items].index(name)
            if i not in self._img_cache:
                img = self.image_items[i][1]()
                self._img_cache[i] = img if img is not None else np.full((480, 640, 3), 200, np.uint8)
            img = self._img_cache[i]
            ih, iw = img.shape[:2]
            s = min(w / iw, h / ih)
            self._thumb_cache[key] = resize_nn(img, max(1, int(iw * s)), max(1, int(ih * s)))
        return self._thumb_cache[key]

    def _draw_button(self, img, b):
        x0, y0, x1, y1 = b.rect
        hover = self.pointer is not None and b.contains(self.pointer)
        col = np.array(b.color, float)
        if hover:
            col = col * 0.75 + 255 * 0.25
        fill_rect(img, x0, y0, x1, y1, col.astype(np.uint8))
        if b.thumb is not None:
            th = self._thumb(b.thumb, x1 - x0 - 20, y1 - y0 - 60)
            tx = x0 + ((x1 - x0) - th.shape[1]) // 2
            img[y0 + 10:y0 + 10 + th.shape[0], tx:tx + th.shape[1]] = th
            put_text(img, b.label, (x0 + x1) // 2, y1 - 18, 0.8, INK, 2, center=True)
        elif b.swatch is not None:
            fill_circle(img, (x0 + x1) // 2, (y0 + y1) // 2, 18, b.swatch)
        else:
            big = (y1 - y0) > 150
            scale = 1.4 if big else 0.7
            tw, _ = text_size(b.label, scale, 3 if big else 2)
            scale *= min(1.0, (x1 - x0 - 14) / max(tw, 1))
            put_text(img, b.label, (x0 + x1) // 2, (y0 + y1) // 2 + int(10 * scale + (8 if big else 0)),
                     scale, (255, 255, 255), 3 if big else 2, center=True)
        if b.selected:
            rect_outline(img, x0, y0, x1, y1, (0, 0, 0), 4)

    def render(self):
        img = np.empty((self.H, self.W, 3), np.uint8)
        img[:] = BG
        pg = self.page
        if pg != "main":
            fill_rect(img, 0, 0, self.W, TOP, C_BAR)
        # --- page content under the layers
        if pg == "main":
            put_text(img, "SMART INTERACTIVE WHITEBOARD", self.W // 2, 100, 1.6, C_ACC, 4, center=True)
            put_text(img, "Point with the marker, NEXT = left click, PREVIOUS = right click",
                     self.W // 2, 150, 0.8, GREY, 2, center=True)
        elif pg == "lesson":
            lesson.draw_slide(img, self.lesson_idx, TOP)
            put_text(img, f"{self.lesson_idx + 1}/{lesson.N_SLIDES}", self.W - 110, TOP + 60, 0.9, GREY, 2)
        elif pg == "draw":
            rect_outline(img, 0, TOP, self.W, self.H, (200, 200, 200), 3)
            if self.canvas.empty() and not self.pen_down:
                put_text(img, "Left click = pen down / up    Right click = pen up / back",
                         self.W // 2, self.H // 2, 0.9, (190, 190, 190), 2, center=True)
        elif pg == "viewer" and self.cur_image is not None:
            view, self.view_scale = render_view(self.cur_image, self.zoom, self.vc[0], self.vc[1],
                                                self.W, self.H - TOP, (225, 225, 225))
            img[TOP:] = view
            put_text(img, f"zoom x{self.zoom:.2f}", self.W - 190, self.H - 30, 0.8, (255, 255, 255), 4)
            put_text(img, f"zoom x{self.zoom:.2f}", self.W - 190, self.H - 30, 0.8, INK, 2)
        elif pg == "objects":
            put_text(img, "Left click: grab / drop an object. Right click: drop.", 40, TOP + 50, 0.9, GREY, 2)
            for i, o in enumerate(self.objects):
                self._draw_object(img, o, grabbed=(self.grabbed is not None and self.grabbed["i"] == i))
        elif pg == "validate":
            self._draw_validate(img)
        elif pg == "settings":
            put_text(img, "Settings", self.W // 2, 150, 1.2, C_ACC, 3, center=True)
        elif pg == "images":
            put_text(img, "Choose an image", 200, TOP + 22, 0.7, GREY, 2)
        # --- drawing layers
        layer = self.active_layer()
        if pg == "draw":
            self.canvas.composite(img)
        elif pg == "lesson" and self.lesson_idx in self.annot:
            self.annot[self.lesson_idx].composite(img)
        # --- buttons then pointer
        for b in self.buttons():
            self._draw_button(img, b)
        if self.flags["show_status"]:
            self._draw_status(img)
        self._draw_pointer(img)
        return img

    def _draw_object(self, img, o, grabbed):
        x, y, s, c = int(o["x"]), int(o["y"]), int(o["s"]), o["c"]
        if o["kind"] == "circle":
            fill_circle(img, x, y, s, c)
        elif o["kind"] == "square":
            fill_rect(img, x - s, y - s, x + s, y + s, c)
        else:
            fill_rect(img, x - int(2.2 * s), y - int(0.35 * s), x + int(2.2 * s), y + int(0.35 * s), c)
        if grabbed:
            circle_outline(img, x, y, s + 14, (0, 0, 0), 3)

    def _draw_validate(self, img):
        v = self.val
        if v is None:
            return
        n = len(v["targets"])
        if v["idx"] < n:
            for k, t in enumerate(v["targets"]):
                if k < v["idx"]:
                    fill_circle(img, t[0], t[1], 6, (150, 200, 150))
            t = v["targets"][v["idx"]]
            circle_outline(img, t[0], t[1], 26, rgb(200, 30, 30), 4)
            draw_line(img, (t[0] - 40, t[1]), (t[0] + 40, t[1]), rgb(200, 30, 30), 2)
            draw_line(img, (t[0], t[1] - 40), (t[0], t[1] + 40), rgb(200, 30, 30), 2)
            put_text(img, f"Target {v['idx'] + 1}/{n}: put the marker on the red cross, then NEXT (left click)",
                     self.W // 2, TOP + 40, 0.75, INK, 2, center=True)
        else:
            e = np.array(v["errs"])
            put_text(img, "Accuracy of the calibration", self.W // 2, 220, 1.2, C_ACC, 3, center=True)
            lines = [f"mean error : {e.mean():6.1f} board px  ({100 * e.mean() / self.W:.2f}% of board width)",
                     f"RMS error  : {np.sqrt((e ** 2).mean()):6.1f} board px",
                     f"max error  : {e.max():6.1f} board px"]
            for i, t in enumerate(lines):
                put_text(img, t, 300, 320 + 60 * i, 0.95, INK, 2)

    def _draw_status(self, img):
        p = self.pointer
        ptxt = "pointer: lost" if p is None else f"pointer: ({p[0]:.0f}, {p[1]:.0f})"
        txt = f"{self.page} | {ptxt} | last click: {self.last_event}" + ("  | PEN DOWN" if self.pen_down else "")
        blend_rect(img, 0, self.H - 26, 560, self.H, (255, 255, 255), 0.7)
        put_text(img, txt, 8, self.H - 8, 0.5, (60, 60, 60), 1)

    def _draw_pointer(self, img):
        now = time.time()
        if self.flash is not None:
            fx, fy, kind, t0 = self.flash
            dt = now - t0
            if dt < 0.35:
                col = rgb(240, 130, 0) if kind == "left" else rgb(200, 30, 30)   # never green: it is the marker colour
                circle_outline(img, fx, fy, 18 + int(60 * dt), col, 5)
            else:
                self.flash = None
        if not self.flags["show_pointer"] or self.pointer is None:
            return
        x, y = self.pointer
        col = rgb(210, 30, 30) if self.pen_down else rgb(20, 70, 200)
        if not self.pointer_fresh:
            col = (150, 150, 150)
        circle_outline(img, x, y, 16, (255, 255, 255), 6)
        circle_outline(img, x, y, 16, col, 3)
        fill_circle(img, x, y, 4, col)
