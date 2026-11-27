"""Whiteboard UI state machine: set_pointer(pos | None) + click('left' | 'right') in, render() -> BGR image out.
LEFT click: press the button under the pointer / pen down<->up on a canvas / grab<->drop an object.
RIGHT click: pen up if drawing, drop if dragging, otherwise back one level.
Pages: main -> lesson | draw | images -> viewer | objects | settings -> validate | quit"""
import time
from collections import defaultdict, namedtuple
from functools import lru_cache, partial
import numpy as np
import lesson
from graphics import (rgb, fill_rect, blend_rect, rect_outline, fill_circle, circle_outline, draw_line, put_text,
                      text_size, resize_nn, render_view)
from sample_images import load_images

# ---- constants (colours are BGR) ---------------------------------------------------------------
BG, INK, GREY, RED = (252, 252, 250), (40, 40, 40), (140, 140, 140), rgb(200, 30, 30)
C_BAR, C_BTN, C_ACC = rgb(235, 238, 245), rgb(60, 90, 170), rgb(20, 70, 160)
CARD_COLORS = [rgb(40, 90, 170), rgb(200, 90, 30), rgb(150, 50, 140), rgb(30, 130, 150), rgb(90, 90, 110), rgb(170, 40, 40)]
PEN_COLORS = [("Black", rgb(25, 25, 25)), ("Red", rgb(220, 30, 30)), ("Blue", rgb(30, 60, 220)), ("Orange", rgb(240, 130, 0))]
PEN_THICK, ERASER_SIZE, TOP, MODES = [("Thin", 3), ("Medium", 8), ("Thick", 16)], 40, 80, ("marker", "finger", "skin")
OBJECTS = [dict(kind="square", x=250, y=330, s=90, c=rgb(40, 90, 170)), dict(kind="circle", x=620, y=380, s=55, c=rgb(220, 120, 20)),
           dict(kind="bar", x=980, y=330, s=120, c=rgb(150, 50, 140))]
HALF = {"square": (1.0, 1.0), "bar": (2.2, 0.35)}      # half width / height of an object, in units of its size "s"
put_center = partial(put_text, center=True)


def clamp(v, lo, hi): return min(max(v, lo), hi)


def grid_rects(cols, rows, x0, y0, x1, y1, gap=24):
    """Row-major (x0, y0, x1, y1) cells filling the area."""
    w, h = (x1 - x0 - gap * (cols - 1)) / cols, (y1 - y0 - gap * (rows - 1)) / rows
    return [tuple(int(v) for v in (x0 + c * (w + gap), y0 + r * (h + gap), x0 + c * (w + gap) + w, y0 + r * (h + gap) + h))
            for r in range(rows) for c in range(cols)]


def hit_object(o, p):
    if o["kind"] == "circle": return np.hypot(p[0] - o["x"], p[1] - o["y"]) <= o["s"]
    kw, kh = HALF[o["kind"]]
    return abs(p[0] - o["x"]) <= kw * o["s"] and abs(p[1] - o["y"]) <= kh * o["s"]


class Btn(namedtuple("Btn", "rect label fn selected color thumb swatch", defaults=(False, C_BTN, None, None))):
    """thumb = image name drawn inside the button, swatch = colour disc instead of text."""
    def contains(self, p): return self.rect[0] <= p[0] < self.rect[2] and self.rect[1] <= p[1] < self.rect[3]


class DrawingLayer:
    """Transparent drawing surface (RGB + mask). Strokes keep their segments so undo can redraw."""
    def __init__(self, w, h):
        self.rgb, self.mask = np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool)
        self.strokes, self.last = [], None             # stroke = ((color, thick, erase), [(prev_point | None, point), ...])

    def begin(self, color, thick, erase): self.strokes.append(((color, thick, erase), [])); self.last = None
    def break_(self): self.last = None
    def composite(self, img): img[self.mask] = self.rgb[self.mask]
    def empty(self): return not self.mask.any()
    def undo(self): self.strokes[-1:] = []; self._redraw()          # drops the last stroke, if any
    def clear(self): self.strokes.clear(); self._redraw()

    def add_point(self, p):
        p = (float(p[0]), float(p[1]))
        self.strokes[-1][1].append((self.last, p)); self._draw(self.strokes[-1][0], self.last, p); self.last = p

    def _draw(self, style, a, b):
        color, thick, erase = style
        draw_line(self.rgb, b if a is None else a, b, color, thick, self.mask, erase)

    def _redraw(self):
        self.last, self.rgb[:], self.mask[:] = None, 0, False
        for style, segs in self.strokes:
            for a, b in segs: self._draw(style, a, b)


class App:
    def __init__(self, cfg):
        self.cfg, self.W, self.H = cfg, cfg.board_w, cfg.board_h
        self.flags = {"show_pointer": True, "show_status": True, "debug": False, "tracking": cfg.tracking}
        self.requests, self.image_items = [], load_images(cfg.images_dir)      # requests: 'calibrate' | 'quit' | 'debug'
        self.pointer, self.pointer_fresh, self.flash, self.last_event = None, False, None, "-"
        self.reset()

    # ---- state & navigation ------------------------------------------------------------------
    def _set(self, **attrs): self.__dict__.update(attrs)

    def reset(self):
        self._set(page="main", history=[], canvas=DrawingLayer(self.W, self.H), annot=defaultdict(partial(DrawingLayer, self.W, self.H)),
                  pen_color=0, pen_thick=1, eraser=False, pen_down=False, lesson_idx=0, annotate=False,
                  cur_image=None, zoom=1.0, vc=(0.0, 0.0), view_scale=1.0, val=None)
        self._reset_objects()

    def _reset_objects(self): self._set(objects=[dict(o) for o in OBJECTS], grabbed=None)
    def _go(self, page, history): self._lift_pen(); self._set(grabbed=None, page=page, history=history)
    def goto(self, page): self._go(page, self.history + [self.page])
    def back(self): self._go(self.history[-1] if self.history else self.page, self.history[:-1])
    def home(self): self._go("main", [])

    def _lift_pen(self):
        self.pen_down = False
        if (layer := self.active_layer()) is not None: layer.break_()

    def active_layer(self):
        """Layer the pen writes on right now (annotation layers are created on demand), else None."""
        if self.page == "draw": return self.canvas
        if self.page == "lesson" and self.annotate: return self.annot[self.lesson_idx]

    def _layer_op(self, op):             # Undo / Clear; on a slide this also works while annotation mode is off
        layer = self.annot[self.lesson_idx] if self.page == "lesson" else self.active_layer()
        if layer is not None: getattr(layer, op)()

    # ---- inputs --------------------------------------------------------------------------------
    def set_pointer(self, pos, fresh=True):
        """pos: (x, y) in board pixels, or None when the pointer is lost."""
        p = self.pointer = None if pos is None else (float(pos[0]), float(pos[1]))
        self.pointer_fresh, layer, g = fresh, self.active_layer(), self.grabbed
        if self.pen_down and layer is not None:
            if p is not None and p[1] > TOP: layer.add_point(p)
            else: layer.break_()
        if g is not None and p is not None and self.page == "objects":
            self.objects[g["i"]].update(x=clamp(p[0] + g["dx"], 0, self.W), y=clamp(p[1] + g["dy"], TOP, self.H))

    def click(self, kind):
        self.last_event, p = kind, self.pointer
        if p is None: return
        self.flash = (p[0], p[1], kind, time.time())
        if kind == "right":                                         # pen up / drop / back
            if self.pen_down: self._lift_pen()
            elif self.grabbed is not None: self.grabbed = None
            elif self.page != "main": self.back()
        elif hit := next((b for b in self.buttons() if b.contains(p)), None): hit.fn()
        elif p[1] > TOP: self._canvas_click(p)

    def _canvas_click(self, p):
        layer, pg, v = self.active_layer(), self.page, self.val
        if layer is not None:                                       # draw / annotate: pen down <-> up
            if self.pen_down: return self._lift_pen()
            self.pen_down = True
            layer.begin(PEN_COLORS[self.pen_color][1], ERASER_SIZE if self.eraser else PEN_THICK[self.pen_thick][1], self.eraser)
            layer.add_point(p)
        elif pg == "viewer" and self.cur_image is not None:         # centre the view on the click
            ih, iw = self.cur_image.shape[:2]
            self.vc = (clamp(self.vc[0] + (p[0] - self.W / 2) / self.view_scale, 0, iw),
                       clamp(self.vc[1] + (p[1] - TOP - (self.H - TOP) / 2) / self.view_scale, 0, ih))
        elif pg == "objects":                                       # drop, or grab the top-most object under the pointer
            i = next((i for i in reversed(range(len(self.objects))) if hit_object(self.objects[i], p)), None)
            if self.grabbed is None and i is not None:
                o = self.objects.pop(i); self.objects.append(o)     # bring to front
                self.grabbed = {"i": len(self.objects) - 1, "dx": o["x"] - p[0], "dy": o["y"] - p[1]}
            else: self.grabbed = None
        elif pg == "validate" and v is not None and v["idx"] < len(v["targets"]):
            t = v["targets"][v["idx"]]
            v["errs"].append(float(np.hypot(p[0] - t[0], p[1] - t[1]))); v["idx"] += 1

    # ---- actions -------------------------------------------------------------------------------
    @lru_cache(maxsize=None)                                        # one App per process, so caching on self is fine
    def _image(self, i):
        img = self.image_items[i][1]()                              # full-size image, loaded once (grey if it fails)
        return np.full((480, 640, 3), 200, np.uint8) if img is None else img

    @lru_cache(maxsize=None)
    def _thumb(self, name, w, h):                                   # image `name` scaled to fit w x h
        img = self._image([n for n, _ in self.image_items].index(name))
        s = min(w / img.shape[1], h / img.shape[0])
        return resize_nn(img, max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))

    def _zoom_reset(self): self._set(zoom=1.0, vc=(self.cur_image.shape[1] / 2.0, self.cur_image.shape[0] / 2.0))
    def _zoom_by(self, f): self.zoom = clamp(self.zoom * f, 1.0, 16.0)
    def _step(self, d): self.lesson_idx = clamp(self.lesson_idx + d, 0, lesson.N_SLIDES - 1)
    def _toggle_annotate(self): self._lift_pen(); self.annotate = not self.annotate

    def _open_image(self, i):
        self.cur_image = self._image(i)
        self._zoom_reset(); self.goto("viewer")

    def _toggle(self, key):
        self.flags[key] = not self.flags[key]
        if key == "debug": self.requests.append("debug")

    def toggle_tracking(self):
        self.flags["tracking"] = self.cfg.tracking = MODES[(MODES.index(self.flags["tracking"]) + 1) % len(MODES)]

    def _start_validate(self, goto_page=True):
        fr = (0.1, 0.5, 0.9)                                        # targets stay below the toolbar: a click never hits a button
        self.val = {"targets": [(self.W * fx, TOP + (self.H - TOP) * fy) for fy in fr for fx in fr], "idx": 0, "errs": []}
        if goto_page: self.goto("validate")

    # ---- buttons per page ----------------------------------------------------------------------
    def _bar(self, *items):
        """Toolbar from (label, fn[, Btn kwargs]) items; a Home button always comes first."""
        items = [("Home", self.home), *items]
        rects = grid_rects(len(items), 1, 10, 10, self.W - 10, TOP - 10, 8)
        return [Btn(r, lab, fn, **(kw[0] if kw else {})) for r, (lab, fn, *kw) in zip(rects, items)]

    def _cards(self, items, cols, x0, x1):                          # big coloured menu buttons on a grid
        rects = grid_rects(cols, 2, x0, 190, x1, self.H - 60)
        return [Btn(r, lab, fn, color=CARD_COLORS[i % 6]) for i, (r, (lab, fn)) in enumerate(zip(rects, items))]

    def buttons(self):
        pg, f, v, W = self.page, self.flags, self.val, self.W
        uc = [("Undo", partial(self._layer_op, "undo")), ("Clear", partial(self._layer_op, "clear"))]
        if pg == "main":
            pages = [(p.capitalize(), partial(self.goto, p)) for p in ("lesson", "draw", "images", "objects", "settings")]
            return self._cards(pages + [("Quit", partial(self.requests.append, "quit"))], 3, 80, W - 80)
        if pg == "lesson":
            return self._bar(("< Prev", partial(self._step, -1)), ("Next >", partial(self._step, 1)),
                             ("Annotate", self._toggle_annotate, {"selected": self.annotate}), *uc,
                             ("Color", lambda: self._set(pen_color=(self.pen_color + 1) % len(PEN_COLORS)),
                              {"color": PEN_COLORS[self.pen_color][1]}))
        if pg == "draw":
            pen = not self.eraser
            return self._bar(*[("", partial(self._set, pen_color=i, eraser=False), {"swatch": c, "selected": self.pen_color == i and pen})
                               for i, (_, c) in enumerate(PEN_COLORS)],
                             *[(n, partial(self._set, pen_thick=i, eraser=False), {"selected": self.pen_thick == i and pen})
                               for i, (n, _) in enumerate(PEN_THICK)],
                             ("Eraser", lambda: self._set(eraser=not self.eraser), {"selected": self.eraser}), *uc)
        if pg == "images":
            rects = grid_rects(3, 2, 60, TOP + 30, W - 60, self.H - 50)
            return self._bar() + [Btn(r, n, partial(self._open_image, i), color=C_BAR, thumb=n)
                                  for i, (r, (n, _)) in enumerate(zip(rects, self.image_items))]
        if pg == "viewer":
            return self._bar(("Gallery", self.back), ("Zoom +", partial(self._zoom_by, 1.5)),
                             ("Zoom -", partial(self._zoom_by, 1 / 1.5)), ("Reset", self._zoom_reset))
        if pg == "objects": return self._bar(("Reset objects", self._reset_objects))
        if pg == "settings":
            toggles = [(f"{n}: {'ON' if f[k] else 'OFF'}", partial(self._toggle, k))
                       for n, k in (("Pointer", "show_pointer"), ("Status bar", "show_status"), ("Camera view", "debug"))]
            items = [(f"Tracking: {f['tracking']}", self.toggle_tracking), *toggles,
                     ("Calibrate camera", partial(self.requests.append, "calibrate")),
                     ("Check accuracy", self._start_validate), ("Reset application", self.reset)]
            return self._bar() + self._cards(items, 4, 50, W - 50)
        if pg == "validate" and v is not None and v["idx"] >= len(v["targets"]):
            return self._bar(("Retry", partial(self._start_validate, goto_page=False)))
        return self._bar()

    # ---- rendering -----------------------------------------------------------------------------
    def _draw_button(self, img, b):
        x0, y0, x1, y1 = b.rect
        w, h, cx, cy = x1 - x0, y1 - y0, (x0 + x1) // 2, (y0 + y1) // 2
        col = np.array(b.color, float)
        if self.pointer is not None and b.contains(self.pointer): col = col * 0.75 + 255 * 0.25     # hover: lighten
        fill_rect(img, x0, y0, x1, y1, col.astype(np.uint8))
        if b.thumb is not None:
            th = self._thumb(b.thumb, w - 20, h - 60)
            tx = x0 + (w - th.shape[1]) // 2
            img[y0 + 10:y0 + 10 + th.shape[0], tx:tx + th.shape[1]] = th
            put_center(img, b.label, cx, y1 - 18, 0.8, INK, 2)
        elif b.swatch is not None: fill_circle(img, cx, cy, 18, b.swatch)
        else:
            big = h > 150
            scale, thick = (1.4, 3) if big else (0.7, 2)
            scale *= min(1.0, (w - 14) / max(text_size(b.label, scale, thick)[0], 1))                # shrink long labels
            put_center(img, b.label, cx, cy + int(10 * scale + (8 if big else 0)), scale, (255, 255, 255), thick)
        if b.selected: rect_outline(img, x0, y0, x1, y1, (0, 0, 0), 4)

    def render(self):
        W, H, pg = self.W, self.H, self.page
        img = np.empty((H, W, 3), np.uint8)
        img[:] = BG
        if pg != "main": fill_rect(img, 0, 0, W, TOP, C_BAR)
        if pg == "main":                                            # page content, under the drawing layers
            put_center(img, "SMART INTERACTIVE WHITEBOARD", W // 2, 100, 1.6, C_ACC, 4)
            put_center(img, "Point with the marker, NEXT = left click, PREVIOUS = right click", W // 2, 150, 0.8, GREY, 2)
        elif pg == "lesson":
            lesson.draw_slide(img, self.lesson_idx, TOP)
            put_text(img, f"{self.lesson_idx + 1}/{lesson.N_SLIDES}", W - 110, TOP + 60, 0.9, GREY, 2)
        elif pg == "draw":
            rect_outline(img, 0, TOP, W, H, (200, 200, 200), 3)
            if self.canvas.empty() and not self.pen_down:
                put_center(img, "Left click = pen down / up    Right click = pen up / back", W // 2, H // 2, 0.9, (190, 190, 190), 2)
        elif pg == "viewer" and self.cur_image is not None:
            img[TOP:], self.view_scale = render_view(self.cur_image, self.zoom, *self.vc, W, H - TOP, (225, 225, 225))
            for col, th in (((255, 255, 255), 4), (INK, 2)):        # zoom label with a white halo
                put_text(img, f"zoom x{self.zoom:.2f}", W - 190, H - 30, 0.8, col, th)
        elif pg == "objects":
            put_text(img, "Left click: grab / drop an object. Right click: drop.", 40, TOP + 50, 0.9, GREY, 2)
            for i, o in enumerate(self.objects): self._draw_object(img, o, self.grabbed is not None and self.grabbed["i"] == i)
        elif pg == "validate": self._draw_validate(img)
        elif pg == "settings": put_center(img, "Settings", W // 2, 150, 1.2, C_ACC, 3)
        elif pg == "images": put_text(img, "Choose an image", 200, TOP + 22, 0.7, GREY, 2)
        if pg == "draw": self.canvas.composite(img)                 # drawing layers, then buttons / status / pointer on top
        elif pg == "lesson" and self.lesson_idx in self.annot: self.annot[self.lesson_idx].composite(img)
        for b in self.buttons(): self._draw_button(img, b)
        if self.flags["show_status"]: self._draw_status(img)
        self._draw_pointer(img)
        return img

    def _draw_object(self, img, o, grabbed):
        x, y, s = int(o["x"]), int(o["y"]), int(o["s"])
        if o["kind"] == "circle": fill_circle(img, x, y, s, o["c"])
        else:
            w, h = (int(k * s) for k in HALF[o["kind"]])
            fill_rect(img, x - w, y - h, x + w, y + h, o["c"])
        if grabbed: circle_outline(img, x, y, s + 14, (0, 0, 0), 3)

    def _draw_validate(self, img):
        if (v := self.val) is None: return
        n, i = len(v["targets"]), v["idx"]
        if i < n:
            for x, y in v["targets"][:i]: fill_circle(img, x, y, 6, (150, 200, 150))                 # already measured
            x, y = v["targets"][i]                                                                   # current target: red cross
            circle_outline(img, x, y, 26, RED, 4)
            draw_line(img, (x - 40, y), (x + 40, y), RED, 2); draw_line(img, (x, y - 40), (x, y + 40), RED, 2)
            put_center(img, f"Target {i + 1}/{n}: put the marker on the red cross, then NEXT (left click)", self.W // 2, TOP + 40, 0.75, INK, 2)
        else:
            e = np.array(v["errs"])
            put_center(img, "Accuracy of the calibration", self.W // 2, 220, 1.2, C_ACC, 3)
            for k, t in enumerate([f"mean error : {e.mean():6.1f} board px  ({100 * e.mean() / self.W:.2f}% of board width)",
                                   f"RMS error  : {np.sqrt((e ** 2).mean()):6.1f} board px", f"max error  : {e.max():6.1f} board px"]):
                put_text(img, t, 300, 320 + 60 * k, 0.95, INK, 2)

    def _draw_status(self, img):
        p = self.pointer
        ptxt = "pointer: lost" if p is None else f"pointer: ({p[0]:.0f}, {p[1]:.0f})"
        txt = f"{self.page} | {ptxt} | last click: {self.last_event}" + ("  | PEN DOWN" if self.pen_down else "")
        blend_rect(img, 0, self.H - 26, 560, self.H, (255, 255, 255), 0.7)
        put_text(img, txt, 8, self.H - 8, 0.5, (60, 60, 60), 1)

    def _draw_pointer(self, img):
        if self.flash is not None:                                  # click ripple
            fx, fy, kind, t0 = self.flash
            dt = time.time() - t0
            if dt < 0.35: circle_outline(img, fx, fy, 18 + int(60 * dt), rgb(240, 130, 0) if kind == "left" else RED, 5)   # never green
            else: self.flash = None
        if not self.flags["show_pointer"] or self.pointer is None: return
        x, y = self.pointer
        col = rgb(210, 30, 30) if self.pen_down else rgb(20, 70, 200)
        if not self.pointer_fresh: col = (150, 150, 150)
        circle_outline(img, x, y, 16, (255, 255, 255), 6); circle_outline(img, x, y, 16, col, 3); fill_circle(img, x, y, 4, col)
