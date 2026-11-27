"""Headless tests (no camera, no GUI):  python tests/test_core.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from config import Config
from calibration import compute_homography, apply_homography, BoardMapper, order_corners
from detection import (MarkerDetector, FingerDetector, PointerFilter, bgr_to_hsv, erode,
                       dilate, opening, connected_components, quad_mask)
from remote_input import RemoteInput
from graphics import fill_circle, draw_line, hsv_to_bgr, render_view
from interaction import App, TOP


def test_homography_roundtrip():
    rs = np.random.RandomState(0)
    Ht = np.array([[1.1, 0.12, 40], [-0.08, 0.95, 25], [1e-4, 2e-4, 1.0]])
    src = rs.rand(4, 2) * [640, 480]
    dst = apply_homography(Ht, src)
    H = compute_homography(src, dst)
    pts = rs.rand(50, 2) * [640, 480]
    err = np.abs(apply_homography(H, pts) - apply_homography(Ht, pts)).max()
    assert err < 1e-6, err


def test_hsv_matches_opencv():
    import cv2
    rs = np.random.RandomState(1)
    img = rs.randint(0, 256, (50, 50, 3)).astype(np.uint8)
    h, s, v = bgr_to_hsv(img)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(float)
    dh = np.abs(h - ref[..., 0] * 360 / 255)
    dh = np.minimum(dh, 360 - dh)          # hue is circular
    gray = s < 0.05                        # hue undefined for greys
    assert dh[~gray].max() < 2.5, dh[~gray].max()
    assert np.abs(s * 255 - ref[..., 1]).max() < 1.5
    assert np.abs(v * 255 - ref[..., 2]).max() < 1.5


def test_morphology():
    m = np.zeros((20, 20), bool); m[5:12, 5:12] = True; m[0, 0] = True
    o = opening(m, 1)
    assert not o[0, 0] and o[8, 8]


def test_full_pipeline_synthetic():
    """Fake camera looking at the board with a perspective, green marker at a known place."""
    cfg = Config()
    Ht = np.array([[0.45, 0.06, 90], [-0.03, 0.42, 60], [2e-4, 1e-4, 1.0]])   # board -> image
    corners_b = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], float)
    corners_i = apply_homography(Ht, corners_b) + np.random.RandomState(2).randn(4, 2) * 0.4
    mapper = BoardMapper.from_corners(corners_i[[2, 0, 3, 1]], 1280, 720)  # shuffled order
    det = MarkerDetector(cfg)
    errs = []
    for tb in [(100, 100), (640, 360), (1180, 650), (300, 600)]:
        frame = np.full((480, 640, 3), 90, np.uint8)
        ti = apply_homography(Ht, tb)[0]
        fill_circle(frame, ti[0], ti[1], 6, (40, 220, 60))               # green marker
        fill_circle(frame, 20, 20, 2, (40, 220, 60))                     # tiny noise blob
        m = det.detect(frame)
        assert m is not None
        b = mapper.to_board(m)
        errs.append(np.hypot(*(b - np.array(tb))))
    assert max(errs) < 6, errs
    # no marker -> None
    assert det.detect(np.full((480, 640, 3), 90, np.uint8)) is None


def test_filter():
    f = PointerFilter(hold=3)
    rs = np.random.RandomState(0)
    raw = [(500 + rs.randn() * 1.5, 300 + rs.randn() * 1.5) for _ in range(50)]
    out = np.array([f.update(r)[0] for r in raw])
    assert out[10:].std(axis=0).max() < 0.7 * np.array(raw).std(axis=0).min()
    for _ in range(3):
        assert f.update(None)[0] is not None
    assert f.update(None)[0] is None


def click_at(app, x, y, kind="left"):
    app.set_pointer((x, y)); app.click(kind)


def button_center(app, label):
    for b in app.buttons():
        if b.label == label:
            x0, y0, x1, y1 = b.rect
            return (x0 + x1) / 2, (y0 + y1) / 2
    raise KeyError(label)


def test_app_flow():
    app = App(Config())
    app.render()
    click_at(app, *button_center(app, "Draw")); assert app.page == "draw"
    click_at(app, *button_center(app, "Red")) if False else None
    # pick 2nd colour (swatch buttons have empty labels -> by index)
    sw = [b for b in app.buttons() if b.swatch is not None]
    click_at(app, *[(sw[1].rect[0] + sw[1].rect[2]) / 2, (sw[1].rect[1] + sw[1].rect[3]) / 2]); assert app.pen_color == 1
    click_at(app, *button_center(app, "Thick")); assert app.pen_thick == 2
    click_at(app, 200, 300)                       # pen down
    assert app.pen_down
    for x in range(200, 600, 10):
        app.set_pointer((x, 300 + 40 * np.sin(x / 50)))
    click_at(app, 600, 300)                       # pen up
    assert not app.pen_down and app.canvas.mask.sum() > 1000
    img = app.render()
    click_at(app, *button_center(app, "Undo")); assert app.canvas.empty()
    click_at(app, 10, 10, "right"); assert app.page == "main"     # right click = back (pen is up)
    # images + zoom
    click_at(app, *button_center(app, "Images")); assert app.page == "images"
    click_at(app, *button_center(app, "Test pattern")); assert app.page == "viewer"
    click_at(app, *button_center(app, "Zoom +")); assert abs(app.zoom - 1.5) < 1e-9
    app.render(); click_at(app, 900, 300)         # recentre
    assert app.vc[0] > 320
    click_at(app, *button_center(app, "Reset")); assert app.zoom == 1.0
    app.render()
    click_at(app, 5, 5, "right"); assert app.page == "images"
    click_at(app, 5, 5, "right"); assert app.page == "main"
    # objects drag & drop
    click_at(app, *button_center(app, "Objects"))
    o = app.objects[0]; x0, y0 = o["x"], o["y"]
    click_at(app, x0, y0); assert app.grabbed is not None
    app.set_pointer((x0 + 100, y0 + 50)); click_at(app, x0 + 100, y0 + 50)
    assert app.grabbed is None and abs(app.objects[-1]["x"] - (x0 + 100)) < 1
    app.render()
    app.home()
    # lesson + annotation
    click_at(app, *button_center(app, "Lesson"))
    for i in range(5):
        app.lesson_idx = i; app.render()
    click_at(app, *button_center(app, "Annotate")); assert app.annotate
    click_at(app, 300, 400); app.set_pointer((500, 450)); click_at(app, 500, 450)
    assert app.annot[app.lesson_idx].mask.any()
    app.home()
    # settings / validation
    click_at(app, *button_center(app, "Settings"))
    click_at(app, *button_center(app, "Check accuracy")); assert app.page == "validate"
    for t in list(app.val["targets"]):
        app.render(); click_at(app, t[0] + 3, t[1] - 4)
    app.render(); assert max(app.val["errs"]) < 6
    click_at(app, *button_center(app, "Retry")); assert app.page == "validate" and app.val["idx"] == 0
    app.home()
    click_at(app, *button_center(app, "Quit")); assert "quit" in app.requests
    # pointer lost never crashes
    app.set_pointer(None); app.click("left"); app.render()


def test_render_view():
    img = np.zeros((100, 200, 3), np.uint8); img[:, :100] = 255
    v, s = render_view(img, 2.0, 100, 50, 400, 200, (7, 7, 7))
    assert v.shape == (200, 400, 3) and abs(s - 4.0) < 1e-9
    assert v[100, 150, 0] == 255 and v[100, 250, 0] == 0


SKIN = (120, 150, 200)   # BGR skin-like colour


def _hand(frame, base, tip, thick=22):
    """Arm from `base` to `tip` plus a palm disc, in skin colour."""
    draw_line(frame, base, tip, SKIN, thick)
    mid = (base[0] + 0.8 * (tip[0] - base[0]), base[1] + 0.8 * (tip[1] - base[1]))
    fill_circle(frame, mid[0], mid[1], 22, SKIN)
    draw_line(frame, mid, tip, SKIN, 8)                       # index finger


def test_connected_components():
    m = np.zeros((30, 30), bool); m[2:6, 2:6] = True; m[20:28, 10:25] = True
    comps = connected_components(m, 5)
    assert sorted(len(c[0]) for c in comps) == [16, 120]


def test_finger_tracking():
    cfg = Config(); det = FingerDetector(cfg)
    cases = [((320, 479), (330, 120)),     # arm from the bottom, pointing up
             ((639, 400), (250, 200)),     # arm from the right side
             ((0, 300), (400, 150))]       # arm from the left side
    for base, tip in cases:
        frame = np.full((480, 640, 3), 70, np.uint8)
        _hand(frame, base, tip)
        fill_circle(frame, 560, 80, 30, SKIN)                 # smaller "face" decoy blob
        m = det.detect(frame)
        assert m is not None and np.hypot(m[0] - tip[0], m[1] - tip[1]) < 12, (base, tip, m)
    assert det.detect(np.full((480, 640, 3), 70, np.uint8)) is None


def test_finger_roi_removes_decoy():
    cfg = Config(); det = FingerDetector(cfg)
    quad = np.array([[50, 40], [600, 40], [600, 460], [50, 460]], float)
    det.set_roi_quad(quad)
    frame = np.full((480, 640, 3), 70, np.uint8)
    _hand(frame, (320, 459), (330, 150))
    fill_circle(frame, 20, 20, 60, SKIN)                      # big blob OUTSIDE the board area
    m = det.detect(frame)
    assert m is not None and abs(m[0] - 330) < 8 and abs(m[1] - 150) < 8, m
    assert quad_mask(quad, (480, 640), 1)[240, 300] and not quad_mask(quad, (480, 640), 1)[10, 10]


def test_keyboard_arrows():
    cfg = Config(); cfg.debounce_s = 0
    r = RemoteInput(cfg)
    assert r.poll(2555904) == ["left"] and r.poll(2424832) == ["right"]     # Right=Next, Left=Prev
    assert r.poll(2228224) == ["left"] and r.poll(2162688) == ["right"]     # PgDn / PgUp
    cfg.arrows_literal = True
    assert r.poll(2424832) == ["left"] and r.poll(2555904) == ["right"]     # literal
    assert r.poll(-1) == [] and r.poll(ord("a")) == []


def test_settings_tracking_toggle():
    app = App(Config()); app.render()
    click_at(app, *button_center(app, "Settings"))
    for before, after in [("marker", "finger"), ("finger", "skin"), ("skin", "marker")]:
        app.render(); click_at(app, *button_center(app, f"Tracking: {before}"))
        assert app.flags["tracking"] == after


def _fake_hand(tip_xy):
    pts = [(0.5, 0.9)] * 21
    pts[8] = tip_xy
    return pts


def test_mediapipe_wrapper():
    from hand_tracking import MediaPipeFingerDetector, point_in_convex_quad
    seen = {}

    def backend(rgb):
        seen["shape"] = rgb.shape; seen["px"] = rgb[0, 0].tolist()
        return [_fake_hand((0.25, 0.5))]
    det = MediaPipeFingerDetector(Config(), backend=backend)
    frame = np.zeros((480, 640, 3), np.uint8); frame[0, 0] = (10, 20, 30)      # BGR
    x, y = det.detect(frame)
    assert (x, y) == (160.0, 240.0)                       # landmark 8, scaled to pixels
    assert seen["px"] == [30, 20, 10] and seen["shape"] == (480, 640, 3)      # RGB order
    assert len(det.last_landmarks) == 21
    # two hands: the one whose index tip is inside the board area wins
    det2 = MediaPipeFingerDetector(Config(), backend=lambda rgb: [_fake_hand((0.05, 0.05)), _fake_hand((0.6, 0.6))])
    det2.set_roi_quad([[200, 150], [600, 150], [600, 450], [200, 450]])
    x, y = det2.detect(frame)
    assert abs(x - 384) < 1e-6 and abs(y - 288) < 1e-6
    # no hand
    assert MediaPipeFingerDetector(Config(), backend=lambda rgb: []).detect(frame) is None
    assert point_in_convex_quad((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)])


def test_mediapipe_tasks_api_present():
    """Real-API smoke test (model file itself cannot be downloaded in the sandbox)."""
    try:
        from mediapipe.tasks.python import vision
    except ImportError:
        print("  (mediapipe not installed: skipped)"); return
    assert hasattr(vision, "HandLandmarker") and hasattr(vision, "RunningMode")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
