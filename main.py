"""Smart Interactive Whiteboard -- entry point.

    python main.py                 # real use: camera + projector + remote
    python main.py --sim           # no camera: mouse = pointer, mouse buttons / z,x = clicks
    python main.py --fullscreen    # board window in full screen (drag it to the projector first)
    python main.py --log-keys      # print the key codes received (to identify the remote)

Keys (board window):  m = cycle tracking (marker / finger-MediaPipe / skin) | c = calibrate | d = camera/debug view | f = full screen
                      Left/Right arrows (or z / x) = clicks from the keyboard | q or Esc = quit
Debug view: click on the marker in the camera image to learn its colour.
"""
import argparse
import faulthandler
import os
import sys
import traceback

faulthandler.enable()            # a native crash now prints a Python stack instead of dying silently
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # silence TensorFlow/oneDNN info lines
os.environ.setdefault("GLOG_minloglevel", "2")         # silence MediaPipe/absl info + warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # config.json / calibration.json live next to main.py

import numpy as np
import cv2

from config import Config
from calibration import BoardMapper, calibrate_interactive
from detection import MarkerDetector, FingerDetector, PointerFilter, bgr_to_hsv, bgr_to_ycrcb
from graphics import draw_line, circle_outline, fill_circle, put_text
from interaction import App
from remote_input import RemoteInput

BOARD_WIN, DEBUG_WIN = "Board", "Camera view"


def debug_view(frame, detector, mapper, meas):
    vis = frame.copy()
    if detector.last_mask is not None:
        m = detector.mask_fullres(vis.shape)
        vis[m] = (0.4 * vis[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    lms = getattr(detector, "last_landmarks", None)
    if lms is not None:                                  # MediaPipe skeleton
        for (x, y) in lms:
            fill_circle(vis, x, y, 3, (255, 200, 0))
        fill_circle(vis, lms[8][0], lms[8][1], 6, (0, 0, 255))   # index fingertip
    if mapper is not None:
        q = mapper.corners_img
        for i in range(4):
            draw_line(vis, q[i], q[(i + 1) % 4], (0, 255, 255), 2)
    if meas is not None:
        circle_outline(vis, meas[0], meas[1], 12, (0, 0, 255), 2)
    put_text(vis, "click on the marker / your skin to learn its colour", 8, 20, 0.5, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--log-keys", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    app = App(cfg)
    remote = RemoteInput(cfg, args.log_keys)
    detectors = {"marker": MarkerDetector(cfg), "skin": FingerDetector(cfg)}   # "finger" (MediaPipe) is built on demand

    def get_detector(mode):
        """MediaPipe is heavy: load it only when first selected; fall back to the marker on failure."""
        if mode == "finger" and "finger" not in detectors:
            try:
                from hand_tracking import MediaPipeFingerDetector
                d = MediaPipeFingerDetector(cfg)
                if mapper is not None:
                    d.set_roi_quad(mapper.corners_img)
                detectors["finger"] = d
                print("finger tracking backend:", d.backend_name)
            except Exception as e:
                print("MediaPipe finger tracking unavailable:", e)
                print("  -> pip install mediapipe   (Python 3.9-3.12). Falling back to the marker.")
                app.flags["tracking"] = cfg.tracking = "marker"
                return detectors["marker"]
        return detectors[mode]

    def set_rois(corners):
        for d in detectors.values():
            d.set_roi_quad(corners)
    last_mode = [app.flags["tracking"]]
    pfilter = PointerFilter(cfg.filt_min_alpha, cfg.filt_max_alpha, cfg.filt_speed_ref,
                            cfg.filt_dead_zone, cfg.hold_frames)
    cam = None
    mapper = BoardMapper.load(cfg.calib_path) if os.path.exists(cfg.calib_path) else None
    if mapper is not None:
        set_rois(mapper.corners_img)
    if not args.sim:
        from camera import Camera
        cam = Camera(cfg)
    state = {"frame": None, "meas": None, "debug_open": False, "sim_click": []}

    def on_board_mouse(event, x, y, flags, param):
        if not args.sim:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            app.set_pointer((x, y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            app.set_pointer((x, y))
            state["sim_click"].append("left")
        elif event == cv2.EVENT_RBUTTONDOWN:
            app.set_pointer((x, y))
            state["sim_click"].append("right")

    def on_debug_mouse(event, x, y, flags, param):
        """Learn the marker colour from a clicked pixel (median of a 7x7 patch)."""
        if event != cv2.EVENT_LBUTTONDOWN or state["frame"] is None:
            return
        f = state["frame"]
        patch = f[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        if app.flags["tracking"] == "finger":
            print("MediaPipe finger mode needs no colour learning.")
            return
        if app.flags["tracking"] == "skin":            # learn the skin colour
            _, cr, cb = bgr_to_ycrcb(patch)
            crm, cbm = float(np.median(cr)), float(np.median(cb))
            cfg.skin_cr = [crm - 14, crm + 14]
            cfg.skin_cb = [cbm - 14, cbm + 14]
            cfg.save()
            print(f"skin colour learned: Cr={crm:.0f} Cb={cbm:.0f} -> saved to config.json")
            return
        h, s, v = bgr_to_hsv(patch)
        hm, sm, vm = float(np.median(h)), float(np.median(s)), float(np.median(v))
        cfg.h_min, cfg.h_max = (hm - 28) % 360, (hm + 28) % 360
        cfg.s_min, cfg.v_min = max(0.25, sm - 0.3), max(0.2, vm - 0.3)
        cfg.save()
        print(f"marker colour learned: hue={hm:.0f} s={sm:.2f} v={vm:.2f} -> saved to config.json")

    cv2.namedWindow(BOARD_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(BOARD_WIN, on_board_mouse)
    if args.fullscreen:
        cv2.setWindowProperty(BOARD_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    running = True
    while running:
        # ---- 1. camera -> detection -> board coordinates -> pointer ----------
        if cam is not None:
            frame = cam.read()
            if frame is not None:
                state["frame"] = frame
                mode = app.flags["tracking"]
                if mode != last_mode[0]:
                    pfilter.reset(); last_mode[0] = mode
                detector = get_detector(mode)
                meas = detector.detect(frame)
                state["meas"] = meas
                board = None
                if meas is not None and mapper is not None:
                    b = mapper.to_board(meas)
                    if -40 <= b[0] <= cfg.board_w + 40 and -40 <= b[1] <= cfg.board_h + 40:
                        board = (min(max(b[0], 0), cfg.board_w - 1), min(max(b[1], 0), cfg.board_h - 1))
                pos, fresh = pfilter.update(board)
                app.set_pointer(pos, fresh)
        # ---- 2. display ----------------------------------------------------------
        cv2.imshow(BOARD_WIN, app.render())
        if app.flags["debug"] and state["frame"] is not None:
            if not state["debug_open"]:
                cv2.namedWindow(DEBUG_WIN)
                cv2.setMouseCallback(DEBUG_WIN, on_debug_mouse)
                state["debug_open"] = True
            cv2.imshow(DEBUG_WIN, debug_view(state["frame"], detectors.get(app.flags["tracking"], detectors["marker"]), mapper, state["meas"]))
        elif state["debug_open"]:
            cv2.destroyWindow(DEBUG_WIN)
            state["debug_open"] = False
        # ---- 3. keyboard / remote -------------------------------------------------
        key = cv2.waitKeyEx(1)
        events = remote.poll(key) + state["sim_click"]
        state["sim_click"] = []
        if key in (ord("z"), ord("x")):
            events.append("left" if key == ord("z") else "right")
        for ev in events:
            app.click(ev)
        if key in (27, ord("q")):
            print("Exit: q / Esc pressed")
            running = False
        elif key == ord("m"):
            app.toggle_tracking()
        elif key == ord("d"):
            app._toggle("debug")
        elif key == ord("f"):
            cv2.setWindowProperty(BOARD_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord("c") and "calibrate" not in app.requests:
            app.requests.append("calibrate")
        # ---- 4. requests coming from the interface ---------------------------------
        while app.requests:
            r = app.requests.pop(0)
            if r == "quit":
                print("Exit: Quit button")
                running = False
            elif r == "calibrate" and cam is not None:
                m = calibrate_interactive(cam.read, cfg.board_w, cfg.board_h)
                if m is not None:
                    mapper = m
                    set_rois(mapper.corners_img)
                    mapper.save(cfg.calib_path)
                    pfilter.reset()
                    print(f"calibration saved ({cfg.calib_path}); corner fit error "
                          f"{mapper.corner_error():.3f} px. Use Settings > Check accuracy.")
            elif r == "calibrate":
                print("Calibration needs a camera (not available in --sim mode).")
    if cam is not None:
        cam.close()
    remote.close()
    cv2.destroyAllWindows()





# exit conds
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit: Ctrl+C")
    except Exception:
        traceback.print_exc()
        print("\nCRASH (Python exception) -- copy the text above.")
        sys.exit(1)
