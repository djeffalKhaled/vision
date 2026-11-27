# Smart Interactive Whiteboard (Python + NumPy)

## Install
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python   # only ONE opencv package allowed
    pip install -r requirements.txt        # numpy<2, opencv-contrib-python<4.12, mediapipe==0.10.21
    python mediapipe_test.py               # stand-alone check of MediaPipe + camera (prints each step)
    python tests/test_core.py              # headless self-test (no camera needed)

## Files
| file | role |
|---|---|
| `main.py` | entry point, main loop (camera -> detection -> mapping -> UI -> remote) |
| `camera.py` | threaded camera acquisition, optional lens undistortion |
| `detection.py` | marker tracking (HSV, thresholding, morphology, robust centroid), **index-finger tracking** (YCrCb skin mask, flood-fill components, fingertip = farthest point from the arm base), pointer filter (all NumPy) |
| `calibration.py` | homography (DLT + SVD, Hartley normalisation), corner ordering, interactive calibration |
| `remote_input.py` | remote buttons -> left / right click (OpenCV key codes or pynput) |
| `interaction.py` | menus, drawing, image viewer + zoom, drag & drop, accuracy test (state machine) |
| `graphics.py` | line / circle / rectangle / resize / zoom-view primitives written with NumPy |
| `hand_tracking.py` | **MediaPipe** index-fingertip tracking (landmark 8), legacy + Tasks API |
| `lesson.py`, `sample_images.py` | demo lesson (5 slides) and generated images |
| `mediapipe_test.py` | stand-alone MediaPipe + camera check, prints every step (use it if something crashes) |
| `key_discovery.py` | **run first**: shows the key codes your remote produces |
| `calibrate_intrinsics.py` | optional chessboard calibration of the lens (cv2.calibrateCamera) |
| `config.json` | created automatically (colour thresholds, key codes...) |

## First use (hardware)
1. `python key_discovery.py` -> press NEXT and PREVIOUS several times; note the codes.
   Put them in `config.json` (`next_keys`, `prev_keys`). Defaults cover PgDn/PgUp/Right/Left on Windows and Linux.
   NEXT = **left click**, PREVIOUS = **right click**.
2. `python main.py`; drag the *Board* window to the projector, press `f` (full screen).
3. Press `d` (camera view), click on the marker in the camera image: its colour is learned and saved.
4. Press `c` (or Settings > Calibrate camera): click the 4 board corners in the camera image, Enter.
5. Settings > **Check accuracy**: put the marker on the 9 targets, NEXT each time -> mean / RMS / max error.
6. No hardware? `python main.py --sim` : mouse = pointer, mouse buttons (or `z` / `x`) = clicks.

## Tracking modes (key `m` or Settings > Tracking cycles through them)
* **marker** (default): coloured tip, most reliable under a projector.
* **finger**: index finger with **MediaPipe Hands** -- the pointer is the index fingertip (landmark 8).
  No marker, no colour tuning. Use **mediapipe==0.10.21** (pinned in `requirements.txt`): it has the classic
  `mp.solutions.hands` API with the model bundled, so nothing is downloaded. (Newer versions also work through the
  Tasks API but need the `hand_landmarker.task` model file, downloaded automatically into `models/`.)
  Press `d` to see the 21 detected hand points. If two hands are visible, the one pointing inside the board area
  is used. If MediaPipe is missing the app prints a message and falls back to the marker.
  MediaPipe is a predefined model: justify it in the report; calibration, mapping, filtering and the interface
  remain your own code.
* **skin**: my own skin-colour finger tracker (YCrCb + largest blob + farthest point). Experimental fallback,
  sensitive to lighting and skin tone.

## Keyboard instead of the remote
Page Down = Next = left click, Page Up = Previous = right click, and also the arrows:
Right arrow = Next (left click), Left arrow = Previous (right click). If you prefer *literal* arrows
(Left arrow = left click, Right arrow = right click) set `"arrows_literal": true` in `config.json`. `z` / `x` also click.

## Controls
LEFT click: press a button / pen down-up / grab-drop an object.  RIGHT click: pen up, drop, else **back**.
Keys: `m` marker/finger, `c` calibrate, `d` camera view, `f` full screen, `z`/`x` keyboard clicks, `q`/Esc quit.

## Predefined functions used (to justify in the report / demo)
Camera and window I/O (`VideoCapture`, `imshow`, `waitKeyEx`, `setMouseCallback`), text (`putText`, `getTextSize`),
image file decoding (`imread`), optional lens calibration (`calibrateCamera`, `remap`).
Everything else (HSV, thresholding, morphology, centroid, homography/SVD, filtering, drawing, zoom) is NumPy.

## Known limits
Marker colour must differ from the projected content (the UI avoids saturated green); strong ambient light
or a hidden marker lowers tracking quality; the homography assumes a flat board and a fixed camera.
