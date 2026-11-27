"""Stand-alone MediaPipe check (does NOT use the rest of the project).

    python mediapipe_test.py

Opens the camera, shows the 21 hand points and prints the index fingertip.
Every step prints a line, so if the program dies you can see the last step
reached.  q = quit.
"""
import faulthandler
import os
faulthandler.enable()
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "2")

print("1. importing numpy / cv2 ...", flush=True)
import numpy as np
import cv2
print("   numpy", np.__version__, "| cv2", cv2.__version__, "|", cv2.__file__, flush=True)
print("2. importing mediapipe ...", flush=True)
import mediapipe as mp
print("   mediapipe", mp.__version__, flush=True)
print("3. creating Hands ...", flush=True)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("4. opening camera 0 ...", flush=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
print("   opened:", cap.isOpened(), flush=True)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("   camera read failed", flush=True)
        continue
    if n == 0:
        print("5. first frame", frame.shape, frame.dtype, "-> calling hands.process ...", flush=True)
    rgb = np.ascontiguousarray(frame[..., ::-1])
    res = hands.process(rgb)
    if n == 0:
        print("6. process() returned OK", flush=True)
    n += 1
    if res.multi_hand_landmarks:
        h, w = frame.shape[:2]
        for lm in res.multi_hand_landmarks[0].landmark:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 200, 0), -1)
        t = res.multi_hand_landmarks[0].landmark[8]
        cv2.circle(frame, (int(t.x * w), int(t.y * h)), 8, (0, 0, 255), -1)
        if n % 15 == 0:
            print(f"   index tip: ({t.x * w:.0f}, {t.y * h:.0f})", flush=True)
    cv2.imshow("mediapipe test (q = quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
print("done", flush=True)
