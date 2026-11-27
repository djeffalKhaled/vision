"""Run this FIRST with the remote plugged in.

Click on the window, press each remote button several times, and write down
the codes shown.  They go into config.json (next_keys / prev_keys).
Optional: `pip install pynput` also prints the key *names* seen by the OS
(e.g. 'page_down'), used by next_names / prev_names.
"""
import numpy as np
import cv2
from graphics import put_text

try:
    from pynput import keyboard
    names = []
    keyboard.Listener(on_press=lambda k: names.append(str(getattr(k, "name", None) or getattr(k, "char", k)))).start()
except Exception:
    names = None

log = []
cv2.namedWindow("Key discovery")
while True:
    img = np.full((360, 640, 3), 255, np.uint8)
    put_text(img, "Press the remote buttons (click here first). q = quit", 10, 30, 0.6)
    for i, line in enumerate(log[-8:]):
        put_text(img, line, 10, 80 + 30 * i, 0.7, (150, 0, 0), 2)
    if names:
        put_text(img, "OS key names: " + ", ".join(names[-4:]), 10, 340, 0.55, (0, 100, 0), 1)
    cv2.imshow("Key discovery", img)
    k = cv2.waitKeyEx(50)
    if k == -1:
        continue
    if k == ord("q"):
        break
    line = f"waitKeyEx code = {k}   (low byte = {k & 0xFF})"
    log.append(line)
    print(line)
cv2.destroyAllWindows()
