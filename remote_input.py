"""Presentation-remote input.

A USB presentation remote is seen by the operating system as a *keyboard*
(HID device).  Its two buttons simply generate key presses, typically
Page Down (Next) and Page Up (Previous) -- some models send Right/Left arrows
or F5/Esc/B instead.  Two ways of reading them are supported:

* default: the key codes returned by cv2.waitKeyEx() in the board window
  (the window must have the focus);
* optional (use_pynput=True): a global listener (pynput) that receives the
  keys even when the window is not focused.

Keyboard arrows also work (see config: arrows_literal).
Mapping used by the application:
    Next     (e.g. Page Down) -> LEFT  click at the current pointer position
    Previous (e.g. Page Up)   -> RIGHT click at the current pointer position
Use key_discovery.py to find the real codes of your remote and put them in
config.json (next_keys / prev_keys, next_names / prev_names).
"""
import time
from collections import deque


class RemoteInput:
    def __init__(self, cfg, log_keys=False):
        self.cfg = cfg
        self.log_keys = log_keys
        self._last = {"left": 0.0, "right": 0.0}
        self._queue = deque()
        self._listener = None
        if cfg.use_pynput:
            try:
                from pynput import keyboard

                def on_press(key):
                    name = getattr(key, "name", None) or getattr(key, "char", None)
                    if self.log_keys:
                        print("pynput key:", name)
                    kind = self._classify_name(name)
                    if kind:
                        self._queue.append(kind)

                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.start()
            except Exception as e:  # pynput missing or no display access
                print("pynput unavailable, falling back to OpenCV keys:", e)
                cfg.use_pynput = False

    def _classify_name(self, name):
        c = self.cfg
        if name in c.next_names:
            return "left"
        if name in c.prev_names:
            return "right"
        if name in ("left", "right"):
            return self._arrow(name == "left")
        return None

    def _classify_code(self, code):
        c = self.cfg
        if code in c.next_keys:
            return "left"
        if code in c.prev_keys:
            return "right"
        if code in c.arrow_left_keys:
            return self._arrow(True)
        if code in c.arrow_right_keys:
            return self._arrow(False)
        return None

    def _arrow(self, is_left_arrow):
        """Arrow keys: Next/Previous style (default) or literal left/right click."""
        if self.cfg.arrows_literal:
            return "left" if is_left_arrow else "right"
        return "right" if is_left_arrow else "left"     # Left arrow = Previous

    def _debounced(self, kind):
        now = time.time()
        if now - self._last[kind] < self.cfg.debounce_s:
            return False
        self._last[kind] = now
        return True

    def poll(self, cv_key=-1):
        """Return the list of click events ('left' / 'right') since last call."""
        events = []
        if self.cfg.use_pynput:
            while self._queue:
                k = self._queue.popleft()
                if self._debounced(k):
                    events.append(k)
            return events
        if cv_key is not None and cv_key != -1:
            if self.log_keys:
                print("key code:", cv_key)
            kind = self._classify_code(cv_key)
            if kind and self._debounced(kind):
                events.append(kind)
        return events

    def close(self):
        if self._listener is not None:
            self._listener.stop()
