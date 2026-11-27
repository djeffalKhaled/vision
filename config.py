"""Central configuration (saved to / loaded from config.json).

Every tunable value of the project lives here so that nothing has to be edited
in the source code during a demonstration: colour thresholds, remote key codes,
camera index, file names...
"""
import json
import os
from dataclasses import dataclass, field, asdict

CONFIG_PATH = "config.json"


@dataclass
class Config:
    # ---- board (projected interface) size in "board pixels" ------------------
    board_w: int = 1280
    board_h: int = 720
    # ---- camera ---------------------------------------------------------------
    camera_index: int = 0
    cam_w: int = 640
    cam_h: int = 480
    intrinsics_path: str = "intrinsics.npz"   # optional (calibrate_intrinsics.py)
    # ---- marker segmentation (HSV, hue in degrees 0..360) --------------------
    h_min: float = 85.0       # default marker: bright GREEN sticker / LED / cap
    h_max: float = 160.0
    s_min: float = 0.45
    v_min: float = 0.35
    # "marker" = coloured tip | "finger" = index finger with MediaPipe | "skin" = own skin-colour finger tracker
    tracking: str = "marker"
    hand_max_hands: int = 2
    hand_model_path: str = "models/hand_landmarker.task"
    hand_model_url: str = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                           "hand_landmarker/float16/1/hand_landmarker.task")
    min_area: int = 12        # minimum number of mask pixels (at detection scale)
    # ---- index-finger tracking (skin segmentation in YCrCb, 0..255) ----------
    skin_cr: list = field(default_factory=lambda: [135, 175])
    skin_cb: list = field(default_factory=lambda: [85, 130])
    skin_y_min: float = 60.0
    finger_stride: int = 3
    finger_min_area: int = 60
    detect_stride: int = 2    # process one pixel out of 2 in x and y (speed)
    # ---- pointer stabilisation (board pixels) --------------------------------
    filt_min_alpha: float = 0.25
    filt_max_alpha: float = 0.85
    filt_speed_ref: float = 120.0
    filt_dead_zone: float = 3.0
    hold_frames: int = 8      # frames during which a lost pointer is kept
    # ---- remote control -------------------------------------------------------
    # "Next" button     -> LEFT  click      "Previous" button -> RIGHT click
    # Codes returned by cv2.waitKeyEx(): Windows PgDn/PgUp/Right/Left,
    # Linux(GTK) PgDn/PgUp/Right/Left. Run key_discovery.py to find YOUR codes.
    next_keys: list = field(default_factory=lambda: [2228224, 65366])   # Page Down
    prev_keys: list = field(default_factory=lambda: [2162688, 65365])   # Page Up
    next_names: list = field(default_factory=lambda: ["page_down"])
    prev_names: list = field(default_factory=lambda: ["page_up"])
    # Keyboard arrows (work like the remote, handy without a remote):
    #   arrows_literal = False : Right arrow = Next (left click),  Left arrow = Previous (right click)
    #   arrows_literal = True  : Left arrow = LEFT click,          Right arrow = RIGHT click
    arrows_literal: bool = False
    arrow_left_keys: list = field(default_factory=lambda: [2424832, 65361])
    arrow_right_keys: list = field(default_factory=lambda: [2555904, 65363])
    use_pynput: bool = False  # True: global key listener (works without window focus)
    debounce_s: float = 0.15
    # ---- files ----------------------------------------------------------------
    calib_path: str = "calibration.json"
    images_dir: str = "images"

    # ------------------------------------------------------------------------
    @staticmethod
    def load(path: str = CONFIG_PATH) -> "Config":
        cfg = Config()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
