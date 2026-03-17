from __future__ import annotations

import importlib


def import_cv_np():
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing. Install it with: pip install opencv-python") from exc

    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("NumPy is missing. Install it with: pip install numpy") from exc

    return cv2, np
