"""Webcam hand-tracking teleoperation of a simulated SO-101 arm (MuJoCo + MediaPipe + Mink)."""

import os

# Must be set before cv2 is first imported, otherwise Windows Media Foundation can take ~20 s
# to open a webcam.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
