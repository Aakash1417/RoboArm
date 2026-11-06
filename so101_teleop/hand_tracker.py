"""Webcam capture (background thread) and MediaPipe hand landmark detection."""

import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .config import HAND_MODEL, HAND_MODEL_URL, HandConfig
from .mapping import HandObservation

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
PALM_POINTS = [0, 5, 9, 13, 17]  # wrist + finger bases: stable while pinching
THUMB_TIP, INDEX_TIP = 4, 8


def ensure_hand_model():
    if not HAND_MODEL.exists():
        print(f"Downloading MediaPipe hand model to {HAND_MODEL} ...")
        HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
    return HAND_MODEL


class Camera:
    """Reads frames in a background thread so a slow frame never stalls the simulation."""

    def __init__(self, cfg: HandConfig):
        # Media Foundation decodes the webcam's compressed 1080p stream at ~30 FPS. DirectShow
        # often falls back to uncompressed YUY2, which only manages ~5 FPS at 1080p.
        self.cap = cv2.VideoCapture(cfg.camera_index, BACKENDS[cfg.backend])
        if not self.cap.isOpened() and cfg.backend != "dshow":
            self.cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {cfg.camera_index}. Try --camera 1, close other apps "
                "using the camera, or check Windows camera privacy settings."
            )
        self.backend = self.cap.getBackendName()
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.request_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.request_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def latest(self):
        """Returns (frame_id, frame). frame_id only changes when a new frame arrives."""
        with self._lock:
            return self._frame_id, self._frame

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class HandTracker:
    """MediaPipe HandLandmarker (pretrained, CPU) in VIDEO mode for one hand."""

    def __init__(self, cfg: HandConfig):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def detect(self, frame_bgr):
        """frame_bgr must already be mirrored. Returns (HandObservation | None, landmark pixels | None)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.hand_landmarks:
            return None, None
        lm = result.hand_landmarks[0]
        norm = np.array([[p.x, p.y] for p in lm])
        px = norm * np.array([w, h])
        return observation_from_landmarks(norm, px, w), px

    def close(self):
        self.landmarker.close()


def observation_from_landmarks(norm, px, width):
    palm = norm[PALM_POINTS].mean(axis=0)
    palm_length = np.linalg.norm(px[9] - px[0])  # wrist to middle-finger base
    palm_width = np.linalg.norm(px[17] - px[5])  # index base to pinky base
    pinch = np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP]) / max(palm_length, 1e-6)
    return HandObservation(
        x=float(palm[0]), y=float(palm[1]),
        size=float((palm_length + palm_width) / width),
        pinch_ratio=float(pinch),
    )


def draw_hand(img, px, pinched):
    pts = px.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 180, 255), -1, cv2.LINE_AA)
    color = (0, 0, 255) if pinched else (0, 255, 0)
    cv2.line(img, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), color, 3, cv2.LINE_AA)
    palm = pts[PALM_POINTS].mean(axis=0).astype(int)
    cv2.drawMarker(img, tuple(palm), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
