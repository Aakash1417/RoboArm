"""All tunable settings in one place. Edit values here, or override common ones on the command line."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "assets" / "so101" / "scene_tabletop.xml"
ROBOT_XML = ROOT / "assets" / "so101" / "so101_new_calib.xml"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Joint/actuator order of the official SO-101 MuJoCo model (so101_new_calib.xml).
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINT_NAMES[:5]
EE_SITE = "gripperframe"


@dataclass
class RobotConfig:
    # Home pose in radians (arm joints only). Gripper in front of the base, pointing down-forward.
    home_q: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.2, 0.8, 1.3, 1.5708]))

    # Gripper joint range in the MuJoCo model is [-0.1745, 1.7453] rad.
    # Larger value = more open. Slightly negative "closed" squeezes objects.
    gripper_open: float = 1.2
    gripper_closed: float = -0.1
    gripper_speed: float = 4.0  # rad/s

    # Cartesian workspace box for the IK target, in world metres (robot base at the origin).
    # +X = forward (away from you), +Y = robot's left, +Z = up. Table top is at Z = 0.
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.10, -0.28, 0.02]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.38, 0.28, 0.36]))
    # Max distance of the target from the shoulder-lift axis (arm reach is about 0.36 m).
    shoulder_center: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.0, 0.12]))
    max_reach: float = 0.33
    min_reach: float = 0.10

    max_target_speed: float = 0.35  # m/s, caps how fast the target point can move
    max_joint_speed: float = 3.0    # rad/s, IK velocity limit per joint
    joint_limit_margin: float = 0.05  # rad kept away from each joint limit


@dataclass
class HandConfig:
    camera_index: int = 0
    backend: str = "msmf"  # "msmf" (Media Foundation), "dshow" (DirectShow) or "any"
    request_width: int = 1920
    request_height: int = 1080
    process_width: int = 640  # frames are resized to this width before MediaPipe

    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass
class MappingConfig:
    # Metres of robot motion per full image width / height of hand motion.
    gain_xy: float = 0.55
    gain_z: float = 0.55
    # Forward/back from apparent hand size (estimate only): metres per +100% change in hand size.
    gain_depth: float = 0.20
    depth_enabled: bool = False

    smoothing: float = 0.35   # EMA factor per frame (0..1). Lower = smoother but laggier.
    dead_zone: float = 0.004  # normalized image units of hand jitter that are ignored
    depth_dead_zone: float = 0.03

    # Grip ratio = thumb-tip to fingertip distance divided by palm length (wrist to middle-finger base),
    # measured on MediaPipe's 3D hand landmarks.
    #   "all":   uses the FARTHEST of index/middle/ring/pinky, so the gripper only closes when every
    #            fingertip is brought to the thumb, and opens as soon as any finger moves away.
    #   "index": thumb-index pinch only.
    grip_mode: str = "all"
    all_close_below: float = 0.45
    all_open_above: float = 0.65
    pinch_close_below: float = 0.30
    pinch_open_above: float = 0.45
    pinch_frames: int = 3  # frames the new state must persist before the gripper switches

    @property
    def grip_thresholds(self):
        if self.grip_mode == "all":
            return self.all_close_below, self.all_open_above
        return self.pinch_close_below, self.pinch_open_above


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
