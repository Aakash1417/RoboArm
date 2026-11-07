"""Turns hand observations into a smooth, bounded Cartesian target and a gripper command."""

from dataclasses import dataclass

import numpy as np

from .config import MappingConfig, RobotConfig


@dataclass
class HandObservation:
    x: float            # palm centre, 0..1 across the (mirrored) image, 0 = your left
    y: float            # palm centre, 0..1 down the image, 0 = top
    size: float         # apparent palm size (fraction of image width); grows as the hand nears the camera
    pinch_ratio: float  # thumb-index distance / palm length; small = pinched
    finger_ratios: tuple = ()  # thumb to index, middle, ring, pinky tips / palm length
    angle: float = 0.0  # hand twist in radians: 0 = fingers up, positive = clockwise as you see it


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def grip_ratio(obs: HandObservation, cfg: MappingConfig) -> float:
    """The single number the gripper thresholds are applied to.

    In majority mode this is the k-th closest fingertip (k = grip_min_fingers): it is below the
    close threshold exactly when at least k fingertips are at the thumb.
    """
    if cfg.grip_mode == "majority" and obs.finger_ratios:
        k = int(np.clip(cfg.grip_min_fingers, 1, len(obs.finger_ratios)))
        return sorted(obs.finger_ratios)[k - 1]
    return obs.pinch_ratio


class GripperHysteresis:
    """Two thresholds plus a frame count so the gripper does not chatter near the threshold."""

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.closed = False
        self._count = 0

    def update(self, ratio):
        close_below, open_above = self.cfg.grip_thresholds
        if ratio < close_below:
            want = True
        elif ratio > open_above:
            want = False
        else:
            want = self.closed
        if want != self.closed:
            self._count += 1
            if self._count >= self.cfg.pinch_frames:
                self.closed = want
                self._count = 0
        else:
            self._count = 0
        return self.closed


class Teleop:
    """Relative hand-to-end-effector mapping.

    target = neutral_target + gain * (hand - neutral_hand), where "neutral" is captured on the
    first detection and whenever you calibrate, resume, or finish homing. The commanded target
    then moves toward that goal no faster than max_target_speed, so the robot never jumps.
    """

    def __init__(self, map_cfg: MappingConfig, robot_cfg: RobotConfig, home_target, roll_limits=(-np.pi, np.pi)):
        self.cfg = map_cfg
        self.rcfg = robot_cfg
        self.home_target = np.array(home_target, dtype=float)
        self.gripper = GripperHysteresis(map_cfg)
        self.target = self.home_target.copy()
        self.goal = self.home_target.copy()
        # Wrist roll is a joint angle (rad), commanded directly rather than through Cartesian IK.
        margin = robot_cfg.joint_limit_margin
        self.roll_limits = (roll_limits[0] + margin, roll_limits[1] - margin)
        self.roll_home = float(robot_cfg.home_q[4])
        self.roll_target = self.roll_home
        self.roll_goal = self.roll_home
        self.hand_angle = float("nan")
        self.paused = False
        self.homing = False
        self.tracking = False
        self.sensitivity = 1.0
        self.last_pinch = float("nan")
        self.last_fingers = ()
        self._needs_calibration = True
        self._ema = None
        self._filtered = None
        self._neutral_hand = None
        self._neutral_target = self.home_target.copy()
        self._neutral_roll = self.roll_home

    # ---- user commands -------------------------------------------------------------------
    def calibrate(self):
        """Next hand reading becomes neutral for the robot's current target (no jump)."""
        self._needs_calibration = True

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self.calibrate()  # resume like a clutch: wherever your hand is now maps to "here"
        self.goal = self.target.copy()
        self.roll_goal = self.roll_target

    def reset_home(self):
        self.homing = True
        self.goal = self.home_target.copy()
        self.roll_goal = self.roll_home

    def toggle_depth(self):
        self.cfg.depth_enabled = not self.cfg.depth_enabled
        self.calibrate()

    def toggle_wrist(self):
        self.cfg.wrist_enabled = not self.cfg.wrist_enabled
        self.roll_goal = self.roll_target
        self.calibrate()

    def change_sensitivity(self, factor):
        self.sensitivity = float(np.clip(self.sensitivity * factor, 0.25, 4.0))
        self.calibrate()

    # ---- per camera frame ----------------------------------------------------------------
    def observe(self, obs: HandObservation | None):
        self.tracking = obs is not None
        if obs is None:
            # Hold position. The filter restarts on re-acquire; the speed limit keeps that gradual.
            self._ema = None
            self._filtered = None
            self.hand_angle = float("nan")
            if not self.homing:
                self.goal = self.target.copy()
                self.roll_goal = self.roll_target
            return

        self.last_pinch = grip_ratio(obs, self.cfg)
        self.last_fingers = obs.finger_ratios
        self.hand_angle = obs.angle
        meas = np.array([obs.x, obs.y, np.log(max(obs.size, 1e-4)), obs.angle])
        if self._ema is None:
            self._ema = meas.copy()
            self._filtered = meas.copy()
        else:
            meas[3] = self._ema[3] + wrap_angle(meas[3] - self._ema[3])  # no jump across +/-180 deg
            self._ema += self.cfg.smoothing * (meas - self._ema)
            # Dead zone as backlash: output only follows once the hand moves beyond the band.
            band = np.array([self.cfg.dead_zone, self.cfg.dead_zone, self.cfg.depth_dead_zone,
                             self.cfg.roll_dead_zone])
            diff = self._ema - self._filtered
            self._filtered += np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)

        if self.paused:
            return
        self.gripper.update(self.last_pinch)
        if self.homing:
            return

        if self._needs_calibration or self._neutral_hand is None:
            self._neutral_hand = self._filtered.copy()
            self._neutral_target = self.target.copy()
            self._neutral_roll = self.roll_target
            self._needs_calibration = False

        d = self._filtered - self._neutral_hand
        if self.cfg.wrist_enabled:
            roll = self._neutral_roll + self.rcfg.wrist_roll_sign * self.cfg.gain_roll * wrap_angle(d[3])
            self.roll_goal = float(np.clip(roll, *self.roll_limits))
        s = self.sensitivity
        offset = np.array([
            s * self.cfg.gain_depth * d[2] if self.cfg.depth_enabled else 0.0,  # bigger hand -> forward
            -s * self.cfg.gain_xy * d[0],  # hand to your left -> robot +Y (its left, screen left)
            -s * self.cfg.gain_z * d[1],   # hand up (image y decreases) -> robot up
        ])
        self.goal = self.clamp_workspace(self._neutral_target + offset)

    # ---- every control tick --------------------------------------------------------------
    def step(self, dt):
        """Move the commanded target toward the goal with a speed limit. Returns the target."""
        delta = self.goal - self.target
        dist = np.linalg.norm(delta)
        max_step = self.rcfg.max_target_speed * dt
        if dist > max_step:
            delta *= max_step / dist
        self.target = self.target + delta
        max_roll = self.rcfg.max_roll_speed * dt
        self.roll_target += float(np.clip(self.roll_goal - self.roll_target, -max_roll, max_roll))
        if (self.homing and np.linalg.norm(self.goal - self.target) < 1e-3
                and abs(self.roll_goal - self.roll_target) < 1e-3):
            self.homing = False
            self.calibrate()
        return self.target

    def clamp_workspace(self, p):
        r = self.rcfg
        p = np.clip(p, r.workspace_min, r.workspace_max)
        v = p - r.shoulder_center
        n = np.linalg.norm(v)
        if n > 1e-9:
            p = r.shoulder_center + v * (np.clip(n, r.min_reach, r.max_reach) / n)
        return np.clip(p, r.workspace_min, r.workspace_max)

    @property
    def status(self):
        if self.paused:
            return "PAUSED"
        if self.homing:
            return "HOMING"
        return "TRACKING" if self.tracking else "HAND LOST - holding"
