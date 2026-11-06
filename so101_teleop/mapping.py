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


class GripperHysteresis:
    """Two thresholds plus a frame count so the gripper does not chatter near the threshold."""

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.closed = False
        self._count = 0

    def update(self, pinch_ratio):
        if pinch_ratio < self.cfg.pinch_close_below:
            want = True
        elif pinch_ratio > self.cfg.pinch_open_above:
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

    def __init__(self, map_cfg: MappingConfig, robot_cfg: RobotConfig, home_target):
        self.cfg = map_cfg
        self.rcfg = robot_cfg
        self.home_target = np.array(home_target, dtype=float)
        self.gripper = GripperHysteresis(map_cfg)
        self.target = self.home_target.copy()
        self.goal = self.home_target.copy()
        self.paused = False
        self.homing = False
        self.tracking = False
        self.sensitivity = 1.0
        self.last_pinch = float("nan")
        self._needs_calibration = True
        self._ema = None
        self._filtered = None
        self._neutral_hand = None
        self._neutral_target = self.home_target.copy()

    # ---- user commands -------------------------------------------------------------------
    def calibrate(self):
        """Next hand reading becomes neutral for the robot's current target (no jump)."""
        self._needs_calibration = True

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self.calibrate()  # resume like a clutch: wherever your hand is now maps to "here"
        self.goal = self.target.copy()

    def reset_home(self):
        self.homing = True
        self.goal = self.home_target.copy()

    def toggle_depth(self):
        self.cfg.depth_enabled = not self.cfg.depth_enabled
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
            if not self.homing:
                self.goal = self.target.copy()
            return

        self.last_pinch = obs.pinch_ratio
        meas = np.array([obs.x, obs.y, np.log(max(obs.size, 1e-4))])
        if self._ema is None:
            self._ema = meas.copy()
            self._filtered = meas.copy()
        else:
            self._ema += self.cfg.smoothing * (meas - self._ema)
            # Dead zone as backlash: output only follows once the hand moves beyond the band.
            band = np.array([self.cfg.dead_zone, self.cfg.dead_zone, self.cfg.depth_dead_zone])
            diff = self._ema - self._filtered
            self._filtered += np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)

        if self.paused:
            return
        self.gripper.update(obs.pinch_ratio)
        if self.homing:
            return

        if self._needs_calibration or self._neutral_hand is None:
            self._neutral_hand = self._filtered.copy()
            self._neutral_target = self.target.copy()
            self._needs_calibration = False

        d = self._filtered - self._neutral_hand
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
        if self.homing and np.linalg.norm(self.goal - self.target) < 1e-3:
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
