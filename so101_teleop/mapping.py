"""Turns hand observations into a smooth, bounded Cartesian target and a gripper command."""

import time
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
    finger_curls: tuple = ()  # fingertip-to-wrist / knuckle-to-wrist, per finger; small = curled


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012): a low-pass whose cutoff rises with speed, so a
    still hand is smoothed hard (no jitter) while a fast one is followed with little lag.
    cutoff = min_cutoff + beta * |filtered speed|, per channel."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.d_cutoff = d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def reset(self):
        self.x = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt) * (x - self.x)
        return self.x.copy()


def approach(pos, vel, goal, vmax, amax, response, dt):
    """One step toward goal with capped speed and acceleration. The wanted speed is
    response * distance (so a moving goal is followed about 1/response s behind, without the
    stop-start of snapping onto it), never more than vmax or than braking at amax allows.
    Works on vectors (by length) and on 1-element arrays."""
    delta = goal - pos
    dist = np.linalg.norm(delta)
    if dist < 1e-9:
        want = np.zeros_like(vel)
    else:
        want = delta / dist * min(vmax, np.sqrt(2 * amax * dist), response * dist)
    dv = want - vel
    n = np.linalg.norm(dv)
    if n > amax * dt:
        dv *= amax * dt / n
    vel = vel + dv
    return pos + vel * dt, vel


def stopping_point(pos, vel, amax):
    """Where braking at amax from vel comes to rest."""
    return pos + vel * np.linalg.norm(vel) / (2 * amax)


def finger_scores(obs: HandObservation, cfg: MappingConfig) -> tuple:
    """Per finger: below 1.0 when the tip touches the thumb or the finger is curled into a fist."""
    curls = obs.finger_curls or (np.inf,) * len(obs.finger_ratios)
    return tuple(min(t / cfg.touch_below, c / cfg.curl_below) for t, c in zip(obs.finger_ratios, curls))


def grip_ratio(obs: HandObservation, cfg: MappingConfig) -> float:
    """The single number the gripper thresholds are applied to.

    In majority mode this is the k-th lowest finger score (k = grip_min_fingers): it is below 1.0
    exactly when at least k fingers are closed.
    """
    if cfg.grip_mode == "majority" and obs.finger_ratios:
        scores = finger_scores(obs, cfg)
        k = int(np.clip(cfg.grip_min_fingers, 1, len(scores)))
        return sorted(scores)[k - 1]
    return obs.pinch_ratio  # "index" and "proportional"


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
    first detection and whenever you calibrate, resume, finish homing, or bring your hand back after
    losing it. The hand is One Euro filtered, and the commanded target moves toward the goal
    with capped speed and acceleration, so the robot never jumps or jerks. The target is the gripper tip or the wrist joint, depending on the IK.
    """

    def __init__(self, map_cfg: MappingConfig, robot_cfg: RobotConfig, home_target, roll_limits=(-np.pi, np.pi),
                 clamp=None):
        self.cfg = map_cfg
        self.rcfg = robot_cfg
        self.clamp = clamp or self.clamp_workspace  # keeps a target point reachable
        self._lost_frames = 0
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
        c = map_cfg
        self._euro = OneEuroFilter([c.euro_min_cutoff] * 2 + [c.euro_min_cutoff_depth, c.euro_min_cutoff_roll],
                                   [c.euro_beta] * 2 + [c.euro_beta_depth, c.euro_beta_roll], c.euro_d_cutoff)
        self._smoothed = None
        self._filtered = None
        self.velocity = np.zeros(3)
        self.roll_velocity = np.zeros(1)
        self._grip_euro = OneEuroFilter(c.grip_min_cutoff, c.grip_beta)
        self.grip_open = 1.0  # 0 = closed .. 1 = open
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
        self._hold()

    def _hold(self):
        """Stop where braking can, without a jolt."""
        r = self.rcfg
        self.goal = self.clamp(stopping_point(self.target, self.velocity, r.max_target_accel))
        roll = stopping_point(np.array([self.roll_target]), self.roll_velocity, r.max_roll_accel)[0]
        self.roll_goal = float(np.clip(roll, *self.roll_limits))

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
    def observe(self, obs: HandObservation | None, t=None):
        """t: when the frame was captured (perf_counter seconds); now if not given."""
        t = time.perf_counter() if t is None else t
        self.tracking = obs is not None
        if obs is None:
            # Hold position. After a short dropout the old neutral still applies; after a longer
            # one the returning hand re-anchors, like lifting a mouse off the desk.
            self._lost_frames += 1
            if self.cfg.reanchor_on_loss and self._lost_frames >= self.cfg.reanchor_frames:
                self._needs_calibration = True
            self._euro.reset()
            self._grip_euro.reset()
            self._smoothed = None
            self._filtered = None
            self.hand_angle = float("nan")
            if not self.homing:
                self._hold()
            return

        self._lost_frames = 0
        self.last_pinch = grip_ratio(obs, self.cfg)
        self.last_fingers = finger_scores(obs, self.cfg) if self.cfg.grip_mode == "majority" else obs.finger_ratios
        self.hand_angle = obs.angle
        meas = np.array([obs.x, obs.y, np.log(max(obs.size, 1e-4)), obs.angle])
        if self._smoothed is None:
            self._smoothed = self._euro(meas, t)
            self._filtered = meas.copy()
        else:
            meas[3] = self._smoothed[3] + wrap_angle(meas[3] - self._smoothed[3])  # no jump across +/-180 deg
            self._smoothed = self._euro(meas, t)
            # Dead zone as backlash: output only follows once the hand moves beyond the band.
            band = np.array([self.cfg.dead_zone, self.cfg.dead_zone, self.cfg.depth_dead_zone,
                             self.cfg.roll_dead_zone])
            diff = self._smoothed - self._filtered
            self._filtered += np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)

        if self.paused:
            return
        if self.cfg.grip_mode == "proportional":
            lo, hi = self.cfg.grip_thresholds
            self.grip_open = float(np.clip(self._grip_euro(np.array([self.last_pinch]), t)[0] - lo, 0, hi - lo)
                                   / (hi - lo))
            self.gripper.closed = self.grip_open < 0.5
        else:
            self.grip_open = 0.0 if self.gripper.update(self.last_pinch) else 1.0
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
        raw = self._neutral_target + offset
        self.goal = self.clamp(raw)
        if self.cfg.slide_anchor:
            self._neutral_target += self.goal - raw

    # ---- every control tick --------------------------------------------------------------
    def step(self, dt):
        """Move the commanded target toward the goal with speed and acceleration limits, so it
        eases in and out instead of starting and stopping in steps. Returns the target."""
        r = self.rcfg
        self.target, self.velocity = approach(self.target, self.velocity, self.goal, r.max_target_speed,
                                              r.max_target_accel, r.target_response, dt)
        roll, self.roll_velocity = approach(np.array([self.roll_target]), self.roll_velocity,
                                            np.array([self.roll_goal]), r.max_roll_speed, r.max_roll_accel,
                                            r.target_response, dt)
        self.roll_target = float(roll[0])
        if (self.homing and np.linalg.norm(self.goal - self.target) < 1e-3
                and abs(self.roll_goal - self.roll_target) < 1e-3):
            self.homing = False
            self.calibrate()
        return self.target

    @property
    def gripper_goal(self):
        """Gripper joint angle (rad) for the current grip_open."""
        r = self.rcfg
        return r.gripper_closed + (r.gripper_open - r.gripper_closed) * self.grip_open

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
