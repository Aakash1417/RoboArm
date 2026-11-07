"""Headless checks of the whole pipeline without a webcam or windows.

    python selftest.py            # simulation, IK, mapping, MediaPipe model
    python selftest.py --camera   # also try to open webcam 0 and read a frame

Writes snapshot images to selftest_output/.
"""

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import mujoco
import numpy as np

from so101_teleop.config import JOINT_NAMES, Config
from so101_teleop.mapping import HandObservation, Teleop
from so101_teleop.robot import CartesianIK, SO101Sim

OUT = Path(__file__).resolve().parent / "selftest_output"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


class Rig:
    """Same control loop as main.py, but driven by scripted observations at 30 Hz."""

    def __init__(self):
        self.cfg = Config()
        self.sim = SO101Sim(self.cfg.robot)
        self.ik = CartesianIK(self.cfg.robot)
        lo, hi = self.ik.joint_limits
        self.teleop = Teleop(self.cfg.mapping, self.cfg.robot, self.ik.home_target, roll_limits=(lo[4], hi[4]))
        self.grip = self.cfg.robot.gripper_open
        self.dt = 0.01
        self.max_target_speed = 0.0
        self.max_joint_speed = 0.0
        self.limit_violation = 0.0

    def run(self, seconds, obs_fn):
        steps = int(seconds / self.dt)
        lo, hi = self.sim.ctrl_lo, self.sim.ctrl_hi
        for i in range(steps):
            if i % 3 == 0:
                self.teleop.observe(obs_fn(i * self.dt))
            prev = self.teleop.target.copy()
            target = self.teleop.step(self.dt)
            self.max_target_speed = max(self.max_target_speed, np.linalg.norm(target - prev) / self.dt)
            self.ik.set_wrist_roll(self.teleop.roll_target)
            arm_q = self.ik.solve(target, self.dt)
            want = self.cfg.robot.gripper_closed if self.teleop.gripper.closed else self.cfg.robot.gripper_open
            s = self.cfg.robot.gripper_speed * self.dt
            self.grip += np.clip(want - self.grip, -s, s)
            self.sim.set_command(arm_q, self.grip)
            self.sim.step(self.dt)
            q = self.sim.joint_q
            self.max_joint_speed = max(self.max_joint_speed, np.abs(self.sim.data.qvel[self.sim.qpos_adr[:5]]).max())
            self.limit_violation = max(self.limit_violation, np.max(q - hi), np.max(lo - q))
        return self.sim.ee_pos


def hand(x=0.5, y=0.5, size=0.12, pinch=0.8, fingers=None, angle=0.0, curls=(1.8,) * 4):
    fingers = fingers or (pinch,) * 4
    return lambda t: HandObservation(x, y, size, fingers[0], tuple(fingers), angle, tuple(curls))


def test_model():
    sim = SO101Sim(Config().robot)
    for i, n in enumerate(JOINT_NAMES):
        jid = sim.model.joint(n).id
        print(f"    {n:<13} joint range [{sim.model.jnt_range[jid, 0]:+.3f}, {sim.model.jnt_range[jid, 1]:+.3f}]"
              f"  actuator ctrlrange [{sim.ctrl_lo[i]:+.3f}, {sim.ctrl_hi[i]:+.3f}]")
    check("model loads with 6 actuators", sim.model.nu == 6)


def test_ik_grid():
    cfg = Config()
    ik = CartesianIK(cfg.robot)
    teleop = Teleop(cfg.mapping, cfg.robot, ik.home_target)
    lo, hi = ik.joint_limits
    errs = []
    for x in np.linspace(0.14, 0.34, 5):
        for y in np.linspace(-0.2, 0.2, 5):
            for z in np.linspace(0.03, 0.25, 4):
                tgt = teleop.clamp_workspace(np.array([x, y, z]))
                ik.reset()
                for _ in range(300):
                    q = ik.solve(tgt, 0.01)
                errs.append(np.linalg.norm(ik.ee_position() - tgt))
                assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)
    errs = np.array(errs)
    check("IK reaches workspace targets", np.percentile(errs, 95) < 0.005,
          f"(median {np.median(errs) * 1000:.1f} mm, 95th pct {np.percentile(errs, 95) * 1000:.1f} mm, "
          f"max {errs.max() * 1000:.1f} mm over {len(errs)} targets)")


def test_pipeline():
    rig = Rig()
    tp = rig.teleop
    home = rig.run(1.5, hand())
    check("calibration does not move the robot", np.linalg.norm(home - rig.ik.home_target) < 0.01,
          f"(drift {np.linalg.norm(home - rig.ik.home_target) * 1000:.1f} mm)")

    p = rig.run(2.0, hand(x=0.35))
    check("hand left -> gripper moves left (+Y)", p[1] - home[1] > 0.05, f"(dy {p[1] - home[1]:+.3f} m)")
    p = rig.run(3.0, hand(x=0.65))
    check("hand right -> gripper moves right (-Y)", p[1] - home[1] < -0.05, f"(dy {p[1] - home[1]:+.3f} m)")
    rig.run(2.0, hand())
    p = rig.run(2.0, hand(y=0.35))
    check("hand up -> gripper moves up (+Z)", p[2] - home[2] > 0.05, f"(dz {p[2] - home[2]:+.3f} m)")
    p = rig.run(3.0, hand(y=0.62))
    check("hand down -> gripper moves down (-Z)", p[2] - home[2] < -0.03, f"(dz {p[2] - home[2]:+.3f} m)")
    rig.run(2.0, hand())

    rig.run(1.0, hand(pinch=0.1))
    g_closed = rig.sim.joint_q[5]
    rig.run(1.0, hand(pinch=0.8))
    g_open = rig.sim.joint_q[5]
    check("fingertips to thumb closes, opening releases gripper",
          tp.gripper.closed is False and g_closed < 0.1 < 1.0 < g_open,
          f"(closed {g_closed:+.2f} rad, open {g_open:+.2f} rad)")

    rig.run(1.0, hand(fingers=(0.1, 0.1, 0.9, 0.9)))
    two_open = not tp.gripper.closed
    rig.run(1.0, hand(fingers=(0.1, 0.9, 0.1, 0.1)))
    three_closed = tp.gripper.closed
    rig.run(1.0, hand(fingers=(0.1, 0.9, 0.9, 0.1)))
    check("majority grip: 2 of 4 fingertips stays open, any 3 of 4 closes, back to 2 opens",
          two_open and three_closed and not tp.gripper.closed)

    rig.run(1.0, hand(fingers=(0.9,) * 4, curls=(0.6,) * 4))
    fist = tp.gripper.closed
    rig.run(1.0, hand())
    reopened = not tp.gripper.closed
    rig.run(1.0, hand(fingers=(0.1, 0.1, 0.9, 0.9), curls=(1.8, 1.8, 0.7, 1.8)))
    mixed = tp.gripper.closed
    rig.run(1.0, hand())
    rig.run(1.0, hand(fingers=(0.9,) * 4, curls=(1.2,) * 4))
    half_bent = not tp.gripper.closed
    check("fist closes, open hand reopens, 2 touching + 1 curled closes, half-bent fingers stay open",
          fist and reopened and mixed and half_bent)
    rig.run(1.0, hand())

    flicker = iter([0.1, 0.8, 0.1, 0.8, 0.55, 0.55, 0.55] * 50)
    rig.run(1.0, lambda t: HandObservation(0.5, 0.5, 0.12, r := next(flicker), (r,) * 4))
    check("gripper ignores single-frame flicker", tp.gripper.closed is False)

    tp.cfg.grip_mode = "index"
    rig.run(1.0, hand(fingers=(0.1, 0.9, 0.9, 0.9)))
    check("--grip index mode: thumb-index pinch alone closes", tp.gripper.closed)
    rig.run(1.0, hand())
    tp.cfg.grip_mode = "majority"

    rig.run(0.5, hand(x=0.4, y=0.45))
    before = rig.sim.ee_pos
    rig.run(3.0, lambda t: None)
    after = rig.sim.ee_pos
    check("hand lost -> robot holds position", np.linalg.norm(after - before) < 0.01,
          f"(moved {np.linalg.norm(after - before) * 1000:.1f} mm in 3 s)")

    rig.max_target_speed = 0.0
    rig.run(3.0, hand(x=0.95, y=0.1))
    check("re-acquire far away -> bounded target speed",
          rig.max_target_speed <= rig.cfg.robot.max_target_speed + 1e-6,
          f"(max {rig.max_target_speed:.2f} m/s, limit {rig.cfg.robot.max_target_speed} m/s)")

    rig.run(3.0, hand(x=-2.0, y=-2.0))
    t = tp.target
    inside = np.all(t >= rig.cfg.robot.workspace_min - 1e-9) and np.all(t <= rig.cfg.robot.workspace_max + 1e-9)
    check("extreme hand position stays inside the workspace", inside, f"(target {np.round(t, 3)})")

    tp.toggle_pause()
    before = rig.sim.ee_pos
    rig.run(1.5, hand(x=0.2, y=0.8, pinch=0.1))
    check("pause holds arm and gripper", np.linalg.norm(rig.sim.ee_pos - before) < 0.01 and not tp.gripper.closed)
    tp.toggle_pause()
    p0 = rig.sim.ee_pos
    rig.run(1.0, hand(x=0.2, y=0.8))
    check("resume does not jump", np.linalg.norm(rig.sim.ee_pos - p0) < 0.01,
          f"(moved {np.linalg.norm(rig.sim.ee_pos - p0) * 1000:.1f} mm)")

    tp.reset_home()
    p = rig.run(4.0, hand(x=0.2, y=0.8))
    check("R returns to home", np.linalg.norm(p - rig.ik.home_target) < 0.01 and not tp.homing,
          f"(error {np.linalg.norm(p - rig.ik.home_target) * 1000:.1f} mm)")

    rig.run(1.0, hand(x=0.2, y=0.8))
    roll0, pos0 = rig.sim.joint_q[4], rig.sim.ee_pos
    p = rig.run(2.0, hand(x=0.2, y=0.8, angle=0.5))
    roll1 = rig.sim.joint_q[4]
    check("hand twist clockwise -> wrist_roll turns 0.5 rad (clockwise as seen from behind)",
          abs((roll1 - roll0) - rig.cfg.robot.wrist_roll_sign * 0.5) < 0.03,
          f"(wrist_roll {roll0:+.2f} -> {roll1:+.2f} rad)")
    check("twisting does not move the gripper position", np.linalg.norm(p - pos0) < 0.01,
          f"(moved {np.linalg.norm(p - pos0) * 1000:.1f} mm)")
    rig.run(3.0, hand(x=0.2, y=0.8, angle=-2.0))
    hi_roll = rig.ik.joint_limits[1][4]
    check("large twist stops at the wrist_roll limit", rig.sim.joint_q[4] <= hi_roll + 0.01,
          f"(wrist_roll {rig.sim.joint_q[4]:+.2f}, limit {hi_roll:+.2f} rad)")
    held = rig.sim.joint_q[4]
    rig.run(2.0, lambda t: None)
    check("hand lost -> wrist roll holds", abs(rig.sim.joint_q[4] - held) < 0.02)
    tp.toggle_wrist()
    rig.run(1.0, hand(x=0.2, y=0.8))
    held = rig.sim.joint_q[4]
    rig.run(1.5, hand(x=0.2, y=0.8, angle=0.8))
    check("T (wrist off) ignores hand twist", abs(rig.sim.joint_q[4] - held) < 0.02)
    tp.toggle_wrist()
    tp.reset_home()
    rig.run(4.0, hand(x=0.2, y=0.8, angle=0.8))
    check("R also returns the wrist roll home", abs(rig.sim.joint_q[4] - rig.cfg.robot.home_q[4]) < 0.03,
          f"(wrist_roll {rig.sim.joint_q[4]:+.2f} rad)")

    tp.toggle_depth()
    rig.run(1.0, hand(size=0.12))
    x0 = rig.sim.ee_pos[0]
    p = rig.run(2.0, hand(size=0.18))
    check("depth estimate: bigger hand -> forward (+X)", p[0] - x0 > 0.03, f"(dx {p[0] - x0:+.3f} m)")

    check("joints never exceed model limits", rig.limit_violation < 0.02,
          f"(worst overshoot {rig.limit_violation:.4f} rad)")
    check("joint speeds stay moderate", rig.max_joint_speed < 6.0, f"(max {rig.max_joint_speed:.2f} rad/s)")


def test_snapshots():
    OUT.mkdir(exist_ok=True)
    rig = Rig()
    r = mujoco.Renderer(rig.sim.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.17, 0, 0.08]
    cam.distance, cam.azimuth, cam.elevation = 0.85, -20, -30
    tiles = []
    for label, fn in [("home", hand()), ("hand left", hand(x=0.3)), ("hand right + up", hand(x=0.7, y=0.35)),
                      ("pinch (closed)", hand(x=0.7, y=0.35, pinch=0.1))]:
        rig.run(2.5, fn)
        rig.sim.set_target_marker(rig.teleop.target)
        mujoco.mj_forward(rig.sim.model, rig.sim.data)
        r.update_scene(rig.sim.data, cam)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    path = OUT / "sim_snapshots.png"
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    check("offscreen render", path.exists(), f"({path})")


def test_mediapipe():
    from so101_teleop.config import HandConfig
    from so101_teleop.hand_tracker import HandTracker

    tracker = HandTracker(HandConfig())
    blank = np.zeros((360, 640, 3), np.uint8)
    t0 = time.perf_counter()
    for _ in range(20):
        obs, _ = tracker.detect(blank)
    ms = (time.perf_counter() - t0) / 20 * 1000
    check("MediaPipe hand model loads and runs", obs is None, f"({ms:.1f} ms per 640x360 frame on CPU)")
    tracker.close()

    from so101_teleop.hand_tracker import FINGER_TIPS, THUMB_TIP, finger_ratios

    pts = np.zeros((21, 3))
    pts[9] = [0, 0.09, 0]  # palm length 9 cm
    pts[THUMB_TIP] = [0.03, 0.12, 0.02]
    for tip, dx in zip(FINGER_TIPS, (0.01, 0.015, 0.02, 0.03)):
        pts[tip] = pts[THUMB_TIP] + [dx, 0, 0]
    ratios = finger_ratios(pts)
    pts[20] = [-0.04, 0.17, 0]  # pinky straightened out
    check("finger ratios: bunched fingertips small, extended pinky large",
          max(ratios) < 0.4 and finger_ratios(pts)[3] > 0.7,
          f"(bunched {np.round(ratios, 2)}, pinky out {finger_ratios(pts)[3]:.2f})")

    from so101_teleop.hand_tracker import FINGER_KNUCKLES, finger_curls

    pts = np.zeros((21, 3))  # wrist at the origin, fingers pointing +Y
    for i, (mcp, tip) in enumerate(zip(FINGER_KNUCKLES, FINGER_TIPS)):
        pts[mcp] = [0.02 * i - 0.03, 0.09, 0]
        pts[tip] = pts[mcp] + [0, 0.075, 0]  # straight finger
    straight = finger_curls(pts)
    for mcp, tip in zip(FINGER_KNUCKLES, FINGER_TIPS):
        pts[tip] = pts[mcp] + [0, -0.035, 0.02]  # folded back into the palm
    fist = finger_curls(pts)
    check("finger curls: straight fingers large, fist small", min(straight) > 1.5 and max(fist) < 0.8,
          f"(straight {np.round(straight, 2)}, fist {np.round(fist, 2)})")


def test_camera():
    from so101_teleop.config import HandConfig
    from so101_teleop.hand_tracker import Camera

    try:
        cam = Camera(HandConfig())
    except RuntimeError as e:
        check("webcam opens", False, str(e))
        return
    t0 = time.perf_counter()
    while cam.latest()[1] is None and time.perf_counter() - t0 < 5:
        time.sleep(0.05)
    n0 = cam.latest()[0]
    time.sleep(2.0)
    fps = (cam.latest()[0] - n0) / 2.0
    frame = cam.latest()[1]
    cam.close()
    check("webcam delivers frames", frame is not None, f"(resolution {cam.resolution}, ~{fps:.0f} FPS)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="also test webcam 0")
    args = ap.parse_args()
    test_model()
    test_ik_grid()
    test_pipeline()
    test_snapshots()
    test_mediapipe()
    if args.camera:
        test_camera()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if failures else 0)
