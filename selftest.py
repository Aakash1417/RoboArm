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
from so101_teleop.robot import SO101Sim, make_ik

OUT = Path(__file__).resolve().parent / "selftest_output"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


class Rig:
    """Same control loop as main.py, but driven by scripted observations at 30 Hz."""

    def __init__(self, ik_mode="jaws"):
        self.cfg = Config()
        self.cfg.robot.ik_mode = self.mode = ik_mode
        self.sim = SO101Sim(self.cfg.robot)
        self.ik = make_ik(self.cfg.robot)
        lo, hi = self.ik.joint_limits
        self.teleop = Teleop(self.cfg.mapping, self.cfg.robot, self.ik.home_target, roll_limits=(lo[4], hi[4]),
                             clamp=getattr(self.ik, "clamp", None))
        self.max_tilt_error = 0.0
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
        q = self.sim.joint_q
        if self.mode != "tip" and self.ik.lo[3] + 0.02 < q[3] < self.ik.hi[3] - 0.02:
            self.max_tilt_error = max(self.max_tilt_error, abs(q[1:4].sum() - self.ik.tilt))
        return self.pos

    @property
    def pos(self):
        """The point the hand steers: the wrist joint in wrist mode, otherwise the jaws."""
        return self.sim.control_pos(self.mode)


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


def test_ik_grid(ik_mode):
    rig = Rig(ik_mode)
    ik, clamp = rig.ik, rig.teleop.clamp
    lo, hi = ik.joint_limits
    errs = []
    xs, ys, zs = ((0.06, 0.34, 6), (-0.26, 0.26, 5), (0.03, 0.36, 5)) if ik_mode != "tip" else \
        ((0.14, 0.34, 5), (-0.2, 0.2, 5), (0.03, 0.25, 4))
    for x in np.linspace(*xs):
        for y in np.linspace(*ys):
            for z in np.linspace(*zs):
                tgt = clamp(np.array([x, y, z]))
                ik.reset()
                for _ in range(300):
                    q = ik.solve(tgt, 0.01)
                errs.append(np.linalg.norm(ik.position() - tgt))
                assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)
    errs = np.array(errs)
    check(f"[{ik_mode} IK] reaches workspace targets", np.percentile(errs, 95) < 0.005,
          f"(median {np.median(errs) * 1000:.1f} mm, 95th pct {np.percentile(errs, 95) * 1000:.1f} mm, "
          f"max {errs.max() * 1000:.1f} mm over {len(errs)} targets)")


def test_pipeline(ik_mode):
    rig = Rig(ik_mode)
    tp = rig.teleop
    def check(name, ok, detail=""):
        globals()["check"](f"[{ik_mode} IK] {name}", ok, detail)

    home = rig.run(1.5, hand())
    check("calibration does not move the robot", np.linalg.norm(home - rig.ik.home_target) < 0.01,
          f"(drift {np.linalg.norm(home - rig.ik.home_target) * 1000:.1f} mm)")

    p = rig.run(2.0, hand(x=0.35))
    check("hand left -> gripper moves left (+Y)", p[1] - home[1] > 0.05, f"(dy {p[1] - home[1]:+.3f} m)")
    if ik_mode != "wrist":
        off = p - home - np.array([0, 0.15 * rig.cfg.mapping.gain_xy, 0])
        check("hand left 0.15 -> the jaws move straight left the same amount (1:1, no swing)",
              np.linalg.norm(off) < 0.01, f"(off by {np.round(off * 1000, 1)} mm, the dead zone is 2.2)")
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
    before = rig.pos
    rig.run(3.0, lambda t: None)
    after = rig.pos
    check("hand lost -> robot holds position", np.linalg.norm(after - before) < 0.01,
          f"(moved {np.linalg.norm(after - before) * 1000:.1f} mm in 3 s)")

    p = rig.run(2.0, hand(x=0.9, y=0.1))
    check("hand comes back somewhere else -> arm stays put (re-anchored, like lifting a mouse)",
          np.linalg.norm(p - after) < 0.01, f"(moved {np.linalg.norm(p - after) * 1000:.1f} mm)")
    rig.run(1.0, lambda t: None)
    p0 = rig.run(1.0, hand(x=0.5, y=0.5))
    blink = iter([None] * 3 + [hand(x=0.4, y=0.5)] * 1000)  # 0.1 s dropout, hand reappears 0.1 to the left
    p1 = rig.run(2.0, lambda t: (o := next(blink)) and o(t))
    expected = 0.1 * rig.cfg.mapping.gain_xy
    check("a short dropout (3 frames) keeps the anchor, so the move still counts",
          abs((p1[1] - p0[1]) - expected) < 0.01, f"(dy {p1[1] - p0[1]:+.3f} m, expected {expected:+.3f})")

    rig.run(3.0, hand(x=-2.0, y=-2.0))
    t = tp.target
    check("extreme hand position stays inside the workspace", np.allclose(tp.clamp(t), t, atol=1e-6),
          f"(target {np.round(t, 3)})")
    p = rig.run(1.5, hand(x=-1.8, y=-2.0))
    check("from the workspace edge, moving back responds immediately (sliding anchor)", t[1] - p[1] > 0.05,
          f"(y {t[1]:+.3f} -> {p[1]:+.3f} m)")
    rig.run(1.0, lambda t: None)
    rig.run(1.0, hand())

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
    roll0, pos0 = rig.sim.joint_q[4], rig.pos
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
    x0 = rig.pos[0]
    p = rig.run(2.0, hand(size=0.18))
    check("depth estimate: bigger hand -> forward (+X)", p[0] - x0 > 0.03, f"(dx {p[0] - x0:+.3f} m)")

    check("joints never exceed model limits", rig.limit_violation < 0.02,
          f"(worst overshoot {rig.limit_violation:.4f} rad)")
    check("joint speeds stay moderate", rig.max_joint_speed < 6.0, f"(max {rig.max_joint_speed:.2f} rad/s)")
    check("target point never moves faster than the speed cap",
          rig.max_target_speed <= rig.cfg.robot.max_target_speed + 1e-6,
          f"(max {rig.max_target_speed:.2f} m/s, limit {rig.cfg.robot.max_target_speed} m/s)")
    if ik_mode != "tip":
        check("gripper keeps its home tilt (except where wrist_flex is at its limit)", rig.max_tilt_error < 0.05,
              f"(worst {np.degrees(rig.max_tilt_error):.1f} deg)")


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


def test_arm_tools():
    """Physical-arm tools against the simulated bus (no hardware needed)."""
    from so101_real import calibration as calib
    from so101_real.bus import MIDDLE, BusError, FakeBus, decode, encode

    check("arm: sign-magnitude encoding round-trips (offsets -2047..2047)",
          all(decode(encode(v, 11), 11) == v for v in range(-2047, 2048)) and encode(-5, 11) == 2048 + 5)

    ids = list(calib.JOINTS.values())
    bus = FakeBus()  # horns at random angles: several joints' travel runs across the 4095 -> 0 wrap
    check("arm: scan finds the six motors", sorted(bus.scan()) == ids)
    bus.torque(1, True)
    bus.write(1, "Min_Position_Limit", 100)
    check("arm: EEPROM writes stick and leave the motor limp",
          bus.read(1, "Min_Position_Limit") == 100 and bus.read(1, "Torque_Enable") == 0)
    bus.write(1, "Min_Position_Limit", 0)

    calib.reset(bus, ids)
    rec = calib.RangeRecorder(ids)
    for i in ids:
        lo, hi = bus.stops[i]
        for raw in np.linspace(lo - 50, hi + 50, 60):
            bus.move_by_hand(i, raw)
            rec.update({i: bus.read(i, "Present_Position")})
        bus.move_by_hand(i, (lo + hi) / 2)
    crosses = {i for i, (lo, hi) in bus.stops.items() if lo < 0 or hi > 4095}
    check("arm: without centring, every joint whose travel crosses 4095/0 is caught",
          rec.wrapped == crosses and 5 in crosses, f"(wrapped {sorted(rec.wrapped)}, expected {sorted(crosses)})")

    offsets = calib.center(bus, ids)
    reads = [bus.read(i, "Present_Position") for i in ids]
    check("arm: centring makes the middle pose read 2047 on every motor", all(abs(r - MIDDLE) <= 1 for r in reads),
          f"(offsets {offsets})")
    rec = calib.RangeRecorder(ids)
    for i in ids:
        lo, hi = bus.stops[i]
        for raw in np.linspace(lo - 50, hi + 50, 60):  # pushing past the stops does nothing
            bus.move_by_hand(i, raw)
            rec.update({i: bus.read(i, "Present_Position")})
    margin = 34  # 3 deg
    limits, problems = calib.limits_from_recording(rec, margin)
    expected = {i: (round(lo - offsets[i]) + margin, round(hi - offsets[i]) - margin) for i, (lo, hi) in bus.stops.items()}
    check("arm: recorded limits = the hard stops pulled in by the margin, no wrap after centring",
          not problems and all(abs(limits[i][0] - expected[i][0]) <= 1 and abs(limits[i][1] - expected[i][1]) <= 1
                               for i in ids), f"({problems or limits[5]})")
    calib.write_limits(bus, limits)
    cal = {n: calib.JointCalibration(i, 0, offsets[i], *limits[i]) for n, i in calib.JOINTS.items()}
    check("arm: servos match the calibration", calib.check(bus, cal) == [])

    i = 3
    bus.move_by_hand(i, (bus.stops[i][0] + bus.stops[i][1]) / 2)
    bus.write(i, "Goal_Position", 0)  # a stale goal, far outside the limits
    bus.torque(i, True)
    time.sleep(0.05)
    still = abs(bus.read(i, "Present_Position") - MIDDLE) < 5
    bus.write(i, "Goal_Position", 4095)
    for _ in range(40):
        bus.read(i, "Present_Position")
        time.sleep(0.01)
    check("arm: torque-on holds the pose (ignores a stale goal); a goal past the limit stops at the limit",
          still and bus.read(i, "Present_Position") == limits[i][1], f"(at {bus.read(i, 'Present_Position')}, "
                                                                        f"max {limits[i][1]})")
    bus.torque_off(i)

    i = 2
    bus.move_by_hand(i, bus.stops[i][1] + 100)  # resting on its end stop, past the margin
    try:
        bus.torque(i, True)
        refused = False
    except BusError:
        refused = True
    bus.torque(i, True, max_snap=60)
    for _ in range(40):
        bus.read(i, "Present_Position")
        time.sleep(0.01)
    check("arm: torque-on is refused outside the limits; jog's small ease-in moves it to the limit",
          refused and bus.read(i, "Present_Position") == limits[i][1], f"(at {bus.read(i, 'Present_Position')})")
    bus.torque_off(i)
    odd = FakeBus(raw={1: 3877})  # like a real scan: old limits 3997..2278 (min above max)
    odd.mem[1][9], odd.mem[1][11] = 3997, 2278
    try:
        odd.torque(1, True)
        refused = False
    except BusError:
        refused = True
    check("arm: torque-on is refused when the stored limits are nonsense (min above max)", refused)

    rec = calib.RangeRecorder([1])
    rec.update({1: 2000})
    rec.update({1: 2100})
    _, problems = calib.limits_from_recording(rec, margin)
    check("arm: a joint that was barely moved is not saved", len(problems) == 1)

    bus = FakeBus(raw={1: 2000})
    bus.change_id(1, 4)
    check("arm: changing a motor's ID", sorted(bus.scan()) == [4])


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
    for mode in ("jaws", "wrist", "tip"):
        test_ik_grid(mode)
        test_pipeline(mode)
    test_snapshots()
    test_mediapipe()
    test_arm_tools()
    if args.camera:
        test_camera()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if failures else 0)
