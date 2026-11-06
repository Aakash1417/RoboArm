"""Milestone 1 check: jog each SO-101 joint with the keyboard and verify directions and ranges.

    python joint_test.py

Keys (focus the "Joint Test" window): 1-6 select joint, W / S move it +/-, H home, Z all zeros,
O / P gripper open / closed, Q quit.
"""

import time

import cv2
import mujoco.viewer
import numpy as np

from so101_teleop.config import JOINT_NAMES, RobotConfig
from so101_teleop.robot import SO101Sim

WINDOW = "Joint Test"
STEP = 0.05  # rad per key press


def main():
    cfg = RobotConfig()
    sim = SO101Sim(cfg)
    cmd = np.append(cfg.home_q, cfg.gripper_open)
    selected = 0
    cv2.namedWindow(WINDOW)

    with mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.17, 0.0, 0.08]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = -20
            viewer.cam.elevation = -30
        last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now
            sim.set_command(cmd[:5], cmd[5])
            sim.set_target_marker(sim.ee_pos)
            sim.step(dt)
            viewer.sync()

            img = np.full((260, 620, 3), 30, np.uint8)
            q = sim.joint_q
            for i, name in enumerate(JOINT_NAMES):
                lo, hi = sim.ctrl_lo[i], sim.ctrl_hi[i]
                text = f"{i + 1} {name:<13} cmd {cmd[i]:+.2f}  actual {q[i]:+.2f}  range [{lo:+.2f}, {hi:+.2f}]"
                color = (0, 255, 255) if i == selected else (220, 220, 220)
                cv2.putText(img, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            ee = sim.ee_pos
            cv2.putText(img, f"gripper site x/y/z: {ee[0]:+.3f} {ee[1]:+.3f} {ee[2]:+.3f} m", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(img, "1-6 select  W/S move  H home  Z zeros  O/P grip open/close  Q quit", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, img)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                selected = key - ord("1")
            elif key == ord("w"):
                cmd[selected] += STEP
            elif key == ord("s"):
                cmd[selected] -= STEP
            elif key == ord("h"):
                cmd = np.append(cfg.home_q, cfg.gripper_open)
            elif key == ord("z"):
                cmd = np.zeros(6)
            elif key == ord("o"):
                cmd[5] = cfg.gripper_open
            elif key == ord("p"):
                cmd[5] = cfg.gripper_closed
            cmd = np.clip(cmd, sim.ctrl_lo, sim.ctrl_hi)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
