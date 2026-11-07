"""Hand-controlled SO-101 (SO-ARM101) simulation.

    python main.py                    # webcam hand tracking (default camera 0)
    python main.py --camera 1         # another webcam
    python main.py --input keyboard   # no webcam: move a virtual hand with keys
    python main.py --input synthetic  # no webcam: scripted hand motion demo

Simulation only. Nothing is ever sent to a physical robot.
"""

import os

# Must be set before cv2 is imported, otherwise Media Foundation can take ~20 s to open a webcam.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import mujoco
import mujoco.viewer
import numpy as np

from so101_teleop.config import Config
from so101_teleop.mapping import HandObservation, Teleop
from so101_teleop.robot import CartesianIK, SO101Sim

WINDOW = "Hand Tracking (keys work when this window is focused)"
CONTROL_HZ = 100.0


class WebcamSource:
    def __init__(self, cfg):
        from so101_teleop.hand_tracker import Camera, HandTracker, draw_hand

        self.cfg = cfg
        self.camera = Camera(cfg)
        self.tracker = HandTracker(cfg)
        self.draw_hand = draw_hand
        self._last_id = 0
        print(f"Webcam {cfg.camera_index} opened with {self.camera.backend} at "
              f"{self.camera.resolution[0]}x{self.camera.resolution[1]}, processing at width {cfg.process_width}.")

    def poll(self):
        """Returns (frame, observation, landmark_pixels) for a new frame, or None if no new frame."""
        frame_id, frame = self.camera.latest()
        if frame is None or frame_id == self._last_id:
            return None
        self._last_id = frame_id
        h, w = frame.shape[:2]
        if w > self.cfg.process_width:
            frame = cv2.resize(frame, (self.cfg.process_width, int(h * self.cfg.process_width / w)),
                               interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)  # mirror so moving your hand left moves it left on screen
        obs, px = self.tracker.detect(frame)
        return frame, obs, px

    def handle_key(self, key):
        return False

    def close(self):
        self.camera.close()
        self.tracker.close()


class VirtualHandSource:
    """Stand-in for the webcam. Produces the same HandObservation the tracker would."""

    def __init__(self, mode, fps=30.0):
        self.mode = mode
        self.period = 1.0 / fps
        self._next = time.perf_counter()
        self._t0 = time.perf_counter()
        self.x, self.y, self.size, self.pinched, self.visible = 0.5, 0.5, 0.12, False, True
        self.draw_hand = None

    def poll(self):
        now = time.perf_counter()
        if now < self._next:
            return None
        self._next = max(self._next + self.period, now - self.period)
        if self.mode == "synthetic":
            t = now - self._t0
            self.x = 0.5 + 0.20 * math.sin(2 * math.pi * t / 6.0)
            self.y = 0.5 + 0.15 * math.sin(2 * math.pi * t / 4.0)
            self.pinched = (t % 6.0) > 3.0
            self.visible = (t % 15.0) < 13.0  # hand "disappears" for 2 s every 15 s
        frame = np.full((360, 640, 3), 40, np.uint8)
        cv2.putText(frame, f"{self.mode.upper()} INPUT (no webcam)", (10, 340), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
        if not self.visible:
            return frame, None, None
        c = (int(self.x * 640), int(self.y * 360))
        r = int(self.size * 640 / 2)
        cv2.circle(frame, c, r, (0, 0, 255) if self.pinched else (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, c, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        r = 0.1 if self.pinched else 0.8
        obs = HandObservation(self.x, self.y, self.size, r, (r, r, r, r))
        return frame, obs, None

    def handle_key(self, key):
        if self.mode != "keyboard":
            return False
        step = 0.02
        moves = {ord("a"): (-step, 0), ord("d"): (step, 0), ord("w"): (0, -step), ord("s"): (0, step)}
        if key in moves:
            self.x = float(np.clip(self.x + moves[key][0], 0, 1))
            self.y = float(np.clip(self.y + moves[key][1], 0, 1))
        elif key == ord("e"):
            self.size = min(self.size * 1.05, 0.5)
        elif key == ord("x"):
            self.size = max(self.size / 1.05, 0.03)
        elif key == ord("g"):
            self.pinched = not self.pinched
        elif key == ord("h"):
            self.visible = not self.visible
        else:
            return False
        return True

    def close(self):
        pass


def draw_hud(img, teleop, sim, mcfg, cam_fps, mode):
    lines = [
        (f"Status: {teleop.status}", (0, 255, 0) if teleop.status == "TRACKING" else (0, 200, 255)),
        (f"Gripper: {'CLOSED' if teleop.gripper.closed else 'OPEN'}",
         (0, 0, 255) if teleop.gripper.closed else (0, 255, 0)),
        (f"Grip ratio ({'all fingers' if mcfg.grip_mode == 'all' else 'index pinch'}): {teleop.last_pinch:.2f}"
         f"  (close<{mcfg.grip_thresholds[0]:.2f} open>{mcfg.grip_thresholds[1]:.2f})", (255, 255, 255)),
        ("Thumb to  " + "  ".join(f"{n} {r:.2f}" for n, r in zip(("index", "middle", "ring", "pinky"),
                                                                teleop.last_fingers)), (200, 200, 200)),
        (f"Camera FPS: {cam_fps:4.1f}   Sensitivity: x{teleop.sensitivity:.2f}", (255, 255, 255)),
        (f"Depth (forward/back): {'ON - ESTIMATE from hand size' if mcfg.depth_enabled else 'OFF'}",
         (255, 255, 0) if mcfg.depth_enabled else (180, 180, 180)),
        ("Target x/y/z [m]: " + " ".join(f"{v:+.3f}" for v in teleop.target), (255, 255, 255)),
        ("Gripper   x/y/z [m]: " + " ".join(f"{v:+.3f}" for v in sim.ee_pos), (200, 200, 200)),
    ]
    h, w = img.shape[:2]
    scale = 0.55 * w / 960
    line_h = int(26 * w / 960)
    pad = int(8 * w / 960)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_w = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t, _ in lines)
    shade(img, 0, 0, text_w + 2 * pad, line_h * len(lines) + pad)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (pad, line_h * (i + 1) - pad // 2), font, scale, color, 1, cv2.LINE_AA)

    help_text = "Q quit  SPACE pause  C calibrate  R home  F depth  +/- sensitivity"
    if mode == "keyboard":
        help_text += "  | WASD move  E/X fwd/back  G grip  H hide hand"
    help_scale = 0.85 * scale
    shade(img, 0, h - line_h - pad, w, h)
    cv2.putText(img, help_text, (pad, h - pad - line_h // 4), font, help_scale, (255, 255, 255), 1, cv2.LINE_AA)


def shade(img, x0, y0, x1, y1, alpha=0.55):
    """Darken a rectangle so text on top stays readable over any background."""
    roi = img[max(y0, 0):y1, max(x0, 0):x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description="Hand-controlled SO-101 MuJoCo simulation")
    p.add_argument("--input", choices=["webcam", "keyboard", "synthetic"], default="webcam")
    p.add_argument("--camera", type=int, default=0, help="webcam index (0, 1, 2, ...)")
    p.add_argument("--backend", choices=["msmf", "dshow", "any"], default="msmf",
                   help="Windows camera backend (try dshow if msmf fails)")
    p.add_argument("--width", type=int, default=1920, help="requested capture width")
    p.add_argument("--height", type=int, default=1080, help="requested capture height")
    p.add_argument("--process-width", type=int, default=640, help="resize frames to this width for tracking")
    p.add_argument("--sensitivity", type=float, default=1.0, help="multiplier on hand-to-robot gain")
    p.add_argument("--smoothing", type=float, default=None, help="EMA factor 0..1 (lower = smoother)")
    p.add_argument("--speed", type=float, default=None, help="max gripper target speed in m/s")
    p.add_argument("--depth", action="store_true", help="enable forward/back from hand size (estimate)")
    p.add_argument("--grip", choices=["all", "index"], default="all",
                   help="close the gripper when all fingertips touch the thumb, or on thumb-index pinch only")
    p.add_argument("--duration", type=float, default=0, help="auto-quit after N seconds (0 = never)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.hand.camera_index = args.camera
    cfg.hand.backend = args.backend
    cfg.hand.request_width, cfg.hand.request_height = args.width, args.height
    cfg.hand.process_width = args.process_width
    cfg.mapping.depth_enabled = args.depth
    cfg.mapping.grip_mode = args.grip
    if args.smoothing is not None:
        cfg.mapping.smoothing = float(np.clip(args.smoothing, 0.01, 1.0))
    if args.speed is not None:
        cfg.robot.max_target_speed = args.speed

    sim = SO101Sim(cfg.robot)
    ik = CartesianIK(cfg.robot)
    teleop = Teleop(cfg.mapping, cfg.robot, ik.home_target)
    teleop.sensitivity = args.sensitivity
    source = WebcamSource(cfg.hand) if args.input == "webcam" else VirtualHandSource(args.input)

    # AUTOSIZE shows the image 1:1, so overlay text is drawn at screen resolution and stays sharp.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    screen_w = ctypes.windll.user32.GetSystemMetrics(0) if sys.platform == "win32" else 1920
    display_w = min(960, screen_w // 2 - 20)
    cv2.moveWindow(WINDOW, screen_w - display_w - 20, 40)  # MuJoCo opens on the left, webcam on the right
    gripper_cmd = cfg.robot.gripper_open
    cam_fps, last_frame_t = 0.0, None
    n_frames = n_hand = n_ticks = 0
    print("Controls (focus the hand-tracking window): Q quit, SPACE pause, C calibrate, R home, "
          "F depth, +/- sensitivity. Drag in the MuJoCo window to rotate the view.")

    with mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.17, 0.0, 0.08]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = -20   # behind the robot, so screen-left is the robot's left
            viewer.cam.elevation = -30

        start = last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now

            polled = source.poll()
            if polled is not None:
                frame, obs, px = polled
                if last_frame_t is not None:
                    cam_fps += 0.1 * (1.0 / max(now - last_frame_t, 1e-3) - cam_fps)
                last_frame_t = now
                teleop.observe(obs)
                n_frames += 1
                n_hand += obs is not None
                zoom = display_w / frame.shape[1]
                view = cv2.resize(frame, (display_w, round(frame.shape[0] * zoom)), interpolation=cv2.INTER_LINEAR)
                if px is not None:
                    close_below = cfg.mapping.grip_thresholds[0]
                    fingers = [r < close_below for r in obs.finger_ratios]
                    if cfg.mapping.grip_mode != "all":
                        fingers = fingers[:1]
                    source.draw_hand(view, px * zoom, fingers, teleop.gripper.closed)
                draw_hud(view, teleop, sim, cfg.mapping, cam_fps, args.input)
                cv2.imshow(WINDOW, view)

            target = teleop.step(dt)
            arm_q = ik.solve(target, dt)
            want = cfg.robot.gripper_closed if teleop.gripper.closed else cfg.robot.gripper_open
            gripper_cmd += np.clip(want - gripper_cmd, -cfg.robot.gripper_speed * dt, cfg.robot.gripper_speed * dt)
            sim.set_command(arm_q, gripper_cmd)
            sim.set_target_marker(target)
            sim.step(dt)
            viewer.sync()
            n_ticks += 1

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not source.handle_key(key):
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == ord(" "):
                    teleop.toggle_pause()
                elif key in (ord("c"), ord("C")):
                    teleop.calibrate()
                elif key in (ord("r"), ord("R")):
                    teleop.reset_home()
                elif key in (ord("f"), ord("F")):
                    teleop.toggle_depth()
                elif key in (ord("+"), ord("=")):
                    teleop.change_sensitivity(1.25)
                elif key in (ord("-"), ord("_")):
                    teleop.change_sensitivity(0.8)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.duration and now - start > args.duration:
                break

            spare = 1.0 / CONTROL_HZ - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

    elapsed = max(time.perf_counter() - start, 1e-6)
    print(f"Ran {elapsed:.1f} s: {n_frames / elapsed:.1f} camera frames/s processed, "
          f"{n_ticks / elapsed:.0f} control updates/s, hand detected in {100 * n_hand / max(n_frames, 1):.0f}% of frames.")
    source.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
