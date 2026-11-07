# Hand-controlled SO-ARM101 simulation (Windows, no ROS)

Move your hand in front of an ordinary webcam and a simulated SO-101 (SO-ARM101 follower) arm
follows it in MuJoCo. Close most of your fingers (3 of 4), either by touching their tips to your
thumb or by curling them into a fist, to close the gripper
(or use `--grip index` for a simple thumb-index pinch).

- **Window 1 - Hand Tracking (OpenCV):** mirrored webcam feed, MediaPipe hand landmarks, a line
  from the thumb to each fingertip (red = that finger counts as closed), tracking status, gripper
  state, per-finger scores,
  FPS, target position.
- **Window 2 - MuJoCo:** the official SO-101 3D model on a table with a small blue cube. The
  translucent red ball is the point your hand is commanding.

Simulation only. No serial port is opened and nothing is sent to a physical robot.

## 1. Install (PowerShell, once)

Requires Python 3.11 (`py -3.11 --version` should work). No CUDA, GPU drivers, ROS, WSL or Docker.

```powershell
cd C:\Aakash\robo
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `.venv`, installs `requirements.txt`, downloads the MediaPipe hand model into
`models\` (already present in this folder), and runs `selftest.py`. You should see
`ALL CHECKS PASSED`.

Manual equivalent:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe selftest.py
```

## 2. Run

```powershell
.\.venv\Scripts\python.exe main.py                     # webcam 0
.\.venv\Scripts\python.exe main.py --camera 1          # a different webcam
.\.venv\Scripts\python.exe main.py --input keyboard    # no webcam: WASD moves a virtual hand
.\.venv\Scripts\python.exe main.py --input synthetic   # no webcam: scripted demo motion
.\.venv\Scripts\python.exe main.py --grip index        # close on thumb-index pinch only
.\.venv\Scripts\python.exe joint_test.py               # jog each joint by keyboard (milestone 1 check)
```

If you activate the venv first (`.\.venv\Scripts\Activate.ps1`), plain `python main.py` works.
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once,
or just use the full `.\.venv\Scripts\python.exe` path as above.

Useful options: `--sensitivity 1.5`, `--smoothing 0.2` (lower = smoother, laggier),
`--speed 0.2` (max target speed in m/s), `--depth` (forward/back estimate on at start),
`--width 1280 --height 720` (capture size), `--process-width 480` (faster tracking),
`--backend dshow` (alternative camera driver).

### Controls (click the Hand Tracking window first so it has keyboard focus)

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `SPACE` | Pause / resume. The robot holds still; on resume your current hand position becomes the new neutral, so it does not jump |
| `C` | Calibrate: your current hand position becomes neutral for wherever the robot is now (no jump) |
| `R` | Return smoothly to the home pose, then re-calibrate |
| `F` | Toggle forward/back control from hand size (**an estimate, not real depth**) |
| `T` | Toggle wrist roll following your hand twist (on by default; `--no-wrist` starts it off) |
| `+` / `-` | Increase / decrease sensitivity |
| `W A S D`, `E` / `X`, `[` / `]`, `G`, `H` | Keyboard mode only: move, forward/back, twist, toggle grip, hide hand |

In the MuJoCo window: left-drag rotates, right-drag pans, scroll zooms, double-click selects a body.
The MuJoCo window has its own built-in shortcuts, so use the Hand Tracking window for the keys above.

| Your hand | Simulated robot |
|---|---|
| Moves left / right | Gripper moves left / right |
| Moves up / down | Gripper moves up / down |
| Moves toward / away from the camera (with `F`) | Gripper moves forward / back (estimated) |
| Twists clockwise / counter-clockwise (like a doorknob, palm toward the camera) | Wrist rolls the same way, same angle |
| At least 3 of the 4 fingers closed (tip touching the thumb, or curled into a fist, in any mix) | Gripper closes |
| Two or more fingers open again | Gripper opens |
| Leaves the view | Robot holds its current position and gripper state |

The MuJoCo camera starts behind the robot, so "left" on your screen, "left" for your hand and the
robot's own left all agree. If you orbit the camera to the front, the robot will look mirrored.

## 3. How it works

```
webcam (1080p, thread) -> resize to 640 px -> mirror -> MediaPipe HandLandmarker (CPU)
   -> palm centre, palm size, thumb-to-fingertip ratios -> smoothing + dead zone -> relative mapping
   -> speed limit + workspace clamp -> 3D target point -> Mink IK (joint limits, velocity limits)
   -> MuJoCo position actuators -> physics -> viewer
```

- **Hand position** is the average of the wrist and the four finger-base landmarks. That point barely
  moves when you pinch, so pinching does not drag the arm.
- **Mapping is relative (Cartesian):** `target = neutral_target + gain * (hand - neutral_hand)`.
  Neutral is captured on the first detection and on `C`, resume, or after `R`. With default gains,
  moving your hand across the full image width moves the gripper about 0.55 m, which is limited by the
  workspace. +Y is the robot's left, +Z is up, +X is forward. Units are metres; the base sits at the
  origin on a table top at Z = 0.
- **IK:** [Mink](https://github.com/kevinzakka/mink) solves for the 5 arm joints so the model's
  `gripperframe` site reaches the target point. Only position is controlled. The SO-101 has 5 arm
  joints, so full 6D pose control isn't always possible. A light posture task keeps the wrist close
  to its home angle. The joint limits come from the model, with a 0.05 rad margin, and joint speed is
  capped at 3 rad/s. This is true end-effector control: the gripper moves left, not a single joint.
- **Wrist roll:** the hand twist is the angle of the line from your wrist to your middle-finger
  knuckle in the mirrored image. Those points don't move when you bunch your fingers, so gripping
  doesn't twist the wrist.
  - The robot's `wrist_roll` joint moves by the same angle, measured from its value at
    calibration, so twisting 30° clockwise turns the gripper 30° clockwise as seen from the default
    camera.
  - It uses the same smoothing, dead zone (about 2°), hold-on-loss and calibration as the position.
    It's capped at 3 rad/s and kept inside the joint's limits, minus a 0.05 rad margin.
  - The IK keeps the gripper point in place while the wrist turns.
  - The home roll is 90° (`1.5708` rad) so the jaw opens sideways, where you can see it. That
    leaves about 70° of counter-clockwise twist before the joint limit, and much more clockwise.
    Change `home_q[4]` in `config.py` for a different balance.
  - Twisting is measured in the image plane, so it works best with your palm or the back of your
    hand toward the camera. If your fingers point straight at the camera, the angle is unreliable.
- **Depth (optional)** uses the change in apparent palm size relative to calibration. It's noisy,
  and it also changes if you tilt your hand. It isn't depth-camera data.
- **Smoothing:** an exponential moving average (`smoothing`), followed by a backlash-style dead zone
  (`dead_zone`). The dead zone ignores tiny jitter without causing steps. On top of that, the target
  point can't move faster than `max_target_speed` (0.35 m/s), so re-acquiring the hand far away, or
  pressing `R`, produces a gradual, bounded motion.
- **Gripper:** each of index, middle, ring and pinky counts as closed in either of two ways. Both
  use MediaPipe's 3D hand landmarks and divide one hand distance by another, so they don't depend
  on how far your hand is from the camera, and tilting your hand doesn't make fingers look closer
  together than they are.
  - **Touching the thumb:** the thumb-tip to fingertip distance ÷ palm length (wrist to
    middle-finger knuckle) is below `touch_below` (0.45).
  - **Curled into a fist:** the fingertip-to-wrist distance ÷ that finger's knuckle-to-wrist
    distance is below `curl_below` (1.0), meaning the tip has folded back past the knuckle. A
    straight finger measures about 1.8 and a fist about 0.7. Half-bent fingers (about 1.2) don't
    count.
  - Each finger gets a score: the smaller of those two measurements, each divided by its threshold.
    Below 1.0 means closed. This is the "Finger scores" line in the overlay.
  - In the default `majority` mode, the gripper closes when at least 3 fingers score below 1.0, in
    any mix of touching and curled (`grip_min_fingers = 3`, or `--grip-fingers N`). It opens once
    fewer than 3 are below `open_score` (1.4). The gap between 1.0 and 1.4 stops flicker. Set the
    count to 4 to require every finger, or 2 for thumb plus any two.
  - `--grip index` uses the thumb-index ratio alone (closes below 0.30, opens above 0.45).
  - In both modes, a change must persist for 3 frames. The gripper joint then ramps at 4 rad/s.
  - This adds a few distance calculations per frame on top of MediaPipe, so there's no measurable
    cost.
- **Gripper values** come from the MuJoCo model, not LeRobot's 0-100 scale. The `gripper` joint
  range is -0.175 to 1.745 rad, and larger values mean more open. The app uses 1.2 rad for open and
  -0.1 rad for closed; the slight squeeze helps hold the cube.
- **Hand lost:** the target freezes and the arm holds position. When the hand returns, the filter
  restarts and the arm moves toward the new position at the capped speed.

All tunables live in `so101_teleop/config.py`: gains, thresholds, home pose, workspace box, and speeds.

## 4. Calibrating and tuning

1. Sit about 50-80 cm from the camera, with your palm facing it and roughly in the middle of the image.
2. Press `C`. Small hand movements should now produce small robot movements around its current pose.
3. Watch the `Finger scores` in the overlay. Closed fingers (touching the thumb or curled into a
   fist) should read below 1.0, and an open hand well above 1.4. If a gesture doesn't register,
   raise `touch_below` or `curl_below` in `config.py` a little, or change how many fingers are
   needed with `--grip-fingers`.
4. If the motion is too twitchy, lower `--smoothing` (for example 0.2) or raise `dead_zone`. If it's
   too laggy, raise `--smoothing` (for example 0.5).
5. If your movements feel too small or too big, use `+` / `-`, or `--sensitivity`.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open webcam index 0` | Close Zoom, Teams, Camera app, OBS, etc. Try `--camera 1`. Check *Settings > Privacy > Camera > Allow desktop apps*. |
| Webcam takes ~20 s to open | Make sure you start via `main.py`, which sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before OpenCV loads. |
| Low camera FPS (about 5-10) | `--backend dshow` often falls back to uncompressed video at 1080p. Use the default `msmf`, or `--width 1280 --height 720`. Low room light also lowers webcam FPS through auto-exposure, so add light. |
| Hand flickers or is often lost | Add light, use a plain background, keep the whole hand in frame, and move a bit slower. |
| Gripper toggles unexpectedly | Raise `pinch_frames` or widen the gap between the close and open thresholds in `config.py`. |
| Gripper won't close | Check the finger scores in the overlay. Raise `touch_below` or `curl_below`, use `--grip-fingers 2`, or use `--grip index`. |
| Gripper closes when you don't want it to | Lower `curl_below` (for example 0.9) if relaxed, slightly bent fingers count as curled. |
| Robot drifts toward an edge | Press `C` with your hand at a comfortable centre, or `R` to go home. |
| Keys do nothing | Click the Hand Tracking window. Keys typed into the MuJoCo window go to MuJoCo. |
| MuJoCo window is black or crashes | Update the NVIDIA driver. MuJoCo needs OpenGL 1.5+. |
| `ModuleNotFoundError` | You're using the wrong Python. Run with `.\.venv\Scripts\python.exe`. |
| `selftest.py` | Run it any time. `--camera` also tests that webcam 0 delivers frames. |

## 6. What has been tested (on this PC, Windows 10, Python 3.11.9)

Tested automatically (`selftest.py`, all passing):
- The official SO-101 model loads. The six actuator ranges match the joint ranges, as listed in the
  self-test output.
- IK reaches 100 targets across the workspace (median error 0.0 mm, 95th percentile 0.1 mm). Joints
  stayed within the model limits for every target.
- The full control loop with MuJoCo physics, driven by scripted hand readings:
  - Hand left/right/up/down moves the gripper left/right/up/down.
  - Calibrating doesn't move the robot.
  - Any 3 of the 4 fingertips at the thumb closes the gripper, 2 of 4 keeps it open, dropping back
    to 2 opens it, and single-frame flicker is ignored. A fist closes it, 2 touching plus 1 curled
    closes it, and half-bent fingers don't. `--grip index` closes on the thumb-index
    pinch alone.
  - Losing the hand holds position (0 mm drift over 3 s).
  - Re-acquiring the hand far away stays at or below the 0.35 m/s speed cap.
  - Extreme hand positions stay inside the workspace.
  - Pause holds, resume doesn't jump, and `R` returns home.
  - A clockwise hand twist turns `wrist_roll` by the same angle without moving the gripper point.
    A large twist stops at the joint limit, the roll holds when the hand is lost, `T` turns it
    off, and `R` returns the roll home. A render confirmed the on-screen direction.
  - The depth estimate moves the gripper forward.
  - No joint-limit overshoot occurred.
- The MediaPipe model loads and runs in about 12 ms per 640x360 frame on CPU.

Tested by running the app:
- Webcam 0 opened at 1920x1080 with Media Foundation, ready in about 4 s.
- A 15 s webcam run processed about 27 frames/s, and MediaPipe detected a hand in 72% of frames.
- `--input synthetic` and `--input keyboard` both ran with the two windows open. `joint_test.py`
  started and ran without errors.
- Rendered snapshots confirmed the arm visibly moves in the right directions and the jaw opens and
  closes. `selftest.py` writes these to `selftest_output/sim_snapshots.png`.

Still needs you, in front of the camera:
- Checking that the grip thresholds (touch 0.45, curl 1.0, open 1.4) suit your hand. They're set from typical
  hand proportions, not measured on a real hand. The values are shown live; tune in `config.py`.
- Checking how natural it feels, including gains and smoothing.
- Trying the depth estimate. It's off by default because hand-size depth is noisy.
- Picking up the cube. Contact grasping with the mesh-based jaw collisions works only sometimes;
  pushing the cube is easy.

## 7. Reused projects

- **[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101`**
  (Apache-2.0): `so101_new_calib.xml` and all STL meshes are copied into `assets/so101/`, unchanged
  except for one edit. The collision class got `contype="2" conaffinity="1"` so the robot's
  convex-hull collision meshes don't collide with each other but still touch the table and cube.
  The joint names, ranges, actuators (`sts3215` class gains), and calibration ("new calib",
  matching LeRobot's current convention) are the model's own. `scene_tabletop.xml` adds the table,
  cube, lights, and target marker.
- **[MuJoCo](https://github.com/google-deepmind/mujoco):** physics, rendering, and the interactive
  passive viewer.
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** `HandLandmarker` (Tasks API) with
  Google's pretrained `hand_landmarker.task`. MediaPipe 1.x no longer ships the old
  `mp.solutions.hands`.
- **[Mink](https://github.com/kevinzakka/mink):** differential IK (FrameTask, PostureTask,
  ConfigurationLimit, VelocityLimit) with the DAQP solver.
- **[sruthihsr/SO101-Gesture-Teleoperation](https://github.com/sruthihsr/SO101-Gesture-Teleoperation):**
  I reviewed it for the overall idea (MediaPipe gestures driving an SO-101 in MuJoCo). It is built on
  ROS 2 and TCP under WSL, so none of its code is used; this project replaces that with one native
  Python process.

## 8. Moving to the physical SO-ARM101 later (not implemented)

The output of this app each tick is `arm_q` (5 joint angles in radians, in the model's "new calib"
convention) plus `gripper_cmd` (radians). To drive the real follower arm:

1. Use [LeRobot](https://github.com/huggingface/lerobot)'s `SO101Follower` (Feetech STS3215 bus)
   instead of writing serial code, and calibrate the arm with LeRobot's calibration tool first.
2. Convert units. With `use_degrees=True`, LeRobot's arm joints are in degrees
   (`np.degrees(arm_q)`); otherwise they use a -100..100 scale. Its gripper is 0-100, so map this
   model's closed (about 0 rad) to open (about 1.745 rad) onto that range, then check
   the direction and zero offset of each joint against the simulation with `joint_test.py` before
   enabling torque.
3. Send `arm_q` only after the simulation has accepted it, and keep all the limits: joint margins,
   the speed cap, and the workspace box. Start with `--speed 0.1` and lower gains.
4. Add a hardware e-stop or torque-off key, and rate-limit bus writes to about 30-50 Hz.
5. Mirror the real joint readings back into the simulation for comparison.

## 9. Files

```
main.py                    app: two windows, control loop, keys, input modes
joint_test.py              milestone 1: jog joints, see ranges
selftest.py                headless checks (no webcam needed); --camera to test webcam
setup.ps1                  one-time Windows setup
requirements.txt           pinned dependencies
so101_teleop/config.py     all tunable settings
so101_teleop/robot.py      MuJoCo sim + Mink IK
so101_teleop/hand_tracker.py  webcam thread, MediaPipe, landmark drawing
so101_teleop/mapping.py    smoothing, calibration, limits, gripper hysteresis
assets/so101/              official SO-101 model + meshes, tabletop scene
models/hand_landmarker.task   MediaPipe pretrained model
```

`_so_arm100_src/` (upstream sparse clone) and `_scratch/` (my test renders) aren't needed to run the
app and can be deleted.
