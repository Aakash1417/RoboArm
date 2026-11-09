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
.\.venv\Scripts\python.exe main.py --ik tip            # original motion: all 5 joints, tilt free
.\.venv\Scripts\python.exe main.py --ik wrist          # hand moves the wrist joint (so101-leap-teleop style)
.\.venv\Scripts\python.exe joint_test.py               # jog each joint by keyboard (milestone 1 check)
```

If you activate the venv first (`.\.venv\Scripts\Activate.ps1`), plain `python main.py` works.
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once,
or just use the full `.\.venv\Scripts\python.exe` path as above.

The gripper follows your thumb-index gap proportionally (`--grip majority` or `--grip index` for
on/off). When your hand leaves the view and comes back, the arm smoothly goes back to where your
hand now maps; `--reanchor` makes it stay put instead.

Useful options: `--sensitivity 1.5`, `--min-cutoff 0.5` (steadier when still, laggier),
`--beta 15` (less lag when moving, a little more jitter),
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
| Comes back somewhere else (after about 0.25 s away) | Robot stays put; your hand's new position becomes neutral, like lifting a mouse off the desk |

The MuJoCo camera starts behind the robot, so "left" on your screen, "left" for your hand and the
robot's own left all agree. If you orbit the camera to the front, the robot will look mirrored.

## 3. How it works

```
webcam (1080p, thread) -> resize to 640 px -> mirror -> MediaPipe HandLandmarker (CPU)
   -> palm centre, palm size, thumb-to-fingertip ratios -> smoothing + dead zone -> relative mapping
   -> speed limit + workspace clamp -> 3D target point -> split IK (or Mink IK with --ik tip)
   -> MuJoCo position actuators -> physics -> viewer
```

- **Hand position** is the average of the wrist and the four finger-base landmarks. That point barely
  moves when you pinch, so pinching does not drag the arm.
- **Mapping is relative (Cartesian):** `target = neutral_target + gain * (hand - neutral_hand)`.
  Neutral is captured on the first detection and on `C`, resume, or after `R`. With default gains,
  moving your hand across the full image width moves the gripper about 0.55 m, which is limited by the
  workspace. +Y is the robot's left, +Z is up, +X is forward. Units are metres; the base sits at the
  origin on a table top at Z = 0.
- **IK, default `--ik jaws`:** the point between the jaws (the model's `gripperframe` site) follows
  your hand 1:1, and the gripper keeps the tilt it has in the home pose (pointing down-forward).
  - Shoulder, elbow and wrist flex all turn about the same axis, so the gripper's tilt in the world
    is their sum. `wrist_flex` is always set to keep that sum at its home value, and `wrist_roll`
    follows your twist.
  - Base, shoulder and elbow are solved with damped least squares to put the jaws on the target,
    with `wrist_flex` moving along with them. The base is first pointed straight at the target and
    each correction step is capped, so a far target can't throw the arm into a folded pose.
  - The tilt holds everywhere except when you push the gripper high and close to the base, where
    `wrist_flex` reaches its limit and the gripper tilts up a little.
  - Limits: a box, a reach shell around the shoulder (35-95% of the upper arm plus forearm length),
    and a floor that keeps the jaws 2 cm above the table. Limits are checked on the wrist position
    that a jaw target implies, since that's what the reach depends on.
  - Each joint moves at most 3 rad/s, and the joint limits keep a 0.05 rad margin.
- **IK, `--ik wrist`:** the same, except the wrist joint follows your hand, as in
  [so101-leap-teleop](https://github.com/Jordaine777/so101-leap-teleop). That project also tracks
  hand tilt, and then the gripper pivoting about its wrist like your hand does is the natural
  choice. Without tilt tracking it's worse: the jaws hang 12.5 cm below and 10 cm in front of the
  wrist, and at home the wrist is only about 6 cm in front of the base axis. Moving it sideways
  turns the base a lot, so the jaws swing roughly twice as far as your hand, on a curve, and the
  red target marker sits at the wrist instead of the jaws.
- **IK, `--ik tip`** (the previous behaviour): [Mink](https://github.com/kevinzakka/mink) solves
  for all 5 arm joints so the model's `gripperframe` site reaches the target point. A light posture
  task keeps the wrist close to its home angle, but the wrist does help with positioning, so the
  gripper's tilt changes as you move. Same joint margins and speed cap, with the older
  workspace box from `workspace_min` / `workspace_max`.
- **Edges:** when the target hits the workspace edge, the neutral point slides with it (also from
  that repo). Moving your hand back responds immediately, instead of first having to return to
  where it crossed the edge.
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
- **Smoothing:** a [One Euro filter](https://gery.casiez.net/1euro/) on the hand (`euro_*` in
  `config.py`). Its cutoff rises with hand speed, so a still hand is smoothed hard and a moving one
  barely lags. A small backlash-style dead zone (`dead_zone`) follows it. The target point then
  moves toward the hand with capped speed (`max_target_speed`, 0.35 m/s) and acceleration
  (`max_target_accel`, 2.5 m/s²), so it eases in and out instead of stepping at the camera's 30 Hz,
  and re-acquiring the hand far away, or pressing `R`, gives a gradual, bounded motion.
- **Measured** (`selftest.py`, synthetic 30 Hz frames): a still hand with MediaPipe-like jitter
  (0.003 image widths) moves the target 0.13 mm rms instead of 1.65 mm; a calm sweep is followed
  75 ms behind. The overlay shows the live camera-to-command latency (capture, MediaPipe, mapping)
  and how far the target trails the hand.
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
- **Hand lost:** the target freezes and the arm holds position. If the hand is back within
  `reanchor_frames` (8 camera frames, about 0.25 s), the old neutral still applies, so a brief
  tracking dropout doesn't lose the movement you made during it. After a longer absence, the
  returning hand re-anchors: wherever it is now maps to where the robot is, so the arm doesn't move.

All tunables live in `so101_teleop/config.py`: gains, thresholds, home pose, workspace box, and speeds.

## 4. Calibrating and tuning

1. Sit about 50-80 cm from the camera, with your palm facing it and roughly in the middle of the image.
2. Press `C`. Small hand movements should now produce small robot movements around its current pose.
3. Watch the `Finger scores` in the overlay. Closed fingers (touching the thumb or curled into a
   fist) should read below 1.0, and an open hand well above 1.4. If a gesture doesn't register,
   raise `touch_below` or `curl_below` in `config.py` a little, or change how many fingers are
   needed with `--grip-fingers`.
4. If the arm shimmers while your hand is still, lower `--min-cutoff` (for example 0.5). If it
   lags when you move, raise `--beta` (for example 15).
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
- The jaws and wrist IKs each reach 150 targets across the workspace (worst error 0.1 mm), and the
  tip IK reaches 100 (95th percentile 0.1 mm). Joints stayed within the model limits for every target.
- The full control loop with MuJoCo physics, driven by scripted hand readings, once with each of the
  three IKs (every check below passes in all three):
  - Hand left/right/up/down moves the gripper left/right/up/down.
  - Calibrating doesn't move the robot.
  - Any 3 of the 4 fingertips at the thumb closes the gripper, 2 of 4 keeps it open, dropping back
    to 2 opens it, and single-frame flicker is ignored. A fist closes it, 2 touching plus 1 curled
    closes it, and half-bent fingers don't. `--grip index` closes on the thumb-index
    pinch alone.
  - Losing the hand holds position (0 mm drift over 3 s).
  - A hand that comes back somewhere else after being lost doesn't move the arm. A 3-frame dropout
    keeps the anchor, so a move made during it still counts.
  - The target never exceeds the 0.35 m/s speed cap.
  - Extreme hand positions stay inside the workspace. From the edge, moving back responds right away.
  - Jaws and wrist IK: the gripper's tilt stays within 0.1° of home, except where `wrist_flex` is
    at its limit. Jaws (and tip) IK: moving the hand 0.15 image widths left moves the jaws straight
    left by the matching 8.2 cm, minus the 2.2 mm dead zone.
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
- **[Jordaine777/so101-leap-teleop](https://github.com/Jordaine777/so101-leap-teleop):** the
  motion ideas behind the split IK: base/shoulder/elbow do the positioning while the wrist joints
  are set directly (`--ik wrist` is that repo's version; the default `--ik jaws` targets the jaws instead), the reach shell around the shoulder, the sliding anchor at the edges,
  and re-anchoring when the hand comes back. It uses a Leap Motion sensor and drives a real arm
  through LeRobot. Neither is used here, and its code was rewritten rather than copied.
- **[sruthihsr/SO101-Gesture-Teleoperation](https://github.com/sruthihsr/SO101-Gesture-Teleoperation):**
  I reviewed it for the overall idea (MediaPipe gestures driving an SO-101 in MuJoCo). It is built on
  ROS 2 and TCP under WSL, so none of its code is used; this project replaces that with one native
  Python process.

## 8. The physical SO-ARM101: motor setup, safe limits, jogging (`arm.py`)

`arm.py` talks to the real follower arm's six Feetech STS3215 servos through the bus adapter board.
`main.py` doesn't use it yet; the hand-tracking app is still simulation only. Every command also
runs with `--fake` (a simulated arm with its own files in `calibration/fake_arm/`), so you can try
the steps first.

**Why limits matter here:** the servo horns were fitted at random angles, so a raw reading says
nothing about where a joint's mechanical stops are. A joint's travel can even cross the encoder's
4095 -> 0 wrap, where a naive program would think it jumped a full turn and drive it the wrong way
into a stop. Calibration fixes both, the way LeRobot does:

1. With every joint at roughly the middle of its travel, each servo's `Homing_Offset` is set so that
   pose reads 2047 (the middle of 0..4095). Each joint's travel is then one continuous range around
   2047, far from the wrap. A guess within about 90° of the true middle is fine.
2. You move each joint by hand to both end stops. The extremes, pulled 3° in (`--margin-deg`),
   are written to the servo's own `Min_Position_Limit` / `Max_Position_Limit` (EEPROM, kept when
   powered off). The servo then refuses goals outside them, whatever program sends them.
3. The values are saved to `calibration/so101_follower.json`, in LeRobot's calibration format.

**Steps:**

1. Wire it: 12 V supply to the adapter board (USB only carries data), USB to the PC, and the 3-pin
   cable from the board to the first motor. `python arm.py ports` should list a COM port. If it
   doesn't, install the board's USB driver (CH340 or CH343 chip, depending on the board).
2. `python arm.py scan` lists every motor that answers: ID, position, limits, voltage, temperature.
   - If only ID 1 answers, the motors are new (they all ship as ID 1). Run
     `python arm.py setup-ids` and connect one motor at a time when asked; it gives gripper = 6,
     wrist roll = 5, wrist flex = 4, elbow = 3, shoulder lift = 2, base = 1 (LeRobot's numbering).
3. `python arm.py identify` keeps torque off and shows every motor's reading live: move a joint by
   hand and its row shows "moving", so you can check that each ID sits on the joint it should.
   After calibrating, `python arm.py identify 3 --wiggle` wiggles motor 3 by ±3.5° at 30 % torque.
   - Before calibration, motors can hold leftover limits that don't fit how the arm was
     assembled; turning torque on would make the servo swing to the nearest one. Every command
     refuses to turn torque on for a motor outside its stored limits (jog allows easing in from
     up to 5° outside, for a joint resting on its end stop).
4. `python arm.py calibrate`: torque goes off on all motors, so hold the arm. Put every joint at
   about the middle of its travel and press Enter, then move each joint slowly to both stops while
   the min/max table fills in, and press Enter. Stop at resistance; don't force it.
   - It refuses to save if a joint barely moved, or if a reading jumped across 4095/0 (the middle
     pose was too far off for that joint: run it again).
5. `python arm.py show` confirms the servos and the file agree.
6. `python arm.py jog`: torque on, holding the current pose.
   - `1`-`6` picks a joint, `LEFT` / `RIGHT` (or `A` / `D`) move it, `UP` / `DOWN` change the step
     (5 to 100 ticks, 0.4° to 9°), `M` sends it to the middle of its range.
   - `SPACE` freezes everything where it is. `T` turns torque off (limp; hold the arm) and on again.
     `Q` quits and asks whether to release torque.
   - Goals are clipped to the calibrated limits, and the servo enforces them again on its side.
     Speed is capped at 40°/s (`--speed-deg`) and torque at 50 % (`--torque`).
   - If a joint pushes harder than 60 % load (`--stall-load`) for 0.3 s without getting there,
     its goal is reset to where it is, so it stops pushing into whatever is in the way.
   - It won't start if the servos don't match the calibration file (for example a motor was
     swapped or re-zeroed).

Turning torque on always sets the goal to the current position first, so a motor never jumps to
an old goal left in its memory.

Tested without hardware against the simulated bus (`selftest.py`): offsets centre every joint at
2047; joints whose travel crosses the wrap are caught without centring and stay continuous with it;
the saved limits equal the end stops minus the margin; a goal past a limit stops at the limit;
EEPROM writes are refused while torque is on; a stale goal doesn't move a motor when torque comes on;
an unmoved joint isn't saved; ID changes work. Jogging was driven with scripted keys: a joint stops
at its max, `M` returns it to the middle, and a blocked joint is stopped. None of this has run on
the real arm yet.

## 9. Driving the physical arm with your hand (`main.py --real`)

The sim model ("new calib") puts each joint's zero at the middle of its range, and calibration
centred each servo's travel, so a sim angle maps to `ticks = middle of range + trim + sign * angle`
(`so101_real/follower.py`). Direction and trim per joint are in `calibration/sim_map.json`.

1. `python arm.py mirror`: torque off, the sim copies the real arm as you move it by hand. Every
   joint must turn the same way as the real one: pick a joint with `1`-`6`, `F` flips it, `[` `]`
   trim it by 2° (the wrist roll usually needs this, it has no end stops to centre on). `S` saves.
   Keys go to the console window.
2. `python main.py --real`: after Enter, torque comes on and the arm moves to the sim's home pose,
   then follows the sim's joint commands at 50 Hz. Speed is capped at 90°/s (`--real-speed`), torque
   at 50 % (`--real-torque`), and every goal is clamped to the calibrated limits. On quit it asks
   whether to keep holding or go limp. `--fake-arm` tries it with the simulated bus.

Start with `--real-speed 30`. `SPACE` (pause) in the hand window freezes the target.

## 10. Files

```
main.py                    app: two windows, control loop, keys, input modes
arm.py                     physical arm: ports, scan, setup-ids, identify, calibrate, show, jog
so101_real/bus.py          STS3215 registers, real serial bus, simulated bus (--fake)
so101_real/calibration.py  centring, range recording, limits, calibration file
so101_real/follower.py     sim angles <-> servo ticks, the arm following the sim (--real)
calibration/               so101_follower.json after you calibrate the real arm
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
