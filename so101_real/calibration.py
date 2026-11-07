"""Per-motor calibration: centre each joint's readings, record its real range, store limits on the servo.

Why: the servo horns were fitted at arbitrary angles, so raw encoder values say nothing about
where a joint's mechanical stops are, and a joint's travel may even cross the encoder's 4095 -> 0
wrap. Calibration fixes both:
  1. With the arm at the middle of every joint's travel, set Homing_Offset so that pose reads 2047.
     Every joint's travel is then a continuous range around 2047, far from the wrap.
  2. Move each joint by hand to both mechanical stops; the extremes, pulled in by a safety margin,
     become Min/Max_Position_Limit in the servo's EEPROM. The servo itself then refuses goals
     outside them, whatever program talks to it.
The file format matches LeRobot's calibration JSON, so LeRobot can use it too.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .bus import MIDDLE, RESOLUTION, BusError

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = ROOT / "calibration" / "so101_follower.json"
FAKE_CALIBRATION_FILE = ROOT / "calibration" / "fake_arm" / "so101_follower.json"

# LeRobot's SO-101 follower IDs, base to gripper.
JOINTS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}


@dataclass
class JointCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def load(path=CALIBRATION_FILE):
    data = json.loads(Path(path).read_text())
    return {name: JointCalibration(**v) for name, v in data.items()}


def save(cal, path=CALIBRATION_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({n: asdict(c) for n, c in cal.items()}, indent=4) + "\n")


def reset(bus, ids):
    """Torque off, no offset, full range: raw encoder values come straight through."""
    for i in ids:
        bus.torque_off(i)
    for i in ids:
        bus.write(i, "Homing_Offset", 0)
        bus.write(i, "Min_Position_Limit", 0)
        bus.write(i, "Max_Position_Limit", RESOLUTION - 1)
        bus.write(i, "Operating_Mode", 0)


def center(bus, ids):
    """Make the current pose read MIDDLE on every motor. Returns {id: homing_offset}."""
    offsets = {}
    for i in ids:
        raw = bus.read(i, "Present_Position")
        offset = max(-2047, min(2047, raw - MIDDLE))  # Present = raw - offset; 11-bit magnitude
        bus.write(i, "Homing_Offset", offset)
        offsets[i] = offset
    for i in ids:
        now = bus.read(i, "Present_Position")
        if abs(now - MIDDLE) > 20:
            raise BusError(f"motor {i}: reads {now} after centring, expected about {MIDDLE}. "
                           "Was the joint moving? Run calibrate again.")
    return offsets


class RangeRecorder:
    """Tracks the extremes each joint reaches while you move it, and notices encoder wrap."""

    def __init__(self, ids):
        self.lo = {i: None for i in ids}
        self.hi = {i: None for i in ids}
        self.last = {i: None for i in ids}
        self.wrapped = set()

    def update(self, positions):
        for i, p in positions.items():
            if self.last[i] is not None and abs(p - self.last[i]) > RESOLUTION // 2:
                self.wrapped.add(i)  # jumped across 4095 -> 0: the centring pose was too far off
            self.last[i] = p
            self.lo[i] = p if self.lo[i] is None else min(self.lo[i], p)
            self.hi[i] = p if self.hi[i] is None else max(self.hi[i], p)

    def span(self, i):
        return 0 if self.lo[i] is None else self.hi[i] - self.lo[i]


def limits_from_recording(rec, margin_ticks, min_span=300):
    """{id: (min, max)} pulled in from the recorded extremes by margin_ticks on each side."""
    out, problems = {}, []
    for i in rec.lo:
        if i in rec.wrapped:
            problems.append(f"motor {i}: its reading jumped across the 4095/0 wrap. Calibrate again and put "
                            "that joint closer to the middle of its travel in the first step.")
            continue
        if rec.span(i) < min_span:
            problems.append(f"motor {i}: only moved {rec.span(i)} ticks ({rec.span(i) * 360 / RESOLUTION:.0f} deg). "
                            "Move it all the way to both ends.")
            continue
        out[i] = (rec.lo[i] + margin_ticks, rec.hi[i] - margin_ticks)
    return out, problems


def write_limits(bus, limits):
    for i, (lo, hi) in limits.items():
        bus.write(i, "Min_Position_Limit", lo)
        bus.write(i, "Max_Position_Limit", hi)
    for i, (lo, hi) in limits.items():
        got = (bus.read(i, "Min_Position_Limit"), bus.read(i, "Max_Position_Limit"))
        if got != (lo, hi):
            raise BusError(f"motor {i}: limits read back as {got}, wrote {(lo, hi)}")


def check(bus, cal):
    """Compare the servos' EEPROM with the calibration file. Returns a list of differences."""
    issues = []
    for name, c in cal.items():
        if bus.ping(c.id) is None:
            issues.append(f"{name} (ID {c.id}) does not answer")
            continue
        got = (bus.read(c.id, "Homing_Offset"), bus.read(c.id, "Min_Position_Limit"),
               bus.read(c.id, "Max_Position_Limit"))
        want = (c.homing_offset, c.range_min, c.range_max)
        if got != want:
            issues.append(f"{name} (ID {c.id}): servo has offset/min/max {got}, file has {want}")
    return issues
