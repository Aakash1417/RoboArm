"""Drive the physical arm from the simulation's joint angles, and read it back as sim angles.

The sim model (so101_new_calib) puts each joint's zero at the middle of its range, and calibration
centred each servo's travel around 2047, so a sim angle q maps onto a servo as

    ticks = centre + sign * (q - model_middle) * TICKS_PER_RAD,   centre = middle of the range + trim

sign and trim live in calibration/sim_map.json; 'python arm.py mirror' shows the limp arm in the
sim so both can be checked and adjusted. Goals are clamped to the calibrated limits, which the
servos also enforce themselves.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import calibration as calib
from .bus import RESOLUTION, BusError

TICKS_PER_RAD = RESOLUTION / (2 * math.pi)
SIM_MAP_FILE = calib.ROOT / "calibration" / "sim_map.json"
FAKE_SIM_MAP_FILE = calib.ROOT / "calibration" / "fake_arm" / "sim_map.json"
SNAP_TICKS = 60  # a joint resting this far outside its limit may ease back in when torque comes on


@dataclass
class JointMap:
    sign: int = 1
    trim_deg: float = 0.0


def load_map(path, names):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    return {n: JointMap(**data.get(n, {})) for n in names}


def save_map(maps, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({n: asdict(m) for n, m in maps.items()}, indent=4) + "\n")


def model_ranges(model, names):
    return {n: tuple(model.jnt_range[model.joint(n).id]) for n in names}


class Mapping:
    def __init__(self, cal, ranges, maps):
        self.cal, self.maps = cal, maps
        self.model_mid = {n: (lo + hi) / 2 for n, (lo, hi) in ranges.items()}

    def centre(self, name):
        c = self.cal[name]
        return (c.range_min + c.range_max) / 2 + self.maps[name].trim_deg * RESOLUTION / 360

    def to_ticks(self, name, q):
        c = self.cal[name]
        t = self.centre(name) + self.maps[name].sign * (q - self.model_mid[name]) * TICKS_PER_RAD
        return int(round(min(max(t, c.range_min), c.range_max)))

    def to_rad(self, name, ticks):
        return self.model_mid[name] + self.maps[name].sign * (ticks - self.centre(name)) / TICKS_PER_RAD


def load_checked(bus, path):
    """The calibration file, provided the servos still carry exactly those offsets and limits."""
    try:
        cal = calib.load(path)
    except FileNotFoundError:
        raise BusError("No calibration yet. Run 'python arm.py calibrate' first.")
    issues = calib.check(bus, cal)
    if issues:
        raise BusError("The servos don't match the calibration file, so its limits can't be trusted:\n  "
                       + "\n  ".join(issues) + "\nRun 'python arm.py calibrate' again.")
    return cal


class RealArm:
    """Follows sim joint angles. The servos' own speed and torque limits soften every move."""

    # LeRobot's SO-101 follower gains: P 16 instead of the factory 32 "to avoid shakiness".
    GAINS = {"P_Coefficient": 16, "I_Coefficient": 0, "D_Coefficient": 32}
    ACCELERATION = 100  # x100 ticks/s^2 (~880 deg/s^2): the commands are already accel-limited

    def __init__(self, bus, cal, mapping, speed_deg=90.0, torque=50.0, rate_hz=50.0, gripper_speed_deg=300.0):
        self.bus, self.cal, self.mapping = bus, cal, mapping
        self.names = list(cal)
        self.speed_deg, self.torque, self.gripper_speed_deg = speed_deg, torque, gripper_speed_deg
        self.period = 1.0 / rate_hz
        self._next = 0.0
        self.on = False
        self.notes = []

    def _tune(self, motor_id):
        """Write GAINS (EEPROM, so only when they differ). Skipped if P reads as something no STS3215
        ships with, since then this firmware's registers aren't where we expect."""
        p = self.bus.read(motor_id, "P_Coefficient")
        if not 8 <= p <= 64:
            self.notes.append(f"motor {motor_id}: P gain reads {p}, left the gains alone")
            return
        for name, v in self.GAINS.items():
            if self.bus.read(motor_id, name) != v:
                self.bus.write(motor_id, name, v)

    def start(self):
        """Torque on, holding where the arm is. Refused if a joint sits well outside its limits."""
        far = []
        for n, c in self.cal.items():
            self.bus.torque_off(c.id)
            self._tune(c.id)
            speed = self.gripper_speed_deg if n == "gripper" else self.speed_deg
            self.bus.write(c.id, "Acceleration", self.ACCELERATION)
            self.bus.write(c.id, "Goal_Speed", int(speed * RESOLUTION / 360))
            self.bus.write(c.id, "Torque_Limit", int(self.torque * 10))
            pos = self.bus.read(c.id, "Present_Position")
            if not c.range_min - SNAP_TICKS <= pos <= c.range_max + SNAP_TICKS:
                far.append(f"{c.id} at {pos} (limits {c.range_min}..{c.range_max})")
        if far:
            raise BusError("These motors are well outside their limits, so torque-on would swing them:\n  "
                           + "\n  ".join(far) + "\nMove them back inside by hand, then start again.")
        for c in self.cal.values():
            self.bus.torque(c.id, True, max_snap=SNAP_TICKS)
        self.on = True

    def send(self, q):
        """q: sim angles in calibration order (pan .. gripper). Sent at most rate_hz times a second."""
        now = time.perf_counter()
        if not self.on or now < self._next:
            return
        self._next = now + self.period
        self.bus.sync_write("Goal_Position", {self.cal[n].id: self.mapping.to_ticks(n, float(v))
                                              for n, v in zip(self.names, q)})

    def read(self):
        return [self.mapping.to_rad(n, self.bus.read(self.cal[n].id, "Present_Position")) for n in self.names]

    def relax(self):
        for c in self.cal.values():
            self.bus.torque_off(c.id)
        self.on = False
