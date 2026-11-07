"""Physical SO-101 follower arm: find the motors, calibrate safe limits, and jog each joint.

    python arm.py ports                  # list serial ports (find the arm's COM port)
    python arm.py scan                   # which motor IDs answer, and their state
    python arm.py setup-ids              # new motors: give each one its ID (1 = base ... 6 = gripper)
    python arm.py identify               # move joints by hand and see which motor ID changes
    python arm.py identify 3 --wiggle    # after calibrating: wiggle motor 3 slightly under power
    python arm.py calibrate              # centre each joint and record its min/max (writes the servos)
    python arm.py show                   # current positions, limits, and whether they match the file
    python arm.py jog                    # move each joint by keyboard, inside the calibrated limits

Add --port COM5 if more than one serial port exists, and --fake to try any command with a
simulated arm. The simulation (main.py) never talks to the arm; only this script does.
"""

import argparse
import math
import os
import sys
import time

from so101_real import calibration as calib
from so101_real.bus import (MIDDLE, RESOLUTION, STS3215_MODEL, BusError, FakeBus, list_ports, open_bus)

NAMES = {i: n for n, i in calib.JOINTS.items()}
DEG_PER_TICK = 360 / RESOLUTION
SNAP_TICKS = 60  # a joint resting this close outside its limit (5 deg, past the 3 deg margin) may ease back in


def ticks_to_deg(t):
    return t * DEG_PER_TICK


# ---- console helpers (Windows) -----------------------------------------------------------------
os.system("")  # turns on ANSI cursor codes in the Windows console


class Screen:
    """Redraws a block of lines in place."""

    def __init__(self):
        self.n = 0

    def draw(self, lines):
        out = f"\x1b[{self.n}F" if self.n else ""
        out += "".join(f"\x1b[2K{line}\n" for line in lines)
        sys.stdout.write(out)
        sys.stdout.flush()
        self.n = len(lines)


def read_key():
    """Non-blocking key read. Arrow keys come back as 'LEFT', 'RIGHT', 'UP', 'DOWN'."""
    import msvcrt

    if not msvcrt.kbhit():
        return None
    k = msvcrt.getwch()
    if k in ("\x00", "\xe0"):
        return {"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(msvcrt.getwch())
    return k


def wait_enter(prompt):
    input(prompt + " [Enter] ")


def bar(pos, lo, hi, width=30):
    if hi <= lo:
        return " " * width
    x = int(round((pos - lo) / (hi - lo) * (width - 1)))
    cells = ["-"] * width
    cells[min(max(x, 0), width - 1)] = "#" if lo <= pos <= hi else "!"
    return "".join(cells)


# ---- commands ----------------------------------------------------------------------------------
def cmd_ports(args):
    ports = list_ports()
    if not ports:
        print("No serial ports found. Plug in the bus adapter's USB cable. If it still doesn't show up,\n"
              "install its USB driver (the Waveshare / Feetech boards use a CH340 or CH343 chip).")
    for p in ports:
        print(f"{p.device:8} {p.description}  [{p.hwid}]")


def cmd_scan(args, bus):
    print(f"Scanning IDs 1..{args.max_id} ...")
    found = bus.scan(range(1, args.max_id + 1))
    if not found:
        print("No motors answered. Check: 12 V power on (USB only powers the adapter), the 3-pin cable from\n"
              "the adapter to the first motor, and that the right port is used.")
        return
    print(f"{'ID':>3} {'joint':14} {'model':>6} {'pos':>5} {'min':>5} {'max':>5} {'offset':>6} "
          f"{'torque':>6} {'volt':>5} {'temp':>4}")
    for i, model in found.items():
        r = lambda name: bus.read(i, name)  # noqa: E731
        print(f"{i:>3} {NAMES.get(i, '?'):14} {model:>6} {r('Present_Position'):>5} {r('Min_Position_Limit'):>5} "
              f"{r('Max_Position_Limit'):>5} {r('Homing_Offset'):>6} {'on' if r('Torque_Enable') else 'off':>6} "
              f"{r('Present_Voltage') / 10:>4.1f}V {r('Present_Temperature'):>3}C")
    if any(m != STS3215_MODEL for m in found.values()):
        print(f"Note: model {STS3215_MODEL} is the STS3215. Other model numbers are not what the SO-101 uses.")
    missing = [n for n, i in calib.JOINTS.items() if i not in found]
    if missing:
        print("Missing for a complete SO-101:", ", ".join(f"{n} (ID {calib.JOINTS[n]})" for n in missing))
        if list(found) == [1]:
            print("Only ID 1 answers: new motors all ship as ID 1. Run 'python arm.py setup-ids'.")


def set_one_id(bus, new_id):
    found = bus.scan(range(0, 253))
    if len(found) != 1:
        raise BusError(f"Expected exactly one motor on the bus, found IDs {sorted(found) or 'none'}. "
                       "Connect only the motor you want to set.")
    old = next(iter(found))
    if old != new_id:
        bus.change_id(old, new_id)
    print(f"  motor ID {old} -> {new_id} OK")


def cmd_setup_ids(args, bus):
    print("Gives each motor its SO-101 ID. Do this with ONE motor connected to the adapter at a time\n"
          "(unplug the daisy-chain cable from the others). Already-set motors can be skipped with 's'.")
    for name, i in sorted(calib.JOINTS.items(), key=lambda kv: -kv[1]):
        ans = input(f"\nConnect only the {name} motor (will be ID {i}), then Enter (s = skip, q = quit): ").strip()
        if ans == "q":
            return
        if ans == "s":
            continue
        set_one_id(bus, i)
    print("\nDone. Reconnect the daisy chain and run 'python arm.py scan' to see all six.")


def cmd_identify(args, bus):
    if args.id is None or not args.wiggle:
        return identify_by_hand(bus)
    i = args.id
    if bus.ping(i) is None:
        raise BusError(f"motor {i} does not answer")
    lo, hi = bus.read(i, "Min_Position_Limit"), bus.read(i, "Max_Position_Limit")
    start = bus.read(i, "Present_Position")
    if not lo + args.ticks <= start <= hi - args.ticks:
        raise BusError(f"motor {i} is at {start}, too close to or outside its limits {lo}..{hi} to wiggle "
                       "safely. Use 'python arm.py identify' (move by hand), or calibrate first.")
    was_on = bus.read(i, "Torque_Enable")
    bus.write(i, "Torque_Limit", 300)  # gentle: 30 %
    bus.write(i, "Goal_Speed", 300)
    bus.torque(i, True)
    print(f"Wiggling motor {i} ({NAMES.get(i, '?')}) by +/-{ticks_to_deg(args.ticks):.0f} deg ...")
    for _ in range(2):
        for goal in (start + args.ticks, start - args.ticks):
            bus.write(i, "Goal_Position", goal)
            time.sleep(0.4)
    bus.write(i, "Goal_Position", start)
    time.sleep(0.4)
    if not was_on:
        bus.torque_off(i)


def identify_by_hand(bus):
    """No motor moves: torque stays off, you move a joint by hand and its row lights up."""
    ids = sorted(bus.scan(range(1, 21)))
    if not ids:
        raise BusError("no motors answer")
    for i in ids:
        bus.torque_off(i)
    print("IDENTIFY: torque is off on every motor (support the arm). Move one joint by hand and watch\n"
          "which ID's reading changes. Press Enter to stop.")
    base = {i: bus.read(i, "Present_Position") for i in ids}
    screen = Screen()
    while True:
        pos = {i: bus.read(i, "Present_Position") for i in ids}
        lines = []
        for i in ids:
            moved = min(abs(pos[i] - base[i]), RESOLUTION - abs(pos[i] - base[i]))
            lines.append(f"ID {i}  {NAMES.get(i, '?'):14} pos {pos[i]:>4}  moved {ticks_to_deg(moved):5.1f} deg"
                         + ("   <-- moving" if moved > 30 else ""))
        screen.draw(lines)
        if read_key() == "\r":
            break
        time.sleep(0.05)


def cmd_calibrate(args, bus):
    ids = list(calib.JOINTS.values())
    missing = [i for i in ids if bus.ping(i) is None]
    if missing:
        raise BusError(f"motor IDs {missing} don't answer. Run 'python arm.py scan' first.")
    margin = int(round(args.margin_deg / DEG_PER_TICK))
    print("CALIBRATION. All torque is about to turn off: hold the arm so it doesn't drop.")
    wait_enter("Ready?")
    calib.reset(bus, ids)
    print("\nStep 1 of 2: move EVERY joint to roughly the middle of its travel (halfway between its two\n"
          "end stops), including the wrist roll and a half-open gripper. A rough guess is fine; it only\n"
          "has to be within about 90 degrees of the true middle.")
    wait_enter("Holding that pose?")
    offsets = calib.center(bus, ids)
    print("  centred. Homing offsets:", {NAMES[i]: o for i, o in offsets.items()})

    print("\nStep 2 of 2: move each joint SLOWLY all the way to both of its end stops, one joint at a time.\n"
          "Stop where it meets resistance; don't force it. Watch min/max grow. Press Enter when done.")
    rec = calib.RangeRecorder(ids)
    screen = Screen()
    fake_sweep = isinstance(bus, FakeBus)
    t0 = time.perf_counter()
    while True:
        if fake_sweep:  # no hands to move a simulated arm: sweep each joint across its stops
            t = time.perf_counter() - t0
            for i in ids:
                lo, hi = bus.stops[i]
                bus.move_by_hand(i, (lo + hi) / 2 + (hi - lo) * 0.6 * math.sin(t * 2 + i))
        positions = {i: bus.read(i, "Present_Position") for i in ids}
        rec.update(positions)
        lines = [f"{'joint':14} {'pos':>5} {'min':>5} {'max':>5} {'range':>6}"]
        for i in ids:
            warn = "  WRAPPED - redo step 1" if i in rec.wrapped else ""
            lines.append(f"{NAMES[i]:14} {positions[i]:>5} {rec.lo[i]:>5} {rec.hi[i]:>5} "
                         f"{ticks_to_deg(rec.span(i)):>5.0f}d{warn}")
        screen.draw(lines)
        key = read_key()
        if key == "\r" or (fake_sweep and time.perf_counter() - t0 > 4):
            break
        time.sleep(0.03)

    limits, problems = calib.limits_from_recording(rec, margin)
    if problems:
        print("\nNot saved:\n  " + "\n  ".join(problems))
        return
    calib.write_limits(bus, limits)
    cal = {NAMES[i]: calib.JointCalibration(i, 0, offsets[i], *limits[i]) for i in ids}
    calib.save(cal, args.file)
    print(f"\nLimits written to each servo (EEPROM, {args.margin_deg:g} deg inside the stops you showed)"
          f" and saved to {args.file}")
    for n, c in cal.items():
        print(f"  {n:14} {c.range_min:>5} .. {c.range_max:<5} "
              f"({ticks_to_deg(c.range_min - MIDDLE):+.0f} .. {ticks_to_deg(c.range_max - MIDDLE):+.0f} deg from middle)")
    print("Torque is still off. Try it with 'python arm.py jog'.")


def cmd_show(args, bus):
    cmd_scan(args, bus)
    try:
        cal = calib.load(args.file)
    except FileNotFoundError:
        print(f"\nNo calibration file yet ({args.file}). Run 'python arm.py calibrate'.")
        return
    issues = calib.check(bus, cal)
    print("\nCalibration file matches the servos." if not issues else
          "\nCalibration file and servos differ:\n  " + "\n  ".join(issues))


def cmd_jog(args, bus):
    try:
        cal = calib.load(args.file)
    except FileNotFoundError:
        raise BusError("No calibration yet. Run 'python arm.py calibrate' first, so jogging has limits.")
    issues = calib.check(bus, cal)
    if issues:
        raise BusError("The servos don't match the calibration file, so its limits can't be trusted:\n  "
                       + "\n  ".join(issues) + "\nRun 'python arm.py calibrate' again.")
    joints = list(cal.values())
    speed = int(args.speed_deg / DEG_PER_TICK)
    for j in joints:
        bus.torque_off(j.id)
        bus.write(j.id, "Acceleration", 20)
        bus.write(j.id, "Goal_Speed", speed)
        bus.write(j.id, "Torque_Limit", int(args.torque * 10))
    goal = {j.id: bus.read(j.id, "Present_Position") for j in joints}
    far = [f"{NAMES[j.id]} at {goal[j.id]} (limits {j.range_min}..{j.range_max})" for j in joints
           if not j.range_min - SNAP_TICKS <= goal[j.id] <= j.range_max + SNAP_TICKS]
    if far:
        raise BusError("These joints are well outside their limits, so torque-on would swing them:\n  "
                       + "\n  ".join(far) + "\nMove them back inside by hand (torque is off), then run jog again.")
    for j in joints:
        goal[j.id] = min(max(goal[j.id], j.range_min), j.range_max)
        bus.torque(j.id, True, max_snap=SNAP_TICKS)  # a joint resting on its stop eases in by the margin
    torque_on = True
    sent = dict(goal)

    steps = [5, 10, 25, 50, 100]
    step_i, sel = 2, 0
    stalled_since = {}
    message = "Holding the current pose."
    screen = Screen()
    print(f"JOG  (max {args.speed_deg:g} deg/s, torque limit {args.torque:g} %)\n"
          "1-6 pick joint   LEFT/RIGHT or A/D move   UP/DOWN step size   M middle of range\n"
          "SPACE freeze all   T torque on/off (off = limp, hold the arm!)   Q quit")
    try:
        while True:
            key = read_key()
            j = joints[sel]
            if key in tuple("123456"):
                sel = min(int(key) - 1, len(joints) - 1)
            elif key in ("LEFT", "a", "RIGHT", "d") and torque_on:
                d = steps[step_i] * (1 if key in ("RIGHT", "d") else -1)
                goal[j.id] = min(max(goal[j.id] + d, j.range_min), j.range_max)
                if goal[j.id] in (j.range_min, j.range_max):
                    message = f"{NAMES[j.id]} at its {'max' if d > 0 else 'min'} limit."
            elif key == "UP":
                step_i = min(step_i + 1, len(steps) - 1)
            elif key == "DOWN":
                step_i = max(step_i - 1, 0)
            elif key in ("m", "M") and torque_on:
                goal[j.id] = (j.range_min + j.range_max) // 2
            elif key == " ":
                goal = {jj.id: bus.read(jj.id, "Present_Position") for jj in joints}
                message = "Frozen where it is."
            elif key in ("t", "T"):
                torque_on = not torque_on
                for jj in joints:
                    bus.torque(jj.id, torque_on, max_snap=SNAP_TICKS)
                    goal[jj.id] = sent[jj.id] = bus.read(jj.id, "Present_Position")
                message = "Torque ON, holding." if torque_on else "Torque OFF: the arm is limp."
            elif key in ("q", "Q"):
                break

            pos, load, lines = {}, {}, []
            for jj in joints:
                pos[jj.id] = bus.read(jj.id, "Present_Position")
                load[jj.id] = bus.read(jj.id, "Present_Load") / 10
                if torque_on and goal[jj.id] != sent.get(jj.id):
                    bus.write(jj.id, "Goal_Position", goal[jj.id])
                    sent[jj.id] = goal[jj.id]
                # Stall guard: pushing hard but not getting there means something is in the way.
                if torque_on and abs(load[jj.id]) > args.stall_load and abs(goal[jj.id] - pos[jj.id]) > 20:
                    stalled_since.setdefault(jj.id, time.perf_counter())
                    if time.perf_counter() - stalled_since[jj.id] > 0.3:
                        goal[jj.id] = pos[jj.id]
                        message = f"{NAMES[jj.id]} was blocked (load {load[jj.id]:.0f} %): stopped it there."
                else:
                    stalled_since.pop(jj.id, None)
            temps = {jj.id: bus.read(jj.id, "Present_Temperature") for jj in (joints[sel],)}
            lines.append(f"step {steps[step_i]} ticks ({ticks_to_deg(steps[step_i]):.1f} deg)   "
                         f"torque {'ON' if torque_on else 'OFF (limp)'}   {message}")
            for k, jj in enumerate(joints):
                p = pos[jj.id]
                lines.append(f"{'>' if k == sel else ' '}{k + 1} {NAMES[jj.id]:13} pos {p:>4} goal {goal[jj.id]:>4} "
                             f"{ticks_to_deg(p - MIDDLE):+5.0f}d  {jj.range_min:>4} [{bar(p, jj.range_min, jj.range_max)}] "
                             f"{jj.range_max:<4} load {load[jj.id]:+4.0f}%"
                             + (f"  {temps[jj.id]}C" if jj.id in temps else ""))
            screen.draw(lines)
            time.sleep(0.03)
    finally:
        if torque_on:
            ans = input("\nRelease torque? The arm goes limp, so support it first. [y = release / Enter = keep holding] ")
            if ans.strip().lower() == "y":
                for jj in joints:
                    bus.torque_off(jj.id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="serial port, e.g. COM5 (default: the only one found)")
    p.add_argument("--fake", action="store_true", help="use a simulated arm instead of the real one")
    p.add_argument("--file", default=str(calib.CALIBRATION_FILE), help="calibration file")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports", help="list serial ports")
    s = sub.add_parser("scan", help="find motors and show their state")
    s.add_argument("--max-id", type=int, default=20)
    sub.add_parser("setup-ids", help="give new motors their IDs, one motor connected at a time")
    s = sub.add_parser("identify", help="move joints by hand and see which motor ID changes (nothing moves)")
    s.add_argument("id", type=int, nargs="?", help="with --wiggle: the motor to wiggle")
    s.add_argument("--wiggle", action="store_true",
                   help="after calibrating: wiggle motor ID slightly under power instead")
    s.add_argument("--ticks", type=int, default=40, help="wiggle size in encoder ticks (40 = 3.5 deg)")
    s = sub.add_parser("calibrate", help="centre each joint and record its safe min/max")
    s.add_argument("--margin-deg", type=float, default=3.0, help="stay this far inside each end stop")
    s = sub.add_parser("show", help="positions, limits, and whether they match the calibration file")
    s.add_argument("--max-id", type=int, default=20)
    s = sub.add_parser("jog", help="move joints by keyboard inside the calibrated limits")
    s.add_argument("--speed-deg", type=float, default=40.0, help="max joint speed in deg/s")
    s.add_argument("--torque", type=float, default=50.0, help="torque limit in %% of the motor's maximum")
    s.add_argument("--stall-load", type=float, default=60.0, help="stop a joint pushing harder than this %%")
    args = p.parse_args()
    if args.fake and args.file == str(calib.CALIBRATION_FILE):
        args.file = str(calib.FAKE_CALIBRATION_FILE)  # never overwrite the real arm's calibration

    if args.cmd == "ports":
        return cmd_ports(args)
    try:
        bus = open_bus(args.port, args.fake)
    except BusError as e:
        sys.exit(str(e))
    try:
        {"scan": cmd_scan, "setup-ids": cmd_setup_ids, "identify": cmd_identify, "calibrate": cmd_calibrate,
         "show": cmd_show, "jog": cmd_jog}[args.cmd](args, bus)
    except BusError as e:
        sys.exit(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
