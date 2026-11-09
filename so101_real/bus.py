"""Feetech STS3215 serial bus: register table, a real bus (scservo_sdk) and a simulated one.

Addresses come from Feetech's SMS_STS headers (the STS3215 is an STS-series servo). Values on
the wire use sign-magnitude: the sign lives in one bit (sign_bit) instead of two's complement.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

STS3215_MODEL = 777
RESOLUTION = 4096          # encoder ticks per turn
MIDDLE = 2047              # calibration puts the middle of each joint's travel here
DEFAULT_BAUD = 1_000_000   # factory default, and what LeRobot uses


@dataclass(frozen=True)
class Reg:
    addr: int
    size: int
    sign_bit: int | None = None
    eeprom: bool = False   # survives power-off; needs the EEPROM unlocked and torque off to write


REGS = {
    "Model_Number": Reg(3, 2),
    "ID": Reg(5, 1, eeprom=True),
    "Baud_Rate": Reg(6, 1, eeprom=True),
    "Min_Position_Limit": Reg(9, 2, eeprom=True),
    "Max_Position_Limit": Reg(11, 2, eeprom=True),
    "Max_Torque_Limit": Reg(16, 2, eeprom=True),     # 0..1000 = 0..100 % of stall torque
    "P_Coefficient": Reg(21, 1, eeprom=True),         # position loop gains; factory 32 / 32 / 0
    "D_Coefficient": Reg(22, 1, eeprom=True),
    "I_Coefficient": Reg(23, 1, eeprom=True),
    "Homing_Offset": Reg(31, 2, sign_bit=11, eeprom=True),
    "Operating_Mode": Reg(33, 1, eeprom=True),        # 0 = position
    "Torque_Enable": Reg(40, 1),
    "Acceleration": Reg(41, 1),                       # x100 ticks/s^2, 0 = no limit
    "Goal_Position": Reg(42, 2, sign_bit=15),
    "Goal_Speed": Reg(46, 2, sign_bit=15),            # ticks/s, 0 = no limit
    "Torque_Limit": Reg(48, 2),                       # RAM copy of Max_Torque_Limit, resets on power-up
    "Lock": Reg(55, 1),                               # 0 = EEPROM writable, 1 = locked
    "Present_Position": Reg(56, 2, sign_bit=15),
    "Present_Speed": Reg(58, 2, sign_bit=15),
    "Present_Load": Reg(60, 2, sign_bit=10),          # 0..1000 = 0..100 %
    "Present_Voltage": Reg(62, 1),                    # 0.1 V
    "Present_Temperature": Reg(63, 1),                # deg C
    "Moving": Reg(66, 1),
    "Present_Current": Reg(69, 2, sign_bit=15),
}

BAUD_CODES = {1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3, 115_200: 4, 57_600: 5, 38_400: 6, 19_200: 7}


def encode(value, sign_bit):
    if sign_bit is None or value >= 0:
        return int(value)
    return int(-value) | (1 << sign_bit)


def decode(raw, sign_bit):
    if sign_bit is not None and raw & (1 << sign_bit):
        return -(raw & ((1 << sign_bit) - 1))
    return raw


class BusError(RuntimeError):
    pass


class BaseBus:
    """Register access by name, plus the EEPROM and torque-safety rules every caller needs."""

    def read(self, motor_id, name):
        reg = REGS[name]
        return decode(self._read(motor_id, reg.addr, reg.size), reg.sign_bit)

    def write(self, motor_id, name, value):
        """EEPROM registers are written limp with the lock open (see torque_off) and read back,
        because a real arm sometimes reported torque on again mid-way through a motor's writes."""
        reg = REGS[name]
        if reg.eeprom:
            for _ in range(self.eeprom_attempts):
                self.torque_off(motor_id)
                self._write(motor_id, reg.addr, reg.size, encode(value, reg.sign_bit))
                time.sleep(self.eeprom_delay)
                got = self.read(motor_id, name)
                if got == value:
                    return
            raise BusError(f"motor {motor_id}: {name} reads {got} after writing {value} "
                           f"(Torque_Enable={self.read(motor_id, 'Torque_Enable')}, "
                           f"Lock={self.read(motor_id, 'Lock')})")
        else:
            self._write(motor_id, reg.addr, reg.size, encode(value, reg.sign_bit))

    def sync_write(self, name, values):
        """{motor_id: value} for a RAM register; the real bus sends it as one packet."""
        for i, v in values.items():
            self.write(i, name, v)

    def torque_off(self, motor_id):
        """Limp, with the EEPROM lock open, as LeRobot does on Feetech servos. (Closing the lock
        goes with torque on; locking after each EEPROM write turned torque back on on a real arm.)"""
        self._write(motor_id, REGS["Torque_Enable"].addr, 1, 0)
        self._write(motor_id, REGS["Lock"].addr, 1, 0)

    def torque(self, motor_id, on, max_snap=0):
        """Turning torque on holds the current position: the goal is set to where the motor is
        first, otherwise it would jump to whatever goal was last written. Refused when the motor
        sits further than max_snap ticks outside its position limits, because the servo would then
        swing to the nearest limit; within max_snap it moves there (at the set Goal_Speed)."""
        if on:
            pos = self.read(motor_id, "Present_Position")
            lo, hi = self.read(motor_id, "Min_Position_Limit"), self.read(motor_id, "Max_Position_Limit")
            if lo > hi or not lo - max_snap <= pos <= hi + max_snap:
                raise BusError(f"motor {motor_id} is at {pos}, outside its stored limits {lo}..{hi}. Turning "
                               "torque on would swing it to a limit. Run 'python arm.py calibrate' first.")
            self.write(motor_id, "Goal_Position", min(max(pos, lo), hi))
            self.write(motor_id, "Torque_Enable", 1)
            self.write(motor_id, "Lock", 1)
        else:
            self.torque_off(motor_id)

    def change_id(self, old_id, new_id):
        """After the ID write the motor only answers to new_id, so the EEPROM is re-locked there."""
        self.torque_off(old_id)
        self._write(old_id, REGS["ID"].addr, 1, new_id)
        time.sleep(self.eeprom_delay + 0.05)
        if self.ping(new_id) is None:
            raise BusError(f"motor did not answer at its new ID {new_id}")

    def scan(self, ids=range(1, 21)):
        """{id: model_number} for every motor that answers."""
        found = {}
        for i in ids:
            model = self.ping(i)
            if model is not None:
                found[i] = model
        return found

    eeprom_delay = 0.05
    eeprom_attempts = 3


class SerialBus(BaseBus):
    """The real thing: a Feetech/Waveshare bus adapter on a COM port."""

    def __init__(self, port, baud=DEFAULT_BAUD, retries=2):
        import scservo_sdk as scs

        self._scs = scs
        self.port_name = port
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(0)  # 0 = little-endian (STS / SMS series)
        self.retries = retries
        if not self.port.openPort():
            raise BusError(f"Could not open {port}. Is another program (e.g. a terminal) using it?")
        if not self.port.setBaudRate(baud):
            self.port.closePort()
            raise BusError(f"Could not set {port} to {baud} baud")

    def ping(self, motor_id):
        for _ in range(self.retries):
            model, result, _ = self.packet.ping(self.port, motor_id)
            if result == self._scs.COMM_SUCCESS:
                return model
        return None

    def _txrx(self, fn, *args):
        """The SDK raises IndexError on a truncated reply (seen right after EEPROM writes);
        treat that like any failed transfer, and let the line settle before the next try."""
        try:
            *out, result, _ = fn(self.port, *args)
        except IndexError:
            out, result = [None], self._scs.COMM_RX_CORRUPT
        if result != self._scs.COMM_SUCCESS:
            time.sleep(0.02)
            self.port.ser.reset_input_buffer()
        return (out[0] if out else None), result

    def _read(self, motor_id, addr, size):
        fn = {1: self.packet.read1ByteTxRx, 2: self.packet.read2ByteTxRx}[size]
        for _ in range(self.retries + 3):
            value, result = self._txrx(fn, motor_id, addr)
            if result == self._scs.COMM_SUCCESS:
                return value
        raise BusError(f"motor {motor_id}: no reply reading address {addr} "
                       f"({self.packet.getTxRxResult(result)})")

    def _write(self, motor_id, addr, size, value):
        fn = {1: self.packet.write1ByteTxRx, 2: self.packet.write2ByteTxRx}[size]
        for _ in range(self.retries + 3):
            _, result = self._txrx(fn, motor_id, addr, value)
            if result == self._scs.COMM_SUCCESS:
                return
        raise BusError(f"motor {motor_id}: no reply writing address {addr} "
                       f"({self.packet.getTxRxResult(result)})")

    def sync_write(self, name, values):
        reg = REGS[name]
        if reg.eeprom:
            raise BusError(f"{name} is in EEPROM; write it per motor")
        param = []
        for i, v in values.items():
            raw = encode(v, reg.sign_bit)
            param += [i] + [(raw >> (8 * b)) & 0xFF for b in range(reg.size)]  # little-endian
        result = self.packet.syncWriteTxOnly(self.port, reg.addr, reg.size, param, len(param))
        if result != self._scs.COMM_SUCCESS:
            raise BusError(f"sync write of {name} failed ({self.packet.getTxRxResult(result)})")

    def close(self):
        self.port.closePort()


class FakeBus(BaseBus):
    """Simulated STS3215s for trying the tools (and testing them) without an arm.

    Each motor has a hidden raw encoder angle; Present_Position = raw - Homing_Offset, and with
    torque on it walks toward the goal, clipped to the position limits, like the real servo.
    Hard stops are where the fake 'arm' blocks the joint; moving by hand can't pass them.
    """

    eeprom_delay = 0.0

    def __init__(self, raw=None, stops=None):
        raw = raw or {1: 3100, 2: 900, 3: 3900, 4: 2000, 5: 150, 6: 2600}
        stops = stops or {}
        self.mem = {}
        self.raw, self.stops = {}, {}
        for i, r in raw.items():
            self.mem[i] = self.factory_memory(i)
            self.raw[i] = float(r)
            self.stops[i] = stops.get(i, (r - 1000, r + 1000))  # raw ticks, may run past 0/4095
        self._last = time.perf_counter()

    @staticmethod
    def factory_memory(motor_id):
        m = {reg.addr: 0 for reg in REGS.values()}
        for name, v in {"Model_Number": STS3215_MODEL, "ID": motor_id, "Max_Position_Limit": RESOLUTION - 1,
                        "Max_Torque_Limit": 1000, "Torque_Limit": 1000, "Lock": 1, "Present_Voltage": 120,
                        "Present_Temperature": 30, "P_Coefficient": 32, "D_Coefficient": 32}.items():
            m[REGS[name].addr] = v
        return m

    def ping(self, motor_id):
        return STS3215_MODEL if motor_id in self.mem else None

    def _offset(self, i):
        return decode(self.mem[i][REGS["Homing_Offset"].addr], 11)

    def _update(self):
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        for i, m in self.mem.items():
            if not m[REGS["Torque_Enable"].addr]:
                continue
            goal = decode(m[REGS["Goal_Position"].addr], 15)
            goal = min(max(goal, m[REGS["Min_Position_Limit"].addr]), m[REGS["Max_Position_Limit"].addr])
            target_raw = goal + self._offset(i)
            speed = m[REGS["Goal_Speed"].addr] or 3000
            step = speed * max(dt, 0.02)
            delta = min(max(target_raw - self.raw[i], -step), step)
            self.raw[i] = min(max(self.raw[i] + delta, self.stops[i][0]), self.stops[i][1])

    def move_by_hand(self, motor_id, raw):
        """Test helper: the user pushes a limp joint, up to its hard stops."""
        lo, hi = self.stops[motor_id]
        self.raw[motor_id] = float(min(max(raw, lo), hi))

    def _read(self, motor_id, addr, size):
        if motor_id not in self.mem:
            raise BusError(f"motor {motor_id}: no reply")
        self._update()
        if addr == REGS["Present_Position"].addr:
            return encode(int(round(self.raw[motor_id] - self._offset(motor_id))) % RESOLUTION, 15)
        return self.mem[motor_id][addr]

    def _write(self, motor_id, addr, size, value):
        if motor_id not in self.mem:
            raise BusError(f"motor {motor_id}: no reply")
        m = self.mem[motor_id]
        eeprom_addrs = {r.addr for r in REGS.values() if r.eeprom}
        if addr in eeprom_addrs and m[REGS["Lock"].addr]:
            return  # treated as not written, like a real servo whose EEPROM is still locked
        self._update()
        m[addr] = value
        if addr == REGS["Lock"].addr and value == 1:
            m[REGS["Torque_Enable"].addr] = 1  # what a real arm did: closing the lock re-enabled torque
        if addr == REGS["ID"].addr and value != motor_id:
            self.mem[value] = self.mem.pop(motor_id)
            self.raw[value] = self.raw.pop(motor_id)
            self.stops[value] = self.stops.pop(motor_id)

    def close(self):
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(
                {i: {"mem": self.mem[i], "raw": self.raw[i], "stops": self.stops[i]} for i in self.mem}))

    state_file = None

    @classmethod
    def persistent(cls, path):
        """A fake arm that remembers its EEPROM and pose between runs, like a real one."""
        path = Path(path)
        bus = cls()
        if path.exists():
            data = {int(i): v for i, v in json.loads(path.read_text()).items()}
            bus.mem = {i: {int(a): x for a, x in v["mem"].items()} for i, v in data.items()}
            bus.raw = {i: v["raw"] for i, v in data.items()}
            bus.stops = {i: tuple(v["stops"]) for i, v in data.items()}
            for i, m in bus.mem.items():
                for addr, v in cls.factory_memory(i).items():
                    m.setdefault(addr, v)  # registers added since the state was saved
                m[REGS["Torque_Enable"].addr] = 0  # like a power cycle
        bus.state_file = path
        return bus


FAKE_STATE_FILE = Path(__file__).resolve().parent.parent / "calibration" / "fake_arm" / "state.json"


def list_ports():
    from serial.tools import list_ports

    return sorted(list_ports.comports(), key=lambda p: p.device)


def open_bus(port=None, fake=False):
    if fake:
        return FakeBus.persistent(FAKE_STATE_FILE)
    if port is None:
        ports = list_ports()
        if len(ports) != 1:
            names = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none found"
            raise BusError(f"Pick the arm's port with --port. Serial ports: {names}")
        port = ports[0].device
    return SerialBus(port)
