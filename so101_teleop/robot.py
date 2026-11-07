"""MuJoCo simulation of the SO-101 plus Cartesian IK using Mink."""

import mink
import mujoco
import numpy as np

from .config import ARM_JOINTS, EE_SITE, JOINT_NAMES, ROBOT_XML, SCENE_XML, RobotConfig


class SO101Sim:
    """Tabletop scene with the official SO-101 model. Only simulation, never real hardware."""

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.act_ids = np.array([self.model.actuator(n).id for n in JOINT_NAMES])
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_ids, 1].copy()
        self.site_id = self.model.site(EE_SITE).id
        self.target_mocap = self.model.body("target").mocapid[0]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(self.cfg.home_q, self.cfg.gripper_open)
        self.data.qpos[self.qpos_adr] = q
        self.set_command(q[:5], q[5])
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, arm_q, gripper_q):
        cmd = np.clip(np.append(arm_q, gripper_q), self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[self.act_ids] = cmd

    def set_target_marker(self, pos):
        self.data.mocap_pos[self.target_mocap] = pos

    def step(self, dt):
        n = max(1, int(round(dt / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=min(n, 100))

    @property
    def joint_q(self):
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def ee_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    @property
    def wrist_pos(self):
        return self.data.xpos[self.model.body("wrist").id].copy()

    def control_pos(self, ik_mode):
        """The point the IK steers: the gripper tip ("tip") or the wrist joint ("wrist")."""
        return self.wrist_pos if ik_mode == "wrist" else self.ee_pos


class CartesianIK:
    """Moves the gripper site to a 3D point using Mink's QP-based differential IK.

    Only the position of the gripper is controlled (the SO-101 has 5 arm joints, so full 6D
    pose control is not always possible). A light posture task keeps the wrist close to the
    home orientation and resolves the extra freedom. Joint limits and velocity limits from the
    model are enforced by Mink.
    """

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.configuration = mink.Configuration(self.model)
        self.arm_adr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])

        self.ee_task = mink.FrameTask(
            frame_name=EE_SITE, frame_type="site", position_cost=1.0, orientation_cost=0.0, lm_damping=1e-2
        )
        # Per-DoF posture weights: pan, lift, elbow, wrist_flex, wrist_roll, gripper. wrist_roll is
        # weighted higher because its posture target is the hand-twist command (set_wrist_roll).
        self.posture_task = mink.PostureTask(self.model, cost=np.array([1e-3, 1e-3, 1e-3, 2e-2, 1.0, 1.0]))
        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(self.model, min_distance_from_limits=cfg.joint_limit_margin),
            mink.VelocityLimit(self.model, {n: cfg.max_joint_speed for n in ARM_JOINTS}),
        ]
        self.home_target = None
        self.reset()

    def reset(self):
        q = np.zeros(self.model.nq)
        q[self.arm_adr] = self.cfg.home_q
        self.configuration.update(q)
        self.posture_q = q.copy()
        self.posture_task.set_target(self.posture_q)
        self.home_target = self.ee_position()

    def set_wrist_roll(self, roll):
        self.posture_q[self.arm_adr[4]] = roll
        self.posture_task.set_target(self.posture_q)

    def ee_position(self):
        return self.configuration.get_transform_frame_to_world(EE_SITE, "site").translation()

    position = ee_position

    def solve(self, target_pos, dt, iterations=2):
        """Advance the IK solution toward target_pos. Returns the 5 arm joint angles."""
        self.ee_task.set_target(mink.SE3.from_translation(np.asarray(target_pos, dtype=float)))
        sub_dt = dt / iterations
        for _ in range(iterations):
            try:
                vel = mink.solve_ik(self.configuration, self.tasks, sub_dt, solver="daqp",
                                    damping=1e-3, limits=self.limits)
            except mink.NoSolutionFound:
                break  # hold the last valid joint solution
            self.configuration.integrate_inplace(vel, sub_dt)
        return self.configuration.q[self.arm_adr].copy()

    @property
    def joint_limits(self):
        lo = self.model.jnt_range[[self.model.joint(n).id for n in ARM_JOINTS], 0]
        hi = self.model.jnt_range[[self.model.joint(n).id for n in ARM_JOINTS], 1]
        return lo, hi


class SplitIK:
    """Arm/wrist split, after github.com/Jordaine777/so101-leap-teleop, with the gripper tilt held.

    shoulder_lift, elbow_flex and wrist_flex all turn about the same axis, so their sum is the
    gripper's tilt in the world. wrist_flex is always tilt - lift - elbow, which keeps the home tilt,
    and wrist_roll follows the hand twist. Damped least squares on base, shoulder and elbow then
    places the target point:
      "jaws":  the gripperframe site between the jaws. wrist_flex moves with the other two joints
               (its coupling is in the Jacobian), so the jaws go exactly where your hand says.
      "wrist": the wrist joint, as in that repo. The jaws hang off it, so near the base they swing
               further than your hand moves. Best if wrist pitch is ever driven by the hand.
    """

    ITERATIONS = 10
    DAMPING = 1e-4  # lambda^2 in J^T (J J^T + lambda^2 I)^-1; keeps steps sane near full stretch
    MAX_STEP = 0.1  # rad per iteration, so a far target can't throw the arm into a folded pose

    def __init__(self, cfg: RobotConfig, target="jaws"):
        self.cfg = cfg
        self.target = target
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(n) for n in ARM_JOINTS]
        self.arm_adr = np.array([j.qposadr[0] for j in joints])
        self.dof_adr = np.array([j.dofadr[0] for j in joints[:4]])
        self.wrist_body = self.model.body("wrist").id
        self.site_id = self.model.site(EE_SITE).id
        rng = self.model.jnt_range[[j.id for j in joints]]
        self.lo = rng[:, 0] + cfg.joint_limit_margin
        self.hi = rng[:, 1] - cfg.joint_limit_margin
        self._jac = np.zeros((3, self.model.nv))
        self.tilt = float(cfg.home_q[1] + cfg.home_q[2] + cfg.home_q[3])

        # Reach geometry, measured on the model in the zero pose.
        self._fk(np.zeros(5))
        lift = self.data.xanchor[self.model.joint("shoulder_lift").id].copy()
        pan = self.data.xanchor[self.model.joint("shoulder_pan").id].copy()
        elbow = self.data.xanchor[self.model.joint("elbow_flex").id].copy()
        wrist = self.data.xpos[self.wrist_body].copy()
        self.pan_xy = pan[:2]
        self.lift_forward = float(lift[0] - pan[0])  # shoulder sits a bit in front of the base axis
        self.lift_z = float(lift[2])
        arm_len = np.linalg.norm(elbow - lift) + np.linalg.norm(wrist - elbow)
        self.r_max, self.r_min = cfg.reach_max * arm_len, cfg.reach_min * arm_len
        self.reset()
        # Wrist-to-jaws offset in the home pose, in the arm's frame (base turned to 0): about 10 cm
        # forward and 12.5 cm down. It stays the same as the arm moves because the tilt is held.
        self._fk(self.q)
        self.jaw_offset = self.data.site_xpos[self.site_id] - self.data.xpos[self.wrist_body]
        self.tip_drop = float(-self.jaw_offset[2])

    def reset(self):
        self.q = np.array(self.cfg.home_q, dtype=float)
        self.cmd = self.q.copy()
        self.roll = float(self.q[4])
        self.home_target = self.position()

    def _point(self):
        return self.data.site_xpos[self.site_id] if self.target == "jaws" else self.data.xpos[self.wrist_body]

    def set_wrist_roll(self, roll):
        self.roll = float(roll)

    def _fk(self, q):
        self.data.qpos[self.arm_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def solve(self, target_pos, dt, iterations=None):
        """Returns the 5 arm joint angles, each moving no faster than max_joint_speed."""
        q = self.q
        target = np.asarray(target_pos, dtype=float)
        # Point the base at the target first (positive shoulder_pan swings toward -Y); the small
        # sideways offset of the arm is then fixed by the iterations below.
        rel = target[:2] - self.pan_xy
        if np.linalg.norm(rel) > 0.03:
            q[0] = np.clip(-np.arctan2(rel[1], rel[0]), self.lo[0], self.hi[0])
        q[4] = np.clip(self.roll, self.lo[4], self.hi[4])
        for _ in range(iterations or self.ITERATIONS):
            flex = self.tilt - q[1] - q[2]
            q[3] = np.clip(flex, self.lo[3], self.hi[3])
            self._fk(q)
            err = target - self._point()
            if np.linalg.norm(err) < 1e-4:
                break
            if self.target == "jaws":
                mujoco.mj_jacSite(self.model, self.data, self._jac, None, self.site_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, self._jac, None, self.wrist_body)
            J = self._jac[:, self.dof_adr]
            # Lifting or bending by d turns wrist_flex by -d (unless it is pinned at a limit).
            coupled = J[:, :3] - (q[3] == flex) * np.outer(J[:, 3], [0.0, 1.0, 1.0])
            dq = coupled.T @ np.linalg.solve(coupled @ coupled.T + self.DAMPING * np.eye(3), err)
            dq *= min(1.0, self.MAX_STEP / max(np.abs(dq).max(), 1e-12))
            q[:3] = np.clip(q[:3] + dq, self.lo[:3], self.hi[:3])
        q[3] = np.clip(self.tilt - q[1] - q[2], self.lo[3], self.hi[3])
        step = self.cfg.max_joint_speed * dt
        self.cmd += np.clip(q - self.cmd, -step, step)
        return self.cmd.copy()

    def position(self):
        self._fk(self.q)
        return self._point().copy()

    def _jaw_offset(self, heading):
        c, s = np.cos(heading), np.sin(heading)
        o = self.jaw_offset
        return np.array([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]])

    def _heading(self, p):
        return np.arctan2(p[1] - self.pan_xy[1], p[0] - self.pan_xy[0])

    def clamp(self, p):
        """Keep a target reachable. Limits are applied to the wrist point (for a jaws target, the
        wrist it implies), then mapped back. Unchanged when the target is already reachable."""
        p = np.asarray(p, dtype=float)
        if self.target != "jaws":
            return self._clamp_wrist(p)
        heading = self._heading(p)
        wrist = p - self._jaw_offset(heading)
        clamped = self._clamp_wrist(wrist)
        if np.allclose(clamped, wrist, atol=1e-12):
            return p
        out = clamped + self._jaw_offset(heading)
        for _ in range(3):  # the base heading of the new point differs slightly; settle it
            out = clamped + self._jaw_offset(self._heading(out))
        return out

    def _clamp_wrist(self, p):
        """Box, floor (for the gripper tip), and a reach shell around the shoulder, which turns
        with the base toward the target."""
        c = self.cfg
        lo = c.wrist_workspace_min.copy()
        lo[2] = max(lo[2], c.wrist_floor + self.tip_drop)
        p = np.clip(np.asarray(p, dtype=float), lo, c.wrist_workspace_max)
        for _ in range(2):
            heading = np.arctan2(p[1] - self.pan_xy[1], p[0] - self.pan_xy[0])
            shoulder = np.array([*(self.pan_xy + self.lift_forward * np.array([np.cos(heading), np.sin(heading)])),
                                 self.lift_z])
            v = p - shoulder
            n = np.linalg.norm(v)
            if n > 1e-9:
                p = shoulder + v * (np.clip(n, self.r_min, self.r_max) / n)
            p = np.clip(p, lo, c.wrist_workspace_max)
        return p

    @property
    def joint_limits(self):
        rng = self.model.jnt_range[[self.model.joint(n).id for n in ARM_JOINTS]]
        return rng[:, 0].copy(), rng[:, 1].copy()


def make_ik(cfg: RobotConfig):
    return CartesianIK(cfg) if cfg.ik_mode == "tip" else SplitIK(cfg, target=cfg.ik_mode)
