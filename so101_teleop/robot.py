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
