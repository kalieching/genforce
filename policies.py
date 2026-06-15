import numpy as np
import mujoco


class SinusoidalPolicy:
    """Deterministic sinusoidal excitation — each joint completes 1-3 cycles per episode."""

    def __init__(self, nu: int, max_steps: int = 500):
        self.cycles = np.linspace(1, 3, nu)
        self.max_steps = max_steps
        self._step = 0

    def __call__(self, _) -> np.ndarray:
        phase = 2 * np.pi * self._step / self.max_steps
        self._step += 1
        return 0.8 * np.sin(phase * self.cycles)

    def reset(self):
        self._step = 0


class WaypointPolicy:
    """Smoothly interpolates through a sequence of joint control waypoints.

    Each waypoint is a ctrl vector (position targets for the Panda's position actuators).
    Steps are divided equally among segments between consecutive waypoints.

    Franka Panda ctrl layout: [j1, j2, j3, j4, j5, j6, j7, gripper_tendon]
    Home keyframe ctrl:        [ 0,  0,  0, -1.571, 0, 1.571, -0.785, 255]
    """

    # Waypoints designed to reach the cube at [0.5, 0, 0.02]:
    # home → extend forward/down → lower to contact
    WAYPOINTS = np.array([
        [ 0.00,  0.00,  0.00, -1.571,  0.00,  1.571, -0.785, 255],  # home
        [ 0.00,  0.50,  0.00, -1.200,  0.00,  1.800, -0.785, 255],  # reach forward
        [ 0.00,  0.70,  0.00, -0.900,  0.00,  1.600, -0.785, 255],  # pre-grasp above cube
        [ 0.00,  0.85,  0.00, -0.700,  0.00,  1.500, -0.785,   0],  # lower to contact, close gripper
    ])

    def __init__(self, steps_per_segment: int = 100):
        self.steps_per_segment = steps_per_segment
        self._step = 0

    def __call__(self, _) -> np.ndarray:
        n_segments = len(self.WAYPOINTS) - 1
        total = n_segments * self.steps_per_segment
        t = min(self._step, total - 1)

        seg   = min(t // self.steps_per_segment, n_segments - 1)
        alpha = (t % self.steps_per_segment) / self.steps_per_segment

        # Smooth step (ease in/out)
        alpha = alpha * alpha * (3 - 2 * alpha)

        ctrl = (1 - alpha) * self.WAYPOINTS[seg] + alpha * self.WAYPOINTS[seg + 1]
        self._step += 1
        return ctrl

    def reset(self):
        self._step = 0

    @property
    def total_steps(self) -> int:
        return (len(self.WAYPOINTS) - 1) * self.steps_per_segment


class ReachAndPickPolicy:
    """Operational Space Controller: drives the Panda hand to hover above the cube,
    then lowers to contact and closes the gripper.

    Uses damped least-squares pseudoinverse of the 6-DOF Jacobian (position + orientation)
    to compute joint velocity commands, keeping the end-effector pointing straight down.

    Requires a reference to the SimEnv so it can read data.xpos / data.xmat / data.qpos
    and call mj_jacBody at each timestep.
    """

    # (target_pos, gripper_ctrl, position_tolerance)
    PHASES = [
        (np.array([0.5, 0.0, 0.18]), 255, 0.025),   # hover above cube
        (np.array([0.5, 0.0, 0.04]), 0,   0.015),   # lower to contact & close gripper
    ]

    # desired EE z-axis direction (pointing down into the table)
    _DESIRED_Z = np.array([0.0, 0.0, -1.0])

    def __init__(self, env, gain: float = 0.7, lam: float = 0.01):
        self.env   = env
        self.gain  = gain
        self.lam   = lam
        self._phase = 0

        self._hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self._hand_id < 0:
            raise ValueError("Could not find body 'hand' in model — check robot XML.")

    def __call__(self, _obs) -> np.ndarray:
        model = self.env.model
        data  = self.env.data

        target_pos, gripper_ctrl, tol = self.PHASES[self._phase]

        # Current EE position and orientation
        ee_pos = data.xpos[self._hand_id].copy()
        ee_mat = data.xmat[self._hand_id].reshape(3, 3)
        ee_z   = ee_mat[:, 2]  # EE z-axis in world frame

        # Position error (3,)
        pos_err = target_pos - ee_pos

        # Orientation error: cross(current_z, desired_z)  →  axis scaled by sin(angle)
        rot_err = np.cross(ee_z, self._DESIRED_Z)

        err = np.concatenate([pos_err, rot_err])  # (6,)

        # 6-DOF Jacobian (3+3 rows, nv columns)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, self._hand_id)

        # Keep only the 7 arm DOFs
        J = np.vstack([jacp[:, :7], jacr[:, :7]])  # (6, 7)

        # Damped least-squares: dq = Jᵀ (J Jᵀ + λI)⁻¹ err
        dq = J.T @ np.linalg.solve(J @ J.T + self.lam * np.eye(6), err)

        # Integrate: new position targets = current joint positions + gain * dq
        ctrl_arm = data.qpos[:7].copy() + self.gain * dq

        # Advance phase once close enough (and not already at last phase)
        if self._phase < len(self.PHASES) - 1 and np.linalg.norm(pos_err) < tol:
            self._phase += 1

        return np.append(ctrl_arm, gripper_ctrl)

    def reset(self):
        self._phase = 0
