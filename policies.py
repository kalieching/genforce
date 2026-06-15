import numpy as np
import mujoco


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

    _DESIRED_Z = np.array([0.0, 0.0, -1.0])

    def __init__(self, env, gain: float = 0.7, lam: float = 0.01):
        self.env = env
        self.gain = gain
        self.lam = lam
        self._phase = 0

        self._hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self._hand_id < 0:
            raise ValueError("Could not find body 'hand' in model — check robot XML.")

    def __call__(self, _obs) -> np.ndarray:
        model = self.env.model
        data = self.env.data

        target_pos, gripper_ctrl, tol = self.PHASES[self._phase]

        ee_pos = data.xpos[self._hand_id].copy()
        ee_mat = data.xmat[self._hand_id].reshape(3, 3)
        ee_z   = ee_mat[:, 2]

        pos_err = target_pos - ee_pos
        rot_err = np.cross(ee_z, self._DESIRED_Z)
        err = np.concatenate([pos_err, rot_err])

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, self._hand_id)

        J  = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + self.lam * np.eye(6), err)

        ctrl_arm = data.qpos[:7].copy() + self.gain * dq

        if self._phase < len(self.PHASES) - 1 and np.linalg.norm(pos_err) < tol:
            self._phase += 1

        return np.append(ctrl_arm, gripper_ctrl)

    def reset(self):
        self._phase = 0
