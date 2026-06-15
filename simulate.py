import os
import numpy as np
import mujoco

MENAGERIE_DIR = os.path.join(os.path.dirname(__file__), "mujoco_menagerie")

ROBOT_REGISTRY = {
    "franka_panda": os.path.join(MENAGERIE_DIR, "franka_emika_panda", "scene.xml"),
}


def _build_scene(base_xml_path: str) -> mujoco.MjModel:
    """Load a robot scene and add a graspable cube via MjSpec."""
    spec = mujoco.MjSpec.from_file(base_xml_path)

    cube = spec.worldbody.add_body()
    cube.name = "cube"
    cube.pos = [0.5, 0.0, 0.02]
    cube.add_freejoint()

    geom = cube.add_geom()
    geom.name = "cube_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.02, 0.02, 0.02]
    geom.rgba = [0.8, 0.2, 0.2, 1.0]
    geom.mass = 0.05

    return spec.compile()


class SimEnv:
    """Base MuJoCo simulation environment. Subclass and implement get_observation and get_force_state."""

    def __init__(self, robot: str = "franka_panda", inject_noise: bool = False):
        if robot not in ROBOT_REGISTRY:
            raise ValueError(f"Unknown robot '{robot}'. Available: {list(ROBOT_REGISTRY.keys())}")

        self.model = _build_scene(ROBOT_REGISTRY[robot])
        self.data = mujoco.MjData(self.model)
        self.inject_noise = inject_noise
        self._viewer = None
        self._ctrl_range = self.model.actuator_ctrlrange.copy()

    def reset(self) -> dict:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        """Apply action, advance simulation one step, return (observation, done)."""
        self.data.ctrl[:] = np.clip(action, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        mujoco.mj_step(self.model, self.data)
        return self.get_observation(), False

    def get_observation(self) -> dict:
        """Override in subclass to return joint positions, velocities, sensor readings, etc."""
        return {}

    def get_force_state(self) -> dict:
        """Override in subclass to return contact forces and actuator torques."""
        return {}

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class GenForceDataFactory(SimEnv):
    """Runs episodes and collects force-augmented trajectories for dataset generation."""

    _CUBE_SPAWN = np.array([0.5, 0.0, 0.02])  # must match _build_scene cube.pos

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Locate cube freejoint so we can re-place the cube after keyframe resets.
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        matches = np.where(self.model.jnt_bodyid == cube_body_id)[0]
        if len(matches) == 0:
            raise RuntimeError("No joint found for body 'cube'.")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[matches[0]])

    def reset(self) -> dict:
        super().reset()
        # Re-place cube at spawn position: freejoint qpos = [x, y, z, qw, qx, qy, qz]
        adr = self._cube_qpos_adr
        self.data.qpos[adr:adr + 3] = self._CUBE_SPAWN
        self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> dict:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ctrl = self.data.ctrl.copy()
        return {"qpos": qpos, "qvel": qvel, "ctrl": ctrl}

    def get_force_state(self) -> dict:
        """Extract force and contact data for this timestep.
        cfrc_ext shape: (n_bodies, 6) — [torque_x, torque_y, torque_z, force_x, force_y, force_z] in world frame.
        qfrc_actuator shape: (n_joints,) — actuator forces at each joint.
        """
        cfrc_ext = self.data.cfrc_ext.copy()
        qfrc_actuator = self.data.qfrc_actuator.copy()
        return {"cfrc_ext": cfrc_ext, "qfrc_actuator": qfrc_actuator}

    def run_episode(self, policy, max_steps: int = 500) -> list[dict]:
        """Run a full episode under `policy`, returning a list of timestep dicts."""
        obs = self.reset()
        episode = []
        for i in range(max_steps):
            action = policy(obs)
            next_obs, done = self.step(action)
            episode.append({
                "observation": {**obs, **self.get_force_state()},
                "action": action,
                "reward": 0.0,
                "is_first": i == 0,
                "is_last": done or i == max_steps - 1,
                "is_terminal": done,
            })
            obs = next_obs
            if done:
                break
        return episode
