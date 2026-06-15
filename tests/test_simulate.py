import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from simulate import GenForceDataFactory

@pytest.fixture(scope="module")
def env():
    return GenForceDataFactory(robot="franka_panda")

def test_model_loads(env):
    assert env.model is not None
    assert env.data is not None

def test_reset_returns_correct_keys(env):
    obs = env.reset()
    assert set(obs.keys()) == {"qpos", "qvel", "ctrl"}

def test_reset_returns_correct_shapes(env):
    obs = env.reset()
    assert obs["qpos"].shape == (env.model.nq,)
    assert obs["qvel"].shape == (env.model.nv,)
    assert obs["ctrl"].shape == (env.model.nu,)

def test_step_advances_time(env):
    env.reset()
    t0 = env.data.time
    action = np.zeros(env.model.nu)
    env.step(action)
    assert env.data.time > t0

def test_get_force_state_shapes(env):
    env.reset()
    force = env.get_force_state()
    assert force["cfrc_ext"].shape == (env.model.nbody, 6)
    assert force["qfrc_actuator"].shape == (env.model.nv,)

def test_run_episode_length(env):
    policy = lambda obs: np.zeros(env.model.nu)
    episode = env.run_episode(policy, max_steps=10)
    assert len(episode) == 10

def test_run_episode_step_keys(env):
    policy = lambda obs: np.zeros(env.model.nu)
    episode = env.run_episode(policy, max_steps=5)
    required = {"observation", "action", "reward", "is_first", "is_last", "is_terminal"}
    for step in episode:
        assert required.issubset(step.keys())

def test_run_episode_flags(env):
    policy = lambda obs: np.zeros(env.model.nu)
    episode = env.run_episode(policy, max_steps=5)
    assert episode[0]["is_first"] is True
    assert episode[-1]["is_last"] is True
