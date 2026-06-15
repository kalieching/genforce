import numpy as np
import pytest
import tensorflow as tf
from simulate import GenForceDataFactory
from pipeline import EpisodeWriter, DataPipeline, _make_feature

@pytest.fixture(scope="module")
def env():
    return GenForceDataFactory(robot="franka_panda")

@pytest.fixture(scope="module")
def episode(env):
    policy = lambda _: np.zeros(env.model.nu)
    return env.run_episode(policy, max_steps=10)

def test_make_feature_keys(episode):
    example = _make_feature(episode[0])
    keys = set(example.features.feature.keys())
    expected = {
        "observation/qpos", "observation/qvel", "observation/ctrl",
        "observation/cfrc_ext", "observation/qfrc_actuator",
        "action", "reward", "is_first", "is_last", "is_terminal",
    }
    assert expected.issubset(keys)

def test_episode_writer_creates_file(tmp_path, episode):
    with EpisodeWriter(out_dir=str(tmp_path)) as writer:
        writer.write(episode)
    files = list(tmp_path.glob("*.tfrecord"))
    assert len(files) == 1

def test_round_trip(tmp_path, episode):
    with EpisodeWriter(out_dir=str(tmp_path)) as writer:
        writer.write(episode)

    pipeline = DataPipeline(out_dir=str(tmp_path))
    ds = pipeline.load()
    steps = list(ds.take(len(episode)))

    assert len(steps) == len(episode)
    np.testing.assert_allclose(
        steps[0]["observation/qpos"].numpy(),
        episode[0]["observation"]["qpos"].astype(np.float32),
        rtol=1e-5,
    )

def test_collect_creates_shards(tmp_path):
    pipeline = DataPipeline(out_dir=str(tmp_path))
    policy = lambda _: np.zeros(pipeline.env.model.nu)
    pipeline.collect(policy, n_episodes=3, max_steps=5)
    files = list(tmp_path.glob("*.tfrecord"))
    assert len(files) == 3

def test_load_returns_dataset(tmp_path, episode):
    with EpisodeWriter(out_dir=str(tmp_path)) as writer:
        writer.write(episode)
    pipeline = DataPipeline(out_dir=str(tmp_path))
    ds = pipeline.load()
    assert isinstance(ds, tf.data.Dataset)
    batch = next(iter(ds))
    assert "observation/qpos" in batch
    assert batch["observation/qpos"].shape == (pipeline.env.model.nq,)
