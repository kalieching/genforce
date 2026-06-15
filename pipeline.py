"""RLDS-compatible TFRecord pipeline for GenForce.

Each episode is stored as a separate TFRecord shard. Each step carries
proprioceptive observation, contact force state, action, and RLDS metadata
flags (is_first / is_last / is_terminal), following the Open X-Embodiment spec.
"""
import os
import uuid
import numpy as np
import tensorflow as tf
from simulate import GenForceDataFactory

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _to_floats(arr) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=np.asarray(arr).flatten()))

def _to_ints(arr) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.asarray(arr).flatten().astype(np.int64)))

def _make_feature(step: dict) -> tf.train.Example:
    obs = step["observation"]
    return tf.train.Example(features=tf.train.Features(feature={
        "observation/qpos": _to_floats(obs["qpos"]),
        "observation/qvel": _to_floats(obs["qvel"]),
        "observation/ctrl": _to_floats(obs["ctrl"]),
        "observation/cfrc_ext": _to_floats(obs["cfrc_ext"]),
        "observation/qfrc_actuator": _to_floats(obs["qfrc_actuator"]),
        "action": _to_floats(step["action"]),
        "reward": _to_floats([step["reward"]]),
        "is_first": _to_ints([step["is_first"]]),
        "is_last": _to_ints([step["is_last"]]),
        "is_terminal": _to_ints([step["is_terminal"]]),
    }))


class EpisodeWriter:
    """Serializes a list of timestep dicts into a TFRecord shard on disk.

    Usage:
        with EpisodeWriter(out_dir) as writer:
            writer.write(episode)   # episode = list of step dicts from run_episode()
    """

    def __init__(self, out_dir: str = DATA_DIR):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self._writer = None

    def __enter__(self):
        path = os.path.join(self.out_dir, f"{uuid.uuid4()}.tfrecord")
        self._writer = tf.io.TFRecordWriter(path)
        return self

    def write(self, episode: list[dict]):
        """Serialize one episode (list of step dicts) to TFRecord."""
        for step in episode:
            self._writer.write(_make_feature(step).SerializeToString())

    def __exit__(self, *args):
        if self._writer:
            self._writer.close()


class DataPipeline:
    """Orchestrates episode collection from the simulator and writing to disk."""

    def __init__(self, robot: str = "franka_panda", out_dir: str = DATA_DIR):
        self.env = GenForceDataFactory(robot=robot)
        self.out_dir = out_dir

    def collect(self, policy, n_episodes: int = 10, max_steps: int = 500):
        """Run n_episodes under policy and write each to its own TFRecord shard."""
        for i in range(n_episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            episode = self.env.run_episode(policy, max_steps)
            with EpisodeWriter(self.out_dir) as writer:
                writer.write(episode)
            print(f"Episode {i + 1}/{n_episodes} collected ({len(episode)} steps)")

    def load(self, nq: int = None, nv: int = None, nu: int = None, nbody: int = None) -> tf.data.Dataset:
        """Load written TFRecords as a tf.data.Dataset of per-step dicts.
        Dimensions default to the current env's model if not provided.
        """
        m = self.env.model
        nq = nq or m.nq
        nv = nv or m.nv
        nu = nu or m.nu
        nbody = nbody or m.nbody
        parse_spec = {
            "observation/qpos": tf.io.FixedLenFeature([nq], tf.float32),
            "observation/qvel": tf.io.FixedLenFeature([nv], tf.float32),
            "observation/ctrl": tf.io.FixedLenFeature([nu], tf.float32),
            "observation/cfrc_ext": tf.io.FixedLenFeature([nbody * 6], tf.float32),
            "observation/qfrc_actuator": tf.io.FixedLenFeature([nv], tf.float32),
            "action": tf.io.FixedLenFeature([nu], tf.float32),
            "reward": tf.io.FixedLenFeature([1], tf.float32),
            "is_first": tf.io.FixedLenFeature([1], tf.int64),
            "is_last": tf.io.FixedLenFeature([1], tf.int64),
            "is_terminal": tf.io.FixedLenFeature([1], tf.int64),
        }
        pattern = os.path.join(self.out_dir, "*.tfrecord")
        files = tf.io.gfile.glob(pattern)
        return tf.data.TFRecordDataset(files).map(
            lambda x: tf.io.parse_single_example(x, parse_spec)
        )
