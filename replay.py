"""Replay collected TFRecord data in the MuJoCo viewer or save as MP4.

Live viewer (requires mjpython on macOS):
    mjpython replay.py

Save video (regular python, no display needed):
    python replay.py --save
"""
import os
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
import tensorflow as tf

from pipeline import DataPipeline, DATA_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_episode(data_dir: str = DATA_DIR, n_steps: int = 300):
    """Load one episode worth of qpos from the most recently written TFRecord."""
    pipeline = DataPipeline(out_dir=data_dir)

    all_files = tf.io.gfile.glob(os.path.join(data_dir, "*.tfrecord"))
    if not all_files:
        raise FileNotFoundError(f"No .tfrecord files found in {data_dir}")
    latest = max(all_files, key=os.path.getmtime)
    print(f"Replaying: {os.path.basename(latest)}")

    m = pipeline.env.model
    parse_spec = {
        "observation/qpos": tf.io.FixedLenFeature([m.nq], tf.float32),
    }
    ds = tf.data.TFRecordDataset([latest]).map(
        lambda x: tf.io.parse_single_example(x, parse_spec)
    )
    steps = [np.array(batch["observation/qpos"]) for batch in ds.take(n_steps)]
    return steps, pipeline.env.model


def replay(steps: list, model: mujoco.MjModel, dt: float = 0.01):
    """Step through saved qpos values in the passive viewer."""
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Replaying {len(steps)} steps — close the window to exit.")
        for qpos in steps:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def save_video(steps: list, model: mujoco.MjModel,
               width: int = 640, height: int = 480, fps: int = 30,
               out_path: str = None):
    """Render each qpos frame offscreen and write an MP4."""
    import imageio

    if out_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "replay.mp4")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    frames = []
    for qpos in steps:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    renderer.close()

    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Saved {len(frames)} frames → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Render to MP4 instead of opening viewer")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output path for MP4 (default: output/replay.mp4)")
    args = parser.parse_args()

    steps, model = load_episode(n_steps=args.steps)

    if args.save:
        save_video(steps, model, fps=args.fps, out_path=args.out)
    else:
        replay(steps, model)
