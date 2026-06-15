import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from simulate import GenForceDataFactory

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_episode_with_render(robot: str = "franka_panda", n_steps: int = 100, width: int = 640, height: int = 480):
    """Run a short episode, capture rendered frames and force data at each step."""
    env = GenForceDataFactory(robot=robot)
    env.reset()

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cycles = np.linspace(1, 3, env.model.nu)

    frames, force_records = [], []

    for i in range(n_steps):
        phase = 2 * np.pi * i / n_steps
        ctrl = 0.8 * np.sin(phase * cycles)
        env.step(ctrl)

        renderer.update_scene(env.data)
        frame = renderer.render().copy()
        frames.append(frame)

        force = env.get_force_state()
        force_records.append({
            "cfrc_ext":      force["cfrc_ext"].copy(),
            "qfrc_actuator": force["qfrc_actuator"].copy(),
        })

    renderer.close()
    return env, frames, force_records


def save_visualization(frames, force_records, out_path: str, robot: str = "franka_panda", task: str = "sinusoidal_excitation"):
    """Save a composite figure: rendered frame + force magnitude per body + joint torques."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pick frames evenly spaced across the episode
    n_snapshots = min(4, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_snapshots, hspace=0.4, wspace=0.3)

    # Row 0: rendered frames with force magnitude as color per body
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[idx])
        ax.set_title(f"Step {idx}", fontsize=9)
        ax.axis("off")

        # Overlay force magnitude text per body as a simple annotation
        cfrc = force_records[idx]["cfrc_ext"]
        force_mag = np.linalg.norm(cfrc[:, 3:], axis=1)  # linear force magnitude
        top_body = int(np.argmax(force_mag))
        ax.set_xlabel(f"Max force: body {top_body} ({force_mag[top_body]:.2f} N)", fontsize=7)

    # Row 1: force magnitude per body over time (heatmap)
    ax_heat = fig.add_subplot(gs[1, :])
    all_cfrc = np.array([f["cfrc_ext"] for f in force_records])
    force_mag_time = np.linalg.norm(all_cfrc[:, :, 3:], axis=2)  # (T, n_bodies)
    im = ax_heat.imshow(force_mag_time.T, aspect="auto", origin="lower", cmap="hot")
    ax_heat.set_xlabel("Timestep")
    ax_heat.set_ylabel("Body index")
    ax_heat.set_title("External contact force magnitude per body over time")
    fig.colorbar(im, ax=ax_heat, label="Force (N)")

    # Row 2: joint torques over time
    ax_torque = fig.add_subplot(gs[2, :])
    all_qfrc = np.array([f["qfrc_actuator"] for f in force_records])
    for j in range(all_qfrc.shape[1]):
        ax_torque.plot(all_qfrc[:, j], label=f"joint {j}", alpha=0.7)
    ax_torque.set_xlabel("Timestep")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_title("Actuator torques over time")
    ax_torque.legend(fontsize=7, ncol=4)

    fig.suptitle(f"{robot} — {task}", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    robot = "franka_panda"
    task = "sinusoidal_excitation"
    env, frames, force_records = run_episode_with_render(robot=robot, n_steps=100)
    out_path = os.path.join(OUTPUT_DIR, f"{robot}_{task}.png")
    save_visualization(frames, force_records, out_path, robot=robot, task=task)
