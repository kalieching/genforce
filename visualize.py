import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from simulate import GenForceDataFactory
from policies import ReachAndPickPolicy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_episode_with_render(n_steps: int = 300, width: int = 640, height: int = 480):
    """Run a reach-and-pick episode, capturing rendered frames and force data."""
    env = GenForceDataFactory(robot="franka_panda")
    policy = ReachAndPickPolicy(env=env)
    obs = env.reset()

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames, force_records = [], []

    for _ in range(n_steps):
        action = policy(obs)
        obs, _ = env.step(action)

        renderer.update_scene(env.data)
        frames.append(renderer.render().copy())
        force = env.get_force_state()
        force_records.append({
            "cfrc_ext": force["cfrc_ext"].copy(),
            "qfrc_actuator": force["qfrc_actuator"].copy(),
        })

    renderer.close()
    return frames, force_records


def save_visualization(frames, force_records, out_path: str):
    """Save a composite figure: rendered snapshots + force heatmap + joint torques."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_snapshots = 4
    indices = np.linspace(0, len(frames) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_snapshots, hspace=0.4, wspace=0.3)

    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[idx])
        ax.set_title(f"Step {idx}", fontsize=9)
        ax.axis("off")
        cfrc = force_records[idx]["cfrc_ext"]
        force_mag = np.linalg.norm(cfrc[:, 3:], axis=1)
        top_body = int(np.argmax(force_mag))
        ax.set_xlabel(f"Max force: body {top_body} ({force_mag[top_body]:.2f} N)", fontsize=7)

    ax_heat = fig.add_subplot(gs[1, :])
    all_cfrc = np.array([f["cfrc_ext"] for f in force_records])
    force_mag_time = np.linalg.norm(all_cfrc[:, :, 3:], axis=2)
    im = ax_heat.imshow(force_mag_time.T, aspect="auto", origin="lower", cmap="hot")
    ax_heat.set_xlabel("Timestep")
    ax_heat.set_ylabel("Body index")
    ax_heat.set_title("External contact force magnitude per body over time")
    fig.colorbar(im, ax=ax_heat, label="Force (N)")

    ax_torque = fig.add_subplot(gs[2, :])
    all_qfrc = np.array([f["qfrc_actuator"] for f in force_records])
    for j in range(all_qfrc.shape[1]):
        ax_torque.plot(all_qfrc[:, j], label=f"joint {j}", alpha=0.7)
    ax_torque.set_xlabel("Timestep")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_title("Actuator torques over time")
    ax_torque.legend(fontsize=7, ncol=4)

    fig.suptitle("Franka Panda — Reach and Pick", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_path = os.path.join(OUTPUT_DIR, "franka_panda_reach_and_pick.png")
    frames, force_records = run_episode_with_render()
    save_visualization(frames, force_records, out_path)
