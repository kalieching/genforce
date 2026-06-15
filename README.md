# GenForce

A proof-of-concept robotics dataset generation system that produces **force-augmented trajectories** for training robot manipulation policies. Standard vision-language-action (VLA) datasets capture images and joint positions; GenForce additionally records contact forces and actuator torques at every timestep — providing the physical grounding needed for contact-rich manipulation.

## Architecture

```
collect.py          CLI: run a policy in simulation and write episodes to disk
    │
    ├── simulate.py     MuJoCo environment (Franka Panda + procedurally-added cube)
    ├── policies.py     Motion policies (sinusoidal, waypoint, OSC reach-and-pick)
    └── pipeline.py     RLDS/TFRecord writer and tf.data loader

train.py            JAX/Flax behavior cloning loop (reads TFRecords, trains MLP)
    └── model.py        ForceConditionedPolicy — MLP over (qpos, qvel, ctrl, cfrc_ext, qfrc_actuator)

replay.py           Replay saved episodes in the MuJoCo viewer or export to MP4
visualize.py        Offscreen render + force heatmap + torque plot → PNG
```

## Key ideas

- **Force-augmented observations**: every timestep captures `cfrc_ext` (external contact forces per body, shape `[nbody, 6]`) and `qfrc_actuator` (actuator torques, shape `[nv]`) alongside the standard proprioceptive state.
- **RLDS-compatible storage**: each episode is a TFRecord shard with `is_first / is_last / is_terminal` flags, matching the [Open X-Embodiment](https://robotics-transformer-x.github.io/) format.
- **Operational Space Controller**: the `ReachAndPickPolicy` uses a damped least-squares pseudoinverse of the 6-DOF Jacobian (position + orientation) to drive the end-effector to a target pose while keeping it perpendicular to the table.
- **Behavior cloning in JAX**: `train.py` uses `jax.value_and_grad` + Optax Adam with Orbax checkpointing. The model learns to imitate the OSC policy purely from logged (observation, action) pairs.

## Setup

```bash
# Clone with submodule (MuJoCo Menagerie robot models)
git clone --recurse-submodules <repo-url>
cd genforce

# Create conda environment
conda create -n genforce python=3.11
conda activate genforce
pip install -r requirements.txt
```

## Usage

**1. Collect training data**
```bash
python collect.py --policy reach --episodes 20 --steps 300
```
Policies: `reach` (OSC, default), `waypoint` (fixed interpolation), `sinusoidal` (joint excitation).

**2. Train the policy**
```bash
python train.py
```
Checkpoints are saved to `checkpoints/` every 10 epochs.

**3. Replay an episode**
```bash
# Live viewer (requires mjpython on macOS)
mjpython replay.py

# Export to MP4 (standard python)
python replay.py --save
```

**4. Visualize force data**
```bash
python visualize.py
# → output/franka_panda_sinusoidal_excitation.png
```

**5. Run tests**
```bash
pytest tests/
```

## Dataset schema

Each TFRecord step contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `observation/qpos` | `[nq]` | Joint positions (arm + gripper + cube freejoint) |
| `observation/qvel` | `[nv]` | Joint velocities |
| `observation/ctrl` | `[nu]` | Applied actuator controls |
| `observation/cfrc_ext` | `[nbody × 6]` | External contact forces per body |
| `observation/qfrc_actuator` | `[nv]` | Actuator-generated joint forces |
| `action` | `[nu]` | Policy action (position targets) |
| `reward` | `[1]` | Scalar reward (0 for BC) |
| `is_first / is_last / is_terminal` | `[1]` | RLDS episode boundary flags |

For the Franka Panda scene: `nq=16, nv=15, nu=8, nbody=13`.
