import jax.numpy as jnp
import flax.linen as nn


class ForceConditionedPolicy(nn.Module):
    """Behavior cloning MLP conditioned on proprioception + force state.

    Input: flattened observation vector (qpos, qvel, ctrl, cfrc_ext, qfrc_actuator)
    Output: action vector (shape: action_dim)
    """
    hidden_dims: tuple
    action_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        action = nn.Dense(self.action_dim)(x)
        return action

def flatten_obs(batch: dict) -> jnp.ndarray:
    """Flatten a TFRecord batch (keys like 'observation/qpos') into a single vector."""
    fields = [
        batch["observation/qpos"],
        batch["observation/qvel"],
        batch["observation/ctrl"],
        batch["observation/cfrc_ext"],
        batch["observation/qfrc_actuator"],
    ]
    return jnp.concatenate([jnp.asarray(f) for f in fields], axis=-1)


def flatten_obs_raw(obs: dict, force: dict) -> jnp.ndarray:
    """Flatten a live simulation observation for inference.
    Takes the dicts returned by get_observation() and get_force_state().
    """
    fields = [
        obs["qpos"],
        obs["qvel"],
        obs["ctrl"],
        force["cfrc_ext"].flatten(),
        force["qfrc_actuator"],
    ]
    return jnp.concatenate([jnp.asarray(f) for f in fields], axis=-1)
