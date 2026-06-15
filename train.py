import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp

from pipeline import DataPipeline, DATA_DIR
from model import ForceConditionedPolicy, flatten_obs

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

HIDDEN_DIMS  = (256, 256, 128)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
N_EPOCHS      = 50


def make_dataset(data_dir: str = DATA_DIR, batch_size: int = BATCH_SIZE):
    """Load TFRecords and return a batched, shuffled tf.data pipeline."""
    pipeline = DataPipeline(out_dir=data_dir)
    ds = pipeline.load()
    ds = ds.shuffle(buffer_size=10_000).batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(4)
    return ds


def create_train_state(key, obs_dim: int, action_dim: int) -> train_state.TrainState:
    """Initialize model parameters and optimizer."""
    model = ForceConditionedPolicy(hidden_dims=HIDDEN_DIMS, action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(key, dummy_obs)
    tx = optax.adam(LEARNING_RATE)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss(params, apply_fn, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Behavior cloning loss: MSE between predicted and demonstrated actions."""
    predictions = apply_fn(params, obs)
    mse_loss = jnp.mean((predictions - actions) ** 2)
    return mse_loss


@jax.jit
def train_step(state: train_state.TrainState, obs: jnp.ndarray, actions: jnp.ndarray):
    """Single gradient update step. Returns (new_state, loss)."""
    loss_fn = lambda params: compute_loss(params, state.apply_fn, obs, actions)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def train(data_dir: str = DATA_DIR, ckpt_dir: str = CKPT_DIR):
    """Full training loop."""
    ds = make_dataset(data_dir)

    # Infer dims from first batch
    sample = next(iter(ds))
    obs_sample = flatten_obs(sample)
    obs_dim    = obs_sample.shape[-1]
    action_dim = sample["action"].shape[-1]

    key = jax.random.PRNGKey(0)
    state = create_train_state(key, obs_dim, action_dim)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in ds:
            obs     = flatten_obs(batch)
            actions = jnp.asarray(batch["action"])
            state, loss = train_step(state, obs, actions)
            epoch_loss += float(loss)
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}  loss={avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}")
            checkpointer.save(path, state, force=True)
            checkpointer.wait_until_finished()
            print(f"  Checkpoint saved: {path}")


if __name__ == "__main__":
    train()
