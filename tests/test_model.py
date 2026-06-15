import jax
import jax.numpy as jnp
import pytest
from model import ForceConditionedPolicy, flatten_obs
from train import compute_loss, train_step, create_train_state

OBS_DIM    = 132
ACTION_DIM = 8
BATCH_SIZE = 4
KEY        = jax.random.PRNGKey(0)


@pytest.fixture
def model():
    return ForceConditionedPolicy(hidden_dims=(64, 64), action_dim=ACTION_DIM)

@pytest.fixture
def params(model):
    dummy = jnp.zeros((1, OBS_DIM))
    return model.init(KEY, dummy)

@pytest.fixture
def state():
    return create_train_state(KEY, OBS_DIM, ACTION_DIM)

@pytest.fixture
def batch():
    obs     = jax.random.normal(KEY, (BATCH_SIZE, OBS_DIM))
    actions = jax.random.normal(KEY, (BATCH_SIZE, ACTION_DIM))
    return obs, actions


def test_forward_pass_shape(model, params):
    obs = jnp.zeros((BATCH_SIZE, OBS_DIM))
    out = model.apply(params, obs)
    assert out.shape == (BATCH_SIZE, ACTION_DIM)

def test_forward_pass_no_nan(model, params):
    obs = jax.random.normal(KEY, (BATCH_SIZE, OBS_DIM))
    out = model.apply(params, obs)
    assert not jnp.any(jnp.isnan(out))

def test_flatten_obs_shape():
    fake_batch = {
        "observation/qpos":          jnp.zeros((BATCH_SIZE, 16)),
        "observation/qvel":          jnp.zeros((BATCH_SIZE, 15)),
        "observation/ctrl":          jnp.zeros((BATCH_SIZE, 8)),
        "observation/cfrc_ext":      jnp.zeros((BATCH_SIZE, 78)),
        "observation/qfrc_actuator": jnp.zeros((BATCH_SIZE, 15)),
    }
    out = flatten_obs(fake_batch)
    assert out.shape == (BATCH_SIZE, OBS_DIM)

def test_compute_loss_is_scalar(state, batch):
    obs, actions = batch
    loss = compute_loss(state.params, state.apply_fn, obs, actions)
    assert loss.shape == ()

def test_compute_loss_is_nonnegative(state, batch):
    obs, actions = batch
    loss = compute_loss(state.params, state.apply_fn, obs, actions)
    assert float(loss) >= 0.0

def test_train_step_returns_new_state(state, batch):
    obs, actions = batch
    new_state, loss = train_step(state, obs, actions)
    assert new_state.step == 1

def test_train_step_loss_is_finite(state, batch):
    obs, actions = batch
    _, loss = train_step(state, obs, actions)
    assert jnp.isfinite(loss)

def test_train_step_updates_params(state, batch):
    obs, actions = batch
    new_state, _ = train_step(state, obs, actions)
    orig = jax.tree_util.tree_leaves(state.params)
    updated = jax.tree_util.tree_leaves(new_state.params)
    assert any(not jnp.allclose(a, b) for a, b in zip(orig, updated))

def test_loss_decreases_after_steps(state, batch):
    obs, actions = batch
    _, loss0 = train_step(state, obs, actions)
    s = state
    for _ in range(50):
        s, loss = train_step(s, obs, actions)
    assert float(loss) < float(loss0)
