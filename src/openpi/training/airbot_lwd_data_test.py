"""Transition semantics and robot action-coordinate regression checks."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from openpi.training.airbot_lwd_data import JOINT_MASK
from openpi.training.airbot_lwd_data import AirbotReplay
from openpi.training.airbot_lwd_data import Episode
from openpi.training.airbot_lwd_data import episode_chunks
from openpi.training.qam import libero_actions_for_critic
from openpi.training.qam_training import QAMActionStats
from openpi.training.qam_training import QAMTrainingConfig


def make_replay(outcome):
    replay = object.__new__(AirbotReplay)
    replay.horizon, replay.gamma = 32, 0.9
    state = np.broadcast_to(np.arange(14, dtype=np.float32), (70, 14)).copy()
    actions = state + 2.0
    replay.episodes = [Episode("test", 0, Path("unused"), "test.mcap", "", outcome, state, actions, 40)]
    return replay


def test_chunks_never_cross_recording_gap():
    chunks = episode_chunks(0, 70, 32, 40)
    assert [(c.start, c.length, c.next_index, c.terminal) for c in chunks] == [
        (0, 32, 32, False),
        (32, 8, 40, False),
        (41, 29, 69, True),
    ]
    covered = [i for c in chunks for i in range(c.start, c.start + c.length)]
    assert covered == [i for i in range(70) if i != 40]
    replay = make_replay("success")
    _, _, reward, discount = replay.transition(chunks[1])
    assert reward == 0
    assert discount == pytest.approx(0.9**8)


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_terminal_rewards_and_delta_grippers(outcome):
    replay = make_replay(outcome)
    chunk = episode_chunks(0, 70, 32, 40)[-1]
    actions, mask, reward, discount = replay.transition(chunk)
    assert discount == 0
    assert reward == pytest.approx(0.9**28 if outcome == "success" else 0)
    np.testing.assert_array_equal(actions[:29, JOINT_MASK], 2.0)
    np.testing.assert_array_equal(actions[:29, ~JOINT_MASK], replay.episodes[0].actions[41:, ~JOINT_MASK])
    assert mask.sum() == 29
    assert np.all(actions[29:] == 0)


def test_last_partial_chunk_has_terminal_placeholder_only():
    chunks = episode_chunks(0, 65, 32, None)
    assert [(c.start, c.length, c.next_index) for c in chunks] == [(0, 32, 32), (32, 32, 64), (64, 1, 64)]
    assert [c.terminal for c in chunks] == [False, False, True]


def test_airbot_action_mapping_keeps_right_arm_and_does_not_clip():
    actions = jnp.full((1, 32, 32), 3.0)
    low, high = jnp.full(14, -2.0), jnp.full(14, 2.0)

    def map_actions(x):
        return libero_actions_for_critic(
            x, low, high, low, high, replan_steps=32, critic_horizon=32, action_dim=14, action_clip=None
        )[0]

    mapped = map_actions(actions)
    np.testing.assert_allclose(mapped, 3.0, atol=1e-6)
    grad = jax.grad(lambda x: jnp.sum(map_actions(x)))(actions)
    np.testing.assert_allclose(grad[..., :14], 1.0, atol=1e-6)
    np.testing.assert_array_equal(grad[..., 14:], 0)


def test_libero_default_mapping_still_clips_and_masks():
    low, high = -jnp.ones(7), jnp.ones(7)
    mapped, mask = libero_actions_for_critic(jnp.full((1, 10, 32), 3.0), low, high, low, high)
    assert mapped.shape == (1, 30, 7)
    np.testing.assert_allclose(mapped[:, :5], 1.0, atol=2e-6)
    np.testing.assert_array_equal(mapped[:, 5:], 0)
    assert mask.sum() == 5


def test_airbot_stats_require_all_14_dimensions():
    stats = {"actions": NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))}
    with pytest.raises(ValueError, match="14"):
        QAMActionStats.from_norm_stats(stats, stats, dimensions=14)
    config = QAMTrainingConfig(
        action_dim=14, replan_steps=32, critic_horizon=32, action_clip=None, sample_from_reference=True
    )
    assert config.replan_steps == 32
    with pytest.raises(ValueError, match="action_clip"):
        QAMTrainingConfig(action_clip=(1.0, -1.0))


def test_data_parallel_gradient_matches_global_batch():
    """Run with four virtual CPU devices to check global gradient reduction."""
    if len(jax.devices()) < 4:
        pytest.skip("Set XLA_FLAGS=--xla_force_host_platform_device_count=4")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4]), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    x = np.arange(32, dtype=np.float32) / 32
    y = x**2
    weight = jax.device_put(jnp.asarray(0.5), replicated)

    @jax.jit
    def update(w, inputs, targets):
        loss, grad = jax.value_and_grad(lambda v: jnp.mean((v * inputs - targets) ** 2))(w)
        return w - 0.01 * grad, loss, grad

    result, loss, gradient = update(weight, jax.device_put(x, sharded), jax.device_put(y, sharded))
    expected_gradient = np.mean(2 * (0.5 * x - y) * x)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-7)
    np.testing.assert_allclose(loss, np.mean((0.5 * x - y) ** 2), atol=1e-7)
    np.testing.assert_allclose(result, 0.5 - 0.01 * expected_gradient, atol=1e-7)
    assert result.sharding.is_fully_replicated
    assert len(result.sharding.device_set) == 4
