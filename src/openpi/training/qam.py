import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0


def qam_velocity(
    actor: pi0.Pi0,
    observation: _model.Observation,
    noisy_actions: _model.Actions,
    w: jax.Array | float,
) -> _model.Actions:
    """Evaluate the velocity field in QAM's time convention.

    w=0: noise; w=1: clean actions.
    observation must already be preprocessed.
    """
    batch_size = noisy_actions.shape[0]
    w = jnp.broadcast_to(
        jnp.asarray(w, dtype=jnp.float32),
        (batch_size,),
    )

    # openpi uses t=1 for noise and t=0 for clean actions
    # With t=1-w, the chain rule gives da/dw = -da/dt.
    openpi_time = 1.0 - w
    velocity = actor.predict_velocity(
        observation=observation,
        noisy_actions=noisy_actions,
        time=openpi_time,
    )

    return -velocity.astype(jnp.float32)


def libero_actions_for_critic(
    actor_actions: jax.Array,
    actor_q01: jax.Array,
    actor_q99: jax.Array,
    critic_q01: jax.Array,
    critic_q99: jax.Array,
    *,
    replan_steps: int = 5,
    critic_horizon: int = 30,
    action_dim: int = 7,
    action_clip: tuple[float, float] | None = (-1.0, 1.0),
) -> tuple[jax.Array, jax.Array]:
    """Map generated endpoints between actor and critic quantile coordinates.

    Both models must use the SAME physical action representation. AIRBOT uses
    joint deltas relative to the chunk's initial state, absolute grippers,
    action_dim=14 and action_clip=None. Defaults preserve LIBERO behavior.
    """

    # LIBERO adaptation: evaluate the executed prefix of a 10-step plan.
    # The 30-step critic array is storage capacity, not execution length.
    prefix = actor_actions[:, :replan_steps, :action_dim].astype(jnp.float32)

    actor_low = jnp.asarray(actor_q01, dtype=jnp.float32)[:action_dim]
    actor_high = jnp.asarray(actor_q99, dtype=jnp.float32)[:action_dim]
    critic_low = jnp.asarray(critic_q01, dtype=jnp.float32)[:action_dim]
    critic_high = jnp.asarray(critic_q99, dtype=jnp.float32)[:action_dim]

    # Match openpi's quantile unnormalization.
    physical_actions = (prefix + 1.0) / 2.0 * (actor_high - actor_low + 1e-6) + actor_low

    # Match the actions actually passed to env.step during collection.
    if action_clip is not None:
        physical_actions = jnp.clip(physical_actions, *action_clip)

    # Match the normalization used by the DIVL replay adapter.
    normalized_actions = (physical_actions - critic_low) / (critic_high - critic_low + 1e-6) * 2.0 - 1.0

    actions = jnp.pad(
        normalized_actions,
        (
            (0, 0),
            (0, critic_horizon - replan_steps),
            (0, 0),
        ),
    )
    mask = jnp.broadcast_to(
        jnp.arange(critic_horizon) < replan_steps,
        actions.shape[:2],
    )

    return actions, mask


def qam_terminal_adjoint(
    critic: nnx.Dict,
    z: jax.Array,
    endpoint_actions: jax.Array,
    action_q01: jax.Array,
    action_q99: jax.Array,
    critic_q01: jax.Array,
    critic_q99: jax.Array,
    *,
    temperature: float = 2.0,
    replan_steps: int = 5,
    critic_horizon: int = 30,
    action_dim: int = 7,
    action_clip: tuple[float, float] | None = (-1.0, 1.0),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the terminal adjoint in actor action coordinates.

    endpoint_actions must be path[-1], after the final reference ODE step.
    action_q01/action_q99 are the actor's normalization statistics.
    temperature is a positive static configuration value.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = jax.lax.stop_gradient(z)
    endpoint_actions = jax.lax.stop_gradient(endpoint_actions.astype(jnp.float32))

    def objective(actions):
        critic_actions, mask = libero_actions_for_critic(
            actions,
            action_q01,
            action_q99,
            critic_q01,
            critic_q99,
            replan_steps=replan_steps,
            critic_horizon=critic_horizon,
            action_dim=action_dim,
            action_clip=action_clip,
        )
        q1, q2 = critic.q_head(z, critic_actions, mask)

        # Engineering choice: use clipped double Q for the actor
        # gradient. This aggregation is not explicitly specified
        # for the actor update in LWD.
        q = jnp.minimum(q1, q2)

        # Independent samples: sum avoids scaling each sample's
        # action gradient by 1 / batch_size.
        return -jnp.sum(q) / temperature, (q1, q2)

    (_, (q1, q2)), adjoint = jax.value_and_grad(objective, has_aux=True)(endpoint_actions)

    metrics = {
        "endpoint_q_mean": jnp.mean(jnp.minimum(q1, q2)),
        "endpoint_q_gap": jnp.mean(jnp.abs(q1 - q2)),
        "terminal_adjoint_norm": jnp.mean(
            jnp.linalg.norm(
                adjoint.reshape(adjoint.shape[0], -1),
                axis=-1,
            ),
        ),
    }

    return (
        jax.lax.stop_gradient(adjoint),
        jax.tree.map(jax.lax.stop_gradient, metrics),
    )


def sample_qam_path(
    actor: pi0.Pi0,
    observation: _model.Observation,
    rng: jax.Array,
    *,
    reference: pi0.Pi0,
    num_steps: int = 10,
    epsilon: float = 0.1,
) -> tuple[jax.Array, jax.Array]:
    """Sample N-1 current-actor SDE steps and one reference ODE step.

    observation must already be preprocessed.
    actor and the independent, fixed reference must be in evaluation mode.
    num_steps and epsilon are static configuration values.

    The grid remains [epsilon, 1], with approximate Gaussian initialization
    at epsilon. Only the noiseless endpoint rule follows QAM's implementation;
    we do not adopt its shifted-time noise schedule.

    Returns:
        times: [N + 1]
        path: [N + 1, B, H, D]
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if (reference.action_horizon, reference.action_dim) != (actor.action_horizon, actor.action_dim):
        raise ValueError("actor and reference must use the same action shape")

    # Limit the explicit step relative to the singular 1/w term.
    # This is not a general stability guarantee for the neural drift.
    if num_steps > 1 and (1.0 - epsilon) / num_steps > epsilon:
        raise ValueError("Increase num_steps so that step_size <= epsilon")

    times = jnp.linspace(epsilon, 1.0, num_steps + 1, dtype=jnp.float32)
    init_rng, path_rng = jax.random.split(rng)

    shape = (
        observation.state.shape[0],
        actor.action_horizon,
        actor.action_dim,
    )

    # Boundary approximation: N(0, I) at epsilon, not exact p_epsilon.
    initial_actions = jax.random.normal(init_rng, shape, dtype=jnp.float32)
    step_keys = jax.random.split(path_rng, num_steps)

    def step(actions, inputs):
        w, dw, step_rng = inputs
        velocity = qam_velocity(actor, observation, actions, w)

        # Agreed LWD adaptation: sample from the CURRENT actor.
        # Use the paired QAM coefficients: sigma^2 divides by w.
        drift = 2.0 * velocity - actions / w
        sigma_squared = 2.0 * (1.0 - w) / w

        noise = jax.random.normal(step_rng, shape, dtype=jnp.float32)
        next_actions = actions + dw * drift + jnp.sqrt(dw * sigma_squared) * noise

        # Sampling constructs supervision; do not backpropagate
        # the later actor loss through this solver.
        next_actions = jax.lax.stop_gradient(next_actions)
        return next_actions, next_actions

    # Left-endpoint Euler-Maruyama on the first N-1 intervals.
    # Keep the original key allocation so those SDE samples are unchanged.
    penultimate_actions, following_actions = jax.lax.scan(
        step,
        initial_actions,
        (times[:-2], jnp.diff(times)[:-1], step_keys[:-1]),
    )

    # QAM / Adjoint Matching Appendix G.1: evaluate Q on a denoised
    # endpoint, using the base/reference velocity and no final noise.
    # https://arxiv.org/html/2409.08861v5#A7.SS1
    final_dw = times[-1] - times[-2]
    endpoint_actions = penultimate_actions + final_dw * qam_velocity(
        reference, observation, penultimate_actions, times[-2]
    )
    endpoint_actions = jax.lax.stop_gradient(endpoint_actions)

    path = jnp.concatenate([initial_actions[None], following_actions, endpoint_actions[None]], axis=0)
    return times, jax.lax.stop_gradient(path)


def qam_backward_adjoint(
    reference: pi0.Pi0,
    observation: _model.Observation,
    times: jax.Array,
    path: jax.Array,
    terminal_adjoint: jax.Array,
) -> jax.Array:
    """Integrate the reference lean adjoint backward on a saved path.

    times and path must come from sample_qam_path:
    times increase from epsilon to 1.
    terminal_adjoint must be evaluated at its denoised endpoint path[-1].

    As in Adjoint Matching Appendix G.1, endpoint denoising changes the
    terminal evaluation, while propagation retains the lean-adjoint ODE.
    This is a finite-step boundary approximation, not the discrete gradient
    of the hybrid sampler. In particular, do not substitute the Jacobian of
    the last deterministic map for the final lean-adjoint interval.

    observation must already be preprocessed.
    reference must be in evaluation mode with fixed parameters.

    Returns:
        adjoints: same shape and time order as path, [N+1, B, H, D].
    """
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a 1D grid with at least two points")
    if path.ndim != 4 or path.shape[0] != times.shape[0]:
        raise ValueError("path must have shape [len(times), B, H, D]")
    if terminal_adjoint.shape != path.shape[1:]:
        raise ValueError("terminal_adjoint must have shape [B, H, D]")

    times = jax.lax.stop_gradient(times.astype(jnp.float32))
    path = jax.lax.stop_gradient(path.astype(jnp.float32))
    terminal_adjoint = jax.lax.stop_gradient(terminal_adjoint.astype(jnp.float32))

    def step(adjoint_right, inputs):
        actions_right, w_right, dw = inputs

        def reference_drift(actions):
            velocity = qam_velocity(reference, observation, actions, w_right)

            # Keep differentiation with respect to actions here.
            return 2.0 * velocity - actions / w_right

        _, pullback = jax.vjp(reference_drift, actions_right)
        jacobian_transpose_g = pullback(adjoint_right)[0]

        # dg/dw = -J_b^T g; backward integration gives a plus sign.
        # Explicit reverse-time Euler uses BOTH right-end state/time.
        adjoint_left = adjoint_right + dw * jacobian_transpose_g
        adjoint_left = jax.lax.stop_gradient(adjoint_left)

        return adjoint_left, adjoint_left

    _, earlier_adjoints = jax.lax.scan(
        step,
        terminal_adjoint,
        (path[1:], times[1:], jnp.diff(times)),
        reverse=True,
    )

    # reverse=True executes backward but stacks outputs in grid order.
    adjoints = jnp.concatenate(
        [earlier_adjoints, terminal_adjoint[None]],
        axis=0,
    )

    return jax.lax.stop_gradient(adjoints)


def qam_actor_loss(
    actor: pi0.Pi0,
    reference: pi0.Pi0,
    observation: _model.Observation,
    times: jax.Array,
    path: jax.Array,
    adjoints: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Local QAM regression on a detached path and lean adjoints.

    Use times/path from sample_qam_path and matching adjoints from
    qam_backward_adjoint. observation is already preprocessed.
    Actor and reference must be distinct models for the same state.

    Reduction: sum over all H*D action coordinates, mean over batch,
    and integrate over [epsilon, 1] with left-endpoint interval weights.
    Temperature is already included in the terminal adjoint.
    """
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a 1D grid with at least two points")
    if path.ndim != 4 or path.shape[0] != times.shape[0]:
        raise ValueError("path must have shape [len(times), B, H, D]")
    if adjoints.shape != path.shape:
        raise ValueError("adjoints must have the same shape as path")
    if actor is reference:
        raise ValueError("actor and reference must be independent models")

    times = jax.lax.stop_gradient(times.astype(jnp.float32))
    path = jax.lax.stop_gradient(path.astype(jnp.float32))
    adjoints = jax.lax.stop_gradient(adjoints.astype(jnp.float32))

    def step(_, inputs):
        w, actions, adjoint = inputs
        sigma_squared = 2.0 * (1.0 - w) / w

        reference_velocity = jax.lax.stop_gradient(qam_velocity(reference, observation, actions, w))
        target = jax.lax.stop_gradient(reference_velocity - 0.5 * sigma_squared * adjoint)

        # Keep the actor's parameter gradients, including its encoder.
        velocity = qam_velocity(actor, observation, actions, w)

        squared_error = jnp.sum(jnp.square(velocity - target), axis=(-2, -1))
        loss_at_time = jnp.mean(4.0 / sigma_squared * squared_error)

        deviation_at_time = jnp.mean(
            jnp.sum(
                jnp.square(velocity - reference_velocity),
                axis=(-2, -1),
            )
        )
        return None, (loss_at_time, deviation_at_time)

    # As in QAM, regress all pre-endpoint velocities, including the one at
    # times[-2], even though the endpoint was generated by the reference ODE.
    # Exclude w=1: sigma_squared is zero at that endpoint.
    _, (losses, deviations) = jax.lax.scan(
        jax.checkpoint(step),
        None,
        (times[:-1], path[:-1], adjoints[:-1]),
    )

    interval_widths = jnp.diff(times)
    loss = jnp.sum(interval_widths * losses)

    metrics = {
        "qam_loss": loss,
        "velocity_deviation_integral": jnp.sum(interval_widths * deviations),
    }
    return loss, jax.tree.map(jax.lax.stop_gradient, metrics)
