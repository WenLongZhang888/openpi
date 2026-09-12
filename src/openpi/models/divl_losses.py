import jax
import jax.numpy as jnp

# TODO: 后续可考虑用 HL-Gauss (Histogram Loss with Gaussian smoothing) 替代当前的
# Categorical DRL,soft label 使训练信号更平滑,target 构造也更简洁。


def categorical_quantile(
    logits: jax.Array,
    atoms: jax.Array,
    tau: float | jax.Array,
) -> jax.Array:
    """Extract a quantile from a categorical distribution.

    Args:
        logits: Logits of shape (batch, num_atoms).
        atoms: Atom values of shape (num_atoms,).
        tau: Scalar or [B], with values in (0, 1].

    Returns:
        Quantile values of shape (batch,).

    """
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    cdf = jnp.cumsum(probabilities, axis=-1)
    cdf = cdf.at[..., -1].set(1.0)

    tau = jnp.asarray(tau, dtype=jnp.float32)
    indices = jnp.argmax(cdf >= tau[..., None], axis=-1)

    return atoms[indices]


def normalized_entropy(logits: jax.Array) -> jax.Array:
    """Compute normalized entropy from finite logits [B, K], K > 1.

    Returns:
        Entropy for each sample, with shape [B].
    """

    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    probs = jnp.exp(log_probs)

    entropy = -jnp.sum(probs * log_probs, axis=-1)
    num_atoms = logits.shape[-1]

    return entropy / jnp.log(jnp.asarray(num_atoms, dtype=jnp.float32))


def adaptive_quantile_level(
    logits: jax.Array,
    *,
    tau_base: float,
    alpha: float,
    tau_min: float,
    tau_max: float,
) -> jax.Array:
    """Compute a stopped-gradient quantile level for each sample."""

    entropy = normalized_entropy(logits)

    tau = jnp.clip(
        tau_base - alpha * entropy,
        tau_min,
        tau_max,
    )

    return jax.lax.stop_gradient(tau)


def project_scalar_to_atoms(
    values: jax.Array,
    atoms: jax.Array,
) -> jax.Array:
    """Project scalar targets onto neighboring value atoms.

    Args:
        values: Target Q values, shape [B].
        atoms: Strictly increasing support, shape [K], K > 1.

    Returns:
        Stopped-gradient target probabilities, shape [B, K].
    """
    values = jnp.clip(
        values.astype(jnp.float32),
        atoms[0],
        atoms[-1],
    )
    num_atoms = atoms.shape[0]

    upper = jnp.searchsorted(atoms, values, side="right")
    upper = jnp.clip(upper, 1, num_atoms - 1)
    lower = upper - 1

    upper_weight = (values - atoms[lower]) / (atoms[upper] - atoms[lower])

    target_probs = (1.0 - upper_weight[..., None]) * jax.nn.one_hot(lower, num_atoms, dtype=jnp.float32) + upper_weight[
        ..., None
    ] * jax.nn.one_hot(upper, num_atoms, dtype=jnp.float32)

    return jax.lax.stop_gradient(target_probs)


def distributional_loss(
    logits: jax.Array,
    target_q: jax.Array,
    atoms: jax.Array,
) -> jax.Array:
    """Compute the distributional V cross-entropy loss

    Args:
        logits: V predictions for current states, shape [B, K].
        target_q: Target critic value Q_bar(s, a), shape [B].
        atoms: Value supprot, shape [K].

    Returns:
        Scalar mean loss in float32.
    """
    target_probs = project_scalar_to_atoms(target_q, atoms)

    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    per_sample_loss = -jnp.sum(target_probs * log_probs, axis=-1)

    return jnp.mean(per_sample_loss)


def divl_q_target(
    rewards: jax.Array,
    discounts: jax.Array,
    next_v_logits: jax.Array,
    atoms: jax.Array,
    *,
    tau_base: float,
    alpha: float,
    tau_min: float,
    tau_max: float,
) -> jax.Array:
    """Build stopped-gradient DIVL TD targets.

    Args:
        rewards: Discounted reward sums over the window, [B].
        discounts: Bootstrap discounts, zero for terminal windows, [B].
        next_v_logits: V logits at the bootstrap states, [B, K].
        atoms: Value support, [K].

    Returns:
        Scalar Q targets for each sample, shape [B].
    """
    tau = adaptive_quantile_level(
        next_v_logits,
        tau_base=tau_base,
        alpha=alpha,
        tau_min=tau_min,
        tau_max=tau_max,
    )

    next_value = categorical_quantile(next_v_logits, atoms, tau)

    target = rewards.astype(jnp.float32) + discounts.astype(jnp.float32) * next_value.astype(jnp.float32)

    return jax.lax.stop_gradient(target)


def twin_q_loss(
    q1: jax.Array,
    q2: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Compute the sum of two critic MSE losses.

    Args:
        q1: First critic predictions, shape [B].
        q2: Second critic predictions, shape [B].
        target: Stopped-gradient TD targets, shape [B].

    Returns:
        Scalar mean loss in float32.
    """
    target = target.astype(jnp.float32)

    error1 = q1.astype(jnp.float32) - target
    error2 = q2.astype(jnp.float32) - target

    return jnp.mean(jnp.square(error1) + jnp.square(error2))
