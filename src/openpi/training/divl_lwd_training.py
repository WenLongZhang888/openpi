"""Sequential V -> Q -> target EMA updates following LWD Algorithm 2.

The shared encoder participates in both optimizer steps, with separate Adam
moments. TD targets use the EMA target encoder and V head, as in the legacy
critic implementation. LWD Appendix B describes EMA for both target Q and V;
we retain this target choice alongside Algorithm 2's sequential updates.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models.divl_losses import distributional_loss
from openpi.models.divl_losses import divl_q_target
from openpi.models.divl_losses import twin_q_loss


def initialize_training(critic, total_steps, learning_rate=5e-4):
    """Initialize once after loading critic backbones; no critic-only warmup."""
    if total_steps < 1 or learning_rate <= 0:
        raise ValueError("total_steps and learning_rate must be positive")
    nnx.update(critic, jax.tree.map(lambda x: x.astype(jnp.float32), nnx.state(critic, nnx.Param)))
    target = nnx.clone(critic)
    schedule = optax.cosine_decay_schedule(learning_rate, total_steps)
    value_filter = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("q_head")))
    q_filter = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("v_head")))
    value_optimizer = nnx.Optimizer(critic, optax.adam(schedule), wrt=value_filter)
    q_optimizer = nnx.Optimizer(critic, optax.adam(schedule), wrt=q_filter)
    return target, value_optimizer, q_optimizer


def _value_loss(critic, batch, target_q):
    z = critic.encoder(**batch["obs"], train=True)
    return distributional_loss(critic.v_head(z), target_q, critic.v_head.atoms)


def _q_loss(critic, batch, td_target):
    z = critic.encoder(**batch["obs"], train=True)
    q1, q2 = critic.q_head(z, batch["actions"], batch["action_mask"])
    return twin_q_loss(q1, q2, td_target), {"q1_mean": jnp.mean(q1), "q2_mean": jnp.mean(q2)}


def _stable_global_norm(tree):
    """Avoid overflowing sum(g**2) when the norm itself fits in float32."""
    leaves = jax.tree.leaves(tree)
    scale = jnp.max(jnp.stack([jnp.max(jnp.abs(x.astype(jnp.float32))) for x in leaves]))
    divisor = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)
    squares = sum(jnp.sum(jnp.square(x.astype(jnp.float32) / divisor)) for x in leaves)
    return scale * jnp.sqrt(squares)


def _prepare_gradients(grads, loss, max_norm):
    norm = _stable_global_norm(grads)
    valid = jnp.isfinite(loss) & jnp.isfinite(norm) & (max_norm > 0) & jnp.isfinite(max_norm)
    # Invalid gradients never enter Adam's squared moments.
    factor = jnp.minimum(1.0, max_norm / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny))
    clipped = jax.tree.map(lambda x: jnp.where(valid, x * factor, jnp.zeros_like(x)), grads)
    return clipped, norm, valid


def _guarded_update(optimizer, grads, valid):
    nnx.cond(valid, lambda opt, g: opt.update(g), lambda opt, g: None, optimizer, grads)


@nnx.jit
def train_step(
    critic,
    target,
    value_optimizer,
    q_optimizer,
    batch,
    tau_base=0.6,
    alpha=0.3,
    tau_min=0.0,
    tau_max=1.0,
    ema_rate=0.005,
    max_grad_norm=1.0,
):
    if critic is target or value_optimizer.model is not critic or q_optimizer.model is not critic:
        raise ValueError("Expected an independent target and optimizers belonging to critic")
    critic.train()
    target.eval()
    bundle = nnx.Dict(critic=critic, value_optimizer=value_optimizer, q_optimizer=q_optimizer)
    before = nnx.state(bundle)
    target_z = target.encoder(**batch["obs"], train=False)
    target_q = jax.lax.stop_gradient(jnp.minimum(*target.q_head(target_z, batch["actions"], batch["action_mask"])))

    # 1. Fit V to the pre-update target Q; do not update the Q head.
    loss_v, value_grads = nnx.value_and_grad(_value_loss, argnums=nnx.DiffState(0, value_optimizer.wrt))(
        critic, batch, target_q
    )
    clipped_v, value_norm, value_valid = _prepare_gradients(value_grads, loss_v, max_grad_norm)
    _guarded_update(value_optimizer, clipped_v, value_valid)

    # 2. Bootstrap from the pre-EMA target encoder AND V head. The online V
    # update affects future targets through EMA; TD labels are detached.
    next_z = target.encoder(**batch["next_obs"], train=False)
    td_target = divl_q_target(
        batch["rewards"],
        batch["discounts"],
        target.v_head(next_z),
        target.v_head.atoms,
        tau_base=tau_base,
        alpha=alpha,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    # 3. Fit Q; do not update the V head or differentiate through TD targets.
    (loss_q, metrics), q_grads = nnx.value_and_grad(_q_loss, argnums=nnx.DiffState(0, q_optimizer.wrt), has_aux=True)(
        critic, batch, td_target
    )
    clipped_q, q_norm, q_valid = _prepare_gradients(q_grads, loss_q, max_grad_norm)
    _guarded_update(q_optimizer, clipped_q, value_valid & q_valid)

    # Reject the whole V/Q transaction if either loss/gradient or candidate
    # parameter/optimizer state is nonfinite. In particular, Q rejection rolls
    # back the earlier V update and its Adam counters/moments.
    candidate = nnx.state(bundle)
    state_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(candidate)]))
    applied = value_valid & q_valid & state_finite
    nnx.update(bundle, jax.lax.cond(applied, lambda: candidate, lambda: before))

    # 4. EMA after both steps. ema_rate is the NEW parameter weight, so .005
    # means .995 * old + .005 * new (rho=.995 in Algorithm 2's notation).
    nnx.update(
        target,
        jax.tree.map(
            lambda old, new: jnp.where(applied, (1 - ema_rate) * old + ema_rate * new, old),
            nnx.state(target, nnx.Param),
            nnx.state(critic, nnx.Param),
        ),
    )
    return {
        **metrics,
        "loss": jnp.where(applied, loss_v + loss_q, jnp.inf),
        "loss_v": loss_v,
        "loss_q": loss_q,
        "target_q_mean": jnp.mean(td_target),
        "value_grad_norm": value_norm,
        "q_grad_norm": q_norm,
        "value_grad_valid": value_valid,
        "q_grad_valid": q_valid,
        "update_applied": applied,
    }
