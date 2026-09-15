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
):
    if critic is target or value_optimizer.model is not critic or q_optimizer.model is not critic:
        raise ValueError("Expected an independent target and optimizers belonging to critic")
    critic.train()
    target.eval()
    target_z = target.encoder(**batch["obs"], train=False)
    target_q = jax.lax.stop_gradient(jnp.minimum(*target.q_head(target_z, batch["actions"], batch["action_mask"])))

    # 1. Fit V to the pre-update target Q; do not update the Q head.
    loss_v, value_grads = nnx.value_and_grad(_value_loss, argnums=nnx.DiffState(0, value_optimizer.wrt))(
        critic, batch, target_q
    )
    value_optimizer.update(value_grads)

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
    q_optimizer.update(q_grads)

    # 4. EMA after both steps. ema_rate is the NEW parameter weight, so .005
    # means .995 * old + .005 * new (rho=.995 in Algorithm 2's notation).
    nnx.update(
        target,
        jax.tree.map(
            lambda old, new: (1 - ema_rate) * old + ema_rate * new,
            nnx.state(target, nnx.Param),
            nnx.state(critic, nnx.Param),
        ),
    )
    return {
        **metrics,
        "loss": loss_v + loss_q,
        "loss_v": loss_v,
        "loss_q": loss_q,
        "target_q_mean": jnp.mean(td_target),
        "value_grad_norm": optax.global_norm(value_grads),
        "q_grad_norm": optax.global_norm(q_grads),
    }
