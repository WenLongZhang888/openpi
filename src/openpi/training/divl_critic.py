import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.divl_losses import distributional_loss
from openpi.models.divl_losses import divl_q_target
from openpi.models.divl_losses import twin_q_loss


def critic_loss(
    critic: nnx.Dict,
    target_critic: nnx.Dict,
    batch: dict,
    tau_base: float,
    alpha: float,
    tau_min: float,
    tau_max: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute V and Q losses without updating parameters or EMA."""

    z = critic.encoder(**batch["obs"], train=True)
    v_logits = critic.v_head(z)
    q1, q2 = critic.q_head(z, batch["actions"], batch["action_mask"])

    target_z = target_critic.encoder(**batch["obs"], train=False)

    target_q1, target_q2 = target_critic.q_head(target_z, batch["actions"], batch["action_mask"])
    v_target = jnp.minimum(target_q1, target_q2)

    loss_v = distributional_loss(v_logits, v_target, critic.v_head.atoms)

    # 目标 V 评估窗口结束状态,构造 Q 的 TD 标签
    next_z = target_critic.encoder(**batch["next_obs"], train=False)
    next_v_logits = target_critic.v_head(next_z)
    td_target = divl_q_target(
        batch["rewards"],
        batch["discounts"],
        next_v_logits,
        target_critic.v_head.atoms,
        tau_base=tau_base,
        alpha=alpha,
        tau_min=tau_min,
        tau_max=tau_max,
    )

    loss_q = twin_q_loss(q1, q2, td_target)

    loss = loss_v + loss_q
    metrics = {
        "loss_v": loss_v,
        "loss_q": loss_q,
        "q1_mean": jnp.mean(q1),
        "q2_mean": jnp.mean(q2),
        "target_q_mean": jnp.mean(td_target),
    }
    return loss, metrics
