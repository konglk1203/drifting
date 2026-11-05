"""JAX semi-discrete OT (SDOT) drift loss -- a drop-in replacement for
drifting's drift_loss / W-Flow's Sinkhorn velocity, for apple-to-apple
comparison. Same signature and return as drift_loss.drift_loss:

    sdot_drift_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None,
                    weight_pos=None, weight_neg=None, ...) -> (loss[B], info)

Idea (SDOT-Drift, this repo's merge algorithm), applied per batch element in
the frozen MAE feature space, exactly where drifting/W-Flow compute their
field:
  * targets = the fixed_pos points (real-data features); optionally include
    fixed_neg. gen = current generated features (the continuous source).
  * Run a few steps of semi-discrete-OT DUAL ASCENT on dual weights g over
    the target points, treating stop_grad(gen) as source samples:
        grad_g = b - hist(argmin_i ||x-y_i||^2 - g_i)
    (b = target marginal; uniform here). This is exact SDOT, not Sinkhorn.
  * Assign each gen point to its Laguerre cell x* = y_{argmin}, and regress
    toward the small-step target goal = sg(gen) + eta*(x* - sg(gen)).
    eta=1 (AlignFlow hard coupling) -> goal = x*.
  * loss = mean ||gen - goal||^2 over points/dims, per batch element.

Everything except the field computation matches drift_loss: same coordinate
scaling by the mean pairwise distance so the regression target is on the
same scale, same stop_gradient structure, same [B] loss reduction.
"""

import jax
import jax.numpy as jnp
from functools import partial


def _sqdist(x, y, eps=1e-8):
    # x [B,N,D], y [B,M,D] -> [B,N,M]
    xy = jnp.einsum("bnd,bmd->bnm", x, y)
    xn = jnp.einsum("bnd,bnd->bn", x, x)
    yn = jnp.einsum("bmd,bmd->bm", y, y)
    return jnp.clip(xn[:, :, None] + yn[:, None, :] - 2 * xy, min=eps)


def sdot_drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    inner_steps=20,
    lr_g=0.5,
    eta=1.0,
    eps_entropic=0.0,
    R_list=None,   # ignored; accepted so train.py can forward its loss_kwargs
):
    """Thin wrapper: drops the ignored R_list (drift_loss's kernel temps,
    forwarded by train.py) and calls the jit'd SDOT core."""
    return _sdot_drift_loss_core(
        gen, fixed_pos, fixed_neg, weight_gen, weight_pos, weight_neg,
        inner_steps=int(inner_steps), lr_g=float(lr_g), eta=float(eta),
        eps_entropic=float(eps_entropic),
    )


@partial(jax.jit, static_argnames=("inner_steps", "eta", "eps_entropic"))
def _sdot_drift_loss_core(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    inner_steps=20,
    lr_g=0.5,
    eta=1.0,
    eps_entropic=0.0,
):
    B, C_g, S = gen.shape
    if fixed_neg is None:
        fixed_neg = jnp.zeros_like(gen[:, :0, :])
    gen = gen.astype(jnp.float32)
    fixed_pos = fixed_pos.astype(jnp.float32)
    fixed_neg = fixed_neg.astype(jnp.float32)

    old_gen = jax.lax.stop_gradient(gen)
    # SDOT targets: the fixed real-data points (+ optional negatives).
    targets = jnp.concatenate([fixed_neg, fixed_pos], axis=1)  # [B, M, S]
    M = targets.shape[1]

    def goal_fn(old_gen_in, targets_in):
        # ---- coordinate scaling (match drift_loss): normalize by mean dist ----
        dist = jnp.sqrt(_sqdist(old_gen_in, targets_in))
        scale = dist.mean()
        scale_inputs = jnp.clip(scale / jnp.sqrt(S), min=1e-3)
        g_scaled = old_gen_in / scale_inputs
        t_scaled = targets_in / scale_inputs

        # ---- semi-discrete OT dual ascent on g over the M target cells ----
        cost = _sqdist(g_scaled, t_scaled)          # [B, C_g, M], cached (fixed across inner)
        nu = jnp.ones((B, M), jnp.float32) / M       # uniform target marginal

        def step(dual, _):
            S_adj = cost - dual[:, None, :]          # [B, C_g, M]
            if eps_entropic > 0.0:
                assign = jax.nn.softmax(-S_adj / eps_entropic, axis=-1)  # soft hist
                hist = assign.mean(axis=1)
            else:
                idx = jnp.argmin(S_adj, axis=-1)     # [B, C_g]
                onehot = jax.nn.one_hot(idx, M)      # [B, C_g, M]
                hist = onehot.mean(axis=1)           # [B, M]
            grad = hist - nu                          # ascent direction (max dual)
            dual = dual - lr_g * grad                 # g += lr*(nu - hist)
            return dual, None

        dual0 = jnp.zeros((B, M), jnp.float32)
        dual, _ = jax.lax.scan(step, dual0, None, length=inner_steps)

        # ---- hard Laguerre assignment -> small-step goal (eta) ----
        S_adj = cost - dual[:, None, :]
        idx = jnp.argmin(S_adj, axis=-1)             # [B, C_g]
        x_star = jnp.take_along_axis(
            t_scaled, idx[:, :, None], axis=1
        )  # gather [B, C_g, S]
        goal_scaled = g_scaled + eta * (x_star - g_scaled)
        return goal_scaled, scale_inputs

    goal_scaled, scale_inputs = jax.lax.stop_gradient(goal_fn(old_gen, targets))
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))        # [B]
    info = {"scale": (scale_inputs * jnp.sqrt(S)).mean()}
    return loss, info
