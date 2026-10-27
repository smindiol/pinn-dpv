"""Weighting strategies for the components of Eq. (15).

Every rule acts on the compressed components ell_i = log(1 + L_i / sigma_i^2) of
Eq. (15) (r09: "cada L_i se lee como ell_i") and returns the weights lambda_i of
the loss sum_i lambda_i ell_i; with the linear control form, ell_i = L_i / sigma_i^2.
Fixed (baseline): lambda_i = 1.
NTK, Eq. (16): K_i = || d sqrt(s_i) / d theta ||_2^2, lambda_i <- K_bar / K_i with
K_bar the mean over present components, recomputed every 10 steps.
BRDR, Eq. (17): bias-corrected running mean of s_i^2, w_hat_i = s_i / sqrt(m_hat_i + eps),
normalized by its mean, then a beta_w running mean renormalized to mean one;
beta_c = beta_w = 0.999, eps = 1e-14.
EMA, Eq. (18): m_i = beta m_i + (1 - beta) s_i, lambda_i = gamma lambda_i +
(1 - gamma) m_i / sum_l m_l; beta = gamma = 0.999.
Gummel alternation (Sec. 6, strategy 6): blocks of steps alternate between the
phi network trained on {Poisson, int_phi} and the (phi_n, phi_p) network trained
on {n, p, BCn, BCp, int_n, int_p}, each with fixed weights inside its block.
TPE (Eq. 19) chooses fixed weights and lives in `search`.

The state is a plain dict of arrays so that a checkpoint carries it whole.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["6"],
    "equations": ["16", "17", "18"],
    "invariants": [
        "ntk_equalizes_gradient_norms",
        "ntk_invariant_to_sigma",
        "brdr_ema_fixed_not_invariant_to_sigma",
        "brdr_weights_mean_one",
        "ema_weights_track_loss_shares",
        "gummel_blocks_alternate",
    ],
}

STRATEGIES = ("fixed", "ntk", "brdr", "ema", "tpe", "gummel")
NTK_EVERY = 10
BRDR_BETA_C = BRDR_BETA_W = 0.999
BRDR_EPS = 1e-14
EMA_BETA = EMA_GAMMA = 0.999
GUMMEL_PHI = ("Poisson", "int_phi")
GUMMEL_QF = ("n", "p", "BCn", "BCp", "int_n", "int_p")


def init_state(names, strategy: str, fixed=None):
    """Initial weights (all one, or the given fixed ones for TPE) and the strategy's statistics."""
    n = len(names)
    lam = jnp.ones(n) if fixed is None else jnp.asarray([fixed[k] for k in names])
    state = {"lam": lam, "t": jnp.asarray(0, jnp.int32)}
    if strategy == "brdr":
        state["m"] = jnp.zeros(n)
    if strategy == "ema":
        state["m"] = jnp.zeros(n)
    return state


def ntk_weights(scaled_fn, params):
    """Eq. (16): lambda_i = K_bar / K_i with K_i = ||grad sqrt(s_i)||^2.

    `scaled_fn(params)` returns the vector of scaled components s_i (> 0).
    """
    jac = jax.jacrev(lambda p: jnp.sqrt(scaled_fn(p)))(params)
    leaves = jax.tree_util.tree_leaves(jac)
    k = sum(jnp.sum(a.reshape(a.shape[0], -1) ** 2, axis=1) for a in leaves)
    return jnp.mean(k) / k


def brdr_update(state, s):
    """Eq. (17) on the scaled components s (vector)."""
    t = state["t"] + 1
    m = BRDR_BETA_C * state["m"] + (1 - BRDR_BETA_C) * s**2
    m_hat = m / (1 - BRDR_BETA_C ** t)
    w_hat = s / jnp.sqrt(m_hat + BRDR_EPS)
    w_tilde = w_hat / (jnp.mean(w_hat) + BRDR_EPS)
    raw = BRDR_BETA_W * state["lam"] + (1 - BRDR_BETA_W) * w_tilde
    lam = raw / (jnp.mean(raw) + BRDR_EPS)
    return dict(state, lam=lam, m=m, t=t)


def ema_update(state, s):
    """Eq. (18) on the scaled components s (vector)."""
    m = EMA_BETA * state["m"] + (1 - EMA_BETA) * s
    lam = EMA_GAMMA * state["lam"] + (1 - EMA_GAMMA) * m / jnp.sum(m)
    return dict(state, lam=lam, m=m, t=state["t"] + 1)


def gummel_phase(step: int, block: int) -> str:
    """'phi' on even blocks, 'qf' on odd blocks of `block` steps."""
    return "phi" if (step // block) % 2 == 0 else "qf"


def gummel_mask(names, phase: str):
    """1 for the components the active network trains on, 0 otherwise."""
    active = GUMMEL_PHI if phase == "phi" else GUMMEL_QF
    return jnp.asarray([1.0 if n in active else 0.0 for n in names])
