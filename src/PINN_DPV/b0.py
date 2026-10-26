"""Baseline B0: direct regression of the PCE from the device parameters.

Sec. 8: a perceptron that receives only xi_hat and returns eta_PCE, trained by
squared error on dPV's PCE of the same training subset, without profiles,
residuals or sweep; evaluated on the same test devices and seeds as the PINNs
(H4: what deriving the PCE from the physics adds over interpolating it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from PINN_DPV import network

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r05.md",
    "sections": ["8"],
    "equations": [],
    "invariants": [
        "b0_sees_only_xi",
        "b0_fits_smooth_target",
    ],
}


def init(key, d_xi: int, width: int = 512, depth: int = 6):
    return network._mlp_init(key, [d_xi] + [width] * depth + [1])


def predict(params, xi_hat):
    return network._mlp_apply(params, xi_hat)[..., 0]


def fit(xi_train, pce_train, seed: int, width: int = 512, depth: int = 6, iters: int = 5000,
        lr: float = 3e-4):
    """Full-batch Adam on the mean squared PCE error; returns the parameters and the loss trace."""
    x, y = jnp.asarray(xi_train), jnp.asarray(pce_train)
    params = init(jax.random.PRNGKey(seed), x.shape[1], width, depth)
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        val, g = jax.value_and_grad(lambda q: jnp.mean((predict(q, x) - y) ** 2))(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, val

    trace = []
    for i in range(iters):
        params, state, val = step(params, state)
        if i % 100 == 0 or i == iters - 1:
            trace.append(float(val))
    return params, trace


def e_pce_pp(params, xi_test, pce_test):
    """|eta_B0 - eta_ref| per device in percentage points (Eq. 20)."""
    return 100.0 * np.abs(np.asarray(predict(params, jnp.asarray(xi_test))) - np.asarray(pce_test))
