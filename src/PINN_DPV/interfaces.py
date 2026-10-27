"""Heterointerface conditions: conservation of displacement and of both currents.

Eq. (22): at every heterointerface x_hat_m the jumps [F_k] = F_k(x_m^+) - F_k(x_m^-)
vanish for F_phi = eps dphi/dx_hat, F_n = J_n and F_p = J_p, while eps, chi, Eg,
Nc, Nv and mu_n,p jump. The potentials are continuous by the architecture of
Eq. (24); the one-sided traces come from fixing the sign of rho_m on each side,
which makes d/dx the exact lateral derivative dU/dx +- dU/drho_m.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from PINN_DPV import physics

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r06.md",
    "sections": ["3", "8"],
    "equations": ["22", "24"],
    "invariants": [
        "jumps_vanish_for_smooth_flux_continuous_fields",
        "jump_detects_displacement_mismatch",
    ],
}


def sides(x_m, m, s):
    """Signs of rho_k at x = x_m from side s (+1 right, -1 left): fixed by position for k != m."""
    sign_other = jnp.where(x_m[m] >= x_m, 1.0, -1.0)
    return sign_other.at[m].set(s)


def flux_jumps(side_fields, m, x_m, mat_left, mat_right, delta):
    """([F_phi], [F_n], [F_p]) at interface m.

    `side_fields(side)` returns a function x -> (phi, phi_n, phi_p) whose cusp
    inputs carry the fixed signs `side` (see `sides`); the material on each
    side is the constant of the adjacent layer.
    """
    xm = x_m[m]
    out = []
    for s, mat in ((1.0, mat_right), (-1.0, mat_left)):
        f = side_fields(sides(x_m, m, s))
        dphi = jax.grad(lambda x: f(x)[0])(xm)
        jn, jp = physics.currents(f, xm, mat, delta)
        out.append(jnp.stack([mat["eps"] * dphi, jn, jp]))
    return out[0] - out[1]
