"""Stationary coupled Poisson and drift-diffusion in quasi-Fermi variables, first-order currents.

Eq. (4): Boltzmann densities under dPV's sign convention,
n = Nc exp(phi + phi_n + chi), p = Nv exp(-phi - phi_p - chi - Eg), ni^2 = Nc Nv exp(-Eg).
Eq. (5): R_Poisson = delta^2 d/dx(eps dphi/dx) - (n - p - Ndop).
Eq. (6): J_n = delta mu_n n dphi_n/dx, J_p = delta mu_p p dphi_p/dx.
Eq. (7): R_n = delta dJ_n/dx + (G - R), R_p = delta dJ_p/dx - (G - R).
Eq. (8): R = (np - ni^2) [Br + 1 / (tau_p (n + ni e^Et) + tau_n (p + ni e^-Et))].

Every residual is returned with its additive terms T_{i,q} (Eq. 23 needs their
magnitudes), evaluated at one x_hat inside one layer, where the material is
constant (Sec. 3: bulk residuals are evaluated only in the interior of each
layer). `fields` is any function x_hat -> (phi, phi_n, phi_p); derivatives
come from automatic differentiation, never from finite differences.

`sg_currents` is dPV's own discrete counterpart of Eq. (6) (Scharfetter-Gummel,
`current.py`): it is what the dataset stores as Jn, Jp and is used here only to
read the reference, never inside the loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["3"],
    "equations": ["4", "5", "6", "7", "8"],
    "invariants": [
        "densities_positive_and_mass_action",
        "recombination_vanishes_at_equilibrium",
        "continuity_sum_is_total_current_divergence",
        "sg_currents_match_dataset",
        "poisson_residual_small_on_dpv_profiles",
        "discrete_continuity_small_on_dpv_sweep",
    ],
}


def densities(phi, phi_n, phi_p, mat):
    """n, p of Eq. (4) in units of C_0."""
    n = mat["Nc"] * jnp.exp(phi + phi_n + mat["Chi"])
    p = mat["Nv"] * jnp.exp(-phi - phi_p - mat["Chi"] - mat["Eg"])
    return n, p


def ni2(mat):
    return mat["Nc"] * mat["Nv"] * jnp.exp(-mat["Eg"])


def recombination(n, p, mat):
    """R_total of Eq. (8) in units of G_0 (Auger omitted: Cn = Cp = 0 in every family)."""
    ni = jnp.sqrt(ni2(mat))
    srh = 1.0 / (mat["tp"] * (n + ni * jnp.exp(mat["Et"])) + mat["tn"] * (p + ni * jnp.exp(-mat["Et"])))
    return (n * p - ni**2) * (mat["Br"] + srh)


def _derivs(fields, x):
    """Values, first and second x_hat-derivatives of the three potentials at x."""
    f = lambda s: jnp.stack(fields(s))  # noqa: E731
    d1 = jax.jacfwd(f)
    d2 = jax.jacfwd(d1)
    return f(x), d1(x), d2(x)


def currents(fields, x, mat, delta):
    """J_n, J_p of Eq. (6) at x_hat = x, in units of J_0."""
    u, du, _ = _derivs(fields, x)
    n, p = densities(u[0], u[1], u[2], mat)
    return delta * mat["mn"] * n * du[1], delta * mat["mp"] * p * du[2]


def poisson_terms(fields, x, mat, delta):
    """Terms of R_Poisson (Eq. 5): delta^2 eps phi'', -n, +p, +Ndop (eps constant in a layer)."""
    u, _, d2u = _derivs(fields, x)
    n, p = densities(u[0], u[1], u[2], mat)
    return jnp.stack([delta**2 * mat["eps"] * d2u[0], -n, p, mat["Ndop"] + 0.0 * n])


def continuity_terms(fields, x, mat, delta, G):
    """Terms of R_n and R_p (Eq. 7), each (delta dJ/dx, +-G, -+R), with Eq. (6) expanded.

    dJ_n/dx = delta mu_n n (phi_n'' + (phi' + phi_n') phi_n'); and
    dJ_p/dx = delta mu_p p (phi_p'' - (phi' + phi_p') phi_p').
    """
    u, du, d2u = _derivs(fields, x)
    n, p = densities(u[0], u[1], u[2], mat)
    djn = delta * mat["mn"] * n * (d2u[1] + (du[0] + du[1]) * du[1])
    djp = delta * mat["mp"] * p * (d2u[2] - (du[0] + du[2]) * du[2])
    r = recombination(n, p, mat)
    tn = jnp.stack([delta * djn, G + 0.0 * n, -r])
    tp = jnp.stack([delta * djp, -G + 0.0 * n, r])
    return tn, tp


def residuals(fields, x, mat, delta, G):
    """(R_Poisson, R_n, R_p) at one point: each residual is the sum of its terms."""
    tn, tp = continuity_terms(fields, x, mat, delta, G)
    return jnp.sum(poisson_terms(fields, x, mat, delta)), jnp.sum(tn), jnp.sum(tp)


def total_current_divergence(fields, x, mat, delta):
    """R_J = delta d/dx (J_n + J_p) (Eq. 7, third line), by automatic differentiation."""
    def jt(s):
        jn, jp = currents(fields, s, mat, delta)
        return jn + jp
    return delta * jax.grad(jt)(x)


# --- dPV's discrete currents (reference only) -------------------------------
def sg_currents(x_debye, pots, nodes):
    """Scharfetter-Gummel Jn, Jp on each grid edge, as dPV's current.Jn/Jp compute them.

    `pots` is (3, N) or (K, 3, N); `nodes` holds the per-node material arrays of
    `data.node_materials`. Returns arrays of shape (..., N-1) in units of J_0.
    """
    dgrid = jnp.diff(jnp.asarray(x_debye))
    phi, phi_n, phi_p = pots[..., 0, :], pots[..., 1, :], pots[..., 2, :]

    psi_n = nodes["Chi"] + jnp.log(nodes["Nc"]) + phi
    dpsin = psi_n[..., :-1] - psi_n[..., 1:]
    norm = jnp.where(jnp.abs(dpsin) < 1e-5, 1e-5, dpsin)
    tay = jnp.clip(dpsin, -1e-5, 1e-5)
    bn = jnp.where(jnp.abs(dpsin) < 1e-5,
                   jnp.exp(psi_n[..., :-1]) / (1 + tay / 2 + tay**2 / 6),
                   jnp.exp(psi_n[..., :-1]) * norm / (jnp.exp(norm) - 1))
    jn = nodes["mn"][:-1] * bn * (jnp.exp(phi_n[..., 1:]) - jnp.exp(phi_n[..., :-1])) / dgrid

    psi_p = nodes["Chi"] + nodes["Eg"] - jnp.log(nodes["Nv"]) + phi
    dpsip = psi_p[..., :-1] - psi_p[..., 1:]
    norm = jnp.where(jnp.abs(dpsip) < 1e-5, 1e-5, dpsip)
    tay = jnp.clip(dpsip, -1e-5, 1e-5)
    bp = jnp.where(jnp.abs(dpsip) < 1e-5,
                   jnp.exp(-psi_p[..., :-1]) / (-1 + tay / 2 - tay**2 / 6),
                   jnp.exp(-psi_p[..., :-1]) * norm / (jnp.exp(-norm) - 1))
    jp = nodes["mp"][:-1] * bp * (jnp.exp(-phi_p[..., 1:]) - jnp.exp(-phi_p[..., :-1])) / dgrid
    return jn, jp
