"""Contacts, terminal current, the J-V sweep and the conversion efficiency, as dPV computes them.

Eq. (11): ohmic contact potentials phi_eq(c) = -chi + ln(Ndop / Nc) for Ndop > 0
and -chi - Eg - ln(-Ndop / Nv) for Ndop < 0 (dPV's `bcond.boundary_phi`); the
voltage is applied at the back contact.
Eq. (12): contact conditions in multiplied form,
B_n0 = J_n(0) - S_n0 (n(0) - n_eq(0)),  B_p0 = J_p(0) + S_p0 (p(0) - p_eq(0)),
B_n1 = J_n(1) + S_n1 (n(1) - n_eq(1)),  B_p1 = J_p(1) - S_p1 (p(1) - p_eq(1)).
Eq. (13): terminal current J_0 times the spatial mean of J_n + J_p over dPV's nodes.
Eq. (14): PCE = P_max / sum_k P_in,k with P_max the maximum of dPV's piecewise
quadratic spline of V J (`spline.calcPmax`), J in A/m^2, and the standard spectrum's
power as denominator, also in E4. J_sc = J(V_0), V_oc by linear interpolation of
the zero crossing, FF = P_max / (V_oc J_sc).

The sweep is dPV's: V_j = j 0.05 V from 0, stopped at the first j >= 2 whose
current changes sign or turns negative.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from PINN_DPV import optics, physics, scaling

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["5", "8"],
    "equations": ["11", "12", "13", "14"],
    "invariants": [
        "pce_matches_dpv_dataset",
        "terminal_current_matches_dpv",
        "contact_potentials_match_dpv",
        "contact_residuals_vanish_on_equilibrium_densities",
        "calcpmax_exact_on_quadratic_power",
        "sweep_stops_like_dpv",
    ],
}

DV = 0.05  # V, dPV's vincr with 20 values per volt
MAX_STEPS = 100


def contact_potential(mat):
    """phi_eq at a contact (Eq. 11), from the dimensionless material of its node."""
    return jnp.where(
        mat["Ndop"] > 0,
        -mat["Chi"] + jnp.log(jnp.abs(mat["Ndop"] / mat["Nc"])),
        -mat["Chi"] - mat["Eg"] - jnp.log(jnp.abs(-mat["Ndop"] / mat["Nv"])))


def equilibrium_densities(mat):
    """(n_eq, p_eq) at a contact: Eq. (4) with phi = phi_eq(c), phi_n = phi_p = 0."""
    return physics.densities(contact_potential(mat), 0.0, 0.0, mat)


def contact_residuals(fields, mat0, mat1, s, delta):
    """(B_n0, B_n1, B_p0, B_p1) of Eq. (12); s = (S_n0, S_n1, S_p0, S_p1) in units of v_0."""
    sn0, sn1, sp0, sp1 = s
    jn0, jp0 = physics.currents(fields, 0.0, mat0, delta)
    jn1, jp1 = physics.currents(fields, 1.0, mat1, delta)
    u0, u1 = fields(0.0), fields(1.0)
    n0, p0 = physics.densities(*u0, mat0)
    n1, p1 = physics.densities(*u1, mat1)
    neq0, peq0 = equilibrium_densities(mat0)
    neq1, peq1 = equilibrium_densities(mat1)
    return (jn0 - sn0 * (n0 - neq0), jn1 + sn1 * (n1 - neq1),
            jp0 + sp0 * (p0 - peq0), jp1 - sp1 * (p1 - peq1))


def terminal_current(jn, jp):
    """Eq. (13): J_0 times the mean of J_n + J_p over the evaluation points, in A/cm^2."""
    return scaling.CURRENT * jnp.mean(jnp.asarray(jn) + jnp.asarray(jp), axis=-1)


# --- dPV's spline.calcPmax ----------------------------------------------------
def qspline(x, y):
    """Piecewise quadratic through (x, y) with continuous slope and a = 0 on the first piece."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = x.size
    M = np.zeros((3 * (n - 1), 3 * (n - 1)))
    z = np.zeros(3 * (n - 1))
    M[0, 0] = 1
    z[1] = y[0]
    for i in range(n - 1):
        M[3 * i + 1, 3 * i:3 * i + 3] = [x[i] ** 2, x[i], 1]
        z[3 * i + 1] = y[i]
        M[3 * i + 2, 3 * i:3 * i + 3] = [x[i + 1] ** 2, x[i + 1], 1]
        z[3 * i + 2] = y[i + 1]
    for i in range(n - 2):
        M[3 * i + 3, 3 * i:3 * i + 6] = [2 * x[i + 1], 1, 0, -2 * x[i + 1], -1, 0]
    coef = np.linalg.solve(M, z)
    return coef[::3], coef[1::3], coef[2::3]


def _quadratic(x, coef):
    a, b, c = coef
    return a * x**2 + b * x + c


def calc_pmax(v, j):
    """(P_max, V_max) of dPV's calcPmax: the largest spline value among piece ends and vertices."""
    v = np.asarray(v, dtype=np.float64)
    p = v * np.asarray(j, dtype=np.float64)
    coef = qspline(v, p)
    a, b, _ = coef
    xl, xu = v[:-1], v[1:]
    filla = np.where(a != 0, a, 1)
    xm = np.clip(-b / (2 * filla), xl, xu)
    xall = np.concatenate([xl, xu, xm])
    yall = np.concatenate([_quadratic(xl, coef), _quadratic(xu, coef), _quadratic(xm, coef)])
    k = int(np.argmax(yall))
    return float(yall[k]), float(xall[k])


# --- The sweep and the summary -------------------------------------------------
def sweep_length(j) -> int:
    """Number of voltages dPV keeps: it stops at the first k >= 2 with J_{k-1} J_k <= 0 or J_k < 0."""
    j = np.asarray(j, dtype=np.float64)
    for k in range(2, min(j.size, MAX_STEPS)):
        if j[k - 1] * j[k] <= 0 or j[k] < 0:
            return k + 1
    return min(j.size, MAX_STEPS)


def voltages(k: int):
    return DV * np.arange(k)


def jv_summary(v, j_acm2):
    """PCE, P_max, V_max, J_sc, V_oc and FF from a J-V curve in V and A/cm^2 (Eq. 14).

    The curve is first cut where dPV would have stopped it.
    """
    k = sweep_length(j_acm2)
    v = np.asarray(v, dtype=np.float64)[:k]
    j = np.asarray(j_acm2, dtype=np.float64)[:k]
    pmax, vmax = calc_pmax(v, j * 1e4)  # A/cm^2 -> A/m^2
    jsc = j[0] * 1e4
    voc = float("nan")
    if j.size >= 2 and j[-1] * j[-2] <= 0 and j[-2] != j[-1]:
        voc = float(v[-2] + j[-2] * (v[-1] - v[-2]) / (j[-2] - j[-1]))
    ff = pmax / (voc * jsc) if np.isfinite(voc) and jsc > 0 else float("nan")
    return {"pce": pmax / optics.standard_power(), "pmax": pmax, "vmax": vmax,
            "jsc": jsc, "voc": voc, "ff": ff, "n_voltages": int(k)}
