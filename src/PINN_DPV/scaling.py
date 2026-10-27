"""Debye scaling of dPV, the unit-interval coordinate and the output normalization.

Eq. (1) fixes the characteristic scales exactly as dPV's `scales.py` does
(same constants, same expressions), so every dimensionless quantity here is
bit-comparable with the dataset. Eq. (2) maps the Debye coordinate to
x_hat = x / L_total and carries delta = L_D / L_total into every derivative.
Eq. (3) normalizes each device parameter to [0, 1] (log10 for dopings, identity
for thicknesses and eta_PL), builds phi with a hard constraint that meets both
Dirichlet conditions of Eq. (11) for every voltage, and maps the quasi-Fermi
outputs affinely between their training-set extremes.

Assumption stated by the revision: T = 300 K. The dataset was generated with
CODATA 2022 constants (scipy >= 1.15); the length scale it implies matches
this module's to relative error 0.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy import constants as const

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["2"],
    "equations": ["1", "2", "3"],
    "invariants": [
        "debye_scales_match_dpv",
        "dataset_grid_is_debye_scaled",
        "xi_normalization_roundtrip",
        "phi_hard_constraint_meets_dirichlet",
        "quasi_fermi_affine_roundtrip",
        "float64_enabled",
    ],
}

# --- Eq. (1), spelled as dPV's scales.py spells it --------------------------
NM = 1e-9  # m
CM = 1e-2  # m
KB = const.k  # J / K
HC = const.c * const.h  # J m
EV = const.eV  # J
Q = const.e  # C
EPS0 = const.epsilon_0 * 1e-2  # C / (V cm)
T = 300.0  # K

DENSITY = 1e19  # C_0, 1 / cm^3
MOBILITY = 1.0  # mu_0, cm^2 / (V s)
ENERGY = KB * T / Q  # V_th, V
TIME = EPS0 / Q / DENSITY / MOBILITY  # t_0, s
LENGTH = math.sqrt(EPS0 * KB * T / (Q**2 * DENSITY))  # L_D, cm
VELOCITY = LENGTH / TIME  # v_0, cm / s
CURRENT = KB * T * DENSITY * MOBILITY / LENGTH  # J_0, A / cm^2
GRATEDENS = DENSITY * MOBILITY * KB * T / (Q * LENGTH**2)  # G_0, 1 / (cm^3 s)

#: Scale that divides each physical input, as in dPV's `scales.units`.
UNITS = {
    "grid": LENGTH, "eps": 1.0, "Chi": ENERGY, "Eg": ENERGY, "Nc": DENSITY,
    "Nv": DENSITY, "mn": MOBILITY, "mp": MOBILITY, "Ndop": DENSITY,
    "Et": ENERGY, "tn": TIME, "tp": TIME, "Br": 1.0 / (TIME * DENSITY),
    "Snl": VELOCITY, "Snr": VELOCITY, "Spl": VELOCITY, "Spr": VELOCITY,
}


def debye_scales() -> dict:
    """The eight characteristic scales of Eq. (1)."""
    return {"V_th": ENERGY, "C_0": DENSITY, "mu_0": MOBILITY, "L_D": LENGTH,
            "t_0": TIME, "v_0": VELOCITY, "G_0": GRATEDENS, "J_0": CURRENT}


# --- Eq. (2) ----------------------------------------------------------------
def delta(l_total_cm):
    """delta = L_D / L_total, the factor each x_hat derivative carries (Eq. 2)."""
    return LENGTH / l_total_cm


def x_hat_from_debye(x_debye):
    """x_hat = x / L_total from a dPV grid in Debye units (dPV's `design.grid`).

    The dataset stores that grid under the name `x_cm`, but its values are
    dPV's dimensionless grid (cm divided by L_D), not centimetres.
    """
    x = np.asarray(x_debye, dtype=np.float64)
    return x / x[-1]


def debye_to_cm(x_debye):
    """Physical position in cm of a dPV grid stored in Debye units."""
    return np.asarray(x_debye, dtype=np.float64) * LENGTH


# --- Eq. (3), device parameters ---------------------------------------------
def _g(value, escala):
    return np.log10(value) if escala == "log" else np.asarray(value, dtype=np.float64)


def normalize_xi(values, specs):
    """xi_hat_j = (g_j(xi_j) - g_j_min) / (g_j_max - g_j_min) (Eq. 3).

    `specs` is the dataset's `parametros_xi`: a list of dicts with `nombre`,
    `escala` ("log" or "lin"), `min` and `max`, in the order of xi.
    """
    out = []
    for v, s in zip(values, specs):
        lo, hi = _g(s["min"], s["escala"]), _g(s["max"], s["escala"])
        out.append((_g(v, s["escala"]) - lo) / (hi - lo))
    return np.asarray(out, dtype=np.float64)


def denormalize_xi(xi_hat, specs):
    """Inverse of `normalize_xi`: physical xi from xi_hat in [0, 1]^d."""
    out = []
    for u, s in zip(xi_hat, specs):
        lo, hi = _g(s["min"], s["escala"]), _g(s["max"], s["escala"])
        g = lo + float(u) * (hi - lo)
        out.append(10.0**g if s["escala"] == "log" else g)
    return np.asarray(out, dtype=np.float64)


# --- Eq. (3), outputs ---------------------------------------------------------
def phi_linear(x_hat, phi_eq0, phi_eq1, v_tilde):
    """phi_lin = phi_eq(0) + (phi_eq(1) + V~ - phi_eq(0)) x_hat."""
    return phi_eq0 + (phi_eq1 + v_tilde - phi_eq0) * x_hat


def phi_from_output(x_hat, u_phi, phi_eq0, phi_eq1, v_tilde, d_phi):
    """phi = phi_lin + 4 x_hat (1 - x_hat) D_phi u_phi: Dirichlet by construction."""
    return phi_linear(x_hat, phi_eq0, phi_eq1, v_tilde) + 4.0 * x_hat * (1.0 - x_hat) * d_phi * u_phi


def quasi_fermi_from_output(u_k, y_min, y_max):
    """Y_k = Y_min + (Y_max - Y_min) u_k, k in {phi_n, phi_p}."""
    return y_min + (y_max - y_min) * u_k


def quasi_fermi_to_output(y_k, y_min, y_max):
    """Inverse of `quasi_fermi_from_output`."""
    return (y_k - y_min) / (y_max - y_min)


def output_scales(profiles):
    """D_phi and the extremes of phi_n, phi_p over training profiles (Sec. 2).

    `profiles` yields (x_hat, pots, phi_eq0, phi_eq1, v_tilde) per device, where
    pots has shape (K, 3, N): phi, phi_n, phi_p at each voltage of the sweep.
    D_phi = max |phi - phi_lin| over every device, voltage and node.
    """
    d_phi, lo, hi = 0.0, np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for x_hat, pots, phi_eq0, phi_eq1, v_tilde in profiles:
        lin = phi_linear(x_hat[None, :], phi_eq0, phi_eq1, np.asarray(v_tilde)[:, None])
        d_phi = max(d_phi, float(np.max(np.abs(pots[:, 0, :] - lin))))
        lo = np.minimum(lo, pots[:, 1:, :].min(axis=(0, 2)))
        hi = np.maximum(hi, pots[:, 1:, :].max(axis=(0, 2)))
    return {"D_phi": d_phi, "Y_min": lo, "Y_max": hi}


def as_jax(x):
    return jnp.asarray(x, dtype=jnp.float64)
