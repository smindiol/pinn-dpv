"""The PINN: a tanh perceptron with one cusp input per heterointerface, and its potentials.

Sec. 8: 6 hidden layers of 512 tanh units, Kaiming initialization, input
(x_hat, xi_hat) in [0, 1]^{1 + d}, output u_hat in R^3.
Eq. (24): u_hat(x_hat, xi_hat) = U(x_hat, rho_1, ..., rho_Nint, xi_hat) with
rho_m = |x_hat - x_hat_m(xi)|, so u_hat is continuous at x_hat_m and its x-derivative
may jump by 2 dU/drho_m; the one-sided traces are dU/dx +- dU/drho_m, exact by
automatic differentiation.
Fourier features (Sec. 8, the base architecture in r08; the plain perceptron is the
control): x_hat alone is projected on sin/cos(2 pi b_l x_hat) with random fixed b_l
before the first layer; rho_m and xi_hat enter unprojected.
Voltage as input (Sec. 8 variant): one network for the whole curve receives
V_hat = V / V_max in [0, 1] unprojected; every input is on a comparable scale:
x_hat, xi_hat, rho_m, V_hat in [0, 1], the Fourier features in [-1, 1].
Zero start of u_phi (A2 step 1, r08): the last-layer row of u_phi is zeroed with zero
bias, so Eq. (3) gives phi = phi_lin at initialization, whatever the features.
Equilibrium start (A2 step 2): the last-layer rows of u_phi_n, u_phi_p are zeroed
and their bias set to -Y_min / (Y_max - Y_min), so that phi_n = phi_p = 0.
Gummel alternation (Sec. 6, strategy 6): two networks, one for phi and one for
(phi_n, phi_p).
The potentials follow Eq. (3): phi with the hard Dirichlet constraint, phi_n and
phi_p affine between their training extremes; at equilibrium phi_n = phi_p = 0 hard.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from PINN_DPV import scaling

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r08.md",
    "sections": ["2", "6", "8"],
    "equations": ["3", "24"],
    "invariants": [
        "network_continuous_at_interfaces",
        "cusp_trace_jump_is_twice_drho",
        "plain_network_when_no_interface",
        "kaiming_initialization_scale",
        "potentials_meet_dirichlet_for_any_weights",
        "equilibrium_start_gives_zero_quasi_fermi",
        "phi_zero_start_gives_linear_profile",
        "voltage_input_on_unit_interval",
        "no_cusp_ablation_is_smooth_at_interfaces",
    ],
}


@dataclass(frozen=True)
class NetConfig:
    d_xi: int
    n_int: int
    width: int = 512
    depth: int = 6
    fourier_features: int = 0  # 0: base architecture; >0: Fourier variant on x_hat
    fourier_scale: float = 1.0
    split: bool = False  # True: Gummel alternation, separate phi and (phi_n, phi_p) networks
    v_max: float = 0.0  # > 0: voltage-as-input variant, V_hat = V_tilde / v_max (V_tilde units)
    cusp: bool = True  # False: experiments v04 E6 ablation, no rho_m inputs (Eq. 24 removed)

    @property
    def v_input(self) -> bool:
        return self.v_max > 0

    @property
    def n_in(self) -> int:
        nx = 2 * self.fourier_features if self.fourier_features else 1
        return nx + (self.n_int if self.cusp else 0) + self.d_xi + (1 if self.v_input else 0)


def _mlp_init(key, sizes):
    """Kaiming (He) normal weights, std sqrt(2 / fan_in); zero biases."""
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append({"W": jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), "b": jnp.zeros(n)})
    return params


def _mlp_apply(params, z):
    for layer in params[:-1]:
        z = jnp.tanh(z @ layer["W"] + layer["b"])
    return z @ params[-1]["W"] + params[-1]["b"]


def init(key, cfg: NetConfig):
    """Parameters: {"phi": mlp, "qf": mlp} when split, else {"all": mlp}; plus Fourier B if any."""
    hidden = [cfg.width] * cfg.depth
    k_net, k_net2, k_b = jax.random.split(key, 3)
    params = {}
    if cfg.split:
        params["phi"] = _mlp_init(k_net, [cfg.n_in] + hidden + [1])
        params["qf"] = _mlp_init(k_net2, [cfg.n_in] + hidden + [2])
    else:
        params["all"] = _mlp_init(k_net, [cfg.n_in] + hidden + [3])
    if cfg.fourier_features:
        # Frozen by the training loop (it never updates "B"): Tancik et al. random features.
        params["B"] = jax.random.normal(k_b, (cfg.fourier_features,)) * cfg.fourier_scale
    return params


def _features(params, cfg, x_hat, rho, xi_hat, v_hat=0.0):
    if cfg.fourier_features:
        arg = 2.0 * jnp.pi * jax.lax.stop_gradient(params["B"]) * x_hat
        fx = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)])
    else:
        fx = jnp.atleast_1d(x_hat)
    parts = [fx, jnp.atleast_1d(rho), xi_hat] if cfg.cusp else [fx, xi_hat]
    if cfg.v_input:
        parts.append(jnp.atleast_1d(v_hat))
    return jnp.concatenate(parts)


def raw(params, cfg: NetConfig, x_hat, rho, xi_hat, v_hat=0.0):
    """U(x_hat, rho, xi_hat[, V_hat]) in R^3 for explicit cusp inputs rho (shape (n_int,))."""
    z = _features(params, cfg, x_hat, rho, xi_hat, v_hat)
    if cfg.split:
        return jnp.concatenate([_mlp_apply(params["phi"], z), _mlp_apply(params["qf"], z)])
    return _mlp_apply(params["all"], z)


def u_hat(params, cfg: NetConfig, x_hat, xi_hat, x_m, side=None, v_hat=0.0):
    """u_hat of Eq. (24) at one point.

    `side` selects a one-sided trace at an interface: None uses rho_m = |x - x_m|
    for every m; an integer array of +1/-1 per interface fixes rho_m = side_m (x - x_m),
    so that d/dx gives dU/dx + side_m dU/drho_m exactly at x = x_m.
    """
    diff = x_hat - x_m
    rho = jnp.abs(diff) if side is None else side * diff
    return raw(params, cfg, x_hat, rho, xi_hat, v_hat)


@dataclass(frozen=True)
class OutputScales:
    """D_phi and the extremes of phi_n, phi_p of Eq. (3), from the training subset."""
    d_phi: float
    y_min: tuple
    y_max: tuple


def potentials(params, cfg: NetConfig, scales: OutputScales, x_hat, xi_hat, x_m,
               phi_eq0, phi_eq1, v_tilde, equilibrium=False, side=None):
    """(phi, phi_n, phi_p) of Eq. (3) at one point; phi_n = phi_p = 0 hard at equilibrium."""
    v_hat = v_tilde / cfg.v_max if cfg.v_input else 0.0
    u = u_hat(params, cfg, x_hat, xi_hat, x_m, side, v_hat)
    phi = scaling.phi_from_output(x_hat, u[0], phi_eq0, phi_eq1, v_tilde, scales.d_phi)
    if equilibrium:
        return phi, 0.0 * phi, 0.0 * phi
    phi_n = scaling.quasi_fermi_from_output(u[1], scales.y_min[0], scales.y_max[0])
    phi_p = scaling.quasi_fermi_from_output(u[2], scales.y_min[1], scales.y_max[1])
    return phi, phi_n, phi_p


def phi_zero_start(params, cfg: NetConfig):
    """A2 step 1 (r08): u_phi = 0 at every point, so Eq. (3) gives phi = phi_lin.

    The last-layer weights and bias of u_phi are zeroed; the hidden layers and the
    Fourier features keep their random initialization and receive gradient from the
    second iteration on.
    """
    params = dict(params)
    if cfg.split:
        last = dict(params["phi"][-1])
        last["W"], last["b"] = jnp.zeros_like(last["W"]), jnp.zeros_like(last["b"])
        params["phi"] = params["phi"][:-1] + [last]
    else:
        last = dict(params["all"][-1])
        last["W"] = last["W"].at[:, 0].set(0.0)
        last["b"] = last["b"].at[0].set(0.0)
        params["all"] = params["all"][:-1] + [last]
    return params


def equilibrium_start(params, cfg: NetConfig, scales: OutputScales):
    """A2 step 2 (r08): phi_n = phi_p = 0 exactly, the equilibrium solution of step 1.

    The last-layer weights of u_phi_n, u_phi_p are zeroed and their biases set to
    -Y_min / (Y_max - Y_min); this needs Y_min <= 0 <= Y_max for both.
    """
    lo, hi = np.asarray(scales.y_min, float), np.asarray(scales.y_max, float)
    if np.any(lo > 0) or np.any(hi < 0):
        raise ValueError(f"equilibrium start needs Y_min <= 0 <= Y_max, got {lo}, {hi}")
    bias = jnp.asarray(-lo / (hi - lo))
    params = dict(params)
    if cfg.split:
        last = dict(params["qf"][-1])
        last["W"], last["b"] = jnp.zeros_like(last["W"]), bias
        params["qf"] = params["qf"][:-1] + [last]
    else:
        last = dict(params["all"][-1])
        last["W"] = last["W"].at[:, 1:].set(0.0)
        last["b"] = last["b"].at[1:].set(bias)
        params["all"] = params["all"][:-1] + [last]
    return params


def count_parameters(params) -> int:
    return int(sum(np.prod(a.shape) for k, a in jax.tree_util.tree_leaves_with_path(params)
                   if "B" not in jax.tree_util.keystr(k)))
