"""The doubly weighted loss and the per-component scales and floors.

Eq. (15): L(theta; V_j) = sum_i lambda_i / sigma_i^2 L_i over
I = {Poisson, n, p, BCn, BCp, int_phi, int_n, int_p, dat}: bulk residuals
averaged over N_Q collocation points in (x_hat, xi_hat), contact terms
<B_0^2 + B_1^2> and interface terms <sum_m [F_k]^2> averaged over the devices of
the batch, and the sparse-state data term with each error divided by the scale
that normalizes its output in Eq. (3) (sigma_dat = 1). R_J is not a component.
Eq. (23): sigma_i^2 = < 1/N_g sum_g (sum_q |T_iq|)^2 > and
L_i^floor = < 1/N_g sum_g R_i^2 >, over the training subset and every voltage,
with the network's own continuous operators applied to a per-layer cubic
interpolant of dPV's potentials at its grid nodes: layer interiors for bulk
residuals, contacts for Eq. (12), lateral traces at x_hat_m for Eq. (22).

Every component is computed from one function returning its additive terms
T_iq; the loss sums them and the scale sums their magnitudes, so the two can
never describe different residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from PINN_DPV import contacts, interfaces, network, optics, physics, scaling  # noqa: F401

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r05.md",
    "sections": ["6"],
    "equations": ["15", "23"],
    "invariants": [
        "loss_is_weighted_sum_of_components",
        "residual_equals_sum_of_terms",
        "spline_fields_interpolate_nodes",
        "sigma_bounds_residual",
        "data_term_zero_on_reference",
    ],
}

COMPONENTS = ("Poisson", "n", "p", "BCn", "BCp", "int_phi", "int_n", "int_p", "dat")
EQUILIBRIUM_COMPONENTS = ("Poisson", "int_phi")


def active_components(n_int: int, equilibrium: bool = False, with_data: bool = False):
    """Components present: no interface terms in E1; Poisson and int_phi at equilibrium."""
    base = EQUILIBRIUM_COMPONENTS if equilibrium else COMPONENTS[:-1]
    out = [c for c in base if n_int > 0 or not c.startswith("int")]
    if with_data and not equilibrium:
        out.append("dat")
    return tuple(out)


# --- Per-device quantities, from xi_hat ----------------------------------------
def device(fs, xi_hat):
    """Layout plus contact potentials and the materials of both contact layers."""
    lay = fs.layout(xi_hat)
    mat0 = fs.material(jnp.asarray(0), lay["ndop"])
    mat1 = fs.material(jnp.asarray(fs.n_layers - 1), lay["ndop"])
    lay["mat0"], lay["mat1"] = mat0, mat1
    lay["phi_eq0"] = contacts.contact_potential(mat0)
    lay["phi_eq1"] = contacts.contact_potential(mat1)
    return lay


# --- Terms of each component at one point / device -------------------------------
def bulk_terms(fields, x_hat, fs, dev, equilibrium=False):
    """Terms of R_Poisson (4,), R_n (3,), R_p (3,) at x_hat inside a layer."""
    layer = fs.layer_of(x_hat, dev["x_layers"])
    mat = fs.material(layer, dev["ndop"])
    g = 0.0 if equilibrium else optics.generation_jax(fs, x_hat, dev)
    tp = physics.poisson_terms(fields, x_hat, mat, dev["delta"])
    tn, tpp = physics.continuity_terms(fields, x_hat, mat, dev["delta"], g)
    return tp, tn, tpp


def contact_terms(fields, fs, dev):
    """Terms of B_n0, B_n1, B_p0, B_p1 (Eq. 12), each (3,): J, -+S n, +-S n_eq."""
    sn0, sn1, sp0, sp1 = fs.contacts
    mat0, mat1, delta = dev["mat0"], dev["mat1"], dev["delta"]
    jn0, jp0 = physics.currents(fields, 0.0, mat0, delta)
    jn1, jp1 = physics.currents(fields, 1.0, mat1, delta)
    n0, p0 = physics.densities(*fields(0.0), mat0)
    n1, p1 = physics.densities(*fields(1.0), mat1)
    neq0, peq0 = contacts.equilibrium_densities(mat0)
    neq1, peq1 = contacts.equilibrium_densities(mat1)
    bn = jnp.stack([jnp.stack([jn0, -sn0 * n0, sn0 * neq0]), jnp.stack([jn1, sn1 * n1, -sn1 * neq1])])
    bp = jnp.stack([jnp.stack([jp0, sp0 * p0, -sp0 * peq0]), jnp.stack([jp1, -sp1 * p1, sp1 * peq1])])
    return bn, bp  # (2 contacts, 3 terms) each


def interface_terms(side_fields, fs, dev):
    """Terms (F(x_m^+), -F(x_m^-)) of [F_phi], [F_n], [F_p] at each interface: (n_int, 3, 2)."""
    out = []
    for m in range(fs.n_int):
        mat_l = fs.material(jnp.asarray(m), dev["ndop"])
        mat_r = fs.material(jnp.asarray(m + 1), dev["ndop"])
        f_right = _lateral_flux(side_fields, m, dev, mat_r, 1.0)
        f_left = _lateral_flux(side_fields, m, dev, mat_l, -1.0)
        out.append(jnp.stack([f_right, -f_left], axis=-1))
    return jnp.stack(out) if out else jnp.zeros((0, 3, 2))


def _lateral_flux(side_fields, m, dev, mat, s):
    f = side_fields(interfaces.sides(dev["x_m"], m, s))
    xm = dev["x_m"][m]
    dphi = jax.grad(lambda x: f(x)[0])(xm)
    jn, jp = physics.currents(f, xm, mat, dev["delta"])
    return jnp.stack([mat["eps"] * dphi, jn, jp])


# --- The network as fields ------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    """Everything fixed for one family-stage: architecture, output scales, sigma_i^2."""
    fs: object
    cfg: network.NetConfig
    scales: network.OutputScales
    sigma2: dict = field(default_factory=dict)


def net_fields(params, prob: Problem, xi_hat, dev, v_tilde, equilibrium=False, side=None):
    def f(x):
        return network.potentials(params, prob.cfg, prob.scales, x, xi_hat, dev["x_m"],
                                  dev["phi_eq0"], dev["phi_eq1"], v_tilde, equilibrium, side)
    return f


def components(params, prob: Problem, batch, v_tilde, equilibrium=False):
    """Unweighted L_i of Eq. (15) for one batch.

    batch: {"xq": (N_Q,), "xiq": (N_Q, d)} interior collocation, {"xib": (N_B, d)}
    boundary/interface devices, and optionally {"xid": (B, d), "xs": (B, K_s),
    "ref": (B, 3, K_s)} for the data term.
    """
    fs = prob.fs

    def interior(x, xi):
        dev = device(fs, xi)
        tp, tn, tpp = bulk_terms(net_fields(params, prob, xi, dev, v_tilde, equilibrium), x, fs, dev,
                                 equilibrium)
        return jnp.sum(tp), jnp.sum(tn), jnp.sum(tpp)

    rp, rn, rpp = jax.vmap(interior)(batch["xq"], batch["xiq"])
    out = {"Poisson": jnp.mean(rp**2)}
    if not equilibrium:
        out["n"], out["p"] = jnp.mean(rn**2), jnp.mean(rpp**2)

        def bc(xi):
            dev = device(fs, xi)
            bn, bp = contact_terms(net_fields(params, prob, xi, dev, v_tilde), fs, dev)
            return jnp.sum(jnp.sum(bn, -1) ** 2), jnp.sum(jnp.sum(bp, -1) ** 2)

        lbn, lbp = jax.vmap(bc)(batch["xib"])
        out["BCn"], out["BCp"] = jnp.mean(lbn), jnp.mean(lbp)

    if fs.n_int:
        def jumps(xi):
            dev = device(fs, xi)
            sf = lambda s: net_fields(params, prob, xi, dev, v_tilde, equilibrium, side=s)  # noqa: E731
            t = interface_terms(sf, fs, dev)
            return jnp.sum(jnp.sum(t, -1) ** 2, axis=0)  # (3,): sum over interfaces

        j = jnp.mean(jax.vmap(jumps)(batch["xib"]), axis=0)
        out["int_phi"] = j[0]
        if not equilibrium:
            out["int_n"], out["int_p"] = j[1], j[2]

    if "xid" in batch and not equilibrium:
        out["dat"] = data_term(params, prob, batch, v_tilde)
    return out


def data_term(params, prob: Problem, batch, v_tilde):
    """L_dat of Eq. (15): normalized squared errors at K_s positions per device."""
    span = jnp.asarray(prob.scales.y_max) - jnp.asarray(prob.scales.y_min)

    def one(xi, xs, ref):
        dev = device(prob.fs, xi)
        f = net_fields(params, prob, xi, dev, v_tilde)
        pred = jax.vmap(lambda x: jnp.stack(f(x)))(xs)  # (K_s, 3)
        e_phi = ((pred[:, 0] - ref[0]) / prob.scales.d_phi) ** 2
        e_qf = jnp.sum(((pred[:, 1:] - ref[1:].T) / span) ** 2, axis=-1)
        return jnp.mean(e_phi + e_qf)

    return jnp.mean(jax.vmap(one)(batch["xid"], batch["xs"], batch["ref"]))


def total(comps: dict, lambdas: dict, sigma2: dict):
    """Eq. (15): sum_i lambda_i / sigma_i^2 L_i over the components present."""
    return sum(lambdas[k] / sigma2.get(k, 1.0) * v for k, v in comps.items())


# --- Eq. (23): reference fields from dPV profiles ----------------------------------
def layer_splines(x_hat, pots_k, layer_index, n_layers):
    """Per-layer cubic interpolant of (phi, phi_n, phi_p) at one voltage.

    Returns breakpoints (L, N) padded with +inf and coefficients (L, N-1, 3, 4);
    each layer is fitted only on its own nodes and extrapolated by its end pieces.
    """
    n = x_hat.size
    bps = np.full((n_layers, n), np.inf)
    coef = np.zeros((n_layers, n - 1, 3, 4))
    for layer in range(n_layers):
        idx = np.flatnonzero(layer_index == layer)
        xs = x_hat[idx]
        bps[layer, :idx.size] = xs
        for c in range(3):
            cs = CubicSpline(xs, pots_k[c, idx])
            coef[layer, :idx.size - 1, c, :] = cs.c.T
        coef[layer, idx.size - 1:, :, :] = coef[layer, idx.size - 2, :, :]
        bps[layer, idx.size:] = np.inf
    return jnp.asarray(bps), jnp.asarray(coef)


def spline_fields(bps, coef, layer):
    """x -> (phi, phi_n, phi_p) from the spline of one layer (differentiable in x)."""
    def f(x):
        b = bps[layer]
        k = jnp.clip(jnp.searchsorted(b, x, side="right") - 1, 0, None)
        npts = jnp.sum(jnp.isfinite(b))
        k = jnp.minimum(k, npts - 2)
        dx = x - b[k]
        c = coef[layer, k]  # (3, 4), highest power first
        v = ((c[:, 0] * dx + c[:, 1]) * dx + c[:, 2]) * dx + c[:, 3]
        return v[0], v[1], v[2]
    return f


def _reference_core(fs, equilibrium):
    """Jitted Eq. (23) evaluation for one family-stage at fixed grid shape."""

    def core(xi_hat, x_hat, layer_index, interior, bps, coef):
        dev = device(fs, xi_hat)

        def bulk(x, layer):
            f = spline_fields(bps, coef, layer)
            return bulk_terms(f, x, fs, dev, equilibrium)

        tp, tn, tpp = jax.vmap(bulk)(x_hat, layer_index)
        w = interior / jnp.sum(interior)

        def stats(t):  # mean over interior nodes of (sum|T|)^2 and R^2
            return jnp.sum(w * jnp.sum(jnp.abs(t), -1) ** 2), jnp.sum(w * jnp.sum(t, -1) ** 2)

        out = {"Poisson": stats(tp)}
        if not equilibrium:
            out["n"], out["p"] = stats(tn), stats(tpp)
            f0 = spline_fields(bps, coef, 0)
            f1 = spline_fields(bps, coef, fs.n_layers - 1)
            both = lambda x: jax.lax.cond(x < 0.5, f0, f1, x)  # noqa: E731
            bn, bp = contact_terms(both, fs, dev)
            for name, t in (("BCn", bn), ("BCp", bp)):
                out[name] = (jnp.mean(jnp.sum(jnp.abs(t), -1) ** 2), jnp.mean(jnp.sum(t, -1) ** 2))
        if fs.n_int:
            lat_all = []
            for m in range(fs.n_int):
                fl, fr = spline_fields(bps, coef, m), spline_fields(bps, coef, m + 1)
                mat_l = fs.material(jnp.asarray(m), dev["ndop"])
                mat_r = fs.material(jnp.asarray(m + 1), dev["ndop"])
                xm_m = dev["x_m"][m]
                lat = []
                for f, mat in ((fr, mat_r), (fl, mat_l)):
                    dphi = jax.grad(lambda x: f(x)[0])(xm_m)
                    jn, jp = physics.currents(f, xm_m, mat, dev["delta"])
                    lat.append(jnp.stack([mat["eps"] * dphi, jn, jp]))
                lat_all.append(jnp.stack([lat[0], -lat[1]], axis=-1))  # (3, 2)
            t = jnp.stack(lat_all)  # (n_int, 3, 2)
            names = ("int_phi",) if equilibrium else ("int_phi", "int_n", "int_p")
            for i, name in enumerate(names):
                ti = t[:, i, :]
                out[name] = (jnp.mean(jnp.sum(jnp.abs(ti), -1) ** 2), jnp.mean(jnp.sum(ti, -1) ** 2))
        return out

    return jax.jit(core)


_CORES = {}


def reference_terms(fs, dev_ref, xi_hat, k, equilibrium=False):
    """(sigma-term, floor-term) of every component on dPV's profiles at voltage k (Eq. 23).

    `dev_ref` is a `data.Device` with profiles; each value is the mean over the
    component's evaluation points g of (sum_q |T_iq|)^2 and of R_i^2.
    """
    key = (id(fs), equilibrium)
    if key not in _CORES:
        _CORES[key] = _reference_core(fs, equilibrium)
    x_hat = scaling.x_hat_from_debye(dev_ref.x)
    dev = device(fs, jnp.asarray(xi_hat))
    xl = np.asarray(dev["x_layers"])
    layer_index = np.sum(x_hat[:, None] >= xl[None, :], axis=1)
    pots = np.stack([dev_ref.phi_eq, 0 * dev_ref.phi_eq, 0 * dev_ref.phi_eq]) if equilibrium \
        else dev_ref.pots[k]
    bps, coef = layer_splines(x_hat, pots, layer_index, fs.n_layers)
    interior = np.zeros(x_hat.size)
    for layer in range(fs.n_layers):
        idx = np.flatnonzero(layer_index == layer)
        interior[idx[1:-1]] = 1.0
    out = _CORES[key](jnp.asarray(xi_hat), jnp.asarray(x_hat), jnp.asarray(layer_index),
                      jnp.asarray(interior), bps, coef)
    return {c: (float(a), float(b)) for c, (a, b) in out.items()}


def sigma_and_floor(fs, devices, equilibrium_too=True):
    """sigma_i^2 and L_i^floor of Eq. (23) over training devices and every voltage."""
    acc = {}
    for d in devices:
        xi_hat = np.asarray(d.u)
        ks = [(k, False) for k in range(len(d.V))]
        if equilibrium_too:
            ks.append((None, True))
        for k, eq in ks:
            terms = reference_terms(fs, d, xi_hat, k, equilibrium=eq)
            for c, (s2, r2) in terms.items():
                key = f"{c}@eq" if eq else c
                a = acc.setdefault(key, [0.0, 0.0, 0])
                a[0] += s2
                a[1] += r2
                a[2] += 1
    sigma2 = {k: v[0] / v[2] for k, v in acc.items()}
    floor = {k: v[1] / v[2] for k, v in acc.items()}
    return sigma2, floor
