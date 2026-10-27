"""Level 2 - properties and invariants. The bridge between proposal and code.

One test per mathematical claim the proposal makes. The name MUST be
`test_<invariant_id>`, matching an id declared in a module's
`__provenance__["invariants"]`; verification pairs them by that exact name.

Each docstring cites the passage it enforces. Tests marked `dataset` read the
dPV reference files (r08 Sec. 7) from `data.dataset_root()`; they compare this
package against values dPV itself stored, never against values computed here.
"""

from __future__ import annotations

import functools
import os

import jax.numpy as jnp
import numpy as np
import pytest

from PINN_DPV import contacts, data, optics, physics, scaling

TOL = 1e-10
FAMILY_STAGES = ["E1A", "E1B", "E2A", "E2B", "E3A", "E3B", "E4A", "E4B"]
SAMPLE_KEYS = ["000000", "000007", "000500"]

needs_dataset = pytest.mark.skipif(
    not all(os.path.exists(p) for fs in FAMILY_STAGES for p in data.canonical_paths(fs)),
    reason="dPV reference files not found under data.dataset_root()")


@functools.lru_cache(maxsize=None)
def sample(family_stage):
    """A few converged devices of the first block of one family-stage, with profiles."""
    path = data.canonical_paths(family_stage)[0]
    return tuple(data.iter_devices(path, keys=SAMPLE_KEYS))


def nodes_of(dev):
    return data.node_materials(dev.family, dev.xi, dev.x)


def jnodes(dev):
    return {k: jnp.asarray(v) for k, v in nodes_of(dev).items()}


def discrete_poisson(dev, nd, phi, phi_n, phi_p):
    """dPV's poisson.pois on interior nodes, and the sum of its term magnitudes."""
    dg = np.diff(dev.x)
    ad = (dg[:-1] + dg[1:]) / 2
    ae = (nd["eps"][1:] + nd["eps"][:-1]) / 2
    n, p = (np.asarray(a) for a in physics.densities(phi, phi_n, phi_p, nd))
    flux = (ae[:-1] * np.diff(phi)[:-1] / dg[:-1] - ae[1:] * np.diff(phi)[1:] / dg[1:]) / ad
    charge = (-n + p + nd["Ndop"])[1:-1]
    scale = np.abs(flux) + np.abs(n[1:-1]) + np.abs(p[1:-1]) + np.abs(nd["Ndop"][1:-1])
    return flux - charge, scale


# --- scaling ----------------------------------------------------------------
def test_debye_scales_match_dpv() -> None:
    """r08 Sec. 2, Eq. (1): G_0 = C_0 / t_0 and J_0 = q C_0 L_D / t_0 equal dPV's own forms."""
    s = scaling.debye_scales()
    assert abs(s["G_0"] / (s["C_0"] / s["t_0"]) - 1) < 1e-12
    assert abs(s["J_0"] / (scaling.Q * s["C_0"] * s["L_D"] / s["t_0"]) - 1) < 1e-12
    assert abs(s["v_0"] - s["L_D"] / s["t_0"]) < 1e-12 * s["v_0"]


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_dataset_grid_is_debye_scaled(fs) -> None:
    """r08 Sec. 2, Eq. (1)-(2): the stored grid `x_cm` is x / L_D, so its end times L_D is L_total."""
    for dev in sample(fs):
        assert abs(dev.x[-1] * scaling.LENGTH / data.l_total_cm(dev.family, dev.xi) - 1) < 1e-9
        assert dev.x[0] == 0.0


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_xi_normalization_roundtrip(fs) -> None:
    """r08 Sec. 2, Eq. (3): xi_hat reproduces the stored u, and its inverse recovers xi."""
    specs = data.xi_specs(data.canonical_paths(fs)[0])
    for dev in sample(fs):
        values = [dev.xi[s["nombre"]] for s in specs]
        u = scaling.normalize_xi(values, specs)
        assert np.max(np.abs(u - dev.u)) < 1e-9
        back = scaling.denormalize_xi(u, specs)
        assert np.max(np.abs(back / np.asarray(values) - 1)) < 1e-9


def test_phi_hard_constraint_meets_dirichlet() -> None:
    """r08 Sec. 2, Eq. (3) and (11): phi(0) = phi_eq(0) and phi(1) = phi_eq(1) + V~ for any output."""
    for u_phi in (-3.0, 0.0, 7.5):
        for v in (0.0, 12.3):
            assert abs(scaling.phi_from_output(0.0, u_phi, -40.0, -55.0, v, 9.0) - (-40.0)) < TOL
            assert abs(scaling.phi_from_output(1.0, u_phi, -40.0, -55.0, v, 9.0) - (-55.0 + v)) < TOL
    assert abs(scaling.phi_from_output(0.5, 1.0, 0.0, 0.0, 0.0, 2.0) - 2.0) < TOL


def test_quasi_fermi_affine_roundtrip() -> None:
    """r08 Sec. 2, Eq. (3): Y_k = Y_min + (Y_max - Y_min) u_k is inverted exactly."""
    y = np.linspace(-30.0, 4.0, 11)
    u = scaling.quasi_fermi_to_output(y, -30.0, 4.0)
    assert u[0] == 0.0 and abs(u[-1] - 1.0) < TOL
    assert np.max(np.abs(scaling.quasi_fermi_from_output(u, -30.0, 4.0) - y)) < TOL


def test_float64_enabled() -> None:
    """r08 Sec. 8: every computation runs in double precision, as dPV does."""
    assert jnp.zeros(1).dtype == jnp.float64
    assert (jnp.asarray(1.0) + 1e-12).dtype == jnp.float64
    assert float(jnp.asarray(1.0) + 1e-12) != 1.0


# --- data ---------------------------------------------------------------------
@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_device_materials_match_dataset(fs) -> None:
    """r08 Sec. 2 and 7: material parameters are piecewise constant, jumping only at x_hat_m(xi).

    The rebuilt node arrays change value exactly once per declared heterointerface,
    on the grid interval that contains x_hat_m.
    """
    for dev in sample(fs):
        nd = nodes_of(dev)
        x_hat = scaling.x_hat_from_debye(dev.x)
        jumps = np.flatnonzero(np.diff(nd["Eg"]) != 0)
        xm = data.interfaces_hat(dev.family, dev.xi)
        assert len(jumps) == len(xm)
        for j, m in zip(jumps, xm):
            assert x_hat[j] - 1e-9 <= m <= x_hat[j + 1] + 1e-9
        assert np.all(nd["eps"] > 0) and np.all(nd["Nc"] > 0)


def test_split_is_partition_by_device() -> None:
    """r08 Sec. 7: 80/10/10 by device, never by point or voltage; no device in two subsets."""
    keys = [f"b000/sims/{i:06d}" for i in range(1000)]
    parts = data.split(keys, seed=3)
    assert (len(parts["train"]), len(parts["val"]), len(parts["test"])) == (800, 100, 100)
    union = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    assert union == set(keys)
    assert not (set(parts["train"]) & set(parts["test"]))
    assert not (set(parts["val"]) & set(parts["test"]))


def test_split_depends_on_seed() -> None:
    """r08 Sec. 8-9: each seed changes the device partition; the same seed reproduces it."""
    keys = [f"b000/sims/{i:06d}" for i in range(200)]
    first = data.split(keys, 0)
    again = data.split(list(reversed(keys)), 0)  # same devices, listed in another order
    assert first == again
    assert first["test"] != data.split(keys, 1)["test"]


def test_corners_go_to_test() -> None:
    """r08 Sec. 7: the 2^d corners of the xi hypercube belong to test whatever the seed."""
    keys = [f"b000/sims/{i:06d}" for i in range(50)]
    corners = ["b000/esquinas/-00001", "b000/esquinas/-00002"]
    for seed in range(5):
        parts = data.split(keys, seed, corners=corners)
        assert set(corners) <= set(parts["test"])
        assert not set(corners) & (set(parts["train"]) | set(parts["val"]))


# --- physics --------------------------------------------------------------------
def test_densities_positive_and_mass_action() -> None:
    """r08 Sec. 3, Eq. (4): n, p > 0 and n p = ni^2 exp(phi_n - phi_p)."""
    mat = {"Nc": 0.08, "Nv": 1.8, "Chi": 150.9, "Eg": 58.0}
    for phi, pn, pp in [(-160.0, 0.0, 0.0), (-150.0, 2.0, -3.0), (-170.0, -5.0, 4.0)]:
        n, p = physics.densities(phi, pn, pp, mat)
        assert n > 0 and p > 0
        assert abs(n * p / (physics.ni2(mat) * np.exp(pn - pp)) - 1) < 1e-12


def test_recombination_vanishes_at_equilibrium() -> None:
    """r08 Sec. 3, Eq. (8): R = 0 when n p = ni^2 (phi_n = phi_p), for SRH and radiative terms."""
    mat = {"Nc": 0.39, "Nv": 0.27, "Chi": 150.9, "Eg": 58.0, "Et": 0.0, "Br": 1.6, "tn": 18.0, "tp": 18.0}
    for phi in (-170.0, -160.0, -150.0):
        n, p = physics.densities(phi, 1.5, 1.5, mat)
        assert abs(float(physics.recombination(n, p, mat))) < 1e-12 * float(n * p)
    n, p = physics.densities(-160.0, 3.0, -3.0, mat)
    assert float(physics.recombination(n, p, mat)) > 0


def test_continuity_sum_is_total_current_divergence() -> None:
    """r08 Sec. 3, Eq. (7): R_n + R_p = R_J = delta d/dx (J_n + J_p), the source terms cancel."""
    mat = {"Nc": 0.08, "Nv": 1.8, "Chi": 150.9, "Eg": 58.0, "Et": 0.0, "Br": 0.0,
           "tn": 180.0, "tp": 180.0, "mn": 100.0, "mp": 100.0, "eps": 9.4, "Ndop": 0.01}

    def fields(x):
        return (-155.0 + 3.0 * jnp.sin(2.0 * x), -2.0 * x**2 + 0.3 * x, 1.5 * jnp.cos(x))

    for x in (0.1, 0.45, 0.9):
        _, rn, rp = physics.residuals(fields, x, mat, 1.9e-4, 3e-11)
        rj = physics.total_current_divergence(fields, x, mat, 1.9e-4)
        assert abs(float(rn + rp - rj)) <= 1e-9 * (abs(float(rn)) + abs(float(rp)) + 1e-300)


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_sg_currents_match_dataset(fs) -> None:
    """r08 Sec. 7: the stored Jn, Jp are dPV's Scharfetter-Gummel currents of the stored potentials."""
    for dev in sample(fs):
        jn, jp = physics.sg_currents(dev.x, jnp.asarray(dev.pots), jnodes(dev))
        for k in range(len(dev.V)):
            scale = np.max(np.abs(dev.Jn[k])) + np.max(np.abs(dev.Jp[k]))
            assert np.max(np.abs(np.asarray(jn)[k] - dev.Jn[k])) < 1e-6 * scale
            assert np.max(np.abs(np.asarray(jp)[k] - dev.Jp[k])) < 1e-6 * scale


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_poisson_residual_small_on_dpv_profiles(fs) -> None:
    """r08 Sec. 3, Eq. (4)-(5): with these densities dPV's solutions satisfy discrete Poisson.

    dPV stops Newton when its update is below 1e-6, so the residual it leaves is
    bounded by about 1e-6 of the largest term; a wrong sign or scale in Eq. (4)
    would leave a residual of the order of the charge.
    """
    for dev in sample(fs):
        nd = nodes_of(dev)
        zero = np.zeros_like(dev.phi_eq)
        r, scale = discrete_poisson(dev, nd, dev.phi_eq, zero, zero)
        assert np.max(np.abs(r)) < 1e-6 * np.max(scale)
        for k in range(len(dev.V)):
            r, scale = discrete_poisson(dev, nd, *dev.pots[k])
            assert np.max(np.abs(r)) < 1e-6 * np.max(scale)


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_discrete_continuity_small_on_dpv_sweep(fs) -> None:
    """r08 Sec. 3-4, Eq. (7), (8), (10): G and R balance dPV's discrete current divergence.

    Per node the residual relative to its own terms has a median below 1e-4 at
    every voltage (measured up to 1.4e-6 in E1): dPV's Newton stops on an update
    norm of 1e-6, so nodes keep residuals of that relative order, while a wrong G
    or R would put the median at order one.
    """
    for dev in sample(fs):
        nd = nodes_of(dev)
        dg = np.diff(dev.x)
        ad = (dg[:-1] + dg[1:]) / 2
        for k in range(len(dev.V)):
            n, p = physics.densities(*dev.pots[k], nd)
            r = np.asarray(physics.recombination(n, p, nd))[1:-1]
            g = dev.G[1:-1]
            for djdx, sign in ((np.diff(dev.Jn[k]) / ad, 1.0), (np.diff(dev.Jp[k]) / ad, -1.0)):
                res = djdx + sign * (g - r)
                terms = np.abs(djdx) + np.abs(g) + np.abs(r)
                assert np.median(np.abs(res) / terms) < 1e-4


# --- optics -----------------------------------------------------------------------
@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_generation_matches_dpv(fs) -> None:
    """r08 Sec. 4, Eq. (9)-(10): G recomputed here equals dPV's stored G, LDS of E4 included."""
    for dev in sample(fs):
        lam, P = optics.spectrum(dev.family, dev.xi)
        g = np.asarray(optics.generation_on_grid(dev.x, nodes_of(dev), lam, P))
        assert np.max(np.abs(g - dev.G)) < 1e-12 * np.max(np.abs(dev.G))


def test_lds_photon_balance() -> None:
    """r08 Sec. 4, Eq. (9): photons out = photons in - (1 - eta_geom eta_PL) * absorbed."""
    lam, P = optics.standard_spectrum()
    window = (lam >= 300.0) & (lam <= 400.0)
    absorbed = 0.95 * np.sum((P * lam)[window])
    for eta in (0.0, 0.5, 0.9):
        lam2, P2 = optics.lds_spectrum(eta)
        out = np.sum(P2 * lam2)
        expected = np.sum(P * lam) - (1 - optics.LDS_ETA_GEOM * eta) * absorbed
        assert abs(out / expected - 1) < 1e-12
    assert np.sum(optics.lds_spectrum(0.9)[1]) != np.sum(P)


def test_lds_identity_outside_window() -> None:
    """r08 Sec. 4, Eq. (9): bands outside 300-400 nm other than k_em are untouched."""
    lam, P = optics.standard_spectrum()
    _, P2 = optics.lds_spectrum(0.7)
    k_em = int(np.argmin(np.abs(lam - optics.LDS_LAMBDA_EM_NM)))
    keep = ~((lam >= 300.0) & (lam <= 400.0))
    keep[k_em] = False
    assert np.max(np.abs(P2[keep] - P[keep])) < 1e-12 * np.max(P)
    assert P2[k_em] > P[k_em]


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_continuous_generation_agrees_with_dpv_grid(fs) -> None:
    """r08 Sec. 4, Eq. (10): the exact optical depth agrees with dPV's left-Riemann G.

    Exact (1e-12) in E1, one material; within 1e-3 of max G with heterointerfaces,
    away from the interface node itself, whose layer dPV assigns by rounding.
    """
    for dev in sample(fs):
        x_hat = scaling.x_hat_from_debye(dev.x)
        g = np.asarray(optics.generation_at(x_hat, dev.family, dev.xi))
        away = np.ones_like(x_hat, dtype=bool)
        for m in data.interfaces_hat(dev.family, dev.xi):
            away &= np.abs(x_hat - m) > 1e-9
        tol = 1e-12 if dev.family == "E1" else 1e-3
        assert np.max(np.abs(g - dev.G)[away]) < tol * np.max(np.abs(dev.G))


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_standard_spectrum_power(fs) -> None:
    """r08 Sec. 4-5, Eq. (14): denominator = standard spectrum power (~899.9 W/m^2), also in E4."""
    for dev in sample(fs):
        assert abs(optics.standard_power() - dev.attrs["P_in_std"]) < 1e-9
        _, P = optics.spectrum(dev.family, dev.xi)
        assert abs(np.sum(P) - dev.attrs["P_in_luz"]) < 1e-9


# --- contacts ---------------------------------------------------------------------
@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_pce_matches_dpv_dataset(fs) -> None:
    """r08 Sec. 5, Eq. (14): calcPmax reimplemented here reproduces the stored PCE, Voc, Jsc, FF."""
    for dev in sample(fs):
        s = contacts.jv_summary(dev.V, dev.J)
        assert abs(s["pce"] - dev.pce) < 1e-12
        assert abs(s["pmax"] - dev.attrs["pmax"]) < 1e-9
        assert abs(s["vmax"] - dev.attrs["vmax"]) < 1e-12
        assert abs(s["jsc"] - dev.attrs["jsc"]) < 1e-9
        assert abs(s["voc"] - dev.attrs["voc"]) < 1e-12
        assert abs(s["ff"] - dev.attrs["ff"]) < 1e-12


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_terminal_current_matches_dpv(fs) -> None:
    """r08 Sec. 5, Eq. (13): the stored J is J_0 times the mean of Jn + Jp, as current.total_current."""
    for dev in sample(fs):
        j = np.asarray(contacts.terminal_current(dev.Jn, dev.Jp))
        assert np.max(np.abs(j - dev.J)) < 1e-12 * np.max(np.abs(dev.J))


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_contact_potentials_match_dpv(fs) -> None:
    """r08 Sec. 5, Eq. (11): phi_eq(0), phi_eq(1) close dPV's equilibrium, and phi(1) = phi_eq(1) + V~."""
    for dev in sample(fs):
        nd = nodes_of(dev)
        phi0 = float(contacts.contact_potential({k: v[0] for k, v in nd.items()}))
        phi1 = float(contacts.contact_potential({k: v[-1] for k, v in nd.items()}))
        assert abs(phi0 - dev.phi_eq[0]) < 1e-10 and abs(phi1 - dev.phi_eq[-1]) < 1e-10
        assert np.max(np.abs(dev.pots[:, 0, 0] - phi0)) < 1e-10
        assert np.max(np.abs(dev.pots[:, 0, -1] - (phi1 + dev.V / scaling.ENERGY))) < 1e-10


def test_contact_residuals_vanish_on_equilibrium_densities() -> None:
    """r08 Sec. 5, Eq. (12): at equilibrium (flat phi_n = phi_p = 0, n = n_eq) every B vanishes,
    and a carrier excess at a contact with S > 0 makes its B nonzero."""
    mat = {"Nc": 0.08, "Nv": 1.8, "Chi": 150.9, "Eg": 58.0, "mn": 100.0, "mp": 100.0, "Ndop": 0.01}
    mat1 = dict(mat, Ndop=-0.01)
    phi0 = contacts.contact_potential(mat)
    phi1 = contacts.contact_potential(mat1)
    s = (3.0, 2.0, 1.0, 4.0)

    def eq(x):
        return (phi0 + (phi1 - phi0) * x, 0.0 * x, 0.0 * x)

    b = contacts.contact_residuals(eq, mat, mat1, s, 1e-3)
    assert all(abs(float(v)) < 1e-12 for v in b)

    def excess(x):
        return (phi0 + (phi1 - phi0) * x, 0.1 + 0.0 * x, 0.0 * x)

    b = contacts.contact_residuals(excess, mat, mat1, s, 1e-3)
    assert abs(float(b[0])) > 1e-6


def test_calcpmax_exact_on_quadratic_power() -> None:
    """r08 Sec. 5, Eq. (14): the spline is exact when V J is linear in V, and its vertex is found
    when the maximum falls inside a piece (a = 0 on the first piece, dPV's closure)."""
    v = np.linspace(0.0, 1.0, 21)
    j = 2.0 - 0.0 * v  # P = 2 V: linear, maximum at the last node
    pmax, vmax = contacts.calc_pmax(v, j)
    assert abs(pmax - 2.0) < 1e-12 and abs(vmax - 1.0) < 1e-12
    j = 1.0 - v  # P = V - V^2: maximum 0.25 at V = 0.5, a node
    pmax, vmax = contacts.calc_pmax(v, j)
    assert abs(pmax - 0.25) < 1e-3 and abs(vmax - 0.5) < 0.05


@needs_dataset
@pytest.mark.parametrize("fs", FAMILY_STAGES)
def test_sweep_stops_like_dpv(fs) -> None:
    """r08 Sec. 5: V_j = j 0.05 V from 0, stopped at the first sign change, as dPV's simulate."""
    for dev in sample(fs):
        assert contacts.sweep_length(dev.J) == len(dev.V)
        assert np.max(np.abs(contacts.voltages(len(dev.V)) - dev.V)) < 1e-12
        extended = np.concatenate([dev.J, [-1.0, -2.0]])
        assert contacts.sweep_length(extended) == len(dev.V)
