"""Level 3 - synthetic data. Deterministic, fixed seed, ground truth by construction.

State each expectation in the docstring BEFORE the assertion, so a passing test
cannot be mistaken for a hypothesis fitted after seeing the output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from PINN_DPV import contacts, physics, scaling

SEED = 0


def test_ideal_diode_pce_from_spline() -> None:
    """Expectation: for an ideal-diode curve J = Jsc - J0 (exp(V/Vt) - 1) sampled every 0.05 V
    up to its sign change, dPV's spline maximum lies within 1 % of the dense-grid maximum of V J,
    never below the best sampled node, and V_oc matches the analytic one within one step."""
    rng = np.random.default_rng(SEED)
    for _ in range(5):
        jsc = rng.uniform(1e-2, 3e-2)  # A/cm^2
        j0 = 10 ** rng.uniform(-18, -14)
        vt = 0.025852
        voc = vt * np.log(jsc / j0 + 1)
        k = int(np.ceil(voc / 0.05)) + 1
        v = contacts.voltages(k)
        j = jsc - j0 * (np.exp(v / vt) - 1)
        s = contacts.jv_summary(v, j)
        dense = np.linspace(0, voc, 200001)
        pd = np.max(dense * (jsc - j0 * (np.exp(dense / vt) - 1))) * 1e4
        assert abs(s["pmax"] / pd - 1) < 1e-2
        assert s["pmax"] >= np.max(v * j * 1e4) - 1e-12
        assert abs(s["voc"] - voc) < 0.05


def test_manufactured_poisson_residual() -> None:
    """Expectation: for phi = a sin(b x) with phi_n = phi_p = 0, the Poisson residual of Eq. (5)
    equals delta^2 eps (-a b^2 sin(b x)) - (n - p - Ndop) computed by hand."""
    mat = {"Nc": 0.08, "Nv": 1.8, "Chi": 150.9, "Eg": 58.0, "eps": 9.4, "Ndop": 0.01,
           "Et": 0.0, "Br": 0.0, "tn": 1.0, "tp": 1.0, "mn": 1.0, "mp": 1.0}
    a, b, delta = -155.0, 1.3, 2e-3
    rng = np.random.default_rng(SEED)
    for x in rng.uniform(0.05, 0.95, size=4):
        def fields(s):
            return (a * jnp.sin(b * s), 0.0 * s, 0.0 * s)
        phi = a * np.sin(b * x)
        n, p = physics.densities(phi, 0.0, 0.0, mat)
        expected = delta**2 * mat["eps"] * (-a * b**2 * np.sin(b * x)) - (n - p - mat["Ndop"])
        got = jnp.sum(physics.poisson_terms(fields, x, mat, delta))
        assert abs(float(got - expected)) < 1e-10 * (abs(float(n)) + abs(float(p)) + 0.01)


def test_hard_constraint_holds_under_autodiff() -> None:
    """Expectation: whatever the network output u_phi(x), phi from Eq. (3) takes the contact values
    exactly and its x-derivative at the contacts depends on u_phi only through 4 D_phi u_phi."""
    w = np.random.default_rng(SEED).normal(size=3)

    def u_phi(x):
        return jnp.tanh(w[0] * x + w[1]) * w[2]

    phi = lambda x: scaling.phi_from_output(x, u_phi(x), -40.0, -55.0, 3.0, 8.0)  # noqa: E731
    assert abs(float(phi(0.0)) + 40.0) < 1e-12
    assert abs(float(phi(1.0)) + 52.0) < 1e-12
    dphi0 = float(jax.grad(phi)(0.0))
    assert abs(dphi0 - ((-55.0 + 3.0 + 40.0) + 4.0 * 8.0 * float(u_phi(0.0)))) < 1e-10
