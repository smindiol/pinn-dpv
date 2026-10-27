"""Evaluation against dPV: the derived PCE, diagnostic metrics, the reference floor and statistics.

Algorithm A3 (Sec. 8): J_hat(V_j) by Eq. (13) on dPV's own spatial grid and voltage
grid, the curve cut at the first sign change, PCE by Eq. (14) with the same
calcPmax and denominator, and V_oc, J_sc, FF from that curve.
Eq. (20): e_PCE = |eta - eta_ref| in percentage points; e_field,k the RMS error of
each normalized output over dPV's nodes; e_J = max_g |J_n + J_p - J_bar| / |J_bar|.
Eq. (21): e_floor = |eta_ref(N_g) - eta_ref(4 N_g)| and r = Q_0.95 of it.
Sec. 9: IQM of e_PCE over seeds with a stratified percentile bootstrap, paired
Wilcoxon signed-rank against the reference configuration with Holm correction,
Friedman ranking, and the Bayesian signed-rank test (Dirichlet process prior,
s = 0.5, z_0 = 0) with the region of practical equivalence [-r, r].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from PINN_DPV import contacts, loss, physics, scaling

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["8", "9"],
    "equations": ["13", "14", "20", "21"],
    "invariants": [
        "online_evaluation_matches_offline",
        "e_pce_zero_for_reference_curve",
        "iqm_trims_quartiles",
        "floor_is_q95_of_refinement_gap",
        "bayesian_signed_rank_probabilities_sum_to_one",
    ],
}


# --- Network quantities on dPV's grid -------------------------------------------------
def network_currents(params, prob, xi_hat, x_hat_nodes, v_tilde):
    """J~_n + J~_p of Eq. (6) from the network at each node, and the three potentials there."""
    fs = prob.fs
    dev = loss.device(fs, jnp.asarray(xi_hat))
    f = loss.net_fields(params, prob, jnp.asarray(xi_hat), dev, v_tilde)

    def at(x):
        layer = fs.layer_of(x, dev["x_layers"])
        mat = fs.material(layer, dev["ndop"])
        jn, jp = physics.currents(f, x, mat, dev["delta"])
        return jn + jp, jnp.stack(f(x))

    return jax.vmap(at)(jnp.asarray(x_hat_nodes))


def voltage_record(params, prob, dev_ref, j: int):
    """What A3 needs from one device at voltage index j: J (A/cm^2), e_J and field errors."""
    x_hat = scaling.x_hat_from_debye(dev_ref.x)
    v_tilde = contacts.DV * j / scaling.ENERGY
    jt, fields = network_currents(params, prob, dev_ref.u, x_hat, v_tilde)
    jt = np.asarray(jt)
    jbar = float(np.mean(jt))
    rec = {"J": float(contacts.terminal_current(jt, 0.0 * jt)),
           "e_J": float(np.max(np.abs(jt - jbar)) / abs(jbar)) if jbar != 0 else float("inf")}
    if dev_ref.pots is not None and j < len(dev_ref.V):
        rec.update(field_errors(np.asarray(fields).T, dev_ref.pots[j], prob.scales))
    return rec


def field_errors(pred, ref, scales):
    """e_field,k of Eq. (20): RMS over nodes of each output, normalized as in Eq. (3)."""
    span = np.asarray(scales.y_max) - np.asarray(scales.y_min)
    return {"e_phi": float(np.sqrt(np.mean(((pred[0] - ref[0]) / scales.d_phi) ** 2))),
            "e_phi_n": float(np.sqrt(np.mean(((pred[1] - ref[1]) / span[0]) ** 2))),
            "e_phi_p": float(np.sqrt(np.mean(((pred[2] - ref[2]) / span[1]) ** 2)))}


def curve_metrics(currents, dev_ref):
    """PCE and the J-V summary of a predicted curve, and its errors against dPV (Eq. 14, 20)."""
    v = contacts.voltages(len(currents))
    s = contacts.jv_summary(v, np.asarray(currents))
    ref = dev_ref.attrs
    out = dict(s)
    out["e_pce_pp"] = 100.0 * abs(s["pce"] - float(ref["eff"]))
    out["d_voc"] = abs(s["voc"] - float(ref["voc"]))
    out["d_jsc"] = abs(s["jsc"] - float(ref["jsc"]))
    out["d_ff"] = abs(s["ff"] - float(ref["ff"]))
    return out


def evaluate_offline(params_per_voltage, prob, dev_ref):
    """A3 from stored per-voltage weights: the reference against which online evaluation is held."""
    recs = [voltage_record(p, prob, dev_ref, j) for j, p in enumerate(params_per_voltage)]
    return recs, curve_metrics([r["J"] for r in recs], dev_ref)


# --- Eq. (21) -------------------------------------------------------------------------------
def floor(eff_base, eff_refined):
    """e_floor per device in percentage points and r = Q_0.95 (Eq. 21)."""
    e = 100.0 * np.abs(np.asarray(eff_base) - np.asarray(eff_refined))
    return e, float(np.quantile(e, 0.95))


# --- Sec. 9 statistics -------------------------------------------------------------------------
def iqm(x):
    """Interquartile mean: the mean of the middle 50 % (scipy's 25 % trimmed mean)."""
    return float(stats.trim_mean(np.asarray(x, dtype=np.float64), 0.25))


def bootstrap_iqm_ci(runs_by_stratum, n_boot=2000, seed=0, level=0.95):
    """Stratified percentile bootstrap of the IQM over runs, resampled within each stratum."""
    rng = np.random.default_rng(seed)
    strata = [np.asarray(v, dtype=np.float64) for v in runs_by_stratum.values()]
    pooled = np.concatenate(strata)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = iqm(np.concatenate([s[rng.integers(0, s.size, s.size)] for s in strata]))
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return iqm(pooled), float(lo), float(hi)


def wilcoxon_holm(reference, others: dict):
    """Paired Wilcoxon signed-rank of each configuration against the reference, Holm-corrected."""
    names = list(others)
    p = np.array([stats.wilcoxon(np.asarray(others[n]), np.asarray(reference)).pvalue for n in names])
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return {n: {"p": float(p[i]), "p_holm": float(adj[i])} for i, n in enumerate(names)}


def friedman(scores):
    """Friedman test over configurations (columns) and datasets (rows), with mean ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    stat = stats.friedmanchisquare(*scores.T)
    ranks = np.mean(stats.rankdata(scores, axis=1), axis=0)
    return {"statistic": float(stat.statistic), "p": float(stat.pvalue), "mean_ranks": ranks.tolist()}


def bayesian_signed_rank(x, y, rope, s=0.5, z0=0.0, n_samples=20000, seed=0):
    """Benavoli et al. 2017 Bayesian signed-rank test: P(x < y), P(rope), P(x > y).

    Differences z = y - x (positive when x has the lower error, i.e. is better);
    Dirichlet process prior of strength s with pseudo-observation z0.
    """
    rng = np.random.default_rng(seed)
    z = np.concatenate([[z0], np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)])
    alpha = np.concatenate([[s], np.ones(z.size - 1)])
    w = rng.dirichlet(alpha, size=n_samples)  # (S, n+1)
    pair = z[:, None] + z[None, :]
    right = (pair > 2 * rope).astype(np.float64)
    left = (pair < -2 * rope).astype(np.float64)
    p_right = np.einsum("si,ij,sj->s", w, right, w)
    p_left = np.einsum("si,ij,sj->s", w, left, w)
    p_rope = 1.0 - p_right - p_left
    winner = np.argmax(np.stack([p_left, p_rope, p_right]), axis=0)
    return {"x_worse": float(np.mean(winner == 0)), "equivalent": float(np.mean(winner == 1)),
            "x_better": float(np.mean(winner == 2))}


__all__ = ["voltage_record", "curve_metrics", "evaluate_offline", "floor", "iqm",
           "bootstrap_iqm_ci", "wilcoxon_holm", "friedman", "bayesian_signed_rank"]
