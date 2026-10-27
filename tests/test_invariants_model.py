"""Level 2 - invariants of the network, the loss, the weighting, training, persistence and evaluation.

Same contract as test_invariants.py: one `test_<id>` per invariant declared in a
module's `__provenance__`, each docstring citing the passage it enforces.
Networks here are tiny (width 8-16) so the suite runs on a laptop CPU.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from PINN_DPV import (b0, data, evaluation, interfaces, loss, network, optics, persistence, physics, scaling,
                      search, steps, training, weighting)

FAMILY_STAGES = ["E1A", "E3A"]
needs_dataset = pytest.mark.skipif(
    not all(os.path.exists(p) for fs in FAMILY_STAGES for p in data.canonical_paths(fs)),
    reason="dPV reference files not found under data.dataset_root()")

MAT = {"Nc": 0.08, "Nv": 1.8, "Chi": 150.9, "Eg": 58.0, "eps": 9.4, "Ndop": 0.01, "Et": 0.0,
       "Br": 0.0, "tn": 180.0, "tp": 180.0, "mn": 100.0, "mp": 100.0}


def tiny_cfg(d_xi=2, n_int=1, **kw):
    return network.NetConfig(d_xi=d_xi, n_int=n_int, width=kw.pop("width", 12), depth=kw.pop("depth", 2), **kw)


# --- network ----------------------------------------------------------------------------------
def test_network_continuous_at_interfaces() -> None:
    """r09 Sec. 8, Eq. (24): u_hat is continuous at x_hat_m because rho_m enters through |x - x_m|."""
    cfg = tiny_cfg(n_int=2)
    p = network.init(jax.random.PRNGKey(0), cfg)
    xi, xm = jnp.asarray([0.3, 0.6]), jnp.asarray([0.25, 0.7])
    for m in range(2):
        left = network.u_hat(p, cfg, xm[m] - 1e-9, xi, xm)
        right = network.u_hat(p, cfg, xm[m] + 1e-9, xi, xm)
        assert float(jnp.max(jnp.abs(left - right))) < 1e-6


def test_cusp_trace_jump_is_twice_drho() -> None:
    """r09 Eq. (24) and E0 protocol: [du/dx] at x_m equals 2 dU/drho_m, by AD and by lateral
    finite differences within 1e-6 relative."""
    cfg = tiny_cfg(n_int=1)
    p = network.init(jax.random.PRNGKey(3), cfg)
    xi, xm = jnp.asarray([0.4, 0.2]), jnp.asarray([0.37])
    f_right = lambda x: network.u_hat(p, cfg, x, xi, xm, side=jnp.asarray([1.0]))  # noqa: E731
    f_left = lambda x: network.u_hat(p, cfg, x, xi, xm, side=jnp.asarray([-1.0]))  # noqa: E731
    jump_ad = jax.jacfwd(f_right)(xm[0]) - jax.jacfwd(f_left)(xm[0])
    drho = jax.jacfwd(lambda r: network.raw(p, cfg, xm[0], jnp.asarray([r]), xi))(0.0)
    assert float(jnp.max(jnp.abs(jump_ad - 2 * drho))) < 1e-12
    h = 1e-6
    g = lambda x: network.u_hat(p, cfg, x, xi, xm)  # noqa: E731
    fd_right = (-3 * g(xm[0]) + 4 * g(xm[0] + h) - g(xm[0] + 2 * h)) / (2 * h)
    fd_left = (3 * g(xm[0]) - 4 * g(xm[0] - h) + g(xm[0] - 2 * h)) / (2 * h)
    jump_fd = fd_right - fd_left
    assert float(jnp.max(jnp.abs(jump_fd - jump_ad) / jnp.abs(jump_ad))) < 1e-6


def test_plain_network_when_no_interface() -> None:
    """r09 Sec. 8: in E1 (N_int = 0) the network is the perceptron on (x_hat, xi_hat) unchanged."""
    cfg = tiny_cfg(n_int=0)
    p = network.init(jax.random.PRNGKey(1), cfg)
    assert p["all"][0]["W"].shape[0] == 1 + 2
    x, xi = 0.3, jnp.asarray([0.1, 0.9])
    z = jnp.concatenate([jnp.atleast_1d(x), xi])
    manual = z
    for layer in p["all"][:-1]:
        manual = jnp.tanh(manual @ layer["W"] + layer["b"])
    manual = manual @ p["all"][-1]["W"] + p["all"][-1]["b"]
    assert float(jnp.max(jnp.abs(network.u_hat(p, cfg, x, xi, jnp.zeros(0)) - manual))) < 1e-14


def test_kaiming_initialization_scale() -> None:
    """r09 Sec. 8: Kaiming initialization, weight std sqrt(2 / fan_in), zero biases."""
    cfg = network.NetConfig(d_xi=2, n_int=0, width=512, depth=2)
    p = network.init(jax.random.PRNGKey(0), cfg)
    w = np.asarray(p["all"][1]["W"])
    assert abs(w.std() / np.sqrt(2.0 / 512) - 1) < 0.02
    assert all(float(jnp.max(jnp.abs(layer["b"]))) == 0.0 for layer in p["all"])


def test_potentials_meet_dirichlet_for_any_weights() -> None:
    """r09 Eq. (3), (11): phi(0) = phi_eq(0), phi(1) = phi_eq(1) + V~ whatever the weights."""
    cfg = tiny_cfg(n_int=1)
    sc = network.OutputScales(30.0, (-5.0, -5.0), (5.0, 5.0))
    for seed in range(3):
        p = network.init(jax.random.PRNGKey(seed), cfg)
        args = (jnp.asarray([0.5, 0.5]), jnp.asarray([0.4]), -150.0, -205.0, 30.0)
        assert abs(float(network.potentials(p, cfg, sc, 0.0, *args)[0]) + 150.0) < 1e-10
        assert abs(float(network.potentials(p, cfg, sc, 1.0, *args)[0]) + 175.0) < 1e-10


# --- interfaces --------------------------------------------------------------------------------
def _smooth_side_fields(side):
    return lambda x: (-150.0 + 2.0 * x, 0.1 * x, -0.2 * x)


def test_jumps_vanish_for_smooth_flux_continuous_fields() -> None:
    """r09 Eq. (22): same material on both sides and smooth fields give zero flux jumps."""
    j = interfaces.flux_jumps(_smooth_side_fields, 0, jnp.asarray([0.5]), MAT, MAT, 1e-3)
    assert float(jnp.max(jnp.abs(j))) < 1e-14


def test_jump_detects_displacement_mismatch() -> None:
    """r09 Eq. (22): with eps jumping and dphi/dx continuous, [eps dphi/dx] = (eps_r - eps_l) phi'."""
    right = dict(MAT, eps=12.0)
    j = interfaces.flux_jumps(_smooth_side_fields, 0, jnp.asarray([0.5]), MAT, right, 1e-3)
    assert abs(float(j[0]) - (12.0 - 9.4) * 2.0) < 1e-12


# --- loss --------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def problem(fs_name="E3A"):
    fs = data.FamilyStage.load(fs_name)
    cfg = network.NetConfig(d_xi=fs.d, n_int=fs.n_int, width=8, depth=2)
    sc = network.OutputScales(40.0, (-3.0, -3.0), (3.0, 3.0))
    return loss.Problem(fs=fs, cfg=cfg, scales=sc, sigma2={})


def small_batch(d, n=6, with_data=False):
    rng = np.random.default_rng(0)
    b = {"xq": jnp.asarray(rng.uniform(0.01, 0.99, n)), "xiq": jnp.asarray(rng.uniform(0, 1, (n, d))),
         "xib": jnp.asarray(rng.uniform(0, 1, (2, d)))}
    if with_data:
        b.update({"xid": jnp.asarray(rng.uniform(0, 1, (2, d))),
                  "xs": jnp.asarray(rng.uniform(0, 1, (2, 3))), "ref": jnp.zeros((2, 3, 3))})
    return b


@needs_dataset
def test_loss_is_weighted_sum_of_components() -> None:
    """r09 Eq. (15): L = sum_i lambda_i log(1 + L_i / sigma_i^2); doubling one lambda adds exactly
    its compressed share."""
    prob = problem()
    p = network.init(jax.random.PRNGKey(0), prob.cfg)
    comps = loss.components(p, prob, small_batch(prob.fs.d), 0.5)
    lam = {k: 1.0 for k in comps}
    sig = {k: 2.0 + i for i, k in enumerate(comps)}
    base = loss.total(comps, lam, sig)
    lam2 = dict(lam, Poisson=2.0)
    assert abs(float(loss.total(comps, lam2, sig) - base - jnp.log1p(comps["Poisson"] / sig["Poisson"]))) \
        <= 1e-12 * abs(float(base))
    assert set(comps) == set(loss.active_components(prob.fs.n_int))


@needs_dataset
def test_residual_equals_sum_of_terms() -> None:
    """r09 Eq. (23): every residual is the sum of its terms T_iq, the same terms sigma_i is built on."""
    prob = problem()
    fs = prob.fs
    p = network.init(jax.random.PRNGKey(2), prob.cfg)
    xi = jnp.asarray([0.3, 0.7])
    dev = loss.device(fs, xi)
    f = loss.net_fields(p, prob, xi, dev, 1.0)
    x = jnp.asarray(0.5)
    tp, tn, tpp = loss.bulk_terms(f, x, fs, dev)
    layer = fs.layer_of(x, dev["x_layers"])
    mat = fs.material(layer, dev["ndop"])
    g = optics.generation_jax(fs, x, dev)
    rp, rn, rpp = physics.residuals(f, x, mat, dev["delta"], g)
    assert abs(float(jnp.sum(tp) - rp)) <= 1e-12 * float(jnp.sum(jnp.abs(tp)))
    assert abs(float(jnp.sum(tn) - rn)) <= 1e-12 * float(jnp.sum(jnp.abs(tn)))


@needs_dataset
def test_spline_fields_interpolate_nodes() -> None:
    """r09 Eq. (23): the per-layer cubic interpolant reproduces dPV's potentials at its nodes,
    so the floor measures only the operator applied between nodes, not a misplaced reference."""
    fs = data.FamilyStage.load("E3A")
    dev = data.load("E3A", data.device_keys("E3A")[:1])[0]
    x_hat = scaling.x_hat_from_debye(dev.x)
    lay = loss.device(fs, jnp.asarray(dev.u))
    li = np.sum(x_hat[:, None] >= np.asarray(lay["x_layers"])[None, :], axis=1)
    bps, coef = loss.layer_splines(x_hat, dev.pots[3], li, fs.n_layers)
    vals = np.stack([np.asarray(jax.vmap(lambda x, l: jnp.stack(loss.spline_fields(bps, coef, l)(x)))(
        jnp.asarray(x_hat), jnp.asarray(li)))]).squeeze(0).T
    assert np.max(np.abs(vals - dev.pots[3])) < 1e-9 * np.max(np.abs(dev.pots[3]))


@needs_dataset
def test_sigma_bounds_residual() -> None:
    """r09 Eq. (23): |R_i| <= sum_q |T_iq| pointwise, so L_i^floor <= sigma_i^2 in every component."""
    fs = data.FamilyStage.load("E3A")
    devs = data.load("E3A", data.device_keys("E3A")[:2])
    s2, fl = loss.sigma_and_floor(fs, devs)
    assert set(s2) >= {"Poisson", "n", "p", "BCn", "BCp", "int_phi", "int_n", "int_p", "Poisson@eq"}
    for k in s2:
        assert fl[k] <= s2[k] * (1 + 1e-12)
        assert s2[k] > 0


@needs_dataset
def test_data_term_zero_on_reference() -> None:
    """r09 Eq. (15): L_dat vanishes when the reference equals the network's own potentials."""
    prob = problem()
    p = network.init(jax.random.PRNGKey(5), prob.cfg)
    b = small_batch(prob.fs.d, with_data=True)
    refs = []
    for xi, xs in zip(b["xid"], b["xs"]):
        dev = loss.device(prob.fs, xi)
        f = loss.net_fields(p, prob, xi, dev, 0.7)
        refs.append(np.stack([np.asarray(f(x)) for x in np.asarray(xs)]).T)
    b["ref"] = jnp.asarray(np.stack(refs))
    assert float(loss.data_term(p, prob, b, 0.7)) < 1e-24
    b["ref"] = b["ref"] + 0.5
    assert float(loss.data_term(p, prob, b, 0.7)) > 1e-4


# --- weighting -----------------------------------------------------------------------------------
def _quad_problem():
    params = {"w": jnp.asarray([1.0, -2.0, 0.5])}
    a = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 100.0]])

    def comps(p):
        return (a @ p["w"]) ** 2 + 1e-3
    return params, comps


def test_ntk_equalizes_gradient_norms() -> None:
    """r09 Eq. (16): after lambda_i = K_bar / K_i, every lambda_i K_i equals K_bar."""
    params, comps = _quad_problem()
    lam = weighting.ntk_weights(comps, params)
    k = jax.jacrev(lambda p: jnp.sqrt(comps(p)))(params)["w"]
    ki = jnp.sum(k**2, axis=1)
    prod = lam * ki
    assert float(jnp.max(jnp.abs(prod - jnp.mean(ki)))) < 1e-9 * float(jnp.mean(ki))


def test_ntk_invariant_to_sigma() -> None:
    """r09 Sec. 6 and E0 protocol: NTK's descent direction does not change when each component is
    divided by sigma_i^2 (cosine > 1 - 1e-10), while fixed weights' direction does."""
    params, comps = _quad_problem()
    sig = jnp.asarray([1e-3, 1.0, 1e4])

    def direction(fn):
        lam = weighting.ntk_weights(fn, params)
        return jax.grad(lambda p: jnp.sum(lam * fn(p)))(params)["w"]

    def cos(a, b):
        return float(jnp.dot(a, b) / jnp.linalg.norm(a) / jnp.linalg.norm(b))

    assert cos(direction(comps), direction(lambda p: comps(p) / sig)) > 1 - 1e-10
    fixed = lambda fn: jax.grad(lambda p: jnp.sum(fn(p)))(params)["w"]  # noqa: E731
    assert cos(fixed(comps), fixed(lambda p: comps(p) / sig)) < 1 - 1e-3


def test_brdr_weights_mean_one() -> None:
    """r09 Eq. (17): BRDR's lambda is renormalized to mean one at every step."""
    st = weighting.init_state(["a", "b", "c"], "brdr")
    rng = np.random.default_rng(0)
    for _ in range(50):
        st = weighting.brdr_update(st, jnp.asarray(rng.uniform(1e-3, 1.0, 3)))
        assert abs(float(jnp.mean(st["lam"])) - 1.0) < 1e-10


def test_ema_weights_track_loss_shares() -> None:
    """r09 Eq. (18): under constant components the EMA weights converge to s_i / sum_l s_l."""
    st = weighting.init_state(["a", "b"], "ema")
    s = jnp.asarray([3.0, 1.0])
    for _ in range(40000):
        st = weighting.ema_update(st, s)
    assert float(jnp.max(jnp.abs(st["lam"] - s / jnp.sum(s)))) < 1e-3


def test_gummel_blocks_alternate() -> None:
    """r09 Sec. 6, strategy 6: blocks alternate between the phi network and the (phi_n, phi_p) one,
    each with its own components."""
    phases = [weighting.gummel_phase(i, 5) for i in range(20)]
    assert phases[:5] == ["phi"] * 5 and phases[5:10] == ["qf"] * 5 and phases[10] == "phi"
    names = ("Poisson", "n", "p", "BCn", "BCp", "int_phi", "int_n", "int_p")
    assert list(np.asarray(weighting.gummel_mask(names, "phi"))) == [1, 0, 0, 0, 0, 1, 0, 0]
    assert list(np.asarray(weighting.gummel_mask(names, "qf"))) == [0, 1, 1, 1, 1, 0, 1, 1]


# --- persistence ---------------------------------------------------------------------------------
def test_checkpoint_roundtrip_exact(tmp_path) -> None:
    """Persistence (AGREED): a checkpoint restores parameters, optimizer state and counters exactly."""
    import optax
    p = network.init(jax.random.PRNGKey(0), tiny_cfg())
    st = {"params": p, "opt_state": optax.adam(1e-3).init(p), "j": 4, "it": 120,
          "key": jax.random.PRNGKey(7)}
    persistence.save_checkpoint(str(tmp_path), st)
    back = persistence.load_checkpoint(str(tmp_path))
    for a, b in zip(jax.tree_util.tree_leaves(st), jax.tree_util.tree_leaves(back)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert back["j"] == 4 and back["it"] == 120


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch) -> None:
    """Persistence (AGREED): a write that dies midway leaves the previous checkpoint intact
    and no temporary file behind."""
    persistence.save_checkpoint(str(tmp_path), {"it": 1})

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(persistence.os, "replace", boom)
    with pytest.raises(OSError):
        persistence.save_checkpoint(str(tmp_path), {"it": 2})
    monkeypatch.undo()
    assert persistence.load_checkpoint(str(tmp_path))["it"] == 1
    assert [n for n in os.listdir(tmp_path) if n.startswith(".ckpt-")] == []


def _bare_repo(tmp_path):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    return f"file://{bare}"


def test_sync_writes_results_branch_and_asset(tmp_path) -> None:
    """Persistence (AGREED): light results land on the orphan branch `results` under runs/<id>/,
    two runs accumulate there, and the checkpoint lands as the run's release asset."""
    url = _bare_repo(tmp_path)
    releases = persistence.LocalReleases(str(tmp_path / "releases"))
    for run_id in ("runA", "runB"):
        run_dir = tmp_path / run_id
        persistence.save_checkpoint(str(run_dir), {"it": run_id})
        persistence.write_json(str(run_dir), "summary.json", {"run": run_id})
        persistence.append_jsonl(str(run_dir), "metrics.jsonl", {"j": 0})
        sync = persistence.Syncer(str(run_dir), run_id, url, releases)
        sync({"it": run_id}, voltage_done=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "-b", "results", url, str(clone)], check=True)
    for run_id in ("runA", "runB"):
        with open(clone / "runs" / run_id / "summary.json") as f:
            assert json.load(f) == {"run": run_id}
        assert (tmp_path / "releases" / f"ckpt-{run_id}" / "checkpoint.pkl").exists()
    log = subprocess.run(["git", "log", "--format=%s", "results"], cwd=clone, capture_output=True,
                         text=True, check=True).stdout.split("\n")
    assert "results: runA" in log and "results: runB" in log
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=clone,
                           capture_output=True, text=True, check=True).stdout
    assert "checkpoint.pkl" not in files
    restored = tmp_path / "restored"
    assert persistence.restore(str(restored), "runA", releases=releases) == "release"
    assert persistence.load_checkpoint(str(restored))["it"] == "runA"


def test_drive_mirror_copies_run(tmp_path) -> None:
    """Persistence (AGREED): on Colab the run directory is mirrored to Drive and restorable from it."""
    run_dir, drive = tmp_path / "run", tmp_path / "MyDrive" / "pinn-dpv-runs"
    (tmp_path / "MyDrive").mkdir()
    persistence.save_checkpoint(str(run_dir), {"it": 9})
    assert persistence.mirror(str(run_dir), str(drive), "r1") == str(drive / "r1")  # created on first use
    assert (drive / "r1" / "checkpoint.pkl").exists()
    unmounted = tmp_path / "unmounted" / "pinn-dpv-runs"
    assert persistence.mirror(str(run_dir), str(unmounted), "r1") is None
    assert not unmounted.parent.exists()
    fresh = tmp_path / "fresh"
    assert persistence.restore(str(fresh), "r1", mirror_root=str(drive)) == "mirror"
    assert persistence.load_checkpoint(str(fresh))["it"] == 9


def test_token_never_in_argv(tmp_path, monkeypatch) -> None:
    """Persistence (AGREED): the token reaches git only through the environment, never argv or a file."""
    secret = "ghp_" + "x" * 36
    seen = []
    real = subprocess.run

    def spy(args, *a, **k):
        seen.append(list(args))
        env = k.get("env") or {}
        if env.get("GIT_ASKPASS"):
            with open(env["GIT_ASKPASS"]) as f:
                assert secret not in f.read()
        return real(args, *a, **k)
    monkeypatch.setattr(persistence.subprocess, "run", spy)
    run_dir = tmp_path / "run"
    persistence.write_json(str(run_dir), "summary.json", {"ok": 1})
    persistence.push_results(str(run_dir), "rid", _bare_repo(tmp_path), token=secret)
    assert seen and all(secret not in " ".join(map(str, a)) for a in seen)


# --- training ----------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def tiny_setup(fs_name="E3A", strategy="fixed", k_s=0, seed=1, v_input=False, qf_start="equilibrium",
               start="zero"):
    cfg = training.TrainConfig(family_stage=fs_name, seed=seed, strategy=strategy, width=8, depth=2,
                               n_q=24, adam_iters=6, lbfgs_iters=2, max_devices=2, max_voltages=2,
                               data_batch=2, k_s=k_s, checkpoint_every=3, eval_marks=(3,),
                               v_input=v_input, qf_start=qf_start, grad_share_marks=(1, 4),
                               start=start, pretrain_iters=8, pretrain_nodes=16)
    fs = data.FamilyStage.load(fs_name)
    tr = data.load(fs_name, data.split(data.device_keys(fs_name), seed)["train"][:2])
    s2, _ = loss.sigma_and_floor(fs, tr[:1])
    sc = scaling.output_scales([(scaling.x_hat_from_debye(d.x), d.pots, d.phi_eq[0], d.phi_eq[-1],
                                 d.V / scaling.ENERGY) for d in tr])
    sigma = {"sigma2": s2, "scales": {"D_phi": sc["D_phi"], "Y_min": sc["Y_min"].tolist(),
                                      "Y_max": sc["Y_max"].tolist()}}
    return training.prepare(cfg, sigma)


@needs_dataset
def test_resume_equals_uninterrupted(tmp_path) -> None:
    """Persistence (AGREED): a run stopped mid-voltage, checkpointed and resumed ends with the same
    weights and the same online evaluation as the run that was never interrupted."""
    setup = tiny_setup(strategy="ema", k_s=4)
    full = training.train(setup)
    part = training.train(setup, stop_after=9)
    assert not part["done"]
    persistence.save_checkpoint(str(tmp_path), part)
    resumed = training.train(setup, state=persistence.load_checkpoint(str(tmp_path)))
    for a, b in zip(jax.tree_util.tree_leaves(full["params"]), jax.tree_util.tree_leaves(resumed["params"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert full["eval"].keys() == resumed["eval"].keys()
    for k in full["eval"]:
        assert [r["J"] for r in full["eval"][k]] == [r["J"] for r in resumed["eval"][k]]


def test_collocation_deterministic_per_step() -> None:
    """r09 Sec. 8: Halton points are resampled every iteration, as a function of (seed, voltage, it)."""
    cfg = training.TrainConfig(family_stage="E1A", seed=3, n_q=50)
    a = training.collocation(cfg, 2, 4, 17)
    b = training.collocation(cfg, 2, 4, 17)
    c = training.collocation(cfg, 2, 4, 18)
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
    assert not np.array_equal(np.asarray(a["xq"]), np.asarray(c["xq"]))
    assert a["xq"].shape == (50,) and a["xib"].shape == (5, 2)


@needs_dataset
def test_equilibrium_trains_phi_only() -> None:
    """r09 Algorithm A2 (1): at equilibrium only u_phi is trained; phi_n = phi_p = 0 hard, so the
    network's phi_n and phi_p output weights receive no gradient."""
    setup = tiny_setup()
    names, scaled, objective = training.make_fns(setup, equilibrium=True)
    assert names == ("Poisson", "int_phi")
    p = network.init(jax.random.PRNGKey(0), setup.prob.cfg)
    batch = training.collocation(setup.cfg, setup.fs.d, 0, 0)
    g = jax.grad(lambda q: objective(q, batch, 0.0, jnp.ones(2))[0])(p)
    w_last = g["all"][-1]["W"]
    assert float(jnp.max(jnp.abs(w_last[:, 1:]))) == 0.0
    assert float(jnp.max(jnp.abs(w_last[:, 0]))) > 0.0


@needs_dataset
def test_loss_decreases_on_tiny_problem() -> None:
    """r09 Sec. 8: Adam reduces the equilibrium objective on a fixed batch (the optimizer is wired
    to the loss of Eq. 15, with the sigma scaling of Eq. 23)."""
    import optax
    setup = tiny_setup()
    names, scaled, objective = training.make_fns(setup, equilibrium=True)
    p = network.init(jax.random.PRNGKey(0), setup.prob.cfg)
    batch = training.collocation(setup.cfg, setup.fs.d, 0, 0)
    opt = optax.adam(1e-3)
    s = opt.init(p)
    f = jax.jit(lambda q: objective(q, batch, 0.0, jnp.ones(2))[0])
    first = float(f(p))
    for _ in range(200):
        g = jax.grad(f)(p)
        u, s = opt.update(g, s, p)
        p = optax.apply_updates(p, u)
    assert float(f(p)) < 0.5 * first


# --- evaluation -----------------------------------------------------------------------------------
@needs_dataset
def test_online_evaluation_matches_offline(tmp_path) -> None:
    """AGREED: evaluating J(V_j) at the end of each voltage gives the same curve and PCE as
    evaluating afterwards from the stored weights of every voltage (Algorithm A3)."""
    setup = tiny_setup()
    st = training.train(setup, models_dir=str(tmp_path))
    per_v = [persistence.load_checkpoint(str(tmp_path / f"v{j:03d}"))["params"] for j in range(setup.k_max)] \
        if setup.cfg.keep_all_weights else None
    if per_v is None:  # the cached setup does not keep weights: rerun keeping them
        import dataclasses
        setup2 = training.Setup(dataclasses.replace(setup.cfg, keep_all_weights=True), setup.fs, setup.prob,
                                setup.split, setup.train, setup.evaluate, setup.k_max, setup.sparse)
        st = training.train(setup2, models_dir=str(tmp_path))
        per_v = [persistence.load_checkpoint(str(tmp_path / f"v{j:03d}"))["params"] for j in range(setup.k_max)]
    for d in setup.evaluate:
        recs, metrics = evaluation.evaluate_offline(per_v, setup.prob, d)
        assert [r["J"] for r in recs] == [r["J"] for r in st["eval"][d.key]]
        online = evaluation.curve_metrics([r["J"] for r in st["eval"][d.key]], d)
        assert metrics["pce"] == online["pce"]


@needs_dataset
def test_e_pce_zero_for_reference_curve() -> None:
    """r09 Eq. (14), (20): fed dPV's own J-V curve, A3 returns e_PCE = 0 and zero diagnostics."""
    for fs in ("E1A", "E3A"):
        dev = data.load(fs, data.device_keys(fs)[:1], profiles=False)[0]
        m = evaluation.curve_metrics(dev.J, dev)
        assert m["e_pce_pp"] < 1e-10 and m["d_voc"] < 1e-12 and m["d_ff"] < 1e-12


def test_iqm_trims_quartiles() -> None:
    """r09 Sec. 9: the IQM is the mean of the middle half; outliers in the outer quartiles do not move it."""
    x = np.arange(1.0, 9.0)  # 1..8, middle half 3..6
    assert abs(evaluation.iqm(x) - 4.5) < 1e-12
    y = x.copy()
    y[-1] = 1e9
    assert abs(evaluation.iqm(y) - 4.5) < 1e-12


@needs_dataset
def test_floor_is_q95_of_refinement_gap() -> None:
    """r09 Eq. (21): r = Q_0.95 of |eta(N_g) - eta(4 N_g)| over the re-simulated devices; for E1A
    the dataset card states 0.047 pp."""
    rec = steps.reference_floor("E1A")
    assert abs(rec["r_pp"] - 0.047) < 5e-4
    e, r = evaluation.floor([0.1] * 20, [0.1 + 1e-4 * i for i in range(20)])
    assert abs(r - np.quantile(np.arange(20) * 1e-2, 0.95)) < 1e-12


def test_bayesian_signed_rank_probabilities_sum_to_one() -> None:
    """r09 Sec. 9: the Bayesian signed-rank test returns P(worse) + P(rope) + P(better) = 1, and a
    configuration that is better on every pair by more than the rope wins."""
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 2, 30)
    out = evaluation.bayesian_signed_rank(y - 0.5, y, rope=0.01, n_samples=4000)
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert out["x_better"] > 0.95
    eq = evaluation.bayesian_signed_rank(y, y + 1e-4, rope=0.01, n_samples=4000)
    assert eq["equivalent"] > 0.95


# --- search and B0 ---------------------------------------------------------------------------------
def test_tpe_space_is_log10_minus3_to_3(tmp_path) -> None:
    """r09 Eq. (19): every suggested lambda lies in [1e-3, 1e3], sampled in log10."""
    seen = []

    def objective(lam):
        seen.append(lam)
        return sum((np.log10(v) - 1.0) ** 2 for v in lam.values())

    best, study = search.run(objective, ["Poisson", "n"], str(tmp_path / "s.db"), "t", n_trials=12)
    assert all(1e-3 <= v <= 1e3 for lam in seen for v in lam.values())
    assert len(study.trials) == 12 and set(best) == {"Poisson", "n"}


def test_tpe_resumes_finished_trials(tmp_path) -> None:
    """Persistence of the search: an interrupted study resumes and only runs the missing trials."""
    calls = []

    def objective(lam):
        calls.append(1)
        return float(sum(np.log10(v) ** 2 for v in lam.values()))

    search.run(objective, ["a"], str(tmp_path / "s.db"), "t", n_trials=5)
    search.run(objective, ["a"], str(tmp_path / "s.db"), "t", n_trials=8)
    assert len(calls) == 8


def test_b0_sees_only_xi() -> None:
    """r09 Sec. 8: B0 receives only xi_hat (input width d_xi) and returns one PCE per device."""
    p = b0.init(jax.random.PRNGKey(0), 3, width=8, depth=2)
    assert p[0]["W"].shape[0] == 3
    assert b0.predict(p, jnp.zeros((5, 3))).shape == (5,)


def test_b0_fits_smooth_target() -> None:
    """r09 Sec. 8: B0 is trained by squared error; on a smooth synthetic PCE map it reaches a
    small training error (the regression machinery works before it meets dPV's data)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (64, 2))
    y = 0.1 + 0.05 * np.sin(3 * x[:, 0]) + 0.02 * x[:, 1]
    p, trace = b0.fit(x, y, seed=0, width=32, depth=2, iters=1500, lr=3e-3)
    assert trace[-1] < 1e-2 * trace[0]
    assert float(np.mean(b0.e_pce_pp(p, x, y))) < 0.5


# --- steps ---------------------------------------------------------------------------------------
@needs_dataset
def test_sigma_per_seed_uses_only_training_devices(tmp_path) -> None:
    """r09 Sec. 2, 6 and A1 (iv): D_phi, Y extremes and sigma_i come only from each seed's training
    subset; two seeds with different training subsets get different aggregates."""
    paths = steps.compute_sigma("E1A", seeds=(0, 1), out_dir=str(tmp_path), max_devices=12)
    recs = [json.load(open(p)) for p in paths]
    keys = data.device_keys("E1A")[:12]
    for rec, seed in zip(recs, (0, 1)):
        assert rec["n_train"] == len(data.split(keys, seed)["train"])
    assert recs[0]["sigma2"] != recs[1]["sigma2"]


def test_pilot_config_matches_protocol() -> None:
    """Experiments v01, E0: fixed weights, pure PINN, base network 6 x 512, Adam 5000 + L-BFGS 500,
    E1A and E3A with seeds 0 to 2."""
    cfg = steps.pilot_config("E1A", 0)
    assert (cfg.strategy, cfg.k_s, cfg.fourier_features, cfg.width, cfg.depth) == ("fixed", 0, 0, 512, 6)
    assert (cfg.adam_iters, cfg.lbfgs_iters, cfg.lr) == (5000, 500, 3e-4)
    assert steps.PILOT_FAMILY_STAGES == ("E1A", "E3A") and steps.PILOT_SEEDS == (0, 1, 2)


def test_budget_rule_stops_where_drop_falls_below_half_r() -> None:
    """Experiments v01, E0: the budget is the mark after which validation e_PCE stops dropping
    by more than r/2 per 1000 iterations, and is undefined while it still drops faster."""
    r = 0.05
    curve = {"1000": 1.0, "2000": 0.4, "3000": 0.38, "4000": 0.37, "5000": 0.369}
    assert steps.budget_rule(curve, r) == 2000
    falling = {1000: 1.0, 2000: 0.8, 3000: 0.6}
    assert steps.budget_rule(falling, r) is None


def test_canonical_paths_accept_flat_layout(tmp_path) -> None:
    """The Kaggle dataset holds the canonical files flat; dPV writes them under <run>/."""
    for rel in data.CANONICAL_FILES["E2A"]:
        (tmp_path / os.path.basename(rel)).write_bytes(b"")
    assert data.canonical_paths("E2A", str(tmp_path)) == [
        str(tmp_path / os.path.basename(rel)) for rel in data.CANONICAL_FILES["E2A"]]
    nested = tmp_path / "nested"
    for rel in data.CANONICAL_FILES["E2A"]:
        (nested / rel).parent.mkdir(parents=True, exist_ok=True)
        (nested / rel).write_bytes(b"")
    assert data.canonical_paths("E2A", str(nested)) == [str(nested / rel) for rel in data.CANONICAL_FILES["E2A"]]


def test_initial_component_spread_is_log10_max_over_min() -> None:
    """Experiments v01, E0: the spread of the initial scaled components is log10(max/min) per
    regime, so 'same order of magnitude' reads as a spread below 1."""
    summ = [{"run_id": "r", "initial_components": [
        {"stage": "eq", "j": 0, "s": {"Poisson": 1e8, "int_phi": 1e2}},
        {"stage": "sweep", "j": 0, "s": {"Poisson": 2.0, "n": 0.5, "p": 1.0}}]}]
    rows = steps.initial_component_spread(summ)
    assert abs(rows[0]["log10_spread"] - 6.0) < 1e-12
    assert abs(rows[1]["log10_spread"] - np.log10(4.0)) < 1e-12


# --- r09 ----------------------------------------------------------------------------------------
def test_log_loss_matches_linear_near_reference() -> None:
    """r09 Sec. 6: near the reference (L_i << sigma_i^2) ell_i = log(1 + L_i / sigma_i^2) agrees
    with the linear scaled sum to second order, and the linear control form is kept exact."""
    s = jnp.asarray([1e-8, 1e-5, 1e-3])
    ell = loss.compress(s, "log1p")
    assert float(jnp.max(jnp.abs(ell - s) / s)) <= float(jnp.max(s))
    assert np.array_equal(np.asarray(loss.compress(s, "linear")), np.asarray(s))
    with pytest.raises(ValueError):
        loss.compress(s, "clip")


def test_log_loss_gradient_never_vanishes() -> None:
    """r09 Sec. 6: grad ell_i = grad L_i / (sigma_i^2 + L_i) stays nonzero far from the reference,
    where cutting with min(L_i / sigma_i^2, c) leaves no gradient at all."""
    big = lambda t: jnp.exp(t)  # noqa: E731 -- an exponential component, as Eq. (4) makes them
    g_log = jax.grad(lambda t: loss.compress(big(t), "log1p"))(200.0)
    g_clip = jax.grad(lambda t: jnp.minimum(big(t), 10.0))(200.0)
    assert abs(float(g_log) - 1.0) < 1e-12
    assert float(g_clip) == 0.0


@needs_dataset
def test_sg_floor_small_on_dpv_profiles() -> None:
    """r09 Eq. (23): with dPV's own Scharfetter-Gummel currents the continuity floor discriminates:
    L^floor / sigma^2 well below one in n and p (the spline operator left it at ~1, rehearsal)."""
    fs = data.FamilyStage.load("E1A")
    devs = data.load("E1A", data.device_keys("E1A")[:3])
    s2, fl = loss.sigma_and_floor(fs, devs, equilibrium_too=False)
    for k in ("n", "p"):
        assert fl[k] / s2[k] < 1e-2, (k, fl[k] / s2[k])
        assert fl[k] <= s2[k] * (1 + 1e-12)


def test_equilibrium_start_gives_zero_quasi_fermi() -> None:
    """r09 A2 step 2: zeroed last-layer rows and bias -Y_min / (Y_max - Y_min) give phi_n = phi_p = 0
    at every point, for the joint and the split (Gummel) network; random weights do not."""
    sc = network.OutputScales(40.0, (-3.0, -5.0), (4.0, 2.0))
    xi, xm = jnp.asarray([0.3, 0.6]), jnp.asarray([0.4])
    for split in (False, True):
        cfg = tiny_cfg(n_int=1, split=split)
        p = network.init(jax.random.PRNGKey(4), cfg)
        before = network.potentials(p, cfg, sc, 0.37, xi, xm, 1.0, -1.0, 0.0)
        assert abs(float(before[1])) > 1e-6
        q = network.equilibrium_start(p, cfg, sc)
        for x in (0.05, 0.37, 0.81):
            _, pn, pp = network.potentials(q, cfg, sc, x, xi, xm, 1.0, -1.0, 0.0)
            assert abs(float(pn)) < 1e-14 and abs(float(pp)) < 1e-14
    with pytest.raises(ValueError):
        network.equilibrium_start(p, cfg, network.OutputScales(40.0, (1.0, -1.0), (2.0, 1.0)))


def test_phi_zero_start_gives_linear_profile() -> None:
    """r09 A2 step 1: the zeroed u_phi row gives phi = phi_lin at every point, with and without
    Fourier features and for the split network; the random row does not."""
    sc = network.OutputScales(40.0, (-3.0, -5.0), (4.0, 2.0))
    xi, xm = jnp.asarray([0.3, 0.6]), jnp.asarray([0.4])
    for split, ff in ((False, 0), (False, 8), (True, 8)):
        cfg = tiny_cfg(n_int=1, split=split, fourier_features=ff, fourier_scale=5.0)
        p = network.init(jax.random.PRNGKey(5), cfg)
        lin = scaling.phi_linear(0.37, 1.0, -1.0, 0.0)
        assert abs(float(network.potentials(p, cfg, sc, 0.37, xi, xm, 1.0, -1.0, 0.0)[0]) - lin) > 1e-6
        q = network.phi_zero_start(p, cfg)
        for x in (0.05, 0.37, 0.81):
            phi = network.potentials(q, cfg, sc, x, xi, xm, 1.0, -1.0, 0.0)[0]
            assert abs(float(phi) - float(scaling.phi_linear(x, 1.0, -1.0, 0.0))) < 1e-14


def test_tg_activation_is_tanh_times_gaussian() -> None:
    """r09 Sec. 8: tg(z) = tanh(z) exp(-z^2/2) (zhang2025), and a network built with it differs
    from the tanh network with the same weights."""
    z = jnp.linspace(-3.0, 3.0, 13)
    assert float(jnp.max(jnp.abs(network.activation("tg", z) - jnp.tanh(z) * jnp.exp(-z ** 2 / 2)))) == 0.0
    xi, xm = jnp.asarray([0.3, 0.6]), jnp.asarray([0.4])
    a, b = tiny_cfg(n_int=1), tiny_cfg(n_int=1, activation="tg")
    p = network.init(jax.random.PRNGKey(6), a)
    ua = network.u_hat(p, a, 0.37, xi, xm)
    ub = network.u_hat(p, b, 0.37, xi, xm)
    assert float(jnp.max(jnp.abs(ua - ub))) > 1e-6
    with pytest.raises(ValueError):
        network.init(jax.random.PRNGKey(6), tiny_cfg(n_int=1, activation="relu"))


def test_separate_networks_one_per_output() -> None:
    """r09 Sec. 8 (riganti2025): with separate networks each output depends only on its own
    network; the zero start of u_phi and the equilibrium start act on the right one."""
    sc = network.OutputScales(40.0, (-3.0, -5.0), (4.0, 2.0))
    xi, xm = jnp.asarray([0.3, 0.6]), jnp.asarray([0.4])
    cfg = tiny_cfg(n_int=1, separate=True, fourier_features=8, fourier_scale=5.0)
    p = network.init(jax.random.PRNGKey(7), cfg)
    assert set(p) == {"phi", "qf", "B"} and len(p["qf"]) == 2
    u = network.u_hat(p, cfg, 0.37, xi, xm)
    for k, net in enumerate([p["phi"], p["qf"][0], p["qf"][1]]):
        q = dict(p)
        bumped = net[:-1] + [dict(net[-1], b=net[-1]["b"] + 1.0)]
        if k == 0:
            q["phi"] = bumped
        else:
            q["qf"] = [bumped if i == k - 1 else p["qf"][i] for i in range(2)]
        diff = network.u_hat(q, cfg, 0.37, xi, xm) - u
        assert abs(float(diff[k]) - 1.0) < 1e-12
        assert float(jnp.max(jnp.abs(jnp.delete(diff, k)))) == 0.0
    q = network.equilibrium_start(network.phi_zero_start(p, cfg), cfg, sc)
    for x in (0.05, 0.37, 0.81):
        phi, pn, pp = network.potentials(q, cfg, sc, x, xi, xm, 1.0, -1.0, 0.0)
        assert abs(float(phi) - float(scaling.phi_linear(x, 1.0, -1.0, 0.0))) < 1e-14
        assert abs(float(pn)) < 1e-14 and abs(float(pp)) < 1e-14


def test_voltage_input_on_unit_interval() -> None:
    """r09 Sec. 8 variant: V_hat = V_tilde / V_max enters as one more unprojected input in [0, 1],
    and the potentials then depend on the voltage beyond the Dirichlet line."""
    cfg = tiny_cfg(n_int=0, v_max=20.0)
    assert cfg.n_in == 1 + 2 + 1
    p = network.init(jax.random.PRNGKey(5), cfg)
    z = network._features(p, cfg, 0.3, jnp.zeros(0), jnp.asarray([0.1, 0.2]), 0.75)
    assert float(z[-1]) == 0.75
    u0 = network.u_hat(p, cfg, 0.3, jnp.asarray([0.1, 0.2]), jnp.zeros(0), v_hat=0.0)
    u1 = network.u_hat(p, cfg, 0.3, jnp.asarray([0.1, 0.2]), jnp.zeros(0), v_hat=1.0)
    assert float(jnp.max(jnp.abs(u0 - u1))) > 0.0


@needs_dataset
def test_equilibrium_start_applied_entering_v0() -> None:
    """r09 A2 step 2: leaving the equilibrium stage sets phi_n = phi_p = 0; the 'random' control
    leaves the weights untouched."""
    for start in ("equilibrium", "random"):
        setup = tiny_setup(qf_start=start)
        st = training.new_state(setup)
        w = np.asarray(st["params"]["all"][-1]["W"]).copy()
        training._advance(setup, st)
        assert st["stage"] == "sweep"
        w_after = np.asarray(st["params"]["all"][-1]["W"])
        if start == "equilibrium":
            assert np.all(w_after[:, 1:] == 0.0)
        else:
            assert np.array_equal(w, w_after)


@needs_dataset
def test_data_start_fits_before_physics() -> None:
    """r09 data start: every regime opens with a data-only step 0 with a finite L_dat, the
    zero start's phi row is not imposed, and the physics stages follow; the zero start has none."""
    setup = tiny_setup(fs_name="E1A", start="data")
    st0 = training.new_state(setup)
    assert st0["phase"] == "pre"
    assert np.any(np.asarray(st0["params"]["all"][-1]["W"])[:, 0] != 0.0)
    st = training.train(setup)
    pre = [h for h in st["history"] if h.get("phase") == "pre"]
    regimes = {(h["stage"], h["j"]) for h in pre}
    assert regimes == {("eq", 0)} | {("sweep", j) for j in range(setup.k_max)}
    first = [h["loss"] for h in pre if h["stage"] == "eq" and h["it"] == 1][0]
    last = [h["loss"] for h in pre if h["stage"] == "eq" and h["it"] == setup.cfg.pretrain_iters][0]
    assert np.isfinite(first) and np.isfinite(last) and first > 0.0
    assert {h["stage"] for h in st["history"] if "names" in h} == {"eq", "sweep"}
    zero = training.train(tiny_setup(fs_name="E1A"))
    assert not [h for h in zero["history"] if h.get("phase") == "pre"]


@needs_dataset
def test_voltage_input_trains_one_stage() -> None:
    """r09 Sec. 8 variant: after the equilibrium, one stage over V_hat with the total budget, then
    every voltage of the curve is evaluated from the same weights."""
    setup = tiny_setup(fs_name="E1A", v_input=True)
    st = training.train(setup)
    stages = {h["stage"] for h in st["history"]}
    assert stages == {"eq", "vin"}
    assert max(h["it"] for h in st["history"] if h["stage"] == "vin" and h["it"] != "lbfgs") \
        == setup.cfg.adam_iters * setup.k_max
    assert all(len(recs) == setup.k_max for recs in st["eval"].values())


def test_gradient_shares_sum_to_one() -> None:
    """Experiments v05, E0 (ii): the shares ||grad lambda_i ell_i|| / sum_l ||grad lambda_l ell_l||
    sum to one, and a component scaled up by 1e6 takes almost all of it."""
    w = {"a": jnp.asarray([1.0, 2.0]), "b": jnp.asarray(0.5)}
    f = lambda p: jnp.stack([jnp.sum(p["a"] ** 2), p["b"] ** 2 * 1e6])  # noqa: E731
    sh = training.gradient_shares(f, w, jnp.ones(2))
    assert abs(float(jnp.sum(sh)) - 1.0) < 1e-12
    assert float(sh[1]) > 0.99


def test_screen_verdicts_follow_caps() -> None:
    """Experiments v05, E0: (ii) early gradient share <= 0.9 in regimes with at least two active
    components, (iii) no non-finite value and no exact zero; each verdict flips when its quantity
    crosses its cap, and a one-component regime does not count for (ii)."""
    ok = {"nonfinite": 0, "zero_components": 0,
          "grad_shares": [{"it": 1, "share": [1.0]}, {"it": 100, "share": [0.5, 0.5]},
                          {"it": 900, "share": [0.99, 0.01]}]}
    v = steps.screen_verdicts(ok)
    assert v["h_ii_no_dominant_gradient"] and v["h_iii_finite_and_nonzero"]
    assert "h_i_initial_bounded" not in v
    bad = dict(ok, zero_components=1, grad_shares=[{"it": 200, "share": [0.95, 0.05]}])
    v = steps.screen_verdicts(bad)
    assert not (v["h_ii_no_dominant_gradient"] or v["h_iii_finite_and_nonzero"])


def test_batch_rule_halves_only_on_out_of_memory(monkeypatch, tmp_path) -> None:
    """Experiments v05 batch rule: an out-of-memory error halves N_Q and resumes, recording the
    change; any other error propagates untouched."""
    import dataclasses

    calls = []
    cfg = training.TrainConfig(family_stage="E1A", seed=0, n_q=512)

    class S:
        def __init__(self, c):
            self.cfg = c

    def fake_train(setup, state, cb, models_dir=None):
        calls.append(setup.cfg.n_q)
        if len(calls) == 1:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate")
        return {"done": True, "batch_events": (state or {}).get("batch_events", [])}

    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr(training, "prepare", lambda c, sig, root=None: S(c))
    monkeypatch.setattr(steps, "load_sigma", lambda *a, **k: {})
    monkeypatch.setattr(persistence, "load_checkpoint",
                        lambda d: {"stage": "sweep", "j": 3, "cfg": dataclasses.asdict(cfg)})
    st = steps._train_with_batch_rule(S(cfg), None, None, str(tmp_path), None, None, "x")
    assert calls == [512, 256]
    assert st["batch_events"] == [{"stage": "sweep", "j": 3, "from_n_q": 512, "to_n_q": 256}]

    def other(setup, state, cb, models_dir=None):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(training, "train", other)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        steps._train_with_batch_rule(S(cfg), None, None, str(tmp_path), None, None, "x")


def test_brdr_ema_fixed_not_invariant_to_sigma() -> None:
    """r09 Sec. 6: BRDR's lambda is unchanged by sigma_i but its effective weight lambda_i / sigma_i^2
    moves, and EMA and fixed weights change their descent direction: none is invariant."""
    params, comps = _quad_problem()
    sig = jnp.asarray([1e-3, 1.0, 1e4])

    def cos(a, b):
        return float(jnp.dot(a, b) / jnp.linalg.norm(a) / jnp.linalg.norm(b))

    def direction(update, fn, steps=50):
        names = ("a", "b", "c")
        st = weighting.init_state(names, "brdr" if update is weighting.brdr_update else "ema")
        for _ in range(steps):
            st = update(st, fn(params))
        return jax.grad(lambda p: jnp.sum(st["lam"] * fn(p)))(params)["w"]

    for upd in (weighting.brdr_update, weighting.ema_update):
        assert cos(direction(upd, comps), direction(upd, lambda p: comps(p) / sig)) < 1 - 1e-3


def test_no_cusp_ablation_is_smooth_at_interfaces() -> None:
    """Experiments v05 E6: without rho_m the network is a smooth perceptron on (x_hat, xi_hat), so its
    x-derivative cannot jump at x_hat_m."""
    cfg = tiny_cfg(n_int=1, cusp=False)
    assert cfg.n_in == 1 + 2
    p = network.init(jax.random.PRNGKey(6), cfg)
    xi, xm = jnp.asarray([0.4, 0.2]), jnp.asarray([0.37])
    jump = jax.jacfwd(lambda x: network.u_hat(p, cfg, x, xi, xm, side=jnp.asarray([1.0])))(xm[0]) - \
        jax.jacfwd(lambda x: network.u_hat(p, cfg, x, xi, xm, side=jnp.asarray([-1.0])))(xm[0])
    assert float(jnp.max(jnp.abs(jump))) == 0.0


@needs_dataset
def test_no_sigma_ablation_uses_unit_scales() -> None:
    """Experiments v05 E6: sigma_scale=False divides every component by one, so the scaled vector
    equals the raw components."""
    from dataclasses import replace
    setup = tiny_setup()
    plain = training.Setup(replace(setup.cfg, sigma_scale=False), setup.fs, setup.prob, setup.split,
                           setup.train, setup.evaluate, setup.k_max)
    names, scaled, _ = training.make_fns(plain, equilibrium=True)
    p = network.init(jax.random.PRNGKey(0), setup.prob.cfg)
    batch = training.collocation(setup.cfg, setup.fs.d, 0, 0)
    comps = loss.components(p, setup.prob, batch, 0.0, equilibrium=True)
    want = loss.compress(jnp.stack([comps[c] for c in names]), "log1p")
    assert float(jnp.max(jnp.abs(scaled(p, batch, 0.0) - want))) <= 1e-12 * float(jnp.max(want))


def test_screen_conclusion_follows_rows() -> None:
    """Experiments v05, E0 success criterion: the chosen configuration is the lowest validation e_PCE
    among runs meeting (ii) and (iii); the chosen tools are the lowest mean e_PCE over fixed weights
    and BRDR among combinations meeting both with both; changing a value changes the choice, and a
    control that violates (ii) is detected."""
    good = dict(h_ii_no_dominant_gradient=True, h_iii_finite_and_nonzero=True)

    def arch(ff, act, w, e, **kw):
        o = {"family_stage": "E1A", "strategy": w, "fourier_features": ff, "activation": act, "separate": False}
        return dict(run_id=f"{ff}{act}{w}", overrides=o, mean_e_pce_val=e, **dict(good, **kw))

    rows = [dict(run_id="base", overrides={"family_stage": "E1A"}, mean_e_pce_val=0.5, **good),
            dict(run_id="ctrl", overrides={"family_stage": "E1A", "loss_form": "linear"}, mean_e_pce_val=0.1,
                 **dict(good, h_ii_no_dominant_gradient=False)),
            arch(64, "tanh", "fixed", 0.3), arch(64, "tanh", "brdr", 0.1),
            arch(0, "tg", "fixed", 0.15), arch(0, "tg", "brdr", 0.35),
            arch(64, "tg", "fixed", 0.01), arch(64, "tg", "brdr", 0.02, h_iii_finite_and_nonzero=False)]
    c = steps.screen_conclusion(rows)
    assert c["base_holds"] == {"E1A": True} and c["a_control_violates"]
    assert c["chosen"] == "64tgfixed"
    assert c["chosen_tools"] == {"fourier": True, "activation": "tanh", "separate": False}
    rows[2]["mean_e_pce_val"] = 0.6
    assert steps.screen_conclusion(rows)["chosen_tools"] == {"fourier": False, "activation": "tg", "separate": False}
