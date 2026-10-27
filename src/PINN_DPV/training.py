"""Training by voltage continuation (Algorithm A2) with online evaluation (A3) and checkpoints.

Sec. 8, A2: (1) dark equilibrium, G = 0, V = 0, phi_n = phi_p = 0 hard, only u_phi
trained on R_Poisson and the int_phi jump; (2) illumination at V_0 = 0, warm-started,
every component of Eq. (15) active; (3) for j = 1..K_max - 1, V_j = j 0.05 V in
Eq. (3) and (11), warm-started from V_{j-1} with the same budget, the weighting
strategy acting inside each voltage. Adam at 3e-4 with exponential decay, then a
final L-BFGS stage; Halton collocation resampled every iteration with a 10:1 ratio
between interior points and boundary devices; double precision.

Each voltage ends with the online evaluation of A3 on the validation and test
devices (and the corners), and a checkpoint; a checkpoint is also written every
`checkpoint_every` iterations. Every random draw is a function of (seed, voltage,
iteration), so a resumed run repeats the uninterrupted one exactly.

The L-BFGS stage of a voltage uses one fixed collocation batch, because its line
search needs one fixed objective.

r06: the loss is sum_i lambda_i log(1 + s_i), s_i = L_i / sigma_i^2 (`loss_form`
"linear" keeps the previous linear sum as E0's control), and every weighting strategy acts on
ell_i = log(1 + s_i). Entering V_0 the quasi-Fermi outputs start at the
equilibrium (`qf_start`, "random" is E0's control). The voltage-as-input variant
(`v_input`) replaces steps 2-3 by one stage "vin" over V_hat in [0, 1] with the
same total budget (adam_iters and lbfgs_iters times the number of voltages).
E0 diagnostics: the raw s_i at the first step of each regime, and the share of the
gradient norm of each lambda_i ell_i at `grad_share_marks`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import qmc

from PINN_DPV import contacts, data, evaluation, loss, network, persistence, scaling, weighting

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r06.md",
    "sections": ["6", "8"],
    "equations": ["15", "16", "17", "18"],
    "invariants": [
        "resume_equals_uninterrupted",
        "collocation_deterministic_per_step",
        "equilibrium_trains_phi_only",
        "loss_decreases_on_tiny_problem",
        "equilibrium_start_applied_entering_v0",
        "voltage_input_trains_one_stage",
        "gradient_shares_sum_to_one",
    ],
}


@dataclass(frozen=True)
class TrainConfig:
    family_stage: str
    seed: int
    strategy: str = "fixed"
    k_s: int = 0  # 0: pure PINN; > 0: PINN + sparse state with K_s positions
    fourier_features: int = 0
    fourier_scale: float = 1.0
    width: int = 512
    depth: int = 6
    n_q: int = 2000
    adam_iters: int = 5000
    lbfgs_iters: int = 500
    lr: float = 3e-4
    lr_decay: float = 0.1  # factor reached at the end of each voltage's Adam stage
    data_batch: int = 64
    checkpoint_every: int = 500
    gummel_block: int = 100
    keep_all_weights: bool = False
    max_voltages: int | None = None  # rehearsal cap on the sweep length
    max_devices: int | None = None  # rehearsal cap per subset
    device_pool: int | None = None  # E0 screening: the first N devices, split as sigma splits them
    fixed_lambdas: dict | None = None  # TPE's chosen weights
    eval_marks: tuple = ()  # Adam iterations at which validation devices are also evaluated (E0)
    loss_form: str = "log1p"  # r06 Eq. (15); "linear": the previous linear sum, E0's control
    qf_start: str = "equilibrium"  # r06 A2 step 2; "random": the previous random start, E0's control
    v_input: bool = False  # r06 Sec. 8 variant: one network for the whole curve
    grad_share_marks: tuple = (1, 100, 200, 300, 400, 500)
    run_id: str = ""

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        arch = f"ff{self.fourier_features}x{self.fourier_scale:g}" if self.fourier_features else "mlp"
        dat = f"ks{self.k_s}" if self.k_s else "pure"
        extra = "-vin" if self.v_input else ""
        extra += "" if self.loss_form == "log1p" else f"-{self.loss_form}"
        extra += "" if self.qf_start == "equilibrium" else f"-qf{self.qf_start}"
        return (f"{self.family_stage}-{self.strategy}-{dat}-{arch}-w{self.width}d{self.depth}"
                f"{extra}-s{self.seed}")


# --- Deterministic draws -----------------------------------------------------------------
def _seed_int(*parts) -> int:
    return int(np.random.SeedSequence([abs(int(p)) for p in parts]).generate_state(1)[0])


def collocation(cfg: TrainConfig, d: int, j_code: int, it: int, v_max: float = 0.0):
    """Halton points for one step: N_Q interior (x_hat, xi_hat), N_Q/10 boundary devices.

    With `v_max` > 0 (voltage-as-input stage) every point and device also draws
    V_tilde = V_hat v_max with V_hat in [0, 1] as one more Halton coordinate.
    """
    n_b = max(1, cfg.n_q // 10)
    extra = 1 if v_max > 0 else 0
    pts = qmc.Halton(d=1 + d + extra, scramble=True, seed=_seed_int(cfg.seed, j_code, it, 1)).random(cfg.n_q)
    dev = qmc.Halton(d=d + extra, scramble=True, seed=_seed_int(cfg.seed, j_code, it, 2)).random(n_b)
    out = {"xq": jnp.asarray(pts[:, 0]), "xiq": jnp.asarray(pts[:, 1:1 + d]), "xib": jnp.asarray(dev[:, :d])}
    if extra:
        out["vq"] = jnp.asarray(pts[:, -1] * v_max)
        out["vb"] = jnp.asarray(dev[:, -1] * v_max)
    return out


# --- Setup --------------------------------------------------------------------------------
@dataclass
class Setup:
    cfg: TrainConfig
    fs: data.FamilyStage
    prob: loss.Problem
    split: dict
    train: list
    evaluate: list
    k_max: int
    sparse: dict = field(default_factory=dict)


def sparse_positions(dev, j, k_s):
    """K_s node indices: half uniform in x_hat, half where |dphi| is largest (space-charge regions)."""
    n = dev.x.size
    half = k_s // 2
    uniform = np.unique(np.linspace(0, n - 1, half).round().astype(int))
    grad = np.abs(np.diff(dev.pots[j, 0]))
    steep = np.argsort(-grad)
    picked = [i for i in steep if i not in set(uniform)][: k_s - uniform.size]
    return np.sort(np.concatenate([uniform, np.asarray(picked, int)]))


def prepare(cfg: TrainConfig, sigma: dict, root=None) -> Setup:
    """Load and split the devices, and build the problem with the precomputed scales.

    `sigma` is what the sigma step wrote: {"sigma2": {...}, "scales": {...}}.
    """
    fs = data.FamilyStage.load(cfg.family_stage, root)
    keys = data.device_keys(cfg.family_stage, root)
    if cfg.device_pool:
        keys = keys[: cfg.device_pool]
    corners = data.device_keys(cfg.family_stage, root, group="esquinas")
    parts = data.split(keys, cfg.seed, corners)
    if cfg.max_devices:
        parts = {k: v[: cfg.max_devices] for k, v in parts.items()}
    train = data.load(cfg.family_stage, parts["train"], root)
    evaluate = data.load(cfg.family_stage, parts["val"] + parts["test"], root)
    if cfg.v_input and cfg.k_s:
        raise ValueError("the voltage-as-input variant has no sparse-state data term")
    if cfg.loss_form not in loss.LOSS_FORMS or cfg.qf_start not in ("equilibrium", "random"):
        raise ValueError(f"unknown loss_form {cfg.loss_form!r} or qf_start {cfg.qf_start!r}")
    sc = sigma["scales"]
    scales = network.OutputScales(sc["D_phi"], tuple(sc["Y_min"]), tuple(sc["Y_max"]))
    k_max = max(len(d.V) for d in train)
    if cfg.max_voltages:
        k_max = min(k_max, cfg.max_voltages)
    v_max = contacts.DV * (k_max - 1) / scaling.ENERGY if cfg.v_input else 0.0
    net = network.NetConfig(d_xi=fs.d, n_int=fs.n_int, width=cfg.width, depth=cfg.depth,
                            fourier_features=cfg.fourier_features, fourier_scale=cfg.fourier_scale,
                            split=cfg.strategy == "gummel", v_max=v_max)
    prob = loss.Problem(fs=fs, cfg=net, scales=scales, sigma2=dict(sigma["sigma2"]))
    setup = Setup(cfg, fs, prob, parts, train, evaluate, k_max)
    if cfg.k_s:
        for i, d in enumerate(train):
            x_hat = scaling.x_hat_from_debye(d.x)
            for j in range(min(len(d.V), k_max)):
                idx = sparse_positions(d, j, cfg.k_s)
                setup.sparse[(i, j)] = (x_hat[idx], d.pots[j][:, idx])
    return setup


def data_batch(setup: Setup, j: int, it: int):
    cfg = setup.cfg
    avail = [i for i in range(len(setup.train)) if (i, j) in setup.sparse]
    rng = np.random.default_rng(_seed_int(cfg.seed, j + 1, it, 3))
    pick = rng.choice(avail, size=min(cfg.data_batch, len(avail)), replace=False)
    return {"xid": jnp.asarray(np.stack([setup.train[i].u for i in pick])),
            "xs": jnp.asarray(np.stack([setup.sparse[(i, j)][0] for i in pick])),
            "ref": jnp.asarray(np.stack([setup.sparse[(i, j)][1] for i in pick]))}


# --- Steps ----------------------------------------------------------------------------------
def make_fns(setup: Setup, equilibrium: bool):
    """Jitted (Adam step, scaled components) for one regime, and the component names."""
    prob, cfg = setup.prob, setup.cfg
    names = loss.active_components(setup.fs.n_int, equilibrium, with_data=cfg.k_s > 0)
    key = (lambda c: f"{c}@eq") if equilibrium else (lambda c: c)
    sig = jnp.asarray([prob.sigma2.get(key(c), 1.0) if c != "dat" else 1.0 for c in names])

    def raw(params, batch, v_tilde):  # s_i = L_i / sigma_i^2
        comps = loss.components(params, prob, batch, v_tilde, equilibrium)
        return jnp.stack([comps[c] for c in names]) / sig

    def scaled(params, batch, v_tilde):  # ell_i, what the weighting strategies act on
        return loss.compress(raw(params, batch, v_tilde), cfg.loss_form)

    def objective(params, batch, v_tilde, lam):
        s = raw(params, batch, v_tilde)
        return jnp.sum(lam * loss.compress(s, cfg.loss_form)), s

    return names, jax.jit(scaled), objective


def gradient_shares(scaled_fn, params, lam):
    """Share of the gradient norm carried by each lambda_i ell_i: ||g_i|| / sum_l ||g_l||."""
    jac = jax.jacrev(lambda p: lam * scaled_fn(p))(params)
    leaves = jax.tree_util.tree_leaves(jac)
    norms = jnp.sqrt(sum(jnp.sum(a.reshape(a.shape[0], -1) ** 2, axis=1) for a in leaves))
    return norms / jnp.sum(norms)


def _optimizer(cfg: TrainConfig, n_adam: int | None = None):
    sched = optax.exponential_decay(cfg.lr, transition_steps=max(n_adam or cfg.adam_iters, 1),
                                    decay_rate=cfg.lr_decay)
    return optax.adam(sched)


def stage_budget(setup: Setup, stage: str):
    """(Adam, L-BFGS) iterations of a stage: per voltage, or all voltages' total in "vin"."""
    k = setup.k_max if stage == "vin" else 1
    return setup.cfg.adam_iters * k, setup.cfg.lbfgs_iters * k


def _freeze(grads, strategy, phase):
    """Zero the gradient of the frozen Fourier matrix and, in Gummel, of the idle network."""
    g = dict(grads)
    if "B" in g:
        g["B"] = jnp.zeros_like(g["B"])
    if strategy == "gummel":
        idle = "qf" if phase == "phi" else "phi"
        g[idle] = jax.tree_util.tree_map(jnp.zeros_like, g[idle])
    return g


def new_state(setup: Setup):
    cfg = setup.cfg
    params = network.init(jax.random.PRNGKey(cfg.seed), setup.prob.cfg)
    return {"run_id": cfg.resolved_run_id(), "cfg": asdict(cfg), "stage": "eq", "j": 0,
            "phase": "adam", "it": 0, "params": params, "opt_state": _optimizer(cfg).init(params),
            "wstate": None, "lbfgs_state": None, "eval": {}, "history": [], "done": False,
            "seconds": 0.0, "per_voltage_seconds": []}


def train(setup: Setup, state=None, callback=None, stop_after=None, models_dir=None):
    """Run (or resume) A2 from `state`; `callback(state, voltage_done)` persists it.

    `stop_after` (iterations) interrupts the run deterministically, for tests of
    resumption; the returned state is resumable.
    """
    cfg = setup.cfg
    state = new_state(setup) if state is None else state
    done_iters = 0
    regimes = {}

    def regime(eq):
        if eq not in regimes:
            regimes[eq] = make_fns(setup, eq)
        return regimes[eq]

    steps = {}

    def get_step(eq, phase, n_adam):
        k = (eq, phase, n_adam)
        if k not in steps:
            names, _, objective = regime(eq)
            opt = _optimizer(cfg, n_adam)

            def step(params, opt_state, batch, v_tilde, lam):
                (val, s), g = jax.value_and_grad(objective, has_aux=True)(params, batch, v_tilde, lam)
                g = _freeze(g, cfg.strategy, phase)
                upd, opt_state = opt.update(g, opt_state, params)
                return optax.apply_updates(params, upd), opt_state, val, s

            steps[k] = jax.jit(step)
        return steps[k]

    shares = {}

    def share_fn(eq):
        if eq not in shares:
            _, scaled, _ = regime(eq)
            shares[eq] = jax.jit(lambda p, b, v, lam: gradient_shares(lambda q: scaled(q, b, v), p, lam))
        return shares[eq]

    while not state["done"]:
        eq = state["stage"] == "eq"
        vin = state["stage"] == "vin"
        j = state["j"]
        j_code = 0 if eq else j + 1
        v_tilde = 0.0 if eq or vin else contacts.DV * j / scaling.ENERGY
        v_max = setup.prob.cfg.v_max if vin else 0.0
        n_adam, n_lbfgs = stage_budget(setup, state["stage"])
        names, scaled, objective = regime(eq)
        if state["wstate"] is None or state["wstate"].get("names") != list(names):
            ws = weighting.init_state(names, cfg.strategy, cfg.fixed_lambdas)
            ws["names"] = list(names)
            state["wstate"] = ws
        t0 = time.time()

        if state["phase"] == "adam":
            while state["it"] < n_adam:
                it = state["it"]
                batch = collocation(cfg, setup.fs.d, j_code, it, v_max)
                if cfg.k_s and not eq:
                    batch.update(data_batch(setup, j, it))
                lam = _weights(state, scaled, batch, v_tilde, it, names, setup)
                phase = weighting.gummel_phase(it, cfg.gummel_block) if cfg.strategy == "gummel" else "all"
                if cfg.strategy == "gummel":
                    lam = lam * weighting.gummel_mask(names, phase)
                if it + 1 in cfg.grad_share_marks:
                    g_share = [float(x) for x in share_fn(eq)(state["params"], batch, v_tilde, lam)]
                    state.setdefault("grad_shares", []).append(
                        {"stage": state["stage"], "j": j, "it": it + 1, "names": list(names), "share": g_share})
                step = get_step(eq, phase, n_adam)
                state["params"], state["opt_state"], val, s = step(
                    state["params"], state["opt_state"], batch, v_tilde, lam)
                if cfg.strategy in ("brdr", "ema"):
                    upd = weighting.brdr_update if cfg.strategy == "brdr" else weighting.ema_update
                    ws = upd({k: v for k, v in state["wstate"].items() if k != "names"},
                             loss.compress(s, cfg.loss_form))
                    ws["names"] = list(names)
                    state["wstate"] = ws
                state["it"] = it + 1
                if not eq and state["it"] in cfg.eval_marks:
                    _evaluate_mark(setup, state, j, state["it"])
                # it == 1 records the components at the initial weights of this voltage
                # (s is evaluated before the update): E0's question about their scale.
                if state["it"] % 100 == 0 or state["it"] in (1, n_adam):
                    state["history"].append({"stage": state["stage"], "j": j, "it": state["it"],
                                             "loss": float(val), "s": [float(x) for x in s],
                                             "names": list(names)})
                done_iters += 1
                if callback and state["it"] % cfg.checkpoint_every == 0:
                    callback(state, False)
                if stop_after is not None and done_iters >= stop_after:
                    state["seconds"] += time.time() - t0
                    return state
            state["phase"], state["it"] = "lbfgs", 0

        if state["phase"] == "lbfgs":
            state = _lbfgs(setup, state, objective, names, j_code, v_tilde, eq, n_lbfgs, v_max)

        # the voltage, the equilibrium or the whole voltage-input stage is finished
        dt = time.time() - t0
        state["seconds"] += dt
        state["per_voltage_seconds"].append({"stage": state["stage"], "j": j, "seconds": dt})
        if vin:
            for jj in range(setup.k_max):
                _evaluate_voltage(setup, state, jj)
        elif not eq:
            _evaluate_voltage(setup, state, j)
            if cfg.keep_all_weights and models_dir:
                persistence.save_checkpoint(f"{models_dir}/v{j:03d}", {"params": state["params"]})
        _advance(setup, state)
        state["opt_state"] = _optimizer(cfg).init(state["params"])
        if callback:
            callback(state, True)
    return state


def _weights(state, scaled, batch, v_tilde, it, names, setup):
    cfg = setup.cfg
    ws = state["wstate"]
    if cfg.strategy == "ntk" and it % weighting.NTK_EVERY == 0:
        lam = weighting.ntk_weights(lambda p: scaled(p, batch, v_tilde), state["params"])
        ws["lam"] = lam
    return ws["lam"]


def _lbfgs(setup, state, objective, names, j_code, v_tilde, eq, n_lbfgs, v_max=0.0):
    cfg = setup.cfg
    if n_lbfgs <= 0:
        return state
    batch = collocation(cfg, setup.fs.d, j_code, 10**6, v_max)  # one fixed batch per voltage
    if cfg.k_s and not eq:
        batch.update(data_batch(setup, state["j"], 10**6))
    lam = state["wstate"]["lam"]
    if cfg.strategy == "gummel":
        lam = jnp.ones_like(lam)

    def f(p):
        return objective(p, batch, v_tilde, lam)[0]

    solver = optax.lbfgs()
    vg = optax.value_and_grad_from_state(f)

    @jax.jit
    def step(params, opt_state):
        value, grad = vg(params, state=opt_state)
        grad = _freeze(grad, "fixed", "all")
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=f)
        return optax.apply_updates(params, updates), opt_state, value

    params = state["params"]
    opt_state = solver.init(params)
    for _ in range(n_lbfgs):
        params, opt_state, value = step(params, opt_state)
    state["params"] = params
    state["history"].append({"stage": state["stage"], "j": state["j"], "it": "lbfgs",
                             "loss": float(value)})
    return state


def _evaluate_mark(setup, state, j, mark):
    """Validation devices only, at an intermediate Adam iteration of voltage j (E0 budget curve).

    The curve these records form approximates a run trained with that budget at
    every voltage: each voltage here is warm-started from the full-budget weights.
    """
    val = set(setup.split["val"])
    marks = state.setdefault("eval_marks", {}).setdefault(str(mark), {})
    for d in setup.evaluate:
        if d.key in val:
            marks.setdefault(d.key, []).append(evaluation.voltage_record(state["params"], setup.prob, d, j))


def _evaluate_voltage(setup, state, j):
    for d in setup.evaluate:
        rec = evaluation.voltage_record(state["params"], setup.prob, d, j)
        state["eval"].setdefault(d.key, []).append(rec)


def _advance(setup, state):
    if state["stage"] == "eq":
        state["stage"], state["j"] = ("vin" if setup.cfg.v_input else "sweep"), 0
        if setup.cfg.qf_start == "equilibrium":
            state["params"] = network.equilibrium_start(state["params"], setup.prob.cfg, setup.prob.scales)
    elif state["stage"] == "vin":
        state["done"] = True
    elif state["j"] + 1 < setup.k_max:
        state["j"] += 1
    else:
        state["done"] = True
    state["phase"], state["it"] = "adam", 0


def summarize(setup: Setup, state) -> dict:
    """Per-device curve metrics and the seed's statistical unit: mean e_PCE over test devices."""
    by_key = {d.key: d for d in setup.evaluate}
    out = {"run_id": state["run_id"], "family_stage": setup.cfg.family_stage, "seed": setup.cfg.seed,
           "strategy": setup.cfg.strategy, "devices": {}, "seconds": state["seconds"]}
    for key, recs in state["eval"].items():
        out["devices"][key] = evaluation.curve_metrics([r["J"] for r in recs], by_key[key])
    for subset in ("val", "test"):
        keys = [k for k in setup.split[subset] if k in out["devices"] and "/sims/" in k]
        e = [out["devices"][k]["e_pce_pp"] for k in keys]
        out[f"mean_e_pce_{subset}"] = float(np.mean(e)) if e else float("nan")
    by_mark = {}
    for mark, recs_by_key in state.get("eval_marks", {}).items():
        e = [evaluation.curve_metrics([r["J"] for r in recs], by_key[k])["e_pce_pp"]
             for k, recs in recs_by_key.items()]
        by_mark[mark] = float(np.mean(e)) if e else float("nan")
    out["mean_e_pce_val_by_mark"] = by_mark
    corners = [k for k in setup.split["test"] if "/esquinas/" in k and k in out["devices"]]
    out["mean_e_pce_corners"] = float(np.mean([out["devices"][k]["e_pce_pp"] for k in corners])) \
        if corners else float("nan")
    # E0: the scaled components s_i = L_i / sigma_i^2 at the weights each regime starts
    # from, and whether anything non-finite appeared anywhere in the run.
    out["initial_components"] = [{"stage": h["stage"], "j": h["j"], "s": dict(zip(h["names"], h["s"]))}
                                 for h in state["history"] if h.get("it") == 1 and "names" in h]
    values = [h["loss"] for h in state["history"]] + [x for h in state["history"] for x in h.get("s", [])]
    values += [r["J"] for recs in state["eval"].values() for r in recs]
    out["nonfinite"] = int(sum(not np.isfinite(v) for v in values))
    # E0 (experiments v02): exact zeros among active components (underflow), gradient
    # shares, the mean field errors of Eq. (20) over test devices, and batch changes.
    out["zero_components"] = int(sum(x == 0.0 for h in state["history"] for x in h.get("s", [])))
    out["grad_shares"] = state.get("grad_shares", [])
    test = [k for k in setup.split["test"] if k in state["eval"]]
    for f in ("e_phi", "e_phi_n", "e_phi_p"):
        v = [r[f] for k in test for r in state["eval"][k] if f in r]
        out[f"mean_{f}_test"] = float(np.mean(v)) if v else float("nan")
    out["batch_events"] = state.get("batch_events", [])
    out["n_q_final"] = setup.cfg.n_q
    return out
