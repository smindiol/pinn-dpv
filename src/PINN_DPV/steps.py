"""The declared steps of the E0 pilot: scales on CPU, the pilot runs on a GPU, the report.

`sigma` (local, CPU): for every family-stage, Algorithm A1 (iv)-(v): the output
scales D_phi, Y_min, Y_max of Eq. (3), sigma_i^2 and L_i^floor of Eq. (23) over the
training subset of each seed, and the reference floor r of Eq. (21) from the
re-simulated devices. Per-device sums are computed once and averaged per seed.

`pilot` (remote: rehearsed on a Kaggle GPU, then run on the Colab A100 by the
operator): full-curve training with fixed weights, 3 seeds, E1A and E3A, with
checkpoints, results branch, release asset and Drive mirror, and validation
e_PCE at intermediate iteration marks for the budget rule of E0.

`report` (local): reads both and writes the E0 tables.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from PINN_DPV import data, evaluation, loss, persistence, scaling, training

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["7", "8", "9"],
    "equations": ["3", "21", "23"],
    "invariants": [
        "sigma_per_seed_uses_only_training_devices",
        "pilot_config_matches_protocol",
        "budget_rule_stops_where_drop_falls_below_half_r",
        "initial_component_spread_is_log10_max_over_min",
        "screen_verdicts_follow_caps",
        "screen_conclusion_follows_rows",
        "batch_rule_halves_only_on_out_of_memory",
    ],
}

PRODUCT = "PINN-DPV"
# Anchored to the repository, never the working directory: `step`, a notebook and a
# remote worker each start somewhere else, and all of them must write the same tree.
# $PINN_DPV_RESULTS moves every output elsewhere (a Drive folder on Colab).
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("PINN_DPV_RESULTS") or REPO_ROOT / PRODUCT / "Results")
SIGMA_DIR = str(RESULTS_ROOT / "sigma")
PILOT_DIR = str(RESULTS_ROOT / "pilot")
REPORT_DIR = str(RESULTS_ROOT / "report")
FAMILY_STAGES = ("E1A", "E1B", "E2A", "E2B", "E3A", "E3B", "E4A", "E4B")
SEEDS = tuple(range(10))
PILOT_FAMILY_STAGES = ("E1A", "E3A")
PILOT_SEEDS = (0, 1, 2)
PILOT_EVAL_MARKS = (1000, 2000, 3000, 4000, 5000)
REPO = "smindiol/pinn-dpv"


# --- sigma -----------------------------------------------------------------------------------
def device_sums(fs, dev):
    """Per-device sums over voltages of Eq. (23)'s terms, and its output-scale extremes."""
    sums = {}
    for k, eq in [(k, False) for k in range(len(dev.V))] + [(None, True)]:
        for c, (s2, r2) in loss.reference_terms(fs, dev, np.asarray(dev.u), k, equilibrium=eq).items():
            key = f"{c}@eq" if eq else c
            a = sums.setdefault(key, [0.0, 0.0, 0])
            a[0] += s2
            a[1] += r2
            a[2] += 1
    x_hat = scaling.x_hat_from_debye(dev.x)
    sc = scaling.output_scales([(x_hat, dev.pots, dev.phi_eq[0], dev.phi_eq[-1], dev.V / scaling.ENERGY)])
    return {"sums": sums, "D_phi": sc["D_phi"], "Y_min": sc["Y_min"].tolist(), "Y_max": sc["Y_max"].tolist()}


def aggregate(per_device: dict, keys) -> dict:
    """sigma_i^2, L_i^floor and output scales over the given (training) device keys."""
    acc, d_phi, lo, hi = {}, 0.0, [np.inf, np.inf], [-np.inf, -np.inf]
    for k in keys:
        rec = per_device[k]
        for c, (s2, r2, n) in rec["sums"].items():
            a = acc.setdefault(c, [0.0, 0.0, 0])
            a[0] += s2
            a[1] += r2
            a[2] += n
        d_phi = max(d_phi, rec["D_phi"])
        lo = np.minimum(lo, rec["Y_min"]).tolist()
        hi = np.maximum(hi, rec["Y_max"]).tolist()
    return {"sigma2": {c: v[0] / v[2] for c, v in acc.items()},
            "floor": {c: v[1] / v[2] for c, v in acc.items()},
            "scales": {"D_phi": d_phi, "Y_min": lo, "Y_max": hi}, "n_train": len(list(keys))}


def reference_floor(family_stage: str, root=None) -> dict:
    """Eq. (21) from the re-simulated devices of every block: e_floor in pp and r = Q_0.95."""
    base, refined = [], []
    for path in data.canonical_paths(family_stage, root):
        ref = {d.key.split("/")[-1]: d for d in data.iter_devices(path, "refinados", profiles=False)}
        if not ref:
            continue
        for d in data.iter_devices(path, "sims", keys=sorted(ref), profiles=False):
            name = d.key.split("/")[-1]
            base.append(d.pce)
            refined.append(ref[name].pce)
    e, r = evaluation.floor(base, refined)
    return {"r_pp": r, "n": len(e)}


def compute_sigma(family_stage: str, seeds=SEEDS, root=None, out_dir=SIGMA_DIR, max_devices=None):
    """Write per-seed sigma JSONs for one family-stage; return their paths."""
    fs = data.FamilyStage.load(family_stage, root)
    keys = data.device_keys(family_stage, root)
    if max_devices:
        keys = keys[:max_devices]
    per_device = {}
    t0 = time.time()
    for path in data.canonical_paths(family_stage, root):
        block = f"b{int(data.file_attrs(path)['bloque']):03d}"
        mine = [k.split("/")[-1] for k in keys if k.startswith(block + "/")]
        for dev in data.iter_devices(path, "sims", keys=mine):
            per_device[dev.key] = device_sums(fs, dev)
            n = len(per_device)
            if n % 50 == 0 or n == len(keys):
                spent = time.time() - t0
                print(f"[sigma] {family_stage}: {n}/{len(keys)} devices, "
                      f"remaining ~{spent / n * (len(keys) - n) / 60:.1f} min", flush=True)
    seconds = time.time() - t0
    floor_r = reference_floor(family_stage, root)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for seed in seeds:
        train = data.split(keys, seed)["train"]
        rec = aggregate(per_device, train)
        rec.update({"family_stage": family_stage, "seed": seed, "reference_floor": floor_r,
                    "seconds_all_devices": seconds, "n_devices": len(per_device)})
        path = os.path.join(out_dir, f"{family_stage}_seed{seed}.json")
        persistence.write_json(os.path.dirname(path), os.path.basename(path), rec)
        paths.append(path)
    return paths


def run_sigma(family_stages=FAMILY_STAGES, max_devices=None):
    """Local step: sigma JSONs for every family-stage and seed (CPU); skips finished ones."""
    data.ensure_dataset(family_stages)
    out, t0 = [], time.time()
    for i, fs in enumerate(family_stages):
        print(f"[sigma] {fs} ({i + 1}/{len(family_stages)}), elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        done = [os.path.join(SIGMA_DIR, f"{fs}_seed{s}.json") for s in SEEDS]
        if max_devices is None and all(os.path.exists(p) for p in done):
            out.extend(done)
            continue
        out.extend(compute_sigma(fs, max_devices=max_devices))
    return out


# --- pilot ------------------------------------------------------------------------------------
def pilot_config(family_stage: str, seed: int, **overrides) -> training.TrainConfig:
    """E0's pilot: fixed weights, pure PINN, base architecture, Adam 5000 + L-BFGS 500 per voltage."""
    base = dict(family_stage=family_stage, seed=seed, strategy="fixed", k_s=0, fourier_features=0,
                width=512, depth=6, adam_iters=5000, lbfgs_iters=500, eval_marks=PILOT_EVAL_MARKS)
    base.update(overrides)
    return training.TrainConfig(**base)


def load_sigma(family_stage: str, seed: int, sigma_dir=SIGMA_DIR) -> dict:
    with open(os.path.join(sigma_dir, f"{family_stage}_seed{seed}.json")) as f:
        return json.load(f)


class Progress:
    """Prints how far a sequence of runs has gone and how long the rest should take.

    A run's share is its voltages done over its voltage budget (equilibrium counts as
    one more); the estimate extrapolates the wall time this session has spent so far.
    """

    def __init__(self, n_runs: int, label: str = "", stream=None):
        self.n_runs, self.label, self.stream = n_runs, label, stream
        self.done_runs, self.t0 = 0, time.time()

    def _print(self, msg):
        print(msg, file=self.stream, flush=True)

    def start_run(self, run_id, k_max):
        self.run_id, self.k_max, self.t_run = run_id, k_max, time.time()
        self._print(f"[{self.label}] run {self.done_runs + 1}/{self.n_runs}: {run_id} ({k_max} voltages max)")

    def update(self, st, voltage_done):
        steps_done = 0 if st["stage"] == "eq" and not voltage_done else (st["j"] + 1 if st["stage"] != "eq" else 1)
        frac_run = min(steps_done / (self.k_max + 1), 1.0)
        frac = (self.done_runs + frac_run) / self.n_runs
        spent = time.time() - self.t0
        eta = spent / frac - spent if frac > 0 else float("nan")
        where = "equilibrium" if st["stage"] == "eq" else f"V_{st['j']}"
        tag = "done" if voltage_done else f"{st.get('phase', '')} it {st.get('it', 0)}"
        self._print(f"[{self.label}] {100 * frac:5.1f}%  {self.run_id} {where} {tag}  "
                    f"elapsed {spent / 60:6.1f} min  remaining ~{eta / 60:6.1f} min")

    def end_run(self):
        self.done_runs += 1
        self._print(f"[{self.label}] finished {self.done_runs}/{self.n_runs} runs "
                    f"in {(time.time() - self.t0) / 60:.1f} min")


def run_one(cfg: training.TrainConfig, out_root=PILOT_DIR, sigma_dir=SIGMA_DIR, root=None,
            remote_url=None, releases=None, mirror_root=None, token=None, sync_every_seconds=600.0,
            keep_models_dir=None, progress=None):
    """Train (or resume) one run with persistence, then write and push its summary."""
    run_id = cfg.resolved_run_id()
    run_dir = os.path.join(out_root, run_id)
    origin = persistence.restore(run_dir, run_id, mirror_root, releases)
    state = persistence.load_checkpoint(run_dir) if origin else None
    setup = training.prepare(cfg, load_sigma(cfg.family_stage, cfg.seed, sigma_dir), root)
    persistence.write_json(run_dir, "run.json", {"run_id": run_id, "config": state["cfg"] if state else
                                                 asdict(cfg), "resumed_from": origin,
                                                 "devices": {k: len(v) for k, v in setup.split.items()}})
    sync = persistence.Syncer(run_dir, run_id, remote_url, releases, mirror_root, token, sync_every_seconds)

    if progress:
        progress.start_run(run_id, setup.k_max)

    def callback(st, voltage_done):
        if progress:
            progress.update(st, voltage_done)
        if voltage_done:
            persistence.append_jsonl(run_dir, "metrics.jsonl",
                                     {"stage": st["stage"], "j": st["j"], "seconds": st["seconds"],
                                      "history_tail": st["history"][-3:], "memory": _memory()})
        sync(st, voltage_done)

    state = _train_with_batch_rule(setup, state, callback, run_dir, keep_models_dir, root, sigma_dir)
    setup = state.pop("_setup", setup)
    summary = training.summarize(setup, state)
    persistence.write_json(run_dir, "summary.json", summary)
    sync.push(with_results=True)
    persistence.mirror(run_dir, mirror_root, run_id)
    if progress:
        progress.end_run()
    return summary


def _out_of_memory(exc) -> bool:
    return "RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower()


def _train_with_batch_rule(setup, state, callback, run_dir, models_dir, root, sigma_dir):
    """Experiments v05 batch rule: N_Q is halved only when the device runs out of memory.

    The run resumes from its last checkpoint with the smaller batch; every change is
    recorded in the state ("batch_events") and reported as a covariate.
    """
    from dataclasses import replace

    while True:
        try:
            state = training.train(setup, state, callback, models_dir=models_dir)
            state["_setup"] = setup
            return state
        except Exception as exc:  # noqa: BLE001
            if not _out_of_memory(exc) or setup.cfg.n_q <= 64:
                raise
            new_q = setup.cfg.n_q // 2
            print(f"[batch] out of memory at N_Q={setup.cfg.n_q}; resuming with N_Q={new_q}", flush=True)
            last = persistence.load_checkpoint(run_dir)
            events = (state or {}).get("batch_events", []) + [
                {"stage": (last or {}).get("stage"), "j": (last or {}).get("j"),
                 "from_n_q": setup.cfg.n_q, "to_n_q": new_q}]
            cfg = replace(setup.cfg, n_q=new_q)
            setup = training.prepare(cfg, load_sigma(cfg.family_stage, cfg.seed, sigma_dir), root)
            state = last
            if state is not None:
                state["batch_events"] = events
                state["cfg"]["n_q"] = new_q


def _memory():
    try:
        import jax

        stats = jax.devices()[0].memory_stats() or {}
        return {k: stats[k] for k in ("peak_bytes_in_use", "bytes_limit") if k in stats}
    except Exception:  # noqa: BLE001
        return {}


REHEARSAL = dict(width=512, depth=6, n_q=256, adam_iters=200, lbfgs_iters=20, max_devices=8,
                 max_voltages=4, eval_marks=(100, 200), checkpoint_every=50)


def run_pilot(family_stages=PILOT_FAMILY_STAGES, seeds=PILOT_SEEDS, root=None, mirror_root=None,
              push=True, overrides=None, rehearsal=False):
    """Remote step: E0's pilot runs, one after another, each resumable.

    `rehearsal` shrinks every run to REHEARSAL (the Kaggle GPU check before the A100) and
    tags its run ids, so a rehearsal never resumes into, or overwrites, a full run.
    `overrides` adjusts single fields on top. `root` defaults to $PINN_DPV_DATA and
    `mirror_root` to $PINN_DPV_MIRROR (the Drive folder on Colab).
    """
    mirror_root = mirror_root or os.environ.get("PINN_DPV_MIRROR")
    data.ensure_dataset(family_stages)
    shrink = dict(REHEARSAL) if rehearsal else {}
    shrink.update(overrides or {})
    token = persistence.github_token() if push else None
    remote_url = f"https://github.com/{REPO}.git" if push and token else None
    releases = persistence.GitHubReleases(REPO, token) if push and token else None
    out = []
    progress = Progress(len(family_stages) * len(seeds), "pilot")
    for fs in family_stages:
        for seed in seeds:
            cfg = pilot_config(fs, seed, **shrink)
            if rehearsal:
                cfg = pilot_config(fs, seed, run_id=f"rehearsal-{cfg.resolved_run_id()}", **shrink)
            out.append(run_one(cfg, root=root, remote_url=remote_url, releases=releases,
                               mirror_root=mirror_root, token=token, progress=progress))
    return out


# --- rehearsals (Kaggle) --------------------------------------------------------------------------
def smoke_sigma(family_stages=PILOT_FAMILY_STAGES, max_devices=60):
    """Rehearsal of `sigma`: the same code on the first devices of the pilot's family-stages."""
    data.ensure_dataset(family_stages)
    for fs in family_stages:
        compute_sigma(fs, max_devices=max_devices)
    return sigma_overview(family_stages)


def sigma_overview(family_stages):
    """floor / sigma^2 per component (median over seeds) and r, per family-stage."""
    out = {}
    for fs in family_stages:
        recs = [load_sigma(fs, s) for s in SEEDS if os.path.exists(os.path.join(SIGMA_DIR, f"{fs}_seed{s}.json"))]
        out[fs] = {"r_pp": recs[0]["reference_floor"]["r_pp"], "n_devices": recs[0]["n_devices"],
                   "floor_over_sigma2": {c: float(np.median([r["floor"][c] / r["sigma2"][c] for r in recs]))
                                         for c in recs[0]["sigma2"]}}
        print(fs, json.dumps(out[fs]), flush=True)
    return out


def initial_component_spread(summaries):
    """log10 of max/min of the initial scaled components per regime: E0's 'same order of
    magnitude' hypothesis holds for a regime only if this is below 1."""
    rows = []
    for s in summaries:
        for c in s.get("initial_components", []):
            v = np.array([x for x in c["s"].values() if x > 0])
            rows.append({"run_id": s["run_id"], "stage": c["stage"], "j": c["j"],
                         "log10_spread": float(np.log10(v.max() / v.min())) if v.size else float("nan"),
                         "components": c["s"]})
    return rows


def smoke_pilot(seeds=(0,), max_devices=60):
    """Rehearsal of `pilot` on a Kaggle GPU: sigma on a few devices, then the shrunk E0 runs,
    then the initial-component spread and the non-finite count of every run."""
    smoke_sigma(PILOT_FAMILY_STAGES, max_devices)
    summaries = run_pilot(PILOT_FAMILY_STAGES, seeds, push=False, rehearsal=True)
    spread = initial_component_spread(summaries)
    for row in spread:
        print(f"[E0] {row['run_id']} {row['stage']} V_{row['j']}: log10(max/min) = {row['log10_spread']:.1f}  "
              + " ".join(f"{k}={v:.2e}" for k, v in row["components"].items()), flush=True)
    persistence.write_json(PILOT_DIR, "rehearsal_e0.json",
                           {"summaries": [{k: v for k, v in s.items() if k != "devices"} for s in summaries],
                            "initial_component_spread": spread})
    return {"runs": len(summaries), "nonfinite": sum(s["nonfinite"] for s in summaries),
            "max_log10_spread": max((r["log10_spread"] for r in spread), default=float("nan"))}


# --- E0 screening (experiments v05) ----------------------------------------------------------
SCREEN_DIR = str(RESULTS_ROOT / "screen")
#: What every screening run shares: r09's base (Fourier features, log loss, equilibrium
#: start, continuation) at a short budget on a fixed pool of devices, seed 0.
SCREEN_BASE = dict(seed=0, strategy="fixed", k_s=0, fourier_features=64, fourier_scale=5.0,
                   width=256, depth=4, n_q=512, adam_iters=1000, lbfgs_iters=50, device_pool=400,
                   checkpoint_every=250, loss_form="log1p", qf_start="equilibrium", v_input=False)
#: One group per Kaggle session; each entry overrides SCREEN_BASE (family_stage required).
SCREEN_GROUPS = {
    "loss-init": [dict(family_stage="E1A"),
                  dict(family_stage="E1A", loss_form="linear"),
                  dict(family_stage="E1A", qf_start="random"),
                  dict(family_stage="E1A", loss_form="linear", qf_start="random")],
    "size": [dict(family_stage="E1A", width=128, depth=4),
             dict(family_stage="E1A", width=256, depth=6),
             dict(family_stage="E1A", width=512, depth=4),
             dict(family_stage="E1A", width=512, depth=6)],
    "voltage-interfaces": [dict(family_stage="E1A", v_input=True),
                           dict(family_stage="E3A"),
                           dict(family_stage="E3A", v_input=True),
                           dict(family_stage="E3A", loss_form="linear", qf_start="random")],
}
#: Each group above runs as two Kaggle sessions of two runs ("-a", "-b"); the halves pair
#: a heavy run with a light one where sizes differ.
_HALVES = {"size": ((0, 3), (1, 2))}
for _g, _runs in list(SCREEN_GROUPS.items()):
    _a, _b = _HALVES.get(_g, ((0, 1), (2, 3)))
    SCREEN_GROUPS[f"{_g}-a"] = [_runs[i] for i in _a]
    SCREEN_GROUPS[f"{_g}-b"] = [_runs[i] for i in _b]
# r09 data start (step 0, data only) against the zero start of r09, in E1A and E3A: one session.
SCREEN_GROUPS["start"] = [dict(family_stage="E1A", start="data"), dict(family_stage="E3A", start="data")]
#: Experiments v05: the eight combinations of r09's tools against spectral bias (Fourier
#: features at scale 5, the activation of zhang2025, one network per output), each with
#: fixed weights and with BRDR, in E1A; four sessions of four runs, each with two of the
#: heavier separate-network runs. The Fourier scale sweep follows, on the chosen combination.
for _w in ("fixed", "brdr"):
    for _half, _ff in (("a", 0), ("b", 64)):
        SCREEN_GROUPS[f"arch-{_w}-{_half}"] = [
            dict(family_stage="E1A", strategy=_w, fourier_features=_ff, activation=_act, separate=_sep)
            for _act in ("tanh", "tg") for _sep in (False, True)]
SCREEN_SHARE_CAP = 0.9  # hypothesis (ii): no component above 90 % of the gradient norm


def screen_config(overrides: dict) -> training.TrainConfig:
    base = dict(SCREEN_BASE)
    base.update(overrides)
    return training.TrainConfig(**base)


def screen_verdicts(summary: dict) -> dict:
    """E0's hypotheses (ii) and (iii) for one run, from its summary (experiments v05; v05
    removed hypothesis (i) on the initial scale of the components).

    (ii) is read only in regimes with at least two active components: with one, its share
    is 1 by definition. (iii) counts exact zeros after the first iteration of a regime.
    """
    early = [max(g["share"]) for g in summary.get("grad_shares", [])
             if g["it"] <= 500 and len(g["share"]) >= 2]
    return {"max_early_grad_share": max(early) if early else float("nan"),
            "h_ii_no_dominant_gradient": bool(early) and max(early) <= SCREEN_SHARE_CAP,
            "h_iii_finite_and_nonzero": summary.get("nonfinite", 1) == 0 and summary.get("zero_components", 1) == 0}


def run_screen(group: str, push: bool = False, groups=None):
    """Remote step (Kaggle GPU): one screening group, its runs one after another, resumable.

    sigma is recomputed on the same device pool the runs split (SG floor, r09 Eq. 23),
    so sigma never sees a validation or test device. Writes screen/<group>.json.
    """
    plan = (groups or SCREEN_GROUPS)[group]
    family_stages = sorted({o["family_stage"] for o in plan})
    data.ensure_dataset(family_stages)
    pools = {o.get("device_pool", SCREEN_BASE["device_pool"]) for o in plan}
    if len(pools) != 1:
        raise ValueError(f"a screening group shares one device pool, got {sorted(pools)}")
    pool = pools.pop()
    sigma_dir = os.path.join(SCREEN_DIR, f"sigma-pool{pool}")
    for fs in family_stages:
        if not os.path.exists(os.path.join(sigma_dir, f"{fs}_seed0.json")):
            compute_sigma(fs, seeds=(0,), out_dir=sigma_dir, max_devices=pool)
    token = persistence.github_token() if push else None
    remote_url = f"https://github.com/{REPO}.git" if push and token else None
    progress = Progress(len(plan), f"screen:{group}")
    rows = []
    for o in plan:
        cfg = screen_config(o)
        cfg = training.TrainConfig(**{**asdict(cfg), "run_id": f"screen-{cfg.resolved_run_id()}"})
        try:
            summ = run_one(cfg, out_root=SCREEN_DIR, sigma_dir=sigma_dir, remote_url=remote_url,
                           token=token, progress=progress)
        except Exception as exc:  # noqa: BLE001 -- one failed run never costs the group
            print(f"[screen] {cfg.run_id} FAILED: {exc!r}", flush=True)
            rows.append({"run_id": cfg.run_id, "overrides": o, "error": repr(exc)})
            continue
        row = {"run_id": summ["run_id"], "overrides": o, "seconds": summ["seconds"],
               "mean_e_pce_val": summ["mean_e_pce_val"], "mean_e_pce_test": summ["mean_e_pce_test"],
               "mean_e_pce_corners": summ["mean_e_pce_corners"],
               **{k: summ[k] for k in summ if k.startswith("mean_e_phi")},
               "nonfinite": summ["nonfinite"], "zero_components": summ["zero_components"],
               "batch_events": summ["batch_events"], "n_q_final": summ["n_q_final"],
               **screen_verdicts(summ)}
        rows.append(row)
        print("[screen] " + json.dumps(row), flush=True)
        persistence.write_json(SCREEN_DIR, f"{group}.json", {"group": group, "rows": rows})
    floor = {fs: load_sigma(fs, 0, sigma_dir) for fs in family_stages}
    out = {"group": group, "rows": rows,
           "floor_over_sigma2": {fs: {c: r["floor"][c] / r["sigma2"][c] for c in r["sigma2"] if r["sigma2"][c]}
                                 for fs, r in floor.items()},
           "r_pp": {fs: r["reference_floor"]["r_pp"] for fs, r in floor.items()}}
    persistence.write_json(SCREEN_DIR, f"{group}.json", out)
    return out


def selftest(paths=("tests/test_invariants_model.py",)) -> dict:
    """Run the target's own invariant tests where the screening runs (never on the operator's PC).

    Returns pytest's exit code and the tail of its report; never raises, so the
    rehearsal that follows still runs and both results come back together.
    """
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # One line per test as it finishes, so the session log shows progress (the model tests
    # compile a network each and take about 20 minutes together on a T4).
    proc = subprocess.Popen([sys.executable, "-u", "-m", "pytest", "-v", "-p", "no:cacheprovider", *paths],
                            cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print("[selftest] " + line, end="", flush=True)
        lines.append(line)
    proc.wait()
    tail = "".join(lines)[-6000:]
    out = {"returncode": proc.returncode, "tail": tail}
    persistence.write_json(SCREEN_DIR, "selftest.json", out)
    return out


def smoke_screen(group: str, tests: bool = True):
    """Rehearsal of `screen`: the invariant tests, then the group's first run at a tiny budget."""
    # The model tests read E1A and E3A whatever the group screens.
    data.ensure_dataset(sorted({"E1A", "E3A"} | {o["family_stage"] for o in SCREEN_GROUPS[group]}))
    result = {"selftest": selftest() if tests else None}
    first = dict(SCREEN_GROUPS[group][0])
    first.update(adam_iters=20, lbfgs_iters=2, device_pool=40, max_voltages=2, grad_share_marks=(1, 10))
    result["screen"] = run_screen(group, groups={group: [first]})
    return result


def screen_table(screen_dir=SCREEN_DIR):
    """E0 screening rows from every returned screen/<group>.json, sorted by validation e_PCE."""
    rows = []
    for path in sorted(glob.glob(os.path.join(screen_dir, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        if "rows" in rec:
            rows += [dict(r, group=rec["group"]) for r in rec["rows"]]
    return sorted(rows, key=lambda r: (r.get("error") is not None, r.get("mean_e_pce_val", float("inf"))))


def _tools(overrides: dict) -> tuple:
    """(Fourier, activation, separate) of a screening run: r09's tools against spectral bias."""
    ff = overrides.get("fourier_features", SCREEN_BASE["fourier_features"])
    return bool(ff), overrides.get("activation", "tanh"), bool(overrides.get("separate", False))


def screen_conclusion(rows) -> dict:
    """E0's success criterion (experiments v05) computed from the screening rows.

    The combination of tools is the lowest mean validation e_PCE over fixed weights and
    BRDR among the combinations meeting (ii) and (iii) with both weightings.
    """
    ok = [r for r in rows if "error" not in r]

    def holds(r):
        return r["h_ii_no_dominant_gradient"] and r["h_iii_finite_and_nonzero"]

    passes = [r for r in ok if holds(r)]
    base = {r["overrides"]["family_stage"]: r for r in ok if set(r["overrides"]) == {"family_stage"}}
    controls = [r for r in ok if r["overrides"].get("loss_form") == "linear"
                or r["overrides"].get("qf_start") == "random"]
    best = min(passes, key=lambda r: r["mean_e_pce_val"]) if passes else None
    combos = {}
    for r in ok:
        if "activation" in r["overrides"]:
            combos.setdefault(_tools(r["overrides"]), {})[r["overrides"].get("strategy", "fixed")] = r
    means = {t: (by["fixed"]["mean_e_pce_val"] + by["brdr"]["mean_e_pce_val"]) / 2
             for t, by in combos.items()
             if set(by) >= {"fixed", "brdr"} and holds(by["fixed"]) and holds(by["brdr"])}
    tools = min(means, key=means.get) if means else None
    return {"base_holds": {fs: holds(r) for fs, r in base.items()},
            "a_control_violates": any(not holds(r) for r in controls),
            "chosen": best["run_id"] if best else None,
            "chosen_tools": dict(zip(("fourier", "activation", "separate"), tools)) if tools else None,
            "tools_mean_e_pce_val": {"/".join(map(str, t)): v for t, v in means.items()},
            "n_runs": len(rows), "n_failed": len(rows) - len(ok), "n_passing": len(passes)}


# --- report -------------------------------------------------------------------------------------
def budget_rule(by_mark: dict, r_pp: float):
    """Experiments v01, E0: the first iteration mark after which mean validation e_PCE
    drops by no more than r/2 per 1000 iterations; None if it is still dropping faster
    at the last mark (the cap did not suffice)."""
    marks = sorted((int(k), float(v)) for k, v in by_mark.items())
    for (m0, e0), (m1, e1) in zip(marks, marks[1:]):
        if (e0 - e1) / (m1 - m0) * 1000.0 <= r_pp / 2:
            return m0
    return None


def run_report():
    """Local step: the E0 tables from the sigma JSONs and the pilot summaries."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(SIGMA_DIR, "*_seed*.json"))):
        with open(path) as f:
            rec = json.load(f)
        r = rows.setdefault(rec["family_stage"], {"sigma2": {}, "floor_ratio": {}, "r_pp": None})
        r["r_pp"] = rec["reference_floor"]["r_pp"]
        for c, v in rec["sigma2"].items():
            r["sigma2"].setdefault(c, []).append(v)
            r["floor_ratio"].setdefault(c, []).append(rec["floor"][c] / v if v else float("nan"))
    sigma_table = {fs: {"sigma2_median": {c: float(np.median(v)) for c, v in r["sigma2"].items()},
                        "floor_ratio_median": {c: float(np.median(v)) for c, v in r["floor_ratio"].items()},
                        "r_pp": r["r_pp"]} for fs, r in rows.items()}
    runs = []
    for path in sorted(glob.glob(os.path.join(PILOT_DIR, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        row = {k: s[k] for k in ("run_id", "family_stage", "seed", "seconds", "mean_e_pce_val",
                                 "mean_e_pce_test", "mean_e_pce_val_by_mark") if k in s}
        # A rehearsal checks the machinery at a scale that answers nothing about E0.
        row["rehearsal"] = str(s.get("run_id", "")).startswith("rehearsal-")
        runs.append(row)
    budget = {}
    for fs in sorted({r["family_stage"] for r in runs if not r["rehearsal"]}):
        marks = [r["mean_e_pce_val_by_mark"] for r in runs
                 if r["family_stage"] == fs and not r["rehearsal"] and r.get("mean_e_pce_val_by_mark")]
        if not marks or fs not in sigma_table:
            continue
        mean = {k: float(np.mean([m[k] for m in marks if k in m])) for k in marks[0]}
        budget[fs] = {"mean_e_pce_val_by_mark": mean, "r_pp": sigma_table[fs]["r_pp"],
                      "iterations": budget_rule(mean, sigma_table[fs]["r_pp"]), "seeds": len(marks)}
    out = {"sigma": sigma_table, "pilot": runs, "budget": budget}
    persistence.write_json(REPORT_DIR, "e0.json", out)
    return out
