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
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r05.md",
    "sections": ["7", "8", "9"],
    "equations": ["3", "21", "23"],
    "invariants": [
        "sigma_per_seed_uses_only_training_devices",
        "pilot_config_matches_protocol",
        "budget_rule_stops_where_drop_falls_below_half_r",
        "initial_component_spread_is_log10_max_over_min",
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

    state = training.train(setup, state, callback, models_dir=keep_models_dir)
    summary = training.summarize(setup, state)
    persistence.write_json(run_dir, "summary.json", summary)
    sync.push(with_results=True)
    persistence.mirror(run_dir, mirror_root, run_id)
    if progress:
        progress.end_run()
    return summary


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
