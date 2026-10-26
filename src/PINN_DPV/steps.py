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

import numpy as np

from dataclasses import asdict

from . import data, evaluation, loss, persistence, scaling, training

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r05.md",
    "sections": ["7", "8", "9"],
    "equations": ["3", "21", "23"],
    "invariants": [
        "sigma_per_seed_uses_only_training_devices",
        "pilot_config_matches_protocol",
    ],
}

PRODUCT = "PINN-DPV"
SIGMA_DIR = f"{PRODUCT}/Results/sigma"
PILOT_DIR = f"{PRODUCT}/Results/pilot"
REPORT_DIR = f"{PRODUCT}/Results/report"
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


def run_sigma():
    """Local step: sigma JSONs for every family-stage and seed (CPU)."""
    for fs in FAMILY_STAGES:
        compute_sigma(fs)


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


def run_one(cfg: training.TrainConfig, out_root=PILOT_DIR, sigma_dir=SIGMA_DIR, root=None,
            remote_url=None, releases=None, mirror_root=None, token=None, sync_every_seconds=600.0,
            keep_models_dir=None):
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

    def callback(st, voltage_done):
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
    return summary


def _memory():
    try:
        import jax

        stats = jax.devices()[0].memory_stats() or {}
        return {k: stats[k] for k in ("peak_bytes_in_use", "bytes_limit") if k in stats}
    except Exception:  # noqa: BLE001
        return {}


def run_pilot(family_stages=PILOT_FAMILY_STAGES, seeds=PILOT_SEEDS, root=None, mirror_root=None,
              push=True, overrides=None):
    """Remote step: E0's pilot runs, one after another, each resumable.

    `overrides` shrinks the configuration for a rehearsal (e.g. a Kaggle GPU smoke run).
    """
    token = persistence.github_token() if push else None
    remote_url = f"https://github.com/{REPO}.git" if push and token else None
    releases = persistence.GitHubReleases(REPO, token) if push and token else None
    out = []
    for fs in family_stages:
        for seed in seeds:
            cfg = pilot_config(fs, seed, **(overrides or {}))
            out.append(run_one(cfg, root=root, remote_url=remote_url, releases=releases,
                               mirror_root=mirror_root, token=token))
    return out


# --- report -------------------------------------------------------------------------------------
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
        runs.append({k: s[k] for k in ("run_id", "family_stage", "seed", "seconds", "mean_e_pce_val",
                                       "mean_e_pce_test", "mean_e_pce_val_by_mark") if k in s})
    persistence.write_json(REPORT_DIR, "e0.json", {"sigma": sigma_table, "pilot": runs})
    return {"sigma": sigma_table, "pilot": runs}
