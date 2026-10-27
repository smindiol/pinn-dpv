"""Bayesian optimization of the fixed weights with TPE (Optuna).

Eq. (19): lambda* = argmin over log10 lambda in [-3, 3]^{N_L} of the mean |eta_PCE - eta_ref|
over the validation subset after a short training, centred on lambda_i = 1 (the
baseline) around the scales sigma_i of Eq. (23). The protocol fixes 40 trials,
TPE seed 0, and 10 % of the per-voltage budget per trial, once per family-stage.

The study is stored in SQLite inside the run directory, so an interrupted search
resumes from the trials it already finished.
"""

from __future__ import annotations

import dataclasses
import math

import optuna

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md",
    "sections": ["6"],
    "equations": ["19"],
    "invariants": [
        "tpe_space_is_log10_minus3_to_3",
        "tpe_resumes_finished_trials",
    ],
}

LOG10_LOW, LOG10_HIGH = -3.0, 3.0
N_TRIALS = 40
TPE_SEED = 0
BUDGET_FRACTION = 0.1


def short_config(cfg, fraction: float = BUDGET_FRACTION):
    """The trial's training budget: a fraction of the per-voltage Adam and L-BFGS iterations."""
    return dataclasses.replace(cfg, adam_iters=max(1, math.ceil(cfg.adam_iters * fraction)),
                               lbfgs_iters=math.ceil(cfg.lbfgs_iters * fraction))


def suggest_lambdas(trial, names):
    return {n: 10.0 ** trial.suggest_float(f"log10_lambda_{n}", LOG10_LOW, LOG10_HIGH) for n in names}


def run(objective_fn, names, storage_path: str, study_name: str, n_trials: int = N_TRIALS,
        seed: int = TPE_SEED):
    """Run (or resume) the TPE study; `objective_fn(lambdas) -> validation mean e_PCE`.

    Returns the best weights and the study. Already finished trials count toward n_trials.
    """
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < n_trials:
        study.optimize(lambda t: objective_fn(suggest_lambdas(t, names)), n_trials=n_trials - done)
    best = {n: 10.0 ** study.best_params[f"log10_lambda_{n}"] for n in names}
    return best, study
