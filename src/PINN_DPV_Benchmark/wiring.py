"""How each declared arm reaches the method's own library — no arm reimplements a term.

Every PINN arm is one `training.TrainConfig` differing only in `strategy` (r05 Sec. 6,
experiments v01 E1): same network, scales, data, collocation and iteration budget.
`tpe` first searches lambda with `search.run` on seed 0's validation split and then
trains with those weights fixed. `b0` is the data-only regressor of `b0.fit`.
"""

from __future__ import annotations

from PINN_DPV import b0, search, steps, training

PINN_ARMS = ("fixed", "ntk", "brdr", "ema", "tpe", "gummel")


def pinn_config(arm: str, family_stage: str, seed: int, **overrides) -> training.TrainConfig:
    """The full-scale configuration of one PINN arm; the E0 pilot is the `fixed` arm."""
    if arm not in PINN_ARMS:
        raise ValueError(f"unknown PINN arm {arm!r}; expected one of {PINN_ARMS}")
    return steps.pilot_config(family_stage, seed, strategy=arm, eval_marks=(), **overrides)


#: The TPE arm's lambda search (Eq. 19), resumable on SQLite storage.
tpe_search = search.run


def b0_fit(xi_train, pce_train, seed: int, **kw):
    """B0, the data-only baseline: PCE regressed on xi alone."""
    return b0.fit(xi_train, pce_train, seed, **kw)
