"""Turning measurements into an answer: who wins each dimension, and when nobody does.

Pure stdlib on purpose. The harness that produces the numbers needs torch; deciding
what the numbers say does not, so this can be read and tested without one.

The same table serves both settings. A synthetic sweep and a trained run over real
data measure different instruments and answer the same question — where does each
implementation hold — so they share these rules and this shape rather than each
inventing its own.
"""

from __future__ import annotations

import math
from typing import Iterable

# Which direction is better for a dimension. `None` means the value is descriptive
# and no winner is declared from it — a parameter count is worth reporting and is
# not a contest.
HIGHER, LOWER, DESCRIPTIVE = "higher", "lower", None

TIE = "indistinguishable"
NOT_APPLICABLE = "not applicable"
UNRESOLVED = "no verdict"

#: Below this many repetitions the rule below inverts, so it is not applied at all.
#: One repetition gives a dispersion of zero, hence a threshold of zero, hence a
#: winner on every row from a bare difference — the opposite of what the threshold
#: exists to do, at exactly the scale where the protection matters most. Two is no
#: better: a standard error from two samples is a number, not an estimate.
MINIMUM_REPETITIONS = 3


def resolved(entry: dict | None) -> bool:
    """Did this measurement come from enough repetitions to be contested?"""
    return entry is not None and int(entry.get("n", 1)) >= MINIMUM_REPETITIONS


def standard_error(entry: dict) -> float:
    """Spread of the mean, not of the samples: what several seeds were run to learn."""
    n = max(int(entry.get("n", 1)), 1)
    return float(entry.get("stdev", 0.0)) / math.sqrt(n)


def decide(baseline: dict | None, new: dict | None, better: str | None) -> dict:
    """Who wins this dimension, or nobody.

    The rule is deliberately plain and is a heuristic, not a significance test: a
    winner is declared only when the means differ by more than their combined standard
    error. Anything closer is reported as indistinguishable.

    That threshold is the whole reason several seeds are run. Comparing two bare means
    always produces a winner — every measurement differs from every other at enough
    decimal places — and calling that a result is reading noise out loud.
    """
    if baseline is None or new is None:
        return {"winner": NOT_APPLICABLE, "margin": None,
                "reason": "one side could not be driven into the common reduction"}
    if better is DESCRIPTIVE:
        return {"winner": None, "margin": None,
                "reason": "descriptive: reported, not contested"}

    if not (resolved(baseline) and resolved(new)):
        # The row is still printed, with its numbers: a pilot has to run the same
        # code the campaign will, and hiding the table would make it a different
        # program. What is withheld is the verdict, not the measurement.
        return {"winner": UNRESOLVED,
                "margin": float(new["mean"]) - float(baseline["mean"]),
                "repetitions": min(int(baseline.get("n", 1)), int(new.get("n", 1))),
                "reason": f"fewer than {MINIMUM_REPETITIONS} repetitions: the "
                          f"dispersion is zero, so the threshold is zero and any "
                          f"difference would win. This is a point estimate."}

    left, right = float(baseline["mean"]), float(new["mean"])
    difference = right - left
    combined = standard_error(baseline) + standard_error(new)

    if abs(difference) <= combined:
        return {"winner": TIE, "margin": difference, "threshold": combined,
                "reason": "the means are closer than their combined standard error"}

    ahead = "new" if (difference > 0) == (better == HIGHER) else "baseline"
    return {"winner": ahead, "margin": difference, "threshold": combined,
            "reason": f"{better} is better and the gap exceeds the combined standard error"}


def judge(comparison: Iterable[dict]) -> list[dict]:
    """Attach a verdict to every row, leaving the measurements untouched."""
    judged = []
    for row in comparison:
        if not isinstance(row, dict):
            continue
        judged.append({**row, "verdict": decide(row.get("baseline"), row.get("new"),
                                                row.get("better", DESCRIPTIVE))})
    return judged


def tally(judged: Iterable[dict]) -> dict:
    """Where each side wins, so the reader is not left to count rows by eye."""
    counts = {"new": [], "baseline": [], TIE: [], NOT_APPLICABLE: [], UNRESOLVED: []}
    for row in judged:
        winner = row["verdict"]["winner"]
        if winner in counts:
            counts[winner].append(row.get("dimension"))
    return counts


def render(judged: Iterable[dict], reduction: dict) -> str:
    """The table. Every number carries the reduction that produced it, in the header,
    because a number read without its reduction gets misquoted."""
    judged = list(judged)
    head = (f"setting={reduction.get('setting','?')}  "
            f"backbone={reduction.get('backbone','none')}  "
            f"data={reduction.get('dataset','synthetic')}  "
            f"seeds={reduction.get('seeds','?')}  "
            f"revision={reduction.get('revision','?')}")
    lines = [head, "", f"{'dimension':<34}{'baseline':>16}{'new':>16}  {'winner'}"]

    def cell(entry):
        if entry is None:
            return "—"
        return f"{entry['mean']:.4g} ± {standard_error(entry):.2g}"

    for row in judged:
        verdict = row["verdict"]
        winner = verdict["winner"] or "—"
        lines.append(f"{str(row.get('dimension','?')):<34}"
                     f"{cell(row.get('baseline')):>16}"
                     f"{cell(row.get('new')):>16}  {winner}")
        if verdict["winner"] in (TIE, NOT_APPLICABLE, UNRESOLVED):
            lines.append(f"{'':<34}{'':>32}  ({verdict['reason']})")

    counts = tally(judged)
    if counts[UNRESOLVED]:
        lines.insert(1, f"!! {len(counts[UNRESOLVED])} row(s) below "
                        f"{MINIMUM_REPETITIONS} repetitions: point estimates, not verdicts")
    lines += ["", f"new wins on {len(counts['new'])}, baseline on {len(counts['baseline'])}, "
                  f"{len(counts[TIE])} indistinguishable, {len(counts[NOT_APPLICABLE])} not applicable, "
                  f"{len(counts[UNRESOLVED])} unresolved"]
    if counts["new"]:
        lines.append("  new:      " + ", ".join(map(str, counts["new"])))
    if counts["baseline"]:
        lines.append("  baseline: " + ", ".join(map(str, counts["baseline"])))
    return "\n".join(lines)
