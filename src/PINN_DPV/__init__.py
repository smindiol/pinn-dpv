"""Reference implementation of the PINN-DPV formulation.

Each module declares the sections and equations it implements in
`__provenance__`, and every invariant listed there has a matching
test under tests/.
"""

import jax

# r08 Sec. 8: double precision throughout, as in dPV, because delta^2 reaches
# 1e-9. Set before any array is created by this package.
jax.config.update("jax_enable_x64", True)

__all__ = []

__implementation__ = {
    # The managed revision this declaration is bound to -- a filename under
    # proposals/, asked for by the flow, never invented here.
    # When another skill shares this package, name one revision per
    # document label instead, so each skill reads its own lineage:
    #     "revision": {"proposal": "<filename>", "<label>": "<filename>"},
    "revision": {
        "proposal": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r08.md",
        "experiments": "experiments-protocolo-experimental-del-benchmark-pinn-sobre-dpv-v04.md",
    },

    # What kind of prediction the protocol assumes, over which unit,
    # by which metric and in which direction it is judged, e.g.:
    #     "premises": {
    #         "prediction": "a class label per subject",
    #         "statisticalUnit": "subject",
    #         "metric": "balancedAccuracy",
    #         "direction": "higher",
    #     }
    "premises": {
        "prediction": "the power conversion efficiency of each device, derived from the PINN potentials through the J-V curve exactly as dPV computes it",
        "statisticalUnit": "seed: the mean e_PCE over the test devices of one seed, per family-stage",
        "metric": "e_PCE, absolute PCE error against dPV in percentage points (IQM over seeds)",
        "direction": "lower",
    },
}

__backend__ = "jax"

# The ordered ladder of rungs a position-section step can reach, entirely
# in this repository's own words -- the forge holds no rung name of its
# own, only the arithmetic that compares two of these names by position
# (see `impl_position.level_index`). A step earns a rung by naming this
# file's own ladder explicitly on its witness (`` `@rehearsal:level <job>`
# `` in `AGREED.md`'s position section); a step with no `:level` marker is
# two-state and never reads this list at all. Left empty until named -- a
# repository whose position items are entirely two-state needs no ladder
# here, and one is never invented on its behalf. A second, independent
# top-level literal, held apart from `__implementation__` above for the
# same reason `resolve_levels_declaration`'s own docstring states.
#
# Example (a repository with no remote service at all still has a ladder):
#     __levels__ = ["local", "cluster"]
__levels__ = ["cpu-local", "kaggle-gpu", "colab-a100"]

# A callable this repository's own code can run, isolated, under this
# repository's own venv -- named and resolved statically by the forge
# (module + function, never imported here), then imported and called
# inside the target's own interpreter, never the forge's. A third,
# independent top-level literal, held apart from `__implementation__` for
# the identical reason `__levels__` is: see `resolve_steps_declaration`'s
# own docstring. Left empty until a step exists -- a repository with
# nothing local to run in isolation needs none, and one is never invented
# on its behalf.
#
# Each entry carries `module` and `function`, and one further key that is
# asked of every step and defaulted for none: `produces`, the list of
# path roots -- relative to the product folder -- that this step and only
# this step writes into. `step` snapshots the product folder before and
# after every run and reports what changed on each side of those roots,
# which is the only way it can tell a step that returned having written
# nothing from one that produced its whole output, or a step that stayed
# in its own tree from one that wrote into a neighbour's. Leave it out
# and BOTH readings are switched off for that step: `verify` says so, per
# step, in `undeclaredProduces`, and nothing here defaults a root on your
# behalf -- the forge never guesses which work belongs to which step.
#
# The pattern that key exists to make possible, and the one worth
# scaffolding on the first day rather than discovering after a run: a
# flow has two kinds of step, and each one owns a notebook.
#
#   - a step that COMPUTES -- it orchestrates this package's own
#     library, writes data, and draws nothing;
#   - a step that DRAWS -- it reads what the computing step left on
#     disk and renders tables, figures and conclusions.
#
# Each names its own notebook among its `produces` roots, so a pilot
# executes both of them AS notebooks. That is the point of the split:
# the artefact that is later handed to a worker elsewhere is a
# notebook, so a pilot that exercises anything else has not tested
# what gets sent.
#
# Collapsing the two into one step is a legitimate design and nothing
# in this forge refuses it -- but what it costs is written down here
# so the choice is made rather than defaulted into. A figure can no
# longer be redrawn without paying for the computation behind it
# again; and whichever half is left outside a notebook is the half
# the pilot never exercised in the shape it will be sent in. `verify`
# says so, per step, in `undeclaredStepNotebooks`, and it never
# refuses -- it names what the absence costs and leaves the design
# yours.
#
# And one more key, asked of every step and defaulted for none:
# `placement`, which says WHERE the step runs once the flow leaves
# rehearsal scale -- "local" on this machine, or "remote" on a worker.
# A remote one also names two more: the `job` folder that carries it
# and the `service` that folder lives under. Nothing else ties a step
# to a job, and the forge deliberately does not invent that link --
# which work goes through which job folder is this repository's
# layout, not the forge's to guess. The service is yours to name for
# a harder reason: the forge may read a service name to walk a
# directory and must reduce it to a count before returning anything,
# so nothing there can name one, and it cannot discover one either --
# adapters register lazily, so the registry is empty until somebody
# names one.
#
# It is a DECLARATION and not a decision recorded somewhere else, and
# the reason is worth having on the first day. That routing gets
# decided in conversation, and a conversation lands in the ledger
# under `.implementation/` -- free prose, in a directory `.gitignore`
# excludes. So a decision left there can be neither consumed (nothing
# parses a sentence into a route) nor travelled with (a clone
# receives none of it), while the walk that has to act on it runs
# from a clone. The ledger keeps the REASON, with its numbers; this
# key carries the FACT, which is the half a machine reads.
#
# Leave it out and the walk cannot route that step: it knows the step
# exists, what it produces and where it sits in the order, and not
# whether it runs here or elsewhere. `verify` says so, per step, in
# `undeclaredPlacement`, and it never refuses -- but nothing defaults
# it either. Routing an unrouted step by convention is how a run
# measured in days lands somewhere nobody chose, so `probe`'s own
# `flowActs` reports that step as `blocked` and names what is missing
# rather than picking for you.
#
# Example -- two steps, each owning its own notebook, each saying
# where it runs:
#     __steps__ = {
#         "computation": {
#             "module": "Example_Method.steps",
#             "function": "run_computation",
#             # Where this step sits in the order, and what it
#             # consumes from the steps above it. `advances` is the
#             # position item this step produces evidence for;
#             # `reads` is empty here because nothing precedes it,
#             # and an empty list is an answer -- it is what tells a
#             # remote rehearsal there is no upstream output to wait
#             # for.
#             "advances": 1,
#             "reads": [],
#             # Writes data and draws nothing. Its notebook is what
#             # the pilot executes and what a worker elsewhere would
#             # be handed.
#             "produces": ["Results/computation",
#                          "Notebooks/computation.ipynb"],
#             # The expensive half, so it goes to a worker at full
#             # scale -- and it names the job folder that carries it
#             # there.
#             "placement": "remote",
#             "job": "computation",
#             "service": "the-service-you-send-to",
#         },
#         "rendering": {
#             "module": "Example_Method.steps",
#             "function": "run_rendering",
#             "advances": 2,
#             # What it consumes, named: this is the link a remote
#             # rehearsal reads to refuse before opening a notebook
#             # whose inputs are not there yet, naming each missing
#             # root and which step writes it.
#             "reads": ["Results/computation"],
#             # Reads what "computation" left behind and renders. It
#             # writes no data of its own, so its notebook is the
#             # whole of what it produces -- and it can be re-run on
#             # its own, without paying for the computation again.
#             "produces": ["Notebooks/rendering.ipynb"],
#             # Seconds of drawing, and the artefact a person reads.
#             # Sending it to a worker would put the thing somebody
#             # has to read behind a download and buy nothing, so it
#             # stays here and names no job.
#             "placement": "local",
#         },
#     }
__steps__ = {
    # Algorithm A1 (iv)-(v) over every family-stage and seed: output scales,
    # sigma_i^2 and L_i^floor (Eq. 3, 23) and the reference floor r (Eq. 21).
    # Read by every training run. Runs on a Kaggle GPU (or inside the Colab
    # central notebook), never on the operator's laptop, by the operator's choice.
    "sigma": {
        "module": "PINN_DPV.steps",
        "function": "run_sigma",
        "advances": 1,
        "reads": [],
        "produces": ["Results/sigma", "Notebooks/sigma.ipynb"],
        "placement": "remote",
        "job": "sigma",
        "service": "kaggle",
    },
    # Experiments v04, E0: the screening of r08's formulation on Kaggle T4s, one
    # group per account and session (loss-init, arch, size, voltage-interfaces).
    # Computes its own sigma on the screening pool, so it reads nothing upstream.
    "screen": {
        "module": "PINN_DPV.steps",
        "function": "run_screen",
        "advances": 1,
        "reads": [],
        "produces": ["Results/screen"],
        "placement": "remote",
        "job": "screen",
        "service": "kaggle",
    },
    # E0's pilot: E1A and E3A, seeds 0-2, fixed weights, full J-V curve.
    # Rehearsed small on a Kaggle GPU (rehearsal=True), then run at full scale
    # by the operator on the Colab A100 with the same notebook.
    "pilot": {
        "module": "PINN_DPV.steps",
        "function": "run_pilot",
        "advances": 2,
        "reads": ["Results/sigma"],
        "produces": ["Results/pilot", "Notebooks/pilot.ipynb"],
        "placement": "remote",
        "job": "pilot",
        "service": "kaggle",
    },
    # The E0 tables: sigma ratios, r, and validation e_PCE per iteration mark.
    "report": {
        "module": "PINN_DPV.steps",
        "function": "run_report",
        "advances": 3,
        "reads": ["Results/sigma", "Results/pilot"],
        "produces": ["Results/report", "Notebooks/report.ipynb"],
        "placement": "local",
    },
}

# A target-chosen name mapped to the record it addresses -- a leveled
# `@record:level <name>` witness in AGREED.md's position section reaches
# exactly one entry here, deriving its rung through the identical
# arithmetic the benchmark's own `search` block already uses
# (`impl_position._record_scale_level`). A fourth, independent top-level
# literal, held apart from `__implementation__` for the identical reason
# `__levels__`/`__steps__` are: see `resolve_records_declaration`'s own
# docstring. Left empty until a record is named -- a repository whose
# leveled `@record` witness stays the bare, operand-less form needs no
# entry here, and one is never invented on its behalf.
#
# Example:
#     __records__ = {
#         "main": {
#             "path": "product/results.json",
#             "requiredScale": {"seeds": 3},
#         },
#     }
__records__: dict = {}
