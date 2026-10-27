"""Declares the experiment this benchmark package runs.

The harness stage copies this file in once, verbatim, only after a
comparison has been accepted, and substitutes nothing into it — every value
below is prefilled empty, and stays that way until a person fills it in as
the work happens.

Each of the five blocks is prefilled empty on purpose, and only at this level.
Emptiness at the block level is unambiguous: no repository ever means "I
measured that `distribution` is empty" before a single shard has run. One
level down that stops being true — a replication run can measure that
`perEnvironment` is genuinely `[]`, and a template that had already written
`[]` there would be indistinguishable from that measurement. So the shape of
what belongs inside each block is described in a comment beside it, never
written as a value, and nothing here invents a value for `arms`, `search`,
`report` or `distribution` — those are read off the work as it happens, not
guessed at scaffold time.

`revision` and `premises` are declared alongside this package, not inside it:
they are facts about the METHOD this benchmark compares against a baseline,
not about the comparison, so they live in `__implementation__` under
`src/<Package>/__init__.py` (`authored_package_init`'s own template) instead.
`__levels__`, `__steps__` and `__records__` — the target's own first-flow
declarations — moved there too, for the identical reason.
"""

__benchmark__ = {
    # One entry per arm, naming the sections of the proposal it exercises,
    # e.g.:
    #     "arms": {
    #         "baseline": {"sections": ["3.1"]},
    #         "proposed": {"sections": ["3.1", "3.4"]},
    #     }
    "arms": {
        # Experiments v01, E1: the reference arm and the five adaptive strategies of
        # r05 Sec. 6, all on the same network, scales, data and iteration budget;
        # B0 is the data-only baseline of Sec. 8.
        "fixed": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "ntk": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "brdr": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "ema": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "tpe": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "gummel": {"sections": ["2", "3", "4", "5", "6", "8", "9"]},
        "b0": {"sections": ["7", "8", "9"]},
    },

    # What a declared search has to say about itself before its chosen value
    # means anything, e.g.:
    #     "search": {
    #         "what": "the regularization weight",
    #         "requiredScale": {"epochs": 20, "seeds": 3},
    #         "role": "validation split",
    #         "tieRule": "smallest value within one standard error of the best",
    #     }
    #
    # One key is OPTIONAL and asked of nobody, and it is the one every
    # later reading hangs off: `record`, the path -- relative to the
    # product folder -- of the artefact this search writes, e.g.:
    #     "record": "Results/search-record.json"
    # Leave it out and `search.recordFound` answers `null` forever: a
    # ticked `@record` witness has nothing behind it, a leveled
    # `@record:level` one reaches no rung, and `probe` keeps answering
    # `search-first` because a declared `requiredScale` cannot be
    # satisfied by a record nothing was told to look for. The forge never
    # guesses the filename; you name it, and it only looks.
    #
    # A second key is likewise OPTIONAL: `currentWhen`, a
    # dotted path into the record's own file naming where it wrote down the
    # identity of the code that produced it, e.g.:
    #     "currentWhen": "evidence.sourcesDigest"
    # `distribution.currentWhen`'s own idiom, one level up from a shard.
    # Declare it and a found record only counts as evidence while the value
    # there still matches the code as it stands; a record left behind by
    # code this repository has moved past reads as unmeasured rather than as
    # a rung reached. Leave it out -- the default -- and a found record is
    # trusted on arrival alone, exactly as before this key existed. The
    # forge never guesses the field, the same way it never guesses which
    # measurements are poolable: you name it, and it only compares.
    "search": {},

    # Which functions render tables and figures, which produce conclusions,
    # and which way each measured dimension wins, e.g.:
    #     "report": {
    #         "renderers": ["tables.render"],
    #         "conclusions": ["tables.conclusion"],
    #         "figures": ["figures.curves"],
    #         "dimensions": {"accuracy": "higher", "seconds": "lower"},
    #     }
    "report": {
        "renderers": ["PINN_DPV.steps.run_report", "PINN_DPV.steps.screen_table"],
        "conclusions": ["PINN_DPV.steps.budget_rule", "PINN_DPV.steps.screen_conclusion"],
        "figures": [],
        "dimensions": {"e_pce_pp": "lower", "seconds": "lower"},
        # Experiments v04, E0 (ii): the loss terms of r08 Eq. (15) and the dimension that
        # carries each one's share of the gradient norm (training.gradient_shares).
        "components": {"terms": ["Poisson", "n", "p", "BCn", "BCp", "int_phi", "int_n", "int_p", "dat"],
                       "share": "grad_shares"},
    },

    # What a run split across shards has to say about itself: the axis a
    # shard is a subset of, and which measurements are poolable, measured
    # per environment, vary per run, or must agree identically across every
    # shard, e.g.:
    #     "distribution": {
    #         "axis": "seed",
    #         "poolable": ["accuracy"],
    #         "perEnvironment": [],
    #         "perRun": [],
    #         "identicalAcrossShards": ["datasetSize"],
    #     }
    #
    # One further key is OPTIONAL and asked of nobody: `currentWhen`, a
    # dotted path into a shard's own `shard.json` stamp naming where that
    # shard wrote down the identity of the code that produced it, e.g.:
    #     "currentWhen": "evidence.sourcesDigest"
    # Declare it and a returned shard only counts as evidence while the
    # value there still matches the code as it stands; a shard that arrived
    # from code this repository has moved past reads as unmeasured rather
    # than as a step reached. Leave it out -- the default -- and a shard is
    # trusted on arrival alone, exactly as before this key existed. The
    # forge never guesses the field, the same way it never guesses which
    # measurements are poolable: you name it, and it only compares.
    #
    # A second, sibling key is likewise OPTIONAL: `shardsRoot`, naming
    # where a split campaign's returned shards land, relative to this
    # repository's own root (an absolute path is honoured as given), e.g.:
    #     "shardsRoot": "campaign-001/shards"
    # Declare it once and every command that reads a position section --
    # not only the ones that carry their own `--shards` flag -- measures a
    # `@shard` witness against that same directory automatically; a `@shard`
    # item ticked on real evidence stays checkable everywhere, not only at
    # the one command that was told where to look for it. An explicit
    # `--shards <dir>`, where the command accepts one, still overrides this
    # for that single invocation. Leave it out -- the default -- and a
    # command with no `--shards` flag of its own reads a `@shard` witness as
    # unmeasured, exactly as before this key existed; the forge never
    # invents a directory a repository never named.
    "distribution": {},

    # The dotted module and function that actually pull the target's runtime
    # in — the same two values `generate-job --run-module`/`--run-function`
    # already require, so nothing new is asked for. `probe`'s harness
    # resolution and `introspect`'s liveness check both read this rather than
    # assuming a filename of the forge's own choosing, e.g.:
    #     "entry": {
    #         "module": "Example_Method_Benchmark.benchmark",
    #         "function": "run",
    #     }
    "entry": {"module": "PINN_DPV.steps", "function": "run_pilot"},
}
