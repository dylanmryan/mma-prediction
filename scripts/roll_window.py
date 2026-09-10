"""RETIRED 2026-09-09. Replaced by `scripts/revalidate_recipe.py`.

This file is kept, rather than deleted, because it is the record of a real
lesson: a gate that was carefully built, honestly reported that it could not
do its job, and turned out to be asking a question the project had stopped
asking. Deleting it would delete the lesson with it. Nothing here runs a
gate any more -- `main` prints the notice below and exits 0.

WHAT IT WAS. A walk-forward promotion gate. Once 150 graded prospective
fights had accumulated since the model's data cutoff it would retrain the
5-seed torch ensemble with a pushed-forward cutoff, score both that
candidate and the committed incumbent on the same newest-two-years
held-forward slice, and promote if the candidate beat the incumbent by more
than 0.002 log-loss. On promotion it staged artifacts into models/torch and
made no git write; a human re-ran the refit recipe, reviewed the diff and
committed by hand. The weekly Action only ever ran its --dry-run.

WHY IT COULD NOT WORK. Two independent structural faults, both of which it
detected itself and aborted on rather than reporting a biased number:

1. **It gated on one member of four.** Since SP2.2 the winner probability is
   a blend of the torch ensemble and a 5-seed XGBoost ensemble; since SP3 the
   deployed scorer is the hybrid, which takes its winner from that blend and
   its method, finish round and joint distribution from a Monte Carlo
   simulator built on two more models the gate never touched. A number about
   the torch ensemble alone was half of the winner probability and none of
   the joint distribution the app and the prediction records show. It read
   the artifacts on disk to say so, so the abort could not go stale as the
   deployment changed.

2. **Its comparison was in-sample for the incumbent.** Under the deployment
   recipe the harness selected (`refit_through_latest`, see
   models/walkforward/refit_decision_b1.json) the incumbent trains on every
   fight through the latest data date. Its training cutoff therefore equals
   the latest date, and the "held-forward" two-year slice is data it trained
   on -- which biases the comparison against any candidate. It read
   models/torch/metrics_val.json and aborted when the incumbent's mode was
   refit_through and its train_through reached into the slice.

WHY THE QUESTION WENT AWAY. Fault 2 is not a bug to fix; it is the deployment
recipe working. The weekly Action retrains every member on all data whenever
new fights arrive, so there is no candidate-versus-incumbent choice left to
make -- the fresh fit IS the incumbent. A promotion gate needs two rival
models and this project deploys one recipe, continuously refit.

WHAT REPLACED IT, AND WHY THAT IS THE RIGHT QUESTION. The gap this leaves
open is not "should something else ship?" but "does what already ships still
clear the bars it was justified by, now that the table has grown?" The
harness evidence behind the recipe was computed on a table ending
2026-08-08; the models train through 2026-09-05 and further every week, which
is exactly what `stale_harness_warning` fires on in the train scripts.
`scripts/revalidate_recipe.py` re-runs the deployed configuration and its
paired comparisons on the current table and re-applies the bars recorded in
the committed decision artifacts, and its `--check-staleness` mode -- the one
the weekly Action now runs in place of this script's dry run -- reports how
far the deployed models have run ahead of that evidence.

One thing this gate got right is worth carrying forward: it never promoted
anything automatically, and neither does its replacement. A bar that no
longer clears is reported to a human, not acted on.
"""
from __future__ import annotations

import argparse

REPLACEMENT = "scripts/revalidate_recipe.py"

NOTICE = f"""\
scripts/roll_window.py is RETIRED (2026-09-09) and does nothing.

The promotion gate it ran could not work, for two reasons it detected and
aborted on itself: it retrained one member of the four-model deployed hybrid,
and its held-forward slice was in-sample for a refit-through-latest incumbent.
The second is the deployment recipe working as selected, not a bug -- the
weekly Action refits every member on all data whenever new fights arrive, so
there is no candidate and no incumbent to choose between.

What replaced it: {REPLACEMENT}

  python {REPLACEMENT} --check-staleness   # how stale is the recipe's evidence?
  python {REPLACEMENT} --plan              # the harness runs it would make
  python {REPLACEMENT}                     # re-measure every recorded bar

It re-runs the deployed configuration and its paired comparisons on the
current feature table, re-applies the bars recorded in the committed decision
artifacts, writes models/walkforward/recipe_revalidation.json, and exits
non-zero only when a bar is no longer met. Like this gate, it promotes
nothing automatically: a bar that no longer clears is a finding for a human.

The full account of what this gate was and why it could not work is this
file's module docstring. It is kept deliberately -- read it before building
another gate.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The retired flags are still accepted so an old invocation -- a stale
    # workflow step, a note in a runbook -- gets the notice rather than an
    # argparse error that says nothing about what happened to the gate.
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--predictions-dir", default=None, help=argparse.SUPPRESS)
    parser.parse_args(argv)
    print(NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
