"""The retired promotion gate (scripts/roll_window.py).

The gate is gone; the file is kept because it is the record of why it could
not work, and this module pins the parts of that record a future edit could
quietly lose:

* it must not run a gate -- no retrain, no staging, no exit code that reads
  as a promotion decision;
* it must name what replaced it, so a reader who lands here from an old
  runbook or workflow step is sent somewhere useful;
* it must keep BOTH structural reasons the gate failed. The first (it scored
  one member of four) is the obvious one; the second (a refit-through-latest
  incumbent makes the held-forward slice in-sample) is the one that says the
  question itself went away, and it is the reason the replacement asks a
  different question rather than a fixed version of this one.
"""
from __future__ import annotations

import scripts.roll_window as roll_window


def test_it_runs_nothing():
    """No retrain, no staging, no promotion decision -- there is no gate left."""
    for gone in ("_execute", "_retrain_candidate", "_ensemble_val_log_loss",
                 "decide_promotion", "PROMOTION_THRESHOLD", "PROMOTION_MARGIN"):
        assert not hasattr(roll_window, gone), gone


def test_main_prints_the_notice_and_exits_zero(capsys):
    assert roll_window.main([]) == 0
    out = capsys.readouterr().out
    assert "RETIRED" in out
    assert roll_window.REPLACEMENT in out


def test_the_retired_flags_still_reach_the_notice(capsys):
    """An old invocation must get the explanation, not an argparse error that
    says nothing about what happened to the gate."""
    assert roll_window.main(["--execute"]) == 0
    assert "RETIRED" in capsys.readouterr().out


def test_the_notice_names_what_replaced_it_and_how_to_run_it():
    assert "revalidate_recipe.py" in roll_window.NOTICE
    assert "--check-staleness" in roll_window.NOTICE
    assert "recipe_revalidation.json" in roll_window.NOTICE


def test_the_docstring_keeps_both_reasons_the_gate_could_not_work():
    doc = roll_window.__doc__
    # (1) it scored one member of the deployed hybrid
    assert "one member of four" in doc
    assert "simulator" in doc and "blend" in doc
    # (2) the held-forward slice was in-sample for a refit incumbent
    assert "in-sample" in doc.lower()
    assert "refit_through_latest" in doc
    # and why that second one retired the question rather than posing a bug
    assert "no candidate-versus-incumbent choice" in doc


def test_the_docstring_still_says_nothing_is_promoted_automatically():
    """The one thing the gate got right, and the property the replacement
    inherits: a bar that no longer clears goes to a human."""
    doc = " ".join(roll_window.__doc__.split())
    assert "it never promoted anything automatically" in doc
    assert "reported to a human, not acted on" in doc
