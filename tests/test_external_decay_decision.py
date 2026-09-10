"""The decision artifact for the snapshot decay (scripts/external_decay_decision.py).

`mma.decay` owns the rule's arithmetic and `tests/test_decay.py` pins it. What
is worth pinning HERE is everything that could make the artifact state one rule
and apply another, or compare two things that are not comparable:

* the thresholds are READ OUT of the plan's own quoted rule text, so a rule
  edited without touching this script changes what the script applies -- and a
  rule this script can no longer parse is a loud failure, not a fallback to
  some constant;
* `check_pairing` refuses an unpaired comparison instead of producing a
  plausible-looking delta between two different experiments;
* `scored_rows_per_fold` counts the rows the JOINT metric is actually averaged
  over, which is neither the fold's `n` nor its `n_method`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.external_decay_decision as d
from mma import decay


RULE_STUB = """## 5. The rule, stated mechanically

Some prose about deltas.

> **REMOVE the columns iff all three hold:**
>
> 1. `d_pooled_joint < 0.01` -- do no harm; **and**
> 2. `keep_gain_recent < 0.001` -- not worth a tenth of the bar; **and**
> 3. `d_pooled_winner <= 0.000346` -- within sigma_seed.
>
> Otherwise **KEEP**.

More prose.

## 6. Something else

Not part of the rule.
"""


# --- the rule is quoted, and its numbers are read from the quote ----------

def test_rule_text_is_the_plans_own_section_and_stops_at_the_next_heading(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Head\n\n## 4. Metric\n\nx\n\n" + RULE_STUB)
    rule = d.rule_text(plan)
    assert rule.startswith(d.RULE_HEADING)
    assert "REMOVE the columns iff" in rule
    assert "Something else" not in rule


def test_thresholds_are_read_out_of_the_rule_text():
    assert d.thresholds_from(RULE_STUB) == {
        "joint_bar": 0.01, "recent_bar": 0.001, "sigma_seed": 0.000346,
    }


def test_a_rule_whose_numbers_moved_moves_what_the_script_applies():
    """The point of reading rather than retyping: edit the rule, and the
    applied thresholds follow without touching this script."""
    assert d.thresholds_from(RULE_STUB.replace("< 0.001", "< 0.002"))["recent_bar"] == 0.002


def test_a_rule_this_script_cannot_parse_fails_loudly():
    """Silently falling back to a constant is exactly how an artifact comes to
    state one bar and apply another."""
    with pytest.raises(SystemExit, match="could not read recent_bar"):
        d.thresholds_from(RULE_STUB.replace("keep_gain_recent", "keep_recent"))


def test_the_committed_plan_states_the_thresholds_the_helper_defaults_to():
    """The pre-registration and mma.decay's constants must not drift apart."""
    thresholds = d.thresholds_from(d.rule_text())
    assert thresholds["joint_bar"] == decay.JOINT_BAR
    assert thresholds["recent_bar"] == decay.RECENT_BAR
    assert thresholds["sigma_seed"] == pytest.approx(decay.SIGMA_SEED, abs=1e-9)


# --- pairing --------------------------------------------------------------

def config(drop, **over) -> dict:
    return {"config": {
        "candidate": "hybrid", "seeds": "0,1,2,3,4", "n_feature_rows": 11238,
        "features_max_date": "2026-08-08", "train_start": None, "half_life": None,
        "fixed_budget_from": None, "drop_columns": list(drop), **over,
    }}


def test_a_correctly_paired_candidate_reports_what_it_removed():
    candidate = config(list(d.FLAGS) + list(d.GROUPS["EXTERNAL"]))
    out = d.check_pairing(candidate, config(d.FLAGS), ("EXTERNAL",))
    assert out["removed_columns"] == sorted(d.GROUPS["EXTERNAL"])
    assert out["flags_held_out_by_both"] == sorted(d.FLAGS)


def test_an_incumbent_that_is_not_the_deployed_drop_set_is_rejected():
    candidate = config(list(d.FLAGS) + list(d.GROUPS["EXTERNAL"]))
    with pytest.raises(SystemExit, match="expected the five leak-guard flags"):
        d.check_pairing(candidate, config(d.FLAGS[:2]), ("EXTERNAL",))


def test_a_candidate_dropping_the_wrong_columns_is_rejected():
    candidate = config(list(d.FLAGS) + list(d.GROUPS["NOTICE"]))
    with pytest.raises(SystemExit, match="candidate drops"):
        d.check_pairing(candidate, config(d.FLAGS), ("EXTERNAL",))


def test_a_candidate_run_with_different_seeds_is_rejected():
    candidate = config(list(d.FLAGS) + list(d.GROUPS["EXTERNAL"]), seeds="5,6,7,8,9")
    with pytest.raises(SystemExit, match="seeds:"):
        d.check_pairing(candidate, config(d.FLAGS), ("EXTERNAL",))


def test_a_candidate_run_on_a_different_table_is_rejected():
    candidate = config(list(d.FLAGS) + list(d.GROUPS["EXTERNAL"]), n_feature_rows=9000)
    with pytest.raises(SystemExit, match="n_feature_rows:"):
        d.check_pairing(candidate, config(d.FLAGS), ("EXTERNAL",))


def test_the_flags_are_never_part_of_what_a_group_removes():
    """They stay out of both matrices either way; removing them again would
    make the comparison about something else."""
    for group in d.GROUPS:
        assert not set(d.GROUPS[group]) & set(d.FLAGS)


# --- fold weights ---------------------------------------------------------

def test_scored_rows_are_neither_the_fold_n_nor_n_method():
    """A decision with an unknown round is unscorable for the joint metric,
    but its method IS known -- so n_method overcounts it."""
    features = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-02-01",  # decision, scored
            "2024-03-01",  # ko with a round, scored
            "2024-04-01",  # ko with no round, NOT scored (method known)
            "2024-05-01",  # method unknown, not scored
            "2025-02-01",  # decision, scored
        ]),
        "y_method": ["decision", "ko", "ko", None, "decision"],
        "y_finish_round": [None, "1", None, None, None],
    })
    weights = d.scored_rows_per_fold(features)
    assert weights["winner"]["2024"] == 4
    assert weights["joint"]["2024"] == 2      # not 4 (n) and not 3 (n_method)
    assert weights["joint"]["2025"] == 1


def test_scored_rows_reproduce_the_committed_reports_fold_row_counts():
    """The winner weights are the fold `n` the harness itself recorded, which
    is the check that make_folds is being replayed correctly here."""
    report = d.load(d.INCUMBENT)
    features = d.features_as_reported(pd.read_parquet(d.FEATURES), report)
    weights = d.scored_rows_per_fold(features)
    for year, block in report["folds"].items():
        assert weights["winner"][year] == block["n"], year
        # And the joint weight is strictly smaller, since unscorable rows exist.
        assert 0 < weights["joint"][year] <= block["n_method"]


def test_the_fold_weights_do_not_move_when_the_feature_table_grows():
    """The last fold is unbounded above, so every fight the weekly refresh
    appends lands in it. Weighing a recorded fold by a count taken from a
    table the harness never saw would silently reweight the comparison."""
    report = d.load(d.INCUMBENT)
    features = pd.read_parquet(d.FEATURES)
    before = d.scored_rows_per_fold(d.features_as_reported(features, report))

    grown = pd.concat([features, features.tail(20).assign(
        date=pd.Timestamp("2026-12-31"))], ignore_index=True)
    after = d.scored_rows_per_fold(d.features_as_reported(grown, report))

    assert before == after


def test_a_feature_table_that_no_longer_contains_the_reported_one_is_loud():
    report = d.load(d.INCUMBENT)
    features = pd.read_parquet(d.FEATURES)
    with pytest.raises(SystemExit, match="was computed on"):
        d.features_as_reported(features.iloc[:-100], report)


# --- the decision routes both seed sets, not just the first ---------------

def results_stub(consolidated="KEEP") -> dict:
    return {"R_all": {"verdict": consolidated}}


def fresh_stub(**verdicts) -> dict:
    ids = {group: cid for cid, (_, groups) in d.CANDIDATES.items()
           if len(groups) == 1 for group in groups}
    return {"decided": True,
            "candidates": {ids[g]: {"verdict": v} for g, v in verdicts.items()}}


def test_a_removal_confirmed_at_both_seed_sets_is_deployed():
    out = d.decision_block({"CONTEXT": "REMOVE"}, results_stub(),
                           fresh_stub(CONTEXT="REMOVE"))
    assert out["groups_to_remove"] == ["CONTEXT"]
    assert out["outcome"] == "remove CONTEXT"


def test_a_removal_the_fresh_seeds_do_not_confirm_deploys_nothing():
    """The seeds 0-4 verdict is a proposal; the artifact must not read as a
    decision to change the model when the confirmation did not agree."""
    out = d.decision_block({"CONTEXT": "REMOVE"}, results_stub(),
                           fresh_stub(CONTEXT="AMBIGUOUS"))
    assert out["confirmed_per_group"] == {"CONTEXT": "AMBIGUOUS"}
    assert out["groups_to_remove"] == []
    assert "nothing is deployed" in out["outcome"]
    assert out["fresh_seed_agreement"]["agree"] is False


def test_a_keep_needs_no_fresh_seed_run_to_stand():
    out = d.decision_block({"EXTERNAL": "KEEP", "NOTICE": "KEEP"}, results_stub(),
                           {"decided": False, "reason": "not run"})
    assert out["confirmed_per_group"] == {"EXTERNAL": "KEEP", "NOTICE": "KEEP"}
    assert out["outcome"] == "keep every snapshot-dependent column"


def test_a_mixed_outcome_names_both_halves():
    out = d.decision_block({"EXTERNAL": "KEEP", "NOTICE": "REMOVE", "CONTEXT": "REMOVE"},
                           results_stub(), fresh_stub(NOTICE="REMOVE", CONTEXT="KEEP"))
    assert out["groups_to_remove"] == ["NOTICE"]
    assert out["ambiguous_groups"] == ["CONTEXT"]
    assert out["outcome"].startswith("remove NOTICE; CONTEXT is AMBIGUOUS")


def test_the_committed_artifact_deploys_nothing_and_says_why():
    """The result of this experiment, pinned so a later edit that turns it
    back into a deployment has to be deliberate."""
    artifact = json.loads((d.OUT).read_text())
    assert artifact["decision"]["confirmed_per_group"] == {
        "EXTERNAL": "KEEP", "NOTICE": "KEEP", "CONTEXT": "AMBIGUOUS",
    }
    assert artifact["decision"]["groups_to_remove"] == []
    assert artifact["decision"]["fresh_seed_agreement"]["run"] is True
