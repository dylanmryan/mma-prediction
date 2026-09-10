"""The out-of-fold model-vs-market path in scripts/build_odds_benchmark.py.

The July 2026 benchmark compared a model that had never seen 2021+ against
the market on 2021+ fights. The deployed model is refit through the latest
event, so that comparison is no longer out of sample for the model and
recomputing it as-is would flatter the model against an honest market. The
honest recomputation pairs the walk-forward OUT-OF-FOLD predictions -- where
each fold's model never saw that fold's fights -- with the same devigged
lines, joined on the shared fight id alone.

These tests exercise that path on synthetic frames: no network, no kagglehub,
no model artifacts, and nothing here recomputes a committed benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.build_odds_benchmark as bob
from mma.walkforward import make_folds, pool

ROOT = Path(__file__).resolve().parents[1]
OOF_BENCHMARK = ROOT / "models" / "market_benchmark_oof.json"


def _features(n_per_year: int = 3, years=range(2015, 2027)) -> pd.DataFrame:
    """A date-sorted stand-in for features.parquet with the columns used here."""
    rows = []
    for year in years:
        for i in range(n_per_year):
            rows.append(
                {
                    "fight_id": f"{year}{i:012d}ab",
                    "date": pd.Timestamp(year=year, month=1 + i, day=5),
                    "y_winner": float((year + i) % 2),
                    "swapped": bool(i % 2),
                }
            )
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def _dump(pooled: pd.DataFrame, p=None) -> dict:
    n = len(pooled)
    if p is None:
        p = np.linspace(0.2, 0.8, n)
    return {
        "name": "synthetic",
        "n": n,
        "fight_id": [str(v) for v in pooled["fight_id"]],
        "fold_year": [int(v) for v in pooled["fold_year"]],
        "y_winner": [float(v) for v in pooled["y_winner"]],
        "p_winner": [float(v) for v in p],
    }


# --------------------------------------------------------------------------
# Rebuilding the pooled row order the prediction dump is written in
# --------------------------------------------------------------------------


def test_pooled_frame_holds_only_fold_year_rows():
    features = _features()
    pooled = bob.pooled_frame(features)
    assert pooled["date"].min() >= pd.Timestamp("2018-01-01")
    # the last fold absorbs everything from 2025 on, 2026 included
    assert pooled["date"].max() >= pd.Timestamp("2026-01-01")
    assert set(pooled["fold_year"]) == set(range(2018, 2026))
    assert int((pooled["fold_year"] == 2025).sum()) == 6  # 2025 + 2026


def test_pooled_frame_row_order_is_walkforward_pools():
    """The rebuilt frame must be the rows the harness pooled.

    The dump is paired by fight id, so the row ORDER no longer carries the
    pairing -- but the rebuilt frame still has to be the same row SET, or the
    join would silently drop fights the harness did in fact evaluate.
    """
    features = _features()
    folds = make_folds(features["date"])
    harness, _ = pool(
        features, [(f.eval, {"winner": np.zeros(int(f.eval.sum()))}) for f in folds]
    )
    rebuilt = bob.pooled_frame(features)
    assert list(rebuilt["fight_id"]) == list(harness["fight_id"])


def test_pooled_frame_scores_no_fight_twice():
    pooled = bob.pooled_frame(_features())
    assert not pooled["fight_id"].duplicated().any()


# --------------------------------------------------------------------------
# Pairing the dump to those rows
# --------------------------------------------------------------------------


def test_attach_oof_predictions_pairs_by_fight_id():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    out = bob.attach_oof_predictions(pooled, dump)
    assert list(out["fight_id"]) == list(pooled["fight_id"])
    assert list(out["model_p_a"]) == dump["p_winner"]


def test_attach_oof_predictions_ignores_the_dump_row_order():
    """An id-keyed pairing is a join, so a reordered dump is the same dump."""
    pooled = bob.pooled_frame(_features())
    straight = bob.attach_oof_predictions(pooled, _dump(pooled))
    dump = _dump(pooled)
    order = list(reversed(range(dump["n"])))
    for key in ("fight_id", "fold_year", "y_winner", "p_winner"):
        dump[key] = [dump[key][i] for i in order]
    reversed_dump = bob.attach_oof_predictions(pooled, dump)
    pd.testing.assert_frame_equal(straight, reversed_dump)


def test_attach_oof_predictions_pairs_through_a_table_that_has_since_grown():
    """The bug this pairing exists to kill: a daily scrape added 52 fights to
    the feature table and the committed dump -- positional, and 52 rows short
    -- became unpairable. Keyed on the id, a grown table just means the dump
    covers fewer of its rows."""
    grown = _features(n_per_year=4)
    small = grown[grown["date"] < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    dump = _dump(bob.pooled_frame(small))
    out = bob.attach_oof_predictions(bob.pooled_frame(grown), dump)
    assert len(out) == dump["n"] < len(bob.pooled_frame(grown))
    assert list(out["fight_id"]) == dump["fight_id"]


def test_attach_oof_predictions_rejects_a_fight_the_table_does_not_evaluate():
    """The pairing failure that IS a failure: the dump names a fight the
    current table's walk-forward never scores, so the two describe different
    fight sets and no metric over the join would mean anything."""
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["fight_id"][0] = "ffffffffffffffff"
    with pytest.raises(ValueError, match="not walk-forward evaluation rows"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_dump_with_no_fight_id():
    """The legacy positional shape is refused outright rather than paired by
    position, which is the fragility being removed."""
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    del dump["fight_id"]
    with pytest.raises(ValueError, match="fight_id"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_ragged_lists():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["p_winner"] = dump["p_winner"][:-1]
    with pytest.raises(ValueError, match="n"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_duplicated_fight_in_the_dump():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["fight_id"][1] = dump["fight_id"][0]
    with pytest.raises(ValueError, match="duplicate"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_disagreeing_outcome():
    """The id says which fight; this says the table still records the same
    result for it. A relabelled outcome would silently rescore the model."""
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["y_winner"][0] = 1.0 - dump["y_winner"][0]
    with pytest.raises(ValueError, match="y_winner"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_disagreeing_fold_year():
    """A fight that has moved between folds is being scored by a different
    model than the dump's, even though the id still matches."""
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["fold_year"][0] = 2019 if dump["fold_year"][0] != 2019 else 2020
    with pytest.raises(ValueError, match="fold_year"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_degenerate_probabilities():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["p_winner"][0] = 0.0
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        bob.attach_oof_predictions(pooled, dump)


def test_check_dump_reproduces_report_accepts_the_matching_report():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    y = oof["y_winner"].to_numpy(dtype=float)
    p = oof["model_p_a"].to_numpy(dtype=float)
    from mma.evaluate import accuracy, brier_score, log_loss

    report = {
        "pooled": {
            "n": len(oof),
            "winner_log_loss": round(log_loss(y, p), 4),
            "accuracy": round(accuracy(y, p), 4),
            "brier": round(brier_score(y, p), 4),
        }
    }
    assert bob.check_dump_reproduces_report(oof, report)["n"] == len(oof)


def test_check_dump_reproduces_report_rejects_a_foreign_report():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    report = {"pooled": {"n": len(oof), "winner_log_loss": 0.1,
                         "accuracy": 0.9, "brier": 0.05}}
    with pytest.raises(ValueError, match="does not reproduce"):
        bob.check_dump_reproduces_report(oof, report)


# --------------------------------------------------------------------------
# The id-only join and the intersection
# --------------------------------------------------------------------------


def _aligned(fight_ids) -> pd.DataFrame:
    n = len(fight_ids)
    return pd.DataFrame(
        {
            "fight_id": list(fight_ids),
            "market_implied_fights_a": np.linspace(0.3, 0.7, n),
            "market_implied_fights_b": 1.0 - np.linspace(0.3, 0.7, n),
            "decimal_fights_a": np.linspace(1.5, 3.0, n),
            "decimal_fights_b": np.linspace(3.0, 1.5, n),
            "align_method": ["id"] * n,
            "n_books": [4] * n,
        }
    )


def test_join_odds_by_id_keeps_exactly_the_intersection():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    kept = list(oof["fight_id"])[::2]
    aligned = _aligned(kept + ["ffffffffffffffff"])  # an odds fight we never scored
    merged = bob.join_odds_by_id(oof, aligned)
    assert list(merged["fight_id"]) == kept
    assert len(merged) == len(kept)


def test_join_odds_by_id_is_keyed_on_the_id_not_the_row_order():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    aligned = _aligned(list(oof["fight_id"]))
    shuffled = aligned.sample(frac=1.0, random_state=0).reset_index(drop=True)
    straight = bob.join_odds_by_id(oof, aligned)
    scrambled = bob.join_odds_by_id(oof, shuffled)
    pd.testing.assert_frame_equal(straight, scrambled)


def test_join_odds_by_id_rejects_duplicate_ids_on_the_odds_side():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    ids = list(oof["fight_id"])[:4]
    aligned = _aligned(ids + ids[:1])
    with pytest.raises(ValueError, match="duplicate fight_id"):
        bob.join_odds_by_id(oof, aligned)


def test_join_odds_by_id_rejects_duplicate_ids_on_the_prediction_side():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    doubled = pd.concat([oof, oof.iloc[:1]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="duplicate fight_id"):
        bob.join_odds_by_id(doubled, _aligned(list(oof["fight_id"])))


def test_join_odds_by_id_requires_the_key():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    with pytest.raises(ValueError, match="no fight_id column"):
        bob.join_odds_by_id(oof.drop(columns=["fight_id"]),
                            _aligned(list(oof["fight_id"])))


def test_join_odds_by_id_survives_the_corner_convention_mapping():
    """The joined odds still map onto features' a/b via the `swapped` flag."""
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    merged = bob.to_features_convention(
        bob.join_odds_by_id(oof, _aligned(list(oof["fight_id"])))
    )
    swapped = merged["swapped"].to_numpy(dtype=bool)
    expected = np.where(swapped, merged["market_implied_fights_b"],
                        merged["market_implied_fights_a"])
    assert np.allclose(merged["market_implied_a"], expected)
    assert np.allclose(merged["market_implied_a"] + merged["market_implied_b"], 1.0)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_winner": [1.0, 0.0, 1.0, 0.0],
            "model_p_a": [0.6, 0.4, 0.55, 0.45],
            "deployed_p_a": [0.9, 0.1, 0.9, 0.1],
            "market_implied_a": [0.7, 0.3, 0.65, 0.35],
            "market_implied_b": [0.3, 0.7, 0.35, 0.65],
            "decimal_a": [1.5, 3.0, 1.6, 2.8],
            "decimal_b": [2.8, 1.4, 2.5, 1.5],
            "date": pd.to_datetime(["2019-01-01", "2021-06-01", "2022-01-01", "2024-01-01"]),
        }
    )


def test_compare_block_delta_is_model_minus_market():
    block = bob.compare_block(_scored())
    for key in ("accuracy", "log_loss", "brier"):
        assert block["delta_model_minus_market"][key] == round(
            block["model"][key] - block["market"][key], 4
        )
    assert block["n_fights"] == 4
    assert len(block["calibration"]["model"]) == 10


def test_compare_block_reads_the_named_model_column():
    """The in-sample diagnostic and the headline differ only in this column."""
    df = _scored()
    oof = bob.compare_block(df, model_col="model_p_a")
    deployed = bob.compare_block(df, model_col="deployed_p_a")
    assert oof["market"] == deployed["market"]  # same fights, same lines
    assert deployed["model"]["log_loss"] < oof["model"]["log_loss"]


def test_roi_sweep_reads_the_named_model_column():
    df = _scored()
    assert bob.roi_sweep(df, model_col="deployed_p_a") != bob.roi_sweep(df)


def test_fold_year_coverage_splits_matched_from_unmatched():
    pooled = bob.pooled_frame(_features())
    oof = bob.attach_oof_predictions(pooled, _dump(pooled))
    matched = oof.iloc[::2]
    coverage = bob.fold_year_coverage(oof, matched)
    assert set(coverage) == {str(y) for y in range(2018, 2026)}
    assert sum(row["n_walkforward"] for row in coverage.values()) == len(oof)
    assert sum(row["n_odds_matched"] for row in coverage.values()) == len(matched)
    for row in coverage.values():
        assert 0.0 <= row["odds_coverage"] <= 1.0


def test_july_comparison_reports_the_direction_of_the_change():
    july = json.loads(bob.FROZEN_BENCHMARK.read_text())["headline_2021_plus"]
    july_gap = july["delta_model_minus_market"]["log_loss"]
    narrower = bob.july_comparison(
        {"n_fights": 100, "model": {"log_loss": 0.60}, "market": {"log_loss": 0.60},
         "delta_model_minus_market": {"log_loss": july_gap - 0.01}}
    )
    assert narrower["direction"] == "narrowed"
    assert narrower["change_in_gap"] == round(-0.01, 4)
    wider = bob.july_comparison(
        {"n_fights": 100, "model": {"log_loss": 0.70}, "market": {"log_loss": 0.60},
         "delta_model_minus_market": {"log_loss": july_gap + 0.02}}
    )
    assert wider["direction"] == "widened"
    assert wider["change_in_gap"] == round(0.02, 4)


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------

oof_only = pytest.mark.skipif(
    not OOF_BENCHMARK.exists(), reason="out-of-fold benchmark not built yet"
)


@pytest.fixture(scope="module")
def artifact():
    return json.loads(OOF_BENCHMARK.read_text())


@oof_only
def test_artifact_has_its_provenance(artifact):
    assert {
        "computed_on", "describes", "odds_dataset", "note", "provenance",
        "alignment", "intersection", "headline_out_of_fold",
        "out_of_fold_2021_plus", "in_sample_diagnostic",
        "comparison_with_frozen_july_artifact", "odds_coverage_by_fold_year",
    } <= set(artifact)
    provenance = artifact["provenance"]
    assert provenance["predictions"].startswith("models/walkforward/preds/")
    assert provenance["walkforward_report"].startswith("models/walkforward/")
    assert len(provenance["deployed_model_version"]) == 12
    assert provenance["fold_years"][0] == 2018
    assert provenance["deployed_training_recipe"]["mode"] == "refit_through"


@oof_only
def test_artifact_intersection_is_a_subset_of_the_walkforward_rows(artifact):
    intersection = artifact["intersection"]
    assert 0 < intersection["n_intersection"] <= intersection["n_walkforward_rows"]
    assert intersection["n_intersection"] <= intersection["n_odds_aligned_fights"]
    assert artifact["headline_out_of_fold"]["n_fights"] == intersection["n_intersection"]
    assert "fight_id" in intersection["join_key"]


@oof_only
def test_artifact_confirms_the_trap_it_exists_to_avoid(artifact):
    """The deployed model trained on every fight in the comparison set, and
    the in-sample diagnostic is duly better than the out-of-fold headline."""
    assert artifact["intersection"][
        "deployed_training_window_covers_the_whole_intersection"
    ] is True
    in_sample = artifact["in_sample_diagnostic"]["model"]["log_loss"]
    out_of_fold = artifact["headline_out_of_fold"]["model"]["log_loss"]
    assert in_sample < out_of_fold
    assert "NOT the headline" in artifact["in_sample_diagnostic"]["warning"]


@oof_only
def test_artifact_honesty_gate_holds_out_of_fold(artifact):
    delta = artifact["headline_out_of_fold"]["delta_model_minus_market"]["log_loss"]
    assert delta > -0.02, (
        f"model beats market by {-delta:.4f} log-loss out of fold -- "
        "investigate alignment/leakage before trusting this artifact"
    )


@oof_only
def test_artifact_market_log_loss_is_sharp(artifact):
    for cut in ("headline_out_of_fold", "out_of_fold_2021_plus"):
        assert 0.5 < artifact[cut]["market"]["log_loss"] < 0.72, cut


@oof_only
def test_artifact_calibration_tables_cover_every_fight(artifact):
    for cut in ("headline_out_of_fold", "out_of_fold_2021_plus"):
        block = artifact[cut]
        for label in ("model", "market"):
            table = block["calibration"][label]
            assert len(table) == 10
            assert sum(row["count"] for row in table) == block["n_fights"]


@oof_only
def test_artifact_roi_sweep_has_all_thresholds(artifact):
    roi = artifact["headline_out_of_fold"]["roi"]
    assert set(roi) == {"0.00", "0.05", "0.10"}
    for block in roi.values():
        for side in ("favorite_edge_on_a", "underdog_edge_on_b"):
            assert block[side]["n_bets"] >= 0


@oof_only
def test_artifact_coverage_check_accounts_for_every_walkforward_row(artifact):
    coverage = artifact["odds_coverage_by_fold_year"]
    assert sum(row["n_walkforward"] for row in coverage.values()) == (
        artifact["intersection"]["n_walkforward_rows"]
    )
    assert sum(row["n_odds_matched"] for row in coverage.values()) == (
        artifact["intersection"]["n_intersection"]
    )


def test_july_comparison_declares_the_cuts_the_same_when_the_market_agrees():
    """The market block depends on the fights and the lines, not on the
    model, so identical n plus identical market metrics is what licenses
    subtracting one gap from the other."""
    july = json.loads(bob.FROZEN_BENCHMARK.read_text())["headline_2021_plus"]
    same = bob.july_comparison(
        {
            "n_fights": july["n_fights"],
            "model": {"log_loss": 0.60},
            "market": dict(july["market"]),
            "delta_model_minus_market": {
                "log_loss": july["delta_model_minus_market"]["log_loss"] - 0.005
            },
        }
    )
    assert same["same_fight_set"] is True
    assert "same fights" in same["same_fight_set_evidence"]

    different = bob.july_comparison(
        {
            "n_fights": july["n_fights"] - 100,
            "model": {"log_loss": 0.60},
            "market": dict(july["market"]),
            "delta_model_minus_market": {"log_loss": 0.03},
        }
    )
    assert different["same_fight_set"] is False
    assert "do NOT cover the same fights" in different["same_fight_set_evidence"]


@oof_only
def test_artifact_records_the_odds_coverage_of_the_walkforward_rows(artifact):
    intersection = artifact["intersection"]
    coverage = intersection["odds_coverage_of_walkforward_rows"]
    assert 0.0 < coverage <= 1.0
    assert coverage == round(
        intersection["n_intersection"] / intersection["n_walkforward_rows"], 4
    )
    thinnest = intersection["thinnest_fold_year"]
    by_year = artifact["odds_coverage_by_fold_year"]
    assert thinnest["odds_coverage"] == min(
        row["odds_coverage"] for row in by_year.values()
    )


@oof_only
def test_artifact_records_that_the_honesty_gate_misses_the_in_sample_run(artifact):
    """A gate that fires on an implausibly large edge cannot catch a model
    that trained on the evaluation set, and the artifact must say so rather
    than leave the gate looking sufficient."""
    diagnostic = artifact["in_sample_diagnostic"]
    gap = diagnostic["delta_model_minus_market"]["log_loss"]
    assert diagnostic["beats_the_market"] is (gap < 0)
    assert diagnostic["honesty_gate_would_have_caught_it"] is (gap < -0.02)


@oof_only
def test_artifact_july_comparison_is_like_for_like(artifact):
    comparison = artifact["comparison_with_frozen_july_artifact"]
    assert comparison["direction"] in {"narrowed", "widened", "unchanged"}
    assert comparison["change_in_gap"] == round(
        comparison["out_of_fold_2021_plus"]["gap_market_favour"]
        - comparison["july_2026_artifact"]["gap_market_favour"],
        4,
    )
    assert (
        comparison["out_of_fold_2021_plus"]["n_fights"]
        == artifact["out_of_fold_2021_plus"]["n_fights"]
    )


# --------------------------------------------------------------------------
# The committed dump, against the table as it stands today
# --------------------------------------------------------------------------

FEATURES = ROOT / "data" / "processed" / "features.parquet"

needs_table = pytest.mark.skipif(
    not FEATURES.exists() or not bob.OOF_PREDICTIONS.exists(),
    reason="feature table or committed prediction dump not built yet",
)


@needs_table
def test_the_committed_dump_still_pairs_with_the_current_feature_table():
    """The regression this whole path was rewritten for.

    A daily scrape grew `features.parquet` from 11,238 to 11,290 fights; the
    unbounded last fold absorbed all 52; and the committed dump -- paired by
    position back then -- could no longer be read against the table at all, so
    `models/market_benchmark_oof.json` stopped being regenerable. Nothing in
    the suite noticed, because every pairing test ran on synthetic frames.
    This one runs on the artifacts that actually ship: it fails the moment the
    committed dump and the committed table stop describing the same fights.
    """
    features = bob.load_features()
    dump = json.loads(bob.OOF_PREDICTIONS.read_text())
    oof = bob.attach_oof_predictions(bob.pooled_frame(features), dump)
    assert len(oof) == dump["n"]
    assert not oof["fight_id"].duplicated().any()
    assert ((oof["model_p_a"] > 0.0) & (oof["model_p_a"] < 1.0)).all()


@needs_table
def test_the_committed_dump_reproduces_the_report_it_is_named_with():
    """Pairing is necessary and not sufficient: the dump also has to be the
    run its report describes, or the benchmark quotes a model nothing vouches
    for."""
    features = bob.load_features()
    dump = json.loads(bob.OOF_PREDICTIONS.read_text())
    report = json.loads(bob.OOF_REPORT.read_text())
    oof = bob.attach_oof_predictions(bob.pooled_frame(features), dump)
    assert bob.check_dump_reproduces_report(oof, report)["n"] == report["pooled"]["n"]
