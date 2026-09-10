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
    """Row i of the rebuilt frame must be row i of what the harness pooled.

    The prediction dump carries no fight id, only three parallel lists, so
    the pairing is positional and this equality is the whole basis for it.
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


def test_attach_oof_predictions_pairs_by_position():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    out = bob.attach_oof_predictions(pooled, dump)
    assert list(out["model_p_a"]) == dump["p_winner"]
    assert list(out["fight_id"]) == list(pooled["fight_id"])


def test_attach_oof_predictions_rejects_a_different_row_count():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["n"] = dump["n"] - 1
    dump["fold_year"] = dump["fold_year"][:-1]
    dump["y_winner"] = dump["y_winner"][:-1]
    dump["p_winner"] = dump["p_winner"][:-1]
    with pytest.raises(ValueError, match="pooled rows"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_misaligned_outcome_sequence():
    """A shuffled dump still has the right n and the right multiset of
    outcomes; only an element-wise check catches it, and a mis-paired join
    would produce a plausible-looking log-loss for the wrong fights."""
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    ys = dump["y_winner"]
    swap = next(i for i, y in enumerate(ys) if y != ys[0])
    ys[0], ys[swap] = ys[swap], ys[0]
    with pytest.raises(ValueError, match="y_winner"):
        bob.attach_oof_predictions(pooled, dump)


def test_attach_oof_predictions_rejects_a_misaligned_fold_year_sequence():
    pooled = bob.pooled_frame(_features())
    dump = _dump(pooled)
    dump["fold_year"] = list(reversed(dump["fold_year"]))
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
# Which (report, dump) pair --mode oof can score on today's table
# --------------------------------------------------------------------------


def _pair(tmp_path: Path, name: str, n: int, *, dump: bool = True):
    """A (report, dump) pair on disk, of the shape the resolver reads."""
    report = tmp_path / f"{name}.json"
    predictions = tmp_path / "preds" / f"{name}.json"
    report.write_text(json.dumps({"name": name, "pooled": {"n": n}}))
    if dump:
        predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_text(json.dumps({"name": name, "n": n}))
    return report, predictions


def test_pooled_row_count_counts_what_pooled_frame_builds():
    """The cheap read must agree with the expensive one, or the staleness
    check would clear a pairing `attach_oof_predictions` then rejects."""
    features = _features()
    assert bob.pooled_row_count(features["date"]) == len(bob.pooled_frame(features))


def test_oof_source_status_says_which_pair_still_matches_the_table(tmp_path):
    stale = _pair(tmp_path, "stale", 4804)
    fresh = _pair(tmp_path, "fresh", 4856)
    status = bob.oof_source_status(4856, sources=(stale, fresh))
    assert [row["pairs"] for row in status] == [False, True]
    assert status[0]["n_dump"] == 4804
    assert status[0]["n_pooled_walkforward_rows"] == 4856


def test_oof_source_status_reports_a_pair_whose_dump_was_never_written(tmp_path):
    """The re-validation writes its report whether or not the dump was asked
    for, so 'report present, dump absent' is a real state and must not read
    as a matching pair."""
    status = bob.oof_source_status(
        4856, sources=(_pair(tmp_path, "nodump", 4856, dump=False),)
    )
    assert status[0]["exists"] is False
    assert status[0]["pairs"] is False
    assert status[0]["n_dump"] is None


def test_resolve_oof_source_takes_the_first_pair_that_matches(tmp_path):
    stale = _pair(tmp_path, "stale", 4804)
    fresh = _pair(tmp_path, "fresh", 4856)
    assert bob.resolve_oof_source(4856, sources=(stale, fresh)) == fresh


def test_resolve_oof_source_refuses_rather_than_scoring_a_stale_dump(tmp_path):
    """The positional join is only valid against the table the dump was
    written from. With no matching pair there is no benchmark to build, and
    the message has to name the command that makes one."""
    stale = _pair(tmp_path, "stale", 4804)
    with pytest.raises(SystemExit) as excinfo:
        bob.resolve_oof_source(4856, sources=(stale,))
    message = str(excinfo.value)
    assert "4804" in message and "4856" in message
    assert "revalidate_recipe.py" in message


def test_the_revalidation_pair_is_preferred_over_the_frozen_sp3_evidence():
    """`hybrid_e2.json` is the report SP3's decision rests on and
    `models/simulator.json` names as its source; it is never refreshed in
    place, so the fresh pair has to come first."""
    assert bob.OOF_SOURCES[0] == (bob.OOF_REVALIDATION_REPORT,
                                  bob.OOF_REVALIDATION_PREDICTIONS)
    assert bob.OOF_SOURCES[-1] == (bob.OOF_REPORT, bob.OOF_PREDICTIONS)

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
    # The named pair is one the resolver would pick, and both halves are on
    # disk -- provenance that names a file nobody can open is not provenance.
    named = (ROOT / provenance["walkforward_report"], ROOT / provenance["predictions"])
    assert named in [(ROOT / report, ROOT / dump) for report, dump in bob.OOF_SOURCES]
    assert all(path.exists() for path in named)
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
