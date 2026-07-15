from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.prospective import (
    build_name_index,
    event_filename,
    fold_accents,
    load_event_record,
    match_fighter_id,
    normalize_name,
    predict_event,
    predict_fight,
    write_event_prediction,
)

ROOT = Path(__file__).resolve().parents[1]


# --- name normalization / matching -----------------------------------------


def test_normalize_name_equates_nfc_and_nfd_forms():
    nfc = "José Aldo"  # single-codepoint e-acute
    nfd = "José Aldo"  # e + combining acute accent
    assert normalize_name(nfc) == normalize_name(nfd)


def test_normalize_name_is_case_insensitive():
    assert normalize_name("Conor McGregor") == normalize_name("CONOR MCGREGOR")


def test_normalize_name_does_not_strip_accents():
    assert normalize_name("José Aldo") != normalize_name("Jose Aldo")


def test_fold_accents_strips_diacritics():
    assert fold_accents("Aleksandar Rakić") == fold_accents("Aleksandar Rakic")
    assert fold_accents("Uroš Medić") == "uros medic"
    assert fold_accents("Ľudovít Klein") == "ludovit klein"
    assert fold_accents("Mateusz Rębecki") == "mateusz rebecki"


def _fighters():
    return pd.DataFrame(
        {
            "fighter_id": ["id1", "id2", "id3", "id4"],
            # id1/id3 collide exactly; id4 is the ASCII-stored form of a
            # name Wikipedia renders with diacritics ("Rakić").
            "name": ["Jon Jones", "Israel Adesanya", "Jon Jones", "Aleksandar Rakic"],
        }
    )


def test_match_fighter_id_unique_exact_match():
    index = build_name_index(_fighters())
    fighter_id, tier, reason = match_fighter_id("israel adesanya", index)
    assert fighter_id == "id2"
    assert tier == "exact"
    assert reason is None


def test_match_fighter_id_zero_matches_is_skipped():
    index = build_name_index(_fighters())
    fighter_id, tier, reason = match_fighter_id("Some Unknown Fighter", index)
    assert fighter_id is None
    assert tier is None
    assert "no fighter" in reason.lower()


def test_match_fighter_id_ambiguous_is_skipped():
    index = build_name_index(_fighters())
    fighter_id, tier, reason = match_fighter_id("Jon Jones", index)
    assert fighter_id is None
    assert tier is None
    assert "ambiguous" in reason.lower()


def test_match_fighter_id_accent_folded_fallback():
    # Wikipedia's "Rakić" has no exact match but folds to the stored "Rakic".
    index = build_name_index(_fighters())
    fighter_id, tier, reason = match_fighter_id("Aleksandar Rakić", index)
    assert fighter_id == "id4"
    assert tier == "accent_folded"
    assert reason is None


def test_match_fighter_id_folded_collision_still_skips():
    # Two DISTINCT fighters whose names collide after accent folding.
    fighters = pd.DataFrame(
        {
            "fighter_id": ["idx", "idy"],
            "name": ["José Silva", "Jose Silva"],
        }
    )
    index = build_name_index(fighters)
    # An accented spelling that exists exactly resolves at tier 1...
    fighter_id, tier, _ = match_fighter_id("José Silva", index)
    assert (fighter_id, tier) == ("idx", "exact")
    # ...but a third spelling that only matches via folding hits BOTH
    # fighters in the folded index -> ambiguous -> skip, never guess.
    fighter_id, tier, reason = match_fighter_id("Josè Silva", index)  # grave accent
    assert fighter_id is None
    assert tier is None
    assert "ambiguous after accent folding" in reason.lower()


def test_match_fighter_id_exact_ambiguity_does_not_fall_through_to_folding():
    index = build_name_index(_fighters())
    fighter_id, tier, reason = match_fighter_id("JON JONES", index)
    assert fighter_id is None
    assert "ambiguous" in reason.lower()
    assert "after accent folding" not in reason.lower()  # stopped at tier 1


# --- event_filename -----------------------------------------------------


def test_event_filename_slugifies_punctuation():
    name = event_filename("UFC Fight Night: du Plessis vs. Usman", "2026-07-18")
    assert name == "UFC_UFC_Fight_Night_du_Plessis_vs_Usman_2026-07-18.json"


def test_event_filename_handles_plain_numbered_event():
    assert event_filename("UFC 331", "2026-09-19") == "UFC_UFC_331_2026-09-19.json"


# --- predict_fight (fake ensemble, no torch load) ------------------------


class _FakeEnsemble:
    """Deterministic stand-in for mma.inference.Ensemble.predict."""

    def predict(self, features: pd.DataFrame) -> dict:
        # Slightly favor whichever row has higher pre_overall Elo, encoded
        # indirectly via elo_diff (first column build_matchup always sets).
        elo_diff = float(features["elo_diff"].iloc[0])
        p = 0.5 + np.clip(elo_diff / 1000.0, -0.3, 0.3)
        return {
            "winner_prob": np.array([p]),
            "winner_spread": np.array([0.05]),
            "method_probs": np.array([[0.5, 0.2, 0.3]]),
            "round_probs": np.array([[0.4, 0.3, 0.2, 0.1]]),
            "method_classes": ["decision", "ko_tko", "submission"],
            "round_classes": ["1", "2", "3", "45"],
        }


def _snapshots():
    return pd.DataFrame(
        {
            "career_fights": [10, 8], "career_wins": [8.0, 5.0],
            "career_win_rate": [0.8, 0.625], "career_finish_rate": [0.5, 0.4],
            "kd_pf": [0.4, 0.2], "sub_att_pf": [0.5, 0.1], "td_landed_pf": [1.5, 1.0],
            "td_acc": [0.5, 0.4], "td_def": [0.7, 0.6], "sig_pm": [4.5, 3.5],
            "sig_absorbed_pm": [3.0, 3.5], "ctrl_share": [0.2, 0.15],
            "streak": [3, -1], "last5_win_rate": [0.8, 0.4],
            "last5_avg_opp_elo": [1550.0, 1500.0],
            "elo_overall": [1600.0, 1450.0], "elo_striking": [1580.0, 1440.0],
            "elo_grappling": [1570.0, 1430.0],
            "last_date": pd.to_datetime(["2025-06-01", "2025-05-01"]),
        },
        index=pd.Index(["fid_a", "fid_b"], name="fighter_id"),
    )


def _fighters_bio():
    return pd.DataFrame(
        {
            "dob": pd.to_datetime(["1993-01-01", "1991-01-01"]),
            "height_cm": [180.0, 178.0],
            "reach_cm": [185.0, 180.0],
            "stance": ["Orthodox", "Southpaw"],
        },
        index=pd.Index(["fid_a", "fid_b"], name="fighter_id"),
    )


def _name_index():
    return build_name_index(
        pd.DataFrame(
            {
                "fighter_id": ["fid_a", "fid_b", "dup1", "dup2"],
                "name": ["Fighter A", "Fighter B", "Duplicate Name", "Duplicate Name"],
            }
        )
    )


def test_predict_fight_success_case():
    wiki_fight = {
        "fighter_a_name": "Fighter A", "fighter_b_name": "Fighter B",
        "weight_class": "Lightweight", "title_fight": False, "main_event": True,
    }
    result = predict_fight(
        wiki_fight, _name_index(), _snapshots(), _fighters_bio(),
        _FakeEnsemble(), as_of=pd.Timestamp("2026-07-18"),
    )
    assert result["skipped"] is False
    assert result["fighter_a_id"] == "fid_a"
    assert result["fighter_b_id"] == "fid_b"
    assert result["match_tier"] == "exact"
    assert 0.0 <= result["p_a_wins"] <= 1.0
    assert result["p_a_wins"] > 0.5  # A has the higher Elo
    assert result["scheduled_rounds"] == 5  # main event
    assert result["elo_a"] == 1600.0
    assert result["elo_b"] == 1450.0
    assert set(result["method_probs"]) == {"decision", "ko_tko", "submission"}
    assert sum(result["method_probs"].values()) == pytest.approx(1.0)


def test_predict_fight_unmatched_name_is_skipped():
    wiki_fight = {
        "fighter_a_name": "Totally Unknown Person", "fighter_b_name": "Fighter B",
        "weight_class": "Lightweight",
    }
    result = predict_fight(
        wiki_fight, _name_index(), _snapshots(), _fighters_bio(),
        _FakeEnsemble(), as_of=pd.Timestamp("2026-07-18"),
    )
    assert result["skipped"] is True
    assert "no fighter" in result["reason"].lower()
    assert result["fighter_a_name"] == "Totally Unknown Person"
    assert "fighter_a_id" not in result


def test_predict_fight_ambiguous_name_is_skipped():
    wiki_fight = {
        "fighter_a_name": "Duplicate Name", "fighter_b_name": "Fighter B",
        "weight_class": "Lightweight",
    }
    result = predict_fight(
        wiki_fight, _name_index(), _snapshots(), _fighters_bio(),
        _FakeEnsemble(), as_of=pd.Timestamp("2026-07-18"),
    )
    assert result["skipped"] is True
    assert "ambiguous" in result["reason"].lower()


def test_predict_fight_accent_folded_match_records_tier():
    name_index = build_name_index(
        pd.DataFrame(
            {
                "fighter_id": ["fid_a", "fid_b"],
                # stored ASCII; Wikipedia will render "Fíghter A" with an accent
                "name": ["Fighter A", "Fighter B"],
            }
        )
    )
    wiki_fight = {
        "fighter_a_name": "Fíghter A", "fighter_b_name": "Fighter B",
        "weight_class": "Lightweight", "title_fight": False, "main_event": False,
    }
    result = predict_fight(
        wiki_fight, name_index, _snapshots(), _fighters_bio(),
        _FakeEnsemble(), as_of=pd.Timestamp("2026-07-18"),
    )
    assert result["skipped"] is False
    assert result["fighter_a_id"] == "fid_a"
    assert result["match_tier"] == "accent_folded"


def test_predict_fight_matched_id_but_no_snapshot_is_skipped():
    name_index = _name_index()
    name_index["exact"][normalize_name("No History")] = ["fid_ghost"]
    name_index["folded"][fold_accents("No History")] = ["fid_ghost"]
    wiki_fight = {
        "fighter_a_name": "No History", "fighter_b_name": "Fighter B",
        "weight_class": "Lightweight",
    }
    result = predict_fight(
        wiki_fight, name_index, _snapshots(), _fighters_bio(),
        _FakeEnsemble(), as_of=pd.Timestamp("2026-07-18"),
    )
    assert result["skipped"] is True
    assert "no fight history" in result["reason"].lower()


def test_predict_event_predicts_every_fight_on_card():
    event = {"event_name": "UFC Fight Night: Test", "date": "2026-07-18"}
    wiki_fights = [
        {"fighter_a_name": "Fighter A", "fighter_b_name": "Fighter B",
         "weight_class": "Lightweight", "title_fight": False, "main_event": True},
        {"fighter_a_name": "Fighter A", "fighter_b_name": "Unknown Person",
         "weight_class": "Lightweight", "title_fight": False, "main_event": False},
    ]
    results = predict_event(
        event, wiki_fights, _name_index(), _snapshots(), _fighters_bio(), _FakeEnsemble()
    )
    assert len(results) == 2
    assert results[0]["skipped"] is False
    assert results[1]["skipped"] is True


# --- idempotent writing ---------------------------------------------------


def test_write_event_prediction_creates_new_file(tmp_path):
    fights = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False, "p_a_wins": 0.6}]
    path, record, n_new = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    assert path.exists()
    assert n_new == 1
    assert record["event_name"] == "UFC 999"
    assert record["fights"][0]["predicted_at_utc"] == "2026-07-15T00:00:00Z"
    assert record["fights"][0]["model_version"] == "abc1234"


def test_write_event_prediction_is_idempotent_on_rerun(tmp_path):
    fights = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False, "p_a_wins": 0.6}]
    path, first_record, _ = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    mtime_before = path.stat().st_mtime_ns
    original_bytes = path.read_bytes()

    # Re-predicting with a DIFFERENT probability must not overwrite the
    # original -- the timestamp/prediction is the point.
    changed_fights = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False, "p_a_wins": 0.99}]
    path2, record2, n_new = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-16T00:00:00Z", changed_fights,
    )
    assert path2 == path
    assert n_new == 0
    assert path.read_bytes() == original_bytes
    assert record2["fights"][0]["p_a_wins"] == 0.6  # unchanged
    assert record2["predicted_at_utc"] == "2026-07-15T00:00:00Z"  # unchanged


def test_write_event_prediction_appends_genuinely_new_fights(tmp_path):
    fights = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False, "p_a_wins": 0.6}]
    write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    new_fights = [
        {"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False, "p_a_wins": 0.6},  # dup, skip
        {"fighter_a_name": "C", "fighter_b_name": "D", "skipped": False, "p_a_wins": 0.55},  # new
    ]
    path, record, n_new = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", new_fights,
    )
    assert n_new == 1
    assert len(record["fights"]) == 2
    assert record["fights"][1]["fighter_a_name"] == "C"
    assert record["fights"][1]["predicted_at_utc"] == "2026-07-20T00:00:00Z"
    assert record["fights"][1]["model_version"] == "def5678"
    # first fight is completely untouched
    assert record["fights"][0]["predicted_at_utc"] == "2026-07-15T00:00:00Z"


def test_write_event_prediction_replaces_skipped_stub_with_real_prediction(tmp_path):
    # Re-attempt policy: a skipped stub holds no prediction, so a later run
    # that CAN predict the fight replaces the stub (fresh timestamp, still
    # pre-event).
    stub = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
             "reason": "no fighter matches name 'A'"}]
    path, _, _ = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", stub,
    )
    real = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False,
             "p_a_wins": 0.7}]
    _, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", real,
    )
    assert n_written == 1
    assert len(record["fights"]) == 1
    fight = record["fights"][0]
    assert fight["skipped"] is False
    assert fight["p_a_wins"] == 0.7
    assert fight["predicted_at_utc"] == "2026-07-20T00:00:00Z"  # fresh stamp
    assert fight["model_version"] == "def5678"
    assert "reason" not in fight


def test_write_event_prediction_skip_reattempted_and_still_skipped_is_untouched(tmp_path):
    stub = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
             "reason": "original reason"}]
    path, _, _ = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", stub,
    )
    original_bytes = path.read_bytes()
    still_skipped = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
                      "reason": "different reason"}]
    _, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", still_skipped,
    )
    assert n_written == 0
    assert path.read_bytes() == original_bytes
    assert record["fights"][0]["reason"] == "original reason"


def test_write_event_prediction_never_replaces_a_real_prediction(tmp_path):
    real = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False,
             "p_a_wins": 0.6}]
    path, _, _ = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", real,
    )
    original_bytes = path.read_bytes()
    # Even a later SKIP for the same pair must not displace a prediction.
    late_skip = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
                  "reason": "spurious"}]
    _, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", late_skip,
    )
    assert n_written == 0
    assert path.read_bytes() == original_bytes
    assert record["fights"][0]["p_a_wins"] == 0.6


def test_load_event_record_returns_none_for_missing_file(tmp_path):
    assert load_event_record(tmp_path / "nonexistent.json") is None


# --- dedup on unordered id pair, not ordered name tuple --------------------


def test_write_event_prediction_swapped_corner_order_does_not_duplicate(tmp_path):
    # Real prediction with a & b matched to specific ids.
    fights = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False,
               "p_a_wins": 0.6, "fighter_a_id": "id_x", "fighter_b_id": "id_y"}]
    write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    # Wikipedia re-renders the card with corners swapped -- same ids, same
    # fight, just fighter_a/fighter_b flipped.
    swapped = [{"fighter_a_name": "B", "fighter_b_name": "A", "skipped": False,
                "p_a_wins": 0.4, "fighter_a_id": "id_y", "fighter_b_id": "id_x"}]
    path, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", swapped,
    )
    assert n_written == 0
    assert len(record["fights"]) == 1
    assert record["fights"][0]["p_a_wins"] == 0.6  # original untouched
    assert record["fights"][0]["fighter_a_id"] == "id_x"


def test_write_event_prediction_diacritic_name_change_same_ids_does_not_duplicate(tmp_path):
    fights = [{"fighter_a_name": "Aleksandar Rakic", "fighter_b_name": "Fighter B",
               "skipped": False, "p_a_wins": 0.6,
               "fighter_a_id": "id_x", "fighter_b_id": "id_y"}]
    write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    # Wikipedia later renders the name with full diacritics; same ids.
    reaccented = [{"fighter_a_name": "Aleksandar Rakić", "fighter_b_name": "Fighter B",
                   "skipped": False, "p_a_wins": 0.9,
                   "fighter_a_id": "id_x", "fighter_b_id": "id_y"}]
    path, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", reaccented,
    )
    assert n_written == 0
    assert len(record["fights"]) == 1
    assert record["fights"][0]["p_a_wins"] == 0.6  # original untouched


def test_write_event_prediction_stub_to_real_replacement_uses_name_pair_dedup(tmp_path):
    # A skipped stub has no ids, so it must still dedup on the unordered
    # NORMALIZED name pair -- and a matched re-run replaces it.
    stub = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
             "reason": "no fighter matches name 'A'"}]
    write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", stub,
    )
    real = [{"fighter_a_name": "B", "fighter_b_name": "A", "skipped": False,
             "p_a_wins": 0.7, "fighter_a_id": "id_y", "fighter_b_id": "id_x"}]
    path, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", real,
    )
    assert n_written == 1
    assert len(record["fights"]) == 1
    fight = record["fights"][0]
    assert fight["skipped"] is False
    assert fight["p_a_wins"] == 0.7
    assert "reason" not in fight


def test_write_event_prediction_real_prediction_beats_colliding_stub_name_pair(tmp_path):
    # A real prediction exists for an id pair. A later run produces a
    # SKIPPED stub whose name pair happens to collide (e.g. a different
    # spelling that fails to match). The real prediction must not be
    # replaced by the stub.
    real = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False,
             "p_a_wins": 0.6, "fighter_a_id": "id_x", "fighter_b_id": "id_y"}]
    path, _, _ = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", real,
    )
    original_bytes = path.read_bytes()
    colliding_stub = [{"fighter_a_name": "A", "fighter_b_name": "B", "skipped": True,
                       "reason": "spurious re-match failure"}]
    _, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", colliding_stub,
    )
    assert n_written == 0
    assert path.read_bytes() == original_bytes
    assert record["fights"][0]["p_a_wins"] == 0.6


def test_write_event_prediction_two_different_fights_same_card_both_written(tmp_path):
    fights = [
        {"fighter_a_name": "A", "fighter_b_name": "B", "skipped": False,
         "p_a_wins": 0.6, "fighter_a_id": "id_x", "fighter_b_id": "id_y"},
        {"fighter_a_name": "C", "fighter_b_name": "D", "skipped": False,
         "p_a_wins": 0.55, "fighter_a_id": "id_p", "fighter_b_id": "id_q"},
    ]
    path, record, n_written = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "abc1234", "2026-07-15T00:00:00Z", fights,
    )
    assert n_written == 2
    assert len(record["fights"]) == 2

    # Re-running with the SAME two (genuinely different) fights writes nothing new.
    rerun = [
        {"fighter_a_name": "B", "fighter_b_name": "A", "skipped": False,
         "p_a_wins": 0.4, "fighter_a_id": "id_y", "fighter_b_id": "id_x"},
        {"fighter_a_name": "D", "fighter_b_name": "C", "skipped": False,
         "p_a_wins": 0.45, "fighter_a_id": "id_q", "fighter_b_id": "id_p"},
    ]
    _, record2, n_written2 = write_event_prediction(
        tmp_path, "UFC 999", "2026-08-01", "https://en.wikipedia.org/wiki/UFC_999",
        "def5678", "2026-07-20T00:00:00Z", rerun,
    )
    assert n_written2 == 0
    assert len(record2["fights"]) == 2


# --- integration test against real committed artifacts --------------------


pytestmark_integration = pytest.mark.skipif(
    not (ROOT / "models" / "torch" / "metrics_val.json").exists(),
    reason="ensemble artifacts not built",
)


@pytestmark_integration
def test_predict_fight_integration_with_real_artifacts():
    from mma.inference import Ensemble
    from mma.snapshots import build_snapshots

    fights_df = pd.read_parquet(ROOT / "data" / "processed" / "fights.parquet")
    stats_df = pd.read_parquet(ROOT / "data" / "processed" / "fight_stats.parquet")
    fighters_df = pd.read_parquet(ROOT / "data" / "processed" / "fighters.parquet")
    ratings_df = pd.read_parquet(ROOT / "data" / "processed" / "ratings.parquet")
    snapshots = build_snapshots(fights_df, stats_df, ratings_df)
    fighters_indexed = fighters_df.set_index("fighter_id")
    name_index = build_name_index(fighters_df)
    ensemble = Ensemble.load()

    # Pick two real fighters with a unique name match and a snapshot.
    candidates = [
        (name, fid)
        for name, ids in name_index["exact"].items()
        for fid in ids
        if len(ids) == 1 and fid in snapshots.index
    ]
    assert len(candidates) >= 2
    name_a_norm, id_a = candidates[0]
    name_b_norm, id_b = candidates[1]
    name_a = fighters_df.set_index("fighter_id").loc[id_a, "name"]
    name_b = fighters_df.set_index("fighter_id").loc[id_b, "name"]

    wiki_fight = {
        "fighter_a_name": name_a, "fighter_b_name": name_b,
        "weight_class": "Lightweight", "title_fight": False, "main_event": False,
    }
    result = predict_fight(
        wiki_fight, name_index, snapshots, fighters_indexed, ensemble,
        as_of=pd.Timestamp("2026-08-01"),
    )
    assert result["skipped"] is False
    assert 0.0 <= result["p_a_wins"] <= 1.0
    assert result["p_a_wins"] + (1 - result["p_a_wins"]) == pytest.approx(1.0)
