"""Per-round hazard rows and decision rows (SP3 Task 1).

The orientation assertions are the point of this file: `a_*` labels must
mean "the FEATURE table's corner A", which is the fights table's corner b
whenever `swapped` is True. Getting that backwards inverts every method
prediction silently, so it is checked on both a hand-built fixture and the
real table.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.hazard import (
    HAZARD_CLASSES, build_decision_rows, build_hazard_rows, mirror_corners,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


# --------------------------------------------------------------------------
# hand-built fixture
# --------------------------------------------------------------------------

def _fixture():
    """Six fights covering every branch the builder has to handle.

    f_ko2   A (feature frame) wins by KO in round 2 of 3, not swapped
    f_dec3  decision over 3 rounds, feature-A wins
    f_ko1   finish in round 1 of 5
    f_sub_b feature-corner B wins by submission in round 3, swapped
    f_na    finish in round 2 with NA scheduled_rounds
    f_r6    finish in round 6 with NA scheduled_rounds (the real R6 shape)
    """
    fights = pd.DataFrame(
        {
            "fight_id": ["f_ko2", "f_dec3", "f_ko1", "f_sub_b", "f_na", "f_r6"],
            "winner": ["a", "a", "b", "a", "a", "b"],
            "method": ["ko_tko", "decision", "ko_tko", "submission",
                       "submission", "ko_tko"],
            "finish_round": pd.array([2, None, 1, 3, 2, 6], dtype="Int64"),
            "scheduled_rounds": pd.array([3, 3, 5, 3, None, None], dtype="Int64"),
        }
    )
    # swapped=True means the feature frame's corner A is the fights table's b
    swapped = [False, False, False, True, False, False]
    y_winner = [
        1,  # a wins, not swapped -> feature corner A
        1,  # a wins, not swapped -> feature corner A
        0,  # b wins, not swapped -> feature corner B
        0,  # a wins, SWAPPED     -> feature corner B
        1,  # a wins, not swapped -> feature corner A
        0,  # b wins, not swapped -> feature corner B
    ]
    features = pd.DataFrame(
        {
            "fight_id": fights["fight_id"],
            "date": pd.to_datetime(["2020-01-01"] * 6),
            "swapped": swapped,
            "y_winner": y_winner,
            "y_method": fights["method"].astype("string"),
            "y_finish_round": pd.array(
                ["2", None, "1", "3", "2", "45"], dtype="string"
            ),
            "scheduled_rounds": fights["scheduled_rounds"],
            "elo_diff": [10.0, -5.0, 0.0, 3.0, 1.0, 2.0],
        }
    )
    return features, fights


def _rows_for(frame, fight_id):
    return frame[frame["fight_id"] == fight_id].sort_values("round_no")


def test_ko_in_round_two_of_three_makes_two_rows():
    rows = _rows_for(build_hazard_rows(*_fixture()), "f_ko2")
    assert list(rows["round_no"]) == [1, 2]
    assert list(rows["hazard_label"]) == ["survive", "a_ko"]


def test_three_round_decision_is_three_survive_rows():
    rows = _rows_for(build_hazard_rows(*_fixture()), "f_dec3")
    assert list(rows["round_no"]) == [1, 2, 3]
    assert set(rows["hazard_label"]) == {"survive"}


def test_round_one_finish_contributes_one_row_not_three():
    rows = _rows_for(build_hazard_rows(*_fixture()), "f_ko1")
    assert list(rows["round_no"]) == [1]
    assert list(rows["hazard_label"]) == ["b_ko"]


def test_swapped_submission_labels_the_feature_frames_corner_b():
    """fights says corner a won by submission; swapped=True so that is
    the feature table's corner B."""
    rows = _rows_for(build_hazard_rows(*_fixture()), "f_sub_b")
    assert list(rows["round_no"]) == [1, 2, 3]
    assert list(rows["hazard_label"]) == ["survive", "survive", "b_sub"]


def test_na_scheduled_rounds_is_kept_and_bounded_by_rounds_fought():
    frame = build_hazard_rows(*_fixture())
    rows = _rows_for(frame, "f_na")
    assert list(rows["round_no"]) == [1, 2]
    assert list(rows["hazard_label"]) == ["survive", "a_sub"]
    assert rows["scheduled_rounds"].isna().all()


def test_sixth_round_finish_emits_six_rows():
    rows = _rows_for(build_hazard_rows(*_fixture()), "f_r6")
    assert list(rows["round_no"]) == [1, 2, 3, 4, 5, 6]
    assert list(rows["hazard_label"])[-1] == "b_ko"


def test_feature_columns_are_carried_through():
    frame = build_hazard_rows(*_fixture())
    assert "elo_diff" in frame.columns
    ko2 = _rows_for(frame, "f_ko2")
    assert set(ko2["elo_diff"]) == {10.0}


def test_hazard_label_values_are_the_declared_classes():
    frame = build_hazard_rows(*_fixture())
    assert set(frame["hazard_label"]) <= set(HAZARD_CLASSES)


def test_builders_do_not_mutate_their_inputs():
    features, fights = _fixture()
    before_f = features.copy(deep=True)
    before_g = fights.copy(deep=True)
    build_hazard_rows(features, fights)
    build_decision_rows(features, fights)
    pd.testing.assert_frame_equal(features, before_f)
    pd.testing.assert_frame_equal(fights, before_g)


def test_decision_rows_label_the_feature_frames_winner():
    features, fights = _fixture()
    rows = build_decision_rows(features, fights)
    assert list(rows["fight_id"]) == ["f_dec3"]
    assert list(rows["decision_label"]) == [1]


def test_decision_rows_label_zero_when_feature_corner_b_wins():
    features, fights = _fixture()
    features = features.copy()
    features.loc[features["fight_id"] == "f_dec3", "y_winner"] = 0
    rows = build_decision_rows(features, fights)
    assert list(rows["decision_label"]) == [0]


def test_hazard_rows_have_no_row_beyond_the_round_fought():
    frame = build_hazard_rows(*_fixture())
    finishes = frame[frame["hazard_label"] != "survive"]
    # the finishing row is the last round for its fight
    last = frame.groupby("fight_id")["round_no"].max()
    for _, row in finishes.iterrows():
        assert row["round_no"] == last[row["fight_id"]]


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

real_data = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="processed tables not built",
)


@pytest.fixture(scope="module")
def real():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    return features, fights


# These assert RELATIONS between the built rows and the table they came from,
# never a row count copied out of one particular build. The weekly refresh
# appends fights, so a hard-coded total is a test that fails on the first
# successful data refresh -- and it fails in the Action, before the refreshed
# data is committed, which is the worst possible place for it.


@real_data
def test_real_hazard_rows_are_one_per_fight_and_round_fought(real):
    frame = build_hazard_rows(*real)
    assert not frame.duplicated(["fight_id", "round_no"]).any()
    span = frame.groupby("fight_id")["round_no"].agg(["min", "max", "nunique"])
    assert (span["min"] == 1).all()
    assert (span["nunique"] == span["max"]).all()  # contiguous from round 1
    assert set(frame["fight_id"]) <= set(real[0]["fight_id"])
    assert len(frame) > len(set(frame["fight_id"]))  # fights last many rounds


@real_data
def test_real_decision_rows_are_exactly_the_fights_that_went_to_the_cards(real):
    features, fights = real
    frame = build_decision_rows(features, fights)
    went_the_distance = features["y_method"].astype("string").eq("decision")

    assert len(frame) == int(went_the_distance.fillna(False).sum())
    assert set(frame["fight_id"]) == set(features.loc[went_the_distance.fillna(False), "fight_id"])


@real_data
def test_real_label_distribution_matches_the_method_counts(real):
    features, fights = real
    frame = build_hazard_rows(features, fights)
    counts = frame["hazard_label"].value_counts()
    decisive = fights[fights["winner"].isin(["a", "b"])]
    n_ko = int((decisive["method"] == "ko_tko").sum())
    n_sub = int((decisive["method"] == "submission").sum())
    assert counts["a_ko"] + counts["b_ko"] == n_ko
    assert counts["a_sub"] + counts["b_sub"] == n_sub
    # Every remaining round is one the fight survived.
    assert counts["survive"] == len(frame) - n_ko - n_sub


@real_data
def test_real_orientation_matches_y_winner(real):
    """Every finish row's labelled corner is the corner `y_winner` names."""
    features, fights = real
    frame = build_hazard_rows(features, fights)
    finishes = frame[frame["hazard_label"] != "survive"]
    labelled_a = finishes["hazard_label"].str.startswith("a_").to_numpy()
    assert np.array_equal(labelled_a, finishes["y_winner"].to_numpy() == 1)


@real_data
def test_real_orientation_inverts_with_the_swapped_flag(real):
    """Independently of `y_winner`: `a_*` is the fights table's corner a
    when not swapped and its corner b when swapped."""
    features, fights = real
    frame = build_hazard_rows(features, fights)
    finishes = frame[frame["hazard_label"] != "survive"]
    raw = fights.set_index("fight_id")["winner"].reindex(finishes["fight_id"])
    raw_a = (raw.to_numpy() == "a")
    swapped = finishes["swapped"].to_numpy(dtype=bool)
    expected_a = np.where(swapped, ~raw_a, raw_a)
    labelled_a = finishes["hazard_label"].str.startswith("a_").to_numpy()
    assert np.array_equal(labelled_a, expected_a)
    assert swapped.any() and (~swapped).any()


@real_data
def test_real_method_matches_the_label_suffix(real):
    features, fights = real
    frame = build_hazard_rows(features, fights)
    finishes = frame[frame["hazard_label"] != "survive"]
    suffix = finishes["hazard_label"].str.split("_").str[1]
    expected = finishes["y_method"].map({"ko_tko": "ko", "submission": "sub"})
    assert (suffix.to_numpy() == expected.to_numpy()).all()


@real_data
def test_real_round_numbers_are_contiguous_from_one(real):
    frame = build_hazard_rows(*real)
    grouped = frame.groupby("fight_id")["round_no"]
    assert (grouped.min() == 1).all()
    assert (grouped.count() == grouped.max()).all()


@real_data
def test_real_na_scheduled_fights_are_kept(real):
    features, fights = real
    frame = build_hazard_rows(features, fights)
    na_ids = set(
        fights.loc[fights["scheduled_rounds"].isna(), "fight_id"]
    ) & set(features["fight_id"])
    assert na_ids
    assert na_ids <= set(frame["fight_id"])


@real_data
def test_real_decision_rows_are_balanced(real):
    rows = build_decision_rows(*real)
    assert 0.45 < rows["decision_label"].mean() < 0.55


# --------------------------------------------------------------------------
# corner mirroring (SP3 Task 3's symmetrisation)
# --------------------------------------------------------------------------

def test_mirror_negates_differentials_and_swaps_the_corner_columns():
    features, _ = _fixture()
    features = features.assign(
        age_a=[30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        age_b=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        southpaw_a=[True] * 6,
        southpaw_b=[False] * 6,
    )
    flipped = mirror_corners(features)
    assert list(flipped["elo_diff"]) == [-v for v in features["elo_diff"]]
    assert list(flipped["age_a"]) == list(features["age_b"])
    assert list(flipped["age_b"]) == list(features["age_a"])
    assert list(flipped["southpaw_a"]) == list(features["southpaw_b"])
    assert list(flipped["y_winner"]) == [1 - v for v in features["y_winner"]]
    assert list(flipped["swapped"]) == [not v for v in features["swapped"]]
    # fight-level facts describe the bout, not a corner
    assert list(flipped["scheduled_rounds"]) == list(features["scheduled_rounds"])
    assert list(flipped["y_method"]) == list(features["y_method"])
    assert list(flipped.columns) == list(features.columns)


def test_mirror_leaves_the_caller_s_frame_alone():
    features, _ = _fixture()
    before = features["elo_diff"].tolist()
    mirror_corners(features)
    assert features["elo_diff"].tolist() == before


def test_mirror_rejects_a_corner_column_without_a_twin():
    features, _ = _fixture()
    with pytest.raises(ValueError, match="reach_a"):
        mirror_corners(features.assign(reach_a=[1.0] * 6))


@pytest.mark.skipif(not (PROCESSED / "features.parquet").exists(),
                    reason="processed feature table not built")
def test_mirror_is_an_involution_on_the_real_table():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    twice = mirror_corners(mirror_corners(features))
    for column in features.columns:
        left, right = features[column], twice[column]
        if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
            # -(-0.0) is 0.0 and NaN != NaN, so compare with a null-aware equality
            assert ((left == right) | (left.isna() & right.isna())).all(), column
        else:
            assert left.equals(right), column


@pytest.mark.skipif(not (PROCESSED / "features.parquet").exists(),
                    reason="processed feature table not built")
def test_mirroring_a_self_symmetric_row_changes_nothing():
    """The fixed point the symmetrisation test relies on: a matchup with no
    corner asymmetry must mirror onto itself, so both orientations of the
    simulator see exactly the same input."""
    features = pd.read_parquet(PROCESSED / "features.parquet").head(1).copy()
    for column in features.columns:
        if column.endswith("_diff"):
            features[column] = features[column] * 0
        elif column.endswith("_a"):
            features[column[:-2] + "_b"] = features[column].to_numpy()
    flipped = mirror_corners(features)
    for column in features.columns:
        if column in ("y_winner", "swapped"):
            continue
        left, right = features[column], flipped[column]
        assert ((left == right) | (left.isna() & right.isna())).all(), column
