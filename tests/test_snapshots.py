import pandas as pd
import pytest

from mma.snapshots import build_snapshots


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2"],
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "fighter_a_id": ["x", "x"],
            "fighter_b_id": ["y", "z"],
            "winner": ["a", "b"],
            "method": ["ko_tko", "decision"],
            "duration_sec": [300.0, 900.0],
        }
    )


def _stats():
    rows = []
    for fid, (sa, sb) in (("f1", (30, 10)), ("f2", (50, 40))):
        rows.append({"fight_id": fid, "corner": "a", "sig_landed": sa,
                     "td_landed": 1, "td_attempted": 2, "sub_att": 0, "ctrl_sec": 60.0, "kd": 0})
        rows.append({"fight_id": fid, "corner": "b", "sig_landed": sb,
                     "td_landed": 0, "td_attempted": 1, "sub_att": 1, "ctrl_sec": 30.0, "kd": 0})
    return pd.DataFrame(rows)


def _ratings():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "fighter_id": ["x", "y", "x", "z"],
            "pre_overall": [1500.0, 1500.0, 1520.0, 1500.0],
            "post_overall": [1520.0, 1480.0, 1502.0, 1518.0],
            "pre_striking": [1500.0] * 4,
            "post_striking": [1510.0, 1490.0, 1495.0, 1515.0],
            "pre_grappling": [1500.0] * 4,
            "post_grappling": [1505.0, 1495.0, 1500.0, 1510.0],
            "pre_fights": [0, 0, 1, 0],
            # the Glicko triple rides in the same table (scripts/build_ratings.py)
            "pre_glicko_mu": [1500.0, 1500.0, 1530.0, 1500.0],
            "post_glicko_mu": [1530.0, 1470.0, 1505.0, 1525.0],
            "pre_glicko_phi": [350.0, 350.0, 180.0, 350.0],
            "post_glicko_phi": [180.0, 180.0, 150.0, 190.0],
            "pre_glicko_sigma": [0.06] * 4,
            "post_glicko_sigma": [0.0599] * 4,
        }
    )


def test_snapshot_reflects_full_career():
    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    x = snapshots.loc["x"]
    assert x["career_fights"] == 2
    assert x["career_wins"] == 1.0
    assert x["streak"] == -1            # won f1, lost f2
    assert x["elo_overall"] == 1502.0   # last post_overall
    assert x["last_date"] == pd.Timestamp("2024-06-01")


def test_one_fight_fighters_present():
    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    assert snapshots.loc["y"]["career_fights"] == 1
    assert snapshots.loc["z"]["elo_overall"] == 1518.0


def test_snapshots_supply_every_state_key_the_served_row_needs():
    """`build_matchup` derives its state dict from the block spec and raises on
    a key nothing supplies, so building a real served row off a snapshot is the
    guarantee -- stronger than comparing two hand-maintained name lists, which
    is what this test used to do."""
    from mma.feature_blocks import BASE_BLOCK, columns_for
    from mma.inference import build_matchup

    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    bio = pd.Series({"dob": pd.Timestamp("1990-01-01"), "height_cm": 180.0,
                     "reach_cm": 183.0, "stance": "Orthodox"})
    # The base contract is what a SNAPSHOT has to satisfy; blocks joined from
    # an external table by fighter id are that table's job, not this one's.
    served = build_matchup(
        snapshots.loc["x"], snapshots.loc["y"], bio, bio,
        "Lightweight", False, 3, as_of=pd.Timestamp("2025-01-01"),
        blocks=(BASE_BLOCK,),
    )
    assert len(served) == 1
    assert set(columns_for([BASE_BLOCK])) <= set(served.columns)


def test_snapshots_carry_the_restored_blocks_state_and_serve_a_row():
    """The serving half of the SP2.1 restoration.

    `build_matchup` builds its state dict from the block spec and raises on a
    key nothing supplies, so the check that a snapshot can serve `trajectory`,
    `context` and `opponent_adjusted` is simply building the row. The two
    date-dependent fields are the ones worth naming: `first_date` replaces
    `years_since_ufc_debut` here for the same reason `last_date` replaces
    `days_since_last` -- both need an as-of date the snapshot does not have.
    """
    from mma.feature_blocks import BASE_BLOCK, columns_for
    from mma.inference import build_matchup

    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    x = snapshots.loc["x"]
    assert x["first_date"] == pd.Timestamp("2024-01-01")
    assert x["last_date"] == pd.Timestamp("2024-06-01")
    assert "years_since_ufc_debut" not in snapshots.columns
    assert "days_since_last" not in snapshots.columns
    assert x["glicko_mu"] == 1505.0        # last post_glicko_mu
    assert x["glicko_phi"] == 150.0

    blocks = (BASE_BLOCK, "trajectory", "context", "opponent_adjusted")
    bio = pd.Series({"dob": pd.Timestamp("1990-01-01"), "height_cm": 180.0,
                     "reach_cm": 183.0, "stance": "Orthodox"})
    as_of = pd.Timestamp("2025-01-01")
    # x debuted at f1 and z at f2, so their tenures differ by the gap between
    # the two cards -- the column has to see that, measured from `first_date`
    served = build_matchup(
        snapshots.loc["x"], snapshots.loc["z"], bio, bio,
        "Lightweight", False, 3, as_of=as_of, blocks=blocks,
    )
    assert set(columns_for(blocks)) <= set(served.columns)
    assert served["years_since_ufc_debut_diff"].iloc[0] == pytest.approx(
        (as_of - pd.Timestamp("2024-01-01")).days / 365.25
        - (as_of - pd.Timestamp("2024-06-01")).days / 365.25
    )
    # the served deviation is the post-fight one GROWN over the lay-off
    # (mma.glicko.decay_days), not the stale post-fight number -- which is
    # what makes the served value equal the one the same fight would train on
    from mma import glicko

    days = (as_of - pd.Timestamp("2024-06-01")).days
    grown = [glicko.decay_days(glicko.Rating(1500.0, phi, 0.0599), days).rd
             for phi in (150.0, 190.0)]
    assert served["glicko_phi_diff"].iloc[0] == pytest.approx(grown[0] - grown[1])
    assert grown[0] > 150.0
