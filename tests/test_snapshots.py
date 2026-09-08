import pandas as pd

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
    served = build_matchup(
        snapshots.loc["x"], snapshots.loc["y"], bio, bio,
        "Lightweight", False, 3, as_of=pd.Timestamp("2025-01-01"),
    )
    assert len(served) == 1
    assert set(columns_for([BASE_BLOCK])) <= set(served.columns)
