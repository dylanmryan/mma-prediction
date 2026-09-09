"""Feature blocks: named, independently evaluable groups of columns.

SP2 adds signal in blocks so each can be judged separately against the
walk-forward bar (see docs/superpowers/plans/2026-09-07-sp2-features-v3.md).
`BASE_BLOCK` is the v1 feature set and is always enabled. A block declares
the state fields it consumes and the output columns it produces; the
registry is what `scripts/build_features.py --blocks` and the serving path
both read, so a block cannot be half-enabled.

`mma.serving.feature_row` walks the resolved blocks one at a time, in
registry order; `spec_for` assembles a requested block list into one `Spec`
(used by tests and callers that want the flattened differentials/absolutes/
booleans rather than walking blocks themselves), `columns_for` names the
columns that walk emits, in the order it emits them, and `state_keys` names
the state fields it reads -- the registry is the contract, not a parallel
description of one.
Both the training merge (`mma.features._side_frame`) and the serving state
dict (`mma.inference.build_matchup`) are built from `state_keys`, so a block
whose state key nothing provides raises instead of quietly producing an
all-NaN column that then measures as "no improvement".
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

BASE_BLOCK = "base"

# The sidecar `scripts/build_features.py` writes next to the table it built.
BLOCKS_SIDECAR = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "features_blocks.json"
)


@dataclass(frozen=True)
class Block:
    """One switchable group of feature columns.

    `differentials` are (state key, output stem) pairs emitted as
    "<stem>_diff" (A minus B); `absolutes` and `booleans` are state keys
    emitted per corner as "<stem>_a"/"<stem>_b" (booleans coerced to strict
    bools); `derived_booleans` are (output column, per-corner stem) pairs
    emitted as the XOR of that stem's two corners; `fight_level` columns
    describe the fight rather than a corner and are re-emitted from the
    context at this block's position in the row.
    """

    name: str
    differentials: tuple[tuple[str, str], ...] = ()
    absolutes: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    derived_booleans: tuple[tuple[str, str], ...] = ()
    fight_level: tuple[str, ...] = ()


@dataclass(frozen=True)
class Spec:
    """Several blocks flattened into one builder spec, in block order."""

    differentials: tuple[tuple[str, str], ...] = ()
    absolutes: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    derived_booleans: tuple[tuple[str, str], ...] = ()
    fight_level: tuple[str, ...] = ()


BLOCKS: dict[str, Block] = {}


def register(block: Block) -> Block:
    """Add a block to the registry, rejecting a name or column already taken.

    The column check lives here rather than in a test so a colliding block
    cannot enter the registry at all: whichever of the two blocks lost the
    race would otherwise have its column silently overwritten in the built
    table, which is the same class of bug as a state key nothing provides.
    This also rejects a block that redeclares one of its own columns
    against itself (e.g. the same differential stem twice) -- checking only
    against previously registered blocks would let that in cleanly and then
    break the documented `columns_for` == `feature_row` order contract."""
    if block.name in BLOCKS:
        raise ValueError(f"duplicate feature block {block.name}")
    own_columns = columns_of(block)
    if len(own_columns) != len(set(own_columns)):
        self_dupes = sorted({c for c in own_columns if own_columns.count(c) > 1})
        raise ValueError(f"feature block {block.name} redeclares its own column(s): {self_dupes}")
    claimed = {column: name for name, other in BLOCKS.items()
               for column in columns_of(other)}
    collisions = sorted(set(own_columns) & set(claimed))
    if collisions:
        named = ", ".join(f"{c} (already {claimed[c]})" for c in collisions)
        raise ValueError(f"feature block {block.name} redeclares column(s): {named}")
    BLOCKS[block.name] = block
    return block


@contextmanager
def registry():
    """Snapshot `BLOCKS` and restore it on exit.

    For tests that register throwaway blocks: wrapping the registrations
    themselves means a `register` that raises part-way through setup cannot
    leak the blocks that went in before it into the real registry."""
    original = dict(BLOCKS)
    try:
        yield BLOCKS
    finally:
        BLOCKS.clear()
        BLOCKS.update(original)


def resolve_blocks(names) -> tuple[str, ...]:
    """Normalise a requested block list: base first, then registry order.

    Order-insensitive by construction, so two runs asking for the same set
    of blocks in different orders build the identical table. A bare string is
    accepted as a single name -- it would otherwise iterate its characters
    and report a list of unknown one-letter blocks."""
    if isinstance(names, str):
        names = (names,)
    requested = set(names) | {BASE_BLOCK}
    unknown = sorted(requested - set(BLOCKS))
    if unknown:
        raise ValueError(f"unknown feature block(s): {unknown}; known: {sorted(BLOCKS)}")
    return tuple([BASE_BLOCK] + [n for n in BLOCKS if n != BASE_BLOCK and n in requested])


def table_blocks(sidecar=BLOCKS_SIDECAR) -> tuple[str, ...]:
    """The blocks `data/processed/features.parquet` was built from.

    This is the SERVING contract as well as the training one: the deployed
    model's preprocessor was fitted on that table, so a row served with fewer
    blocks is missing columns the preprocessor asks for (a loud KeyError) and
    one served with more just carries columns it ignores. Reading the sidecar
    is what stops `build_matchup`'s callers -- the app, the prospective run,
    the explainer -- from each having to remember which blocks shipped.

    Falls back to the v1 `base` contract when no sidecar exists, which is the
    state of a checkout that has not run `scripts/build_features.py` yet.
    """
    path = Path(sidecar)
    if not path.exists():
        return (BASE_BLOCK,)
    return resolve_blocks(json.loads(path.read_text())["blocks"])


def spec_for(names) -> Spec:
    """The assembled spec for a requested block list."""
    blocks = [BLOCKS[name] for name in resolve_blocks(names)]
    return Spec(
        differentials=tuple(d for b in blocks for d in b.differentials),
        absolutes=tuple(a for b in blocks for a in b.absolutes),
        booleans=tuple(f for b in blocks for f in b.booleans),
        derived_booleans=tuple(d for b in blocks for d in b.derived_booleans),
        fight_level=tuple(c for b in blocks for c in b.fight_level),
    )


def columns_of(block: Block) -> tuple[str, ...]:
    """One block's own output columns, in the order it emits them.

    No `base` injection -- this is exactly what `block` contributes, which is
    what a per-block uniqueness check needs. `mma.serving.feature_row` walks
    the resolved blocks one at a time and emits each block's columns in this
    order, so `columns_for` is just these, concatenated."""
    columns = [f"{stem}_diff" for _, stem in block.differentials]
    columns += [
        f"{stem}_{corner}"
        for corner in ("a", "b")
        for stem in block.absolutes + block.booleans
    ]
    columns += [name for name, _ in block.derived_booleans]
    columns += list(block.fight_level)
    return tuple(columns)


def columns_for(names) -> tuple[str, ...]:
    """Output columns for a requested block list, in emission order.

    Mirrors `mma.serving.feature_row` exactly: each resolved block's
    `columns_of`, concatenated in registry order."""
    return tuple(
        column
        for name in resolve_blocks(names)
        for column in columns_of(BLOCKS[name])
    )


def state_keys(names) -> tuple[str, ...]:
    """Every state field the resolved blocks read, deduplicated.

    Differential source keys, then absolutes, then booleans, per block in
    registry order. This is the single source of the *values* a block needs,
    the way `columns_for` is the single source of the column *names*: the
    training merge and the serving state dict are both built from it, so a
    block cannot half-exist as an all-NaN column."""
    keys: list[str] = []
    for name in resolve_blocks(names):
        block = BLOCKS[name]
        for key in [k for k, _ in block.differentials] + list(block.absolutes) + list(block.booleans):
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def state_key_blocks(names) -> dict[str, str]:
    """state key -> the first resolved block that declares it (for error messages)."""
    owners: dict[str, str] = {}
    for name in resolve_blocks(names):
        block = BLOCKS[name]
        for key in [k for k, _ in block.differentials] + list(block.absolutes) + list(block.booleans):
            owners.setdefault(key, name)
    return owners


# --- the v1 feature contract -------------------------------------------------
# Column order of the committed feature table; moved here verbatim from
# `mma.serving` so the registry holds the single definition.
register(Block(
    name=BASE_BLOCK,
    differentials=(
        ("career_fights", "career_fights"),
        ("career_wins", "career_wins"),
        ("career_win_rate", "career_win_rate"),
        ("career_finish_rate", "career_finish_rate"),
        ("kd_pf", "kd_pf"),
        ("sub_att_pf", "sub_att_pf"),
        ("td_landed_pf", "td_landed_pf"),
        ("td_acc", "td_acc"),
        ("td_def", "td_def"),
        ("sig_pm", "sig_pm"),
        ("sig_absorbed_pm", "sig_absorbed_pm"),
        ("ctrl_share", "ctrl_share"),
        ("streak", "streak"),
        ("days_since_last", "days_since_last"),
        ("last5_win_rate", "last5_win_rate"),
        ("last5_avg_opp_elo", "last5_avg_opp_elo"),
        ("pre_overall", "elo"),
        ("pre_striking", "striking_elo"),
        ("pre_grappling", "grappling_elo"),
        ("pre_fights", "elo_fights"),
        ("height_cm", "height"),
        ("reach_cm", "reach"),
        ("age", "age"),
    ),
    absolutes=("age", "career_fights"),
    booleans=("reach_missing", "dob_missing", "southpaw", "debut"),
    derived_booleans=(
        ("debut_matchup", "debut"),
        ("stance_mismatch", "southpaw"),
    ),
))


# --- SP2 block: external -----------------------------------------------------
# Pre-UFC career and origin, from the MIT `ehan03/jds-mma-data` snapshot via
# `data/external/fighter_external.parquet`. This is the first block whose
# information is not a recombination of the fight table: it describes what a
# fighter did BEFORE the UFC, which our sources do not record at all.
#
# The state comes from `mma.external` (joined by fighter id only, never by
# name) in both directions -- `mma.features._side_frame` for training,
# `mma.inference.build_matchup` for serving.
#
# MISSINGNESS IS FIGHT-LEVEL ONLY, and that is a measured correction rather
# than a style choice. The SP1 rule asks every externally-sourced column to
# carry a `*_missing` boolean, but a PER-CORNER pair leaks: membership of the
# snapshot's cross-source fighter mapping is a function of how long a
# fighter's UFC career turned out to be (among 2013-2022 debutants, 24% of
# one-and-done fighters are in it against 100% of those with 11+ bouts), so
# `external_missing_a` / `_b` would tell the model which corner went on to
# have a career -- the unmapped corner loses 76% of the time overall and 90%
# of the time in the debut slice. The row-level OR keeps the missingness
# visible (and is what `mma.walkforward.slice_masks` reads for the
# `external_missing` slice) while being symmetric in the two corners, so it
# cannot say which of them wins. See the SP2 plan's Task 11 notes.
#
# `nationality` and `gym_id` are not features (high-cardinality strings); the
# only thing derived from them is the fight-level `same_country` flag.
EXTERNAL_BLOCK = "external"

register(Block(
    name=EXTERNAL_BLOCK,
    differentials=(
        ("pre_ufc_wins", "pre_ufc_wins"),
        ("pre_ufc_losses", "pre_ufc_losses"),
        ("pre_ufc_finish_rate", "pre_ufc_finish_rate"),
        ("pre_ufc_finish_loss_rate", "pre_ufc_finish_loss_rate"),
        ("pre_ufc_avg_opp_wins", "pre_ufc_avg_opp_wins"),
        ("days_since_pro_debut", "days_since_pro_debut"),
    ),
    fight_level=("external_missing", "same_country"),
))



# --- SP2 block: notice (REJECTED IN SP2 AND SP2.1; SHIPS IN SP2.2 AS PART OF S1)
# Short notice and missed weight, from the Bet MMA tables in the same
# `ehan03/jds-mma-data` snapshot. Deliberately NOT registered: it was
# built, evaluated and reverted in SP2 Task 12, where it did not clear the bar
# (torch pooled 0.6472 against the incumbent 0.6476 with the coverage flag held
# out of the model, 0.6482 with it in -- a fifth of the 0.003 bar at best), and
# the reason is coverage rather than signal: the source's bout list ends
# 2024-12-14, so the columns are unknown on 100% of 2025 and 2026 rows, which
# is exactly where the deployed model predicts. The raw signal is real and
# large where it exists -- of the 484 observed fights with one corner on <=30
# days' notice, that corner wins 35% of the time, and 26% at <=7 days -- but it
# reaches only 4% of the table.
#
# What was kept, because it is reusable and correct: the derivation in
# `scripts/build_external.py`, the committed `data/external/fight_notice.parquet`,
# the loader `mma.notice`, and `mma.wiki_cards.parse_background`, which is the
# only route to these facts for a FUTURE event.
#
# SHIPPING FORM, if it is ever registered again (`notice_noflag`, the better
# of the two SP2 variants): the fight-level `notice_unknown` stays in the
# TABLE -- something has to record that the source has not seen a fight -- and
# is held out of both model matrices via `mma.tensors.DROPPED` and
# `mma.models.xgb.MODEL_EXCLUDED`, exactly as `external`'s two flags are.
# Modelling it cost 0.0010 on torch (0.6482 against 0.6472). Both exclusion
# lists still name it, so re-registering the block needs no change there.
#
# SP2.1 RESULT (2026-09-08): UNREGISTERED AGAIN. The pre-registered capacity
# experiment combined this block with the other three whose torch point
# estimate was negative or whose scorers disagreed into S1 (11,238 x 87) and
# ran one 25-configuration torch architecture search over both S1 (arm T) and
# the shipped S0 (arm C, the control that separates a feature effect from an
# architecture effect). Neither arm cleared the 0.003 bar against the deployed
# incumbent -- C -0.0013, T -0.0012 -- and T - C = +0.0001 is inside
# sigma_seed = 0.000346, so the pre-registration's fourth attribution branch
# applied: the blocks are dead and the architecture is adequate; record and
# revert. The only result that cleared any bar, A1 on XGBoost at -0.0039 on
# seed 0, failed its fresh-seed confirmation (mean -0.0024 over seeds 1-3
# against a newly measured XGB sigma_seed of 0.00087-0.00125). Nothing shipped
# and nothing was redeployed. See
# docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md and
# models/walkforward/sp2_1_decision.json.
#
# The wiring the block needs -- `notice.attach` and `notice.fight_context` in
# `mma.features._side_frame` / `build_features`, and the `notice_a` /
# `notice_b` arguments to `mma.inference.build_matchup` -- sits behind
# `if NOTICE_BLOCK in resolved` guards, which are live again now that the
# registration below is permanent. `notice_unknown` stays in the table (the
# `notice` slice needs it) and out of both model matrices.
NOTICE_BLOCK = "notice"

# SP2.2 RESULT (2026-09-08): **REGISTERED PERMANENTLY -- THIS BLOCK SHIPS.**
# Not as a block that finally cleared a bar on its own: it never did, and the
# SP2.1 verdict above stands unedited. It ships as one sixth of S1
# (`base,external,trajectory,notice,context,opponent_adjusted`, 11,238 x 87),
# the feature table SP2.2's candidate B1 was scored on. B1 -- the equal-weight,
# temperature-scaled blend of a 5-seed XGBoost ensemble and the 5-seed torch
# ensemble -- cleared the 0.003 bar at seeds 0-4 (-0.0039 against I) and again
# at seeds 5-9 (-0.0037 against its own fresh-seed paired incumbent), and the
# amended ECE gate passes. What SP2.1 could not see is that these blocks are
# alive THROUGH THE XGBOOST MEMBER: the XGB arm on S1 is 0.6462 against 0.6490
# on S0, while the torch arm is 0.6475 either way. The blocks are dead to the
# MLP and were never dead to the trees. See
# docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md and
# models/walkforward/sp2_2_decision.json.
#
# Re-verified before it was relied on: rebuilt on the feature set the committed
# SP2 report's own config names, this registration reproduces that report
# EXACTLY -- pooled, all eight folds, all slices and every per-fold fit budget.
register(Block(
    name=NOTICE_BLOCK,
    differentials=(("notice_shortfall_days", "notice_shortfall_days"),
                   ("missed_weight_over_lbs", "missed_weight_over_lbs")),
    booleans=("short_notice_7", "short_notice_30", "missed_weight"),
    fight_level=("notice_unknown",),
))



# --- SP2 block: rankings (MEASURED AND REJECTED) -----------------------------
# The UFC's own weekly divisional rankings, from the CC0 Kaggle dataset
# `jerzyszocik/ufc-rankings-history` via `data/external/rankings.parquet`.
# Deliberately NOT registered: built, evaluated and reverted in SP2 Task 12.
# All three variants were WORSE on torch (pooled 0.6487-0.6493 against the
# incumbent 0.6476), so the block ships in no form.
#
# The interesting part is that the signal is real and the columns are clean.
# Unlike `external` this source is contemporaneous and refreshed weekly, so
# its coverage does not decay: `rank_diff` is populated on 0.19-0.27 of rows in
# every year from 2015 to 2026, never trending. Among the 1,502 fights where
# both corners were ranked at different ranks, the better-ranked corner wins
# 0.563 of the time -- BETTER than `elo_diff` picks the same rows (0.533) --
# and the champion wins 0.673 of 199 champion-vs-challenger bouts. The per-
# corner flags are also not the `external` leak in disguise: where exactly one
# corner is ranked (n=1,300) that corner wins only 0.545 of the time, flat
# across years, against the unmapped corner's 0.24 in `external`.
#
# It still costs, because `rank_diff` reaches 13.8% of the table and correlates
# -0.40 with `elo_diff` and -0.54 with `last5_avg_opp_elo_diff` on the rows it
# does reach. Same lesson as `in_fight` and `opponent_adjusted`: a correlated
# addition that is missing on most rows costs the MLP more than its marginal
# signal is worth. See the SP2 plan's Task 12 notes.
#
#     register(Block(
#         name="rankings",
#         differentials=(("rank", "rank"),),
#         booleans=("is_champion", "is_ranked"),
#         fight_level=("rank_missing", "ranking_regime_post_2026_06"),
#     ))


# --- SP2 block: trajectory (REJECTED IN SP2 AND SP2.1; SHIPS IN SP2.2 IN S1) -
# Rating DYNAMICS on top of the base block's rating LEVELS: Glicko-2's
# deviation and volatility, recent Elo momentum, the drawdown from a
# fighter's own peak, tenure, and two age interactions. Built, evaluated and
# reverted in SP2 Task 7; restored and re-measured inside the SP2.1 combined
# set because it is one of the blocks whose two scorers disagreed by the most,
# then unregistered again when that set did not clear the bar either.
#
# XGB liked it (pooled 0.6486 against the incumbent 0.6506, delta -0.0020) and
# torch barely moved (0.6472 against 0.6476, delta **-0.0004** -- a seventh of
# the 0.003 bar). A second variant with the most collinear column held out of
# the model (`glicko_mu_diff`, r = 0.88 with `elo_diff`) landed in the same
# place, -0.0005, which rules out collinearity-with-Elo as the whole story:
# the block is simply a recombination of the rating history the base block
# already carries, and the deployed scorer has that information.
#
# What was kept, because it is reusable and correct: `mma.glicko` -- a
# tested Glicko-2 implementation pinned to the worked example in Glickman's
# paper -- and the `run_glicko` pass in `scripts/build_ratings.py`, so
# `data/processed/ratings.parquet` carries `{pre,post}_glicko_{mu,phi,sigma}`
# whether or not anything models them.
#
# The block's own wiring: the four accumulator fields in
# `mma.history._FighterState` (`elo_delta_3`, `elo_delta_5`,
# `elo_peak_minus_current`, `years_since_ufc_debut`), the `first_date` and
# Glicko pass-through in `mma.snapshots`, the two age interactions in
# `mma.features._side_frame`, and the as-of derivations in
# `mma.inference.build_matchup` (which grows the served Glicko deviation over
# the days since the fighter's last bout, so the served value equals the
# trained one). See the SP2 plan's Task 7 notes.
#
# SP2.1 RESULT (2026-09-08): UNREGISTERED AGAIN. The pre-registered capacity
# experiment combined this block with the other three whose torch point
# estimate was negative or whose scorers disagreed into S1 (11,238 x 87) and
# ran one 25-configuration torch architecture search over both S1 (arm T) and
# the shipped S0 (arm C, the control that separates a feature effect from an
# architecture effect). Neither arm cleared the 0.003 bar against the deployed
# incumbent -- C -0.0013, T -0.0012 -- and T - C = +0.0001 is inside
# sigma_seed = 0.000346, so the pre-registration's fourth attribution branch
# applied: the blocks are dead and the architecture is adequate; record and
# revert. The only result that cleared any bar, A1 on XGBoost at -0.0039 on
# seed 0, failed its fresh-seed confirmation (mean -0.0024 over seeds 1-3
# against a newly measured XGB sigma_seed of 0.00087-0.00125). Nothing shipped
# and nothing was redeployed. See
# docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md and
# models/walkforward/sp2_1_decision.json.
#
# All of the wiring named above is in the tree: the accumulators are computed
# unconditionally in `mma.history`, and the `features` / `inference`
# derivations sit behind `if TRAJECTORY_BLOCK in resolved` guards, live again
# now that the registration below is permanent. The serving path grows the
# Glicko deviation over the lay-off exactly as the training pass does
# (`mma.inference.build_matchup`), so a served row is not a stale one.
TRAJECTORY_BLOCK = "trajectory"

# SP2.2 RESULT (2026-09-08): **REGISTERED PERMANENTLY -- THIS BLOCK SHIPS.**
# Not as a block that finally cleared a bar on its own: it never did, and the
# SP2.1 verdict above stands unedited. It ships as one sixth of S1
# (`base,external,trajectory,notice,context,opponent_adjusted`, 11,238 x 87),
# the feature table SP2.2's candidate B1 was scored on. B1 -- the equal-weight,
# temperature-scaled blend of a 5-seed XGBoost ensemble and the 5-seed torch
# ensemble -- cleared the 0.003 bar at seeds 0-4 (-0.0039 against I) and again
# at seeds 5-9 (-0.0037 against its own fresh-seed paired incumbent), and the
# amended ECE gate passes. What SP2.1 could not see is that these blocks are
# alive THROUGH THE XGBOOST MEMBER: the XGB arm on S1 is 0.6462 against 0.6490
# on S0, while the torch arm is 0.6475 either way. The blocks are dead to the
# MLP and were never dead to the trees. See
# docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md and
# models/walkforward/sp2_2_decision.json.
#
# Re-verified before it was relied on: rebuilt on the feature set the committed
# SP2 report's own config names, this registration reproduces that report
# EXACTLY -- pooled, all eight folds, all slices and every per-fold fit budget.
register(Block(
    name=TRAJECTORY_BLOCK,
    differentials=(
        ("pre_glicko_mu", "glicko_mu"),
        ("pre_glicko_phi", "glicko_phi"),
        ("pre_glicko_sigma", "glicko_sigma"),
        ("elo_delta_3", "elo_delta_3"),
        ("elo_delta_5", "elo_delta_5"),
        ("elo_peak_minus_current", "elo_peak_minus_current"),
        ("years_since_ufc_debut", "years_since_ufc_debut"),
        ("age_x_fights", "age_x_fights"),
    ),
    absolutes=("age_squared",),
))


# --- SP2 block: context (REJECTED IN SP2 AND SP2.1; SHIPS IN SP2.2 IN S1) ---
# The fight's SETTING rather than either fighter's record: referee tendency,
# home advantage, and how often a fighter's wins carried a post-fight bonus.
# Built, evaluated and reverted in SP2 Task 8; restored in its best-measured
# `context_nohome` form for the SP2.1 capacity experiment, then unregistered
# again when that experiment's combined set did not clear the bar either.
# Three variants were measured and all three are worse than or level with the
# incumbent on torch (0.6474 to 0.6483 against 0.6476); the best,
# `context_nohome`, is -0.0002, a fifteenth of the 0.003 bar.
#
# Two findings worth carrying forward:
#
#   * `home_country_a`/`_b` is the `external` coverage leak in a second
#     channel. A per-corner boolean is False when the corner's nationality is
#     unknown, and nationality exists only for fighters the snapshot mapped,
#     so over the 1,448 feature rows where exactly one corner is mapped the
#     pair is NOT constant -- and the mapped corner wins 0.745 of those. The
#     variant that models it is the WORST of the three (+0.0007). Asserted one
#     level lower, on `mma.context` over the real tables, in
#     `tests/test_context.py::test_a_per_corner_home_country_pair_is_a_coverage_channel`,
#     so it survives this revert.
#   * The referee columns are worth nothing even before the serving-asymmetry
#     gate is applied. Holding them out of the model moves torch from 0.6474
#     to 0.6481 and XGB from 0.6499 to 0.6498 -- inside seed noise on the
#     screen. The gate would have blocked them regardless: they are present on
#     97.7% of training rows and on 0% of servable ones, because Wikipedia
#     cards do not name a referee.
#
# `mma.context` owns the derivations: the prior-fights-only referee pass, the
# event-country parser and its nationality/location normalisation, and the
# bonus fight-id loader. The rest of the wiring is `bonus_wins` in
# `mma.history._FighterState`, the `home_country`/`event_country` derivations
# in `mma.features._side_frame`, the referee and home-advantage context in
# `build_features`, and the `referee` / `referee_rates` / `event_country`
# arguments to `mma.inference.build_matchup`; see the SP2 plan's Task 8 notes.
#
# SHIPPING FORM (`context_nohome`): `home_country_a`/`_b` stay in the TABLE
# and are held out of both model matrices, because they FAIL the constant-
# vector leak check -- over the rows where exactly one corner is mapped by the
# `external` snapshot the pair takes three distinct values and the mapped
# corner wins 0.745 of them. That is not a tuning choice; it is the same
# coverage-selection leak `external_missing_a`/`_b` was corrected for. Both
# exclusion lists still name the pair, so re-registering needs no change there.
#
# SP2.1 RESULT (2026-09-08): UNREGISTERED AGAIN. The pre-registered capacity
# experiment combined this block with the other three whose torch point
# estimate was negative or whose scorers disagreed into S1 (11,238 x 87) and
# ran one 25-configuration torch architecture search over both S1 (arm T) and
# the shipped S0 (arm C, the control that separates a feature effect from an
# architecture effect). Neither arm cleared the 0.003 bar against the deployed
# incumbent -- C -0.0013, T -0.0012 -- and T - C = +0.0001 is inside
# sigma_seed = 0.000346, so the pre-registration's fourth attribution branch
# applied: the blocks are dead and the architecture is adequate; record and
# revert. The only result that cleared any bar, A1 on XGBoost at -0.0039 on
# seed 0, failed its fresh-seed confirmation (mean -0.0024 over seeds 1-3
# against a newly measured XGB sigma_seed of 0.00087-0.00125). Nothing shipped
# and nothing was redeployed. See
# docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md and
# models/walkforward/sp2_1_decision.json.
#
# `mma.context` and the wiring named above sit behind `if CONTEXT_BLOCK in
# resolved` guards, live again now that the registration below is permanent.
# `home_country_a` / `home_country_b` stay in the table and out of both model
# matrices: the pair FAILS the constant-vector leak check outright -- it is the
# `external` coverage-selection artifact in a second channel -- and the shipping
# form of this block is therefore the `context_nohome` one.
CONTEXT_BLOCK = "context"

# SP2.2 RESULT (2026-09-08): **REGISTERED PERMANENTLY -- THIS BLOCK SHIPS.**
# Not as a block that finally cleared a bar on its own: it never did, and the
# SP2.1 verdict above stands unedited. It ships as one sixth of S1
# (`base,external,trajectory,notice,context,opponent_adjusted`, 11,238 x 87),
# the feature table SP2.2's candidate B1 was scored on. B1 -- the equal-weight,
# temperature-scaled blend of a 5-seed XGBoost ensemble and the 5-seed torch
# ensemble -- cleared the 0.003 bar at seeds 0-4 (-0.0039 against I) and again
# at seeds 5-9 (-0.0037 against its own fresh-seed paired incumbent), and the
# amended ECE gate passes. What SP2.1 could not see is that these blocks are
# alive THROUGH THE XGBOOST MEMBER: the XGB arm on S1 is 0.6462 against 0.6490
# on S0, while the torch arm is 0.6475 either way. The blocks are dead to the
# MLP and were never dead to the trees. See
# docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md and
# models/walkforward/sp2_2_decision.json.
#
# Re-verified before it was relied on: rebuilt on the feature set the committed
# SP2 report's own config names, this registration reproduces that report
# EXACTLY -- pooled, all eight folds, all slices and every per-fold fit budget.
register(Block(
    name=CONTEXT_BLOCK,
    differentials=(("bonus_rate", "bonus_rate"),),
    booleans=("home_country",),
    fight_level=("referee_finish_rate", "referee_decision_rate",
                 "referee_missing", "home_country_unknown"),
))


# --- SP2 block: opponent_adjusted (REJECTED IN SP2/SP2.1; SHIPS IN SP2.2) ---
# Each core rate priced against the opposition it was produced against: the
# career mean of (own value in a fight) minus (what that fight's opponent had
# historically ALLOWED before it), plus the mean pre-fight Elo of the
# opponents a fighter has beaten and lost to.
#
# Built, evaluated and reverted in SP2 Task 6, and the block the SP2.1
# capacity hypothesis is really about: XGB screened at pooled winner LL 0.6519
# against a 0.6537 incumbent (-0.0018, better) while torch came in at 0.6532
# against 0.6510 (+0.0022, worse), with seven of eight folds regressing. Seven
# columns heavily collinear with the base rates they derive from is exactly
# the shape a tree ensemble can exploit per split and a fixed-width MLP cannot.
#
# The accumulators live in `mma.history._FighterState`: `allowed()` is the
# expectation side (what a fighter's opponents have achieved on them, mirrored
# rate by rate) and both corners' `allowed()` are read before either corner is
# updated, so the value used is the opponent's PRE-fight one. `mma.snapshots`
# carries the same fields for serving, which is why no separate serving
# derivation is needed here.
#
# SP2.1 RESULT (2026-09-08): UNREGISTERED AGAIN. The pre-registered capacity
# experiment combined this block with the other three whose torch point
# estimate was negative or whose scorers disagreed into S1 (11,238 x 87) and
# ran one 25-configuration torch architecture search over both S1 (arm T) and
# the shipped S0 (arm C, the control that separates a feature effect from an
# architecture effect). Neither arm cleared the 0.003 bar against the deployed
# incumbent -- C -0.0013, T -0.0012 -- and T - C = +0.0001 is inside
# sigma_seed = 0.000346, so the pre-registration's fourth attribution branch
# applied: the blocks are dead and the architecture is adequate; record and
# revert. The only result that cleared any bar, A1 on XGBoost at -0.0039 on
# seed 0, failed its fresh-seed confirmation (mean -0.0024 over seeds 1-3
# against a newly measured XGB sigma_seed of 0.00087-0.00125). Nothing shipped
# and nothing was redeployed. See
# docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md and
# models/walkforward/sp2_1_decision.json.
#
# SP2's revert of this block was TOTAL: the accumulators survived in no git
# tree, and SP2.1 Task 1 had to reconstruct them from the plan's written
# contract rather than recover them with `git show`. That is why they were kept
# live through SP2.1's second revert, and it is why the block could be scored
# at all in SP2.2. The implementation lives in `mma.history`
# (`VS_EXPECTATION_RATES`, `_FighterState.allowed` and the `*_vs_exp` /
# `avg_opp_elo_*` accumulators) and in `mma.snapshots`, computed
# unconditionally like every base rate, with unit tests in
# `tests/test_history.py` and `tests/test_snapshots.py` that call
# `build_history` / `build_snapshots` directly.
OPPONENT_ADJUSTED_BLOCK = "opponent_adjusted"

# SP2.2 RESULT (2026-09-08): **REGISTERED PERMANENTLY -- THIS BLOCK SHIPS.**
# Not as a block that finally cleared a bar on its own: it never did, and the
# SP2.1 verdict above stands unedited. It ships as one sixth of S1
# (`base,external,trajectory,notice,context,opponent_adjusted`, 11,238 x 87),
# the feature table SP2.2's candidate B1 was scored on. B1 -- the equal-weight,
# temperature-scaled blend of a 5-seed XGBoost ensemble and the 5-seed torch
# ensemble -- cleared the 0.003 bar at seeds 0-4 (-0.0039 against I) and again
# at seeds 5-9 (-0.0037 against its own fresh-seed paired incumbent), and the
# amended ECE gate passes. What SP2.1 could not see is that these blocks are
# alive THROUGH THE XGBOOST MEMBER: the XGB arm on S1 is 0.6462 against 0.6490
# on S0, while the torch arm is 0.6475 either way. The blocks are dead to the
# MLP and were never dead to the trees. See
# docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md and
# models/walkforward/sp2_2_decision.json.
#
# Re-verified before it was relied on: rebuilt on the feature set the committed
# SP2 report's own config names, this registration reproduces that report
# EXACTLY -- pooled, all eight folds, all slices and every per-fold fit budget.
register(Block(
    name=OPPONENT_ADJUSTED_BLOCK,
    differentials=(
        ("sig_pm_vs_exp", "sig_pm_vs_exp"),
        ("sig_absorbed_pm_vs_exp", "sig_absorbed_pm_vs_exp"),
        ("td_landed_pf_vs_exp", "td_landed_pf_vs_exp"),
        ("td_def_vs_exp", "td_def_vs_exp"),
        ("ctrl_share_vs_exp", "ctrl_share_vs_exp"),
        ("avg_opp_elo_wins", "avg_opp_elo_wins"),
        ("avg_opp_elo_losses", "avg_opp_elo_losses"),
    ),
))
