"""Feature blocks: named, independently evaluable groups of columns.

SP2 adds signal in blocks so each can be judged separately against the
walk-forward bar (see docs/superpowers/plans/2026-09-07-sp2-features-v3.md).
`BASE_BLOCK` is the v1 feature set and is always enabled. A block declares
the state fields it consumes and the output columns it produces; the
registry is what `scripts/build_features.py --blocks` and the serving path
both read, so a block cannot be half-enabled.

`spec_for` flattens a requested block list into the one assembled spec
`mma.serving.feature_row` walks, and `columns_for` names the columns that
walk emits, in the order it emits them -- the registry is the contract, not
a parallel description of one.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE_BLOCK = "base"


@dataclass(frozen=True)
class Block:
    """One switchable group of feature columns.

    `differentials` are (state key, output stem) pairs emitted as
    "<stem>_diff" (A minus B); `absolutes` and `booleans` are state keys
    emitted per corner as "<stem>_a"/"<stem>_b" (booleans coerced to strict
    bools); `derived_booleans` are (output column, per-corner stem) pairs
    emitted as the XOR of that stem's two corners; `fight_level` columns
    describe the fight rather than a corner and are passed through the
    context verbatim. `external` marks a block fed by a source outside the
    UFCStats scrape, which is what the `external_missing` evaluation slice
    keys off.
    """

    name: str
    differentials: tuple[tuple[str, str], ...] = ()
    absolutes: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    derived_booleans: tuple[tuple[str, str], ...] = ()
    fight_level: tuple[str, ...] = ()
    external: bool = False


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
    if block.name in BLOCKS:
        raise ValueError(f"duplicate feature block {block.name}")
    BLOCKS[block.name] = block
    return block


def resolve_blocks(names) -> tuple[str, ...]:
    """Normalise a requested block list: base first, then registry order.

    Order-insensitive by construction, so two runs asking for the same set
    of blocks in different orders build the identical table."""
    requested = set(names) | {BASE_BLOCK}
    unknown = sorted(requested - set(BLOCKS))
    if unknown:
        raise ValueError(f"unknown feature block(s): {unknown}; known: {sorted(BLOCKS)}")
    return tuple([BASE_BLOCK] + [n for n in BLOCKS if n != BASE_BLOCK and n in requested])


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


def columns_for(names) -> tuple[str, ...]:
    """Output columns for a requested block list, in emission order.

    Mirrors `mma.serving.feature_row` exactly: every differential, then the
    per-corner absolutes and booleans (corner a, then corner b), then the
    derived flags, then the fight-level columns."""
    spec = spec_for(names)
    columns = [f"{stem}_diff" for _, stem in spec.differentials]
    columns += [
        f"{stem}_{corner}"
        for corner in ("a", "b")
        for stem in spec.absolutes + spec.booleans
    ]
    columns += [name for name, _ in spec.derived_booleans]
    columns += list(spec.fight_level)
    return tuple(columns)


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
