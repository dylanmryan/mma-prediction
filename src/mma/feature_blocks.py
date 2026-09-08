"""Feature blocks: named, independently evaluable groups of columns.

SP2 adds signal in blocks so each can be judged separately against the
walk-forward bar (see docs/superpowers/plans/2026-09-07-sp2-features-v3.md).
`BASE_BLOCK` is the v1 feature set and is always enabled. A block declares
the state fields it consumes and the output columns it produces; the
registry is what `scripts/build_features.py --blocks` and the serving path
both read, so a block cannot be half-enabled.

`spec_for` flattens a requested block list into the one assembled spec
`mma.serving.feature_row` walks; `columns_for` names the columns that walk
emits, in the order it emits them, and `state_keys` names the state fields
it reads -- the registry is the contract, not a parallel description of one.
Both the training merge (`mma.features._side_frame`) and the serving state
dict (`mma.inference.build_matchup`) are built from `state_keys`, so a block
whose state key nothing provides raises instead of quietly producing an
all-NaN column that then measures as "no improvement".
"""
from __future__ import annotations

from contextlib import contextmanager
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
    table, which is the same class of bug as a state key nothing provides."""
    if block.name in BLOCKS:
        raise ValueError(f"duplicate feature block {block.name}")
    claimed = {column: name for name, other in BLOCKS.items()
               for column in columns_of(other)}
    collisions = sorted(set(columns_of(block)) & set(claimed))
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
