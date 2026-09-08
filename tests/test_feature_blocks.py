"""The feature-block registry: what `--blocks` may name and what it produces.

SP2 adds signal in named blocks so each can be judged separately against the
walk-forward bar. These tests pin the two invariants the machinery rests on:
`base` (the v1 contract) is always on, and no two blocks claim the same
output column.
"""
import pandas as pd
import pytest

from mma.feature_blocks import BASE_BLOCK, BLOCKS, Block, columns_for, register, resolve_blocks


def test_base_block_is_always_present():
    assert resolve_blocks([]) == (BASE_BLOCK,)
    assert resolve_blocks([BASE_BLOCK]) == (BASE_BLOCK,)


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown feature block"):
        resolve_blocks(["not_a_block"])


def test_every_block_declares_columns_and_they_are_unique():
    seen = set()
    for name in BLOCKS:
        cols = columns_for([name])
        assert cols, f"block {name} declares no columns"
        overlap = seen & set(cols)
        assert not overlap, f"block {name} redeclares {overlap}"
        seen |= set(cols)


def test_base_columns_match_the_committed_feature_table():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
    if not path.exists():
        pytest.skip("processed data not built")
    table = pd.read_parquet(path)
    identifiers = {"fight_id", "date", "swapped", "y_winner", "y_method", "y_finish_round",
                   "weight_class", "title_fight", "scheduled_rounds"}
    assert set(columns_for([BASE_BLOCK])) == set(table.columns) - identifiers


@pytest.fixture
def dummy_blocks():
    """Two throwaway blocks registered after `base`, removed afterwards."""
    original = dict(BLOCKS)
    register(Block(name="zzz_second", absolutes=("zzz_second_stat",)))
    register(Block(name="aaa_first", booleans=("aaa_first_flag",)))
    try:
        yield ("aaa_first", "zzz_second")
    finally:
        BLOCKS.clear()
        BLOCKS.update(original)


def test_resolve_blocks_is_order_insensitive_and_follows_registry_order(dummy_blocks):
    first, second = dummy_blocks
    # requested in either order -> same answer, and it is registration order
    # (base, then zzz_second, then aaa_first), not the order asked for
    assert resolve_blocks([first, second]) == resolve_blocks([second, first])
    assert resolve_blocks([first, second]) == (BASE_BLOCK, "zzz_second", "aaa_first")
    assert columns_for([first, second]) == columns_for([second, first])


def test_columns_for_is_the_column_order_feature_row_emits():
    """The registry is the contract, so it must agree with the builder."""
    from mma import serving

    state = {key: 1.0 for block in BLOCKS.values() for key, _ in block.differentials}
    state.update({stem: 1.0 for stem in BLOCKS[BASE_BLOCK].absolutes})
    state.update({stem: True for stem in BLOCKS[BASE_BLOCK].booleans})
    row = serving.feature_row(state, dict(state), {"weight_class": "Lightweight"})
    emitted = [c for c in row if c != "weight_class"]
    assert tuple(emitted) == columns_for([BASE_BLOCK])
