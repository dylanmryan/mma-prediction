"""The feature-block registry: what `--blocks` may name and what it produces.

SP2 adds signal in named blocks so each can be judged separately against the
walk-forward bar. These tests pin the two invariants the machinery rests on:
`base` (the v1 contract) is always on, and no two blocks claim the same
output column.
"""
import pandas as pd
import pytest

from mma.feature_blocks import (
    BASE_BLOCK, BLOCKS, Block, columns_for, register, resolve_blocks,
    state_keys,
)


def test_base_block_is_always_present():
    assert resolve_blocks([]) == (BASE_BLOCK,)
    assert resolve_blocks([BASE_BLOCK]) == (BASE_BLOCK,)


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown feature block"):
        resolve_blocks(["not_a_block"])


def test_every_block_declares_columns_and_they_are_unique():
    """No two blocks may claim the same output column, and every block must
    contribute at least one. `columns_for` concatenates the resolved blocks
    in order, so asking for all of them at once surfaces a collision as a
    duplicate entry -- including a collision with `base`, which is implicit
    in every single-block resolve and so cannot be spotted by comparing
    single-block results against each other.
    """
    names = list(BLOCKS)
    all_columns = columns_for(names)
    duplicates = sorted({c for c in all_columns if all_columns.count(c) > 1})
    assert not duplicates, f"blocks redeclare {duplicates}"
    seen: set[str] = set()
    for name in names:
        own = set(columns_for([name])) - seen
        assert own, f"block {name} declares no columns of its own"
        seen |= own


def test_registry_columns_match_the_committed_feature_table():
    """Whatever blocks the committed table was built from -- recorded in the
    sidecar -- the registry must name exactly its feature columns."""
    import json
    from pathlib import Path
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (processed / "features.parquet").exists():
        pytest.skip("processed data not built")
    blocks = json.loads((processed / "features_blocks.json").read_text())["blocks"]
    table = pd.read_parquet(processed / "features.parquet")
    identifiers = {"fight_id", "date", "swapped", "y_winner", "y_method", "y_finish_round",
                   "weight_class", "title_fight", "scheduled_rounds"}
    assert set(columns_for(blocks)) == set(table.columns) - identifiers


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


def test_state_keys_are_cumulative_and_deduplicated(dummy_blocks):
    first, second = dummy_blocks
    base_keys = state_keys([BASE_BLOCK])
    # `age` and `career_fights` are both a differential source and an absolute
    # in `base`; each state key is named once.
    assert len(base_keys) == len(set(base_keys))
    assert "age" in base_keys and "career_fights" in base_keys

    with_dummies = state_keys([first, second])
    assert set(base_keys) < set(with_dummies)
    assert set(with_dummies) - set(base_keys) == {"zzz_second_stat", "aaa_first_flag"}
    # registry order, not requested order
    assert state_keys([first, second]) == state_keys([second, first])
    assert with_dummies[:len(base_keys)] == base_keys


def test_feature_row_raises_when_a_declared_state_key_is_absent():
    """A key nothing provides must be a loud error, not an all-NaN column:
    a half-wired block would otherwise measure as 'no improvement'."""
    from mma import serving

    full = {key: 1.0 for key in state_keys([BASE_BLOCK])}
    # dict states (serving): key absent from the mapping
    partial = {k: v for k, v in full.items() if k != "td_def"}
    with pytest.raises(KeyError, match="td_def"):
        serving.feature_row(partial, dict(full))
    with pytest.raises(KeyError, match=BASE_BLOCK):
        serving.feature_row(dict(full), partial)

    # frame states (training): column absent from `.columns`
    frame = pd.DataFrame({k: [1.0] for k in full})
    with pytest.raises(KeyError, match="southpaw"):
        serving.feature_row(frame.drop(columns=["southpaw"]), frame)


def test_build_features_raises_for_a_state_key_no_source_table_provides(dummy_blocks):
    """The registry single-sources values, not just names: registering a block
    whose state key no source table carries fails the build."""
    from mma.features import build_features
    from tests.test_features import _tables

    with pytest.raises(ValueError, match="zzz_second_stat"):
        build_features(*_tables(), blocks=["zzz_second"])
