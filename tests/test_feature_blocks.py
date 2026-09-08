"""The feature-block registry: what `--blocks` may name and what it produces.

SP2 adds signal in named blocks so each can be judged separately against the
walk-forward bar. These tests pin the two invariants the machinery rests on:
`base` (the v1 contract) is always on, and no two blocks claim the same
output column.
"""
import pandas as pd
import pytest

from mma.feature_blocks import (
    BASE_BLOCK, BLOCKS, Block, columns_for, columns_of, register, registry,
    resolve_blocks, state_keys,
)


def test_base_block_is_always_present():
    assert resolve_blocks([]) == (BASE_BLOCK,)
    assert resolve_blocks([BASE_BLOCK]) == (BASE_BLOCK,)


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown feature block"):
        resolve_blocks(["not_a_block"])


def test_every_block_declares_columns_and_they_are_unique():
    """No two blocks may claim the same output column, and every block must
    contribute at least one.

    Checked with `columns_of`, which is a single block's own columns.
    `columns_for([name])` always resolves `base` in alongside it, so
    comparing those would report the second and every later block as
    redeclaring all of base's columns -- a spurious failure that appears the
    moment a second block is registered.
    """
    claimed: dict[str, str] = {}
    for name, block in BLOCKS.items():
        own = columns_of(block)
        assert own, f"block {name} declares no columns of its own"
        collisions = sorted(set(own) & set(claimed))
        assert not collisions, f"block {name} redeclares {collisions}"
        claimed.update({column: name for column in own})


def test_column_uniqueness_survives_a_second_block(dummy_blocks):
    """The guard above must still hold with more than one block registered --
    the case the `columns_for`-based version got wrong."""
    test_every_block_declares_columns_and_they_are_unique()


def test_register_rejects_a_block_that_reclaims_a_column():
    """The collision check belongs to `register`, so a bad block cannot enter
    the registry at all and be discovered later by a test."""
    with registry():
        with pytest.raises(ValueError, match="career_fights_diff"):
            register(Block(
                name="clash",
                differentials=(("career_fights", "career_fights"),),
            ))
        assert "clash" not in BLOCKS


def test_register_rejects_a_block_that_redeclares_its_own_column():
    """A block declaring the same output column twice within itself must be
    caught here too, not just against other registered blocks -- otherwise
    it registers cleanly and then breaks the documented `columns_for` ==
    `feature_row` order contract."""
    with registry():
        with pytest.raises(ValueError, match="dup_stem_diff"):
            register(Block(
                name="self_clash",
                differentials=(("k1", "dup_stem"), ("k2", "dup_stem")),
            ))
        assert "self_clash" not in BLOCKS


def test_resolve_blocks_accepts_a_bare_string(dummy_blocks):
    """A single name is a common slip and used to iterate its characters,
    failing with a list of unknown one-letter blocks."""
    first, _ = dummy_blocks
    assert resolve_blocks(first) == resolve_blocks([first])
    assert resolve_blocks(BASE_BLOCK) == (BASE_BLOCK,)
    with pytest.raises(ValueError, match="unknown feature block"):
        resolve_blocks("not_a_block")


def test_registry_context_manager_restores_the_registry():
    original = dict(BLOCKS)
    with pytest.raises(ValueError):
        with registry():
            register(Block(name="temp_block", absolutes=("temp_stat",)))
            raise ValueError("setup blew up after a partial registration")
    assert BLOCKS == original


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
    # Column SETS matching is not enough -- a registry reorder would
    # silently change the committed parquet's physical column order with a
    # green suite. Pin the order too.
    non_identifier_columns = [c for c in table.columns if c not in identifiers]
    assert non_identifier_columns == list(columns_for(blocks))


@pytest.fixture
def dummy_blocks():
    """Two throwaway blocks registered after `base`, removed afterwards.

    `registry()` wraps the registrations too, so a `register` that raises
    during setup cannot leak the block that went in before it."""
    with registry():
        register(Block(name="zzz_second", absolutes=("zzz_second_stat",)))
        register(Block(name="aaa_first", booleans=("aaa_first_flag",)))
        yield ("aaa_first", "zzz_second")


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


def test_columns_for_order_holds_for_a_block_with_fight_level_columns():
    """`columns_for` places fight-level columns at their block's registry
    position, but they arrive in `context` and so land at the front of the
    row. `feature_row` re-emits them in position; without that the documented
    "columns_for order == feature_row order" contract is false for any block
    declaring fight_level."""
    from mma import serving

    with registry():
        register(Block(
            name="ctx_block",
            absolutes=("ctx_stat",),
            fight_level=("ctx_level", "ctx_other"),
        ))
        state = {key: 1.0 for key in state_keys(["ctx_block"])}
        context = {"weight_class": "Lightweight", "ctx_level": 7, "ctx_other": 2}
        row = serving.feature_row(state, dict(state), context, blocks=["ctx_block"])
        emitted = [c for c in row if c != "weight_class"]
        assert tuple(emitted) == columns_for(["ctx_block"])
        assert row["ctx_level"] == 7 and row["ctx_other"] == 2

        with pytest.raises(KeyError, match="ctx_other"):
            serving.feature_row(state, dict(state),
                                {"ctx_level": 7}, blocks=["ctx_block"])
