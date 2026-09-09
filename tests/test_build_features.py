"""Which blocks a bare `python scripts/build_features.py` builds.

The weekly refresh Action runs it bare (.github/workflows/refresh-data.yml,
"Rebuild features"), so its default IS the deployed feature contract. A
default of "base" would silently rebuild the shipped table without the blocks
the deployed model was trained and measured on -- no crash, because the
preprocessor is re-fit at train time; just a quietly worse model.
"""
from __future__ import annotations

import json

import pytest

import scripts.build_features as bf


def _sidecar(tmp_path, blocks):
    path = tmp_path / "features_blocks.json"
    path.write_text(json.dumps({"blocks": list(blocks), "n_rows": 1, "n_columns": 1}))
    return path


def test_a_bare_run_builds_the_blocks_the_committed_table_was_built_from(tmp_path):
    sidecar = _sidecar(tmp_path, ["base", "external", "trajectory"])
    assert bf.resolve_blocks_arg(None, sidecar=sidecar) == ("base", "external", "trajectory")


def test_a_bare_run_falls_back_to_base_when_no_table_has_been_built(tmp_path):
    assert bf.resolve_blocks_arg(None, sidecar=tmp_path / "absent.json") == ("base",)


def test_an_explicit_blocks_flag_still_wins(tmp_path):
    sidecar = _sidecar(tmp_path, ["base", "external", "trajectory"])
    assert bf.resolve_blocks_arg("base,external", sidecar=sidecar) == ("base", "external")
    # base is implied, as it always is
    assert bf.resolve_blocks_arg("external", sidecar=sidecar) == ("base", "external")


def test_an_unknown_block_is_a_loud_error_not_a_silent_base_table(tmp_path):
    with pytest.raises(ValueError, match="unknown feature block"):
        bf.resolve_blocks_arg("nope", sidecar=tmp_path / "absent.json")


def test_the_default_matches_the_committed_sidecar_in_this_tree():
    """The real check: a bare run in this checkout rebuilds the shipped
    contract, whatever that currently is."""
    from mma.feature_blocks import table_blocks

    assert bf.resolve_blocks_arg(None) == table_blocks()


def test_parse_args_leaves_the_default_unresolved():
    assert bf.parse_args([]).blocks is None
    assert bf.parse_args(["--blocks", "external"]).blocks == "external"
