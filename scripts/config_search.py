"""Score the SP2.1 pre-registered 25-configuration search space on one arm.

    python scripts/config_search.py --arm C --blocks base,external \
        --drop-columns external_missing,same_country

The space is fixed by
docs/superpowers/plans/2026-09-08-sp2-1-capacity-experiment.md and is not a
parameter of this script: 25 configurations drawn from ``rng(0)`` over
hidden / dropout / lr / weight_decay / embedding_dim / method_scale /
round_scale. The *same* 25 configurations must score both arms -- the arms
differ only in their feature set, so a re-sample would turn the comparison
into two unrelated searches. They are therefore written once to
``models/walkforward/search/configs.json`` and every later run re-samples
and refuses to start if the file disagrees.

Scoring is not reimplemented here: each configuration is handed to
``scripts/run_walkforward.py`` as a subprocess (``--candidate torch
--config-json ... --seeds 0,1 --out-dir models/walkforward/search/<arm>``),
so a search report is the same artefact, written by the same code, as any
other walk-forward report. Runs are resumable -- a configuration whose
report already exists is skipped -- and the ranked ``summary.json`` is
rebuilt from the reports on disk at the end of every run.

This script builds no features. The table on disk must already be the arm's
feature set; the blocks sidecar is checked against ``--blocks`` and a
mismatch is a hard error rather than 25 wasted runs on the wrong table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.feature_blocks import resolve_blocks, table_blocks  # noqa: E402

SEARCH_DIR = ROOT / "models" / "walkforward" / "search"
CONFIGS_PATH = SEARCH_DIR / "configs.json"
RUN_WALKFORWARD = ROOT / "scripts" / "run_walkforward.py"

# The pre-registered space. Editing any of this after the arms have run
# would invalidate the experiment; the configs.json check below is what
# makes such an edit fail loudly instead of silently re-searching.
N_CONFIGS = 25
SAMPLE_SEED = 0
RANK_SEEDS = "0,1"
HIDDEN_CHOICES = ((128, 64), (256, 128), (256, 128, 64), (384, 192), (512, 256))
EMBEDDING_CHOICES = (4, 8, 16)
DROPOUT_RANGE = (0.2, 0.5)
LR_RANGE = (3e-4, 3e-3)
WEIGHT_DECAY_RANGE = (1e-5, 3e-3)
METHOD_SCALE_RANGE = (0.25, 0.75)
ROUND_SCALE_RANGE = (0.1, 0.4)


def _log_uniform(rng, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_configs(n: int = N_CONFIGS, seed: int = SAMPLE_SEED) -> list[dict]:
    """The pre-registered configurations, in draw order.

    One config is drawn per loop iteration in the order the pre-registration
    lists the parameters, so the sequence is fixed by ``seed`` alone and the
    first k configs of an n-config draw are the first k of any larger draw.
    ``hidden`` is a list rather than a tuple so a config compares equal to
    itself after a JSON round-trip (``resolve_config`` coerces it back).
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({
            "hidden": list(HIDDEN_CHOICES[int(rng.integers(len(HIDDEN_CHOICES)))]),
            "dropout": float(rng.uniform(*DROPOUT_RANGE)),
            "lr": _log_uniform(rng, *LR_RANGE),
            "weight_decay": _log_uniform(rng, *WEIGHT_DECAY_RANGE),
            "embedding_dim": int(EMBEDDING_CHOICES[int(rng.integers(len(EMBEDDING_CHOICES)))]),
            "method_scale": float(rng.uniform(*METHOD_SCALE_RANGE)),
            "round_scale": float(rng.uniform(*ROUND_SCALE_RANGE)),
        })
    return configs


def load_or_write_configs(path: Path = CONFIGS_PATH) -> list[dict]:
    """The shared configurations: written on the first run, verified after.

    Both arms read this file. If the sampler ever drifts from what the first
    arm actually ran, the arms would no longer differ only in their feature
    set and the comparison would be meaningless -- so a disagreement stops
    the run rather than being repaired.
    """
    configs = sample_configs()
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != configs:
            raise SystemExit(
                f"{path} disagrees with the sampler: the stored configurations were "
                "used by an arm that has already run, so re-sampling would leave the "
                "arms searching different spaces. Restore the sampler or delete the "
                "whole search directory and start over."
            )
        return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configs, indent=2) + "\n")
    return configs


def check_table(blocks: str) -> tuple[str, ...]:
    """Refuse to search against a table built from the wrong blocks."""
    requested = resolve_blocks([b.strip() for b in blocks.split(",") if b.strip()])
    on_disk = table_blocks()
    if requested != on_disk:
        raise SystemExit(
            f"data/processed/features.parquet was built from {list(on_disk)}, but this "
            f"arm asks for {list(requested)}; run scripts/build_features.py --blocks "
            f"{','.join(requested)} first"
        )
    return requested


def run_config(index: int, config: dict, out_dir: Path, drop_columns: str | None,
               seeds: str = RANK_SEEDS) -> Path:
    """Score one configuration by shelling out to the walk-forward harness."""
    report = out_dir / f"config_{index:02d}.json"
    cmd = [
        sys.executable, str(RUN_WALKFORWARD),
        "--candidate", "torch",
        "--name", report.stem,
        "--seeds", seeds,
        "--config-json", json.dumps(config),
        "--out-dir", str(out_dir),
    ]
    if drop_columns:
        cmd += ["--drop-columns", drop_columns]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL)
    return report


def _repo_path(path: Path) -> str:
    """Repo-relative when the path is inside the repo, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def summarise(arm: str, blocks: tuple[str, ...], drop_columns: str | None,
              configs: list[dict], out_dir: Path) -> dict:
    """Rank every config that has a report, and describe the distribution.

    The spread matters as much as the winner: best-of-25 on a metric whose
    seed noise is 0.00035 buys some of its "best" from selection, and the
    min/median/max is what makes that visible.
    """
    rows = []
    for index, config in enumerate(configs):
        path = out_dir / f"config_{index:02d}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        rows.append({
            "index": index,
            "report": _repo_path(path),
            "config": config,
            "pooled_winner_log_loss": report["pooled"]["winner_log_loss"],
            "pooled": report["pooled"],
            "folds": {y: f["winner_log_loss"] for y, f in report["folds"].items()},
            "runtime_sec": report["config"]["runtime_sec"],
        })
    rows.sort(key=lambda r: (r["pooled_winner_log_loss"], r["index"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    losses = [r["pooled_winner_log_loss"] for r in rows]
    return {
        "arm": arm,
        "blocks": list(blocks),
        "drop_columns": [c.strip() for c in drop_columns.split(",")] if drop_columns else [],
        "seeds": RANK_SEEDS,
        "n_configs": len(configs),
        "n_scored": len(rows),
        "distribution": {
            "min": min(losses), "median": float(np.median(losses)), "max": max(losses),
        } if losses else None,
        "best": rows[0] if rows else None,
        "ranked": rows,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, help="arm label; names the output directory")
    parser.add_argument("--blocks", required=True,
                        help="the blocks the table on disk must have been built from")
    parser.add_argument("--drop-columns", default=None,
                        help="passed through to run_walkforward.py unchanged")
    parser.add_argument("--search-dir", type=Path, default=SEARCH_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    blocks = check_table(args.blocks)
    configs = load_or_write_configs(args.search_dir / "configs.json")
    out_dir = args.search_dir / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"arm {args.arm}: {len(configs)} configs x seeds {RANK_SEEDS} on {list(blocks)}")
    print(f"drop-columns: {args.drop_columns or '(none)'}")
    started = time.time()
    best = (None, float("inf"))
    for index, config in enumerate(configs):
        report = out_dir / f"config_{index:02d}.json"
        if report.exists():
            status = "skip"
        else:
            run_config(index, config, out_dir, args.drop_columns)
            status = "ran "
        loss = json.loads(report.read_text())["pooled"]["winner_log_loss"]
        if loss < best[1]:
            best = (index, loss)
        print(f"  [{index + 1:2d}/{len(configs)}] {status} config_{index:02d} "
              f"hidden={config['hidden']} dropout={config['dropout']:.3f} "
              f"lr={config['lr']:.2e} wd={config['weight_decay']:.2e} "
              f"emb={config['embedding_dim']} -> {loss:.4f} "
              f"| best config_{best[0]:02d} {best[1]:.4f} "
              f"| {time.time() - started:6.0f}s",
              flush=True)

    summary = summarise(args.arm, blocks, args.drop_columns, configs, out_dir)
    summary["runtime_sec"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "distribution": summary["distribution"],
                      "best_index": summary["best"]["index"],
                      "best_config": summary["best"]["config"],
                      "best_pooled": summary["best"]["pooled_winner_log_loss"]}, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
