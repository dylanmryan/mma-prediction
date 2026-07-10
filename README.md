# MMA Fight Prediction

Predicting UFC fight winners, method of victory, and finish round.
Elo baseline -> XGBoost -> PyTorch multi-task net, honestly evaluated.

Work in progress. Design: `docs/superpowers/specs/2026-07-10-mma-prediction-design.md`

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py   # Kaggle UFC dataset -> data/raw/
.venv/bin/python scripts/make_dataset.py    # clean parquet -> data/processed/
.venv/bin/pytest
```
