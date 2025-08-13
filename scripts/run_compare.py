#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_pipeline.config.global_config import GlobalConfig
from risk_pipeline.core.evaluator import evaluate_all, build_models
from risk_pipeline.data.dataset import CanonicalDataset


def parse_args():
    p = argparse.ArgumentParser(description="Run fair model comparison")
    p.add_argument("--models", type=str, default="lstm,stockmixer,xgb,arima", help="Comma-separated models")
    p.add_argument("--artifacts", type=str, default="artifacts/compare_run", help="Artifacts directory")
    return p.parse_args()


def main():
    args = parse_args()
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = GlobalConfig(models_to_run=models_to_run, artifacts_dir=args.artifacts)

    # Load demo dataset
    df, y = CanonicalDataset.load_demo()

    # Evaluate
    models = build_models(cfg)
    res = evaluate_all(models, df, y, cfg)

    # Save config dump
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.artifacts_dir) / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(res.sort_values("mse_mean").to_string(index=False))
    print(f"\nSee {cfg.artifacts_dir}/results.csv and RESULTS.md")


if __name__ == "__main__":
    main()
