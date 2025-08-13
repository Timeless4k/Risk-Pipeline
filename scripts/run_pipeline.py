#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add the parent directory to Python path to allow importing risk_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from risk_pipeline.config.global_config import GlobalConfig
from risk_pipeline.core.evaluator import evaluate_all, build_models
from risk_pipeline.core.io_utils import write_atomic, tee_logger, setup_basic_logging
from risk_pipeline.data.dataset import CanonicalDataset


def parse_args():
    p = argparse.ArgumentParser(description="Run the full fair pipeline: data -> features -> splits -> train/eval -> artifacts")
    p.add_argument("--data", choices=["demo", "csv"], default="demo", help="Use built-in demo data or provide a CSV")
    p.add_argument("--csv-path", type=str, default=None, help="Path to CSV with columns Close,High,Low and a datetime index/column")
    p.add_argument("--date-col", type=str, default=None, help="Name of datetime column if CSV index is not datetime")
    p.add_argument("--models", type=str, default="lstm,stockmixer,xgb,arima", help="Comma-separated models to run")
    p.add_argument("--artifacts", type=str, default="artifacts/compare_run", help="Directory to store results and logs")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU only (unset CUDA_VISIBLE_DEVICES)")
    p.add_argument("--dry-run", action="store_true", help="Build first split, print shapes, and exit without training")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--log-file", type=str, default=None, help="Log file to tee stdout/stderr")
    p.add_argument("--num-workers", type=int, default=0, help="Optional parallel workers for non-iterative models")
    return p.parse_args()


def load_csv(csv_path: str, date_col: str | None) -> pd.DataFrame:
    if date_col:
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        df = df.set_index(date_col)
    else:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Expect Close/High/Low
    missing = {c for c in ("Close", "High", "Low") if c not in df.columns}
    if missing:
        raise ValueError(f"CSV must contain columns Close,High,Low. Missing: {missing}")
    return df.sort_index()


def main():
    args = parse_args()
    setup_basic_logging(verbose=args.verbose)
    if args.log_file:
        tee_logger(args.log_file)

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] Forcing CPU-only mode (CUDA_VISIBLE_DEVICES cleared)")

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = GlobalConfig(models_to_run=models_to_run, artifacts_dir=args.artifacts)

    # Load data
    if args.data == "demo":
        df, y = CanonicalDataset.load_demo()
    else:
        if not args.csv_path:
            print("[ERROR] --csv-path is required when --data csv")
            return sys.exit(2)
        df = load_csv(args.csv_path, args.date_col)
        # Build target using the same config to preserve parity
        ds = CanonicalDataset.from_prices(df)
        y = ds.build_target(cfg)
        # Align df to target index
        df = df.loc[y.index]

    # Metadata: environment snapshot
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    env = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "models": models_to_run,
    }
    # Try to add git commit
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        env["git_commit"] = commit
    except Exception:
        pass
    # Pip freeze top 50
    try:
        pkgs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
        env["pip_freeze"] = pkgs[:50]
    except Exception:
        env["pip_freeze"] = []
    write_atomic(str(Path(cfg.artifacts_dir) / "env.json"), json.dumps(env, indent=2))

    # Prepare first split info (dry-run support): use evaluator’s splitter to preview windows
    from risk_pipeline.core.splits import generate_sliding_splits
    splits = list(generate_sliding_splits(len(df), cfg.train_size, cfg.val_size, cfg.step))
    splits_json = [
        {
            "train_start": s[0].start,
            "train_end": s[0].stop,
            "val_start": s[1].start,
            "val_end": s[1].stop,
        }
        for s in splits
    ]
    write_atomic(str(Path(cfg.artifacts_dir) / "splits.json"), json.dumps(splits_json, indent=2))

    if args.dry_run:
        # Build canonical features for first split only and print shapes
        from risk_pipeline.core.feature_engineer import FeatureEngineer
        if not splits:
            print("[ERROR] No valid splits produced with current config")
            return sys.exit(3)
        train_slc, val_slc = splits[0]
        fe = FeatureEngineer()
        (X_seq_tr, X_flat_tr, y_tr), (X_seq_va, X_flat_va, y_va) = fe.create_canonical_views(df, y, cfg, train_slc, val_slc)
        print("[DRY-RUN] First split shapes:")
        print("  X_seq_train:", X_seq_tr.shape, "X_flat_train:", X_flat_tr.shape, "y_train:", y_tr.shape)
        print("  X_seq_val:", X_seq_va.shape, "X_flat_val:", X_flat_va.shape, "y_val:", y_va.shape)
        print("[DRY-RUN] Exiting without training.")
        return sys.exit(0)

    # Build models and evaluate on canonical splits/features
    models = build_models(cfg)
    # Note: we keep sequence models serial; tabular models batching/parallelization is left to evaluator (future)
    results = evaluate_all(models, df, y, cfg)

    # Save config used
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    write_atomic(str(Path(cfg.artifacts_dir) / "config.json"), json.dumps(cfg.__dict__, indent=2))

    # Results schema & NaN guard
    required_cols = [
        "model", "mse_mean", "mse_std", "mae_mean", "mae_std", "ic_mean", "ic_std", "fit_time_s_mean", "pred_time_s_mean"
    ]
    missing = [c for c in required_cols if c not in results.columns]
    if missing:
        print(f"[ERROR] Results missing required columns: {missing}")
        return sys.exit(4)
    metric_cols = [c for c in results.columns if any(m in c for m in ["mse", "mae", "ic"]) ]
    if results[metric_cols].isna().any().any() or (~results[metric_cols].replace([float('inf'), float('-inf')], pd.NA).notna()).any().any():
        print("[ERROR] NaN or inf detected in results metrics.")
        print(results)
        return sys.exit(5)

    print("\n=== Fair Comparison Results (sorted by mse_mean) ===")
    print(results.sort_values("mse_mean").to_string(index=False))
    print(f"\nArtifacts written to: {cfg.artifacts_dir}")

    return sys.exit(0)

if __name__ == "__main__":
    main()
