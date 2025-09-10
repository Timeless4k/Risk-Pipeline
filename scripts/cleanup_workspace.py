#!/usr/bin/env python3
"""
Workspace cleanup utility with a safe dry-run by default.

Removes old experiment folders, logs, temporary caches, and test artifacts, while
keeping recent items according to retention arguments.

Defaults (can be overridden):
- Keep last 5 experiment runs by modification time
- Keep last 10 pipeline logs
- Delete temp cache under data_cache/joblib_tmp
- Delete files prefixed with '^' in data_cache (failed/incomplete files)
- Keep SHAP plots from the last 14 days; older are pruned
- Remove test artifact directories (test_shap_dir, test_shap_plots) if present

Usage examples:
  Dry run (default):
    python scripts/cleanup_workspace.py

  Execute deletion:
    python scripts/cleanup_workspace.py --execute

  Custom retention:
    python scripts/cleanup_workspace.py --keep-experiments 8 --keep-logs 20 --shap-retention-days 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DeletionCandidate:
    path: Path
    is_dir: bool
    bytes_size: int
    reason: str


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def get_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except (PermissionError, FileNotFoundError):
            continue
    return total


def list_dirs_sorted_by_mtime(path: Path) -> List[Path]:
    if not path.exists():
        return []
    dirs = [p for p in path.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def list_files_sorted_by_mtime(path: Path, pattern: str = "*.log") -> List[Path]:
    if not path.exists():
        return []
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def collect_experiment_deletions(experiments_dir: Path, keep: int) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    dirs = list_dirs_sorted_by_mtime(experiments_dir)
    to_delete = dirs[keep:]
    for d in to_delete:
        candidates.append(
            DeletionCandidate(path=d, is_dir=True, bytes_size=get_size(d), reason=f"experiments: keep_last={keep}")
        )
    return candidates


def collect_log_deletions(logs_dir: Path, keep: int) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    files = list_files_sorted_by_mtime(logs_dir, pattern="*.log")
    to_delete = files[keep:]
    for f in to_delete:
        candidates.append(
            DeletionCandidate(path=f, is_dir=False, bytes_size=get_size(f), reason=f"logs: keep_last={keep}")
        )
    return candidates


def collect_data_cache_deletions(data_cache_dir: Path) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    # Delete joblib tmp dir entirely
    joblib_tmp = data_cache_dir / "joblib_tmp"
    if joblib_tmp.exists():
        candidates.append(
            DeletionCandidate(path=joblib_tmp, is_dir=True, bytes_size=get_size(joblib_tmp), reason="cache: joblib_tmp")
        )
    # Delete files starting with '^' (likely control-characters or temp/failed downloads)
    for f in data_cache_dir.iterdir() if data_cache_dir.exists() else []:
        if f.is_file() and f.name.startswith("^"):
            candidates.append(
                DeletionCandidate(path=f, is_dir=False, bytes_size=get_size(f), reason="cache: invalid_prefix")
            )
    return candidates


def collect_shap_plot_deletions(shap_dir: Path, retention_days: int) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    if not shap_dir.exists():
        return candidates
    cutoff = dt.datetime.now() - dt.timedelta(days=retention_days)
    # Keep 'quick' entirely; prune older files elsewhere by mtime
    for item in shap_dir.iterdir():
        if item.name.lower() == "quick":
            continue
        if item.is_dir():
            # Delete files inside dir older than cutoff; delete dir if becomes empty
            for f in item.rglob("*"):
                try:
                    if f.is_file():
                        mtime = dt.datetime.fromtimestamp(f.stat().st_mtime)
                        if mtime < cutoff:
                            candidates.append(
                                DeletionCandidate(
                                    path=f, is_dir=False, bytes_size=get_size(f), reason=f"shap: >{retention_days}d"
                                )
                            )
                except (PermissionError, FileNotFoundError):
                    continue
        elif item.is_file():
            try:
                mtime = dt.datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    candidates.append(
                        DeletionCandidate(
                            path=item, is_dir=False, bytes_size=get_size(item), reason=f"shap: >{retention_days}d"
                        )
                    )
            except (PermissionError, FileNotFoundError):
                continue
    return candidates


def collect_test_artifact_deletions(root: Path) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    for name in ["test_shap_dir", "test_shap_plots"]:
        p = root / name
        if p.exists():
            candidates.append(
                DeletionCandidate(path=p, is_dir=True, bytes_size=get_size(p), reason="tests: artifacts")
            )
    return candidates


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            # Python < 3.8 fallback
            if path.exists():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def summarize(candidates: Iterable[DeletionCandidate]) -> Tuple[int, int]:
    total_bytes = 0
    total_count = 0
    for c in candidates:
        total_bytes += c.bytes_size
        total_count += 1
    return total_count, total_bytes


def _collect_root_glob(root: Path, patterns: List[str], reason: str) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []
    for pattern in patterns:
        for p in root.glob(pattern):
            if not p.exists():
                continue
            # Skip directories that we intentionally do not remove here
            candidates.append(
                DeletionCandidate(path=p, is_dir=p.is_dir(), bytes_size=get_size(p), reason=reason)
            )
    return candidates


def collect_production_prune(root: Path) -> List[DeletionCandidate]:
    """Aggressively remove non-runtime artifacts for a slim production workspace.

    Conservatively preserves core package, configs, data_cache CSVs, and scripts/cleanup.
    """
    candidates: List[DeletionCandidate] = []

    # Entire directories safe to remove for production runs (except experiments: keep latest)
    for dir_name in [
        "thesis_reports",
        "visualizations",
    ]:
        p = root / dir_name
        if p.exists():
            candidates.append(
                DeletionCandidate(path=p, is_dir=True, bytes_size=get_size(p), reason="production: remove_dir")
            )

    # Experiments: keep only the most recent experiment directory
    candidates += collect_experiment_deletions(root / "experiments", keep=1)

    # Logs: keep only last 2
    candidates += collect_log_deletions(root / "logs", keep=2)

    # SHAP plots: remove entirely (reproducible) except keep 'quick' optional; here remove all
    shap_root = root / "shap_plots"
    if shap_root.exists():
        candidates.append(
            DeletionCandidate(path=shap_root, is_dir=True, bytes_size=get_size(shap_root), reason="production: shap_plots")
        )

    # Test artifacts and test directories/files
    for name in ["test_shap_dir", "test_shap_plots", "tests"]:
        p = root / name
        if p.exists():
            candidates.append(
                DeletionCandidate(path=p, is_dir=True, bytes_size=get_size(p), reason="production: tests")
            )

    candidates += _collect_root_glob(
        root,
        patterns=[
            "test_*.py",
            "*_test.py",
            "*tests*.py",
            "*_test_*.py",
        ],
        reason="production: test_files",
    )

    # Development/helper scripts that are not needed in production runs
    for rel in [
        "scripts/viz_fix_test.py",
        "debug_pipeline.py",
        "fix_negative_r2_solution.py",
        "quick_hyperparameter_test.py",
        "simple_hyperparameter_test.py",
        "run_simple_pipeline.py",
        "ultimate_pipeline_test.log",
    ]:
        p = root / rel
        if p.exists():
            candidates.append(
                DeletionCandidate(path=p, is_dir=p.is_dir(), bytes_size=get_size(p), reason="production: dev_artifact")
            )

    # Result/benchmark JSON and CSV artifacts in root
    candidates += _collect_root_glob(
        root,
        patterns=[
            "quick_hyperparameter_test_results_*.json",
            "simple_hyperparameter_results_*.json",
            "r2_fixing_results_*.json",
        ],
        reason="production: result_artifact",
    )

    # Data cache: keep CSVs, remove joblib tmp and malformed '^' files (handled already)
    candidates += collect_data_cache_deletions(root / "data_cache")

    return candidates


def plan_cleanup(args: argparse.Namespace) -> List[DeletionCandidate]:
    candidates: List[DeletionCandidate] = []

    if getattr(args, "production", False):
        candidates += collect_production_prune(REPO_ROOT)
    else:
        # Experiments
        candidates += collect_experiment_deletions(REPO_ROOT / "experiments", keep=args.keep_experiments)

        # Logs
        candidates += collect_log_deletions(REPO_ROOT / "logs", keep=args.keep_logs)

        # Data cache
        candidates += collect_data_cache_deletions(REPO_ROOT / "data_cache")

        # SHAP plots retention
        candidates += collect_shap_plot_deletions(REPO_ROOT / "shap_plots", retention_days=args.shap_retention_days)

        # Test artifacts
        candidates += collect_test_artifact_deletions(REPO_ROOT)

    # De-duplicate by path (in case of overlap)
    unique: dict[Path, DeletionCandidate] = {c.path: c for c in candidates}
    return list(unique.values())


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe workspace cleanup with dry-run")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry-run)")
    parser.add_argument("--production", action="store_true", help="Aggressively remove non-runtime artifacts for production")
    parser.add_argument("--keep-experiments", type=int, default=5, help="How many experiment runs to keep")
    parser.add_argument("--keep-logs", type=int, default=10, help="How many recent logs to keep")
    parser.add_argument(
        "--shap-retention-days", type=int, default=14, help="Retention window (days) for SHAP plots"
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    candidates = plan_cleanup(args)
    total_count, total_bytes = summarize(candidates)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Cleanup mode: {mode}")
    print(f"Repository root: {REPO_ROOT}")
    print("Planned deletions (reason | size | path):")
    for c in sorted(candidates, key=lambda x: (x.reason, x.path)):
        print(f"- {c.reason} | {human_bytes(c.bytes_size)} | {c.path}")
    print("")
    print(f"Total items: {total_count}")
    print(f"Total size: {human_bytes(total_bytes)}")

    if not args.execute:
        return 0

    # Execute deletions
    errors: List[Tuple[Path, str]] = []
    for c in candidates:
        try:
            delete_path(c.path)
        except Exception as e:  # noqa: BLE001
            errors.append((c.path, str(e)))

    if errors:
        print("\nSome deletions failed:", file=sys.stderr)
        for p, msg in errors:
            print(f"- {p}: {msg}", file=sys.stderr)
        return 1

    print("\nDeletion completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


