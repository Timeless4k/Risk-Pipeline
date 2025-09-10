#!/usr/bin/env python3
"""
Run cross-asset transfer matrix experiment and save reports/plots.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from risk_pipeline import RiskPipeline


def main():
    start = time.time()
    experiment_name = f"cross_transfer_{int(start)}"
    config_path = str(project_root / 'configs' / 'pipeline_config.json')
    pipe = RiskPipeline(config_path=config_path, experiment_name=experiment_name)

    assets = pipe.config.data.all_assets
    models = ['xgboost', 'lstm']

    print(f"Assets: {assets}")
    print(f"Models: {models}")

    results = pipe.run_cross_asset_matrix(assets=assets, models=models, task='regression')
    print("Cross-asset transfer completed.")
    exp_dir = Path(pipe.results_manager.base_dir) / pipe.experiment_name / 'transfer_matrices'
    print(f"Reports saved under: {exp_dir}")


if __name__ == '__main__':
    main()


