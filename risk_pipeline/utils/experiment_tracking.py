"""
Experiment tracking utilities for RiskPipeline.

Provides experiment versioning, indexing, and metadata management for reproducibility and comparison.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

class ExperimentTracker:
    """
    ExperimentTracker manages experiment versioning, indexing, and metadata.
    """
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "experiment_index.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2, default=str)

    def create_experiment(self, config: Dict[str, Any], description: str = "") -> str:
        """
        Create a new experiment with versioning and metadata.
        Returns the experiment_id.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"experiment_{timestamp}_{uuid.uuid4().hex[:6]}"
        exp_path = self.base_dir / experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "config": config,
            "path": str(exp_path)
        }
        self.index[experiment_id] = meta
        self._save_index()
        # Save config and metadata
        with open(exp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        with open(exp_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return experiment_id

    def get_experiment_meta(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self.index.get(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        return list(self.index.values())

    def find_experiments(self, filter_fn) -> List[Dict[str, Any]]:
        return [meta for meta in self.index.values() if filter_fn(meta)]

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        if experiment_id in self.index:
            self.index[experiment_id].update(updates)
            self._save_index()

    def archive_experiment(self, experiment_id: str):
        if experiment_id in self.index:
            self.index[experiment_id]["archived"] = True
            self._save_index()

    def delete_experiment(self, experiment_id: str):
        if experiment_id in self.index:
            exp_path = Path(self.index[experiment_id]["path"])
            if exp_path.exists():
                for child in exp_path.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(exp_path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                exp_path.rmdir()
            del self.index[experiment_id]
            self._save_index()