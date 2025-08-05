"""
Results Manager for RiskPipeline - Centralized state management.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Centralized results manager for RiskPipeline.
    
    This class provides a single point of access for all pipeline results,
    including model predictions, metrics, SHAP values, and metadata.
    It implements a thread-safe singleton pattern for shared state management.
    """
    
    def __init__(self):
        """Initialize the results manager."""
        self._results: Dict[str, Any] = {}
        self._shap_results: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._predictions: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        
        # Initialize timestamp
        self._metadata['created_at'] = datetime.now().isoformat()
        self._metadata['version'] = '2.0.0'  # Modular version
        
        logger.info("ResultsManager initialized")
    
    def store_results(self, results: Dict[str, Any], asset: Optional[str] = None):
        """
        Store pipeline results.
        
        Args:
            results: Results dictionary
            asset: Optional asset name for asset-specific results
        """
        if asset:
            self._results[asset] = results
        else:
            self._results.update(results)
        
        logger.debug(f"Stored results for asset: {asset}")
    
    def store_shap_results(self, shap_results: Dict[str, Any], asset: Optional[str] = None):
        """
        Store SHAP analysis results.
        
        Args:
            shap_results: SHAP results dictionary
            asset: Optional asset name for asset-specific results
        """
        if asset:
            self._shap_results[asset] = shap_results
        else:
            self._shap_results.update(shap_results)
        
        logger.debug(f"Stored SHAP results for asset: {asset}")
    
    def store_model(self, model: Any, asset: str, model_type: str, task: str):
        """
        Store a trained model.
        
        Args:
            model: Trained model object
            asset: Asset name
            model_type: Type of model
            task: Task type ('regression' or 'classification')
        """
        key = f"{asset}_{model_type}_{task}"
        self._models[key] = {
            'model': model,
            'asset': asset,
            'model_type': model_type,
            'task': task,
            'stored_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Stored model: {key}")
    
    def store_features(self, features: Dict[str, Any], asset: str):
        """
        Store engineered features.
        
        Args:
            features: Feature dictionary
            asset: Asset name
        """
        self._features[asset] = {
            'features': features,
            'stored_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Stored features for asset: {asset}")
    
    def store_predictions(self, predictions: np.ndarray, asset: str, model_type: str, task: str):
        """
        Store model predictions.
        
        Args:
            predictions: Model predictions
            asset: Asset name
            model_type: Type of model
            task: Task type
        """
        key = f"{asset}_{model_type}_{task}"
        self._predictions[key] = {
            'predictions': predictions,
            'asset': asset,
            'model_type': model_type,
            'task': task,
            'stored_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Stored predictions: {key}")
    
    def store_metrics(self, metrics: Dict[str, float], asset: str, model_type: str, task: str):
        """
        Store model metrics.
        
        Args:
            metrics: Metrics dictionary
            asset: Asset name
            model_type: Type of model
            task: Task type
        """
        key = f"{asset}_{model_type}_{task}"
        self._metrics[key] = {
            'metrics': metrics,
            'asset': asset,
            'model_type': model_type,
            'task': task,
            'stored_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Stored metrics: {key}")
    
    def get_results(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stored results.
        
        Args:
            asset: Optional asset name to filter results
            
        Returns:
            Results dictionary
        """
        if asset:
            return self._results.get(asset, {})
        return self._results
    
    def get_shap_results(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stored SHAP results.
        
        Args:
            asset: Optional asset name to filter results
            
        Returns:
            SHAP results dictionary
        """
        if asset:
            return self._shap_results.get(asset, {})
        return self._shap_results
    
    def get_model(self, asset: str, model_type: str, task: str) -> Optional[Any]:
        """
        Get stored model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            Model object or None if not found
        """
        key = f"{asset}_{model_type}_{task}"
        model_data = self._models.get(key)
        return model_data['model'] if model_data else None
    
    def get_features(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Get stored features.
        
        Args:
            asset: Asset name
            
        Returns:
            Features dictionary or None if not found
        """
        features_data = self._features.get(asset)
        return features_data['features'] if features_data else None
    
    def get_predictions(self, asset: str, model_type: str, task: str) -> Optional[np.ndarray]:
        """
        Get stored predictions.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            Predictions array or None if not found
        """
        key = f"{asset}_{model_type}_{task}"
        pred_data = self._predictions.get(key)
        return pred_data['predictions'] if pred_data else None
    
    def get_metrics(self, asset: str, model_type: str, task: str) -> Optional[Dict[str, float]]:
        """
        Get stored metrics.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            Metrics dictionary or None if not found
        """
        key = f"{asset}_{model_type}_{task}"
        metrics_data = self._metrics.get(key)
        return metrics_data['metrics'] if metrics_data else None
    
    def get_all_metrics(self) -> pd.DataFrame:
        """
        Get all metrics as a DataFrame for analysis.
        
        Returns:
            DataFrame with all metrics
        """
        metrics_list = []
        
        for key, data in self._metrics.items():
            metrics = data['metrics'].copy()
            metrics['asset'] = data['asset']
            metrics['model_type'] = data['model_type']
            metrics['task'] = data['task']
            metrics['stored_at'] = data['stored_at']
            metrics_list.append(metrics)
        
        if metrics_list:
            return pd.DataFrame(metrics_list)
        else:
            return pd.DataFrame()
    
    def get_best_model(self, asset: str, task: str, metric: str = 'mse') -> Optional[Dict[str, Any]]:
        """
        Get the best performing model for an asset and task.
        
        Args:
            asset: Asset name
            task: Task type
            metric: Metric to optimize (lower is better)
            
        Returns:
            Best model information or None if no models found
        """
        best_model = None
        best_score = float('inf')
        
        for key, data in self._metrics.items():
            if data['asset'] == asset and data['task'] == task:
                metrics = data['metrics']
                if metric in metrics:
                    score = metrics[metric]
                    if score < best_score:
                        best_score = score
                        best_model = {
                            'model_type': data['model_type'],
                            'metrics': metrics,
                            'model': self.get_model(asset, data['model_type'], task)
                        }
        
        return best_model
    
    def list_assets(self) -> List[str]:
        """Get list of all assets with results."""
        return list(self._results.keys())
    
    def list_models(self) -> List[str]:
        """Get list of all stored models."""
        return list(self._models.keys())
    
    def clear_results(self, asset: Optional[str] = None):
        """
        Clear stored results.
        
        Args:
            asset: Optional asset name to clear specific results
        """
        if asset:
            self._results.pop(asset, None)
            self._shap_results.pop(asset, None)
            self._features.pop(asset, None)
            
            # Clear related predictions and metrics
            keys_to_remove = [k for k in self._predictions.keys() if k.startswith(f"{asset}_")]
            for key in keys_to_remove:
                self._predictions.pop(key, None)
            
            keys_to_remove = [k for k in self._metrics.keys() if k.startswith(f"{asset}_")]
            for key in keys_to_remove:
                self._metrics.pop(key, None)
            
            keys_to_remove = [k for k in self._models.keys() if k.startswith(f"{asset}_")]
            for key in keys_to_remove:
                self._models.pop(key, None)
        else:
            self._results.clear()
            self._shap_results.clear()
            self._features.clear()
            self._predictions.clear()
            self._metrics.clear()
            self._models.clear()
        
        logger.info(f"Cleared results for asset: {asset}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """
        Save results to disk.
        
        Args:
            results: Results to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics as CSV
        metrics_df = self.get_all_metrics()
        if not metrics_df.empty:
            metrics_file = output_path / 'model_metrics.csv'
            metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Saved results to {output_path}")
    
    def save_shap_results(self, shap_results: Dict[str, Any], output_dir: str):
        """
        Save SHAP results to disk.
        
        Args:
            shap_results: SHAP results to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save SHAP results
        shap_file = output_path / 'shap_results.pkl'
        with open(shap_file, 'wb') as f:
            pickle.dump(shap_results, f)
        
        logger.info(f"Saved SHAP results to {output_path}")
    
    def load_results(self, results_file: str):
        """
        Load results from disk.
        
        Args:
            results_file: Path to results file
        """
        results_path = Path(results_file)
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            self.store_results(results)
            logger.info(f"Loaded results from {results_file}")
        else:
            logger.warning(f"Results file not found: {results_file}")
    
    def load_shap_results(self, shap_file: str):
        """
        Load SHAP results from disk.
        
        Args:
            shap_file: Path to SHAP results file
        """
        shap_path = Path(shap_file)
        
        if shap_path.exists():
            with open(shap_path, 'rb') as f:
                shap_results = pickle.load(f)
            self.store_shap_results(shap_results)
            logger.info(f"Loaded SHAP results from {shap_file}")
        else:
            logger.warning(f"SHAP file not found: {shap_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all stored data.
        
        Returns:
            Summary dictionary
        """
        return {
            'assets': len(self._results),
            'models': len(self._models),
            'features': len(self._features),
            'predictions': len(self._predictions),
            'metrics': len(self._metrics),
            'shap_results': len(self._shap_results),
            'created_at': self._metadata['created_at'],
            'version': self._metadata['version']
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export all results to a DataFrame for analysis.
        
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        for asset, asset_results in self._results.items():
            for task_type, task_results in asset_results.items():
                for model_type, model_results in task_results.items():
                    if isinstance(model_results, dict) and 'metrics' in model_results:
                        row = {
                            'asset': asset,
                            'task': task_type,
                            'model_type': model_type,
                            **model_results['metrics']
                        }
                        all_results.append(row)
        
        return pd.DataFrame(all_results)


# Global results manager instance for dependency injection
_global_results_manager: Optional[ResultsManager] = None


def get_results_manager() -> ResultsManager:
    """Get global results manager instance."""
    global _global_results_manager
    if _global_results_manager is None:
        _global_results_manager = ResultsManager()
    return _global_results_manager


def set_results_manager(results_manager: ResultsManager):
    """Set global results manager instance."""
    global _global_results_manager
    _global_results_manager = results_manager 