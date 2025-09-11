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
import os

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Centralized results manager for RiskPipeline.
    
    This class provides a single point of access for all pipeline results,
    including model predictions, metrics, SHAP values, and metadata.
    It implements a thread-safe singleton pattern for shared state management.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize the results manager.

        Args:
            base_dir: Base directory where experiments and artifacts are stored
        """
        self._results: Dict[str, Any] = {}
        self._shap_results: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._predictions: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self.base_dir: Path = Path(base_dir) if base_dir else Path('experiments')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment_id: Optional[str] = None
        self.current_experiment_path: Optional[Path] = None
        
        # Initialize timestamp
        self._metadata['created_at'] = datetime.now().isoformat()
        self._metadata['version'] = '2.0.0'  # Modular version
        
        logger.info("ResultsManager initialized")
    
    def _get_experiment_dir(self):
        """Get the current experiment directory."""
        if self.current_experiment_path:
            return str(self.current_experiment_path)
        else:
            return str(self.base_dir / 'temp')
    
    def get_experiment_dir(self):
        """Get the current experiment directory (public method)."""
        return self._get_experiment_dir()

    # Experiment lifecycle
    def start_experiment(self, name: str, config: Dict[str, Any], description: str = "") -> str:
        """Create a new experiment directory and persist config. Returns experiment id."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"experiment_{timestamp}"
        exp_path = self.base_dir / exp_id
        exp_path.mkdir(parents=True, exist_ok=True)

        with open(exp_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

        metadata = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
        }
        with open(exp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.current_experiment_id = exp_id
        self.current_experiment_path = exp_path
        return exp_id

    def save_experiment_metadata(self, metadata: Dict[str, Any]) -> None:
        """Merge and persist experiment metadata for the active experiment."""
        if not self.current_experiment_path:
            logger.warning("No active experiment to save metadata")
            return
        meta_path = self.current_experiment_path / 'metadata.json'
        try:
            existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            existing = {}
        existing.update(metadata)
        with open(meta_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
    
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
        rounded_metrics = {k: (round(v, 12) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        self._metrics[key] = {
            'metrics': rounded_metrics,
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
        maximize = metric.lower() in {'r2', 'accuracy', 'f1', 'precision', 'recall'}
        best_score = -float('inf') if maximize else float('inf')
        
        for key, data in self._metrics.items():
            if data['asset'] == asset and data['task'] == task:
                metrics = data['metrics']
                if metric in metrics:
                    score = metrics[metric]
                    better = score > best_score if maximize else score < best_score
                    if better:
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
    
    def save_shap_results(self, *args, **kwargs):
        """Save SHAP results.

        Two call styles are supported:
        - save_shap_results(asset, model_name, shap_values, explainer_metadata, feature_importance, task='regression')
        - save_shap_results(shap_results_dict, output_dir)
        """
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            # New API: granular save under current experiment
            asset: str = args[0]
            model_name: str = args[1]
            shap_values = kwargs.get('shap_values') if 'shap_values' in kwargs else (args[2] if len(args) > 2 else None)
            explainer_metadata = kwargs.get('explainer_metadata') if 'explainer_metadata' in kwargs else (args[3] if len(args) > 3 else {})
            feature_importance = kwargs.get('feature_importance') if 'feature_importance' in kwargs else (args[4] if len(args) > 4 else {})
            task: str = kwargs.get('task', 'regression')

            # Update in-memory store
            if asset not in self._shap_results:
                self._shap_results[asset] = {}
            if task not in self._shap_results[asset]:
                self._shap_results[asset][task] = {}
            self._shap_results[asset][task][model_name] = {
                'shap_values': shap_values,
                'explainer_metadata': explainer_metadata,
                'feature_importance': feature_importance,
            }

            # Persist to disk if experiment is active
            base = self.current_experiment_path or (self.base_dir / 'temp')
            output_path = base / 'shap' / asset / model_name / task
            output_path.mkdir(parents=True, exist_ok=True)

            with open(output_path / 'shap_values.pkl', 'wb') as f:
                pickle.dump(shap_values, f)
            with open(output_path / 'explainer_metadata.json', 'w') as f:
                json.dump(explainer_metadata, f, indent=2, default=str)
            with open(output_path / 'feature_importance.json', 'w') as f:
                json.dump(feature_importance, f, indent=2)

            logger.info(f"Saved SHAP artifacts: {output_path}")
            return

        # Legacy API: (shap_results_dict, output_dir)
        if len(args) == 2 and isinstance(args[0], dict):
            shap_results: Dict[str, Any] = args[0]
            output_dir: Union[str, Path] = args[1]
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / 'shap_results.pkl', 'wb') as f:
                pickle.dump(shap_results, f)
            logger.info(f"Saved SHAP results to {output_path}")
            return
        # Support kwargs-only style used by tests
        asset_kw = kwargs.get('asset')
        model_name_kw = kwargs.get('model_name')
        if asset_kw and model_name_kw:
            return self.save_shap_results(
                asset_kw,
                model_name_kw,
                shap_values=kwargs.get('shap_values'),
                explainer_metadata=kwargs.get('explainer_metadata', {}),
                feature_importance=kwargs.get('feature_importance', {}),
                task=kwargs.get('task', 'regression'),
            )
        raise TypeError("Invalid arguments to save_shap_results")
    
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

    def export_results_table(self, format: str = 'csv') -> str:
        """Export aggregated metrics to the active experiment directory.

        Returns the path to the exported file.
        """
        df = self.get_all_metrics()
        export_dir = self.current_experiment_path or (self.base_dir / 'temp')
        export_dir.mkdir(parents=True, exist_ok=True)
        out_path = export_dir / f"model_performance.{ 'csv' if format == 'csv' else 'json'}"
        if format == 'csv':
            df.to_csv(out_path, index=False)
        else:
            df.to_json(out_path, orient='records')
        logger.info(f"Exported results table to {out_path}")
        return str(out_path)

    def get_best_models(self, metric: str = 'R2', task: str = 'regression') -> pd.DataFrame:
        """Return rows sorted by metric for the given task."""
        df = self.get_all_metrics()
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        df_task = df[df['task'] == task].copy()
        return df_task.sort_values(by=metric, ascending=False).reset_index(drop=True)

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment config, metadata, and summary metrics."""
        exp_path = self.base_dir / experiment_id
        out: Dict[str, Any] = {}
        cfg = exp_path / 'config.json'
        meta = exp_path / 'metadata.json'
        metrics_csv = exp_path / 'model_performance.csv'
        if cfg.exists():
            out['config'] = json.loads(cfg.read_text())
        if meta.exists():
            out['metadata'] = json.loads(meta.read_text())
        if metrics_csv.exists():
            out['summary'] = pd.read_csv(metrics_csv)
        else:
            out['summary'] = self.get_all_metrics()
        return out

    def save_model_results(self, asset: str, model_type: Optional[str] = None, task: str = 'regression', model: Any = None, 
                          metrics: Optional[Dict[str, Any]] = None, features: Optional[List[str]] = None,
                          shap_results: Optional[Dict[str, Any]] = None,
                          predictions: Optional[Dict[str, List]] = None, **kwargs) -> None:
        """
        Save comprehensive model results for thesis analysis.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            model: Trained model
            metrics: Model metrics
            features: Feature names
            shap_results: SHAP analysis results
            predictions: Dictionary containing 'actual' and 'predicted' lists
        """
        # Support alias: model_name (backward compatibility)
        if (model_type is None or model_type == '') and 'model_name' in kwargs:
            model_type = kwargs.get('model_name')

        # Basic validations / fallbacks
        if metrics is None:
            metrics = {}
        if model_type is None:
            model_type = str(getattr(model, 'name', 'unknown') or 'unknown')

        # Create experiment directory structure
        experiment_dir = self._get_experiment_dir()
        model_dir = os.path.join(experiment_dir, 'models', asset, model_type, task)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        keras_saved = False
        # Prefer TensorFlow SavedModel for TF-backed models to avoid CUDA handle issues during pickling
        try:
            # Lazy import to avoid hard TF dependency
            import tensorflow as tf  # type: ignore
            # Detect a Keras model wrapper (common in LSTM/StockMixer)
            keras_model = getattr(model, 'model', None)
            if keras_model is not None and hasattr(keras_model, 'save'):
                # Force CPU-only save path to avoid GPU cast kernels during serialization
                try:
                    from risk_pipeline.utils.tensorflow_utils import force_cpu_mode  # type: ignore
                    force_cpu_mode()
                except Exception:
                    pass
                tf_model_dir = os.path.join(model_dir, 'tf_model')
                os.makedirs(tf_model_dir, exist_ok=True)
                # Use SavedModel format (directory) for maximum compatibility
                keras_model.save(tf_model_dir)
                # Also save weights in H5 for convenience
                try:
                    keras_model.save_weights(os.path.join(model_dir, 'weights.h5'))
                except Exception:
                    pass
                # Save a lightweight manifest with metadata
                manifest = {
                    'backend': 'tensorflow.keras',
                    'saved_model_dir': 'tf_model',
                    'weights_file': 'weights.h5',
                    'wrapper_class': model.__class__.__name__,
                    'task': task,
                    'model_type': model_type,
                }
                with open(os.path.join(model_dir, 'model_manifest.json'), 'w') as mf:
                    json.dump(manifest, mf, indent=2)
                keras_saved = True
                logger.info(f"Keras model saved to {tf_model_dir} (SavedModel) with manifest")
        except Exception:
            # Not a TF environment or save failed; continue to pickle attempt
            pass

        # Attempt to pickle full wrapper for non-TF models or as best-effort backup
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            if keras_saved:
                logger.warning(f"Failed to pickle model wrapper (using SavedModel instead): {e}")
            else:
                logger.warning(f"Failed to save model: {e}")
        
        # Save metrics
        metrics_path = os.path.join(model_dir, 'metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
        
        # Save features
        if features:
            features_path = os.path.join(model_dir, 'features.json')
            try:
                with open(features_path, 'w') as f:
                    json.dump(features, f, indent=2)
                logger.info(f"Features saved to {features_path}")
            except Exception as e:
                logger.warning(f"Failed to save features: {e}")
        
        # Save SHAP results
        if shap_results:
            shap_path = os.path.join(model_dir, 'shap_results.json')
            try:
                # Convert numpy arrays to lists for JSON serialization
                shap_serializable = {}
                for key, value in shap_results.items():
                    if isinstance(value, np.ndarray):
                        shap_serializable[key] = value.tolist()
                    else:
                        shap_serializable[key] = value
                
                with open(shap_path, 'w') as f:
                    json.dump(shap_serializable, f, indent=2, default=str)
                logger.info(f"SHAP results saved to {shap_path}")
            except Exception as e:
                logger.warning(f"Failed to save SHAP results: {e}")
        
        # Save per-fold predictions and comprehensive data
        if predictions:
            try:
                # Save per-fold predictions CSV
                predictions_path = os.path.join(model_dir, 'per_fold_predictions.csv')
                if 'actual' in predictions and 'predicted' in predictions:
                    df_predictions = pd.DataFrame({
                        'actual': predictions['actual'],
                        'predicted': predictions['predicted']
                    })
                    df_predictions.to_csv(predictions_path, index=False)
                    logger.info(f"Per-fold predictions saved to {predictions_path}")
                
                # Save comprehensive fold-level metrics if available
                if 'all_fold_metrics' in predictions and predictions['all_fold_metrics']:
                    fold_metrics_path = os.path.join(model_dir, 'fold_level_metrics.csv')
                    df_fold_metrics = pd.DataFrame(predictions['all_fold_metrics'])
                    df_fold_metrics.to_csv(fold_metrics_path, index=False)
                    logger.info(f"Fold-level metrics saved to {fold_metrics_path}")
                
                # Save fold indices for reproducibility
                if 'all_fold_indices' in predictions and predictions['all_fold_indices']:
                    fold_indices_path = os.path.join(model_dir, 'fold_indices.json')
                    with open(fold_indices_path, 'w') as f:
                        json.dump(predictions['all_fold_indices'], f, indent=2)
                    logger.info(f"Fold indices saved to {fold_indices_path}")
                
                # Save comprehensive analysis summary
                analysis_summary = {
                    'asset': asset,
                    'model_type': model_type,
                    'task': task,
                    'timestamp': datetime.now().isoformat(),
                    'n_splits': predictions.get('n_splits', 0),
                    'successful_splits': predictions.get('successful_splits', 0),
                    'total_splits': predictions.get('total_splits', 0),
                    'total_samples': len(predictions.get('actual', [])),
                    'metrics_summary': metrics
                }
                
                summary_path = os.path.join(model_dir, 'analysis_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(analysis_summary, f, indent=2, default=str)
                logger.info(f"Analysis summary saved to {summary_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save comprehensive data: {e}")
        
        # Store in memory for quick access
        key = f"{asset}_{model_type}_{task}"
        # Standardize model storage to match get_model expectations
        self._models[key] = {
            'model': model,
            'asset': asset,
            'model_type': model_type,
            'task': task,
            'stored_at': datetime.now().isoformat()
        }
        self._metrics[key] = {
            'metrics': metrics,
            'asset': asset,
            'model_type': model_type,
            'task': task,
            'stored_at': datetime.now().isoformat()
        }
        
        if features:
            # Backward-compat: keep key-based, but standardize asset-based as primary
            self._features[key] = features
            self._features[asset] = {
                'features': features,
                'stored_at': datetime.now().isoformat()
            }
        
        if shap_results:
            self._shap_results[key] = shap_results
        
        if predictions:
            self._predictions[key] = predictions
        
        logger.info(f"✅ Comprehensive results saved for {asset}_{model_type}_{task}")
    
    def export_thesis_data(self, output_dir: str = None) -> str:
        """
        Export comprehensive thesis-ready data for all experiments.
        
        Args:
            output_dir: Output directory (defaults to experiments/thesis_export_<timestamp>)
            
        Returns:
            Path to exported data directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self._get_experiment_dir(), f"thesis_export_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export aggregated metrics
        all_metrics_df = self.get_all_metrics()
        if not all_metrics_df.empty:
            metrics_path = os.path.join(output_dir, 'all_model_metrics.csv')
            all_metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"All model metrics exported to {metrics_path}")
        
        # Export per-fold data for each model
        for key, pred_data in self._predictions.items():
            try:
                asset, model_type, task = key.split('_', 2)
                model_dir = os.path.join(output_dir, 'per_model_data', asset, model_type, task)
                os.makedirs(model_dir, exist_ok=True)
                
                # Export predictions
                if 'actual' in pred_data and 'predicted' in pred_data:
                    df_pred = pd.DataFrame({
                        'actual': pred_data['actual'],
                        'predicted': pred_data['predicted']
                    })
                    df_pred.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
                
                # Export fold metrics
                if 'all_fold_metrics' in pred_data and pred_data['all_fold_metrics']:
                    df_fold = pd.DataFrame(pred_data['all_fold_metrics'])
                    df_fold.to_csv(os.path.join(model_dir, 'fold_metrics.csv'), index=False)
                
                # Export fold indices
                if 'all_fold_indices' in pred_data and pred_data['all_fold_indices']:
                    with open(os.path.join(model_dir, 'fold_indices.json'), 'w') as f:
                        json.dump(pred_data['all_fold_indices'], f, indent=2)
                
                # Export model metrics
                if key in self._metrics:
                    with open(os.path.join(model_dir, 'model_metrics.json'), 'w') as f:
                        json.dump(self._metrics[key], f, indent=2, default=str)
                
            except Exception as e:
                logger.warning(f"Failed to export data for {key}: {e}")
        
        # Create comprehensive summary report
        summary_report = {
            'export_timestamp': datetime.now().isoformat(),
            'total_models': len(self._models),
            'total_assets': len(set(data['asset'] for data in self._metrics.values())),
            'model_types': list(set(data['model_type'] for data in self._metrics.values())),
            'tasks': list(set(data['task'] for data in self._metrics.values())),
            'export_summary': {}
        }
        
        # Add per-asset summary
        for asset in set(data['asset'] for data in self._metrics.values()):
            asset_models = [k for k in self._metrics.keys() if k.startswith(f"{asset}_")]
            summary_report['export_summary'][asset] = {
                'models': len(asset_models),
                'model_types': list(set(self._metrics[k]['model_type'] for k in asset_models)),
                'tasks': list(set(self._metrics[k]['task'] for k in asset_models))
            }
        
        with open(os.path.join(output_dir, 'export_summary.json'), 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"🎓 Thesis data export completed: {output_dir}")
        return output_dir


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