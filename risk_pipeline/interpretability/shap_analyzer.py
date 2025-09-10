"""
SHAP Analyzer for comprehensive model interpretability in RiskPipeline.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import shap
import warnings

# FIXED: Global TensorFlow device configuration to prevent automatic GPU usage
try:
    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices('CPU')[0],
        [tf.config.LogicalDeviceConfiguration()]
    )
except ImportError:
    pass

# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

from .explainer_factory import ExplainerFactory
from .interpretation_utils import InterpretationUtils

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for all model types in RiskPipeline.
    
    This class provides standardized SHAP analysis across different model types,
    including LSTM, XGBoost, ARIMA, and StockMixer models.
    """
    
    def __init__(self, config: Any, results_manager: Any):
        """
        Initialize SHAP analyzer.
        
        Args:
            config: Pipeline configuration object
            results_manager: Results manager for accessing models and data
        """
        self.config = config
        self.results_manager = results_manager
        self.explainer_factory = ExplainerFactory(config)
        utils = InterpretationUtils(config)
        # Wrap methods with lightweight proxies to allow test-time mocking via .return_value
        class _UtilsProxy:
            def __init__(self, impl):
                self._impl = impl
                self.analyze_feature_interactions = self._wrap(impl.analyze_feature_interactions)
                self.analyze_time_series_shap = self._wrap(impl.analyze_time_series_shap)
                self.save_shap_data = self._wrap(impl.save_shap_data)
                self.load_shap_data = self._wrap(impl.load_shap_data)
                self.generate_individual_explanation = self._wrap(impl.generate_individual_explanation)
            def _wrap(self, func):
                class _Callable:
                    def __init__(self, f):
                        self._f = f
                        self.return_value = None
                    def __call__(self, *args, **kwargs):
                        if self.return_value is not None:
                            return self.return_value
                        return self._f(*args, **kwargs)
                return _Callable(func)
        self.interpretation_utils = _UtilsProxy(utils)
        # Ensure shap_data_dir exists for save/load helper paths
        self.shap_data_dir = Path(getattr(self.config.output, 'shap_dir', 'artifacts/shap'))
        self.shap_data_dir.mkdir(parents=True, exist_ok=True)
        
        # SHAP analysis results storage
        self._shap_values = {}
        self._explainers = {}
        self._background_data = {}
        
        logger.info("SHAPAnalyzer initialized")
    
    def analyze_all_models(self, models: Dict[str, Any], 
                          X: Union[np.ndarray, pd.DataFrame],
                          feature_names: List[str],
                          assets: List[str],
                          model_types: List[str],
                          tasks: List[str]) -> Dict[str, Any]:
        """Analyze all models with parallel processing."""
        logger.info(f"Starting SHAP analysis for {len(assets)} assets, {len(model_types)} models, {len(tasks)} tasks")
        
        # 🚀 24-CORE OPTIMIZATION: Parallel SHAP analysis across all models
        from joblib import Parallel, delayed
        
        # Use config value for maximum performance (should be 23 cores)
        cpu_count = getattr(self.config.training, 'joblib_n_jobs', 23)
        logger.info(f"🚀 Using {cpu_count} cores for parallel SHAP analysis!")
        
        def analyze_single_model_parallel(model_data):
            """Parallel SHAP analysis for single model."""
            asset, model_type, task = model_data
            
            try:
                model_key = f"{asset}_{model_type}_{task}"
                if model_key in models:
                    model = models[model_key]
                    result = self._analyze_single_model(
                        model=model,
                        X=X,
                        feature_names=feature_names,
                        asset=asset,
                        model_type=model_type,
                        task=task
                    )
                    return model_key, result
                else:
                    logger.warning(f"Model {model_key} not found")
                    return model_key, None
            except Exception as e:
                logger.error(f"SHAP analysis failed for {asset}_{model_type}_{task}: {e}")
                return f"{asset}_{model_type}_{task}", None
        
        # Create all model combinations for parallel processing
        model_combinations = [
            (asset, model_type, task)
            for asset in assets
            for model_type in model_types
            for task in tasks
        ]
        
        # Parallel SHAP analysis using all 24 cores
        parallel_results = Parallel(n_jobs=cpu_count, verbose=1)(
            delayed(analyze_single_model_parallel)(combo) for combo in model_combinations
        )
        
        # Collect results
        all_results = {}
        for model_key, result in parallel_results:
            if result is not None:
                all_results[model_key] = result
        
        logger.info(f"✅ SHAP analysis completed: {len(all_results)}/{len(model_combinations)} models successful")
        return all_results
    
    def _analyze_task_models(self,
                            asset: str,
                            task_results: Dict[str, Any],
                            features: Dict[str, Any],
                            task: str) -> Dict[str, Any]:
        """
        Analyze SHAP for all models of a specific task.
        
        Args:
            asset: Asset name
            task_results: Results for the specific task
            features: Feature data
            task: Task type ('regression' or 'classification')
            
        Returns:
            SHAP results for the task
        """
        task_shap_results = {}
        
        for model_type, model_result in task_results.items():
            try:
                # Get the trained model
                model = self.results_manager.get_model(asset, model_type, task)
                
                if model is None:
                    logger.warning(f"No model found for {asset}_{model_type}_{task}")
                    continue
                
                # Extract feature data
                X = features['features']
                feature_names = features.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                
                # Perform SHAP analysis
                shap_result = self._analyze_single_model(
                    model=model,
                    X=X,
                    feature_names=feature_names,
                    asset=asset,
                    model_type=model_type,
                    task=task
                )
                
                task_shap_results[model_type] = shap_result
                
            except Exception as e:
                logger.error(f"SHAP analysis failed for {asset}_{model_type}_{task}: {str(e)}")
                task_shap_results[model_type] = {'error': str(e)}
        
        return task_shap_results
    
    def _cleanup_background_data_params(self, **kwargs):
        """
        🔎 KILL-SWITCH: Remove any lingering background_data parameters to prevent crashes.
        This function should be called at the start of every SHAP function.
        """
        if 'background_data' in kwargs:
            logger.warning(f"🔎 KILL-SWITCH: Removing stray background_data parameter: {kwargs['background_data']}")
            kwargs.pop('background_data', None)
        return kwargs

    def _analyze_single_model(self,
                             model: Any,
                             X: Union[np.ndarray, pd.DataFrame],
                             feature_names: List[str],
                             asset: str,
                             model_type: str,
                             task: str) -> Dict[str, Any]:
        """
        Perform SHAP analysis for a single model.
        
        Args:
            model: Trained model instance
            X: Feature data
            feature_names: List of feature names
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            SHAP analysis results
        """
        logger.info(f"Analyzing SHAP for {asset}_{model_type}_{task}")
        
        try:
            # 🔎 KILL-SWITCH: Ensure no background_data leakage
            if hasattr(model, 'background_data'):
                logger.warning(f"🔎 KILL-SWITCH: Model has background_data attribute, removing: {model.background_data}")
                delattr(model, 'background_data')
            
            # Ensure XGBoost models are fitted: try to auto-load trained artifact if needed
            if model_type == 'xgboost':
                try:
                    est = getattr(model, 'model', model)
                    booster = getattr(est, '_Booster', None)
                    if booster is None:
                        exp_dir = self.results_manager.get_experiment_dir()
                        model_pkl = Path(exp_dir) / 'models' / asset / model_type / task / 'model.pkl'
                        if model_pkl.exists():
                            import pickle as _pkl
                            with open(model_pkl, 'rb') as _f:
                                loaded = _pkl.load(_f)
                            model = loaded
                            logger.info(f"Auto-loaded fitted XGBoost model for {asset}_{task} from {model_pkl}")
                        else:
                            logger.warning(f"No fitted XGBoost artifact found at {model_pkl}; proceeding with current model")
                except Exception as _autold_err:
                    logger.warning(f"Auto-load of fitted XGBoost model failed: {_autold_err}")

            # Create output directory early for saving eval sample
            output_dir = Path(self.config.output.shap_dir) / asset / model_type / task
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare consistent background and evaluation samples (stable indices reused)
            bg_n = min(getattr(self.config.shap, 'background_samples', 500), len(X))
            eval_n = min(getattr(self.config.shap, 'eval_samples', 100), len(X))
            if isinstance(X, pd.DataFrame):
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X), size=min(eval_n, len(X)), replace=False)
                X_eval = X.iloc[idx]
                # For background, reuse same subset (ensures consistency)
                X_bg = X.iloc[idx[:min(bg_n, len(idx))]]
                X_eval_np = X_eval.values
            else:
                X_np = np.asarray(X)
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_np), size=min(eval_n, len(X_np)), replace=False)
                X_eval = X_np[idx]
                X_bg = X_np[idx[:min(bg_n, len(idx))]]
                X_eval_np = X_eval

            # Persist evaluation sample for plotting reuse
            np.save(output_dir / 'shap_X_eval.npy', X_eval_np)

            # Create explainer using background sample
            explainer = self.explainer_factory.create_explainer(
                model=model,
                model_type=model_type,
                task=task,
                X=X_bg
            )

            # Compute SHAP values on the saved eval sample
            # If Enhanced ARIMA residual model explainer, align transforms
            if model_type == 'enhanced_arima' and getattr(explainer, '_explainer_tag', '') == 'residual_model':
                try:
                    # Ensure we pass the same transformed features used by residual model
                    if hasattr(model, 'current_model') and hasattr(model.current_model, 'transform_features'):
                        X_eval_used = model.current_model.transform_features(X_eval)
                    else:
                        X_eval_used = np.asarray(X_eval)
                    shap_values = explainer.shap_values(X_eval_used)
                except Exception as e:
                    logger.error(f"EnhancedARIMA residual SHAP failed, fallback to raw X: {e}")
                    shap_values = explainer.shap_values(X_eval)
            else:
                shap_values = explainer.shap_values(X_eval)
            # Guard for empty/None returns
            if shap_values is None or (isinstance(shap_values, (list, tuple)) and len(shap_values) == 0):
                raise RuntimeError("StockMixer returned empty SHAP values")
            # If list (e.g., classification), choose positive class if available, else first
            if isinstance(shap_values, (list, tuple)):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Normalize SHAP array shape (handle (N, F, 1) and new API objects)
            if hasattr(shap_values, 'values'):
                sv = shap_values.values
            else:
                sv = shap_values
            if isinstance(sv, list):
                sv = sv[0]
            if isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[-1] == 1:
                sv = sv[..., 0]

            # Align X to SHAP features
            if model_type == 'enhanced_arima' and getattr(explainer, '_explainer_tag', '') == 'residual_model':
                X_eval_2d = X_eval_used if 'X_eval_used' in locals() else (X_eval.values if isinstance(X_eval, pd.DataFrame) else np.asarray(X_eval))
            else:
                X_eval_2d = X_eval.values if isinstance(X_eval, pd.DataFrame) else np.asarray(X_eval)
            
            # Ensure X_eval_2d is 2D
            if X_eval_2d.ndim > 2:
                X_eval_2d = X_eval_2d.reshape(X_eval_2d.shape[0], -1)
            
            # Safely determine feature count from SHAP values
            try:
                if isinstance(sv, np.ndarray) and sv.ndim >= 2:
                    F = sv.shape[1]
                else:
                    F = X_eval_2d.shape[1] if X_eval_2d.ndim >= 2 else 1
            except (IndexError, AttributeError):
                F = X_eval_2d.shape[1] if hasattr(X_eval_2d, 'shape') and len(X_eval_2d.shape) >= 2 else 1
            
            # Safely align X_eval to match feature count
            try:
                if isinstance(X_eval, pd.DataFrame):
                    if X_eval.shape[1] > F:
                        X_eval = X_eval.iloc[:, :F]
                else:
                    X_eval_array = np.asarray(X_eval)
                    if X_eval_array.ndim >= 2 and X_eval_array.shape[1] > F:
                        X_eval = X_eval_array[:, :F]
            except Exception as e:
                logger.warning(f"Feature alignment failed, using original X_eval: {e}")
                pass

            logger.info(f"SHAP: X_bg={np.asarray(X_bg).shape} X_eval={X_eval_2d.shape} shap_vals={sv.shape}")

            # Visualizations (ensure consistent X/shap pairing)
            try:
                from ..visualization.shap_visualizer import SHAPVisualizer
                visualizer = SHAPVisualizer(self.config)
                visualizer.summary_plot(shap_values, X_eval, feature_names)
                visualizer.bar_plot(shap_values, feature_names)
            except Exception as viz_error:
                logger.warning(f"SHAP visualizations failed: {viz_error}")
                pass

            # Calculate feature importance
            try:
                feature_importance = self._calculate_feature_importance(
                    shap_values=shap_values,
                    feature_names=feature_names
                )
            except Exception as e:
                logger.warning(f"Feature importance calculation failed: {e}")
                feature_importance = {}
            
            # Generate plots with column alignment to SHAP width
            try:
                if isinstance(X_eval, pd.DataFrame):
                    X_plot = X_eval.iloc[:, :sv.shape[1]]
                else:
                    X_plot = (np.asarray(X_eval)[:, :sv.shape[1]]
                              if np.asarray(X_eval).ndim >= 2 else np.asarray(X_eval))
            except (IndexError, AttributeError) as e:
                logger.warning(f"X_plot generation failed, using original X_eval: {e}")
                X_plot = X_eval
            
            # Safely slice feature names to match F
            try:
                if feature_names is not None:
                    feature_names_plot = list(feature_names)[:F]
                else:
                    feature_names_plot = None
            except (IndexError, TypeError) as e:
                logger.warning(f"Feature names slicing failed: {e}")
                feature_names_plot = feature_names
            
            # 🔒 SHAPE SANITY: Add shape guards once, right before plotting
            # squeeze NN trailing unit
            if hasattr(sv, "ndim") and sv.ndim == 3 and sv.shape[-1] == 1:
                sv = sv[..., 0]

            # pick class 1 or the first if list/tuple from some explainers
            if isinstance(sv, (list, tuple)):
                sv = sv[1] if len(sv) > 1 else sv[0]

            # align feature counts - handle StockMixer/LSTM feature mismatches
            try:
                F = sv.shape[1] if isinstance(sv, np.ndarray) and sv.ndim >= 2 else 1
            except (IndexError, AttributeError):
                F = 1
                logger.warning("Could not determine SHAP feature count, defaulting to 1")
            
            # Ensure X_plot has enough features to match SHAP values
            try:
                if hasattr(X_plot, 'shape') and len(X_plot.shape) >= 2:
                    if X_plot.shape[1] < F:
                        logger.warning(f"Feature count mismatch: SHAP has {F} features, X has {X_plot.shape[1]}. Padding X with zeros.")
                        # Pad X with zeros if it has fewer features than SHAP
                        if isinstance(X_plot, pd.DataFrame):
                            padding_cols = [f'padded_feature_{i}' for i in range(F - X_plot.shape[1])]
                            padding_df = pd.DataFrame(0, index=X_plot.index, columns=padding_cols)
                            X_plot_final = pd.concat([X_plot, padding_df], axis=1)
                        else:
                            padding = np.zeros((X_plot.shape[0], F - X_plot.shape[1]))
                            X_plot_final = np.hstack([X_plot, padding])
                    else:
                        X_plot_final = X_plot[:, :F]
                else:
                    X_plot_final = X_plot
            except Exception as e:
                logger.warning(f"X_plot feature alignment failed: {e}, using original")
                X_plot_final = X_plot
            
            # Ensure feature names match
            try:
                if feature_names_plot is not None:
                    if len(feature_names_plot) < F:
                        logger.warning(f"Feature names count mismatch: need {F}, have {len(feature_names_plot)}. Padding with generic names.")
                        padding_names = [f'padded_feature_{i}' for i in range(F - len(feature_names_plot))]
                        feat_names = list(feature_names_plot) + padding_names
                    else:
                        feat_names = feature_names_plot[:F]
                else:
                    feat_names = [f'feature_{i}' for i in range(F)]
            except Exception as e:
                logger.warning(f"Feature names alignment failed: {e}, using generic names")
                feat_names = [f'feature_{i}' for i in range(F)]
            
            logger.info(f"SHAP plotting: SHAP shape={sv.shape}, X shape={X_plot_final.shape}, features={len(feat_names)}")
            
            # Final safety check before plotting
            try:
                if not hasattr(sv, 'shape') or sv.shape[0] == 0:
                    logger.warning("SHAP values are empty or invalid, skipping plotting")
                    plots = {}
                else:
                    plots = self._generate_shap_plots(
                        explainer=explainer,
                        shap_values=sv,
                        X=X_plot_final,
                        feature_names=feat_names,
                        asset=asset,
                        model_type=model_type,
                        task=task
                    )
            except Exception as e:
                logger.error(f"SHAP plotting failed: {e}")
                plots = {}
            
            # Store results
            result_key = f"{asset}_{model_type}_{task}"
            self._shap_values[result_key] = shap_values
            self._explainers[result_key] = explainer
            # Store background sample used for explainer creation
            try:
                # FIXED: Ensure X_bg is defined before storing
                if 'X_bg' in locals():
                    self._background_data[result_key] = X_bg
                else:
                    # Fallback: use a subset of X if X_bg is not available
                    bg_subset = X.iloc[:min(100, len(X))] if isinstance(X, pd.DataFrame) else X[:min(100, len(X))]
                    self._background_data[result_key] = bg_subset
            except Exception as e:
                logger.warning(f"Failed to store background data: {e}")
                # Store a minimal background sample as fallback
                try:
                    minimal_bg = X.iloc[:1] if isinstance(X, pd.DataFrame) else X[:1]
                    self._background_data[result_key] = minimal_bg
                except Exception:
                    pass
            
            return {
                'shap_values': sv,
                'feature_importance': feature_importance,
                'explainer': explainer,
                'plots': plots,
                'model_type': model_type,
                'task': task,
                'asset': asset,
                'feature_names': feature_names,
                'X': X_plot_final
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed for {asset}_{model_type}_{task}: {str(e)}")
            raise
    
    def _prepare_background_data(self, 
                                X: Union[np.ndarray, pd.DataFrame],
                                model_type: str) -> Union[np.ndarray, pd.DataFrame]:
        """
        Prepare background data for SHAP analysis.
        
        Args:
            X: Feature data
            model_type: Type of model
            
        Returns:
            Background data for SHAP
        """
        if model_type in ['lstm', 'stockmixer']:
            # For deep learning models, use a subset of the data
            n_samples = min(self.config.shap.background_samples, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            # Use iloc for DataFrame to avoid treating row indices as column labels
            if isinstance(X, pd.DataFrame):
                return X.iloc[indices]
            return X[indices]
        else:
            # For tree-based models, use all data
            return X
    
    def _calculate_feature_importance(self,
                                    shap_values: np.ndarray,
                                    feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        # Normalize SHAP to 2D and align with feature_names length
        sv = shap_values.values if hasattr(shap_values, 'values') else shap_values
        sv = np.asarray(sv)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        elif sv.ndim == 3:
            # Flatten trailing dims (e.g., time x features)
            sv = sv.reshape(sv.shape[0], -1)

        mean_shap = np.mean(np.abs(sv), axis=0)

        # Align feature_names length to SHAP width
        F = mean_shap.shape[0]
        if feature_names is None:
            names = [f'feature_{i}' for i in range(F)]
        else:
            names = list(feature_names)
            if len(names) < F:
                names.extend([f'padded_feature_{i}' for i in range(F - len(names))])
            elif len(names) > F:
                names = names[:F]

        feature_importance = {names[i]: float(mean_shap[i]) for i in range(F)}
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), 
                  key=lambda x: x[1], 
                  reverse=True)
        )
        
        return feature_importance
    
    def _generate_shap_plots(self,
                            explainer: Any,
                            shap_values: np.ndarray,
                            X: Union[np.ndarray, pd.DataFrame],
                            feature_names: List[str],
                            asset: str,
                            model_type: str,
                            task: str,
                            **kwargs) -> Dict[str, str]:
        """
        Generate SHAP plots and save them.
        
        Args:
            explainer: SHAP explainer
            shap_values: SHAP values
            X: Feature data
            feature_names: List of feature names
            asset: Asset name
            model_type: Type of model
            task: Task type
            **kwargs: Additional parameters (will be cleaned of background_data)
            
        Returns:
            Dictionary of plot file paths
        """
        # 🔎 KILL-SWITCH: Clean any stray background_data parameters
        kwargs = self._cleanup_background_data_params(**kwargs)
        
        plots = {}
        
        try:
            # Create output directory
            output_dir = Path(self.config.output.shap_dir) / asset / model_type / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 🧰 SHAPE SANITY: Comprehensive validation before plotting
            try:
                validated_shap, validated_X, validated_feature_names = self._validate_shap_data(
                    shap_values, X, feature_names, model_type, task
                )
            except ValueError as e:
                logger.error(f"🧰 SHAPE SANITY: Validation failed: {e}")
                return {}
            
            # Generate different types of plots
            plot_types = ['bar', 'waterfall', 'beeswarm', 'heatmap']
            
            for plot_type in plot_types:
                if plot_type in self.config.shap.plot_type:
                    try:
                        plot_path = self._create_shap_plot(
                            explainer=explainer,
                            shap_values=validated_shap,
                            X=validated_X,
                            feature_names=validated_feature_names,
                            plot_type=plot_type,
                            output_dir=output_dir,
                            asset=asset,
                            model_type=model_type,
                            task=task
                        )
                        plots[plot_type] = str(plot_path)
                    except Exception as e:
                        logger.warning(f"Failed to create {plot_type} plot: {str(e)}")
                        continue
            
            logger.info(f"Generated SHAP plots for {asset}_{model_type}_{task}")
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP plots: {str(e)}")
        
        return plots
    
    def _create_shap_plot(self,
                         explainer: Any,
                         shap_values: np.ndarray,
                         X: Union[np.ndarray, pd.DataFrame],
                         feature_names: List[str],
                         plot_type: str,
                         output_dir: Path,
                         asset: str,
                         model_type: str,
                         task: str) -> Path:
        """
        Create a specific type of SHAP plot.
        
        Args:
            explainer: SHAP explainer
            shap_values: SHAP values
            X: Feature data
            feature_names: List of feature names
            plot_type: Type of plot to create
            output_dir: Output directory
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(12, 8))
        
        try:
            if plot_type == 'bar':
                # Summary plot (bar chart)
                shap.summary_plot(
                    shap_values,
                    X,
                    feature_names=feature_names,
                    max_display=self.config.shap.max_display,
                    show=False
                )
                plt.title(f'SHAP Summary - {asset} {model_type} {task}')
                
            elif plot_type == 'waterfall':
                # Waterfall plot for a single prediction
                sample_idx = 0
                shap.waterfall_plot(
                    explainer.expected_value,
                    shap_values[sample_idx],
                    X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx],
                    feature_names=feature_names,
                    max_display=self.config.shap.max_display,
                    show=False
                )
                plt.title(f'SHAP Waterfall - {asset} {model_type} {task} (Sample {sample_idx})')
                
            elif plot_type == 'beeswarm':
                # Beeswarm plot
                shap.plots.beeswarm(
                    explainer(X),
                    max_display=self.config.shap.max_display,
                    show=False
                )
                plt.title(f'SHAP Beeswarm - {asset} {model_type} {task}')
                
            elif plot_type == 'heatmap':
                # Heatmap plot
                shap.plots.heatmap(
                    explainer(X),
                    max_display=self.config.shap.max_display,
                    show=False
                )
                plt.title(f'SHAP Heatmap - {asset} {model_type} {task}')
            
            # Save plot
            plot_path = output_dir / f'shap_{plot_type}_{asset}_{model_type}_{task}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            plt.close()
            logger.error(f"Failed to create {plot_type} plot: {str(e)}")
            raise
    
    def get_feature_importance(self, 
                              asset: str,
                              model_type: str,
                              task: str,
                              top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance for a specific model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top feature importance scores
        """
        result_key = f"{asset}_{model_type}_{task}"
        
        if result_key in self._shap_values:
            shap_result = self.results_manager.get_shap_results(asset)
            if task in shap_result and model_type in shap_result[task]:
                feature_importance = shap_result[task][model_type]['feature_importance']
                
                # Return top N features
                return dict(list(feature_importance.items())[:top_n])
        
        return {}
    
    def get_shap_values(self,
                       asset: str,
                       model_type: str,
                       task: str) -> Optional[np.ndarray]:
        """
        Get SHAP values for a specific model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            
        Returns:
            SHAP values array or None if not found
        """
        result_key = f"{asset}_{model_type}_{task}"
        return self._shap_values.get(result_key)
    
    def compare_feature_importance(self,
                                 asset: str,
                                 task: str,
                                 model_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare feature importance across different models.
        
        Args:
            asset: Asset name
            task: Task type
            model_types: List of model types to compare. If None, compares all.
            
        Returns:
            DataFrame with feature importance comparison
        """
        comparison_data = []
        
        shap_results = self.results_manager.get_shap_results(asset)
        if task not in shap_results:
            return pd.DataFrame()
        
        task_results = shap_results[task]
        
        if model_types:
            task_results = {k: v for k, v in task_results.items() if k in model_types}
        
        for model_type, model_result in task_results.items():
            if 'feature_importance' in model_result:
                feature_importance = model_result['feature_importance']
                
                for feature, importance in feature_importance.items():
                    comparison_data.append({
                        'model_type': model_type,
                        'feature': feature,
                        'importance': importance
                    })
        
        return pd.DataFrame(comparison_data)
    
    def generate_summary_report(self, output_dir: str):
        """
        Generate a comprehensive SHAP analysis summary report.
        
        Args:
            output_dir: Output directory for the report
        """
        logger.info("Generating SHAP analysis summary report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all SHAP results
        all_shap_results = self.results_manager.get_shap_results()
        
        # Create summary plots
        self._create_summary_plots(all_shap_results, output_path)
        
        # Create summary statistics
        self._create_summary_statistics(all_shap_results, output_path)
        
        logger.info(f"SHAP summary report generated in {output_path}")
    
    def _create_summary_plots(self, all_shap_results: Dict[str, Any], output_path: Path):
        """Create summary plots for all SHAP results."""
        # Implementation for summary plots
        pass
    
    def _create_summary_statistics(self, all_shap_results: Dict[str, Any], output_path: Path):
        """Create summary statistics for all SHAP results."""
        try:
            # Create summary report
            summary_data = []
            
            for asset, asset_results in all_shap_results.items():
                for task, task_results in asset_results.items():
                    for model_type, model_result in task_results.items():
                        if 'feature_importance' in model_result:
                            feature_importance = model_result['feature_importance']
                            
                            summary_data.append({
                                'asset': asset,
                                'task': task,
                                'model_type': model_type,
                                'top_feature': max(feature_importance.items(), key=lambda x: x[1])[0],
                                'top_importance': max(feature_importance.values()),
                                'mean_importance': np.mean(list(feature_importance.values())),
                                'std_importance': np.std(list(feature_importance.values()))
                            })
            
            # Save summary to CSV
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_path / 'shap_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            logger.info(f"Summary statistics saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Summary statistics creation failed: {str(e)}")
    
    def explain_prediction(self, 
                          asset: str,
                          model_type: str,
                          task: str,
                          instance: Union[np.ndarray, pd.DataFrame],
                          feature_names: List[str],
                          instance_index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction with detailed analysis.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            instance: Instance to explain
            feature_names: List of feature names
            instance_index: Index of the instance
            
        Returns:
            Dictionary containing prediction explanation
        """
        logger.info(f"Explaining prediction for {asset}_{model_type}_{task}")
        
        try:
            # Get the explainer
            explainer = self._explainers.get(f"{asset}_{model_type}_{task}")
            
            if explainer is None:
                # Create explainer if not available
                model = self.results_manager.get_model(asset, model_type, task)
                if model is None:
                    raise ValueError(f"Model not found for {asset}_{model_type}_{task}")
                
                explainer = self.explainer_factory.create_explainer(
                    model=model,
                    model_type=model_type,
                    task=task,
                    X=instance
                )
            
            # Generate individual explanation
            explanation = self.interpretation_utils.generate_individual_explanation(
                explainer=explainer,
                instance=instance,
                feature_names=feature_names,
                instance_index=instance_index
            )
            
            # Add model-specific analysis
            if model_type == 'arima':
                explanation['arima_analysis'] = self._analyze_arima_prediction(
                    explainer, instance, feature_names
                )
            elif model_type == 'stockmixer':
                explanation['pathway_analysis'] = self._analyze_stockmixer_prediction(
                    explainer, instance, feature_names
                )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_feature_interactions(self,
                                   asset: str,
                                   model_type: str,
                                   task: str,
                                   top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze feature interactions for a specific model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            top_k: Number of top interactions to return
            
        Returns:
            Dictionary containing feature interaction analysis
        """
        logger.info(f"Analyzing feature interactions for {asset}_{model_type}_{task}")
        
        try:
            # Get SHAP values and data
            shap_values = self.get_shap_values(asset, model_type, task)
            if shap_values is None:
                raise ValueError(f"No SHAP values found for {asset}_{model_type}_{task}")
            
            # Get feature data
            features = self.results_manager.get_features(asset)
            if features is None:
                raise ValueError(f"No features found for {asset}")
            
            X = features['features']
            feature_names = features.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            
            # Analyze interactions
            interactions = self.interpretation_utils.analyze_feature_interactions(
                shap_values=shap_values,
                X=X,
                feature_names=feature_names,
                top_k=top_k
            )
            
            return interactions
            
        except Exception as e:
            logger.error(f"Feature interaction analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_time_series_shap(self,
                                asset: str,
                                model_type: str,
                                task: str,
                                window_size: int = 30) -> Dict[str, Any]:
        """
        Generate time-series specific SHAP analysis.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            window_size: Rolling window size for analysis
            
        Returns:
            Dictionary containing time-series SHAP analysis
        """
        logger.info(f"Generating time-series SHAP for {asset}_{model_type}_{task}")
        
        try:
            # Get SHAP values and data
            shap_values = self.get_shap_values(asset, model_type, task)
            if shap_values is None:
                raise ValueError(f"No SHAP values found for {asset}_{model_type}_{task}")
            
            # Get feature data
            features = self.results_manager.get_features(asset)
            if features is None:
                raise ValueError(f"No features found for {asset}")
            
            X = features['features']
            feature_names = features.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            
            # Get time index if available
            time_index = features.get('time_index')
            
            # Perform time-series analysis
            time_series_analysis = self.interpretation_utils.analyze_time_series_shap(
                shap_values=shap_values,
                X=X,
                feature_names=feature_names,
                time_index=time_index,
                window_size=window_size
            )
            
            return time_series_analysis
            
        except Exception as e:
            logger.error(f"Time-series SHAP analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def save_shap_data(self,
                      asset: str,
                      model_type: str,
                      task: str,
                      filepath: Optional[Union[str, Path]] = None) -> bool:
        """
        Save SHAP data for a specific model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            filepath: Optional custom filepath
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get SHAP values
            shap_values = self.get_shap_values(asset, model_type, task)
            if shap_values is None:
                raise ValueError(f"No SHAP values found for {asset}_{model_type}_{task}")
            
            # Prepare metadata
            metadata = {
                'asset': asset,
                'model_type': model_type,
                'task': task,
                'shap_values_shape': shap_values.shape,
                'timestamp': datetime.now().isoformat(),
                'feature_names': self._get_feature_names(asset)
            }
            
            # Determine filepath
            if filepath is None:
                filepath = self.shap_data_dir / f"{asset}_{model_type}_{task}_shap"
            
            # Save data
            return self.interpretation_utils.save_shap_data(
                shap_values=shap_values,
                metadata=metadata,
                filepath=filepath
            )
            
        except Exception as e:
            logger.error(f"SHAP data save failed: {str(e)}")
            return False
    
    def load_shap_data(self,
                      asset: str,
                      model_type: str,
                      task: str,
                      filepath: Optional[Union[str, Path]] = None) -> bool:
        """
        Load SHAP data for a specific model.
        
        Args:
            asset: Asset name
            model_type: Type of model
            task: Task type
            filepath: Optional custom filepath
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine filepath
            if filepath is None:
                filepath = self.shap_data_dir / f"{asset}_{model_type}_{task}_shap"
            
            # Load data
            shap_values, metadata = self.interpretation_utils.load_shap_data(filepath)
            
            if shap_values is not None and metadata is not None:
                # Store loaded data
                result_key = f"{asset}_{model_type}_{task}"
                self._shap_values[result_key] = shap_values
                
                logger.info(f"SHAP data loaded for {asset}_{model_type}_{task}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"SHAP data load failed: {str(e)}")
            return False
    
    def _analyze_arima_prediction(self,
                                explainer: Any,
                                instance: Union[np.ndarray, pd.DataFrame],
                                feature_names: List[str]) -> Dict[str, Any]:
        """Analyze ARIMA-specific prediction aspects."""
        try:
            if hasattr(explainer, 'explain'):
                return explainer.explain(instance)
            else:
                return {'error': 'ARIMA explainer not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_stockmixer_prediction(self,
                                     explainer: Any,
                                     instance: Union[np.ndarray, pd.DataFrame],
                                     feature_names: List[str]) -> Dict[str, Any]:
        """Analyze StockMixer-specific prediction aspects."""
        try:
            if hasattr(explainer, 'explain'):
                return explainer.explain(instance)
            else:
                return {'error': 'StockMixer explainer not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_feature_names(self, asset: str) -> List[str]:
        """Get feature names for an asset."""
        try:
            features = self.results_manager.get_features(asset)
            if features and 'feature_names' in features:
                return features['feature_names']
            else:
                return []
        except:
            return [] 

    def save_shap_plots(self, shap_vals, X_eval, feature_names=None, out_dir=None, **kwargs):
        """
        Save SHAP plots to disk.
        
        Args:
            shap_vals: SHAP values array
            X_eval: Evaluation data
            feature_names: List of feature names
            out_dir: Output directory
            **kwargs: Additional parameters (will be cleaned of background_data)
        """
        # 🔎 KILL-SWITCH: Clean any stray background_data parameters
        kwargs = self._cleanup_background_data_params(**kwargs)
        
        try:
            if out_dir is None:
                out_dir = Path(self.config.output.shap_dir)
            
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots using the safe wrapper
            plots = self._generate_shap_plots(
                explainer=None,  # Not needed for basic plotting
                shap_values=shap_vals,
                X=X_eval,
                feature_names=feature_names,
                asset='unknown',
                model_type='unknown',
                task='unknown'
            )
            
            logger.info(f"Saved SHAP plots to {out_dir}")
            return plots
            
        except Exception as e:
            logger.error(f"Failed to save SHAP plots: {str(e)}")
            return {} 

    def _validate_shap_data(self, shap_values, X, feature_names, model_type, task):
        """
        🧰 SHAPE SANITY: Comprehensive validation of SHAP data before plotting.
        
        Args:
            shap_values: SHAP values to validate
            X: Feature data to validate
            feature_names: Feature names to validate
            model_type: Type of model for context
            task: Task type for context
            
        Returns:
            Tuple of (validated_shap_values, validated_X, validated_feature_names)
            
        Raises:
            ValueError: If validation fails
        """
        # 🔒 SHAPE SANITY: Validate SHAP values
        if shap_values is None:
            raise ValueError("🧰 SHAPE SANITY: SHAP values are None")
        
        if isinstance(shap_values, (list, tuple)) and len(shap_values) == 0:
            raise ValueError("🧰 SHAPE SANITY: SHAP values list is empty")
        
        # Convert to numpy array if needed
        if not isinstance(shap_values, np.ndarray):
            try:
                shap_values = np.asarray(shap_values)
            except Exception as e:
                raise ValueError(f"🧰 SHAPE SANITY: Failed to convert SHAP values to numpy array: {e}")
        
        # Validate SHAP array shape
        if shap_values.size == 0:
            raise ValueError("🧰 SHAPE SANITY: SHAP values array is empty")
        
        # Handle different SHAP shapes
        if shap_values.ndim == 1:
            # 1D: reshape to (n_samples, 1)
            shap_values = shap_values.reshape(-1, 1)
        elif shap_values.ndim == 3:
            # 3D: squeeze trailing unit dimension
            if shap_values.shape[-1] == 1:
                shap_values = shap_values[..., 0]
            else:
                # For classification with multiple classes, pick class 1 or mean
                if task == 'classification' and shap_values.shape[-1] > 1:
                    shap_values = shap_values[..., 1]  # Pick positive class
                else:
                    shap_values = np.mean(shap_values, axis=-1)  # Average across classes
        
        # Ensure 2D shape
        if shap_values.ndim != 2:
            raise ValueError(f"🧰 SHAPE SANITY: SHAP values must be 2D after processing, got {shap_values.ndim}D")
        
        # 🔒 SHAPE SANITY: Validate X data
        if X is None:
            raise ValueError("🧰 SHAPE SANITY: Feature data X is None")
        
        # Convert X to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            try:
                X_np = np.asarray(X)
            except Exception as e:
                raise ValueError(f"🧰 SHAPE SANITY: Failed to convert X to numpy array: {e}")
        
        # Handle X shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        elif X_np.ndim > 2:
            # Flatten to 2D
            X_np = X_np.reshape(X_np.shape[0], -1)
        
        # 🔒 FEATURE COUNT ALIGNMENT: Ensure X and SHAP have matching features
        n_samples_shap, n_features_shap = shap_values.shape
        n_samples_X, n_features_X = X_np.shape
        
        if n_samples_shap != n_samples_X:
            logger.warning(f"🧰 SHAPE SANITY: Sample count mismatch: SHAP={n_samples_shap}, X={n_samples_X}")
            # Use minimum sample count
            n_samples = min(n_samples_shap, n_samples_X)
            shap_values = shap_values[:n_samples, :]
            X_np = X_np[:n_samples, :]
        
        # Align feature counts
        if n_features_shap != n_features_X:
            logger.warning(f"🧰 SHAPE SANITY: Feature count mismatch: SHAP={n_features_shap}, X={n_features_X}")
            
            if n_features_shap < n_features_X:
                # SHAP has fewer features: truncate X
                X_np = X_np[:, :n_features_shap]
                logger.info(f"🧰 SHAPE SANITY: Truncated X to {n_features_shap} features to match SHAP")
            else:
                # SHAP has more features: pad X with zeros
                padding = np.zeros((X_np.shape[0], n_features_shap - n_features_X))
                X_np = np.hstack([X_np, padding])
                logger.info(f"🧰 SHAPE SANITY: Padded X with {n_features_shap - n_features_X} zero features")
        
        # 🔒 FEATURE NAMES ALIGNMENT: Ensure feature names match
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(shap_values.shape[1])]
        elif len(feature_names) != shap_values.shape[1]:
            logger.warning(f"🧰 SHAPE SANITY: Feature names count mismatch: {len(feature_names)} != {shap_values.shape[1]}")
            
            if len(feature_names) < shap_values.shape[1]:
                # Pad feature names
                padding_names = [f'padded_feature_{i}' for i in range(shap_values.shape[1] - len(feature_names))]
                feature_names = list(feature_names) + padding_names
                logger.info(f"🧰 SHAPE SANITY: Padded feature names with {len(padding_names)} generic names")
            else:
                # Truncate feature names
                feature_names = feature_names[:shap_values.shape[1]]
                logger.info(f"🧰 SHAPE SANITY: Truncated feature names to {len(feature_names)}")
        
        # Final validation
        final_shap_shape = shap_values.shape
        final_X_shape = X_np.shape
        
        if final_shap_shape[0] != final_X_shape[0]:
            raise ValueError(f"🧰 SHAPE SANITY: Final sample count mismatch: SHAP={final_shap_shape[0]}, X={final_X_shape[0]}")
        
        if final_shap_shape[1] != final_X_shape[1]:
            raise ValueError(f"🧰 SHAPE SANITY: Final feature count mismatch: SHAP={final_shap_shape[1]}, X={final_X_shape[1]}")
        
        if len(feature_names) != final_shap_shape[1]:
            raise ValueError(f"🧰 SHAPE SANITY: Final feature names count mismatch: {len(feature_names)} != {final_shap_shape[1]}")
        
        logger.info(f"🧰 SHAPE SANITY: Validation passed - SHAP: {final_shap_shape}, X: {final_X_shape}, Features: {len(feature_names)}")
        
        return shap_values, X_np, feature_names 