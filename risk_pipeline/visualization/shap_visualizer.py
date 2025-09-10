"""
SHAP Visualization Module for RiskPipeline.

This module provides comprehensive visualization capabilities for SHAP analysis
across all model types in the RiskPipeline.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

logger = logging.getLogger(__name__)


class SHAPVisualizer:
    """
    Comprehensive SHAP visualization for all model types.
    
    Provides:
    - Model-specific visualizations (ARIMA, LSTM, StockMixer, XGBoost)
    - Time-series specific plots
    - Pathway analysis visualizations
    - Comparison plots across models
    - Interactive and static plots
    """
    
    def __init__(self, config: Any):
        """
        Initialize SHAP visualizer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.output_dir = Path(config.output.shap_dir) if hasattr(config.output, 'shap_dir') else Path('shap_plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("SHAPVisualizer initialized")
    
    def summary_plot(self,
                    shap_values: Union[np.ndarray, Any],
                    X: Union[np.ndarray, pd.DataFrame],
                    feature_names: Optional[List[str]] = None) -> None:
        """Create and save a SHAP summary plot.
        This lightweight helper matches calls from SHAPAnalyzer.
        """
        try:
            # Normalize SHAP values object/array
            sv = shap_values.values if hasattr(shap_values, 'values') else shap_values
            if isinstance(sv, list):
                sv = sv[0]
            # Ensure X is aligned shape-wise
            X_plot = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            if hasattr(sv, 'ndim') and sv.ndim > 2:
                sv = sv.reshape(sv.shape[0], -1)
                if X_plot.ndim > 2:
                    X_plot = X_plot.reshape(X_plot.shape[0], -1)

            # Prepare output file
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            out_dir = self.output_dir / 'quick'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'shap_summary_{ts}.png'

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv,
                X_plot,
                feature_names=feature_names,
                max_display=getattr(self.config.shap, 'max_display', 20) if hasattr(self.config, 'shap') else 20,
                show=False
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"summary_plot failed: {e}")
            try:
                plt.close()
            except Exception:
                pass
    
    def bar_plot(self,
                 shap_values: Union[np.ndarray, Any],
                 feature_names: Optional[List[str]] = None) -> None:
        """Create and save a mean |SHAP| bar plot.
        This lightweight helper matches calls from SHAPAnalyzer.
        """
        try:
            sv = shap_values.values if hasattr(shap_values, 'values') else shap_values
            if isinstance(sv, list):
                sv = sv[0]
            sv_arr = np.asarray(sv)
            if sv_arr.ndim > 2:
                sv_arr = sv_arr.reshape(sv_arr.shape[0], -1)
            # Compute importance
            importance = np.mean(np.abs(sv_arr), axis=0)
            # Names
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            # Sort
            order = np.argsort(importance)[::-1]
            importance = importance[order]
            names_sorted = [feature_names[i] for i in order]

            # Prepare output file
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            out_dir = self.output_dir / 'quick'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'shap_bar_{ts}.png'

            plt.figure(figsize=(12, 8))
            sns.barplot(x=importance[:20], y=names_sorted[:20], orient='h')
            plt.xlabel('Mean |SHAP value|')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"bar_plot failed: {e}")
            try:
                plt.close()
            except Exception:
                pass
    
    def create_comprehensive_plots(self,
                                 shap_values: np.ndarray,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 feature_names: List[str],
                                 asset: str,
                                 model_type: str,
                                 task: str,
                                 explainer: Any = None) -> Dict[str, str]:
        """
        Create comprehensive SHAP plots for a model.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            feature_names: List of feature names
            asset: Asset name
            model_type: Type of model
            task: Task type
            explainer: SHAP explainer (optional)
            
        Returns:
            Dictionary of plot file paths
        """
        logger.info(f"Creating comprehensive plots for {asset}_{model_type}_{task}")
        
        plots = {}
        # Guard: if shap_values or X are None/empty, skip plotting gracefully
        if shap_values is None or X is None:
            logger.error("SHAP plots skipped: shap_values or X is None")
            return {'error': 'shap_values_or_X_missing'}
        try:
            if isinstance(X, pd.DataFrame) and X.empty:
                logger.error("SHAP plots skipped: X DataFrame is empty")
                return {'error': 'X_empty'}
            if hasattr(shap_values, 'size') and getattr(shap_values, 'size', 0) == 0:
                logger.error("SHAP plots skipped: shap_values empty")
                return {'error': 'shap_values_empty'}
        except Exception:
            pass
        
        try:
            # Create output directory
            output_dir = self.output_dir / asset / model_type / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Basic SHAP plots
            plots.update(self._create_basic_shap_plots(
                shap_values, X, feature_names, output_dir, asset, model_type, task
            ))
            
            # Model-specific plots
            if model_type == 'arima':
                plots.update(self._create_arima_plots(
                    explainer, X, feature_names, output_dir, asset, task
                ))
            elif model_type == 'stockmixer':
                plots.update(self._create_stockmixer_plots(
                    explainer, X, feature_names, output_dir, asset, task
                ))
            elif model_type == 'lstm':
                plots.update(self._create_lstm_plots(
                    shap_values, X, feature_names, output_dir, asset, task
                ))
            elif model_type == 'xgboost':
                plots.update(self._create_xgboost_plots(
                    shap_values, X, feature_names, output_dir, asset, task
                ))
            
            # Time-series specific plots
            plots.update(self._create_time_series_plots(
                shap_values, X, feature_names, output_dir, asset, model_type, task
            ))
            
            # Feature interaction plots
            plots.update(self._create_interaction_plots(
                shap_values, X, feature_names, output_dir, asset, model_type, task
            ))
            
        except Exception as e:
            logger.error(f"Comprehensive plot creation failed: {str(e)}")
            plots['error'] = str(e)
        
        return plots
    
    def _create_basic_shap_plots(self,
                               shap_values: np.ndarray,
                               X: Union[np.ndarray, pd.DataFrame],
                               feature_names: List[str],
                               output_dir: Path,
                               asset: str,
                               model_type: str,
                               task: str) -> Dict[str, str]:
        """Create basic SHAP plots."""
        plots = {}
        
        try:
            # Summary plot (bar chart)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=feature_names,
                max_display=20,
                show=False
            )
            plt.title(f'SHAP Summary - {asset} {model_type} {task}')
            plt.tight_layout()
            
            summary_path = output_dir / f'shap_summary_{asset}_{model_type}_{task}.png'
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['summary'] = str(summary_path)
            
            # Beeswarm plot
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(
                shap.Explanation(shap_values, feature_names=feature_names),
                max_display=20,
                show=False
            )
            plt.title(f'SHAP Beeswarm - {asset} {model_type} {task}')
            plt.tight_layout()
            
            beeswarm_path = output_dir / f'shap_beeswarm_{asset}_{model_type}_{task}.png'
            plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['beeswarm'] = str(beeswarm_path)
            
            # Waterfall plot for a sample
            plt.figure(figsize=(12, 8))
            sample_idx = 0
            shap.waterfall_plot(
                shap.Explanation(
                    shap_values[sample_idx],
                    base_values=0,
                    feature_names=feature_names
                ),
                max_display=20,
                show=False
            )
            plt.title(f'SHAP Waterfall - {asset} {model_type} {task} (Sample {sample_idx})')
            plt.tight_layout()
            
            waterfall_path = output_dir / f'shap_waterfall_{asset}_{model_type}_{task}.png'
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['waterfall'] = str(waterfall_path)
            
        except Exception as e:
            logger.error(f"Basic SHAP plots failed: {str(e)}")
        
        return plots
    
    def _create_arima_plots(self,
                          explainer: Any,
                          X: Union[np.ndarray, pd.DataFrame],
                          feature_names: List[str],
                          output_dir: Path,
                          asset: str,
                          task: str) -> Dict[str, str]:
        """Create ARIMA-specific plots."""
        plots = {}
        
        try:
            if hasattr(explainer, 'explain'):
                explanations = explainer.explain(X)
                
                # Residuals plot
                if 'residuals' in explanations:
                    residuals = explanations['residuals']
                    if 'residuals' in residuals:
                        plt.figure(figsize=(12, 8))
                        plt.plot(residuals['residuals'])
                        plt.title(f'ARIMA Residuals - {asset} {task}')
                        plt.xlabel('Time')
                        plt.ylabel('Residuals')
                        plt.grid(True)
                        
                        residuals_path = output_dir / f'arima_residuals_{asset}_{task}.png'
                        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['residuals'] = str(residuals_path)
                
                # Decomposition plot
                if 'decomposition' in explanations:
                    decomp = explanations['decomposition']
                    if 'trend' in decomp and 'seasonal' in decomp:
                        plt.figure(figsize=(15, 10))
                        
                        plt.subplot(4, 1, 1)
                        plt.plot(decomp.get('trend', []))
                        plt.title('Trend Component')
                        plt.grid(True)
                        
                        plt.subplot(4, 1, 2)
                        plt.plot(decomp.get('seasonal', []))
                        plt.title('Seasonal Component')
                        plt.grid(True)
                        
                        plt.subplot(4, 1, 3)
                        plt.plot(decomp.get('residual', []))
                        plt.title('Residual Component')
                        plt.grid(True)
                        
                        plt.subplot(4, 1, 4)
                        plt.plot(decomp.get('trend', []))
                        plt.plot(decomp.get('seasonal', []))
                        plt.plot(decomp.get('residual', []))
                        plt.title('All Components')
                        plt.legend(['Trend', 'Seasonal', 'Residual'])
                        plt.grid(True)
                        
                        decomp_path = output_dir / f'arima_decomposition_{asset}_{task}.png'
                        plt.savefig(decomp_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['decomposition'] = str(decomp_path)
                
                # Forecast intervals plot
                if 'forecast_intervals' in explanations:
                    forecast = explanations['forecast_intervals']
                    if 'forecast' in forecast and 'confidence_intervals' in forecast:
                        plt.figure(figsize=(12, 8))
                        
                        forecast_vals = forecast['forecast']
                        conf_int = forecast['confidence_intervals']
                        
                        plt.plot(forecast_vals, 'b-', label='Forecast')
                        plt.fill_between(
                            range(len(forecast_vals)),
                            conf_int['lower'],
                            conf_int['upper'],
                            alpha=0.3,
                            label='Confidence Interval'
                        )
                        plt.title(f'ARIMA Forecast - {asset} {task}')
                        plt.xlabel('Time Steps')
                        plt.ylabel('Value')
                        plt.legend()
                        plt.grid(True)
                        
                        forecast_path = output_dir / f'arima_forecast_{asset}_{task}.png'
                        plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['forecast'] = str(forecast_path)
                        
        except Exception as e:
            logger.error(f"ARIMA plots failed: {str(e)}")
        
        return plots
    
    def _create_stockmixer_plots(self,
                               explainer: Any,
                               X: Union[np.ndarray, pd.DataFrame],
                               feature_names: List[str],
                               output_dir: Path,
                               asset: str,
                               task: str) -> Dict[str, str]:
        """Create StockMixer-specific plots."""
        plots = {}
        
        try:
            if hasattr(explainer, 'explain'):
                explanations = explainer.explain(X)
                
                # Pathway analysis plots
                if 'pathways' in explanations:
                    pathways = explanations['pathways']
                    
                    # Pathway activation comparison
                    plt.figure(figsize=(15, 10))
                    
                    pathway_names = list(pathways.keys())
                    pathway_stats = []
                    
                    for pathway_name, pathway_data in pathways.items():
                        if 'mean_activation' in pathway_data:
                            pathway_stats.append({
                                'name': pathway_name,
                                'mean': pathway_data['mean_activation'],
                                'std': pathway_data['std_activation']
                            })
                    
                    if pathway_stats:
                        names = [p['name'] for p in pathway_stats]
                        means = [p['mean'] for p in pathway_stats]
                        stds = [p['std'] for p in pathway_stats]
                        
                        plt.subplot(2, 2, 1)
                        plt.bar(names, means, yerr=stds, capsize=5)
                        plt.title('Pathway Activation Means')
                        plt.ylabel('Mean Activation')
                        plt.xticks(rotation=45)
                        
                        plt.subplot(2, 2, 2)
                        plt.bar(names, stds)
                        plt.title('Pathway Activation Variability')
                        plt.ylabel('Standard Deviation')
                        plt.xticks(rotation=45)
                        
                        # Pathway importance heatmap
                        plt.subplot(2, 2, 3)
                        pathway_matrix = np.array([[p['mean'], p['std']] for p in pathway_stats])
                        sns.heatmap(
                            pathway_matrix.T,
                            annot=True,
                            fmt='.3f',
                            xticklabels=names,
                            yticklabels=['Mean', 'Std'],
                            cmap='viridis'
                        )
                        plt.title('Pathway Statistics Heatmap')
                        
                        plt.tight_layout()
                        
                        pathway_path = output_dir / f'stockmixer_pathways_{asset}_{task}.png'
                        plt.savefig(pathway_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['pathways'] = str(pathway_path)
                
                # Feature mixing analysis
                if 'feature_mixing' in explanations:
                    mixing = explanations['feature_mixing']
                    
                    plt.figure(figsize=(12, 8))
                    
                    mixing_data = []
                    mixing_names = []
                    
                    for pathway_name, pathway_data in mixing.items():
                        if 'mean' in pathway_data:
                            mixing_data.append(pathway_data['mean'])
                            mixing_names.append(pathway_name)
                    
                    if mixing_data:
                        plt.bar(mixing_names, mixing_data)
                        plt.title(f'StockMixer Feature Mixing - {asset} {task}')
                        plt.ylabel('Mean Activation')
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3)
                        
                        mixing_path = output_dir / f'stockmixer_mixing_{asset}_{task}.png'
                        plt.savefig(mixing_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['mixing'] = str(mixing_path)
                        
        except Exception as e:
            logger.error(f"StockMixer plots failed: {str(e)}")
        
        return plots
    
    def _create_lstm_plots(self,
                         shap_values: np.ndarray,
                         X: Union[np.ndarray, pd.DataFrame],
                         feature_names: List[str],
                         output_dir: Path,
                         asset: str,
                         task: str) -> Dict[str, str]:
        """Create LSTM-specific plots."""
        plots = {}
        
        try:
            # Temporal SHAP heatmap
            plt.figure(figsize=(15, 8))
            
            # Reshape SHAP values for temporal visualization
            if len(shap_values.shape) == 2:
                # If 2D, assume time steps are in rows
                temporal_shap = shap_values
            else:
                # If 3D, take mean across time dimension
                temporal_shap = np.mean(shap_values, axis=1)
            
            # Create heatmap
            sns.heatmap(
                temporal_shap.T,
                cmap='RdBu_r',
                center=0,
                xticklabels=100,  # Show every 100th time step
                yticklabels=feature_names,
                cbar_kws={'label': 'SHAP Value'}
            )
            plt.title(f'LSTM Temporal SHAP - {asset} {task}')
            plt.xlabel('Time Steps')
            plt.ylabel('Features')
            
            temporal_path = output_dir / f'lstm_temporal_{asset}_{task}.png'
            plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['temporal'] = str(temporal_path)
            
            # Feature importance over time
            plt.figure(figsize=(12, 8))
            
            feature_importance = np.mean(np.abs(temporal_shap), axis=1)
            plt.plot(feature_importance)
            plt.title(f'LSTM Feature Importance Over Time - {asset} {task}')
            plt.xlabel('Time Steps')
            plt.ylabel('Mean Absolute SHAP Value')
            plt.grid(True, alpha=0.3)
            
            importance_path = output_dir / f'lstm_importance_time_{asset}_{task}.png'
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['importance_time'] = str(importance_path)
            
        except Exception as e:
            logger.error(f"LSTM plots failed: {str(e)}")
        
        return plots
    
    def _create_xgboost_plots(self,
                            shap_values: np.ndarray,
                            X: Union[np.ndarray, pd.DataFrame],
                            feature_names: List[str],
                            output_dir: Path,
                            asset: str,
                            task: str) -> Dict[str, str]:
        """Create XGBoost-specific plots."""
        plots = {}
        
        try:
            # Normalize inputs to numpy for consistent indexing
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            sv_arr = np.asarray(shap_values)
            if sv_arr.ndim > 2:
                sv_arr = sv_arr.reshape(sv_arr.shape[0], -1)
            if X_arr.ndim > 2:
                X_arr = X_arr.reshape(X_arr.shape[0], -1)

            # Align feature names length
            num_features = X_arr.shape[1] if X_arr.ndim == 2 else sv_arr.shape[1]
            if len(feature_names) != num_features:
                feature_names = feature_names[:num_features]
                if len(feature_names) < num_features:
                    feature_names += [f'feature_{i}' for i in range(len(feature_names), num_features)]

            # Dependence plots for top features
            top_features = self._get_top_features(sv_arr, feature_names, top_k=5)
            
            plt.figure(figsize=(15, 10))
            
            for i, feature in enumerate(top_features):
                plt.subplot(2, 3, i + 1)
                
                # Get feature index
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                else:
                    continue
                
                # Create dependence plot
                try:
                    plt.scatter(
                        X_arr[:, feature_idx],
                        sv_arr[:, feature_idx],
                        alpha=0.6,
                        s=20
                    )
                except Exception as dep_err:
                    logger.warning(f"Dependence plot fallback for feature {feature}: {dep_err}")
                    try:
                        plt.plot(sv_arr[:, feature_idx])
                    except Exception:
                        pass
                plt.xlabel(feature)
                plt.ylabel('SHAP Value')
                plt.title(f'Dependence: {feature}')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            dependence_path = output_dir / f'xgboost_dependence_{asset}_{task}.png'
            plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['dependence'] = str(dependence_path)
            
            # Feature interaction plot
            plt.figure(figsize=(12, 8))
            
            # Calculate feature interaction matrix
            sv_for_corr = sv_arr if sv_arr.ndim == 2 else sv_arr.reshape(sv_arr.shape[0], -1)
            if sv_for_corr.shape[1] >= 2:
                interaction_matrix = np.corrcoef(np.abs(sv_for_corr).T)
            else:
                interaction_matrix = np.array([[1.0]])
            
            sns.heatmap(
                interaction_matrix,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                square=True
            )
            plt.title(f'XGBoost Feature Interactions - {asset} {task}')
            
            interaction_path = output_dir / f'xgboost_interactions_{asset}_{task}.png'
            plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['interactions'] = str(interaction_path)
            
        except Exception as e:
            logger.error(f"XGBoost plots failed: {str(e)}")
        
        return plots
    
    def _create_time_series_plots(self,
                                shap_values: np.ndarray,
                                X: Union[np.ndarray, pd.DataFrame],
                                feature_names: List[str],
                                output_dir: Path,
                                asset: str,
                                model_type: str,
                                task: str) -> Dict[str, str]:
        """Create time-series specific plots."""
        plots = {}
        
        try:
            # Rolling SHAP statistics
            plt.figure(figsize=(15, 10))
            
            # Calculate rolling statistics
            window_size = 30
            rolling_mean = pd.DataFrame(shap_values).rolling(window=window_size, min_periods=1).mean()
            rolling_std = pd.DataFrame(shap_values).rolling(window=window_size, min_periods=1).std()
            
            # Plot rolling statistics for top features
            top_features = self._get_top_features(shap_values, feature_names, top_k=5)
            
            for i, feature in enumerate(top_features):
                feature_idx = feature_names.index(feature)
                
                plt.subplot(2, 3, i + 1)
                plt.plot(rolling_mean.iloc[:, feature_idx], label='Rolling Mean', linewidth=2)
                plt.fill_between(
                    range(len(rolling_mean)),
                    rolling_mean.iloc[:, feature_idx] - rolling_std.iloc[:, feature_idx],
                    rolling_mean.iloc[:, feature_idx] + rolling_std.iloc[:, feature_idx],
                    alpha=0.3,
                    label='±1 Std'
                )
                plt.title(f'Rolling SHAP: {feature}')
                plt.xlabel('Time Steps')
                plt.ylabel('SHAP Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            rolling_path = output_dir / f'timeseries_rolling_{asset}_{model_type}_{task}.png'
            plt.savefig(rolling_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['rolling'] = str(rolling_path)
            
            # Regime change detection
            plt.figure(figsize=(12, 8))
            
            # Calculate overall feature importance over time
            feature_importance = np.mean(np.abs(shap_values), axis=1)
            rolling_importance = pd.Series(feature_importance).rolling(
                window=window_size, min_periods=1
            ).mean()
            
            # Detect regime changes
            threshold = np.std(feature_importance) * 2
            changes = np.where(np.abs(np.diff(rolling_importance)) > threshold)[0]
            
            plt.plot(rolling_importance, label='Rolling Importance', linewidth=2)
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Change Threshold')
            
            for change_point in changes:
                plt.axvline(x=change_point, color='red', alpha=0.5, linestyle=':')
            
            plt.title(f'Regime Change Detection - {asset} {model_type} {task}')
            plt.xlabel('Time Steps')
            plt.ylabel('Feature Importance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            regime_path = output_dir / f'timeseries_regime_{asset}_{model_type}_{task}.png'
            plt.savefig(regime_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['regime'] = str(regime_path)
            
        except Exception as e:
            logger.error(f"Time-series plots failed: {str(e)}")
        
        return plots
    
    def _create_interaction_plots(self,
                                shap_values: np.ndarray,
                                X: Union[np.ndarray, pd.DataFrame],
                                feature_names: List[str],
                                output_dir: Path,
                                asset: str,
                                model_type: str,
                                task: str) -> Dict[str, str]:
        """Create feature interaction plots."""
        plots = {}
        
        try:
            # Feature interaction heatmap
            plt.figure(figsize=(12, 10))
            
            # Calculate pairwise interactions
            interaction_matrix = np.corrcoef(np.abs(shap_values).T)
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))
            
            sns.heatmap(
                interaction_matrix,
                mask=mask,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                square=True,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'}
            )
            plt.title(f'Feature Interaction Matrix - {asset} {model_type} {task}')
            
            interaction_path = output_dir / f'interaction_matrix_{asset}_{model_type}_{task}.png'
            plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['interaction_matrix'] = str(interaction_path)
            
            # Top interaction pairs
            plt.figure(figsize=(12, 8))
            
            # Get top interaction pairs
            top_interactions = self._get_top_interactions(interaction_matrix, feature_names, top_k=10)
            
            interaction_names = [f"{pair[0]} vs {pair[1]}" for pair in top_interactions]
            interaction_strengths = [pair[2] for pair in top_interactions]
            
            plt.barh(range(len(interaction_names)), interaction_strengths)
            plt.yticks(range(len(interaction_names)), interaction_names)
            plt.xlabel('Interaction Strength')
            plt.title(f'Top Feature Interactions - {asset} {model_type} {task}')
            plt.grid(True, alpha=0.3)
            
            top_interactions_path = output_dir / f'top_interactions_{asset}_{model_type}_{task}.png'
            plt.savefig(top_interactions_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['top_interactions'] = str(top_interactions_path)
            
        except Exception as e:
            logger.error(f"Interaction plots failed: {str(e)}")
        
        return plots
    
    def create_comparison_plots(self,
                              shap_results: Dict[str, Any],
                              assets: List[str],
                              model_types: List[str],
                              task: str) -> Dict[str, str]:
        """
        Create comparison plots across models and assets.
        
        Args:
            shap_results: Dictionary of SHAP results
            assets: List of assets
            model_types: List of model types
            task: Task type
            
        Returns:
            Dictionary of plot file paths
        """
        logger.info("Creating comparison plots")
        
        plots = {}
        
        try:
            # Create output directory
            output_dir = self.output_dir / 'comparisons' / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Feature importance comparison
            plots['importance_comparison'] = self._create_importance_comparison(
                shap_results, assets, model_types, task, output_dir
            )
            
            # Model performance comparison
            plots['performance_comparison'] = self._create_performance_comparison(
                shap_results, assets, model_types, task, output_dir
            )
            
            # Asset comparison
            plots['asset_comparison'] = self._create_asset_comparison(
                shap_results, assets, model_types, task, output_dir
            )
            
        except Exception as e:
            logger.error(f"Comparison plots failed: {str(e)}")
            plots['error'] = str(e)
        
        return plots
    
    def _create_importance_comparison(self,
                                    shap_results: Dict[str, Any],
                                    assets: List[str],
                                    model_types: List[str],
                                    task: str,
                                    output_dir: Path) -> str:
        """Create feature importance comparison plot."""
        try:
            plt.figure(figsize=(15, 10))
            
            # Collect data for comparison
            comparison_data = []
            
            for asset in assets:
                if asset in shap_results and task in shap_results[asset]:
                    task_results = shap_results[asset][task]
                    
                    for model_type in model_types:
                        if model_type in task_results:
                            model_result = task_results[model_type]
                            
                            if 'feature_importance' in model_result:
                                feature_importance = model_result['feature_importance']
                                
                                for feature, importance in feature_importance.items():
                                    comparison_data.append({
                                        'asset': asset,
                                        'model_type': model_type,
                                        'feature': feature,
                                        'importance': importance
                                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                
                # Create grouped bar plot
                pivot_df = df.pivot_table(
                    values='importance',
                    index='feature',
                    columns=['asset', 'model_type'],
                    aggfunc='mean'
                )
                
                pivot_df.plot(kind='bar', figsize=(15, 10))
                plt.title(f'Feature Importance Comparison - {task}')
                plt.xlabel('Features')
                plt.ylabel('SHAP Importance')
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                comparison_path = output_dir / f'importance_comparison_{task}.png'
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(comparison_path)
            
        except Exception as e:
            logger.error(f"Importance comparison failed: {str(e)}")
        
        return ""
    
    def _create_performance_comparison(self,
                                     shap_results: Dict[str, Any],
                                     assets: List[str],
                                     model_types: List[str],
                                     task: str,
                                     output_dir: Path) -> str:
        """Create model performance comparison plot."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Collect performance metrics
            performance_data = []
            
            for asset in assets:
                if asset in shap_results and task in shap_results[asset]:
                    task_results = shap_results[asset][task]
                    
                    for model_type in model_types:
                        if model_type in task_results:
                            model_result = task_results[model_type]
                            
                            if 'feature_importance' in model_result:
                                feature_importance = model_result['feature_importance']
                                
                                performance_data.append({
                                    'asset': asset,
                                    'model_type': model_type,
                                    'mean_importance': np.mean(list(feature_importance.values())),
                                    'std_importance': np.std(list(feature_importance.values())),
                                    'max_importance': max(feature_importance.values())
                                })
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                
                # Create grouped bar plot
                plt.subplot(1, 2, 1)
                df.groupby('model_type')['mean_importance'].mean().plot(kind='bar')
                plt.title('Mean Feature Importance by Model')
                plt.ylabel('Mean Importance')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                df.groupby('model_type')['std_importance'].mean().plot(kind='bar')
                plt.title('Feature Importance Variability by Model')
                plt.ylabel('Standard Deviation')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                performance_path = output_dir / f'performance_comparison_{task}.png'
                plt.savefig(performance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(performance_path)
            
        except Exception as e:
            logger.error(f"Performance comparison failed: {str(e)}")
        
        return ""
    
    def _create_asset_comparison(self,
                               shap_results: Dict[str, Any],
                               assets: List[str],
                               model_types: List[str],
                               task: str,
                               output_dir: Path) -> str:
        """Create asset comparison plot."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Collect asset-specific data
            asset_data = []
            
            for asset in assets:
                if asset in shap_results and task in shap_results[asset]:
                    task_results = shap_results[asset][task]
                    
                    asset_importance = []
                    for model_type in model_types:
                        if model_type in task_results:
                            model_result = task_results[model_type]
                            
                            if 'feature_importance' in model_result:
                                feature_importance = model_result['feature_importance']
                                asset_importance.append(np.mean(list(feature_importance.values())))
                    
                    if asset_importance:
                        asset_data.append({
                            'asset': asset,
                            'mean_importance': np.mean(asset_importance),
                            'std_importance': np.std(asset_importance)
                        })
            
            if asset_data:
                df = pd.DataFrame(asset_data)
                
                plt.bar(df['asset'], df['mean_importance'], yerr=df['std_importance'], capsize=5)
                plt.title(f'Asset Comparison - {task}')
                plt.xlabel('Assets')
                plt.ylabel('Mean Feature Importance')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                asset_path = output_dir / f'asset_comparison_{task}.png'
                plt.savefig(asset_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(asset_path)
            
        except Exception as e:
            logger.error(f"Asset comparison failed: {str(e)}")
        
        return ""
    
    def _get_top_features(self,
                         shap_values: np.ndarray,
                         feature_names: List[str],
                         top_k: int = 10) -> List[str]:
        """Get top K features by importance."""
        try:
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_importance)[-top_k:][::-1]
            return [feature_names[i] for i in top_indices]
        except:
            return feature_names[:top_k]
    
    def _get_top_interactions(self,
                            interaction_matrix: np.ndarray,
                            feature_names: List[str],
                            top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Get top K feature interaction pairs."""
        try:
            # Get upper triangle indices
            upper_triangle = np.triu_indices_from(interaction_matrix, k=1)
            
            # Get interaction strengths
            interactions = []
            for i, j in zip(upper_triangle[0], upper_triangle[1]):
                interactions.append((
                    feature_names[i],
                    feature_names[j],
                    abs(interaction_matrix[i, j])
                ))
            
            # Sort by strength and return top K
            interactions.sort(key=lambda x: x[2], reverse=True)
            return interactions[:top_k]
            
        except:
            return [] 