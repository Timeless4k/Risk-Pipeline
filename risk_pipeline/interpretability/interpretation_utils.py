"""
Interpretation utilities for RiskPipeline - Advanced SHAP Analysis.

This module provides utilities for comprehensive model interpretability,
including time-series specific analysis, feature interactions, and data persistence.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

logger = logging.getLogger(__name__)


class InterpretationUtils:
    """
    Advanced interpretation utilities for comprehensive SHAP analysis.
    
    Provides:
    - Time-series specific SHAP analysis
    - Feature interaction detection
    - SHAP data persistence and retrieval
    - Advanced visualization utilities
    - Statistical analysis tools
    """
    
    def __init__(self, config: Any):
        """
        Initialize interpretation utilities.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.shap_data_dir = Path(config.output.shap_dir) if hasattr(config.output, 'shap_dir') else Path('shap_data')
        self.shap_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("InterpretationUtils initialized")
    
    def analyze_time_series_shap(self,
                                shap_values: np.ndarray,
                                X: Union[np.ndarray, pd.DataFrame],
                                feature_names: List[str],
                                time_index: Optional[pd.DatetimeIndex] = None,
                                window_size: int = 30) -> Dict[str, Any]:
        """
        Perform time-series specific SHAP analysis.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            feature_names: List of feature names
            time_index: Time index for the data
            window_size: Rolling window size for analysis
            
        Returns:
            Dictionary containing time-series SHAP analysis results
        """
        logger.info("Performing time-series SHAP analysis")
        
        analysis_results = {}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X.copy()
            
            # Add time index if provided
            if time_index is not None:
                X_df.index = time_index
            
            # Calculate rolling SHAP statistics
            analysis_results['rolling_stats'] = self._calculate_rolling_shap_stats(
                shap_values, X_df, window_size
            )
            
            # Analyze temporal feature importance
            analysis_results['temporal_importance'] = self._analyze_temporal_importance(
                shap_values, X_df, feature_names
            )
            
            # Detect regime changes in feature importance
            analysis_results['regime_changes'] = self._detect_regime_changes(
                shap_values, X_df, window_size
            )
            
            # Analyze seasonality in SHAP values
            analysis_results['seasonality'] = self._analyze_shap_seasonality(
                shap_values, X_df
            )
            
        except Exception as e:
            logger.error(f"Time-series SHAP analysis failed: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _calculate_rolling_shap_stats(self,
                                    shap_values: np.ndarray,
                                    X_df: pd.DataFrame,
                                    window_size: int) -> Dict[str, pd.DataFrame]:
        """Calculate rolling statistics for SHAP values."""
        try:
            # Convert SHAP values to DataFrame
            shap_df = pd.DataFrame(shap_values, columns=X_df.columns, index=X_df.index)
            
            # Calculate rolling statistics
            rolling_mean = shap_df.rolling(window=window_size, min_periods=1).mean()
            rolling_std = shap_df.rolling(window=window_size, min_periods=1).std()
            rolling_max = shap_df.rolling(window=window_size, min_periods=1).max()
            rolling_min = shap_df.rolling(window=window_size, min_periods=1).min()
            
            return {
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'rolling_max': rolling_max,
                'rolling_min': rolling_min
            }
        except Exception as e:
            logger.error(f"Rolling SHAP stats calculation failed: {str(e)}")
            return {}
    
    def _analyze_temporal_importance(self,
                                   shap_values: np.ndarray,
                                   X_df: pd.DataFrame,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Analyze how feature importance changes over time."""
        try:
            # Calculate feature importance over time
            temporal_importance = {}
            
            for i, feature in enumerate(feature_names):
                feature_shap = shap_values[:, i]
                
                # Calculate statistics over time
                temporal_importance[feature] = {
                    'mean': float(np.mean(feature_shap)),
                    'std': float(np.std(feature_shap)),
                    'trend': self._calculate_trend(feature_shap),
                    'volatility': float(np.std(feature_shap) / np.mean(np.abs(feature_shap)))
                }
            
            return temporal_importance
            
        except Exception as e:
            logger.error(f"Temporal importance analysis failed: {str(e)}")
            return {}
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend in values."""
        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _detect_regime_changes(self,
                             shap_values: np.ndarray,
                             X_df: pd.DataFrame,
                             window_size: int) -> Dict[str, Any]:
        """Detect regime changes in feature importance."""
        try:
            # Calculate rolling feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=1)
            
            # Calculate rolling mean and detect changes
            rolling_mean = pd.Series(feature_importance).rolling(
                window=window_size, min_periods=1
            ).mean()
            
            # Detect significant changes
            threshold = np.std(feature_importance) * 2
            changes = np.where(np.abs(np.diff(rolling_mean)) > threshold)[0]
            
            return {
                'change_points': changes.tolist(),
                'rolling_importance': rolling_mean.tolist(),
                'threshold': float(threshold)
            }
            
        except Exception as e:
            logger.error(f"Regime change detection failed: {str(e)}")
            return {}
    
    def _analyze_shap_seasonality(self,
                                shap_values: np.ndarray,
                                X_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality patterns in SHAP values."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Calculate overall feature importance over time
            feature_importance = np.mean(np.abs(shap_values), axis=1)
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                feature_importance,
                period=min(12, len(feature_importance) // 4),
                extrapolate_trend='freq'
            )
            
            return {
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist()
            }
            
        except Exception as e:
            logger.error(f"Seasonality analysis failed: {str(e)}")
            return {}
    
    def analyze_feature_interactions(self,
                                   shap_values: np.ndarray,
                                   X: Union[np.ndarray, pd.DataFrame],
                                   feature_names: List[str],
                                   top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP values.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            feature_names: List of feature names
            top_k: Number of top interactions to return
            
        Returns:
            Dictionary containing feature interaction analysis
        """
        logger.info("Analyzing feature interactions")
        
        interaction_results = {}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X.copy()
            
            # Calculate pairwise interactions
            interactions = self._calculate_pairwise_interactions(
                shap_values, X_df, feature_names
            )
            
            # Find top interactions
            top_interactions = self._get_top_interactions(interactions, top_k)
            
            # Analyze interaction patterns
            interaction_patterns = self._analyze_interaction_patterns(
                shap_values, X_df, feature_names
            )
            
            interaction_results = {
                'pairwise_interactions': interactions,
                'top_interactions': top_interactions,
                'interaction_patterns': interaction_patterns
            }
            
        except Exception as e:
            logger.error(f"Feature interaction analysis failed: {str(e)}")
            interaction_results['error'] = str(e)
        
        return interaction_results
    
    def _calculate_pairwise_interactions(self,
                                       shap_values: np.ndarray,
                                       X_df: pd.DataFrame,
                                       feature_names: List[str]) -> Dict[str, float]:
        """Calculate pairwise feature interactions."""
        try:
            interactions = {}
            
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    feature1 = feature_names[i]
                    feature2 = feature_names[j]
                    
                    # Calculate correlation between SHAP values
                    shap_corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                    
                    # Calculate interaction strength
                    interaction_strength = abs(shap_corr)
                    
                    key = f"{feature1}__{feature2}"
                    interactions[key] = float(interaction_strength)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Pairwise interaction calculation failed: {str(e)}")
            return {}
    
    def _get_top_interactions(self,
                            interactions: Dict[str, float],
                            top_k: int) -> List[Tuple[str, float]]:
        """Get top K feature interactions."""
        try:
            sorted_interactions = sorted(
                interactions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_interactions[:top_k]
        except Exception as e:
            logger.error(f"Top interactions extraction failed: {str(e)}")
            return []
    
    def _analyze_interaction_patterns(self,
                                   shap_values: np.ndarray,
                                   X_df: pd.DataFrame,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Analyze patterns in feature interactions."""
        try:
            # Calculate feature importance correlation matrix
            importance_corr = np.corrcoef(np.abs(shap_values).T)
            
            # Find clusters of related features
            clusters = self._find_feature_clusters(importance_corr, feature_names)
            
            return {
                'importance_correlation': importance_corr.tolist(),
                'feature_clusters': clusters
            }
            
        except Exception as e:
            logger.error(f"Interaction pattern analysis failed: {str(e)}")
            return {}
    
    def _find_feature_clusters(self,
                             correlation_matrix: np.ndarray,
                             feature_names: List[str],
                             threshold: float = 0.5) -> List[List[str]]:
        """Find clusters of related features based on correlation."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Use hierarchical clustering to find feature clusters
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - threshold,
                linkage='complete'
            )
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            clusters = clustering.fit_predict(distance_matrix)
            
            # Group features by cluster
            feature_clusters = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in feature_clusters:
                    feature_clusters[cluster_id] = []
                feature_clusters[cluster_id].append(feature_names[i])
            
            return list(feature_clusters.values())
            
        except Exception as e:
            logger.error(f"Feature clustering failed: {str(e)}")
            return []
    
    def save_shap_data(self,
                      shap_values: np.ndarray,
                      metadata: Dict[str, Any],
                      filepath: Union[str, Path]) -> bool:
        """
        Save SHAP values and metadata to disk.
        
        Args:
            shap_values: SHAP values array
            metadata: Metadata dictionary
            filepath: Path to save the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save SHAP values
            shap_file = filepath.with_suffix('.pkl')
            with open(shap_file, 'wb') as f:
                pickle.dump(shap_values, f)
            
            # Save metadata
            meta_file = filepath.with_suffix('.json')
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"SHAP data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save SHAP data: {str(e)}")
            return False
    
    def load_shap_data(self, filepath: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load SHAP values and metadata from disk.
        
        Args:
            filepath: Path to the saved data
            
        Returns:
            Tuple of (shap_values, metadata) or (None, None) if failed
        """
        try:
            filepath = Path(filepath)
            
            # Load SHAP values
            shap_file = filepath.with_suffix('.pkl')
            with open(shap_file, 'rb') as f:
                shap_values = pickle.load(f)
            
            # Load metadata
            meta_file = filepath.with_suffix('.json')
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"SHAP data loaded from {filepath}")
            return shap_values, metadata
            
        except Exception as e:
            logger.error(f"Failed to load SHAP data: {str(e)}")
            return None, None
    
    def generate_individual_explanation(self,
                                      explainer: Any,
                                      instance: Union[np.ndarray, pd.DataFrame],
                                      feature_names: List[str],
                                      instance_index: int = 0) -> Dict[str, Any]:
        """
        Generate individual prediction explanation.
        
        Args:
            explainer: SHAP explainer
            instance: Single instance to explain
            feature_names: List of feature names
            instance_index: Index of the instance
            
        Returns:
            Dictionary containing individual explanation
        """
        try:
            # Generate SHAP values for the instance
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(instance)
                
                # Handle classification case
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values = explainer(instance).values
            
            # Create feature contribution dictionary
            contributions = {}
            for i, feature in enumerate(feature_names):
                if i < len(shap_values[instance_index]):
                    contributions[feature] = float(shap_values[instance_index, i])
            
            # Sort by absolute contribution
            sorted_contributions = dict(
                sorted(contributions.items(), 
                      key=lambda x: abs(x[1]), 
                      reverse=True)
            )
            
            return {
                'instance_index': instance_index,
                'feature_contributions': sorted_contributions,
                'total_contribution': float(np.sum(shap_values[instance_index])),
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            logger.error(f"Individual explanation failed: {str(e)}")
            return {'error': str(e)}
    
    def create_comparison_analysis(self,
                                 shap_results: Dict[str, Any],
                                 assets: List[str],
                                 model_types: List[str],
                                 task: str) -> Dict[str, Any]:
        """
        Create comparison analysis across models and assets.
        
        Args:
            shap_results: Dictionary of SHAP results
            assets: List of assets
            model_types: List of model types
            task: Task type
            
        Returns:
            Dictionary containing comparison analysis
        """
        logger.info("Creating comparison analysis")
        
        comparison_results = {}
        
        try:
            # Compare feature importance across models
            feature_importance_comparison = self._compare_feature_importance(
                shap_results, assets, model_types, task
            )
            
            # Compare model performance patterns
            performance_patterns = self._compare_performance_patterns(
                shap_results, assets, model_types, task
            )
            
            # Create summary statistics
            summary_stats = self._create_summary_statistics(
                shap_results, assets, model_types, task
            )
            
            comparison_results = {
                'feature_importance_comparison': feature_importance_comparison,
                'performance_patterns': performance_patterns,
                'summary_statistics': summary_stats
            }
            
        except Exception as e:
            logger.error(f"Comparison analysis failed: {str(e)}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def _compare_feature_importance(self,
                                  shap_results: Dict[str, Any],
                                  assets: List[str],
                                  model_types: List[str],
                                  task: str) -> pd.DataFrame:
        """Compare feature importance across models and assets."""
        try:
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
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Feature importance comparison failed: {str(e)}")
            return pd.DataFrame()
    
    def _compare_performance_patterns(self,
                                    shap_results: Dict[str, Any],
                                    assets: List[str],
                                    model_types: List[str],
                                    task: str) -> Dict[str, Any]:
        """Compare performance patterns across models."""
        try:
            patterns = {}
            
            for model_type in model_types:
                model_patterns = {}
                
                for asset in assets:
                    if (asset in shap_results and 
                        task in shap_results[asset] and 
                        model_type in shap_results[asset][task]):
                        
                        model_result = shap_results[asset][task][model_type]
                        
                        if 'feature_importance' in model_result:
                            importance = model_result['feature_importance']
                            
                            # Calculate statistics
                            model_patterns[asset] = {
                                'mean_importance': np.mean(list(importance.values())),
                                'std_importance': np.std(list(importance.values())),
                                'top_feature': max(importance.items(), key=lambda x: x[1])[0],
                                'top_importance': max(importance.values())
                            }
                
                patterns[model_type] = model_patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Performance pattern comparison failed: {str(e)}")
            return {}
    
    def _create_summary_statistics(self,
                                 shap_results: Dict[str, Any],
                                 assets: List[str],
                                 model_types: List[str],
                                 task: str) -> Dict[str, Any]:
        """Create summary statistics for all SHAP results."""
        try:
            summary = {
                'total_assets': len(assets),
                'total_models': len(model_types),
                'task': task,
                'analysis_timestamp': datetime.now().isoformat(),
                'asset_coverage': {},
                'model_coverage': {}
            }
            
            # Calculate coverage statistics
            for asset in assets:
                summary['asset_coverage'][asset] = {
                    'has_data': asset in shap_results,
                    'has_task': (asset in shap_results and 
                               task in shap_results[asset]),
                    'model_count': len(shap_results.get(asset, {}).get(task, {}))
                }
            
            for model_type in model_types:
                model_count = 0
                for asset in assets:
                    if (asset in shap_results and 
                        task in shap_results[asset] and 
                        model_type in shap_results[asset][task]):
                        model_count += 1
                
                summary['model_coverage'][model_type] = model_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary statistics creation failed: {str(e)}")
            return {'error': str(e)} 