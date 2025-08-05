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
        self.interpretation_utils = InterpretationUtils(config)
        
        # SHAP analysis results storage
        self._shap_values = {}
        self._explainers = {}
        self._background_data = {}
        
        logger.info("SHAPAnalyzer initialized")
    
    def analyze_all_models(self, 
                          features: Dict[str, Any],
                          results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform SHAP analysis for all trained models.
        
        Args:
            features: Dictionary of features for each asset
            results: Pipeline results containing trained models
            
        Returns:
            Dictionary containing all SHAP analysis results
        """
        logger.info("Starting comprehensive SHAP analysis")
        
        shap_results = {}
        
        for asset, asset_results in results.items():
            logger.info(f"Analyzing SHAP for asset: {asset}")
            
            if asset not in features:
                logger.warning(f"No features found for asset {asset}, skipping")
                continue
            
            asset_features = features[asset]
            asset_shap_results = {}
            
            # Analyze regression models
            if 'regression' in asset_results:
                regression_shap = self._analyze_task_models(
                    asset=asset,
                    task_results=asset_results['regression'],
                    features=asset_features,
                    task='regression'
                )
                asset_shap_results['regression'] = regression_shap
            
            # Analyze classification models
            if 'classification' in asset_results:
                classification_shap = self._analyze_task_models(
                    asset=asset,
                    task_results=asset_results['classification'],
                    features=asset_features,
                    task='classification'
                )
                asset_shap_results['classification'] = classification_shap
            
            shap_results[asset] = asset_shap_results
        
        # Store results
        self.results_manager.store_shap_results(shap_results)
        
        logger.info("SHAP analysis completed for all models")
        return shap_results
    
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
            # Create explainer
            explainer = self.explainer_factory.create_explainer(
                model=model,
                model_type=model_type,
                task=task,
                X=X
            )
            
            # Prepare background data
            background_data = self._prepare_background_data(X, model_type)
            
            # Calculate SHAP values
            if model_type in ['lstm', 'stockmixer']:
                # For deep learning models, use a subset for background
                background_subset = background_data[:self.config.shap.background_samples]
                shap_values = explainer.shap_values(background_subset)
                
                # For classification, SHAP returns a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # For tree-based models, use TreeExplainer
                shap_values = explainer.shap_values(X)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                shap_values=shap_values,
                feature_names=feature_names
            )
            
            # Generate plots
            plots = self._generate_shap_plots(
                explainer=explainer,
                shap_values=shap_values,
                X=X,
                feature_names=feature_names,
                asset=asset,
                model_type=model_type,
                task=task
            )
            
            # Store results
            result_key = f"{asset}_{model_type}_{task}"
            self._shap_values[result_key] = shap_values
            self._explainers[result_key] = explainer
            self._background_data[result_key] = background_data
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'explainer': explainer,
                'plots': plots,
                'model_type': model_type,
                'task': task,
                'asset': asset,
                'feature_names': feature_names
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
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            feature_importance[feature_name] = float(mean_shap[i])
        
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
                            task: str) -> Dict[str, str]:
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
            
        Returns:
            Dictionary of plot file paths
        """
        plots = {}
        
        try:
            # Create output directory
            output_dir = Path(self.config.output.shap_dir) / asset / model_type / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate different types of plots
            plot_types = ['bar', 'waterfall', 'beeswarm', 'heatmap']
            
            for plot_type in plot_types:
                if plot_type in self.config.shap.plot_type:
                    plot_path = self._create_shap_plot(
                        explainer=explainer,
                        shap_values=shap_values,
                        X=X,
                        feature_names=feature_names,
                        plot_type=plot_type,
                        output_dir=output_dir,
                        asset=asset,
                        model_type=model_type,
                        task=task
                    )
                    plots[plot_type] = str(plot_path)
            
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