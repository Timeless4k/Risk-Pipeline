"""
Model factory for RiskPipeline modular architecture.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd

from .base_model import BaseModel
from .arima_model import ARIMAModel
from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel
from .stockmixer_model import StockMixerModel


class ModelFactory:
    """Factory class for creating model instances."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model factory.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger('risk_pipeline.models.ModelFactory')
        self.logger.info("ModelFactory initialized")
        
        # Available model types
        self.available_models = {
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'xgboost': XGBoostModel,
            'stockmixer': StockMixerModel
        }
    
    def create_model(self, model_type: str, task: str = 'regression', 
                    **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('arima', 'lstm', 'xgboost', 'stockmixer')
            task: Task type ('regression' or 'classification')
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model instance
        """
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(self.available_models.keys())}")
        
        # Get model class
        model_class = self.available_models[model_type]
        
        # Get model parameters from config
        model_params = self._get_model_params(model_type, task)
        
        # Update with provided kwargs
        model_params.update(kwargs)
        
        # Create model instance
        if model_type == 'arima':
            # ARIMA only supports regression
            if task != 'regression':
                self.logger.warning("ARIMA only supports regression tasks. Using regression.")
            model = model_class(**model_params)
        elif model_type == 'lstm':
            # LSTM supports both tasks
            model = model_class(**model_params)
        elif model_type == 'xgboost':
            # XGBoost supports both tasks
            model = model_class(task=task, **model_params)
        elif model_type == 'stockmixer':
            # StockMixer supports both tasks
            model = model_class(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.logger.info(f"Created {model_type} model for {task} task")
        return model
    
    def create_models(self, model_types: List[str], task: str = 'regression',
                     **kwargs) -> Dict[str, BaseModel]:
        """
        Create multiple model instances.
        
        Args:
            model_types: List of model types to create
            task: Task type ('regression' or 'classification')
            **kwargs: Additional parameters to pass to all models
            
        Returns:
            Dictionary mapping model types to model instances
        """
        models = {}
        
        for model_type in model_types:
            try:
                models[model_type] = self.create_model(model_type, task, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create {model_type} model: {e}")
                continue
        
        self.logger.info(f"Created {len(models)} models: {list(models.keys())}")
        return models
    
    def _get_model_params(self, model_type: str, task: str) -> Dict[str, Any]:
        """
        Get model parameters from configuration.
        
        Args:
            model_type: Type of model
            task: Task type
            
        Returns:
            Dictionary of model parameters
        """
        params = {}
        
        # Get model-specific config section
        model_config = self.config.get('models', {})
        
        if model_type == 'arima':
            params = {
                'order': model_config.get('arima_order', (1, 1, 1))
            }
        
        elif model_type == 'lstm':
            params = {
                'units': model_config.get('lstm_units', [50, 30]),
                'dropout': model_config.get('lstm_dropout', 0.2),
                'sequence_length': model_config.get('sequence_length', 15),
                'batch_size': model_config.get('batch_size', 16),
                'epochs': model_config.get('epochs', 100),
                'early_stopping_patience': model_config.get('early_stopping_patience', 5),
                'reduce_lr_patience': model_config.get('reduce_lr_patience', 5)
            }
        
        elif model_type == 'xgboost':
            params = {
                'n_estimators': model_config.get('xgboost_n_estimators', 100),
                'max_depth': model_config.get('xgboost_max_depth', 5),
                'learning_rate': model_config.get('xgboost_learning_rate', 0.1),
                'random_state': model_config.get('random_state', 42)
            }
        
        elif model_type == 'stockmixer':
            params = {
                'temporal_units': model_config.get('stockmixer_temporal_units', 64),
                'indicator_units': model_config.get('stockmixer_indicator_units', 64),
                'cross_stock_units': model_config.get('stockmixer_cross_stock_units', 64),
                'fusion_units': model_config.get('stockmixer_fusion_units', 128),
                'dropout': model_config.get('stockmixer_dropout', 0.2),
                'batch_size': model_config.get('batch_size', 16),
                'epochs': model_config.get('epochs', 100),
                'early_stopping_patience': model_config.get('early_stopping_patience', 5),
                'reduce_lr_patience': model_config.get('reduce_lr_patience', 5)
            }
        
        return params
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
        """
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.available_models[model_type]
        
        info = {
            'name': model_class.__name__,
            'module': model_class.__module__,
            'docstring': model_class.__doc__,
            'supports_regression': True,  # All models support regression
            'supports_classification': model_type != 'arima'  # ARIMA only supports regression
        }
        
        return info
    
    def validate_model_config(self, model_type: str, task: str) -> bool:
        """
        Validate model configuration.
        
        Args:
            model_type: Type of model
            task: Task type
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if model type is supported
            if model_type not in self.available_models:
                self.logger.error(f"Unsupported model type: {model_type}")
                return False
            
            # Check task compatibility
            if model_type == 'arima' and task != 'regression':
                self.logger.error("ARIMA only supports regression tasks")
                return False
            
            # Check if required parameters are available
            params = self._get_model_params(model_type, task)
            if not params:
                self.logger.warning(f"No parameters found for {model_type} model")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_ensemble(self, model_types: List[str], task: str = 'regression',
                       weights: Optional[List[float]] = None, **kwargs) -> Dict[str, BaseModel]:
        """
        Create an ensemble of models.
        
        Args:
            model_types: List of model types for ensemble
            task: Task type
            weights: Optional weights for ensemble (if None, equal weights)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing ensemble models and metadata
        """
        # Validate model types
        for model_type in model_types:
            if not self.validate_model_config(model_type, task):
                raise ValueError(f"Invalid configuration for {model_type}")
        
        # Create models
        models = self.create_models(model_types, task, **kwargs)
        
        # Set weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        ensemble = {
            'models': models,
            'weights': weights,
            'task': task,
            'model_types': model_types
        }
        
        self.logger.info(f"Created ensemble with {len(models)} models: {list(models.keys())}")
        return ensemble
    
    def get_model_summary(self) -> str:
        """Get summary of available models."""
        summary = "Available Models:\n"
        summary += "=" * 50 + "\n"
        
        for model_type in self.available_models:
            info = self.get_model_info(model_type)
            summary += f"\n{model_type.upper()}:\n"
            summary += f"  Class: {info['name']}\n"
            summary += f"  Regression: {info['supports_regression']}\n"
            summary += f"  Classification: {info['supports_classification']}\n"
            if info['docstring']:
                summary += f"  Description: {info['docstring'].strip()}\n"
        
        return summary
    
    def list_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        List parameters for a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of parameter names and default values
        """
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get default parameters for each task
        regression_params = self._get_model_params(model_type, 'regression')
        classification_params = self._get_model_params(model_type, 'classification')
        
        return {
            'regression': regression_params,
            'classification': classification_params,
            'common': {k: v for k, v in regression_params.items() 
                      if k in classification_params and regression_params[k] == classification_params[k]}
        } 