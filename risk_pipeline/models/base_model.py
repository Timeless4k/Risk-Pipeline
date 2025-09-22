"""
Base model class for RiskPipeline modular architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class BaseModel(ABC):
    """Abstract base class for all models in RiskPipeline."""
    
    def __init__(self, name: str = None, task: str = 'regression', **kwargs):
        """
        Initialize the base model.
        
        Args:
            name: Model name for identification
            task: Task type ('regression' or 'classification')
            **kwargs: Additional model-specific parameters
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f'risk_pipeline.models.{self.name}')
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.task = task  # 'regression' or 'classification'
        
        # Store model parameters
        self.params = kwargs
        self.logger.info(f"Initialized {self.name} with parameters: {kwargs}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], config: Any = None) -> Dict[str, Any]:
        """
        Fit the model (alias for train to match adapter interface).
        
        Args:
            X: Training features
            y: Training targets
            config: Configuration object (optional)
            
        Returns:
            Dictionary containing training metrics and history
        """
        # Extract parameters from config if provided
        kwargs = {}
        if config:
            if hasattr(config, 'max_epochs'):
                kwargs['epochs'] = config.max_epochs
            if hasattr(config, 'batch_size'):
                kwargs['batch_size'] = config.batch_size
            if hasattr(config, 'patience'):
                kwargs['early_stopping_patience'] = config.patience
            if hasattr(config, 'lr'):
                kwargs['learning_rate'] = config.lr
        
        return self.train(X, y, **kwargs)

    @abstractmethod
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        self.params.update(params)
        self.logger.info(f"Updated parameters: {params}")
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray], 
                       y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and convert input data.
        
        Args:
            X: Input features
            y: Optional target values
            
        Returns:
            Tuple of (X_converted, y_converted)
        """
        # Convert DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(f"X must be pandas DataFrame or numpy array, got {type(X)}")
        
        # Convert Series to numpy array
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            elif isinstance(y, np.ndarray):
                pass
            else:
                raise ValueError(f"y must be pandas Series or numpy array, got {type(y)}")
        
        # Check for NaN values
        if np.isnan(X).any():
            self.logger.warning("NaN values detected in features")
        
        if y is not None and np.isnan(y).any():
            self.logger.warning("NaN values detected in targets")
        
        return X, y
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {e}")
            return {
                'MSE': float('inf'),
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf')
            }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            }
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {e}")
            return {
                'Accuracy': 0.0,
                'F1': 0.0,
                'Precision': 0.0,
                'Recall': 0.0
            }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'task': self.task,
                'name': self.name
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.task = model_data['task']
            self.name = model_data['name']
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_trained or self.feature_names is None:
            return None
        
        # This should be implemented by subclasses that support feature importance
        return None
    
    def __str__(self) -> str:
        return f"{self.name}(task={self.task}, trained={self.is_trained})"
    
    def __repr__(self) -> str:
        return self.__str__() 