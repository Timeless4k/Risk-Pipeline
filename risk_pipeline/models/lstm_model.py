"""
LSTM model implementation for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from ..utils.tensorflow_utils import (
    configure_tensorflow_memory, 
    get_optimal_device, 
    safe_tensorflow_operation,
    cleanup_tensorflow_memory
)


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting and classification."""
    
    def __init__(self, task: str = 'regression', **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            task: 'regression' or 'classification'
            **kwargs: Additional parameters
        """
        super().__init__(name="LSTM", **kwargs)
        self.task = task
        self.units = kwargs.get('units', [50, 30])
        self.dropout = kwargs.get('dropout', 0.2)
        self.sequence_length = kwargs.get('sequence_length', 15)
        self.scaler = StandardScaler()
        self.input_shape = None
        self.model = None
        
        # Training parameters
        self.params = {
            'batch_size': kwargs.get('batch_size', 16),
            'epochs': kwargs.get('epochs', 100),
            'validation_split': kwargs.get('validation_split', 0.2),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 5),
            'reduce_lr_patience': kwargs.get('reduce_lr_patience', 5),
            'learning_rate': kwargs.get('learning_rate', 0.001)
        }
        
        self.logger.info(f"LSTM model initialized with units={self.units}, dropout={self.dropout}")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> 'LSTMModel':
        """Build the LSTM model architecture with GPU fallback."""
        self.input_shape = input_shape
        
        # Handle different input shapes
        if len(input_shape) == 2:
            # [N, F] - flatten to [N, 1, F]
            self.input_shape = (1, input_shape[1])
        elif len(input_shape) == 3:
            # [N, T, F] - use as is
            self.input_shape = input_shape
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        try:
            # Try to configure TensorFlow with GPU fallback
            from ..utils.tensorflow_utils import (
                configure_tensorflow_memory, 
                get_optimal_device, 
                safe_tensorflow_operation,
                cleanup_tensorflow_memory
            )
            
            # Configure TensorFlow with GPU fallback
            device = configure_tensorflow_memory(gpu_memory_growth=True, force_cpu=False)
            
            if device == '/CPU:0':
                self.logger.info("Using CPU for LSTM model building")
            else:
                self.logger.info("Using GPU for LSTM model building")
            
            # Create the model with safe operation
            def _create_model():
                return self._create_model(n_classes=1 if self.task == 'regression' else 2)
            
            self.model = safe_tensorflow_operation(
                _create_model,
                fallback_device='/CPU:0',
                max_retries=1
            )
            
            self.logger.info(f"LSTM model built successfully with input shape: {self.input_shape} on {device}")
            return self
            
        except Exception as e:
            self.logger.error(f"LSTM model building failed: {e}")
            
            # Force CPU mode and retry
            try:
                from ..utils.tensorflow_utils import force_cpu_mode
                force_cpu_mode()
                
                # Clean up any GPU memory
                cleanup_tensorflow_memory()
                
                # Retry on CPU
                self.model = self._create_model(n_classes=1 if self.task == 'regression' else 2)
                self.logger.info("LSTM model built successfully on CPU after GPU failure")
                return self
                
            except Exception as cpu_error:
                self.logger.error(f"LSTM model building failed on both GPU and CPU: {cpu_error}")
                raise RuntimeError(f"Failed to build LSTM model: {cpu_error}")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        if not self.model:
            raise ValueError("Model must be built before training. Call build_model() first.")
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # Configure TensorFlow memory
            configure_tensorflow_memory()
            
            # Ensure X has the right shape for LSTM
            if X.ndim == 2:
                # [N, F] -> [N, 1, F] for single timestep
                X = X.reshape(X.shape[0], 1, X.shape[1])
            elif X.ndim == 3:
                # [N, T, F] - already correct
                pass
            else:
                raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Create sequences if needed
            if self.sequence_length > 1:
                X_seq, y_seq = self._create_sequences(X_scaled, y)
            else:
                X_seq, y_seq = X_scaled, y
            
            self.logger.info(f"Training LSTM with {len(X_seq)} samples, shape: {X_seq.shape}")
            
            # Get optimal device and train
            device = get_optimal_device()
            self.logger.info(f"Using device: {device}")
            
            # Use safe operation with fallback
            def train_model():
                return self.model.fit(
                    X_seq, y_seq,
                    batch_size=self.params.get('batch_size', 16),
                    epochs=self.params.get('epochs', 100),
                    validation_split=0.2,
                    callbacks=self._get_callbacks(),
                    verbose=0
                )
            
            history = safe_tensorflow_operation(train_model)
            
            self.is_trained = True
            
            # Clean up memory
            cleanup_tensorflow_memory()
            
            return {
                'history': history.history,
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history.get('val_loss', [None])[-1]
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            cleanup_tensorflow_memory()
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        try:
            # Ensure X has the right shape for LSTM
            if X.ndim == 2:
                # [N, F] -> [N, 1, F] for single timestep
                X = X.reshape(X.shape[0], 1, X.shape[1])
            elif X.ndim == 3:
                # [N, T, F] - already correct
                pass
            else:
                raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Create sequences if needed
            if self.sequence_length > 1:
                X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
            else:
                X_seq = X_scaled
            
            # Make predictions
            y_pred = self.model.predict(X_seq, verbose=0)
            
            # Ensure output is 1D
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            return y_pred
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            return np.zeros(len(X))
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate LSTM model.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            if self.task == 'classification':
                metrics = self._calculate_classification_metrics(y, y_pred)
            else:
                metrics = self._calculate_regression_metrics(y, y_pred)
            
            self.logger.info(f"LSTM evaluation completed")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"LSTM evaluation failed: {e}")
            if self.task == 'classification':
                return {
                    'Accuracy': 0.0,
                    'F1': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0
                }
            else:
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf')
                }
    
    def _create_model(self, n_classes: int) -> tf.keras.Model:
        """Create LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(self.dropout))
        
        # Second LSTM layer (if specified)
        if len(self.units) > 1:
            model.add(LSTM(self.units[1], return_sequences=False))
            model.add(Dropout(self.dropout))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        
        # Output layer
        if self.task == 'classification':
            model.add(Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def _get_callbacks(self):
        """Get training callbacks."""
        callbacks = []
        
        # Early stopping
        if self.params.get('early_stopping_patience', 5) > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params.get('early_stopping_patience', 5),
                restore_best_weights=True
            ))
        
        # Learning rate reduction
        if self.params.get('reduce_lr_patience', 5) > 0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.params.get('reduce_lr_patience', 5),
                min_lr=1e-7
            ))
        
        return callbacks

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_trained:
            return "Model not trained"
        
        try:
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            return "\n".join(stringlist)
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return "Summary not available"
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting history")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss plot
            axes[0].plot(self.model.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.model.history.history:
                axes[0].plot(self.model.history.history['val_loss'], label='Validation Loss')
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            
            # Metric plot
            if self.task == 'classification':
                metric_name = 'accuracy'
                metric_label = 'Accuracy'
            else:
                metric_name = 'mae'
                metric_label = 'MAE'
            
            if metric_name in self.model.history.history:
                axes[1].plot(self.model.history.history[metric_name], label=f'Training {metric_label}')
                if f'val_{metric_name}' in self.model.history.history:
                    axes[1].plot(self.model.history.history[f'val_{metric_name}'], 
                               label=f'Validation {metric_label}')
                axes[1].set_title(f'Model {metric_label}')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel(metric_label)
                axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training history plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot training history: {e}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model with additional LSTM-specific data."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'task': self.task,
                'name': self.name,
                'scaler': self.scaler,
                'input_shape': self.input_shape,
                'units': self.units,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"LSTM model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save LSTM model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained LSTM model."""
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.task = model_data['task']
            self.name = model_data['name']
            self.scaler = model_data['scaler']
            self.input_shape = model_data['input_shape']
            self.units = model_data['units']
            self.dropout = model_data['dropout']
            self.sequence_length = model_data['sequence_length']
            self.is_trained = True
            
            self.logger.info(f"LSTM model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            raise 