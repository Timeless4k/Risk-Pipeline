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


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting and classification."""
    
    def __init__(self, units: List[int] = [50, 30], dropout: float = 0.2, 
                 sequence_length: int = 15, **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            units: List of LSTM units for each layer
            dropout: Dropout rate
            sequence_length: Length of input sequences
            **kwargs: Additional parameters
        """
        super().__init__(name="LSTM", **kwargs)
        self.units = units
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.input_shape = None
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.epochs = kwargs.get('epochs', 100)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 5)
        self.reduce_lr_patience = kwargs.get('reduce_lr_patience', 5)
        
        self.logger.info(f"LSTM model initialized with units={units}, dropout={dropout}")
    
    # For unit tests compatibility
    def build_model(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        # Build minimal model
        model = Sequential()
        model.add(LSTM(self.units[0], return_sequences=len(self.units) > 1, input_shape=input_shape))
        if len(self.units) > 1:
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.units[1]))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.model = model
        return self
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and history
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        if len(X) < self.sequence_length + 10:
            raise ValueError(f"LSTM requires at least {self.sequence_length + 10} observations")
        
        self.logger.info(f"Training LSTM model with {len(X)} observations")
        
        try:
            # Determine task type
            unique_y = np.unique(y)
            if len(unique_y) <= 10:  # Classification task
                self.task = 'classification'
                n_classes = len(unique_y)
                self.logger.info(f"Classification task with {n_classes} classes")
            else:
                self.task = 'regression'
                n_classes = 1
                self.logger.info("Regression task")
            
            # Flatten 3D input (samples, time, features) to 2D for scaling
            if isinstance(X, np.ndarray) and X.ndim == 3:
                num_samples, time_steps, num_features = X.shape
                X_flat = X.reshape(num_samples, time_steps * num_features)
            else:
                X_flat = X
            # Scale features
            X_scaled = self.scaler.fit_transform(X_flat)
            
            # Create sequences
            y_1d = y.flatten() if isinstance(y, np.ndarray) and y.ndim > 1 else y
            X_seq, y_seq = self._create_sequences(X_scaled, y_1d)
            
            # Set input shape
            self.input_shape = (X_seq.shape[1], X_seq.shape[2])
            
            # Prefer GPU if available
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.logger.info(f"Using GPU for LSTM: {[d.name for d in gpus]}")
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    self.logger.info("No GPU detected; using CPU for LSTM")
            except Exception as _e:
                self.logger.debug(f"GPU config skipped: {_e}")

            # Create model
            self.model = self._create_model(n_classes)
            
            # Callbacks
            callbacks = self._create_callbacks()
            
            # Train model
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.model.predict(X_seq, verbose=0)
            if self.task == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = y_pred.flatten()
            
            metrics = (self._calculate_classification_metrics(y_seq, y_pred) 
                      if self.task == 'classification' 
                      else self._calculate_regression_metrics(y_seq, y_pred))
            
            self.logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
            
            return {
                'metrics': metrics,
                'history': history.history,
                'input_shape': self.input_shape,
                'task': self.task
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
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
            # Flatten if 3D
            if isinstance(X, np.ndarray) and X.ndim == 3:
                num_samples, time_steps, num_features = X.shape
                X_flat = X.reshape(num_samples, time_steps * num_features)
            else:
                X_flat = X
            # Scale features
            X_scaled = self.scaler.transform(X_flat)
            
            # Create sequences
            X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
            
            # Make predictions
            y_pred = self.model.predict(X_seq, verbose=0)
            
            if self.task == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
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
    
    def _create_callbacks(self) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.reduce_lr_patience,
            min_lr=0.00001,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Input features
            y: Target values
            stride: Stride for sequence creation
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(0, len(X) - self.sequence_length, stride):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)
    
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