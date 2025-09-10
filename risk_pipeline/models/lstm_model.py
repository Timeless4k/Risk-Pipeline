"""
LSTM model implementation for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
try:
    import tensorflow as tf  # Optional dependency
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - environment-specific
    tf = None  # type: ignore
    TF_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FIXED: Global TensorFlow device configuration to prevent automatic GPU usage
if TF_AVAILABLE:
    tf.config.set_soft_device_placement(False)
    try:
        # Hide all GPUs to avoid accidental GPU ops during evaluation/inference
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        cpus = tf.config.list_physical_devices('CPU')
        if cpus:
            tf.config.set_logical_device_configuration(
                cpus[0],
                [tf.config.LogicalDeviceConfiguration()]
            )
    except Exception:
        pass

# QUICK CPU OPTIMIZATION: Use all physical cores for maximum performance (if TF available)
import psutil
cpu_count = psutil.cpu_count(logical=False)
if TF_AVAILABLE:
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count)

from .base_model import BaseModel


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
        self.units = kwargs.get('units', [128, 64])
        self.dropout = kwargs.get('dropout', 0.1)
        self.sequence_length = kwargs.get('sequence_length', 15)
        self.input_shape = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Training parameters
        self.params = {
            'batch_size': kwargs.get('batch_size', 32),
            'epochs': kwargs.get('epochs', 200),
            'validation_split': kwargs.get('validation_split', 0.2),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 30),
            'reduce_lr_patience': kwargs.get('reduce_lr_patience', 10),
            'learning_rate': kwargs.get('learning_rate', 0.001)
        }
        
        self.logger.info(f"LSTM model initialized with units={self.units}, dropout={self.dropout}")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> 'LSTMModel':
        """Build the LSTM model architecture with GPU fallback."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow or disable LSTM model.")
        self.input_shape = input_shape
        
        # FIXED: Properly handle 2D input for tabular data
        if len(input_shape) == 2:
            # [N, F] - use as is for tabular data (no time dimension)
            self.input_shape = input_shape
            self.logger.info(f"Using 2D input shape for tabular data: {self.input_shape}")
        elif len(input_shape) == 3:
            # [N, T, F] - use as is for sequence data
            self.input_shape = input_shape
            self.logger.info(f"Using 3D input shape for sequence data: {self.input_shape}")
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # FIXED: Force CPU mode to avoid device conflicts
        device = '/CPU:0'
        self.logger.info(f"Using {device} for LSTM to avoid device conflicts")
        
        try:
            with tf.device(device):
                # FIXED: Build model with correct input shape
                if len(self.input_shape) == 2:
                    # Tabular data - use Dense layers
                    inputs = tf.keras.Input(shape=(self.input_shape[1],))  # Remove batch dimension
                    
                    # Dense layers for tabular data
                    x = tf.keras.layers.Dense(self.units[0], activation='relu')(inputs)
                    x = tf.keras.layers.Dropout(self.dropout)(x)
                    x = tf.keras.layers.Dense(self.units[1] if len(self.units) > 1 else max(16, self.units[0] // 2), activation='relu')(x)
                    x = tf.keras.layers.Dropout(self.dropout)(x)
                    x = tf.keras.layers.Dense(max(16, (self.units[1] if len(self.units) > 1 else self.units[0] // 2) // 2), activation='relu')(x)
                    x = tf.keras.layers.Dropout(self.dropout)(x)
                else:
                    # Sequence data - use LSTM layers
                    inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))  # Remove batch dimension
                    
                    # LSTM layers for sequence data
                    x = tf.keras.layers.LSTM(self.units[0], return_sequences=True)(inputs)
                    x = tf.keras.layers.Dropout(self.dropout)(x)
                    x = tf.keras.layers.LSTM(self.units[1] if len(self.units) > 1 else max(16, self.units[0] // 2))(x)
                    x = tf.keras.layers.Dropout(self.dropout)(x)
                
                # Output layer
                if self.task == 'classification':
                    # FIXED: Use 2 classes for binary classification (regime prediction)
                    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # 2 classes for binary classification
                else:
                    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
                
                self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # Compile model
                if self.task == 'classification':
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                        loss='mse',
                        metrics=['mae']
                    )
                    
        except Exception as build_error:
            self.logger.error(f"LSTM build failed: {build_error}")
            raise RuntimeError(f"Failed to build LSTM model: {build_error}")
        
        self.logger.info(f"LSTM model built successfully with input shape: {self.input_shape}")
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
            Training results dictionary
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow or disable LSTM model.")
        if not self.model:
            raise ValueError("Model must be built before training. Call build_model() first.")
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # FIXED: Ensure X has the right shape for the model
            if X.ndim == 2 and len(self.input_shape) == 2:
                # [N, F] - use as is for tabular data
                X_reshaped = X
                self.logger.info(f"Using 2D input shape for tabular data: {X_reshaped.shape}")
            elif X.ndim == 3 and len(self.input_shape) == 3:
                # [N, T, F] - use as is for sequence data
                X_reshaped = X
                self.logger.info(f"Using 3D input shape for sequence data: {X_reshaped.shape}")
            elif X.ndim == 2 and len(self.input_shape) == 3:
                # [N, F] but model expects [N, T, F] - reshape to single timestep
                X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                self.logger.info(f"Reshaping 2D input {X.shape} to 3D {X_reshaped.shape}")
            elif X.ndim == 3 and len(self.input_shape) == 2:
                # [N, T, F] but model expects [N, F] - flatten to tabular
                X_reshaped = X.reshape(X.shape[0], -1)
                self.logger.info(f"Flattening 3D input {X.shape} to 2D {X_reshaped.shape}")
            else:
                raise ValueError(f"Input shape {X.shape} incompatible with model input shape {self.input_shape}")
            
            # Ensure y has the right shape
            if y.ndim == 1:
                y_reshaped = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
            else:
                y_reshaped = y
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_reshaped, y_reshaped, test_size=0.2, random_state=42
            )
            # Fit scaler on train only, transform val
            try:
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
                self.logger.info("Applied StandardScaler (fit on train) to LSTM features")
            except Exception:
                pass
            
            # FIXED: Train on CPU to match model device
            with tf.device('/CPU:0'):
                # Training callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=self.params.get('early_stopping_patience', 30), restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=self.params.get('reduce_lr_patience', 10), min_lr=1e-6
                    )
                ]
                
                # Train the model
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.params.get('epochs', 200),
                    batch_size=self.params.get('batch_size', 32),
                    callbacks=callbacks,
                    verbose=0
                )
            
            # Store training history
            self.training_history = history.history
            
            # Mark as trained
            self.is_trained = True
            
            # Calculate metrics
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            if self.task == 'classification':
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                metrics = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
            else:
                train_mae = history.history['mae'][-1]
                val_mae = history.history['val_mae'][-1]
                metrics = {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
            
            self.logger.info(f"LSTM training completed successfully. Final val_loss: {val_loss:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            raise RuntimeError(f"LSTM training failed: {e}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow or disable LSTM model.")
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        try:
            # FIXED: Ensure X has the right shape for the model
            if X.ndim == 2 and len(self.input_shape) == 2:
                # [N, F] - use as is for tabular data
                X_reshaped = X
                self.logger.info(f"Using 2D input shape for tabular data: {X_reshaped.shape}")
            elif X.ndim == 3 and len(self.input_shape) == 3:
                # [N, T, F] - use as is for sequence data
                X_reshaped = X
                self.logger.info(f"Using 3D input shape for sequence data: {X_reshaped.shape}")
            elif X.ndim == 2 and len(self.input_shape) == 3:
                # [N, F] but model expects [N, T, F] - reshape to single timestep
                X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                self.logger.info(f"Reshaping 2D input {X.shape} to 3D {X_reshaped.shape}")
            elif X.ndim == 3 and len(self.input_shape) == 2:
                # [N, T, F] but model expects [N, F] - flatten to tabular
                X_reshaped = X.reshape(X.shape[0], -1)
                self.logger.info(f"Flattening 3D input {X.shape} to 2D {X_reshaped.shape}")
            else:
                raise ValueError(f"Input shape {X.shape} incompatible with model input shape {self.input_shape}")
            
            # Scale features using fitted scaler
            try:
                X_reshaped = self.scaler.transform(X_reshaped)
            except Exception:
                pass

            # FIXED: Force CPU device context during prediction to match training
            with tf.device('/CPU:0'):
                # Make predictions
                predictions = self.model.predict(X_reshaped, verbose=0)
            
            # Reshape predictions to match expected output
            if self.task == 'classification':
                # Return class predictions
                return np.argmax(predictions, axis=1)
            else:
                # Return regression predictions
                return predictions.flatten()
                
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            raise RuntimeError(f"LSTM prediction failed: {e}")
    
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
                'name': self.name,
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
            self.input_shape = model_data['input_shape']
            self.units = model_data['units']
            self.dropout = model_data['dropout']
            self.sequence_length = model_data['sequence_length']
            self.is_trained = True
            
            self.logger.info(f"LSTM model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            raise 