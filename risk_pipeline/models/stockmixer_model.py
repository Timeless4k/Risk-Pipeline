"""
StockMixer model implementation for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Model
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


class StockMixerModel(BaseModel):
    """StockMixer model with parallel pathways for temporal, indicator, and cross-stock mixing."""
    
    def __init__(self, task: str = 'regression', **kwargs):
        """
        Initialize StockMixer model.
        
        Args:
            task: 'regression' or 'classification'
            **kwargs: Additional parameters
        """
        super().__init__(name="StockMixer", **kwargs)
        self.task = task
        self.temporal_units = kwargs.get('temporal_units', 64)
        self.indicator_units = kwargs.get('indicator_units', 64)
        self.cross_stock_units = kwargs.get('cross_stock_units', 64)
        self.fusion_units = kwargs.get('fusion_units', 128)
        self.dropout = kwargs.get('dropout', 0.2)
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
        
        self.logger.info(f"StockMixer model initialized with pathways: "
                        f"temporal={self.temporal_units}, indicator={self.indicator_units}, "
                        f"cross_stock={self.cross_stock_units}, fusion={self.fusion_units}")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> 'StockMixerModel':
        """Build the StockMixer model architecture."""
        self.input_shape = input_shape
        
        # Handle different input shapes
        if len(input_shape) == 2:
            # [N, F] - use as is for tabular data
            self.input_shape = input_shape
        elif len(input_shape) == 3:
            # [N, T, F] - flatten to [N, T*F] for tabular processing
            self.input_shape = (input_shape[0], input_shape[1] * input_shape[2])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # Create the model
        self.model = self._create_model(n_classes=1 if self.task == 'regression' else 2)
        self.logger.info(f"StockMixer model built with input shape: {self.input_shape}")
        return self
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train StockMixer model.
        
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
            
            # Ensure X has the right shape for StockMixer
            if X.ndim == 2:
                # [N, F] - already correct for tabular data
                X_flat = X
            elif X.ndim == 3:
                # [N, T, F] - flatten to [N, T*F] for tabular processing
                num_samples, time_steps, num_features = X.shape
                X_flat = X.reshape(num_samples, time_steps * num_features)
            else:
                raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_flat)
            
            self.logger.info(f"Training StockMixer with {len(X_scaled)} samples, shape: {X_scaled.shape}")
            
            # Get optimal device and train
            device = get_optimal_device()
            self.logger.info(f"Using device: {device}")
            
            # Use safe operation with fallback
            def train_model():
                return self.model.fit(
                    X_scaled, y,
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
            self.logger.error(f"StockMixer training failed: {e}")
            cleanup_tensorflow_memory()
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with StockMixer model.
        
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
            # Ensure X has the right shape for StockMixer
            if X.ndim == 2:
                # [N, F] - already correct for tabular data
                X_flat = X
            elif X.ndim == 3:
                # [N, T, F] - flatten to [N, T*F] for tabular processing
                num_samples, time_steps, num_features = X.shape
                X_flat = X.reshape(num_samples, time_steps * num_features)
            else:
                raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
            
            # Scale features
            X_scaled = self.scaler.transform(X_flat)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled, verbose=0)
            
            # Ensure output is 1D
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            return y_pred
            
        except Exception as e:
            self.logger.error(f"StockMixer prediction failed: {e}")
            return np.zeros(len(X))
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate StockMixer model.
        
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
            
            self.logger.info(f"StockMixer evaluation completed")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"StockMixer evaluation failed: {e}")
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
        """Create StockMixer model architecture with three parallel pathways."""
        inputs = Input(shape=self.input_shape)
        
        # Temporal pathway - focuses on time-based patterns
        temporal = Dense(self.temporal_units, activation='relu', name='temporal_dense1')(inputs)
        temporal = BatchNormalization(name='temporal_bn1')(temporal)
        temporal = Dense(self.temporal_units // 2, activation='relu', name='temporal_dense2')(temporal)
        temporal = BatchNormalization(name='temporal_bn2')(temporal)
        
        # Indicator pathway - focuses on technical indicators
        indicator = Dense(self.indicator_units, activation='relu', name='indicator_dense1')(inputs)
        indicator = BatchNormalization(name='indicator_bn1')(indicator)
        indicator = Dense(self.indicator_units // 2, activation='relu', name='indicator_dense2')(indicator)
        indicator = BatchNormalization(name='indicator_bn2')(indicator)
        
        # Cross-stock pathway - focuses on cross-asset relationships
        cross_stock = Dense(self.cross_stock_units, activation='relu', name='cross_stock_dense1')(inputs)
        cross_stock = BatchNormalization(name='cross_stock_bn1')(cross_stock)
        cross_stock = Dense(self.cross_stock_units // 2, activation='relu', name='cross_stock_dense2')(cross_stock)
        cross_stock = BatchNormalization(name='cross_stock_bn2')(cross_stock)
        
        # Concatenate pathways
        merged = Concatenate(name='pathway_fusion')([temporal, indicator, cross_stock])
        
        # Fusion layers
        mixed = Dense(self.fusion_units, activation='relu', name='fusion_dense1')(merged)
        mixed = BatchNormalization(name='fusion_bn1')(mixed)
        mixed = Dropout(self.dropout, name='fusion_dropout1')(mixed)
        
        mixed = Dense(self.fusion_units // 2, activation='relu', name='fusion_dense2')(mixed)
        mixed = BatchNormalization(name='fusion_bn2')(mixed)
        mixed = Dropout(self.dropout, name='fusion_dropout2')(mixed)
        
        # Output layer
        if self.task == 'classification':
            output = Dense(n_classes, activation='softmax', name='output')(mixed)
            model = Model(inputs=inputs, outputs=output)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            output = Dense(1, name='output')(mixed)
            model = Model(inputs=inputs, outputs=output)
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
        """Create sequences for StockMixer training."""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def _analyze_pathways(self, X: np.ndarray) -> Dict[str, float]:
        """Analyze the contribution of each pathway."""
        if not self.is_trained:
            return {}
        
        try:
            # Create intermediate models for each pathway
            pathway_models = {}
            
            # Temporal pathway model
            temporal_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('temporal_bn2').output
            )
            pathway_models['temporal'] = temporal_model
            
            # Indicator pathway model
            indicator_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('indicator_bn2').output
            )
            pathway_models['indicator'] = indicator_model
            
            # Cross-stock pathway model
            cross_stock_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('cross_stock_bn2').output
            )
            pathway_models['cross_stock'] = cross_stock_model
            
            # Calculate pathway activations
            pathway_analysis = {}
            for pathway_name, pathway_model in pathway_models.items():
                activations = pathway_model.predict(X, verbose=0)
                pathway_analysis[f'{pathway_name}_mean_activation'] = float(np.mean(activations))
                pathway_analysis[f'{pathway_name}_std_activation'] = float(np.std(activations))
            
            return pathway_analysis
            
        except Exception as e:
            self.logger.warning(f"Pathway analysis failed: {e}")
            return {}
    
    def get_pathway_weights(self) -> Dict[str, np.ndarray]:
        """Get the weights of each pathway."""
        if not self.is_trained:
            return {}
        
        try:
            pathway_weights = {}
            
            # Get weights for each pathway
            pathway_layers = {
                'temporal': ['temporal_dense1', 'temporal_dense2'],
                'indicator': ['indicator_dense1', 'indicator_dense2'],
                'cross_stock': ['cross_stock_dense1', 'cross_stock_dense2']
            }
            
            for pathway_name, layer_names in pathway_layers.items():
                pathway_weights[pathway_name] = {}
                for layer_name in layer_names:
                    layer = self.model.get_layer(layer_name)
                    pathway_weights[pathway_name][layer_name] = layer.get_weights()
            
            return pathway_weights
            
        except Exception as e:
            self.logger.error(f"Failed to get pathway weights: {e}")
            return {}
    
    def plot_pathway_analysis(self, X: Union[pd.DataFrame, np.ndarray], 
                            save_path: Optional[str] = None) -> None:
        """Plot pathway analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before pathway analysis")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        try:
            import matplotlib.pyplot as plt
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get pathway activations
            pathway_models = {}
            pathway_names = ['temporal', 'indicator', 'cross_stock']
            
            for pathway_name in pathway_names:
                pathway_model = Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(f'{pathway_name}_bn2').output
                )
                pathway_models[pathway_name] = pathway_model
            
            # Calculate activations
            activations = {}
            for pathway_name, pathway_model in pathway_models.items():
                activations[pathway_name] = pathway_model.predict(X_scaled, verbose=0)
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Pathway activation distributions
            for i, pathway_name in enumerate(pathway_names):
                ax = axes[i // 2, i % 2]
                ax.hist(activations[pathway_name].flatten(), bins=30, alpha=0.7)
                ax.set_title(f'{pathway_name.title()} Pathway Activations')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency')
            
            # Pathway comparison
            pathway_means = [np.mean(activations[name]) for name in pathway_names]
            axes[1, 1].bar(pathway_names, pathway_means)
            axes[1, 1].set_title('Mean Pathway Activations')
            axes[1, 1].set_ylabel('Mean Activation')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Pathway analysis plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot pathway analysis: {e}")
    
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
        """Save the trained model with additional StockMixer-specific data."""
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
                'temporal_units': self.temporal_units,
                'indicator_units': self.indicator_units,
                'cross_stock_units': self.cross_stock_units,
                'fusion_units': self.fusion_units,
                'dropout': self.dropout
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"StockMixer model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save StockMixer model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained StockMixer model."""
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
            self.temporal_units = model_data['temporal_units']
            self.indicator_units = model_data['indicator_units']
            self.cross_stock_units = model_data['cross_stock_units']
            self.fusion_units = model_data['fusion_units']
            self.dropout = model_data['dropout']
            self.is_trained = True
            
            self.logger.info(f"StockMixer model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load StockMixer model: {e}")
            raise 