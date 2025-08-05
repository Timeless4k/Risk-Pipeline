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


class StockMixerModel(BaseModel):
    """StockMixer model with parallel pathways for temporal, indicator, and cross-stock mixing."""
    
    def __init__(self, temporal_units: int = 64, indicator_units: int = 64, 
                 cross_stock_units: int = 64, fusion_units: int = 128, 
                 dropout: float = 0.2, **kwargs):
        """
        Initialize StockMixer model.
        
        Args:
            temporal_units: Units for temporal pathway
            indicator_units: Units for indicator pathway
            cross_stock_units: Units for cross-stock pathway
            fusion_units: Units for fusion layer
            dropout: Dropout rate
            **kwargs: Additional parameters
        """
        super().__init__(name="StockMixer", **kwargs)
        self.temporal_units = temporal_units
        self.indicator_units = indicator_units
        self.cross_stock_units = cross_stock_units
        self.fusion_units = fusion_units
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.input_shape = None
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.epochs = kwargs.get('epochs', 100)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 5)
        self.reduce_lr_patience = kwargs.get('reduce_lr_patience', 5)
        
        self.logger.info(f"StockMixer model initialized with pathways: "
                        f"temporal={temporal_units}, indicator={indicator_units}, "
                        f"cross_stock={cross_stock_units}, fusion={fusion_units}")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train StockMixer model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and history
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        if len(X) < 20:
            raise ValueError("StockMixer requires at least 20 observations")
        
        self.logger.info(f"Training StockMixer model with {len(X)} observations")
        
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
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Set input shape
            self.input_shape = (X_scaled.shape[1],)
            
            # Create model
            self.model = self._create_model(n_classes)
            
            # Callbacks
            callbacks = self._create_callbacks()
            
            # Train model
            history = self.model.fit(
                X_scaled, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.model.predict(X_scaled, verbose=0)
            if self.task == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = y_pred.flatten()
            
            metrics = (self._calculate_classification_metrics(y, y_pred) 
                      if self.task == 'classification' 
                      else self._calculate_regression_metrics(y, y_pred))
            
            self.logger.info(f"StockMixer training completed. Final loss: {history.history['loss'][-1]:.4f}")
            
            return {
                'metrics': metrics,
                'history': history.history,
                'input_shape': self.input_shape,
                'task': self.task,
                'pathway_analysis': self._analyze_pathways(X_scaled)
            }
            
        except Exception as e:
            self.logger.error(f"StockMixer training failed: {e}")
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
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled, verbose=0)
            
            if self.task == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
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