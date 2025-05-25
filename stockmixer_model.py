"""
StockMixer: A simple yet strong MLP-based architecture for stock price forecasting
Based on Fan & Shen (2024) - Adapted for volatility forecasting

This module implements the StockMixer architecture as described in the thesis,
with three parallel pathways for temporal, indicator, and cross-stock mixing.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from typing import Tuple, Optional, Dict


class TemporalMixing(layers.Layer):
    """Temporal mixing pathway - captures time-based patterns"""
    
    def __init__(self, units: int = 64, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # First MLP block
        self.dense1 = layers.Dense(
            self.units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='temporal_dense1'
        )
        self.batch_norm1 = layers.BatchNormalization(name='temporal_bn1')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='temporal_dropout1')
        
        # Second MLP block
        self.dense2 = layers.Dense(
            self.units // 2,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='temporal_dense2'
        )
        self.batch_norm2 = layers.BatchNormalization(name='temporal_bn2')
        
        # Attention mechanism for temporal importance
        self.attention = layers.Dense(1, activation='sigmoid', name='temporal_attention')
        
    def call(self, inputs, training=None):
        # Process through MLP blocks
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        
        # Apply temporal attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        return x


class IndicatorMixing(layers.Layer):
    """Indicator mixing pathway - processes technical indicators"""
    
    def __init__(self, units: int = 64, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Feature extraction layers
        self.dense1 = layers.Dense(
            self.units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='indicator_dense1'
        )
        self.batch_norm1 = layers.BatchNormalization(name='indicator_bn1')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='indicator_dropout1')
        
        # Feature combination layer
        self.dense2 = layers.Dense(
            self.units // 2,
            activation='relu',
            name='indicator_dense2'
        )
        self.batch_norm2 = layers.BatchNormalization(name='indicator_bn2')
        
        # Feature importance weighting
        self.feature_weights = layers.Dense(
            self.units // 2,
            activation='softmax',
            name='indicator_weights'
        )
        
    def call(self, inputs, training=None):
        # Extract features
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        
        # Apply feature importance weighting
        weights = self.feature_weights(inputs)
        x = x * weights
        
        return x


class CrossStockMixing(layers.Layer):
    """Cross-stock mixing pathway - captures inter-asset relationships"""
    
    def __init__(self, units: int = 64, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Relationship extraction
        self.dense1 = layers.Dense(
            self.units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='cross_dense1'
        )
        self.batch_norm1 = layers.BatchNormalization(name='cross_bn1')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='cross_dropout1')
        
        # Relationship modeling
        self.dense2 = layers.Dense(
            self.units // 2,
            activation='relu',
            name='cross_dense2'
        )
        self.batch_norm2 = layers.BatchNormalization(name='cross_bn2')
        
        # Correlation attention
        self.correlation_attention = layers.Dense(
            self.units // 2,
            activation='tanh',
            name='correlation_attention'
        )
        
    def call(self, inputs, training=None):
        # Extract cross-asset relationships
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        
        # Apply correlation-based attention
        corr_weights = self.correlation_attention(x)
        x = x * tf.nn.sigmoid(corr_weights)
        
        return x


class StockMixer(Model):
    """
    StockMixer model for volatility forecasting
    
    Architecture:
    1. Three parallel pathways (temporal, indicator, cross-stock)
    2. Feature fusion layer
    3. Task-specific output heads
    """
    
    def __init__(
        self,
        temporal_units: int = 64,
        indicator_units: int = 64,
        cross_stock_units: int = 64,
        fusion_units: int = 128,
        dropout_rate: float = 0.2,
        task: str = 'regression',
        n_classes: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.task = task
        self.n_classes = n_classes
        
        # Three mixing pathways
        self.temporal_mixing = TemporalMixing(temporal_units, dropout_rate)
        self.indicator_mixing = IndicatorMixing(indicator_units, dropout_rate)
        self.cross_stock_mixing = CrossStockMixing(cross_stock_units, dropout_rate)
        
        # Feature fusion layers
        self.fusion_dense1 = layers.Dense(
            fusion_units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.005, l2=0.005),
            name='fusion_dense1'
        )
        self.fusion_bn1 = layers.BatchNormalization(name='fusion_bn1')
        self.fusion_dropout1 = layers.Dropout(dropout_rate, name='fusion_dropout1')
        
        self.fusion_dense2 = layers.Dense(
            fusion_units // 2,
            activation='relu',
            name='fusion_dense2'
        )
        self.fusion_bn2 = layers.BatchNormalization(name='fusion_bn2')
        self.fusion_dropout2 = layers.Dropout(dropout_rate, name='fusion_dropout2')
        
        # Task-specific output heads
        if task == 'regression':
            self.output_layer = layers.Dense(1, name='volatility_output')
        else:
            self.output_layer = layers.Dense(
                n_classes,
                activation='softmax',
                name='regime_output'
            )
        
    def call(self, inputs, training=None):
        # For LSTM-style sequence input, flatten appropriately
        if len(inputs.shape) == 3:
            # (batch, time_steps, features) -> (batch, time_steps * features)
            inputs = layers.Flatten()(inputs)
        
        # Process through three pathways
        temporal_features = self.temporal_mixing(inputs, training=training)
        indicator_features = self.indicator_mixing(inputs, training=training)
        cross_stock_features = self.cross_stock_mixing(inputs, training=training)
        
        # Concatenate pathway outputs
        combined = layers.concatenate([
            temporal_features,
            indicator_features,
            cross_stock_features
        ])
        
        # Feature fusion
        x = self.fusion_dense1(combined)
        x = self.fusion_bn1(x, training=training)
        x = self.fusion_dropout1(x, training=training)
        
        x = self.fusion_dense2(x)
        x = self.fusion_bn2(x, training=training)
        x = self.fusion_dropout2(x, training=training)
        
        # Task-specific output
        output = self.output_layer(x)
        
        return output
    
    def get_pathway_outputs(self, inputs):
        """Get intermediate outputs from each pathway for interpretability"""
        if len(inputs.shape) == 3:
            inputs = layers.Flatten()(inputs)
            
        temporal = self.temporal_mixing(inputs, training=False)
        indicator = self.indicator_mixing(inputs, training=False)
        cross_stock = self.cross_stock_mixing(inputs, training=False)
        
        return {
            'temporal': temporal,
            'indicator': indicator,
            'cross_stock': cross_stock
        }


def create_stockmixer(
    input_shape: Tuple[int, ...],
    task: str = 'regression',
    learning_rate: float = 0.001,
    **kwargs
) -> StockMixer:
    """
    Factory function to create and compile StockMixer model
    
    Args:
        input_shape: Shape of input data
        task: 'regression' or 'classification'
        learning_rate: Learning rate for Adam optimizer
        **kwargs: Additional arguments for StockMixer
    
    Returns:
        Compiled StockMixer model
    """
    model = StockMixer(task=task, **kwargs)
    
    # Build model by calling it with dummy input
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    
    if task == 'regression':
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


class StockMixerExplainer:
    """Helper class for StockMixer interpretability"""
    
    def __init__(self, model: StockMixer):
        self.model = model
        
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract feature importance from each pathway
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with importance scores for each pathway
        """
        pathway_outputs = self.model.get_pathway_outputs(X)
        
        # Calculate importance as mean activation strength
        importance = {}
        for pathway, output in pathway_outputs.items():
            importance[pathway] = np.mean(np.abs(output), axis=0)
            
        return importance
    
    def visualize_pathway_contributions(self, X: np.ndarray, y_true: np.ndarray = None):
        """Create visualization of pathway contributions"""
        import matplotlib.pyplot as plt
        
        # Get pathway outputs
        pathway_outputs = self.model.get_pathway_outputs(X)
        
        # Calculate mean absolute contributions
        contributions = {}
        for pathway, output in pathway_outputs.items():
            contributions[pathway] = np.mean(np.abs(output))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        pathways = list(contributions.keys())
        values = list(contributions.values())
        
        bars = ax.bar(pathways, values)
        ax.set_ylabel('Mean Absolute Contribution')
        ax.set_title('StockMixer Pathway Contributions')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    input_shape = (20, 13)  # 20 time steps, 13 features
    
    # Regression model
    reg_model = create_stockmixer(input_shape, task='regression')
    print("Regression model summary:")
    reg_model.summary()
    
    # Classification model
    clf_model = create_stockmixer(input_shape, task='classification')
    print("\nClassification model summary:")
    clf_model.summary()
    
    # Test forward pass
    test_input = np.random.randn(32, 20, 13)
    reg_output = reg_model(test_input)
    clf_output = clf_model(test_input)
    
    print(f"\nRegression output shape: {reg_output.shape}")
    print(f"Classification output shape: {clf_output.shape}")
