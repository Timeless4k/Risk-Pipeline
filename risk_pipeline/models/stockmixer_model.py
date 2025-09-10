"""
StockMixer model implementation (AAAI-24) for RiskPipeline.
Implements three-stage mixing: Indicator, Time (causal multi-scale), Stock (market-aware).
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None  # type: ignore
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split

import psutil

# TensorFlow device/threading configuration (CPU-only, deterministic)
if TF_AVAILABLE:
    tf.config.set_soft_device_placement(False)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        cpus = tf.config.list_physical_devices('CPU')
        if cpus:
            tf.config.set_logical_device_configuration(
                cpus[0], [tf.config.LogicalDeviceConfiguration()]
            )
    except Exception:
        pass
    threads = psutil.cpu_count(logical=False) or 1
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

from .base_model import BaseModel


class IndicatorMixingBlock(tf.keras.layers.Layer):
    def __init__(self, indicator_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.indicator_dim = indicator_dim
        self.hidden_dim = hidden_dim or indicator_dim
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(indicator_dim)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        mixed = self.dense1(x)
        mixed = self.dropout(mixed, training=training)
        mixed = self.dense2(mixed)
        return inputs + mixed


class TimeMixingBlock(tf.keras.layers.Layer):
    def __init__(self, time_dim: int, patch_sizes: List[int] = [1, 2, 4], hidden_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.time_dim = time_dim
        self.patch_sizes = patch_sizes
        self.hidden_dim = hidden_dim or time_dim
        self.patch_processors = {}
        for patch_size in patch_sizes:
            compressed_dim = max(1, time_dim // patch_size)
            self.patch_processors[patch_size] = {
                'layer_norm': tf.keras.layers.LayerNormalization(),
                'dense1': tf.keras.layers.Dense(compressed_dim, activation='relu'),
                'dropout': tf.keras.layers.Dropout(dropout),
                'dense2': tf.keras.layers.Dense(compressed_dim)
            }
        self.fusion_dense = tf.keras.layers.Dense(time_dim)

    def _create_causal_mask(self, seq_len: Union[int, tf.Tensor]):
        if isinstance(seq_len, int):
            mask = np.triu(np.ones((seq_len, seq_len)), k=0).astype(np.float32)
            return tf.constant(mask)
        seq_len_int = tf.cast(seq_len, tf.int32)
        row = tf.range(seq_len_int)[:, None]
        col = tf.range(seq_len_int)[None, :]
        mask = tf.cast(col >= row, tf.float32)
        return mask

    def _apply_causal_mixing(self, x, mask):
        # x: [B, T, F], mask: [T, T]
        mixed = tf.linalg.matmul(mask, x)
        return mixed

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        patch_outputs = []
        for patch_size in self.patch_sizes:
            if patch_size > 1:
                pad_len = (patch_size - seq_len % patch_size) % patch_size
                padded = tf.pad(inputs, [[0, 0], [0, pad_len], [0, 0]])
                new_seq_len = tf.shape(padded)[1]
                patches = tf.reshape(padded, [batch_size, new_seq_len // patch_size, patch_size, -1])
                patch_input = tf.reduce_mean(patches, axis=2)
            else:
                patch_input = inputs

            patch_len = tf.shape(patch_input)[1]
            scale_mask = self._create_causal_mask(patch_len)
            mixed_patch = self._apply_causal_mixing(patch_input, scale_mask)

            processor = self.patch_processors[patch_size]
            x = processor['layer_norm'](mixed_patch)
            x = processor['dense1'](x)
            x = processor['dropout'](x, training=training)
            x = processor['dense2'](x)
            processed_patch = mixed_patch + x

            if patch_size > 1:
                upsampled = tf.repeat(processed_patch, patch_size, axis=1)
                upsampled = upsampled[:, :seq_len, :]
                patch_outputs.append(upsampled)
            else:
                patch_outputs.append(processed_patch)

        concatenated = tf.concat(patch_outputs, axis=-1)
        fused = self.fusion_dense(concatenated)
        return fused


class StockMixingBlock(tf.keras.layers.Layer):
    def __init__(self, n_stocks: int, market_dim: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.stock_to_market = tf.keras.layers.Dense(market_dim, activation='relu')
        self.market_dropout = tf.keras.layers.Dropout(dropout)
        self.market_to_stock = tf.keras.layers.Dense(n_stocks)

    def call(self, inputs, training=None):
        # inputs: [B, S, F]
        x = self.layer_norm(inputs)
        market_states = tf.reduce_mean(x, axis=1)
        market_states = self.stock_to_market(market_states)
        market_states = self.market_dropout(market_states, training=training)
        market_influence = self.market_to_stock(market_states)
        market_influence = tf.expand_dims(market_influence, axis=-1)
        feature_dim = tf.shape(inputs)[-1]
        market_influence = tf.tile(market_influence, [1, 1, feature_dim])
        return inputs + market_influence


class StockMixerNet(tf.keras.Model):
    def __init__(self, n_stocks: int, n_indicators: int, sequence_length: int,
                 indicator_hidden_dim: int = 64, time_patch_sizes: List[int] = [1, 2, 4],
                 market_dim: int = 32, dropout: float = 0.1, num_classes: int = 1, task: str = 'regression', **kwargs):
        super().__init__(**kwargs)
        self.n_stocks = n_stocks
        self.n_indicators = n_indicators
        self.sequence_length = sequence_length
        self.task = task
        self.num_classes = num_classes

        self.indicator_mixing = IndicatorMixingBlock(indicator_dim=n_indicators, hidden_dim=indicator_hidden_dim, dropout=dropout)
        self.time_mixing = TimeMixingBlock(time_dim=sequence_length, patch_sizes=time_patch_sizes, dropout=dropout)
        self.stock_mixing = StockMixingBlock(n_stocks=n_stocks, market_dim=market_dim, dropout=dropout)
        if task == 'regression':
            self.prediction_head = tf.keras.layers.Dense(1, activation='linear')
        else:
            self.prediction_head = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        # inputs: [B, S, T, I]
        batch_size = tf.shape(inputs)[0]
        reshaped_for_ind = tf.reshape(inputs, [-1, self.n_indicators])
        mixed_ind = self.indicator_mixing(reshaped_for_ind, training=training)
        mixed_ind = tf.reshape(mixed_ind, [batch_size, self.n_stocks, self.sequence_length, self.n_indicators])

        x = tf.transpose(mixed_ind, [0, 1, 3, 2])
        x = tf.reshape(x, [-1, self.sequence_length, 1])
        mixed_time = self.time_mixing(x, training=training)
        mixed_time = tf.reshape(mixed_time, [batch_size, self.n_stocks, self.n_indicators, self.sequence_length])
        mixed_time = tf.transpose(mixed_time, [0, 1, 2, 3])
        mixed_time = tf.transpose(mixed_time, [0, 1, 3, 2])

        stock_features = tf.reshape(mixed_time, [batch_size, self.n_stocks, -1])
        mixed_stocks = self.stock_mixing(stock_features, training=training)
        pooled = tf.reduce_mean(mixed_stocks, axis=1)
        preds = self.prediction_head(pooled)
        return preds


class StockMixerModel(BaseModel):
    """Wrapper integrating StockMixerNet with RiskPipeline BaseModel."""

    def __init__(self, task: str = 'regression', **kwargs):
        super().__init__(name="StockMixer", **kwargs)
        self.task = task
        # architecture params
        self.n_stocks = int(kwargs.get('n_stocks', 1))
        self.n_indicators = int(kwargs.get('n_indicators', 4))
        self.sequence_length = int(kwargs.get('sequence_length', 30))
        self.num_classes = int(kwargs.get('num_classes', 2 if task == 'classification' else 1))
        self.indicator_hidden_dim = int(kwargs.get('indicator_hidden_dim', 64))
        self.time_patch_sizes = list(kwargs.get('time_patch_sizes', [1, 2, 4]))
        self.market_dim = int(kwargs.get('market_dim', 32))
        self.dropout = float(kwargs.get('dropout', 0.1))
        # training params
        self.params.update({
            'batch_size': int(kwargs.get('batch_size', 32)),
            'epochs': int(kwargs.get('epochs', 100)),
            'learning_rate': float(kwargs.get('learning_rate', 1e-3)),
            'validation_split': float(kwargs.get('validation_split', 0.2)),
        })
        self.model: Optional[tf.keras.Model] = None if TF_AVAILABLE else None

    def _ensure_4d(self, X: np.ndarray) -> np.ndarray:
        # Accept [N,T,F] or [N,S,T,F]; convert to [N,S,T,F]
        if X.ndim == 3:
            return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        if X.ndim == 4:
            return X
        raise ValueError(f"Unsupported input shape {X.shape}; expected [N,T,F] or [N,S,T,F]")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'StockMixerModel':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow to use StockMixer.")

        # Infer dims from input_shape
        if len(input_shape) == 3:
            # [N,T,F]
            self.n_stocks = 1
            self.sequence_length = int(input_shape[1])
            self.n_indicators = int(input_shape[2])
        elif len(input_shape) == 4:
            # [N,S,T,F]
            self.n_stocks = int(input_shape[1])
            self.sequence_length = int(input_shape[2])
            self.n_indicators = int(input_shape[3])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        with tf.device('/CPU:0'):
            net = StockMixerNet(
                n_stocks=self.n_stocks,
                n_indicators=self.n_indicators,
                sequence_length=self.sequence_length,
                indicator_hidden_dim=self.indicator_hidden_dim,
                time_patch_sizes=self.time_patch_sizes,
                market_dim=self.market_dim,
                dropout=self.dropout,
                num_classes=self.num_classes,
                task=self.task,
            )
            # Build Keras functional wrapper for compile/fit
            inputs = tf.keras.Input(shape=(self.n_stocks, self.sequence_length, self.n_indicators))
            outputs = net(inputs)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='StockMixer')

            if self.task == 'regression':
                loss = 'mse'
                metrics = ['mae']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'], clipnorm=1.0),
                loss=loss,
                metrics=metrics,
            )

        return self

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow to use StockMixer.")
        if self.model is None:
            # Build from observed data shape
            X_arr, y_arr = self._validate_input(X, y)
            self.build_model(X_arr.shape)
        else:
            X_arr, y_arr = self._validate_input(X, y)

        X4 = self._ensure_4d(X_arr)

        if np.any(np.isnan(X4)) or np.any(np.isinf(X4)):
            raise ValueError("Features contain NaN or infinite values")
        if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
            raise ValueError("Targets contain NaN or infinite values")

        # Train/val split
        idx = np.arange(X4.shape[0])
        train_idx, val_idx = train_test_split(idx, test_size=self.params['validation_split'], random_state=42)
        X_train, X_val = X4[train_idx], X4[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        with tf.device('/CPU:0'):
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=int(kwargs.get('epochs', self.params['epochs'])),
                batch_size=int(kwargs.get('batch_size', self.params['batch_size'])),
                verbose=0,
            )

        self.training_history = history.history
        self.is_trained = True

        result: Dict[str, Any] = {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
        }
        if self.task == 'classification':
            result.update({
                'train_accuracy': history.history.get('accuracy', [np.nan])[-1],
                'val_accuracy': history.history.get('val_accuracy', [np.nan])[-1],
            })
        else:
            result.update({
                'train_mae': history.history.get('mae', [np.nan])[-1],
                'val_mae': history.history.get('val_mae', [np.nan])[-1],
            })
        return result

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow to use StockMixer.")
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_arr, _ = self._validate_input(X)
        X4 = self._ensure_4d(X_arr)
        with tf.device('/CPU:0'):
            preds = self.model.predict(X4, verbose=0)
        if self.task == 'classification':
            return np.argmax(preds, axis=1)
        return preds.flatten()

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        X_arr, y_arr = self._validate_input(X, y)
        y_pred = self.predict(X_arr)
        if self.task == 'classification':
            return self._calculate_classification_metrics(np.asarray(y_arr).astype(int), np.asarray(y_pred).astype(int))
        return self._calculate_regression_metrics(np.asarray(y_arr), np.asarray(y_pred))
