"""
LSTM model implementation for RiskPipeline.

Enhanced architecture for financial time series with:
- Proper sequence handling (overlapping windows)
- Optional multi-scale processing
- Bidirectional LSTMs
- Attention mechanism
- Advanced regularization and training dynamics
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


if TF_AVAILABLE:
    class AttentionLayer(tf.keras.layers.Layer):
        """Simple additive attention over time dimension."""

        def __init__(self, hidden_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.hidden_dim = hidden_dim
            self.W = tf.keras.layers.Dense(hidden_dim, use_bias=False)
            self.V = tf.keras.layers.Dense(1, use_bias=False)

        def call(self, lstm_output):
            score = self.V(tf.nn.tanh(self.W(lstm_output)))
            weights = tf.nn.softmax(score, axis=1)
            attended = tf.reduce_sum(lstm_output * weights, axis=1)
            return attended, weights

    class MultiScaleLSTMBlock(tf.keras.layers.Layer):
        """Processes inputs at multiple temporal scales and fuses representations."""

        def __init__(self, units: int, scales: List[int], dropout: float, recurrent_dropout: float, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.scales = scales
            self.dropout = dropout
            self.recurrent_dropout = recurrent_dropout
            self.scale_lstms: Dict[int, tf.keras.layers.Layer] = {}
            self.scale_attn: Dict[int, AttentionLayer] = {}
            for s in scales:
                self.scale_lstms[s] = tf.keras.layers.LSTM(
                    max(4, units // max(1, len(scales))),
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    name=f"lstm_scale_{s}"
                )
                self.scale_attn[s] = AttentionLayer(max(4, units // max(1, len(scales))), name=f"attn_scale_{s}")
            self.fusion = tf.keras.layers.Dense(units, activation='tanh')
            self.layer_norm = tf.keras.layers.LayerNormalization()

        def _pool(self, x, scale: int):
            if scale == 1:
                return x
            b = tf.shape(x)[0]
            t = tf.shape(x)[1]
            f = tf.shape(x)[2]
            pad = (scale - t % scale) % scale
            x = tf.pad(x, [[0, 0], [0, pad], [0, 0]]) if pad > 0 else x
            new_t = tf.shape(x)[1]
            x = tf.reshape(x, [b, new_t // scale, scale, f])
            x = tf.reduce_mean(x, axis=2)
            return x

        def call(self, inputs, training=None):
            outputs = []
            for s in self.scales:
                xi = self._pool(inputs, s)
                xo = self.scale_lstms[s](xi, training=training)
                att, _ = self.scale_attn[s](xo)
                outputs.append(att)
            fused = self.fusion(tf.concat(outputs, axis=-1))
            return self.layer_norm(fused)


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
        self.dropout = kwargs.get('dropout', 0.2)
        self.recurrent_dropout = kwargs.get('recurrent_dropout', 0.1)
        self.sequence_length = kwargs.get('sequence_length', 15)
        self.input_shape = None
        self.model = None
        self.scaler = StandardScaler()
        # Advanced architecture toggles
        self.use_attention = kwargs.get('use_attention', True)
        self.use_bidirectional = kwargs.get('use_bidirectional', True)
        self.use_multi_scale = kwargs.get('use_multi_scale', True)
        self.scales = kwargs.get('scales', [1, 2, 4])
        self.multi_scale_units = kwargs.get('multi_scale_units', max(self.units[0], 64))
        # Ensure stable classification with fixed class count across folds
        self.num_classes = kwargs.get('num_classes', (3 if self.task == 'classification' else 1))
        # Advanced training options
        self.use_class_weights = kwargs.get('use_class_weights', True)
        self.gradient_clip_norm = kwargs.get('gradient_clip_norm', 1.0)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1 if task == 'classification' else 0.0)
        
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
        
        # Always use LSTM processing. Normalize shapes:
        # - If input is [N, F], treat as single-timestep sequence: [N, 1, F]
        # - If input is [N, T, F], use as-is
        if len(input_shape) == 2:
            # Convert to pseudo-3D shape metadata for consistent handling downstream
            self.input_shape = (input_shape[0], 1, input_shape[1])
            self.logger.info(f"Using 2D input reshaped to sequence [T=1]: {self.input_shape}")
        elif len(input_shape) == 3:
            self.input_shape = input_shape
            self.logger.info(f"Using 3D input shape for sequence data: {self.input_shape}")
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # FIXED: Force CPU mode to avoid device conflicts
        device = '/CPU:0'
        self.logger.info(f"Using {device} for LSTM to avoid device conflicts")
        
        try:
            with tf.device(device):
                # Inputs are always sequences now (T may be 1)
                inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))
                x = tf.keras.layers.BatchNormalization()(inputs)
                x = tf.keras.layers.Dropout(self.dropout * 0.5)(x)

                time_steps = self.input_shape[1]
                enable_multi_scale = bool(self.use_multi_scale and time_steps is not None and time_steps > 1)

                # Optional multi-scale residual features (only meaningful for T>1)
                if enable_multi_scale:
                    ms_block = MultiScaleLSTMBlock(self.multi_scale_units, self.scales, self.dropout, self.recurrent_dropout, name='multi_scale')
                    ms_feat = ms_block(x)
                    ms_feat = tf.keras.layers.Dense(self.input_shape[2], activation='tanh', name='ms_dense')(ms_feat)
                    ms_feat = tf.keras.layers.Reshape((1, self.input_shape[2]))(ms_feat)
                    ms_feat = tf.keras.layers.Lambda(lambda t: tf.tile(t, [1, tf.shape(x)[1], 1]))(ms_feat)
                    x = tf.keras.layers.Add()([x, ms_feat])

                # Stacked (bi)LSTMs with BN+Dropout
                for i, u in enumerate(self.units):
                    # Base rule: return_sequences=True for all but last. If attention=True, force True for last as well.
                    is_last = (i == len(self.units) - 1)
                    return_sequences = (not is_last) or bool(self.use_attention)
                    lstm = tf.keras.layers.LSTM(
                        u,
                        return_sequences=return_sequences,
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        name=f'lstm_{i}'
                    )
                    if self.use_bidirectional:
                        x = tf.keras.layers.Bidirectional(lstm, name=f'bilstm_{i}')(x)
                    else:
                        x = lstm(x)
                    x = tf.keras.layers.BatchNormalization(name=f'bn_lstm_{i}')(x)
                    x = tf.keras.layers.Dropout(self.dropout, name=f'dp_lstm_{i}')(x)

                # Attention or last timestep pooling
                if self.use_attention:
                    last_dim = self.units[-1] * (2 if self.use_bidirectional else 1)
                    attn = AttentionLayer(last_dim, name='attention')
                    x, _ = attn(x)
                else:
                    # If last LSTM returned sequences=False, x is already [B, D]; otherwise take last timestep
                    x = x if len(x.shape) == 2 else x[:, -1, :]
                
                # Output layer
                if self.task == 'classification':
                    outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='classification_output')(x)
                else:
                    outputs = tf.keras.layers.Dense(1, activation='linear', name='regression_output')(x)
                
                self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # Compile model
                opt = tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001), clipnorm=self.gradient_clip_norm, epsilon=1e-7)
                if self.task == 'classification':
                    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing, from_logits=False)
                    self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
                else:
                    loss = tf.keras.losses.Huber(delta=1.0)
                    self.model.compile(optimizer=opt, loss=loss, metrics=['mae', 'mse'])
                    
        except Exception as build_error:
            self.logger.error(f"LSTM build failed: {build_error}")
            raise RuntimeError(f"Failed to build LSTM model: {build_error}")
        
        self.logger.info(f"LSTM model built successfully with input shape: {self.input_shape}")
        return self
    
    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert 2D tabular data to overlapping sequences of length sequence_length."""
        if X.ndim == 3:
            return X, y
        if X.ndim != 2:
            raise ValueError(f"Unsupported input shape for sequence prep: {X.shape}")
        n, f = X.shape
        if n <= self.sequence_length:
            return np.empty((0, self.sequence_length, f)), None if y is None else np.empty((0, 1))
        seqs = []
        targets = []
        for i in range(self.sequence_length, n):
            seqs.append(X[i - self.sequence_length:i])
            if y is not None:
                targets.append(y[i])
        X_seq = np.asarray(seqs)
        y_seq = None if y is None else np.asarray(targets)
        return X_seq, y_seq

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
            # Proper sequence handling
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            y_arr = y.values if isinstance(y, pd.Series) else y
            if len(self.input_shape) == 3:
                if X_arr.ndim == 2:
                    # If model expects T=1, simply expand dims; otherwise prepare overlapping sequences
                    if self.input_shape[1] == 1:
                        X_seq = np.expand_dims(X_arr, axis=1)
                        y_seq = y_arr
                    else:
                        X_seq, y_seq = self._prepare_sequences(X_arr, y_arr)
                else:
                    X_seq, y_seq = X_arr, y_arr
            else:
                X_seq, y_seq = X_arr, y_arr
            if X_seq is None or (isinstance(X_seq, np.ndarray) and X_seq.size == 0):
                raise ValueError("Insufficient data for LSTM training")
            
            # Ensure y shape
            y_reshaped = y_seq.reshape(-1, 1) if y_seq is not None and y_seq.ndim == 1 else y_seq
            
            # Time-aware split: last validation_split as validation
            vs = float(self.params.get('validation_split', 0.2))
            n_samples = X_seq.shape[0]
            val_len = max(1, int(n_samples * vs))
            train_len = n_samples - val_len
            X_train, X_val = X_seq[:train_len], X_seq[train_len:]
            y_train, y_val = (y_reshaped[:train_len], y_reshaped[train_len:]) if y_reshaped is not None else (None, None)
            # Centralized scaling: if inputs are pre-scaled, skip internal scaler
            expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
            if not expects_scaled:
                # Scale features for both 2D and 3D
                if X_train.ndim == 3:
                    tr_shape = X_train.shape
                    vl_shape = X_val.shape
                    X_tr_flat = X_train.reshape(-1, tr_shape[-1])
                    X_vl_flat = X_val.reshape(-1, vl_shape[-1])
                    X_tr_flat = self.scaler.fit_transform(X_tr_flat)
                    X_vl_flat = self.scaler.transform(X_vl_flat)
                    X_train = X_tr_flat.reshape(tr_shape)
                    X_val = X_vl_flat.reshape(vl_shape)
                else:
                    X_train = self.scaler.fit_transform(X_train)
                    X_val = self.scaler.transform(X_val)
            
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
                
                # Class weights and label format
                class_weight = None
                y_train_fit = y_train
                if self.task == 'classification':
                    # Use one-hot targets for CategoricalCrossentropy
                    from tensorflow.keras.utils import to_categorical
                    y_train_fit = to_categorical(y_train.reshape(-1), num_classes=self.num_classes)
                    y_val_fit = to_categorical(y_val.reshape(-1), num_classes=self.num_classes)
                    if self.use_class_weights:
                        try:
                            from sklearn.utils.class_weight import compute_class_weight
                            classes = np.unique(y_train.reshape(-1))
                            weights = compute_class_weight('balanced', classes=classes, y=y_train.reshape(-1))
                            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
                        except Exception:
                            class_weight = None
                else:
                    y_val_fit = y_val

                # Train the model
                history = self.model.fit(
                    X_train, y_train_fit,
                    validation_data=(X_val, y_val_fit),
                    epochs=self.params.get('epochs', 200),
                    batch_size=self.params.get('batch_size', 32),
                    callbacks=callbacks,
                    verbose=0,
                    shuffle=False,
                    class_weight=class_weight
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
                train_mae = history.history.get('mae', [np.nan])[-1]
                val_mae = history.history.get('val_mae', [np.nan])[-1]
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
            # Proper sequence handling
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            if len(self.input_shape) == 3:
                if X_arr.ndim == 2:
                    if self.input_shape[1] == 1:
                        X_reshaped = np.expand_dims(X_arr, axis=1)
                    else:
                        X_reshaped, _ = self._prepare_sequences(X_arr)
                else:
                    X_reshaped = X_arr
            else:
                X_reshaped = X_arr
            
            # Scale features using fitted scaler unless pre-scaled
            expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
            if not expects_scaled:
                try:
                    if X_reshaped.ndim == 3:
                        shp = X_reshaped.shape
                        flat = X_reshaped.reshape(-1, shp[-1])
                        flat = self.scaler.transform(flat)
                        X_reshaped = flat.reshape(shp)
                    else:
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