"""
Enhanced PyTorch LSTM model implementation for RiskPipeline.
Based on paper: "A Hybrid Model Integrating LSTM with Multiple GARCH-Type Models for Volatility and VaR Forecast"

Key improvements:
- Enhanced sequence handling for volatility prediction
- Realized volatility target support
- Improved feature integration
- Better temporal modeling for financial time series
- Support for multiple input types (returns, volatility, features)
"""

import logging
from typing import Dict, Tuple, Optional, Union, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base_model import BaseModel
from risk_pipeline.utils.torch_utils import get_torch_device


class _EnhancedTorchLSTMModule:
    """
    Enhanced LSTM module with improved architecture for volatility prediction.
    
    Based on paper methodology with:
    - Multiple LSTM layers with residual connections
    - Attention mechanism for temporal focus
    - Batch normalization for stability
    - Enhanced dropout for regularization
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], bidirectional: bool, 
                 num_classes: int, task: str, dropout: float, use_attention: bool = True):
        import torch
        import torch.nn as nn
        self.nn = nn
        self.torch = torch
        self.task = task
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Store parameters for serialization
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout

        prev = input_dim
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Build LSTM layers with batch normalization
        for i, h in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if i < len(hidden_dims) - 1 else 0.0
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(h * (2 if bidirectional else 1)))
            self.dropouts.append(nn.Dropout(dropout))
            prev = h * (2 if bidirectional else 1)
        
        # Attention mechanism for temporal focus
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prev,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(prev)
        
        # Output layers
        out_dim = num_classes if task == 'classification' else 1
        self.head = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev // 2, out_dim)
        )
        self.activation = nn.LogSoftmax(dim=1) if task == 'classification' else nn.Identity()

    def to(self, device):
        for m in self.lstm_layers:
            m.to(device)
        for bn in self.batch_norms:
            bn.to(device)
        for d in self.dropouts:
            d.to(device)
        if self.use_attention:
            self.attention.to(device)
            self.attention_norm.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        for m in self.lstm_layers:
            for p in m.parameters():
                yield p
        for bn in self.batch_norms:
            for p in bn.parameters():
                yield p
        for d in self.dropouts:
            for p in d.parameters():
                yield p
        if self.use_attention:
            for p in self.attention.parameters():
                yield p
            for p in self.attention_norm.parameters():
                yield p
        for p in self.head.parameters():
            yield p

    def __call__(self, x):
        # Pass through LSTM layers with batch normalization
        for i, (lstm, bn, dropout) in enumerate(zip(self.lstm_layers, self.batch_norms, self.dropouts)):
            x, _ = lstm(x)
            # Apply batch normalization (transpose for BatchNorm1d)
            x_reshaped = x.transpose(1, 2)  # [batch, features, time]
            x_reshaped = bn(x_reshaped)
            x = x_reshaped.transpose(1, 2)  # [batch, time, features]
            x = dropout(x)
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)  # Residual connection
        
        # Use last time step for prediction
        x = x[:, -1, :]
        logits = self.head(x)
        return self.activation(logits)
    
    def __getstate__(self):
        """Custom serialization for pickle compatibility."""
        state = self.__dict__.copy()
        # Remove non-serializable torch modules
        if 'nn' in state:
            del state['nn']
        if 'torch' in state:
            del state['torch']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization for pickle compatibility."""
        self.__dict__.update(state)
        # Re-import torch modules
        import torch
        import torch.nn as nn
        self.nn = nn
        self.torch = torch
        # Rebuild the model architecture
        self._rebuild_model()
    
    def _rebuild_model(self):
        """Rebuild the model architecture after deserialization."""
        import torch
        import torch.nn as nn
        
        prev = self.input_dim
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Rebuild LSTM layers
        for i, h in enumerate(self.hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                    dropout=self.dropout if i < len(self.hidden_dims) - 1 else 0.0
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(h * (2 if self.bidirectional else 1)))
            self.dropouts.append(nn.Dropout(self.dropout))
            prev = h * (2 if self.bidirectional else 1)
        
        # Rebuild attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prev,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(prev)
        
        # Rebuild output layers
        out_dim = self.num_classes if self.task == 'classification' else 1
        self.head = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(prev // 2, out_dim)
        )
        self.activation = nn.LogSoftmax(dim=1) if self.task == 'classification' else nn.Identity()


class LSTMModel(BaseModel):
    """
    Enhanced LSTM model for volatility prediction with paper-based improvements.
    
    Key features:
    - Realized volatility target support
    - Enhanced sequence handling
    - Attention mechanism for temporal focus
    - Better feature integration
    - Improved regularization
    """
    
    def __init__(self, task: str = 'regression', **kwargs):
        super().__init__(name="EnhancedLSTM", **kwargs)
        self.logger = logging.getLogger(__name__)
        self.task = task
        self.units = kwargs.get('units', [128, 64, 32])  # Deeper network
        self.dropout = float(kwargs.get('dropout', 0.3))  # Higher dropout
        self.use_bidirectional = bool(kwargs.get('use_bidirectional', True))
        self.sequence_length = int(kwargs.get('sequence_length', 20))  # Longer sequences
        self.num_classes = int(kwargs.get('num_classes', (3 if task == 'classification' else 1)))
        self.use_attention = bool(kwargs.get('use_attention', True))  # Attention mechanism
        self.scaler = StandardScaler()
        
        # Enhanced training parameters
        self.params = {
            'batch_size': int(kwargs.get('batch_size', 64)),  # Larger batch size
            'epochs': int(kwargs.get('epochs', 300)),  # More epochs
            'validation_split': float(kwargs.get('validation_split', 0.2)),
            'learning_rate': float(kwargs.get('learning_rate', 5e-4)),  # Lower learning rate
            'weight_decay': float(kwargs.get('weight_decay', 1e-5)),  # L2 regularization
            'patience': int(kwargs.get('patience', 20)),  # Early stopping
        }
        
        # Realized volatility support
        self.use_realized_vol = bool(kwargs.get('use_realized_vol', True))
        self.rv_window = int(kwargs.get('rv_window', 5))
        
        self.device_str = get_torch_device(prefer_gpu=True)
        self.model: Optional[_EnhancedTorchLSTMModule] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.training_history = {'loss': [], 'val_loss': []}

    def calculate_realized_volatility(self, prices: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate realized volatility as per paper methodology.
        
        Args:
            prices: Price series
            window: Rolling window for RV calculation
            
        Returns:
            Realized volatility series
        """
        if window is None:
            window = self.rv_window
            
        # Calculate log returns
        log_returns = 100 * np.log(prices / prices.shift(1))
        
        if len(log_returns) < window:
            self.logger.warning(f"Insufficient data for {window}-day RV calculation")
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate rolling realized volatility
        rv = log_returns.rolling(window=window, min_periods=window//2).std()
        
        # Annualize (assuming daily data)
        rv_annualized = rv * np.sqrt(252)
        
        self.logger.info(f"Calculated {window}-day realized volatility")
        return rv_annualized

    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Enhanced sequence preparation with better handling of financial time series.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (sequences, targets)
        """
        if X.ndim == 3:
            return X, y
        n, f = X.shape
        if n <= self.sequence_length:
            return np.empty((0, self.sequence_length, f)), None if y is None else np.empty((0, 1))
        
        seqs = []
        targets = []
        for i in range(self.sequence_length, n):
            seqs.append(X[i - self.sequence_length:i])
            if y is not None:
                targets.append(y[i])
        return np.asarray(seqs), (None if y is None else np.asarray(targets))

    def build_model(self, input_shape: Tuple[int, ...]) -> 'LSTMModel':
        """
        Build enhanced LSTM model with attention mechanism.
        
        Args:
            input_shape: Input shape tuple
            
        Returns:
            Self for chaining
        """
        if len(input_shape) == 2:
            self.input_shape = (input_shape[0], 1, input_shape[1])
            input_dim = int(input_shape[1])
        elif len(input_shape) == 3:
            self.input_shape = input_shape
            input_dim = int(input_shape[2])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        self.model = _EnhancedTorchLSTMModule(
            input_dim=input_dim,
            hidden_dims=self.units,
            bidirectional=self.use_bidirectional,
            num_classes=self.num_classes,
            task=self.task,
            dropout=self.dropout,
            use_attention=self.use_attention,
        ).to(self.device_str)
        
        self.logger.info(f"Built enhanced LSTM model with {len(self.units)} layers, attention={self.use_attention}")
        return self

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Enhanced training with realized volatility support and improved regularization.
        
        Args:
            X: Input features (can include price data for realized volatility)
            y: Target values (volatility or returns)
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            X_arr, y_arr = self._validate_input(X, y)
            self.build_model(X_arr.shape)
        else:
            X_arr, y_arr = self._validate_input(X, y)

        X_np = X_arr.values if isinstance(X_arr, pd.DataFrame) else X_arr
        y_np = y_arr.values if isinstance(y_arr, pd.Series) else y_arr

        if self.input_shape is None:
            raise ValueError("Model not built")

        # Enhanced data preparation with realized volatility support
        if isinstance(X_arr, pd.DataFrame) and self.use_realized_vol:
            # Look for price columns to calculate realized volatility
            price_cols = ['Close', 'Adj Close', 'Price']
            price_col = None
            for col in price_cols:
                if col in X_arr.columns:
                    price_col = col
                    break
            
            if price_col:
                # Calculate realized volatility and use as additional feature
                prices = X_arr[price_col].dropna()
                if len(prices) > self.rv_window:
                    rv = self.calculate_realized_volatility(prices, self.rv_window)
                    # Add realized volatility as a feature
                    rv_aligned = rv.reindex(X_arr.index).fillna(method='ffill')
                    X_np = np.column_stack([X_np, rv_aligned.values])
                    self.logger.info("Added realized volatility as feature")

        # Prepare sequences
        if self.input_shape[1] == 1 and X_np.ndim == 2:
            X_seq = np.expand_dims(X_np, axis=1)
            y_seq = y_np
        else:
            X_seq, y_seq = self._prepare_sequences(X_np, y_np)

        vs = float(self.params['validation_split'])
        n = X_seq.shape[0]
        v = max(1, int(n * vs))
        tr = n - v
        X_tr, X_vl = X_seq[:tr], X_seq[tr:]
        y_tr, y_vl = y_seq[:tr], y_seq[tr:]

        # Scaling if not pre-scaled
        expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
        if not expects_scaled:
            tr_shape = X_tr.shape
            vl_shape = X_vl.shape
            X_tr_flat = self.scaler.fit_transform(X_tr.reshape(-1, tr_shape[-1]))
            X_vl_flat = self.scaler.transform(X_vl.reshape(-1, vl_shape[-1]))
            X_tr = X_tr_flat.reshape(tr_shape)
            X_vl = X_vl_flat.reshape(vl_shape)

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device_str)
        xtr = torch.tensor(X_tr, dtype=torch.float32)
        xvl = torch.tensor(X_vl, dtype=torch.float32)
        if self.task == 'classification':
            ytr = torch.tensor(y_tr.reshape(-1), dtype=torch.long)
            yvl = torch.tensor(y_vl.reshape(-1), dtype=torch.long)
            criterion = nn.NLLLoss()
        else:
            ytr = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32)
            yvl = torch.tensor(y_vl.reshape(-1, 1), dtype=torch.float32)
            criterion = nn.MSELoss()

        bs = int(kwargs.get('batch_size', self.params['batch_size']))
        dl_tr = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=False)
        dl_vl = DataLoader(TensorDataset(xvl, yvl), batch_size=bs, shuffle=False)

        # Enhanced optimizer with weight decay
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=float(kwargs.get('learning_rate', self.params['learning_rate'])),
            weight_decay=float(kwargs.get('weight_decay', self.params['weight_decay']))
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.model.to(device)
        history = {'loss': [], 'val_loss': []}
        epochs = int(kwargs.get('epochs', self.params['epochs']))
        patience = int(kwargs.get('patience', self.params['patience']))
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.torch.set_grad_enabled(True)
            train_loss = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= max(1, tr)

            # Validation phase
            self.model.torch.set_grad_enabled(False)
            val_loss = 0.0
            with self.model.torch.no_grad():
                for xb, yb in dl_vl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= max(1, v)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 50 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        self.training_history = history
        self.is_trained = True
        return {
            'train_loss': history['loss'][-1], 
            'val_loss': history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(history['loss'])
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        X_arr, _ = self._validate_input(X)
        X_np = X_arr.values if isinstance(X_arr, pd.DataFrame) else X_arr
        if self.input_shape is None:
            raise ValueError("Model not built")

        if self.input_shape[1] == 1 and X_np.ndim == 2:
            X_seq = np.expand_dims(X_np, axis=1)
        else:
            X_seq, _ = self._prepare_sequences(X_np)

        expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
        if not expects_scaled:
            shp = X_seq.shape
            flat = self.scaler.transform(X_seq.reshape(-1, shp[-1]))
            X_seq = flat.reshape(shp)

        import torch
        device = torch.device(self.device_str)
        xt = torch.tensor(X_seq, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = self.model(xt)
        out = preds.detach().cpu().numpy()
        if self.task == 'classification':
            return np.argmax(out, axis=1)
        return out.reshape(-1)

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Enhanced evaluation with additional metrics for volatility prediction.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        X_val, y_val = self._validate_input(X, y)
        y_pred = self.predict(X_val)
        
        if self.task == 'classification':
            return self._calculate_classification_metrics(y_val, y_pred)
        else:
            # Enhanced regression metrics for volatility prediction
            metrics = self._calculate_regression_metrics(y_val, y_pred)
            
            # Add additional volatility-specific metrics
            try:
                # Mean Absolute Percentage Error (MAPE) - important for volatility
                mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
                metrics['MAPE'] = mape
                
                # Directional accuracy (for volatility trends)
                if len(y_val) > 1:
                    actual_direction = np.diff(y_val) > 0
                    pred_direction = np.diff(y_pred) > 0
                    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                    metrics['Directional_Accuracy'] = directional_accuracy
                
                # Volatility clustering (GARCH-like behavior)
                if len(y_val) > 10:
                    actual_vol = np.std(y_val)
                    pred_vol = np.std(y_pred)
                    vol_ratio = pred_vol / (actual_vol + 1e-8)
                    metrics['Volatility_Ratio'] = vol_ratio
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate additional metrics: {e}")
            
            return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model architecture and training info
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "EnhancedLSTM",
            "task": self.task,
            "architecture": {
                "units": self.units,
                "bidirectional": self.use_bidirectional,
                "attention": self.use_attention,
                "sequence_length": self.sequence_length,
                "dropout": self.dropout
            },
            "training_params": self.params,
            "training_history": {
                "final_train_loss": self.training_history['loss'][-1] if self.training_history['loss'] else None,
                "final_val_loss": self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
                "epochs_trained": len(self.training_history['loss'])
            }
        }
        
        return info

    def get_attention_weights(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """
        Get attention weights if attention mechanism is enabled.
        
        Args:
            X: Input features
            
        Returns:
            Attention weights array or None if not available
        """
        if not self.use_attention or not self.is_trained:
            return None
        
        try:
            X_arr, _ = self._validate_input(X)
            X_np = X_arr.values if isinstance(X_arr, pd.DataFrame) else X_arr
            
            if self.input_shape[1] == 1 and X_np.ndim == 2:
                X_seq = np.expand_dims(X_np, axis=1)
            else:
                X_seq, _ = self._prepare_sequences(X_np)
            
            import torch
            device = torch.device(self.device_str)
            xt = torch.tensor(X_seq, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # Get attention weights from the model
                # This would require modifying the forward pass to return attention weights
                # For now, return None as this is a placeholder
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract attention weights: {e}")
            return None