"""
PyTorch StockMixer model implementation for RiskPipeline.
Based on the paper's architecture with adaptations for our data format.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from risk_pipeline.utils.torch_utils import get_torch_device
from .base_model import BaseModel


# Activation function
acv = nn.GELU()


class MixerBlock(nn.Module):
    """Basic MLP mixing block from the paper."""
    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    """2D mixing with time and channel dimensions."""
    def __init__(self, time_steps: int, channels: int):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)  # [B, T, C]

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TriU(nn.Module):
    """Triangular upper matrix for time mixing."""
    def __init__(self, time_step: int):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([
            nn.Linear(i + 1, 1) for i in range(time_step)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    """Time-specific mixing block."""
    def __init__(self, time_step: int):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module):
    """Multi-scale time mixing."""
    def __init__(self, time_step: int, channel: int, scale_count: int = 1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList([
            nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2 ** i, stride=2 ** i),
                TriU(int(time_step / 2 ** i)),
                nn.Hardswish(),
                TriU(int(time_step / 2 ** i))
            ) for i in range(scale_count)
        ])
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, C, T]
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    """2D mixing with TriU for time dimension."""
    def __init__(self, time_steps: int, channels: int):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)  # [B, T, C]

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """Multi-time 2D mixing."""
    def __init__(self, time_step: int, channel: int, scale_dim: int = 8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """Stock-level mixing without graph."""
    def __init__(self, stocks: int, hidden_dim: int = 20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.permute(1, 0)  # [S, B]
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)  # [B, S]
        return x


class StockMixerNet(nn.Module):
    """Enhanced StockMixer model optimized for 12GB VRAM utilization."""
    def __init__(self, stocks: int, time_steps: int, channels: int, market: int, scale: int, 
                 num_classes: int = 1, task: str = 'regression', dropout: float = 0.15):
        super(StockMixerNet, self).__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.market = market
        self.scale = scale
        self.task = task
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Enhanced architecture for better VRAM utilization
        scale_dim = 16  # Increased scale dimension
        hidden_dim = max(256, channels * 2)  # Larger hidden dimensions
        
        # Multiple mixing layers for deeper architecture
        self.mixer_layers = nn.ModuleList([
            MultTime2dMixer(time_steps, channels, scale_dim=scale_dim) for _ in range(2)
        ])
        
        # Enhanced channel processing
        self.channel_processor = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Enhanced time processing
        self.time_processor = nn.Sequential(
            nn.Linear(time_steps * 2 + scale_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Multiple convolution layers for better feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, padding=2)
        ])
        
        # Enhanced stock mixing
        self.stock_mixer = NoGraphMixer(stocks, market)
        
        # Additional processing layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(time_steps * 2 + scale_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Enhanced output layer
        out_dim = (num_classes if task == 'classification' else 1)
        self.output_fc = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_dim)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1) if task == 'classification' else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, S, T, C] or [B, T, C] - adapt to our data format
        if inputs.dim() == 3:
            # [B, T, C] -> [B, 1, T, C] for single stock
            inputs = inputs.unsqueeze(1)
        
        B, S, T, C = inputs.shape
        
        # Process each stock separately with enhanced architecture
        stock_outputs = []
        for s in range(S):
            stock_input = inputs[:, s, :, :]  # [B, T, C]
            
            # Apply multiple convolution layers for better feature extraction
            conv_features = []
            for conv_layer in self.conv_layers:
                x = stock_input.permute(0, 2, 1)  # [B, C, T]
                x = conv_layer(x)  # [B, C, T']
                x = x.permute(0, 2, 1)  # [B, T', C]
                conv_features.append(x)
            
            # Use the first conv output for mixing
            x = conv_features[0]  # [B, T//2, C]
            
            # Apply multiple mixing layers
            mixed_features = stock_input
            for mixer_layer in self.mixer_layers:
                mixed_features = mixer_layer(mixed_features, x)  # [B, T*2 + scale_dim, C]
            
            # Enhanced channel processing
            channel_output = self.channel_processor(mixed_features).squeeze(-1)  # [B, T*2 + scale_dim]
            
            # Enhanced time processing
            time_output = self.time_processor(channel_output)  # [B, 1]
            
            stock_outputs.append(time_output)
        
        # Stack stock outputs: [B, S, 1]
        stock_outputs = torch.stack(stock_outputs, dim=1)  # [B, S, 1]
        
        # Enhanced stock-level mixing
        z = self.stock_mixer(stock_outputs.squeeze(-1))  # [B, S]
        z = self.fusion_layer(z.unsqueeze(-1))  # [B, 1]
        
        # Combine all features
        final_output = time_output + z  # [B, 1]
        
        # Apply enhanced output layer
        output = self.output_fc(final_output)
        
        return self.logsoftmax(output)


class StockMixerModel(BaseModel):
    def __init__(self, task: str = 'regression', **kwargs):
        super().__init__(name="StockMixer", **kwargs)
        self.task = task
        self.stocks = int(kwargs.get('stocks', 1))
        self.time_steps = int(kwargs.get('time_steps', 30))
        self.channels = int(kwargs.get('channels', 42))  # Number of features
        self.market = int(kwargs.get('market', 64))  # Increased for 12GB VRAM
        self.scale = int(kwargs.get('scale', 5))  # Increased scale count
        self.num_classes = int(kwargs.get('num_classes', 2 if task == 'classification' else 1))
        self.dropout = float(kwargs.get('dropout', 0.15))
        
        # Optimized parameters for 12GB VRAM
        self.params.update({
            'batch_size': int(kwargs.get('batch_size', 128)),  # Increased batch size
            'epochs': int(kwargs.get('epochs', 150)),  # More epochs
            'learning_rate': float(kwargs.get('learning_rate', 5e-4)),  # Lower LR for stability
            'validation_split': float(kwargs.get('validation_split', 0.2)),
            'gradient_accumulation_steps': int(kwargs.get('gradient_accumulation_steps', 2)),  # For larger effective batch size
            'mixed_precision': kwargs.get('mixed_precision', True),  # Enable mixed precision
            'gradient_clip_norm': float(kwargs.get('gradient_clip_norm', 1.0)),
        })
        self.model: Optional[StockMixerNet] = None
        self.device_str = get_torch_device(prefer_gpu=True)
        self.scaler = None  # For mixed precision

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        """Convert input to [B, T, C] format for the StockMixer."""
        if X.ndim == 2:
            # [B, C] -> [B, 1, C] (single time step)
            return X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim == 3:
            # [B, T, C] - already correct format
            return X
        elif X.ndim == 4:
            # [B, S, T, C] -> [B, T, C] (flatten stocks)
            return X.reshape(X.shape[0], X.shape[2], X.shape[3])
        else:
            raise ValueError(f"Unsupported input shape {X.shape}; expected [N,C], [N,T,C], or [N,S,T,C]")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'StockMixerModel':
        if len(input_shape) == 2:
            # [B, C] - single time step
            self.time_steps = 1
            self.channels = int(input_shape[1])
        elif len(input_shape) == 3:
            # [B, T, C] - time series
            self.time_steps = int(input_shape[1])
            self.channels = int(input_shape[2])
        elif len(input_shape) == 4:
            # [B, S, T, C] - multiple stocks
            self.stocks = int(input_shape[1])
            self.time_steps = int(input_shape[2])
            self.channels = int(input_shape[3])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        self.model = StockMixerNet(
            stocks=self.stocks,
            time_steps=self.time_steps,
            channels=self.channels,
            market=self.market,
            scale=self.scale,
            num_classes=self.num_classes,
            task=self.task,
            dropout=self.dropout,
        ).to(self.device_str)
        return self

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        if self.model is None:
            X_arr, y_arr = self._validate_input(X, y)
            self.build_model(X_arr.shape)
        else:
            X_arr, y_arr = self._validate_input(X, y)

        X3 = self._ensure_3d(X_arr)
        if np.any(np.isnan(X3)) or np.any(np.isinf(X3)):
            raise ValueError("Features contain NaN or infinite values")
        if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
            raise ValueError("Targets contain NaN or infinite values")

        idx = np.arange(X3.shape[0])
        train_idx, val_idx = train_test_split(idx, test_size=self.params['validation_split'], random_state=42)
        X_train, X_val = X3[train_idx], X3[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        import torch.optim as optim
        from torch.cuda.amp import autocast, GradScaler
        
        device = torch.device(self.device_str)
        self.model.to(device)
        
        # Initialize mixed precision scaler
        if self.params.get('mixed_precision', True) and device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        if self.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        
        # Use AdamW optimizer for better performance
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.params['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.params['epochs'],
            eta_min=self.params['learning_rate'] * 0.01
        )

        bs = int(kwargs.get('batch_size', self.params['batch_size']))
        epochs = int(kwargs.get('epochs', self.params['epochs']))
        grad_accum_steps = int(kwargs.get('gradient_accumulation_steps', self.params.get('gradient_accumulation_steps', 1)))
        grad_clip_norm = float(kwargs.get('gradient_clip_norm', self.params.get('gradient_clip_norm', 1.0)))

        def _iter(Xb, yb, train=True):
            total = 0.0
            count = 0
            rng = range(0, Xb.shape[0], bs)
            
            if train:
                optimizer.zero_grad()
            
            for step, i in enumerate(rng):
                xb = torch.tensor(Xb[i:i+bs], dtype=torch.float32, device=device)
                yb_t = torch.tensor(yb[i:i+bs], dtype=torch.long if self.task=='classification' else torch.float32, device=device)
                
                # Use mixed precision if available
                if self.scaler is not None:
                    with autocast():
                        preds = self.model(xb)
                        loss = criterion(preds, yb_t if self.task=='classification' else yb_t.view(-1,1))
                        loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
                else:
                    preds = self.model(xb)
                    loss = criterion(preds, yb_t if self.task=='classification' else yb_t.view(-1,1))
                    loss = loss / grad_accum_steps
                
                if train:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Gradient accumulation
                    if (step + 1) % grad_accum_steps == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                            optimizer.step()
                        optimizer.zero_grad()
                
                total += loss.item() * xb.size(0) * grad_accum_steps
                count += xb.size(0)
            
            return total / max(1, count)

        history = {'loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = kwargs.get('early_stopping_patience', 20)
        
        for epoch in range(epochs):
            tl = _iter(X_train, y_train, train=True)
            vl = _iter(X_val, y_val, train=False)
            
            # Step the scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            history['loss'].append(tl)
            history['val_loss'].append(vl)
            history['lr'].append(current_lr)
            
            # Early stopping
            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {tl:.6f}, Val Loss: {vl:.6f}, LR: {current_lr:.6f}")

        self.training_history = history
        self.is_trained = True
        result: Dict[str, Any] = {
            'train_loss': history['loss'][-1], 
            'val_loss': history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(history['loss'])
        }
        return result

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        X_arr, _ = self._validate_input(X)
        X3 = self._ensure_3d(X_arr)
        device = torch.device(self.device_str)
        xb = torch.tensor(X3, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = self.model(xb).detach().cpu().numpy()
        if self.task == 'classification':
            return np.argmax(preds, axis=1)
        return preds.reshape(-1)

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        X_val, y_val = self._validate_input(X, y)
        y_pred = self.predict(X_val)
        if self.task == 'classification':
            return self._calculate_classification_metrics(y_val, y_pred)
        else:
            return self._calculate_regression_metrics(y_val, y_pred)
