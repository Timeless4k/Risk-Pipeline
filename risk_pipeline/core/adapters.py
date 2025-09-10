from __future__ import annotations

from typing import Any

from .model_interface import BaseModel


class SeqAdapterModel(BaseModel):
    """Routes X_seq [N,T,F] to the wrapped sequence model; no data changes."""

    def __init__(self, wrapped: Any, name: str):
        self.wrapped = wrapped
        self._name = name

    def name(self) -> str:
        return self._name

    def build_model(self, input_spec: Any) -> None:
        return self.wrapped.build_model(input_spec)

    def fit(self, X_seq, y, config) -> None:
        # Prefer explicit train API if available (allows kwargs)
        if hasattr(self.wrapped, 'train'):
            kwargs = {
                "epochs": getattr(config, "max_epochs", None),
                "batch_size": getattr(config, "batch_size", None),
                "early_stopping_patience": getattr(config, "patience", None),
                "learning_rate": getattr(config, "lr", None),
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return self.wrapped.train(X_seq, y, **kwargs)
        # Fallback to fit without passing config
        if hasattr(self.wrapped, 'fit'):
            return self.wrapped.fit(X_seq, y)
        # Fallback to legacy train signature
        kwargs = {
            "epochs": getattr(config, "max_epochs", None),
            "batch_size": getattr(config, "batch_size", None),
            "early_stopping_patience": getattr(config, "patience", None),
            "learning_rate": getattr(config, "lr", None),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.wrapped.train(X_seq, y, **kwargs)

    def predict(self, X_seq):
        import numpy as np
        y = self.wrapped.predict(X_seq)
        return np.asarray(y).reshape(-1, 1)


class FlatAdapterModel(BaseModel):
    """Routes X_flat [N,TF] to the wrapped tabular model; no data changes."""

    def __init__(self, wrapped: Any, name: str):
        self.wrapped = wrapped
        self._name = name

    def name(self) -> str:
        return self._name

    def build_model(self, input_spec: Any) -> None:
        return self.wrapped.build_model(input_spec)

    def fit(self, X_flat, y, config) -> None:
        if hasattr(self.wrapped, 'train'):
            kwargs = {
                "epochs": getattr(config, "max_epochs", None),
                "early_stopping_patience": getattr(config, "patience", None),
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return self.wrapped.train(X_flat, y, **kwargs)
        if hasattr(self.wrapped, 'fit'):
            return self.wrapped.fit(X_flat, y)
        kwargs = {
            "epochs": getattr(config, "max_epochs", None),
            "early_stopping_patience": getattr(config, "patience", None),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.wrapped.train(X_flat, y, **kwargs)

    def predict(self, X_flat):
        import numpy as np
        y = self.wrapped.predict(X_flat)
        return np.asarray(y).reshape(-1, 1)
