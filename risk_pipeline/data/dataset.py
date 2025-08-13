from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

from risk_pipeline.config.global_config import GlobalConfig


@dataclass
class CanonicalDataset:
    X: pd.DataFrame

    @staticmethod
    def from_prices(prices: pd.DataFrame) -> "CanonicalDataset":
        # Minimal placeholder: pass-through; FeatureEngineer will build tensors
        return CanonicalDataset(X=prices.copy())

    def build_target(self, cfg: GlobalConfig) -> pd.Series:
        if cfg.target == "next_return":
            ret = self.X["Close"].pct_change().shift(-1)
            return ret.dropna()
        elif cfg.target == "next_close":
            tgt = self.X["Close"].shift(-1)
            return tgt.dropna()
        else:
            # custom target hook (identity fallback)
            return self.X["Close"].shift(-1).dropna()
