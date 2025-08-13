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

    @staticmethod
    def load_demo(n: int = 240, seed: int = 1337) -> Tuple[pd.DataFrame, pd.Series]:
        """Return a tiny deterministic demo dataset suitable for T=16.

        Produces CPU-only synthetic price series with columns Close/High/Low.
        n: number of timesteps (default 240)
        """
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        steps = rng.normal(loc=0.0, scale=1.0, size=n)
        price = 100.0 + np.cumsum(steps)
        high = price + np.abs(rng.normal(0, 0.5, size=n))
        low = price - np.abs(rng.normal(0, 0.5, size=n))
        df = pd.DataFrame({
            "Close": price.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
        }, index=idx)
        cfg = GlobalConfig()
        ds = CanonicalDataset.from_prices(df)
        y = ds.build_target(cfg)
        # Align df to target index to avoid NaNs at tail
        df = df.loc[y.index]
        return df, y
