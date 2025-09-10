"""
Market regime detection utilities.

Provides multiple complementary approaches to label regimes as 'Bull', 'Bear', 'Sideways':
- Hidden Markov Model (HMM) on returns
- Threshold-based regime classification using rolling returns
- Volatility-based regime detection using GARCH

Works with US and AU assets; optional dependencies are handled gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Optional dependencies
HMM_AVAILABLE = False
GARCH_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HMM_AVAILABLE = True
except Exception:
    pass

try:
    from arch import arch_model  # type: ignore
    GARCH_AVAILABLE = True
except Exception:
    pass


@dataclass
class RegimeDetectorConfig:
    window: int = 60
    bull_threshold: float = 0.1
    bear_threshold: float = -0.1
    hmm_states: int = 3
    random_state: int = 42


class MarketRegimeDetector:
    """
    Implement proper market regime detection using:
    1. Hidden Markov Model for regime switching
    2. Threshold-based regime classification using rolling returns
    3. Volatility-based regime detection using GARCH

    Outputs one of: 'Bull', 'Bear', 'Sideways'.
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        self.config = config or RegimeDetectorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def detect(self, returns: pd.Series, method: Literal['auto', 'hmm', 'threshold', 'garch', 'slope'] = 'auto') -> pd.Series:
        """Detect regimes using the specified method.

        - auto: prefer HMM, then GARCH, then threshold
        """
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return pd.Series(dtype='object')

        if method == 'auto':
            if HMM_AVAILABLE and len(returns) >= max(200, self.config.window * 3):
                try:
                    return self._detect_hmm(returns)
                except Exception as e:
                    self.logger.warning(f"HMM detection failed, falling back. Reason: {e}")
            if GARCH_AVAILABLE and len(returns) >= self.config.window * 3:
                try:
                    return self._detect_garch(returns)
                except Exception as e:
                    self.logger.warning(f"GARCH detection failed, falling back. Reason: {e}")
            # Prefer slope-based over simple threshold when advanced methods are unavailable
            try:
                return self._detect_slope(returns)
            except Exception as e:
                self.logger.warning(f"Slope detection failed, falling back to threshold. Reason: {e}")
                return self._detect_threshold(returns)
        elif method == 'hmm':
            return self._detect_hmm(returns)
        elif method == 'garch':
            return self._detect_garch(returns)
        elif method == 'slope':
            return self._detect_slope(returns)
        else:
            return self._detect_threshold(returns)

    def _detect_threshold(self, returns: pd.Series) -> pd.Series:
        """Threshold-based classification using rolling cumulative returns."""
        win = max(5, int(self.config.window))
        roll_sum = returns.rolling(window=win, min_periods=max(5, win // 2)).sum()
        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[roll_sum > self.config.bull_threshold] = 'Bull'
        regimes[roll_sum < self.config.bear_threshold] = 'Bear'
        mask_side = regimes.isna()
        regimes[mask_side] = 'Sideways'
        return regimes

    def _detect_hmm(self, returns: pd.Series) -> pd.Series:
        """HMM-based regime detection with 3 states; states mapped by mean return."""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is not available")
        X = returns.values.reshape(-1, 1)

        # Fit GaussianHMM
        model = GaussianHMM(n_components=int(self.config.hmm_states), covariance_type='full', n_iter=200, random_state=self.config.random_state)
        model.fit(X)
        hidden_states = model.predict(X)

        # Map states to regimes by mean of returns in each state
        state_means = {s: returns[hidden_states == s].mean() for s in np.unique(hidden_states)}
        # Highest mean -> Bull, lowest -> Bear, middle -> Sideways
        ordered = sorted(state_means.items(), key=lambda kv: kv[1])
        mapping = {ordered[0][0]: 'Bear', ordered[-1][0]: 'Bull'}
        for s, _ in ordered[1:-1]:
            mapping[s] = 'Sideways'

        regimes = pd.Series([mapping[s] for s in hidden_states], index=returns.index)
        return regimes

    def _detect_garch(self, returns: pd.Series) -> pd.Series:
        """Volatility-regime classification using GARCH(1,1) conditional volatility.

        High conditional volatility => Bear, low => Bull, middle => Sideways.
        """
        if not GARCH_AVAILABLE:
            raise ImportError("arch is not available")

        # Fit GARCH on returns (assume mean zero for short horizons)
        am = arch_model(returns.dropna() * 100.0, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
        res = am.fit(disp='off')
        cond_vol = res.conditional_volatility

        # Use rolling quantiles to adapt through time
        q_low = cond_vol.rolling(self.config.window, min_periods=max(10, self.config.window // 2)).quantile(0.33)
        q_high = cond_vol.rolling(self.config.window, min_periods=max(10, self.config.window // 2)).quantile(0.67)

        regimes = pd.Series(index=cond_vol.index, dtype='object')
        regimes[cond_vol <= q_low] = 'Bull'
        regimes[cond_vol >= q_high] = 'Bear'
        regimes[(cond_vol > q_low) & (cond_vol < q_high)] = 'Sideways'
        return regimes.reindex(returns.index, method='ffill')


    def _detect_slope(self, returns: pd.Series) -> pd.Series:
        """Slope-based classification using rolling OLS slope of cumulative returns.

        We compute the rolling slope of cumulative log-returns over a window. This is
        approximately the average return per step. We compare the slope against mean
        thresholds derived from bull/bear cumulative thresholds.
        """
        win = max(5, int(self.config.window))
        # Cumulative sum of returns (log-price up to a constant)
        cum = returns.cumsum()

        # Precompute centered time index for OLS slope per window
        t = np.arange(win, dtype=float)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        if t_var == 0:
            # Fallback to threshold if something goes wrong
            return self._detect_threshold(returns)

        def slope_window(y: np.ndarray) -> float:
            # OLS slope: cov(t, y)/var(t)
            y_mean = float(np.mean(y))
            cov_ty = float(((t - t_mean) * (y - y_mean)).sum())
            return cov_ty / t_var

        slopes = cum.rolling(window=win, min_periods=max(5, win // 2)).apply(slope_window, raw=True)

        # Convert cumulative thresholds to mean-per-step thresholds
        bull_mean_thr = float(self.config.bull_threshold) / win
        bear_mean_thr = float(self.config.bear_threshold) / win

        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[slopes > bull_mean_thr] = 'Bull'
        regimes[slopes < bear_mean_thr] = 'Bear'
        regimes[regimes.isna()] = 'Sideways'
        return regimes

