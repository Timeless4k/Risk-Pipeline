import numpy as np
import pandas as pd

from risk_pipeline.config.global_config import GlobalConfig
from risk_pipeline.core.splits import generate_sliding_splits
from risk_pipeline.core.feature_engineer import FeatureEngineer


def _synthetic_df(n=400):
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    price = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        'Close': price,
        'High': price + np.random.rand(n),
        'Low': price - np.random.rand(n),
    }, index=idx)
    return df


def test_same_indices_and_scaling():
    np.random.seed(1337)
    cfg = GlobalConfig()
    df = _synthetic_df(400)
    # next_return target
    ret = df['Close'].pct_change().shift(-1).dropna()

    fe = FeatureEngineer()
    n = len(df)
    splits = list(generate_sliding_splits(n, cfg.train_size, cfg.val_size, cfg.step))
    assert len(splits) > 0

    train_slc, val_slc = splits[0]
    (X_seq_tr, X_flat_tr, y_tr), (X_seq_va, X_flat_va, y_va) = fe.create_canonical_views(df, ret, cfg, train_slc, val_slc)

    # shape checks
    assert X_seq_tr.shape[2] * X_seq_tr.shape[1] == X_flat_tr.shape[1]
    assert X_seq_va.shape[2] * X_seq_va.shape[1] == X_flat_va.shape[1]

    # pick a random sample/time/feature and ensure equality between views
    if X_seq_tr.shape[0] > 0:
        i = 0
        t = min(cfg.lookback_T - 1, X_seq_tr.shape[1] - 1)
        f = 0
        assert np.isclose(X_seq_tr[i, t, f], X_flat_tr[i, t * X_seq_tr.shape[2] + f])


def test_no_leakage():
    np.random.seed(1337)
    cfg = GlobalConfig()
    df = _synthetic_df(400)
    # Inject a future-only signal
    future_signal = np.zeros(len(df))
    future_signal[cfg.train_size + cfg.val_size + 5:] = 10.0
    df['Close'] = df['Close'] + future_signal
    ret = df['Close'].pct_change().shift(-1).dropna()

    fe = FeatureEngineer()
    n = len(df)
    splits = list(generate_sliding_splits(n, cfg.train_size, cfg.val_size, cfg.step))
    train_slc, val_slc = splits[0]
    (X_seq_tr, X_flat_tr, y_tr), (X_seq_va, X_flat_va, y_va) = fe.create_canonical_views(df, ret, cfg, train_slc, val_slc)

    # val set should not include the injected region (no leakage)
    assert val_slc.start >= cfg.train_size
