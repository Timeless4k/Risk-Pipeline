### Risk-Pipeline

End-to-end machine learning pipeline for market risk forecasting and classification on financial time series, with experiment tracking, interpretability (SHAP), and thesis-ready reporting.

### What this repository does

- **Assets and tasks**: Trains multiple models across assets:
  - US: `MSFT`, `AAPL`, `^VIX`, `^GSPC`
  - AU: `IOZ.AX`, `CBA.AX`, `BHP.AX`
  For both **regression** (e.g., returns/volatility) and **classification** (e.g., direction, regimes).
- **Pipeline**: Load cached market data → engineer features → optional regime tagging → time-series splits → train/evaluate multiple models → aggregate metrics → export artifacts/plots/SHAP → generate reports.
- **Reproducible experiments**: Each run writes a timestamped folder under `experiments/` with all configs, models, metrics, and summaries.

### Repository structure

```
Risk-Pipeline/
├── configs/
│   └── pipeline_config.json            # Main user-editable pipeline configuration
├── data_cache/                         # Local CSV cache per symbol and date range
├── experiments/                        # Outputs per run (models, metrics, exports)
├── logs/                               # Verbose pipeline logs
├── risk_pipeline/
│   ├── config/                         # Internal config helpers
│   ├── core/                           # Data, features, training, evaluation
│   ├── data/                           # Dataset abstraction
│   ├── interpretability/               # SHAP explainers and analysis
│   ├── models/                         # ARIMA, GARCH, XGBoost, LSTM, StockMixer
│   ├── utils/                          # Logging, persistence, GPU SHAP utils
│   └── visualization/                  # Plotting utilities
├── shap_plots/                         # Saved SHAP plots per symbol/model/task
├── thesis_reports/                     # Thesis-ready exports (CSV, JSON, TeX)
├── run_simple_pipeline.py              # Main entry-point script for a full run
├── scripts/                            # Utilities and experiments
└── requirements.txt                    # Python dependencies
```

### End-to-end flow

1. **Configuration**
   - Read user config from `configs/pipeline_config.json` and internal defaults from `risk_pipeline/config/*.py`.
   - Key sections:
     - `data`: date range, asset lists, cache dir
     - `features` and `feature_engineering`: indicator windows, sequence length, correlation/regime flags
     - `models`: hyperparameters for GARCH, LSTM, StockMixer, XGBoost
     - `training`: walk-forward split params, optimization schedule, mixed precision, parallelism
     - `shap`, `output`, `logging`, and advanced flags
2. **Data loading**
   - `risk_pipeline/core/data_loader.py` reads from `data_cache/` files: `<SYMBOL>_<START>_<END>.csv`.
   - Expected columns (example from `MSFT`): `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Returns, Log_Returns, Volatility, SMA_20, SMA_50, RSI`.
   - `risk_pipeline/data/dataset.py` produces aligned features/targets per task.
3. **Feature engineering**
   - `risk_pipeline/core/feature_engineer.py` uses config:
     - Moving averages: `ma_short=30`, `ma_long=60`
     - Volatility windows: `[5,10,20]`
     - Momentum/oscillators: `RSI(14)`, `MACD(12,26,9)`, stochastic (disabled if 0)
     - Bollinger bands: `period=20`, `std=2`
     - ATR(14)
     - Correlations on window `30` (pairs configurable)
     - Lags: if `use_only_price_lags=false`, includes engineered features alongside `price_lag_days=[30,60,90]`
   - Targets: regression (e.g., next-period return/volatility) and/or classification labels.
4. **Regime detection (optional)**
   - `risk_pipeline/core/regime_detector.py` labels regimes for filtering or classification when `enable_regime_features=true`.
5. **Time-series splits**
   - `risk_pipeline/core/splits.py` with walk-forward:
     - `walk_forward_splits=5`, `max_train_size=1008`, `test_size=126`, optional `validation_split=0.2` inside train.
     - Expanding or rolling behavior per config; preserves temporal order.
6. **Model training**
   - `risk_pipeline/models/*` via `model_factory.py` creates implementations:
     - ARIMA (`arima_model.py`) and GARCH (`garch_model.py`) for classical TS/volatility.
     - XGBoost (`xgboost_model.py`) for tabular features.
     - LSTM (`lstm_model.py`) with options: units `[128,64,32]`, bidirectional, attention, multi-scale with `scales=[1,2,4]` and shared `lstm_multi_scale_units=128`.
     - StockMixer (`stockmixer_model.py`) with temporal/indicator/cross-stock branches and fusion; config fields: `*_units`, `num_layers=4`, `attention_heads=8`, `dropout=0.15`, `market_dim=64`, `scale_count=5`.
   - `risk_pipeline/core/trainer.py` orchestrates training:
     - Optimization schedule: `learning_rate_schedule="cosine"`, `warmup_epochs=5`.
     - Regularization: `early_stopping_patience=20`, `reduce_lr_patience=10`, `gradient_clip_norm=1.0`.
     - Performance aids: `mixed_precision=true`, `gradient_accumulation_steps=2`, `batch_size=128`, `epochs=150`.
     - Parallelism knobs exist (`parallel_backend`, `num_workers`, `joblib_n_jobs`, `ray_num_cpus`, `dask_n_workers`), defaulting to conservative values in config.
7. **Evaluation and metrics**
   - `risk_pipeline/core/evaluator.py`: computes metrics defined in `metrics.py`.
     - Regression: RMSE, MAE, MAPE, R2 (and others depending on task config).
     - Classification: Accuracy, Precision/Recall, F1, ROC-AUC, PR-AUC; supports class weighting (`class_weight_balance=true`).
   - `metrics_summarizer.py`: aggregates fold-level metrics to per-model/per-asset summaries and writes `model_performance.csv` in the experiment folder.
8. **Results management**
   - `risk_pipeline/core/results_manager.py` produces the experiment layout under `experiments/<timestamp>/`:
     - `config.json`, `metadata.json`
     - `models/<SYMBOL>/<MODEL>/<task>/`: saved artifacts and fold outputs
     - `model_performance.csv`: consolidated metrics
     - Optional `thesis_export_<timestamp>/` for publication-ready artifacts
9. **Interpretability (SHAP)**
   - `interpretability/explainer_factory.py` selects SHAP method per model; `shap_analyzer.py` computes values and saves arrays/summaries.
   - `utils/gpu_shap_utils.py` enables GPU-backed SHAP where possible; `install_gpu_shap.py` and `GPU_SHAP_GUIDE.md` provide setup guidance.
   - `visualization/shap_visualizer.py` renders bar/summary/dependence plots to `shap_plots/<SYMBOL>/<MODEL>/<task>/` with `shap.max_display=10` (configurable).
10. **Visualization and reporting**
    - `visualization/volatility_visualizer.py` and other plotters write to `visualizations/`.
    - `utils/thesis_reporting.py` generates `thesis_reports/` content: tables (CSV/TeX), JSON analyses (feature importance, comparisons, statistical tests).

### Data schema

- Minimal expected raw columns: `Date, Open, High, Low, Close, Volume`.
- Engineered columns (examples): `Returns, Log_Returns, Volatility, SMA_20, SMA_50, RSI, MACD, Bollinger_*`, plus lags and correlation features per config.
- Index/Date handling: `Date` is parsed to a time index; ensure consistent timezone and business-day alignment across assets during multi-asset features.

### Training loop details (DL models)

- Batching: `batch_size=128`, optional `gradient_accumulation_steps=2` for effective larger batch.
- Optimizer/scheduler: cosine schedule with warmup; ReduceLROnPlateau fallback via `reduce_lr_patience`.
- Early stopping: monitors validation metric with `early_stopping_patience` epochs.
- Mixed precision: enabled when supported to reduce memory and improve throughput.
- Gradient clipping: `clip_norm=1.0` to stabilize training.

### Reproducibility

- Seeds: `training.random_state=42` used across splitters and model initializations where applicable.
- Determinism: some GPU ops may remain nondeterministic; log versions in `metadata.json` and `logs/pipeline_run_<timestamp>.log`.
- Config snapshot: Effective `config.json` saved in each `experiments/<timestamp>/`.

### Entry points and scripts

- **Full pipeline run**
  - PowerShell:
    ```powershell
    python .\run_simple_pipeline.py
    ```
  - Produces a new `experiments/<timestamp>/` with models, metrics, logs, and optional thesis exports.

- **Cross-market transfer experiments**
  - ```powershell
    python .\scripts\run_cross_transfer.py
    ```
  - Writes transfer matrices under `experiments/simple_run_*/transfer_matrices/`.

- **Cleanup workspace artifacts**
  - ```powershell
    python .\scripts\cleanup_workspace.py
    ```

- **Model/dev tests**
  - ```powershell
    python .\test_model_fixes.py
    python .\test_optimized_stockmixer.py
    python .\test_pytorch_models.py
    ```

- **GPU sanity check**
  - ```powershell
    python .\simple_pytorch_test.py
    ```

### Configuration knobs (from `configs/pipeline_config.json`)

- Data: `start_date`, `end_date`, `us_assets`, `au_assets`, `cache_dir`
- Features: volatility windows, MA windows, RSI/MACD/Bollinger/ATR, stochastic, correlation window/pairs, `sequence_length`, lags
- Feature engineering toggles: enable indicators, correlation, regime features, and feature selection (`mutual_info`, `k=50`)
- Models: GARCH orders/auto-order; LSTM units, dropout, bidirectionality, attention, multi-scale; StockMixer dimensions/layers; XGBoost hyperparameters
- Training: walk-forward counts and sizes, batch/epochs, warmup/cosine LR, early stopping, reduce LR, class weights, mixed precision, gradient accumulation, parallelism knobs
- SHAP: background size, max display, plot type, save flag
- Outputs/logging directories and log formatting

### Outputs

- `experiments/<timestamp>/`
  - `config.json`, `metadata.json`
  - `model_performance.csv`
  - `models/<SYMBOL>/<MODEL>/<task>/...`
  - optional `thesis_export_<timestamp>/`
- `shap_plots/<SYMBOL>/<MODEL>/<task>/...`
- `logs/pipeline_run_<timestamp>.log`

### How to extend

- Add a model: subclass `models/base_model.py`, register in `models/model_factory.py`.
- Add features: edit `core/feature_engineer.py` and reference in config.
- Add metrics: implement in `core/metrics.py`, aggregate in `evaluator.py`/`metrics_summarizer.py`.

### Requirements and environment

- Install dependencies:
  ```powershell
  pip install -r .\requirements.txt
  ```
- Optional GPU SHAP acceleration: see `GPU_SHAP_GUIDE.md` and `install_gpu_shap.py`.

### Troubleshooting

- **Missing data**: Ensure files exist in `data_cache/` named like `<SYMBOL>_<START>_<END>.csv`.
- **Model dependencies**: Install optional libs for XGBoost and Torch; verify CUDA for GPU features.
- **Long SHAP runs**: Reduce `shap.background_samples` or disable for heavy models; prefer GPU path.
- **Windows paths**: Use PowerShell examples; keep consistent slashes.

### At a glance

- **Purpose**: Financial time-series risk forecasting/classification with multi-model benchmarking and interpretability.
- **Core**: Data → Features → Splits → Train → Evaluate → Interpret → Report.
- **Assets**: US: `MSFT`, `AAPL`, `^VIX`, `^GSPC`; AU: `IOZ.AX`, `CBA.AX`, `BHP.AX`.
- **Outputs**: Reproducible experiment folders with metrics, models, plots, SHAP, and thesis-ready exports.
