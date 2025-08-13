from dataclasses import dataclass, field
from typing import List, Literal


ScalingType = Literal["standard", "minmax", "none"]
ScaleFitScope = Literal["train_only", "all"]
TargetType = Literal["next_return", "next_close", "custom"]


@dataclass
class GlobalConfig:
    """Single source of truth for fair, uniform evaluation across models.

    All components should read from this config only. No per-model overrides
    that change data influence.
    """

    # Randomness
    random_seed: int = 1337

    # Windows
    lookback_T: int = 16
    step: int = 5
    train_size: int = 200
    val_size: int = 60

    # Preprocessing
    scaling: ScalingType = "standard"
    scale_fit_scope: ScaleFitScope = "train_only"
    target: TargetType = "next_return"

    # Metrics
    track_mse: bool = True
    track_mae: bool = True
    track_spearman_ic: bool = True

    # Training loop
    max_epochs: int = 100
    patience: int = 10
    batch_size: int | None = None  # None -> full batch for sequence models
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Explainability
    compute_shap: bool = False

    # Which models to run
    models_to_run: List[str] = field(default_factory=lambda: ["arima", "xgb", "lstm", "stockmixer"])

    # Persistence
    artifacts_dir: str = "artifacts/compare_run"
