import numpy as np
import os

from risk_pipeline.models.stockmixer_model import StockMixerModel


def main():
    # Simple sanity test for StockMixerModel
    # Shapes: [N, S, T, F]
    num_samples = 64
    num_stocks = 3
    seq_len = 20
    num_indicators = 4

    rng = np.random.default_rng(42)
    X = rng.normal(size=(num_samples, num_stocks, seq_len, num_indicators)).astype(np.float32)
    y = rng.normal(size=(num_samples,)).astype(np.float32)

    # Build and train model (regression)
    model = StockMixerModel(task='regression',
                            n_stocks=num_stocks,
                            n_indicators=num_indicators,
                            sequence_length=seq_len,
                            epochs=1,
                            batch_size=16,
                            learning_rate=1e-3)

    model.build_model(X.shape)
    metrics = model.train(X, y, epochs=1, batch_size=16)
    print("Train metrics:", metrics)

    # Predict
    preds = model.predict(X[:8])
    print("Predictions sample (first 5):", preds[:5])

    # Evaluate
    eval_metrics = model.evaluate(X[:16], y[:16])
    print("Eval metrics (subset):", eval_metrics)


if __name__ == "__main__":
    # Ensure consistent threading behavior for quick tests
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()


