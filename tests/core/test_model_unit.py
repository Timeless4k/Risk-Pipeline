import unittest
import numpy as np
from risk_pipeline.models.arima_model import ARIMAModel
from risk_pipeline.models.lstm_model import LSTMModel
from risk_pipeline.models.stockmixer_model import StockMixerModel
from risk_pipeline.models.xgboost_model import XGBoostModel

class TestARIMAModel(unittest.TestCase):
    def test_arima_regression(self):
        y = np.sin(np.linspace(0, 10, 100))
        model = ARIMAModel(model_type='arima', task='regression', config={'order': (1,1,1)})
        model.build_model(input_shape=(100,))
        model.train(None, y)
        preds = model.predict(np.zeros(10))
        self.assertEqual(preds.shape[0], 10)
        metrics = model.evaluate(np.zeros(10), y[-10:])
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)

class TestLSTMModel(unittest.TestCase):
    def test_lstm_regression(self):
        X = np.random.randn(100, 5, 3)
        y = np.random.randn(100, 1)
        model = LSTMModel(model_type='lstm', task='regression', config={'units': 4, 'epochs': 2})
        model.build_model(input_shape=(5, 3))
        model.train(X, y, epochs=2, batch_size=8)
        preds = model.predict(X[:10])
        self.assertEqual(preds.shape[0], 10)
        metrics = model.evaluate(X[:10], y[:10])
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)

class TestStockMixerModel(unittest.TestCase):
    def test_stockmixer_regression(self):
        X = np.random.randn(100, 5, 3)
        y = np.random.randn(100, 1)
        model = StockMixerModel(model_type='stockmixer', task='regression', config={'units': 4, 'n_layers': 2, 'epochs': 2})
        model.build_model(input_shape=(5, 3))
        model.train(X, y, epochs=2, batch_size=8)
        preds = model.predict(X[:10])
        self.assertEqual(preds.shape[0], 10)
        metrics = model.evaluate(X[:10], y[:10])
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)

class TestXGBoostModel(unittest.TestCase):
    def test_xgboost_regression(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = XGBoostModel(model_type='xgboost', task='regression', config={'params': {'n_estimators': 10, 'max_depth': 2}})
        model.build_model(input_shape=(5,))
        model.train(X, y)
        preds = model.predict(X[:10])
        self.assertEqual(preds.shape[0], 10)
        metrics = model.evaluate(X[:10], y[:10])
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)

if __name__ == '__main__':
    unittest.main()