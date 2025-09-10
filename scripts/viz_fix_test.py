import numpy as np
from risk_pipeline import RiskPipeline


def main():
    rp = RiskPipeline(experiment_name='viz_fix_test')

    # Minimal results structure
    results = {
        'AAPL': {
            'regression': {
                'arima': {'metrics': {}},
                'xgboost': {'metrics': {}}
            },
            'classification': {}
        }
    }

    # Dummy explainer with required API for ARIMA plots
    class DummyExplainer:
        def explain(self, X):
            return {
                'residuals': {'residuals': [0, 1, 2]},
                'decomposition': {
                    'trend': [0, 0, 0],
                    'seasonal': [0, 0, 0],
                    'residual': [0, 0, 0]
                },
                'forecast_intervals': {
                    'forecast': [0, 1, 2, 3],
                    'confidence_intervals': {
                        'lower': [0, 0, 0, 0],
                        'upper': [1, 1, 1, 1]
                    }
                }
            }

    dummy_explainer = DummyExplainer()

    # Fake SHAP outputs
    sv = np.random.randn(100, 5)
    X = np.random.randn(100, 5)
    feature_names = [f'f{i}' for i in range(5)]

    shap_results = {
        'AAPL': {
            'regression': {
                'arima': {
                    'shap_values': sv,
                    'X': X,
                    'feature_names': feature_names,
                    'explainer': dummy_explainer,
                    'task': 'regression'
                },
                'xgboost': {
                    'shap_values': sv,
                    'X': X,
                    'feature_names': feature_names,
                    'task': 'regression'
                }
            }
        }
    }

    # Should not raise TypeError about 'ARIMAExplainer' being iterable
    rp._generate_comprehensive_visualizations(results, shap_results)
    print('OK')


if __name__ == '__main__':
    main()


