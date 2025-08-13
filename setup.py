from setuptools import setup, find_packages

setup(
    name="risk_pipeline",
    version="0.1.0",
    description="Modular Interpretable Machine Learning for Volatility Forecasting",
    author="Risk Pipeline Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scipy>=1.10.0",
        "numba>=0.56.0",
        "scikit-learn>=1.1.0",
        "imbalanced-learn>=0.10.0",
        "yfinance>=0.2.0",
        "requests>=2.31.0",
        "tensorflow>=2.10.0",
        "xgboost>=1.7.0",
        "shap>=0.41.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "joblib>=1.2.0",
        "psutil>=5.9.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-mock>=3.14.0",
        ]
    }
)
