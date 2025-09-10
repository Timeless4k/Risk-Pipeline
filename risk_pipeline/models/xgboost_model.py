"""
XGBoost model implementation for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for regression and classification tasks."""
    
    def __init__(self, task: str = 'regression', **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            task: 'regression' or 'classification'
            **kwargs: Additional parameters
        """
        super().__init__(name="XGBoost", **kwargs)
        self.task = task
        self.scaler = StandardScaler()
        # Optional log-target support for regression stability
        self.use_log_vol_target = bool(kwargs.get('use_log_vol_target', False)) if task == 'regression' else False
        self.log_target_epsilon = float(kwargs.get('log_target_epsilon', 1e-6)) if task == 'regression' else 1e-6
        
        # Tuned default parameters (apply unless explicitly overridden)
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 800),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'random_state': kwargs.get('random_state', 42),
            'use_label_encoder': False,
            'eval_metric': 'logloss' if task == 'classification' else 'rmse',
            
            # Regularization parameters
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 1.0),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'min_child_weight': kwargs.get('min_child_weight', 5),
            'gamma': kwargs.get('gamma', 0.2),
            # Force CPU to avoid accidental CUDA selection
            'device': 'cpu',
            'predictor': 'cpu_predictor',
        }
        
        # Update with any additional parameters
        self.params.update(kwargs)
        
        # Force reliable CPU defaults; avoid flaky GPU selection in mixed environments
        self.params.setdefault('tree_method', 'hist')
        # Ensure CPU device settings take precedence
        self.params['device'] = 'cpu'
        self.params['predictor'] = 'cpu_predictor'
        # Remove any GPU-specific parameters that may have been passed in
        self.params.pop('gpu_id', None)
        self.params.pop('gpu_id', None)
        # Also guard against any booster device hints
        if 'device' in self.params and self.params['device'] != 'cpu':
            self.params['device'] = 'cpu'
        if 'tree_method' in self.params and str(self.params['tree_method']).startswith('gpu'):
            self.params['tree_method'] = 'hist'

        # Create model
        if task == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        
        self.logger.info(f"XGBoost model initialized for {task} task with params: {self.params}")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> 'XGBoostModel':
        """Build XGBoost model with optimized parameters."""
        self.input_shape = input_shape
        
        # QUICK CPU OPTIMIZATION: Use parallel cores for maximum performance
        cpu_count =  max(1, (__import__('psutil').cpu_count(logical=False) or 4))
        
        # Tuned defaults for stability and generalization
        xgb_params = {
            'n_jobs': cpu_count,
            'tree_method': 'hist',
            'device': 'cpu',
            'predictor': 'cpu_predictor',
            'n_estimators': self.params.get('n_estimators', 800),
            'max_depth': self.params.get('max_depth', 6),
            'learning_rate': self.params.get('learning_rate', 0.05),
            'subsample': self.params.get('subsample', 0.8),
            'colsample_bytree': self.params.get('colsample_bytree', 0.8),
            'min_child_weight': self.params.get('min_child_weight', 5),
            'gamma': self.params.get('gamma', 0.2),
            'reg_alpha': self.params.get('reg_alpha', 0.1),
            'reg_lambda': self.params.get('reg_lambda', 1.0),
            'random_state': self.params.get('random_state', 42),
            'verbosity': 0
        }
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**xgb_params)
        else:
            self.model = xgb.XGBRegressor(**xgb_params)
        
        self.logger.info(f"🚀 XGBoost model built with {cpu_count}-core optimization!")
        self.logger.info(f"⚡ Using tree_method='hist' for maximum speed")
        self.logger.info(f"💪 Parallel processing: {cpu_count} cores")
        self.logger.info(f"Final XGB params: {xgb_params}")
        
        return self
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'XGBoostModel':
        """
        Fit XGBoost model (compatibility method for sklearn interface).
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        self.train(X, y, **kwargs)
        return self
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        if len(X) < 10:
            raise ValueError("XGBoost requires at least 10 observations")
        
        self.logger.info(f"Training XGBoost model with {len(X)} observations")
        
        try:
            # Centralized scaling compatibility: skip internal scaling if provided
            expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
            if expects_scaled:
                X_scaled = X
            else:
                X_scaled = self.scaler.fit_transform(X)
            # Optional log-target transform (regression only)
            if self.task == 'regression' and (kwargs.get('use_log_vol_target', self.use_log_vol_target)):
                eps = kwargs.get('log_target_epsilon', self.log_target_epsilon)
                try:
                    y = np.log(np.asarray(y).ravel() + eps)
                except Exception:
                    pass
            
            # Balance classes if classification (optional, auto-on for heavy imbalance)
            if self.task == 'classification':
                try:
                    class_counts = Counter(np.asarray(y).ravel())
                    imbalance_ratio = max(class_counts.values()) / max(1, min(class_counts.values()))
                    if imbalance_ratio >= 5 and hasattr(self, '_apply_smote_tomek'):
                        X_scaled, y = self._apply_smote_tomek(X_scaled, np.asarray(y).ravel())
                except Exception:
                    pass

            # Fit
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self.logger.info("XGBoost training completed")
            
            return {
                'status': 'success',
                'n_estimators': getattr(self.model, 'n_estimators', None)
            }
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with XGBoost model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        try:
            # Ensure X is 2D array
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            elif X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            
            # Scale features (or pass-through if already scaled)
            expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
            X_scaled = X if expects_scaled else self.scaler.transform(X)
            
            # Make predictions
            if self.task == 'classification' and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)
                # Convert to labels: argmax for multiclass, threshold 0.5 for binary
                if isinstance(proba, np.ndarray) and proba.ndim == 2:
                    if proba.shape[1] > 1:
                        y_pred = np.argmax(proba, axis=1)
                    else:
                        y_pred = (proba.ravel() >= 0.5).astype(int)
                else:
                    y_pred = (np.asarray(proba).ravel() >= 0.5).astype(int)
            else:
                y_pred = self.model.predict(X_scaled)
            
            # Ensure output is 1D
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()
            
            # Final NaN/Inf guard
            try:
                arr = np.asarray(y_pred, dtype=float)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    self.logger.error("XGBoost predict produced NaN/Inf; returning zeros")
                    return np.zeros(len(X), dtype=float)
            except Exception:
                pass
            return y_pred
            
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            return np.zeros(len(X))
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        try:
            # Ensure X is 2D array
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            elif X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            
            # Scale features (or pass-through if already scaled)
            expects_scaled = bool(getattr(self, 'expects_scaled_input', False))
            X_scaled = X if expects_scaled else self.scaler.transform(X)
            
            # Get probabilities
            proba = self.model.predict_proba(X_scaled)
            
            return proba
            
        except Exception as e:
            self.logger.error(f"XGBoost predict_proba failed: {e}")
            return np.zeros((len(X), 1))
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate XGBoost model.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            if self.task == 'classification':
                metrics = self._calculate_classification_metrics(y, y_pred)
            else:
                metrics = self._calculate_regression_metrics(y, y_pred)
            
            self.logger.info(f"XGBoost evaluation completed")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"XGBoost evaluation failed: {e}")
            if self.task == 'classification':
                return {
                    'Accuracy': 0.0,
                    'F1': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0
                }
            else:
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf')
                }
    
    def _apply_smote_tomek(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE+Tomek balancing for classification."""
        try:
            # Log original class distribution
            original_dist = Counter(y)
            self.logger.info(f"Original class distribution: {dict(original_dist)}")
            
            # Apply SMOTE+Tomek
            smt = SMOTETomek(smote=SMOTE(k_neighbors=1), random_state=self.params['random_state'])
            X_resampled, y_resampled = smt.fit_resample(X, y)
            
            # Log new class distribution
            new_dist = Counter(y_resampled)
            self.logger.info(f"New class distribution after SMOTE+Tomek: {dict(new_dist)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.warning(f"SMOTE+Tomek failed: {e}. Using original data.")
            return X, y
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained or self.feature_names is None:
            return None
        
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")
        
        feature_importance = self.get_feature_importance()
        if not feature_importance:
            self.logger.warning("No feature importance available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Get top N features
            top_features = list(feature_importance.items())[:top_n]
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'XGBoost Feature Importance - {self.task.title()}')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot feature importance: {e}")
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_trained:
            return "Model not trained"
        
        try:
            summary = f"XGBoost Model Summary\n"
            summary += f"Task: {self.task}\n"
            summary += f"Parameters: {self.params}\n"
            summary += f"Feature count: {len(self.feature_names) if self.feature_names else 'Unknown'}\n"
            
            if self.task == 'classification':
                summary += f"Classes: {len(self.model.classes_) if hasattr(self.model, 'classes_') else 'Unknown'}\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return "Summary not available"
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model with additional XGBoost-specific data."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'task': self.task,
                'name': self.name,
                'scaler': self.scaler
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"XGBoost model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save XGBoost model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained XGBoost model."""
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.task = model_data['task']
            self.name = model_data['name']
            self.scaler = model_data['scaler']
            self.is_trained = True
            
            self.logger.info(f"XGBoost model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {e}")
            raise
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                       y: Union[pd.Series, np.ndarray], 
                       cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Input features
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing CV results
        """
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Use TimeSeriesSplit for financial data to prevent data leakage
            if cv_folds > 1:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv = tscv
            else:
                cv = cv_folds
            
            # Perform cross-validation
            if self.task == 'classification':
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
            else:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
            
            results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std()
            }
            
            self.logger.info(f"Cross-validation completed: mean={results['mean_score']:.4f}, std={results['std_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {
                'cv_scores': [],
                'mean_score': 0.0,
                'std_score': 0.0
            }
    
    def tune_hyperparameters(self, X: Union[pd.DataFrame, np.ndarray], 
                           y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with time series cross-validation.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Dictionary with best parameters and scores
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        
        # Validate input
        X, y = self._validate_input(X, y)
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Define parameter grid for tuning
            param_grid = {
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'reg_alpha': [0.01, 0.1, 1.0],
                'reg_lambda': [0.5, 1.0, 2.0],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            # Use TimeSeriesSplit for financial data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create base model for tuning
            if self.task == 'classification':
                base_model = xgb.XGBClassifier(
                    random_state=self.params['random_state'],
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                scoring = 'accuracy'
            else:
                base_model = xgb.XGBRegressor(
                    random_state=self.params['random_state'],
                    eval_metric='rmse'
                )
                scoring = 'r2'
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=tscv, 
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_scaled, y)
            
            # Update model with best parameters
            best_params = grid_search.best_params_
            self.params.update(best_params)
            
            # Recreate model with best parameters
            if self.task == 'classification':
                self.model = xgb.XGBClassifier(**self.params)
            else:
                self.model = xgb.XGBRegressor(**self.params)
            
            results = {
                'best_params': best_params,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Hyperparameter tuning completed. Best score: {results['best_score']:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            return {
                'best_params': self.params,
                'best_score': 0.0,
                'cv_results': {}
            } 