"""
Data quality utilities for RiskPipeline.

This module handles financial data validation, cleaning, and quality checks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validator for financial data quality."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds
        self.missing_threshold = self.config.get('missing_threshold', 0.1)  # 10% missing data
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # 3 standard deviations
        self.volatility_threshold = self.config.get('volatility_threshold', 0.5)  # 50% daily volatility
        
    def validate_financial_data(self, data: pd.DataFrame, 
                              required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of financial data.
        
        Args:
            data: Financial data DataFrame
            required_columns: List of required columns
            
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Basic data structure validation
            structure_validation = self._validate_structure(data, required_columns)
            results.update(structure_validation)
            
            if not structure_validation['is_valid']:
                results['is_valid'] = False
                return results
            
            # Data quality checks
            quality_checks = self._check_data_quality(data)
            results.update(quality_checks)
            
            # Financial-specific validation
            financial_validation = self._validate_financial_characteristics(data)
            results.update(financial_validation)
            
            # Overall validation
            if quality_checks['issues'] or financial_validation['issues']:
                results['is_valid'] = False
            
            # Generate summary statistics
            results['stats'] = self._generate_summary_stats(data)
            
            self.logger.info(f"Data validation completed. Valid: {results['is_valid']}")
            if results['issues']:
                self.logger.warning(f"Found {len(results['issues'])} issues: {results['issues']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            results['is_valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")
            return results
    
    def _validate_structure(self, data: pd.DataFrame, 
                           required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate basic data structure."""
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if data is empty
        if data.empty:
            results['is_valid'] = False
            results['issues'].append("Data is empty")
            return results
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                results['is_valid'] = False
                results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            results['warnings'].append("Data index is not datetime - time series operations may fail")
        
        # Check data types
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            results['warnings'].append("No numeric columns found")
        
        return results
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues."""
        results = {
            'issues': [],
            'warnings': []
        }
        
        # Check for missing values
        missing_stats = data.isnull().sum()
        high_missing = missing_stats[missing_stats > len(data) * self.missing_threshold]
        
        if not high_missing.empty:
            for col in high_missing.index:
                missing_pct = (high_missing[col] / len(data)) * 100
                if missing_pct > 50:
                    results['issues'].append(f"Column {col} has {missing_pct:.1f}% missing data")
                else:
                    results['warnings'].append(f"Column {col} has {missing_pct:.1f}% missing data")
        
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        inf_columns = inf_counts[inf_counts > 0]
        
        if not inf_columns.empty:
            for col in inf_columns.index:
                results['issues'].append(f"Column {col} has {inf_columns[col]} infinite values")
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            duplicate_pct = (duplicates / len(data)) * 100
            if duplicate_pct > 5:
                results['issues'].append(f"Data has {duplicate_pct:.1f}% duplicate rows")
            else:
                results['warnings'].append(f"Data has {duplicate_pct:.1f}% duplicate rows")
        
        return results
    
    def _validate_financial_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate financial data characteristics."""
        results = {
            'issues': [],
            'warnings': []
        }
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Check for extreme outliers
            z_scores = np.abs(stats.zscore(col_data))
            extreme_outliers = (z_scores > self.outlier_threshold).sum()
            
            if extreme_outliers > 0:
                outlier_pct = (extreme_outliers / len(col_data)) * 100
                if outlier_pct > 5:
                    results['issues'].append(f"Column {col} has {outlier_pct:.1f}% extreme outliers")
                else:
                    results['warnings'].append(f"Column {col} has {outlier_pct:.1f}% extreme outliers")
            
            # Check for zero variance
            if col_data.std() == 0:
                results['warnings'].append(f"Column {col} has zero variance")
            
            # Check for constant values
            unique_vals = col_data.nunique()
            if unique_vals == 1:
                results['warnings'].append(f"Column {col} has only one unique value")
            elif unique_vals < 10:
                results['warnings'].append(f"Column {col} has only {unique_vals} unique values")
        
        return results
    
    def _generate_summary_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the data."""
        stats = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'dtypes': data.dtypes.to_dict(),
            'missing_counts': data.isnull().sum().to_dict(),
            'numeric_summary': {}
        }
        
        # Numeric column summaries
        numeric_data = data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                stats['numeric_summary'][col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75)
                }
        
        return stats
    
    def clean_data(self, data: pd.DataFrame, 
                   strategy: str = 'conservative') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean financial data based on validation results.
        
        Args:
            data: Input data
            strategy: Cleaning strategy ('conservative', 'aggressive', 'minimal')
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        cleaning_report = {
            'original_shape': data.shape,
            'cleaned_shape': None,
            'removed_rows': 0,
            'removed_columns': 0,
            'imputed_values': 0,
            'outliers_handled': 0
        }
        
        cleaned_data = data.copy()
        
        try:
            # Handle missing values
            if strategy in ['aggressive', 'conservative']:
                # Remove rows with too many missing values
                max_missing = 0.5 if strategy == 'aggressive' else 0.8
                missing_threshold = len(data.columns) * max_missing
                
                rows_to_remove = data.isnull().sum(axis=1) > missing_threshold
                if rows_to_remove.any():
                    cleaned_data = cleaned_data[~rows_to_remove]
                    cleaning_report['removed_rows'] = rows_to_remove.sum()
                    self.logger.info(f"Removed {cleaning_report['removed_rows']} rows with excessive missing data")
                
                # Impute remaining missing values
                if strategy == 'aggressive':
                    # Forward fill for time series data
                    cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
                else:
                    # Only forward fill for numeric columns
                    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                    cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(method='ffill')
            
            # Handle outliers
            if strategy == 'aggressive':
                numeric_data = cleaned_data.select_dtypes(include=[np.number])
                for col in numeric_data.columns:
                    col_data = numeric_data[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = (col_data < lower_bound) | (col_data > upper_bound)
                        if outliers.any():
                            # Cap outliers instead of removing
                            cleaned_data.loc[col_data[col_data < lower_bound].index, col] = lower_bound
                            cleaned_data.loc[col_data[col_data > upper_bound].index, col] = upper_bound
                            cleaning_report['outliers_handled'] += outliers.sum()
            
            # Remove constant columns
            if strategy in ['aggressive', 'conservative']:
                constant_cols = []
                for col in cleaned_data.columns:
                    if cleaned_data[col].nunique() <= 1:
                        constant_cols.append(col)
                
                if constant_cols:
                    cleaned_data = cleaned_data.drop(columns=constant_cols)
                    cleaning_report['removed_columns'] = len(constant_cols)
                    self.logger.info(f"Removed {len(constant_cols)} constant columns")
            
            # Final cleanup
            cleaned_data = cleaned_data.dropna(how='all')  # Remove completely empty rows
            
            cleaning_report['cleaned_shape'] = cleaned_data.shape
            
            self.logger.info(f"Data cleaning completed. Shape: {cleaning_report['original_shape']} -> {cleaning_report['cleaned_shape']}")
            
            return cleaned_data, cleaning_report
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            return data, cleaning_report
    
    def validate_returns_data(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Validate financial returns data specifically.
        
        Args:
            returns: Returns series
            
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) == 0:
                results['is_valid'] = False
                results['issues'].append("No valid returns data")
                return results
            
            # Check for extreme returns
            extreme_returns = np.abs(returns_clean) > self.volatility_threshold
            if extreme_returns.any():
                extreme_pct = (extreme_returns.sum() / len(returns_clean)) * 100
                if extreme_pct > 1:
                    results['issues'].append(f"Returns have {extreme_pct:.1f}% extreme values (>50%)")
                else:
                    results['warnings'].append(f"Returns have {extreme_pct:.1f}% extreme values (>50%)")
            
            # Check for unrealistic returns
            unrealistic_returns = returns_clean < -0.5  # -50% daily return
            if unrealistic_returns.any():
                results['warnings'].append(f"Found {unrealistic_returns.sum()} returns below -50%")
            
            # Generate returns statistics
            results['stats'] = {
                'count': len(returns_clean),
                'mean': returns_clean.mean(),
                'std': returns_clean.std(),
                'skewness': returns_clean.skew(),
                'kurtosis': returns_clean.kurtosis(),
                'min': returns_clean.min(),
                'max': returns_clean.max()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Returns validation failed: {e}")
            results['is_valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")
            return results
