"""
Unit tests for WalkForwardValidator and related components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from risk_pipeline.core.validator import WalkForwardValidator, ValidationConfig

class TestValidationConfig(unittest.TestCase):
    """Test ValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        self.assertEqual(config.n_splits, 5)
        self.assertEqual(config.test_size, 252)
        self.assertEqual(config.min_train_size, 60)
        self.assertEqual(config.min_test_size, 20)
        self.assertEqual(config.gap, 0)
        self.assertTrue(config.expanding_window)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            n_splits=3,
            test_size=100,
            min_train_size=50,
            min_test_size=15,
            gap=5,
            expanding_window=False
        )
        
        self.assertEqual(config.n_splits, 3)
        self.assertEqual(config.test_size, 100)
        self.assertEqual(config.min_train_size, 50)
        self.assertEqual(config.min_test_size, 15)
        self.assertEqual(config.gap, 5)
        self.assertFalse(config.expanding_window)
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ValidationConfig(
            n_splits=5,
            test_size=100,
            min_train_size=50,
            min_test_size=20
        )
        
        self.assertTrue(config.validate())
    
    def test_validate_invalid_n_splits(self):
        """Test validation with invalid n_splits."""
        config = ValidationConfig(n_splits=0)
        
        self.assertFalse(config.validate())
    
    def test_validate_invalid_test_size(self):
        """Test validation with invalid test_size."""
        config = ValidationConfig(test_size=10, min_test_size=20)
        
        self.assertFalse(config.validate())
    
    def test_validate_invalid_min_train_size(self):
        """Test validation with invalid min_train_size."""
        config = ValidationConfig(min_train_size=5)
        
        self.assertFalse(config.validate())

class TestWalkForwardValidator(unittest.TestCase):
    """Test WalkForwardValidator class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        self.sample_data = pd.DataFrame({
            'feature1': np.random.randn(500).cumsum(),
            'feature2': np.random.randn(500).cumsum(),
            'feature3': np.random.randn(500).cumsum()
        }, index=dates)
        
        self.validator = WalkForwardValidator(
            n_splits=3,
            test_size=50,
            min_train_size=100,
            min_test_size=20
        )
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator.config)
        self.assertEqual(self.validator.config.n_splits, 3)
        self.assertEqual(self.validator.config.test_size, 50)
        self.assertEqual(self.validator.config.min_train_size, 100)
        self.assertEqual(self.validator.config.min_test_size, 20)
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid configuration."""
        with self.assertRaises(ValueError):
            WalkForwardValidator(n_splits=0)
    
    def test_split_basic(self):
        """Test basic split generation."""
        splits = self.validator.split(self.sample_data)
        
        self.assertIsInstance(splits, list)
        self.assertGreater(len(splits), 0)
        
        for train_idx, test_idx in splits:
            self.assertIsInstance(train_idx, pd.Index)
            self.assertIsInstance(test_idx, pd.Index)
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
    
    def test_split_small_dataset(self):
        """Test split generation with small dataset."""
        small_data = self.sample_data.iloc[:50]
        validator = WalkForwardValidator(n_splits=5, test_size=20, min_train_size=10, min_test_size=5)
        
        splits = validator.split(small_data)
        
        # Should still generate some splits
        self.assertIsInstance(splits, list)
        self.assertGreater(len(splits), 0)
    
    def test_split_very_small_dataset(self):
        """Test split generation with very small dataset."""
        tiny_data = self.sample_data.iloc[:10]
        validator = WalkForwardValidator(n_splits=5, test_size=20, min_train_size=10, min_test_size=5)
        
        splits = validator.split(tiny_data)
        
        # Should return empty list for insufficient data
        self.assertEqual(len(splits), 0)
    
    def test_split_expanding_window(self):
        """Test expanding window splits."""
        validator = WalkForwardValidator(
            n_splits=3,
            test_size=30,
            min_train_size=50,
            min_test_size=10,
            expanding_window=True
        )
        
        splits = validator.split(self.sample_data)
        
        self.assertGreater(len(splits), 0)
        
        # Check that train sets are expanding
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        for i in range(1, len(train_sizes)):
            self.assertGreaterEqual(train_sizes[i], train_sizes[i-1])
    
    def test_split_sliding_window(self):
        """Test sliding window splits."""
        validator = WalkForwardValidator(
            n_splits=3,
            test_size=30,
            min_train_size=50,
            min_test_size=10,
            expanding_window=False
        )
        
        splits = validator.split(self.sample_data)
        
        self.assertGreater(len(splits), 0)
    
    def test_split_with_gap(self):
        """Test splits with gap between train and test."""
        validator = WalkForwardValidator(
            n_splits=2,
            test_size=30,
            min_train_size=50,
            min_test_size=10,
            gap=10
        )
        
        splits = validator.split(self.sample_data)
        
        self.assertGreater(len(splits), 0)
        
        # Check that there's a gap between train and test sets
        for train_idx, test_idx in splits:
            train_end = train_idx[-1]
            test_start = test_idx[0]
            
            # Find the indices in the original data
            train_end_pos = self.sample_data.index.get_loc(train_end)
            test_start_pos = self.sample_data.index.get_loc(test_start)
            
            # Should have at least the gap size between them
            self.assertGreaterEqual(test_start_pos - train_end_pos, 10)
    
    def test_validate_split_indices(self):
        """Test split index validation."""
        # Valid indices
        self.assertTrue(self.validator._validate_split_indices(100, 150, 200, 500))
        
        # Invalid indices
        self.assertFalse(self.validator._validate_split_indices(200, 150, 200, 500))  # test_start < train_end
        self.assertFalse(self.validator._validate_split_indices(100, 150, 150, 500))  # test_end <= test_start
        self.assertFalse(self.validator._validate_split_indices(100, 150, 600, 500))  # test_end > n_samples
    
    def test_validate_split_sizes(self):
        """Test split size validation."""
        # Valid sizes
        train_idx = pd.Index(range(100))
        test_idx = pd.Index(range(100, 130))
        self.assertTrue(self.validator._validate_split_sizes(train_idx, test_idx))
        
        # Invalid sizes
        small_train_idx = pd.Index(range(50))  # Less than min_train_size
        small_test_idx = pd.Index(range(100, 110))  # Less than min_test_size
        self.assertFalse(self.validator._validate_split_sizes(small_train_idx, test_idx))
        self.assertFalse(self.validator._validate_split_sizes(train_idx, small_test_idx))
    
    def test_get_split_info(self):
        """Test getting split information."""
        splits = self.validator.split(self.sample_data)
        info = self.validator.get_split_info(splits)
        
        self.assertIsInstance(info, dict)
        self.assertIn('n_splits', info)
        self.assertIn('train_sizes', info)
        self.assertIn('test_sizes', info)
        self.assertIn('total_samples_used', info)
        self.assertIn('overlap', info)
        
        if len(splits) > 0:
            self.assertEqual(info['n_splits'], len(splits))
            self.assertGreater(info['total_samples_used'], 0)
    
    def test_get_split_info_empty_splits(self):
        """Test getting split information with empty splits."""
        info = self.validator.get_split_info([])
        
        self.assertEqual(info, {})
    
    def test_calculate_overlap(self):
        """Test overlap calculation."""
        # Create test splits
        train_idx1 = pd.Index(range(100))
        test_idx1 = pd.Index(range(100, 130))
        train_idx2 = pd.Index(range(50, 150))
        test_idx2 = pd.Index(range(150, 180))
        
        splits = [(train_idx1, test_idx1), (train_idx2, test_idx2)]
        overlap = self.validator._calculate_overlap(splits)
        
        self.assertIn('train_overlap', overlap)
        self.assertIn('test_overlap', overlap)
        
        # Train overlap should be high for expanding window
        self.assertGreater(overlap['train_overlap']['mean'], 0)
        # Test overlap should be 0 for proper validation
        self.assertEqual(overlap['test_overlap']['mean'], 0)
    
    def test_calculate_overlap_single_split(self):
        """Test overlap calculation with single split."""
        train_idx = pd.Index(range(100))
        test_idx = pd.Index(range(100, 130))
        splits = [(train_idx, test_idx)]
        
        overlap = self.validator._calculate_overlap(splits)
        
        self.assertEqual(overlap['train_overlap'], 0)
        self.assertEqual(overlap['test_overlap'], 0)
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        quality_report = self.validator.validate_data_quality(self.sample_data)
        
        self.assertIsInstance(quality_report, dict)
        self.assertIn('n_samples', quality_report)
        self.assertIn('n_features', quality_report)
        self.assertIn('missing_values', quality_report)
        self.assertIn('duplicate_indices', quality_report)
        self.assertIn('index_type', quality_report)
        self.assertIn('is_sorted', quality_report)
        self.assertIn('date_range', quality_report)
        self.assertIn('issues', quality_report)
        self.assertIn('is_valid', quality_report)
        
        self.assertEqual(quality_report['n_samples'], len(self.sample_data))
        self.assertEqual(quality_report['n_features'], len(self.sample_data.columns))
    
    def test_validate_data_quality_with_target(self):
        """Test data quality validation with target variable."""
        target = pd.Series(np.random.randn(500), index=self.sample_data.index)
        quality_report = self.validator.validate_data_quality(self.sample_data, target)
        
        self.assertIn('target_missing', quality_report)
        self.assertIn('target_unique', quality_report)
        self.assertIn('target_distribution', quality_report)
    
    def test_validate_data_quality_with_issues(self):
        """Test data quality validation with data issues."""
        # Create data with issues
        problematic_data = self.sample_data.copy()
        problematic_data.iloc[0, 0] = np.nan  # Missing value
        problematic_data.index = problematic_data.index.repeat(2)[::2]  # Duplicate indices
        
        quality_report = self.validator.validate_data_quality(problematic_data)
        
        self.assertGreater(len(quality_report['issues']), 0)
        self.assertFalse(quality_report['is_valid'])
    
    def test_create_time_series_split(self):
        """Test time series split iterator."""
        target = pd.Series(np.random.randn(500), index=self.sample_data.index)
        
        split_iterator = self.validator.create_time_series_split(self.sample_data, target)
        
        for X_train, X_test, y_train, y_test in split_iterator:
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(X_test, np.ndarray)
            self.assertIsInstance(y_train, np.ndarray)
            self.assertIsInstance(y_test, np.ndarray)
            
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(X_train.shape[1], X_test.shape[1])
    
    def test_create_time_series_split_no_target(self):
        """Test time series split iterator without target."""
        split_iterator = self.validator.create_time_series_split(self.sample_data)
        
        for X_train, X_test, y_train, y_test in split_iterator:
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(X_test, np.ndarray)
            self.assertIsNone(y_train)
            self.assertIsNone(y_test)
    
    def test_get_validation_summary(self):
        """Test getting comprehensive validation summary."""
        target = pd.Series(np.random.randn(500), index=self.sample_data.index)
        summary = self.validator.get_validation_summary(self.sample_data, target)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('configuration', summary)
        self.assertIn('actual_splits', summary)
        self.assertIn('split_info', summary)
        self.assertIn('data_quality', summary)
        self.assertIn('validation_status', summary)
        
        self.assertIsInstance(summary['configuration'], dict)
        self.assertIsInstance(summary['split_info'], dict)
        self.assertIsInstance(summary['data_quality'], dict)
        self.assertIn(summary['validation_status'], ['valid', 'invalid'])
    
    def test_plot_splits(self):
        """Test split visualization."""
        splits = self.validator.split(self.sample_data)
        
        # Test without saving
        self.validator.plot_splits(self.sample_data, splits)
        
        # Test with saving
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                self.validator.plot_splits(self.sample_data, splits, save_path=tmp_file.name)
                self.assertTrue(os.path.exists(tmp_file.name))
            finally:
                os.unlink(tmp_file.name)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.validator)
        
        self.assertIsInstance(repr_str, str)
        self.assertIn('WalkForwardValidator', repr_str)
        self.assertIn('n_splits=3', repr_str)
        self.assertIn('test_size=50', repr_str)
    
    def test_adaptive_test_size(self):
        """Test adaptive test size calculation."""
        # Test with dataset smaller than requested test size
        small_data = self.sample_data.iloc[:100]
        validator = WalkForwardValidator(n_splits=2, test_size=80, min_train_size=20, min_test_size=10)
        
        splits = validator.split(small_data)
        
        # Should still generate splits with adapted test size
        self.assertGreater(len(splits), 0)
        
        for _, test_idx in splits:
            # Test size should be at least min_test_size
            self.assertGreaterEqual(len(test_idx), 10)
    
    def test_max_possible_splits(self):
        """Test maximum possible splits calculation."""
        # Test with dataset that can't support requested number of splits
        small_data = self.sample_data.iloc[:80]
        validator = WalkForwardValidator(n_splits=10, test_size=20, min_train_size=30, min_test_size=10)
        
        splits = validator.split(small_data)
        
        # Should generate fewer splits than requested
        self.assertLess(len(splits), 10)
        self.assertGreater(len(splits), 0)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with exactly minimum required data
        min_data = self.sample_data.iloc[:70]  # min_train_size + min_test_size
        validator = WalkForwardValidator(n_splits=1, test_size=20, min_train_size=50, min_test_size=20)
        
        splits = validator.split(min_data)
        self.assertEqual(len(splits), 1)
        
        # Test with data just below minimum
        too_small_data = self.sample_data.iloc[:60]
        splits = validator.split(too_small_data)
        self.assertEqual(len(splits), 0)

if __name__ == '__main__':
    unittest.main() 