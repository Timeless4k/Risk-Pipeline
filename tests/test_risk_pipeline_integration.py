"""
Integration tests for RiskPipeline orchestrator.

Tests the complete pipeline integration including experiment management,
SHAP analysis, model persistence, and all component interactions.
"""

import unittest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
from risk_pipeline import RiskPipeline, PipelineConfig
from risk_pipeline.core.results_manager import ResultsManager
from risk_pipeline.utils.model_persistence import ModelPersistence
from risk_pipeline.utils.experiment_tracking import ExperimentTracker


class TestRiskPipelineIntegration(unittest.TestCase):
    """Test complete RiskPipeline integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "test_config.json"
        
        # Create test configuration
        test_config = {
            "data": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "all_assets": ["AAPL", "MSFT"],
                "cache_dir": str(self.test_dir / "cache")
            },
            "training": {
                "walk_forward_splits": 2,
                "test_size": 50,
                "random_state": 42
            },
            "output": {
                "results_dir": str(self.test_dir / "results"),
                "models_dir": str(self.test_dir / "models"),
                "plots_dir": str(self.test_dir / "plots"),
                "shap_dir": str(self.test_dir / "shap"),
                "shap_plots_dir": str(self.test_dir / "shap_plots"),
                "log_dir": str(self.test_dir / "logs")
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Create directories
        for dir_path in test_config["output"].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('risk_pipeline.core.data_loader.DataLoader.download_data')
    @patch('risk_pipeline.core.feature_engineer.FeatureEngineer.create_features')
    @patch('risk_pipeline.models.model_factory.ModelFactory.create_model')
    @patch('risk_pipeline.core.validator.WalkForwardValidator.evaluate_model')
    def test_run_complete_pipeline(self, mock_evaluate, mock_create_model, 
                                  mock_create_features, mock_download_data):
        """Test complete pipeline execution."""
        # Mock data loading
        mock_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=pd.date_range('2023-01-01', periods=100, freq='D')),
            'MSFT': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 200,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        }
        mock_download_data.return_value = mock_data
        
        # Mock feature engineering
        mock_features = {
            'AAPL': {
                'features': np.random.randn(50, 10),
                'volatility_target': np.random.randn(50),
                'regime_target': np.random.randint(0, 2, 50),
                'feature_names': [f'feature_{i}' for i in range(10)],
                'scaler': Mock()
            },
            'MSFT': {
                'features': np.random.randn(50, 10),
                'volatility_target': np.random.randn(50),
                'regime_target': np.random.randint(0, 2, 50),
                'feature_names': [f'feature_{i}' for i in range(10)],
                'scaler': Mock()
            }
        }
        mock_create_features.return_value = mock_features
        
        # Mock model creation
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        # Mock model evaluation
        mock_evaluate.return_value = {
            'metrics': {'R2': 0.8, 'MAE': 0.1},
            'predictions': np.random.randn(50),
            'actuals': np.random.randn(50),
            'config': {'param1': 'value1'}
        }
        
        # Initialize pipeline
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_integration"
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            assets=['AAPL', 'MSFT'],
            models=['xgboost', 'lstm'],
            save_models=True,
            run_shap=True
        )
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('AAPL', results)
        self.assertIn('MSFT', results)
        
        # Verify experiment was created
        self.assertTrue((self.test_dir / "experiments").exists())
        experiment_dirs = list((self.test_dir / "experiments").glob("test_integration_*"))
        self.assertGreater(len(experiment_dirs), 0)
    
    def test_run_quick_test(self):
        """Test quick test mode."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_quick"
        )
        
        # Mock the complete pipeline method
        with patch.object(pipeline, 'run_complete_pipeline') as mock_run:
            mock_run.return_value = {'AAPL': {'regression': {'xgboost': {'R2': 0.7}}}}
            
            results = pipeline.run_quick_test()
            
            # Verify quick test was called with correct parameters
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            self.assertEqual(call_args[1]['models'], ['xgboost', 'lstm'])
            self.assertFalse(call_args[1]['save_models'])
            self.assertFalse(call_args[1]['run_shap'])
    
    def test_train_models_only(self):
        """Test models-only training."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_models_only"
        )
        
        # Mock data loading and feature engineering
        with patch.object(pipeline.data_loader, 'download_data') as mock_download:
            with patch.object(pipeline.feature_engineer, 'create_features') as mock_features:
                with patch.object(pipeline, '_run_models') as mock_run_models:
                    mock_download.return_value = {'AAPL': pd.DataFrame()}
                    mock_features.return_value = {
                        'AAPL': {
                            'features': np.random.randn(50, 10),
                            'volatility_target': np.random.randn(50)
                        }
                    }
                    mock_run_models.return_value = {'xgboost': {'R2': 0.8}}
                    
                    results = pipeline.train_models_only(
                        assets=['AAPL'],
                        models=['xgboost'],
                        save=True
                    )
                    
                    self.assertIn('AAPL', results)
                    self.assertIn('regression', results['AAPL'])
    
    def test_analyze_saved_models(self):
        """Test analysis of saved models."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_analysis"
        )
        
        # Create a mock experiment
        experiment_id = "test_experiment_123"
        experiment_dir = self.test_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock experiment data
        experiment_data = {
            'config': {'test': 'config'},
            'metadata': {'test': 'metadata'},
            'summary': pd.DataFrame({
                'asset': ['AAPL'],
                'model': ['xgboost'],
                'task': ['regression'],
                'R2': [0.8]
            })
        }
        
        with patch.object(pipeline.results_manager, 'load_experiment') as mock_load:
            mock_load.return_value = experiment_data
            
            results = pipeline.analyze_saved_models(
                experiment_id=experiment_id,
                run_additional_shap=False
            )
            
            self.assertIsInstance(results, dict)
    
    def test_compare_experiments(self):
        """Test experiment comparison."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_comparison"
        )
        
        experiment_ids = ['exp1', 'exp2']
        
        # Mock experiment loading
        with patch.object(pipeline.results_manager, 'load_experiment') as mock_load:
            mock_load.side_effect = [
                {'config': {'exp1': 'config'}, 'summary': pd.DataFrame()},
                {'config': {'exp2': 'config'}, 'summary': pd.DataFrame()}
            ]
            
            results = pipeline.compare_experiments(experiment_ids)
            
            self.assertIsInstance(results, dict)
            self.assertIn('exp1', results)
            self.assertIn('exp2', results)
    
    def test_get_best_models(self):
        """Test getting best models."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_best_models"
        )
        
        # Mock best models retrieval
        mock_best_models = pd.DataFrame({
            'experiment_id': ['exp1', 'exp2'],
            'asset': ['AAPL', 'MSFT'],
            'model': ['xgboost', 'lstm'],
            'R2': [0.8, 0.9]
        })
        
        with patch.object(pipeline.results_manager, 'get_best_models') as mock_get:
            mock_get.return_value = mock_best_models
            
            results = pipeline.get_best_models(metric='R2', task='regression')
            
            self.assertIsInstance(results, pd.DataFrame)
            self.assertEqual(len(results), 2)
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_memory"
        )
        
        # Test memory tracking
        pipeline._track_memory_usage("test_stage")
        
        self.assertIsInstance(pipeline.memory_usage, list)
        self.assertGreater(len(pipeline.memory_usage), 0)
        self.assertIsInstance(pipeline.memory_usage[0], (int, float))
    
    def test_backward_compatibility(self):
        """Test backward compatibility with original interface."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_backward"
        )
        
        # Mock the complete pipeline method
        with patch.object(pipeline, 'run_complete_pipeline') as mock_run:
            mock_run.return_value = {'AAPL': {'regression': {'xgboost': {'R2': 0.7}}}}
            
            # Test original run_pipeline method
            results = pipeline.run_pipeline(
                assets=['AAPL'],
                skip_correlations=False,
                debug=False
            )
            
            # Verify it calls the new method with correct defaults
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            self.assertEqual(call_args[1]['assets'], ['AAPL'])
            self.assertTrue(call_args[1]['save_models'])
            self.assertTrue(call_args[1]['run_shap'])
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_error"
        )
        
        # Mock data loading to raise an exception
        with patch.object(pipeline.data_loader, 'download_data') as mock_download:
            mock_download.side_effect = Exception("Data download failed")
            
            # Test that exception is properly raised
            with self.assertRaises(Exception):
                pipeline.run_complete_pipeline(
                    assets=['AAPL'],
                    models=['xgboost']
                )
            
            # Verify memory tracking still works
            self.assertIsInstance(pipeline.memory_usage, list)
    
    def test_experiment_metadata_saving(self):
        """Test experiment metadata saving."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_metadata"
        )
        
        # Mock the complete pipeline execution
        with patch.object(pipeline, 'run_complete_pipeline') as mock_run:
            mock_run.return_value = {'AAPL': {'regression': {'xgboost': {'R2': 0.7}}}}
            
            # Mock results manager methods
            with patch.object(pipeline.results_manager, 'start_experiment') as mock_start:
                with patch.object(pipeline.results_manager, 'save_experiment_metadata') as mock_save:
                    mock_start.return_value = "test_experiment_id"
                    
                    pipeline.run_complete_pipeline(
                        assets=['AAPL'],
                        models=['xgboost'],
                        save_models=True,
                        run_shap=True
                    )
                    
                    # Verify experiment was started and metadata was saved
                    mock_start.assert_called_once()
                    mock_save.assert_called_once()
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction from SHAP results."""
        pipeline = RiskPipeline(
            config_path=str(self.config_path),
            experiment_name="test_feature_importance"
        )
        
        # Mock SHAP results
        mock_shap_results = {
            'xgboost': {
                'shap_values': np.random.randn(50, 10),
                'feature_names': [f'feature_{i}' for i in range(10)]
            }
        }
        
        feature_importance = pipeline._extract_feature_importance(mock_shap_results)
        
        self.assertIsInstance(feature_importance, dict)
        self.assertEqual(len(feature_importance), 10)
        self.assertTrue(all(isinstance(v, float) for v in feature_importance.values()))


class TestResultsManagerIntegration(unittest.TestCase):
    """Test ResultsManager integration with RiskPipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.results_manager = ResultsManager(base_dir=str(self.test_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        # Start experiment
        config = {'test': 'config'}
        experiment_id = self.results_manager.start_experiment(
            name="test_experiment",
            config=config,
            description="Test experiment"
        )
        
        self.assertIsInstance(experiment_id, str)
        self.assertTrue(experiment_id.startswith("experiment_"))
        
        # Save model results
        mock_model = Mock()
        mock_scaler = Mock()
        metrics = {'R2': 0.8, 'MAE': 0.1}
        predictions = {'actual': [1, 2, 3], 'predicted': [1.1, 1.9, 3.1]}
        
        self.results_manager.save_model_results(
            asset='AAPL',
            model_name='xgboost',
            task='regression',
            metrics=metrics,
            predictions=predictions,
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['feature1', 'feature2'],
            config={'param1': 'value1'}
        )
        
        # Save SHAP results
        shap_values = np.random.randn(100, 5)
        explainer_metadata = {'background_samples': 100}
        feature_importance = {'feature1': 0.5, 'feature2': 0.3}
        
        self.results_manager.save_shap_results(
            asset='AAPL',
            model_name='xgboost',
            shap_values=shap_values,
            explainer_metadata=explainer_metadata,
            feature_importance=feature_importance
        )
        
        # Save experiment metadata
        self.results_manager.save_experiment_metadata({
            'assets_processed': 1,
            'models_run': ['xgboost'],
            'execution_time_minutes': 5.0
        })
        
        # Load experiment
        experiment_data = self.results_manager.load_experiment(experiment_id)
        
        self.assertIsInstance(experiment_data, dict)
        self.assertIn('config', experiment_data)
        self.assertIn('metadata', experiment_data)
        self.assertIn('summary', experiment_data)
    
    def test_get_best_models(self):
        """Test getting best models across experiments."""
        # Create multiple experiments with different performance
        for i in range(3):
            experiment_id = self.results_manager.start_experiment(
                name=f"test_experiment_{i}",
                config={'test': 'config'},
                description=f"Test experiment {i}"
            )
            
            # Save model results with different performance
            self.results_manager.save_model_results(
                asset='AAPL',
                model_name='xgboost',
                task='regression',
                metrics={'R2': 0.7 + i * 0.1},  # Increasing R2 scores
                predictions={'actual': [1], 'predicted': [1]},
                model=Mock(),
                scaler=Mock(),
                feature_names=[],
                config={}
            )
            
            self.results_manager.save_experiment_metadata({})
        
        # Get best models
        best_models = self.results_manager.get_best_models(metric='R2', task='regression')
        
        self.assertIsInstance(best_models, pd.DataFrame)
        self.assertGreater(len(best_models), 0)
        
        # Verify best model has highest R2
        if len(best_models) > 0:
            best_r2 = best_models['R2'].max()
            self.assertGreaterEqual(best_r2, 0.9)  # Should be 0.9 from last experiment


class TestModelPersistenceIntegration(unittest.TestCase):
    """Test ModelPersistence integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_model_save_load(self):
        """Test saving and loading complete model artifacts."""
        # Create mock model and components
        mock_model = Mock()
        mock_scaler = Mock()
        feature_names = ['feature1', 'feature2', 'feature3']
        config = {'param1': 'value1', 'param2': 42}
        metrics = {'R2': 0.8, 'MAE': 0.1}
        
        model_dir = self.test_dir / "test_model"
        
        # Save complete model
        ModelPersistence.save_complete_model(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=feature_names,
            config=config,
            metrics=metrics,
            filepath=model_dir
        )
        
        # Verify files were created
        self.assertTrue((model_dir / 'model.pkl').exists())
        self.assertTrue((model_dir / 'scaler.pkl').exists())
        self.assertTrue((model_dir / 'feature_names.json').exists())
        self.assertTrue((model_dir / 'config.json').exists())
        self.assertTrue((model_dir / 'metrics.json').exists())
        self.assertTrue((model_dir / 'metadata.json').exists())
        
        # Load complete model
        loaded_model, loaded_scaler, loaded_features, loaded_config, loaded_metrics, loaded_metadata = \
            ModelPersistence.load_complete_model(model_dir)
        
        # Verify loaded components
        self.assertEqual(loaded_model, mock_model)
        self.assertEqual(loaded_scaler, mock_scaler)
        self.assertEqual(loaded_features, feature_names)
        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_metrics, metrics)
        self.assertIsInstance(loaded_metadata, dict)
    
    def test_model_integrity_verification(self):
        """Test model integrity verification."""
        # Create a valid model directory
        model_dir = self.test_dir / "valid_model"
        model_dir.mkdir()
        
        # Create mock files
        (model_dir / 'model.pkl').touch()
        (model_dir / 'feature_names.json').write_text('["feature1", "feature2"]')
        (model_dir / 'config.json').write_text('{"test": "config"}')
        (model_dir / 'metrics.json').write_text('{"R2": 0.8}')
        (model_dir / 'metadata.json').write_text('{"test": "metadata"}')
        
        # Test integrity verification
        with patch('risk_pipeline.utils.model_persistence.joblib.load') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([1, 2, 3])
            mock_load.return_value = mock_model
            
            is_valid = ModelPersistence.verify_model_integrity(model_dir)
            self.assertTrue(is_valid)
    
    def test_experiment_config_saving(self):
        """Test experiment configuration saving."""
        config = {'data': {'assets': ['AAPL']}, 'training': {'splits': 5}}
        data_info = {'version': '1.0', 'samples': 1000}
        experiment_path = self.test_dir / "experiment"
        
        ModelPersistence.save_experiment_config(config, data_info, experiment_path)
        
        # Verify config files were created
        self.assertTrue((experiment_path / 'config.json').exists())
        self.assertTrue((experiment_path / 'data_info.json').exists())
        
        # Verify content
        with open(experiment_path / 'config.json', 'r') as f:
            loaded_config = json.load(f)
        self.assertEqual(loaded_config, config)
        
        with open(experiment_path / 'data_info.json', 'r') as f:
            loaded_data_info = json.load(f)
        self.assertEqual(loaded_data_info, data_info)


if __name__ == '__main__':
    unittest.main() 