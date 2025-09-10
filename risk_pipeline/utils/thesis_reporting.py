"""
Thesis reporting utilities for comprehensive model analysis and visualization.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ThesisReporter:
    """Generate comprehensive thesis reports from pipeline results."""
    
    def __init__(self, results_manager):
        self.results_manager = results_manager
    
    def generate_comprehensive_report(self, output_dir: str = None) -> str:
        """
        Generate a comprehensive thesis report with all analysis components.
        
        Args:
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("thesis_reports", f"comprehensive_report_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export all data first
        data_dir = self.results_manager.export_thesis_data()
        
        # Generate comprehensive analysis
        self._generate_model_comparison_report(output_dir, data_dir)
        self._generate_performance_analysis_report(output_dir, data_dir)
        self._generate_feature_importance_report(output_dir, data_dir)
        self._generate_statistical_significance_report(output_dir, data_dir)
        self._generate_visualization_guide(output_dir, data_dir)
        self._generate_executive_summary(output_dir, data_dir)
        
        logger.info(f"🎓 Comprehensive thesis report generated: {output_dir}")
        return output_dir
    
    def _generate_model_comparison_report(self, report_dir: str, data_dir: str):
        """Generate model comparison analysis."""
        try:
            # Load aggregated metrics
            metrics_path = os.path.join(data_dir, 'all_model_metrics.csv')
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Create comprehensive comparison tables
                comparison_data = []
                
                # Normalize common metric column aliases
                col_aliases = {
                    'mse': ['mse', 'MSE', 'mean_squared_error'],
                    'rmse': ['rmse', 'RMSE'],
                    'mae': ['mae', 'MAE'],
                    'R2': ['R2', 'r2', 'r2_score'],
                    'mape': ['mape', 'MAPE'],
                    'baseline_mse': ['baseline_mse', 'baseline_MSE']
                }
                def getv(row, key, default=0):
                    for k in col_aliases.get(key, [key]):
                        if k in row and pd.notna(row[k]):
                            return row[k]
                    return default

                for _, row in df.iterrows():
                    asset = row['asset']
                    model = row['model_type']
                    task = row['task']
                    
                    if task == 'classification':
                        comparison_data.append({
                            'Asset': asset,
                            'Model': model,
                            'Task': task,
                            'Accuracy': f"{row.get('accuracy', 0):.4f} ± {row.get('accuracy_std', 0):.4f}",
                            'F1-Score': f"{row.get('f1', 0):.4f} ± {row.get('f1_std', 0):.4f}",
                            'Precision': f"{row.get('precision', 0):.4f} ± {row.get('precision_std', 0):.4f}",
                            'Recall': f"{row.get('recall', 0):.4f} ± {row.get('recall_std', 0):.4f}",
                            'Fit_Time_s': f"{row.get('fit_time', 0):.3f} ± {row.get('fit_time_std', 0):.3f}",
                            'Pred_Time_s': f"{row.get('pred_time', 0):.3f} ± {row.get('pred_time_std', 0):.3f}"
                        })
                    else:  # regression
                        comparison_data.append({
                            'Asset': asset,
                            'Model': model,
                            'Task': task,
                            'MSE': f"{getv(row, 'mse', 0):.6f} ± {row.get('mse_std', 0):.6f}",
                            'RMSE': f"{getv(row, 'rmse', 0):.6f} ± {row.get('rmse_std', 0):.6f}",
                            'MAE': f"{getv(row, 'mae', 0):.6f} ± {row.get('mae_std', 0):.6f}",
                            'R²': f"{getv(row, 'R2', 0):.4f} ± {row.get('R2_std', 0):.4f}",
                            'MAPE': f"{getv(row, 'mape', 0):.4f} ± {row.get('mape_std', 0):.4f}",
                            'Baseline_MSE': f"{getv(row, 'baseline_mse', 0):.6f} ± {row.get('baseline_mse_std', 0):.6f}",
                            'Fit_Time_s': f"{row.get('fit_time', 0):.3f} ± {row.get('fit_time_std', 0):.3f}",
                            'Pred_Time_s': f"{row.get('pred_time', 0):.3f} ± {row.get('pred_time_std', 0):.3f}"
                        })
                
                # Save comparison tables
                comparison_df = pd.DataFrame(comparison_data)
                comparison_path = os.path.join(report_dir, 'model_comparison_table.csv')
                comparison_df.to_csv(comparison_path, index=False)
                
                # Generate LaTeX table
                latex_table = self._generate_latex_table(comparison_df, task)
                latex_path = os.path.join(report_dir, 'model_comparison_table.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)
                
                logger.info(f"Model comparison report generated: {comparison_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate model comparison report: {e}")
    
    def _generate_performance_analysis_report(self, report_dir: str, data_dir: str):
        """Generate detailed performance analysis."""
        try:
            # Load aggregated metrics
            metrics_path = os.path.join(data_dir, 'all_model_metrics.csv')
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Performance analysis
                performance_analysis = {
                    'timestamp': datetime.now().isoformat(),
                    'total_models': len(df),
                    'assets_analyzed': df['asset'].nunique(),
                    'model_types': df['model_type'].unique().tolist(),
                    'tasks': df['task'].unique().tolist(),
                    'performance_summary': {}
                }
                
                # Per-task performance analysis
                for task in df['task'].unique():
                    task_df = df[df['task'] == task]
                    if task == 'classification':
                        performance_analysis['performance_summary'][task] = {
                            'best_accuracy': {
                                'model': task_df.loc[task_df['accuracy'].idxmax(), 'model_type'],
                                'asset': task_df.loc[task_df['accuracy'].idxmax(), 'asset'],
                                'value': float(task_df['accuracy'].max())
                            },
                            'best_f1': {
                                'model': task_df.loc[task_df['f1'].idxmax(), 'model_type'],
                                'asset': task_df.loc[task_df['f1'].idxmax(), 'asset'],
                                'value': float(task_df['f1'].max())
                            },
                            'avg_accuracy': float(task_df['accuracy'].mean()),
                            'avg_f1': float(task_df['f1'].mean()),
                            'std_accuracy': float(task_df['accuracy'].std()),
                            'std_f1': float(task_df['f1'].std())
                        }
                    else:  # regression
                        # Column aliasing for robustness
                        def pick_col(df_in, candidates, fallback):
                            for c in candidates:
                                if c in df_in.columns:
                                    return c
                            return fallback
                        r2_col = pick_col(task_df, ['R2', 'r2', 'r2_score'], 'R2')
                        mse_col = pick_col(task_df, ['mse', 'MSE', 'mean_squared_error'], 'mse')

                        try:
                            best_r2_idx = task_df[r2_col].idxmax()
                        except Exception:
                            best_r2_idx = task_df.index[0]
                        try:
                            best_mse_idx = task_df[mse_col].idxmin()
                        except Exception:
                            best_mse_idx = task_df.index[0]

                        performance_analysis['performance_summary'][task] = {
                            'best_r2': {
                                'model': task_df.loc[best_r2_idx, 'model_type'] if len(task_df) else None,
                                'asset': task_df.loc[best_r2_idx, 'asset'] if len(task_df) else None,
                                'value': float(task_df[r2_col].max()) if r2_col in task_df else 0.0
                            },
                            'best_mse': {
                                'model': task_df.loc[best_mse_idx, 'model_type'] if len(task_df) else None,
                                'asset': task_df.loc[best_mse_idx, 'asset'] if len(task_df) else None,
                                'value': float(task_df[mse_col].min()) if mse_col in task_df else 0.0
                            },
                            'avg_r2': float(task_df[r2_col].mean()) if r2_col in task_df else 0.0,
                            'avg_mse': float(task_df[mse_col].mean()) if mse_col in task_df else 0.0,
                            'std_r2': float(task_df[r2_col].std()) if r2_col in task_df else 0.0,
                            'std_mse': float(task_df[mse_col].std()) if mse_col in task_df else 0.0
                        }
                
                # Save performance analysis
                perf_path = os.path.join(report_dir, 'performance_analysis.json')
                with open(perf_path, 'w') as f:
                    json.dump(performance_analysis, f, indent=2, default=str)
                
                logger.info(f"Performance analysis report generated: {perf_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate performance analysis report: {e}")
    
    def _generate_feature_importance_report(self, report_dir: str, data_dir: str):
        """Generate feature importance analysis."""
        try:
            # This would analyze SHAP results if available
            feature_analysis = {
                'timestamp': datetime.now().isoformat(),
                'note': 'Feature importance analysis requires SHAP results to be available',
                'recommendations': [
                    'Use SHAP values to identify most important features per model',
                    'Compare feature importance across different model types',
                    'Analyze temporal stability of feature importance',
                    'Identify feature interactions and dependencies'
                ]
            }
            
            feature_path = os.path.join(report_dir, 'feature_importance_analysis.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_analysis, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to generate feature importance report: {e}")
    
    def _generate_statistical_significance_report(self, report_dir: str, data_dir: str):
        """Generate statistical significance analysis."""
        try:
            # Load per-fold metrics for statistical testing
            statistical_analysis = {
                'timestamp': datetime.now().isoformat(),
                'statistical_tests_recommended': [
                    'Paired t-test for model comparison within same asset/task',
                    'Wilcoxon signed-rank test for non-parametric comparison',
                    'ANOVA for comparing multiple models across assets',
                    'Correlation analysis between model performance and market conditions'
                ],
                'sample_size_analysis': {
                    'note': 'Per-fold metrics available for statistical testing',
                    'recommendations': [
                        'Use fold-level metrics for paired comparisons',
                        'Bootstrap confidence intervals for performance metrics',
                        'Cross-validation stability analysis',
                        'Temporal performance drift analysis'
                    ]
                }
            }
            
            stats_path = os.path.join(report_dir, 'statistical_analysis.json')
            with open(stats_path, 'w') as f:
                json.dump(statistical_analysis, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to generate statistical analysis report: {e}")
    
    def _generate_visualization_guide(self, report_dir: str, data_dir: str):
        """Generate visualization recommendations."""
        try:
            visualization_guide = {
                'timestamp': datetime.now().isoformat(),
                'recommended_plots': {
                    'performance_comparison': [
                        'Bar charts comparing metrics across models',
                        'Box plots showing distribution of fold-level metrics',
                        'Heatmaps for model-asset-task performance matrix',
                        'Time series plots showing performance evolution'
                    ],
                    'model_analysis': [
                        'Residual plots for regression models',
                        'Confusion matrices for classification models',
                        'ROC curves and precision-recall curves',
                        'Calibration plots for probability estimates'
                    ],
                    'feature_analysis': [
                        'SHAP summary plots',
                        'Feature importance bar charts',
                        'Partial dependence plots',
                        'Feature correlation heatmaps'
                    ],
                    'temporal_analysis': [
                        'Performance over time plots',
                        'Rolling window performance metrics',
                        'Market regime analysis',
                        'Volatility clustering visualization'
                    ]
                },
                'plotting_libraries': [
                    'matplotlib for basic plots',
                    'seaborn for statistical visualizations',
                    'plotly for interactive plots',
                    'SHAP for model interpretability plots'
                ],
                'data_sources': [
                    'per_fold_predictions.csv for residual analysis',
                    'fold_level_metrics.csv for performance distributions',
                    'model_metrics.json for aggregated comparisons',
                    'shap_results.json for feature importance'
                ]
            }
            
            viz_path = os.path.join(report_dir, 'visualization_guide.json')
            with open(viz_path, 'w') as f:
                json.dump(visualization_guide, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to generate visualization guide: {e}")
    
    def _generate_executive_summary(self, report_dir: str, data_dir: str):
        """Generate executive summary of all findings."""
        try:
            # Load performance analysis
            perf_path = os.path.join(report_dir, 'performance_analysis.json')
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    perf_analysis = json.load(f)
                
                executive_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'report_title': 'Comprehensive Risk Pipeline Model Analysis Report',
                    'executive_summary': {
                        'scope': f"Analysis of {perf_analysis.get('total_models', 0)} models across {perf_analysis.get('assets_analyzed', 0)} assets",
                        'key_findings': [],
                        'recommendations': [],
                        'next_steps': []
                    },
                    'technical_details': {
                        'models_analyzed': perf_analysis.get('model_types', []),
                        'tasks_performed': perf_analysis.get('tasks', []),
                        'data_export_location': data_dir,
                        'report_location': report_dir
                    }
                }
                
                # Add key findings based on performance
                if 'performance_summary' in perf_analysis:
                    for task, summary in perf_analysis['performance_summary'].items():
                        if task == 'classification':
                            executive_summary['executive_summary']['key_findings'].append(
                                f"Best classification model: {summary.get('best_accuracy', {}).get('model', 'Unknown')} "
                                f"on {summary.get('best_accuracy', {}).get('asset', 'Unknown')} "
                                f"(Accuracy: {summary.get('best_accuracy', {}).get('value', 0):.4f})"
                            )
                        else:
                            executive_summary['executive_summary']['key_findings'].append(
                                f"Best regression model: {summary.get('best_r2', {}).get('model', 'Unknown')} "
                                f"on {summary.get('best_r2', {}).get('asset', 'Unknown')} "
                                f"(R²: {summary.get('best_r2', {}).get('value', 0):.4f})"
                            )
                
                # Add recommendations
                executive_summary['executive_summary']['recommendations'] = [
                    "Use per-fold metrics for statistical significance testing",
                    "Implement ensemble methods combining best-performing models",
                    "Analyze feature importance for model interpretability",
                    "Consider temporal performance stability in model selection"
                ]
                
                # Add next steps
                executive_summary['executive_summary']['next_steps'] = [
                    "Generate recommended visualizations for thesis presentation",
                    "Perform statistical significance testing on model comparisons",
                    "Analyze feature importance and model interpretability",
                    "Investigate temporal performance patterns and market regime effects"
                ]
                
                # Save executive summary
                summary_path = os.path.join(report_dir, 'executive_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(executive_summary, f, indent=2, default=str)
                
                logger.info(f"Executive summary generated: {summary_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
    
    def _generate_latex_table(self, df: pd.DataFrame, task: str) -> str:
        """Generate LaTeX table for thesis."""
        if task == 'classification':
            columns = ['Asset', 'Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        else:
            columns = ['Asset', 'Model', 'MSE', 'RMSE', 'MAE', 'R²', 'MAPE']
        
        latex = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:model_comparison}\n"
        latex += "\\begin{tabular}{" + "l" * len(columns) + "}\n"
        latex += "\\toprule\n"
        latex += " & ".join(columns) + " \\\\\n"
        latex += "\\midrule\n"
        
        for _, row in df.iterrows():
            if task == 'classification':
                latex += f"{row['Asset']} & {row['Model']} & {row['Accuracy']} & {row['F1-Score']} & {row['Precision']} & {row['Recall']} \\\\\n"
            else:
                latex += f"{row['Asset']} & {row['Model']} & {row['MSE']} & {row['RMSE']} & {row['MAE']} & {row['R²']} & {row['MAPE']} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex


def create_thesis_report(results_manager, output_dir: str = None) -> str:
    """
    Convenience function to create a comprehensive thesis report.
    
    Args:
        results_manager: Results manager instance
        output_dir: Output directory for the report
        
    Returns:
        Path to the generated report directory
    """
    reporter = ThesisReporter(results_manager)
    return reporter.generate_comprehensive_report(output_dir)
