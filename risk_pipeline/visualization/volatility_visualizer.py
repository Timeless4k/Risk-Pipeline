"""
Volatility visualization components for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class VolatilityVisualizer:
    """Visualization class for volatility forecasting results."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize volatility visualizer.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('risk_pipeline.visualization.VolatilityVisualizer')
        self.logger.info(f"VolatilityVisualizer initialized with output_dir: {output_dir}")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_regression_performance(self, results_df: pd.DataFrame, 
                                  save_path: Optional[str] = None) -> None:
        """
        Plot regression performance comparison.
        
        Args:
            results_df: DataFrame containing regression results
            save_path: Optional path to save the plot
        """
        try:
            # Filter regression results
            reg_results = results_df[results_df['Task'] == 'regression']
            
            if reg_results.empty:
                self.logger.warning("No regression results to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. RMSE comparison by model
            if 'RMSE' in reg_results.columns:
                rmse_pivot = reg_results.pivot_table(
                    index='Model', values='RMSE', aggfunc='mean'
                )
                rmse_pivot.plot(kind='bar', ax=axes[0, 0], color='skyblue')
                axes[0, 0].set_title('Average RMSE by Model')
                axes[0, 0].set_ylabel('RMSE')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. R² comparison by model
            if 'R2' in reg_results.columns:
                r2_pivot = reg_results.pivot_table(
                    index='Model', values='R2', aggfunc='mean'
                )
                r2_pivot.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
                axes[0, 1].set_title('Average R² by Model')
                axes[0, 1].set_ylabel('R²')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Performance by asset
            if 'Asset' in reg_results.columns and 'R2' in reg_results.columns:
                asset_perf = reg_results.pivot_table(
                    index='Asset', columns='Model', values='R2', aggfunc='mean'
                )
                asset_perf.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('R² Score by Asset and Model')
                axes[1, 0].set_ylabel('R² Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 4. MAE comparison
            if 'MAE' in reg_results.columns:
                mae_pivot = reg_results.pivot_table(
                    index='Model', values='MAE', aggfunc='mean'
                )
                mae_pivot.plot(kind='bar', ax=axes[1, 1], color='salmon')
                axes[1, 1].set_title('Average MAE by Model')
                axes[1, 1].set_ylabel('MAE')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Regression performance plot saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'regression_performance.png', dpi=300, bbox_inches='tight')
                self.logger.info("Regression performance plot saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot regression performance: {e}")
    
    def plot_classification_performance(self, results_df: pd.DataFrame, 
                                      save_path: Optional[str] = None) -> None:
        """
        Plot classification performance comparison.
        
        Args:
            results_df: DataFrame containing classification results
            save_path: Optional path to save the plot
        """
        try:
            # Filter classification results
            clf_results = results_df[results_df['Task'] == 'classification']
            
            if clf_results.empty:
                self.logger.warning("No classification results to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Accuracy comparison by model
            if 'Accuracy' in clf_results.columns:
                acc_pivot = clf_results.pivot_table(
                    index='Model', values='Accuracy', aggfunc='mean'
                )
                acc_pivot.plot(kind='bar', ax=axes[0, 0], color='lightcoral')
                axes[0, 0].set_title('Average Accuracy by Model')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. F1 score comparison by model
            if 'F1' in clf_results.columns:
                f1_pivot = clf_results.pivot_table(
                    index='Model', values='F1', aggfunc='mean'
                )
                f1_pivot.plot(kind='bar', ax=axes[0, 1], color='mediumseagreen')
                axes[0, 1].set_title('Average F1 Score by Model')
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Performance by asset
            if 'Asset' in clf_results.columns and 'Accuracy' in clf_results.columns:
                asset_perf = clf_results.pivot_table(
                    index='Asset', columns='Model', values='Accuracy', aggfunc='mean'
                )
                asset_perf.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Accuracy by Asset and Model')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 4. Precision vs Recall
            if 'Precision' in clf_results.columns and 'Recall' in clf_results.columns:
                precision_pivot = clf_results.pivot_table(
                    index='Model', values='Precision', aggfunc='mean'
                )
                recall_pivot = clf_results.pivot_table(
                    index='Model', values='Recall', aggfunc='mean'
                )
                
                x = np.arange(len(precision_pivot.index))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, precision_pivot.values.flatten(), width, 
                              label='Precision', color='gold')
                axes[1, 1].bar(x + width/2, recall_pivot.values.flatten(), width, 
                              label='Recall', color='lightblue')
                
                axes[1, 1].set_title('Precision vs Recall by Model')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(precision_pivot.index, rotation=45)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Classification performance plot saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'classification_performance.png', dpi=300, bbox_inches='tight')
                self.logger.info("Classification performance plot saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot classification performance: {e}")
    
    def plot_predictions_vs_actual(self, results: Dict[str, Dict], 
                                 save_path: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values for regression models.
        
        Args:
            results: Dictionary containing model results with predictions and actuals
            save_path: Optional path to save the plot
        """
        try:
            # Find regression results
            regression_results = {}
            for asset, asset_results in results.items():
                if 'regression' in asset_results:
                    regression_results[asset] = asset_results['regression']
            
            if not regression_results:
                self.logger.warning("No regression results found for predictions vs actual plot")
                return
            
            # Create subplots
            n_assets = len(regression_results)
            n_cols = min(3, n_assets)
            n_rows = (n_assets + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_assets == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (asset, models) in enumerate(regression_results.items()):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Plot each model's predictions
                for model_name, model_results in models.items():
                    if 'predictions' in model_results and 'actuals' in model_results:
                        predictions = model_results['predictions']
                        actuals = model_results['actuals']
                        
                        if len(predictions) > 0 and len(actuals) > 0:
                            ax.scatter(actuals, predictions, alpha=0.6, label=model_name, s=20)
                
                # Add perfect prediction line
                if 'predictions' in list(models.values())[0] and 'actuals' in list(models.values())[0]:
                    all_actuals = list(models.values())[0]['actuals']
                    if len(all_actuals) > 0:
                        min_val = min(all_actuals)
                        max_val = max(all_actuals)
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{asset} - Predictions vs Actual')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for idx in range(n_assets, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Predictions vs actual plot saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
                self.logger.info("Predictions vs actual plot saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot predictions vs actual: {e}")
    
    def plot_cross_market_comparison(self, results_df: pd.DataFrame, 
                                   save_path: Optional[str] = None) -> None:
        """
        Plot cross-market performance comparison.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the plot
        """
        try:
            # Add market classification
            results_df['Market'] = results_df['Asset'].apply(
                lambda x: 'US' if x in ['AAPL', 'MSFT', '^GSPC'] else 'AU'
            )
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Regression performance by market
            reg_results = results_df[results_df['Task'] == 'regression']
            if not reg_results.empty and 'R2' in reg_results.columns:
                market_reg = reg_results.groupby('Market')['R2'].mean()
                market_reg.plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'lightcoral'])
                axes[0, 0].set_title('Average R² by Market')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].tick_params(axis='x', rotation=0)
            
            # 2. Classification performance by market
            clf_results = results_df[results_df['Task'] == 'classification']
            if not clf_results.empty and 'Accuracy' in clf_results.columns:
                market_clf = clf_results.groupby('Market')['Accuracy'].mean()
                market_clf.plot(kind='bar', ax=axes[0, 1], color=['lightgreen', 'gold'])
                axes[0, 1].set_title('Average Accuracy by Market')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].tick_params(axis='x', rotation=0)
            
            # 3. Model performance comparison across markets
            if not reg_results.empty and 'R2' in reg_results.columns:
                model_market_reg = reg_results.pivot_table(
                    index='Model', columns='Market', values='R2', aggfunc='mean'
                )
                model_market_reg.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('R² Score by Model and Market')
                axes[1, 0].set_ylabel('R² Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend()
            
            # 4. Asset-specific performance
            if not reg_results.empty and 'R2' in reg_results.columns:
                asset_perf = reg_results.groupby('Asset')['R2'].mean().sort_values(ascending=True)
                asset_perf.plot(kind='barh', ax=axes[1, 1], color='lightsteelblue')
                axes[1, 1].set_title('R² Score by Asset')
                axes[1, 1].set_xlabel('R² Score')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Cross-market comparison plot saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'cross_market_comparison.png', dpi=300, bbox_inches='tight')
                self.logger.info("Cross-market comparison plot saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot cross-market comparison: {e}")
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                            save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive model comparison.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the plot
        """
        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Overall model ranking (regression)
            reg_results = results_df[results_df['Task'] == 'regression']
            if not reg_results.empty and 'R2' in reg_results.columns:
                model_ranking_reg = reg_results.groupby('Model')['R2'].mean().sort_values(ascending=True)
                model_ranking_reg.plot(kind='barh', ax=axes[0, 0], color='skyblue')
                axes[0, 0].set_title('Model Ranking - Regression (R²)')
                axes[0, 0].set_xlabel('Average R² Score')
            
            # 2. Overall model ranking (classification)
            clf_results = results_df[results_df['Task'] == 'classification']
            if not clf_results.empty and 'Accuracy' in clf_results.columns:
                model_ranking_clf = clf_results.groupby('Model')['Accuracy'].mean().sort_values(ascending=True)
                model_ranking_clf.plot(kind='barh', ax=axes[0, 1], color='lightcoral')
                axes[0, 1].set_title('Model Ranking - Classification (Accuracy)')
                axes[0, 1].set_xlabel('Average Accuracy')
            
            # 3. Performance heatmap
            if not reg_results.empty and 'R2' in reg_results.columns:
                perf_matrix = reg_results.pivot_table(
                    index='Model', columns='Asset', values='R2', aggfunc='mean'
                )
                sns.heatmap(perf_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                           ax=axes[1, 0], fmt='.3f')
                axes[1, 0].set_title('Performance Heatmap - R² Scores')
            
            # 4. Model consistency (std dev of performance)
            if not reg_results.empty and 'R2' in reg_results.columns:
                model_consistency = reg_results.groupby('Model')['R2'].agg(['mean', 'std'])
                model_consistency.plot(kind='bar', ax=axes[1, 1], y=['mean', 'std'])
                axes[1, 1].set_title('Model Consistency (Mean ± Std)')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].legend(['Mean', 'Std Dev'])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Model comparison plot saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
                self.logger.info("Model comparison plot saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot model comparison: {e}")
    
    def generate_summary_report(self, results_df: pd.DataFrame, 
                              config: Dict, output_dir: Path) -> None:
        """
        Generate comprehensive summary report with visualizations.
        
        Args:
            results_df: DataFrame containing results
            config: Configuration dictionary
            output_dir: Output directory for saving files
        """
        try:
            # Check if results dataframe is empty
            if results_df.empty:
                self.logger.warning("Cannot generate summary report - results dataframe is empty")
                self._create_empty_summary_report(output_dir)
                return
            
            # Create visualizations
            self.plot_regression_performance(results_df, 
                                           save_path=output_dir / 'regression_performance.png')
            self.plot_classification_performance(results_df, 
                                               save_path=output_dir / 'classification_performance.png')
            self.plot_cross_market_comparison(results_df, 
                                            save_path=output_dir / 'cross_market_comparison.png')
            self.plot_model_comparison(results_df, 
                                     save_path=output_dir / 'model_comparison.png')
            
            # Generate text summary
            self._create_summary_text_report(results_df, config, output_dir)
            
            self.logger.info("Summary report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def _create_summary_text_report(self, results_df: pd.DataFrame, 
                                  config: Dict, output_dir: Path) -> None:
        """Create text summary report."""
        try:
            with open(output_dir / 'summary_report.txt', 'w') as f:
                f.write("="*80 + "\n")
                f.write("VOLATILITY FORECASTING RESULTS SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                # Study information
                f.write("STUDY INFORMATION:\n")
                f.write("-"*40 + "\n")
                f.write(f"Study Period: {config.get('data', {}).get('start_date', 'Unknown')} to "
                       f"{config.get('data', {}).get('end_date', 'Unknown')}\n")
                
                assets = config.get('data', {}).get('us_assets', []) + config.get('data', {}).get('au_assets', [])
                f.write(f"Assets Analyzed: {', '.join(assets)}\n")
                f.write(f"Walk-Forward Splits: {config.get('training', {}).get('walk_forward_splits', 'Unknown')}\n\n")
                
                # Results summary
                f.write("RESULTS SUMMARY:\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Evaluations: {len(results_df)}\n")
                f.write(f"Assets Processed: {results_df['Asset'].nunique()}\n")
                f.write(f"Models Evaluated: {results_df['Model'].nunique()}\n")
                f.write(f"Tasks: {', '.join(results_df['Task'].unique())}\n\n")
                
                # Best performing models
                f.write("BEST PERFORMING MODELS:\n")
                f.write("-"*40 + "\n")
                
                # Regression
                reg_results = results_df[results_df['Task'] == 'regression']
                if not reg_results.empty and 'RMSE' in reg_results.columns:
                    best_reg = reg_results.loc[reg_results['RMSE'].idxmin()]
                    f.write(f"Regression (lowest RMSE): {best_reg['Model']} on {best_reg['Asset']} "
                           f"(RMSE={best_reg['RMSE']:.4f})\n")
                
                # Classification
                clf_results = results_df[results_df['Task'] == 'classification']
                if not clf_results.empty and 'F1' in clf_results.columns:
                    best_clf = clf_results.loc[clf_results['F1'].idxmax()]
                    f.write(f"Classification (highest F1): {best_clf['Model']} on {best_clf['Asset']} "
                           f"(F1={best_clf['F1']:.4f})\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Detailed results saved in 'model_performance.csv'\n")
                f.write("Visualization plots saved in current directory\n")
                f.write("SHAP interpretability plots saved in 'shap_plots/'\n")
            
            self.logger.info("Text summary report created")
            
        except Exception as e:
            self.logger.error(f"Failed to create text summary report: {e}")
    
    def _create_empty_summary_report(self, output_dir: Path) -> None:
        """Create empty summary report when no results are available."""
        try:
            with open(output_dir / 'summary_report.txt', 'w') as f:
                f.write("="*80 + "\n")
                f.write("VOLATILITY FORECASTING RESULTS SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write("No results were generated. This could be due to:\n")
                f.write("- Insufficient data for model training\n")
                f.write("- Model training failures\n")
                f.write("- Data quality issues\n\n")
                f.write("Check the log files for detailed error information.\n")
            
            self.logger.info("Empty summary report created")
            
        except Exception as e:
            self.logger.error(f"Failed to create empty summary report: {e}")
    
    def format_results_dataframe(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Format results into a summary dataframe.
        
        Args:
            results: Dictionary containing model results
            
        Returns:
            Formatted DataFrame
        """
        try:
            rows = []
            
            for asset, asset_results in results.items():
                for task, task_results in asset_results.items():
                    for model, metrics in task_results.items():
                        row = {
                            'Asset': asset,
                            'Task': task,
                            'Model': model
                        }
                        
                        # Add metrics (excluding predictions and actuals)
                        for metric, value in metrics.items():
                            if metric not in ['predictions', 'actuals']:
                                # Handle inf and -inf values
                                if isinstance(value, float):
                                    if value == float('inf'):
                                        value = None  # Will become NaN in pandas
                                    elif value == -float('inf'):
                                        value = None
                                row[metric] = value
                        
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Log some statistics about the results
            if not df.empty:
                self.logger.info(f"Results summary: {len(df)} total evaluations")
                if 'R2' in df.columns:
                    valid_r2 = df['R2'].dropna()
                    if not valid_r2.empty:
                        self.logger.info(f"R2 scores - Mean: {valid_r2.mean():.3f}, Best: {valid_r2.max():.3f}")
                if 'Accuracy' in df.columns:
                    valid_acc = df['Accuracy'].dropna()
                    if not valid_acc.empty:
                        self.logger.info(f"Accuracy scores - Mean: {valid_acc.mean():.3f}, Best: {valid_acc.max():.3f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to format results dataframe: {e}")
            return pd.DataFrame()
    
    def create_performance_dashboard(self, results_df: pd.DataFrame, 
                                   save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the dashboard
        """
        try:
            # Create large figure for dashboard
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 1. Overall performance summary (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            if not results_df.empty:
                task_counts = results_df['Task'].value_counts()
                task_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
                ax1.set_title('Task Distribution')
            
            # 2. Model performance comparison (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            if not results_df.empty and 'R2' in results_df.columns:
                reg_results = results_df[results_df['Task'] == 'regression']
                if not reg_results.empty:
                    model_perf = reg_results.groupby('Model')['R2'].mean().sort_values(ascending=True)
                    model_perf.plot(kind='barh', ax=ax2, color='skyblue')
                    ax2.set_title('Model Performance (R²)')
                    ax2.set_xlabel('Average R² Score')
            
            # 3. Asset performance (middle left)
            ax3 = fig.add_subplot(gs[1, :2])
            if not results_df.empty and 'R2' in results_df.columns:
                reg_results = results_df[results_df['Task'] == 'regression']
                if not reg_results.empty:
                    asset_perf = reg_results.groupby('Asset')['R2'].mean().sort_values(ascending=True)
                    asset_perf.plot(kind='barh', ax=ax3, color='lightcoral')
                    ax3.set_title('Asset Performance (R²)')
                    ax3.set_xlabel('Average R² Score')
            
            # 4. Market comparison (middle right)
            ax4 = fig.add_subplot(gs[1, 2:])
            if not results_df.empty:
                results_df['Market'] = results_df['Asset'].apply(
                    lambda x: 'US' if x in ['AAPL', 'MSFT', '^GSPC'] else 'AU'
                )
                if 'R2' in results_df.columns:
                    reg_results = results_df[results_df['Task'] == 'regression']
                    if not reg_results.empty:
                        market_perf = reg_results.groupby('Market')['R2'].mean()
                        market_perf.plot(kind='bar', ax=ax4, color=['skyblue', 'lightcoral'])
                        ax4.set_title('Market Performance (R²)')
                        ax4.set_ylabel('Average R² Score')
            
            # 5. Performance heatmap (bottom left)
            ax5 = fig.add_subplot(gs[2:, :2])
            if not results_df.empty and 'R2' in results_df.columns:
                reg_results = results_df[results_df['Task'] == 'regression']
                if not reg_results.empty:
                    perf_matrix = reg_results.pivot_table(
                        index='Model', columns='Asset', values='R2', aggfunc='mean'
                    )
                    sns.heatmap(perf_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                               ax=ax5, fmt='.3f')
                    ax5.set_title('Performance Heatmap')
            
            # 6. Classification performance (bottom right)
            ax6 = fig.add_subplot(gs[2:, 2:])
            if not results_df.empty and 'Accuracy' in results_df.columns:
                clf_results = results_df[results_df['Task'] == 'classification']
                if not clf_results.empty:
                    clf_perf = clf_results.groupby('Model')['Accuracy'].mean().sort_values(ascending=True)
                    clf_perf.plot(kind='barh', ax=ax6, color='lightgreen')
                    ax6.set_title('Classification Performance (Accuracy)')
                    ax6.set_xlabel('Average Accuracy')
            
            plt.suptitle('RiskPipeline Performance Dashboard', fontsize=16, y=0.98)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Performance dashboard saved to {save_path}")
            else:
                plt.savefig(self.output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
                self.logger.info("Performance dashboard saved")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create performance dashboard: {e}") 