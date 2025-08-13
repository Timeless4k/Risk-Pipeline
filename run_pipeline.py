#!/usr/bin/env python3
"""
RiskPipeline - Comprehensive CLI Menu & Launcher

This is the CENTRAL CONTROL SCRIPT for RiskPipeline with:
✅ ALL FEATURES ENABLED BY DEFAULT
✅ MAXIMUM PERFORMANCE SETTINGS
✅ FULL CUSTOMIZATION OPTIONS
✅ ONE-CLICK "RUN EVERYTHING" OPTION

Usage:
    python run_pipeline.py                    # Interactive menu
    python run_pipeline.py --run-all         # Run everything with max settings
    python run_pipeline.py --quick           # Quick run with defaults
    python run_pipeline.py --custom          # Custom configuration
    python run_pipeline.py --help            # Show help
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class RiskPipelineCLI:
    """Comprehensive CLI menu for RiskPipeline with everything enabled by default."""
    
    def __init__(self):
        # DEFAULT CONFIGURATION - EVERYTHING ENABLED AT MAXIMUM
        self.config = {
            # Data Configuration
            'data_source': 'demo',
            'csv_path': None,
            'date_col': None,
            
            # Models - ALL ENABLED BY DEFAULT
            'models': ['arima', 'xgb', 'stockmixer', 'lstm'],
            'models_to_run': 'arima,xgb,stockmixer,lstm',
            
            # Performance Settings - MAXIMUM BY DEFAULT
            'cpu_only': False,
            'num_workers': 8,  # Maximum parallel processing
            'verbose': True,    # Always verbose for full output
            
            # Features - ALL ENABLED BY DEFAULT
            'run_all': True,
            'compute_shap': True,
            'enable_visualizations': True,
            'enable_volatility_analysis': True,
            'save_models': True,
            'comprehensive_report': True,
            
            # Output Settings
            'artifacts_dir': 'artifacts/comprehensive_run',
            'log_file': 'pipeline.log',
            'dry_run': False,
            
            # Advanced Settings
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
            'random_state': 42,
            'shuffle': False,  # Time series - no shuffle
        }
        
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            print(f"{Colors.OKGREEN}[INFO] TensorFlow available - LSTM models enabled{Colors.ENDC}")
        except ImportError:
            self.tensorflow_available = False
            self.config['models'].remove('lstm')
            self.config['models_to_run'] = ','.join(self.config['models'])
            print(f"{Colors.WARNING}[INFO] TensorFlow not available - LSTM models disabled{Colors.ENDC}")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the main header."""
        self.clear_screen()
        print(f"{Colors.HEADER}{'='*80}")
        print(f"{Colors.BOLD}🚀 RiskPipeline - COMPREHENSIVE CLI MENU{Colors.ENDC}")
        print(f"{Colors.BOLD}🎯 ALL FEATURES ENABLED BY DEFAULT - MAXIMUM PERFORMANCE{Colors.ENDC}")
        print(f"{'='*80}{Colors.ENDC}")
        print()
    
    def print_current_config(self):
        """Print current configuration summary."""
        print(f"{Colors.OKCYAN}📋 CURRENT CONFIGURATION (Everything Enabled):{Colors.ENDC}")
        print(f"  📊 Data: {self.config['data_source']}")
        if self.config['csv_path']:
            print(f"  📁 CSV: {self.config['csv_path']}")
        print(f"  🤖 Models: {', '.join(self.config['models'])}")
        print(f"  🔧 SHAP: ✅ ENABLED")
        print(f"  📈 Visualizations: ✅ ENABLED")
        print(f"  💾 Save Models: ✅ ENABLED")
        print(f"  📊 Volatility Analysis: ✅ ENABLED")
        print(f"  📁 Output: {self.config['artifacts_dir']}")
        print(f"  🚀 Parallel Workers: {self.config['num_workers']} (MAXIMUM)")
        print(f"  📝 Verbose Logging: ✅ ENABLED")
        print(f"  📋 Comprehensive Report: ✅ ENABLED")
        print()
    
    def show_main_menu(self):
        """Display the main menu."""
        while True:
            self.print_header()
            self.print_current_config()
            
            print(f"{Colors.OKBLUE}🎮 MAIN MENU - Choose Your Option:{Colors.ENDC}")
            print()
            print("1. 🚀 RUN EVERYTHING (Recommended - All features, max performance)")
            print("2. ⚡ QUICK RUN (Skip settings, use defaults)")
            print("3. 🔧 CUSTOM CONFIGURATION (Tune individual settings)")
            print("4. 📊 DATA CONFIGURATION (Change data source)")
            print("5. 🤖 MODEL SELECTION (Choose which models to run)")
            print("6. ⚙️  PERFORMANCE TUNING (Adjust workers, CPU mode)")
            print("7. 📁 OUTPUT SETTINGS (Change directories, logging)")
            print("8. ❓ HELP & INFORMATION")
            print("9. 🚪 EXIT")
            print()
            
            choice = self.get_menu_choice(9)
            if choice == 1:
                self.run_everything()
            elif choice == 2:
                self.quick_run()
            elif choice == 3:
                self.custom_configuration()
            elif choice == 4:
                self.data_configuration()
            elif choice == 5:
                self.model_selection()
            elif choice == 6:
                self.performance_tuning()
            elif choice == 7:
                self.output_settings()
            elif choice == 8:
                self.show_help()
            elif choice == 9:
                print(f"\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
                break
    
    def get_menu_choice(self, max_options: int) -> int:
        """Get user choice from menu."""
        while True:
            try:
                choice = input(f"Enter your choice (1-{max_options}): ").strip()
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= max_options:
                        return choice_num
                    else:
                        print(f"{Colors.FAIL}Invalid choice. Please enter a number between 1 and {max_options}.{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}Invalid choice. Please enter a number.{Colors.ENDC}")
            except (ValueError, IndexError):
                print(f"{Colors.FAIL}Invalid choice. Please enter a number.{Colors.ENDC}")
    
    def run_everything(self):
        """Run everything with maximum settings."""
        self.print_header()
        print(f"{Colors.OKGREEN}🚀 RUNNING EVERYTHING WITH MAXIMUM SETTINGS!{Colors.ENDC}")
        print()
        print("This will run:")
        print("✅ ALL MODELS: ARIMA, XGBoost, StockMixer, LSTM (if available)")
        print("✅ SHAP ANALYSIS: Full model interpretability")
        print("✅ COMPREHENSIVE VISUALIZATIONS: All plots and charts")
        print("✅ VOLATILITY ANALYSIS: Advanced financial analysis")
        print("✅ MODEL PERSISTENCE: Save all trained models")
        print("✅ MAXIMUM PERFORMANCE: {self.config['num_workers']} parallel workers")
        print("✅ VERBOSE LOGGING: Full detailed output")
        print("✅ COMPREHENSIVE REPORT: Complete analysis summary")
        print()
        
        if self.get_yes_no("Ready to run everything with maximum settings?"):
            self.execute_pipeline()
    
    def quick_run(self):
        """Quick run with current defaults."""
        self.print_header()
        print(f"{Colors.OKBLUE}⚡ QUICK RUN - Using Current Settings{Colors.ENDC}")
        print()
        print("Running with current configuration (no changes):")
        self.print_current_config()
        
        if self.get_yes_no("Run pipeline with these settings?"):
            self.execute_pipeline()
    
    def custom_configuration(self):
        """Custom configuration menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}🔧 CUSTOM CONFIGURATION{Colors.ENDC}")
            print()
            print("1. 📊 Data Settings")
            print("2. 🤖 Model Settings")
            print("3. ⚙️  Performance Settings")
            print("4. 🔧 Feature Toggles")
            print("5. 📁 Output Settings")
            print("6. ⬅ Back to Main Menu")
            print()
            
            choice = self.get_menu_choice(6)
            if choice == 1:
                self.data_configuration()
            elif choice == 2:
                self.model_selection()
            elif choice == 3:
                self.performance_tuning()
            elif choice == 4:
                self.feature_toggles()
            elif choice == 5:
                self.output_settings()
            elif choice == 6:
                break
    
    def data_configuration(self):
        """Data configuration menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}📊 DATA CONFIGURATION{Colors.ENDC}")
            print()
            print(f"Current: {self.config['data_source']}")
            print()
            print("1. 🎯 Demo Data (Built-in sample data)")
            print("2. 📁 Custom CSV File")
            print("3. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(3)
            if choice == 1:
                self.config['data_source'] = 'demo'
                self.config['csv_path'] = None
                self.config['date_col'] = None
                print(f"{Colors.OKGREEN}✅ Demo data selected{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 2:
                self.configure_csv_data()
            elif choice == 3:
                break
    
    def configure_csv_data(self):
        """Configure CSV data settings."""
        self.print_header()
        print(f"{Colors.OKBLUE}📁 CSV CONFIGURATION{Colors.ENDC}")
        print()
        
        # Get CSV path
        while True:
            csv_path = input("Enter path to CSV file: ").strip()
            if os.path.exists(csv_path):
                self.config['csv_path'] = csv_path
                self.config['data_source'] = 'csv'
                break
            else:
                print(f"{Colors.FAIL}File not found: {csv_path}{Colors.ENDC}")
                if not self.get_yes_no("Try again?"):
                    return
        
        # Get date column
        self.config['date_col'] = input("Enter date column name (or press Enter if index is datetime): ").strip()
        if not self.config['date_col']:
            self.config['date_col'] = None
        
        print(f"{Colors.OKGREEN}✅ CSV configuration complete{Colors.ENDC}")
        input("Press Enter to continue...")
    
    def model_selection(self):
        """Model selection menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}🤖 MODEL SELECTION{Colors.ENDC}")
            print()
            
            available_models = ['arima', 'xgb', 'stockmixer']
            if self.tensorflow_available:
                available_models.append('lstm')
            
            print("Available models:")
            for model in available_models:
                status = "✅" if model in self.config['models'] else "❌"
                print(f"  {status} {model.upper()}")
            
            print()
            print("1. ✅ Enable All Models")
            print("2. ❌ Disable All Models")
            print("3. 🔧 Toggle Individual Models")
            print("4. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(4)
            if choice == 1:
                self.config['models'] = available_models.copy()
                self.config['models_to_run'] = ','.join(self.config['models'])
                print(f"{Colors.OKGREEN}✅ All models enabled{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 2:
                self.config['models'] = []
                self.config['models_to_run'] = ''
                print(f"{Colors.WARNING}⚠️  All models disabled{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 3:
                self.toggle_individual_models(available_models)
            elif choice == 4:
                break
    
    def toggle_individual_models(self, available_models):
        """Toggle individual model selection."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}🔧 TOGGLE INDIVIDUAL MODELS{Colors.ENDC}")
            print()
            
            for i, model in enumerate(available_models, 1):
                status = "✅" if model in self.config['models'] else "❌"
                print(f"{i}. {status} {model.upper()}")
            
            print(f"{len(available_models) + 1}. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(len(available_models) + 1)
            if choice <= len(available_models):
                model = available_models[choice - 1]
                if model in self.config['models']:
                    self.config['models'].remove(model)
                    print(f"❌ {model.upper()} disabled")
                else:
                    self.config['models'].append(model)
                    print(f"✅ {model.upper()} enabled")
                
                self.config['models_to_run'] = ','.join(self.config['models'])
                input("Press Enter to continue...")
            else:
                break
    
    def performance_tuning(self):
        """Performance tuning menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}⚙️  PERFORMANCE TUNING{Colors.ENDC}")
            print()
            print(f"Current settings:")
            print(f"  🚀 Parallel Workers: {self.config['num_workers']}")
            print(f"  💻 CPU Only Mode: {'✅' if self.config['cpu_only'] else '❌'}")
            print()
            print("1. 🚀 Set Maximum Parallel Workers (8)")
            print("2. 🔧 Custom Number of Workers")
            print("3. 💻 Toggle CPU Only Mode")
            print("4. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(4)
            if choice == 1:
                self.config['num_workers'] = 8
                print(f"{Colors.OKGREEN}✅ Maximum workers set to 8{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 2:
                try:
                    workers = int(input("Enter number of workers (1-16): "))
                    if 1 <= workers <= 16:
                        self.config['num_workers'] = workers
                        print(f"{Colors.OKGREEN}✅ Workers set to {workers}{Colors.ENDC}")
                    else:
                        print(f"{Colors.FAIL}Invalid number. Must be between 1 and 16.{Colors.ENDC}")
                    input("Press Enter to continue...")
                except ValueError:
                    print(f"{Colors.FAIL}Invalid input. Please enter a number.{Colors.ENDC}")
                    input("Press Enter to continue...")
            elif choice == 3:
                self.config['cpu_only'] = not self.config['cpu_only']
                status = "enabled" if self.config['cpu_only'] else "disabled"
                print(f"{Colors.OKGREEN}✅ CPU only mode {status}{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 4:
                break
    
    def feature_toggles(self):
        """Feature toggles menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}🔧 FEATURE TOGGLES{Colors.ENDC}")
            print()
            
            features = [
                ('compute_shap', 'SHAP Analysis'),
                ('enable_visualizations', 'Visualizations'),
                ('enable_volatility_analysis', 'Volatility Analysis'),
                ('save_models', 'Save Models'),
                ('comprehensive_report', 'Comprehensive Report'),
                ('verbose', 'Verbose Logging')
            ]
            
            for i, (key, name) in enumerate(features, 1):
                status = "✅" if self.config[key] else "❌"
                print(f"{i}. {status} {name}")
            
            print(f"{len(features) + 1}. ✅ Enable All Features")
            print(f"{len(features) + 2}. ❌ Disable All Features")
            print(f"{len(features) + 3}. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(len(features) + 3)
            if choice <= len(features):
                key = features[choice - 1][0]
                self.config[key] = not self.config[key]
                status = "enabled" if self.config[key] else "disabled"
                print(f"{Colors.OKGREEN}✅ {features[choice - 1][1]} {status}{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == len(features) + 1:
                for key, _ in features:
                    self.config[key] = True
                print(f"{Colors.OKGREEN}✅ All features enabled{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == len(features) + 2:
                for key, _ in features:
                    self.config[key] = False
                print(f"{Colors.WARNING}⚠️  All features disabled{Colors.ENDC}")
                input("Press Enter to continue...")
            else:
                break
    
    def output_settings(self):
        """Output settings menu."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}📁 OUTPUT SETTINGS{Colors.ENDC}")
            print()
            print(f"Current settings:")
            print(f"  📁 Output Directory: {self.config['artifacts_dir']}")
            print(f"  📝 Log File: {self.config['log_file']}")
            print()
            print("1. 📁 Change Output Directory")
            print("2. 📝 Change Log File")
            print("3. ⬅ Back")
            print()
            
            choice = self.get_menu_choice(3)
            if choice == 1:
                new_dir = input(f"Enter new output directory [{self.config['artifacts_dir']}]: ").strip()
                if new_dir:
                    self.config['artifacts_dir'] = new_dir
                    print(f"{Colors.OKGREEN}✅ Output directory changed to {new_dir}{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 2:
                new_log = input(f"Enter new log file path [{self.config['log_file']}]: ").strip()
                if new_log:
                    self.config['log_file'] = new_log
                    print(f"{Colors.OKGREEN}✅ Log file changed to {new_log}{Colors.ENDC}")
                input("Press Enter to continue...")
            elif choice == 3:
                break
    
    def show_help(self):
        """Show help and information."""
        while True:
            self.print_header()
            print(f"{Colors.OKBLUE}❓ HELP & INFORMATION{Colors.ENDC}")
            print()
            print(f"{Colors.BOLD}RiskPipeline - Comprehensive CLI Menu{Colors.ENDC}")
            print("This is the central control script for RiskPipeline with everything enabled by default.")
            print()
            print(f"{Colors.BOLD}🚀 Key Features:{Colors.ENDC}")
            print("• ALL MODELS: ARIMA, XGBoost, StockMixer, LSTM (if available)")
            print("• MAXIMUM PERFORMANCE: 8 parallel workers by default")
            print("• COMPREHENSIVE ANALYSIS: SHAP, visualizations, volatility analysis")
            print("• VERBOSE LOGGING: Full detailed output")
            print("• MODEL PERSISTENCE: Save all trained models")
            print()
            print(f"{Colors.BOLD}🎯 Quick Start:{Colors.ENDC}")
            print("• Option 1: Run everything with maximum settings")
            print("• Option 2: Quick run with current defaults")
            print("• Option 3: Customize individual settings")
            print()
            print(f"{Colors.BOLD}💡 Tips:{Colors.ENDC}")
            print("• Start with 'Run Everything' for full analysis")
            print("• Use 'Custom Configuration' to tune specific settings")
            print("• All features are enabled by default for maximum results")
            print()
            
            input("Press Enter to go back...")
            break
    
    def get_yes_no(self, prompt: str) -> bool:
        """Get yes/no input from user."""
        while True:
            response = input(f"{prompt} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print(f"{Colors.FAIL}Please enter 'y' or 'n'{Colors.ENDC}")
    
    def execute_pipeline(self):
        """Execute the pipeline with current configuration."""
        self.print_header()
        print(f"{Colors.OKBLUE}🚀 EXECUTING PIPELINE...{Colors.ENDC}")
        print()
        
        # Build command
        cmd = [sys.executable, "risk_pipeline_main.py"]
        
        # Add data source
        cmd.extend(["--data", self.config['data_source']])
        
        # Add CSV path if using CSV
        if self.config['data_source'] == 'csv' and self.config['csv_path']:
            cmd.extend(["--csv-path", self.config['csv_path']])
            if self.config['date_col']:
                cmd.extend(["--date-col", self.config['date_col']])
        
        # Add models
        if self.config['models']:
            cmd.extend(["--models", self.config['models_to_run']])
        
        # Add artifacts directory
        cmd.extend(["--artifacts", self.config['artifacts_dir']])
        
        # Add feature flags
        if self.config['run_all']:
            cmd.append("--run-all")
        else:
            if self.config['compute_shap']:
                cmd.append("--compute-shap")
            if self.config['enable_visualizations']:
                cmd.append("--enable-visualizations")
            if self.config['enable_volatility_analysis']:
                cmd.append("--enable-volatility-analysis")
            if self.config['save_models']:
                cmd.append("--save-models")
            if self.config['comprehensive_report']:
                cmd.append("--comprehensive-report")
        
        # Add other options
        if self.config['cpu_only']:
            cmd.append("--cpu-only")
        if self.config['verbose']:
            cmd.append("--verbose")
        if self.config['log_file']:
            cmd.extend(["--log-file", self.config['log_file']])
        if self.config['num_workers'] > 0:
            cmd.extend(["--num-workers", str(self.config['num_workers'])])
        if self.config['dry_run']:
            cmd.append("--dry-run")
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Execute
        try:
            result = subprocess.run(cmd, check=True)
            print(f"\n{Colors.OKGREEN}✅ Pipeline completed successfully!{Colors.ENDC}")
            print(f"📁 Check results in: {self.config['artifacts_dir']}")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.FAIL}❌ Pipeline failed with exit code: {e.returncode}{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}⏹️ Pipeline interrupted by user{Colors.ENDC}")
        
        input("\nPress Enter to continue...")

def main():
    """Main entry point."""
    try:
        # Check if we're in the right directory
        if not Path("risk_pipeline_main.py").exists():
            print(f"{Colors.FAIL}Error: Please run this script from the Risk-Pipeline root directory{Colors.ENDC}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Expected file: risk_pipeline_main.py")
            return 1
        
        # Handle command line arguments
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            
            if arg in ['--help', '-h', 'help']:
                print("🚀 RiskPipeline - Comprehensive CLI Menu")
                print("=" * 50)
                print()
                print("Usage:")
                print("  python run_pipeline.py                    # Interactive menu")
                print("  python run_pipeline.py --run-all         # Run everything with max settings")
                print("  python run_pipeline.py --quick           # Quick run with defaults")
                print("  python run_pipeline.py --custom          # Custom configuration")
                print("  python run_pipeline.py --help            # Show this help")
                print()
                print("Features:")
                print("  ✅ ALL MODELS: ARIMA, XGBoost, StockMixer, LSTM")
                print("  ✅ MAXIMUM PERFORMANCE: 8 parallel workers")
                print("  ✅ COMPREHENSIVE ANALYSIS: SHAP, visualizations, volatility")
                print("  ✅ VERBOSE LOGGING: Full detailed output")
                print("  ✅ MODEL PERSISTENCE: Save all trained models")
                return 0
            
            elif arg in ['--run-all', 'run-all']:
                print("🚀 Running everything with maximum settings...")
                cli = RiskPipelineCLI()
                cli.run_everything()
                return 0
            
            elif arg in ['--quick', 'quick']:
                print("⚡ Running quick pipeline with defaults...")
                cli = RiskPipelineCLI()
                cli.quick_run()
                return 0
            
            elif arg in ['--custom', 'custom']:
                print("🔧 Opening custom configuration menu...")
                cli = RiskPipelineCLI()
                cli.custom_configuration()
                return 0
            
            else:
                print(f"❌ Unknown argument: {arg}")
                print("Use --help for available options")
                return 1
        
        # No arguments - show interactive menu
        cli = RiskPipelineCLI()
        cli.show_main_menu()
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Goodbye! 👋{Colors.ENDC}")
        return 0
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
