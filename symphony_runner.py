#!/usr/bin/env python3
"""
Symphony Runner - Main Integration Script

Ties together all components to run symphony backtests and live execution.
This is your main entry point for testing the new symphony system.

Fixed: No circular imports - uses component factory pattern.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import core components without circular dependencies
from symphony_core import SymphonyService, SymphonyConfig
from integrated_symphony_system import IntegratedSymphonySystem

class SymphonyRunner:
    """Main orchestrator for symphony operations"""
    
    def __init__(self, config_path: str = None):
        # Initialize with clean architecture
        symphony_config = SymphonyConfig()
        if config_path:
            symphony_config.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        self.service = SymphonyService(symphony_config)
        self.factory = self.service.factory
        
        # Get all components from factory (no circular imports)
        self.data_config = symphony_config
        self.data_pipeline = self.factory.get_data_pipeline()
        self.data_manager = self.factory.get_data_manager()
        self.engine = self.factory.get_engine()
        self.backtester = self.factory.get_backtester()
        self.visualizer = self.factory.get_visualizer()
        self.forecaster = self.factory.get_forecaster()
        self.optimizer = self.factory.get_optimizer()
        self.composer_parser = self.factory.get_composer_parser()
        
        # Initialize integrated system (no circular dependency)
        self.integrated_system = IntegratedSymphonySystem(symphony_config)
        
        print("🎵 Symphony Runner initialized")
        print(f"📊 Data source: {self.data_config.source if hasattr(self.data_config, 'source') else 'alpha_vantage'}")
        print(f"💾 Cache directory: {self.data_config.data_cache_dir}")
        print(f"🎨 All components loaded via factory pattern")
        print(f"✅ No circular import issues")
    
    def load_symphony_config(self, config_path: str) -> dict:
        """Load symphony configuration from JSON file"""
        return self.service.load_symphony_config(config_path)
    
    def create_sample_symphony(self, output_path: str = "sample_symphony_v2.json"):
        """Create a sample symphony configuration file"""
        return self.service.create_sample_symphony(output_path)
    
    def validate_symphony(self, symphony_config: dict) -> bool:
        """Validate symphony configuration"""
        return self.service.validate_symphony(symphony_config)
    
    def run_backtest(self, symphony_config: dict, start_date: str, end_date: str, 
                    rebalance_frequency: str = "monthly") -> pd.DataFrame:
        """Run a complete backtest"""
        
        print(f"\n🚀 Starting backtest: {symphony_config['name']}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🔄 Rebalance frequency: {rebalance_frequency}")
        
        # Step 1: Prepare data
        print("\n📊 Fetching market data...")
        market_data = self.data_manager.prepare_symphony_data(
            symphony_config, start_date, end_date
        )
        
        if not market_data:
            raise ValueError("No market data available for symphony universe")
        
        # Step 2: Run backtest
        print(f"\n⚡ Running backtest with {len(market_data)} symbols...")
        results = self.backtester.backtest(
            symphony_config, market_data, start_date, end_date, rebalance_frequency
        )
        
        if results.empty:
            raise ValueError("Backtest produced no results")
        
        print(f"✅ Backtest completed: {len(results)} rebalance periods")
        return results
    
    def analyze_backtest_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze backtest results and calculate performance metrics"""
        
        print("\n📈 Analyzing backtest results...")
        
        # Calculate cumulative returns
        results['cumulative_return'] = (1 + results['portfolio_return']).cumprod() - 1
        
        # Performance metrics
        total_return = results['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        volatility = results['portfolio_return'].std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative_returns = 1 + results['cumulative_return']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (results['portfolio_return'] > 0).mean()
        
        analysis = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(results),
            'avg_return_per_period': results['portfolio_return'].mean()
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Return:        {total_return:8.2%}")
        print(f"Annualized Return:   {annual_return:8.2%}")
        print(f"Volatility:          {volatility:8.2%}")
        print(f"Sharpe Ratio:        {sharpe_ratio:8.2f}")
        print(f"Max Drawdown:        {max_drawdown:8.2%}")
        print(f"Win Rate:            {win_rate:8.2%}")
        print(f"Number of Periods:   {len(results):8d}")
        print("=" * 50)
        
        return analysis
    
    def run_single_execution(self, symphony_config: dict, execution_date: str = None) -> Dict[str, Any]:
        """Run symphony for a single execution date"""
        
        if execution_date is None:
            execution_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n🎯 Single execution for date: {execution_date}")
        
        # Prepare data up to execution date
        market_data = self.data_manager.prepare_symphony_data(
            symphony_config, end_date=execution_date
        )
        
        if not market_data:
            raise ValueError("No market data available")
        
        # Execute symphony
        result = self.engine.execute_symphony(symphony_config, market_data, execution_date)
        
        print(f"\n✅ Execution completed")
        print(f"🎯 Triggered condition: {result.triggered_condition}")
        print(f"📊 Portfolio allocation:")
        
        for symbol, weight in result.allocation.items():
            print(f"  {symbol}: {weight:6.2%}")
        
        return {
            'allocation': result.allocation,
            'triggered_condition': result.triggered_condition,
            'metrics_used': result.metrics_used,
            'execution_date': result.execution_date
        }
    
    def export_results(self, results: pd.DataFrame, analysis: Dict[str, Any], 
                      output_dir: str = "./backtest_results"):
        """Export backtest results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export detailed results
        results_file = os.path.join(output_dir, "backtest_results.csv")
        results.to_csv(results_file)
        print(f"📄 Detailed results exported: {results_file}")
        
        # Export analysis summary
        analysis_file = os.path.join(output_dir, "performance_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📊 Performance analysis exported: {analysis_file}")
        
        # Create allocation history file
        allocation_history = []
        for _, row in results.iterrows():
            for symbol, weight in row['allocation'].items():
                allocation_history.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'weight': weight,
                    'condition': row['triggered_condition']
                })
        
        allocation_df = pd.DataFrame(allocation_history)
        allocation_file = os.path.join(output_dir, "allocation_history.csv")
        allocation_df.to_csv(allocation_file, index=False)
        print(f"📈 Allocation history exported: {allocation_file}")
    
    def create_visualizations(self, results: pd.DataFrame, analysis: Dict[str, Any], 
                            output_dir: str = "./backtest_results", 
                            benchmark_symbol: str = 'SPY',
                            start_date: str = None, end_date: str = None) -> None:
        """Create comprehensive visualizations for backtest results"""
        
        print(f"\n🎨 Creating visualizations...")
        
        # Fetch benchmark data if requested
        benchmark_data = None
        if benchmark_symbol:
            print(f"📊 Fetching benchmark data ({benchmark_symbol})...")
            from symphony_visualizer import fetch_benchmark_data
            benchmark_data = fetch_benchmark_data(benchmark_symbol, start_date, end_date)
            
            if not benchmark_data.empty:
                print(f"✅ Benchmark data loaded: {len(benchmark_data)} records")
            else:
                print("⚠️ Could not load benchmark data, proceeding without comparison")
                benchmark_data = None
        
        # Create charts directory
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate all visualizations
        try:
            print("\n📊 Generating performance dashboard...")
            self.visualizer.create_performance_dashboard(
                results, analysis, benchmark_data,
                save_path=os.path.join(charts_dir, "performance_dashboard.png")
            )
            
            print("🌐 Generating interactive dashboard...")
            self.visualizer.create_interactive_dashboard(
                results, analysis, benchmark_data,
                save_path=os.path.join(charts_dir, "interactive_dashboard.html")
            )
            
            print("🌅 Generating allocation sunburst...")
            self.visualizer.create_allocation_sunburst(
                results,
                save_path=os.path.join(charts_dir, "allocation_sunburst.html")
            )
            
            print("📈 Generating rolling metrics...")
            self.visualizer.create_rolling_metrics_chart(
                results, window=min(12, len(results)//4),
                save_path=os.path.join(charts_dir, "rolling_metrics.html")
            )
            
            print(f"\n✅ All visualizations saved to: {charts_dir}")
            print(f"🌐 Open {charts_dir}/interactive_dashboard.html in your browser for interactive charts!")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def run_full_analysis(self, symphony_config: dict, start_date: str, end_date: str,
                         rebalance_frequency: str = "monthly", benchmark_symbol: str = 'SPY',
                         output_dir: str = "./backtest_results") -> Dict[str, Any]:
        """Run complete analysis including backtest and visualizations"""
        
        print(f"\n🚀 Running full analysis: {symphony_config['name']}")
        
        # Step 1: Run backtest
        results = self.run_backtest(symphony_config, start_date, end_date, rebalance_frequency)
        
        # Step 2: Analyze results
        analysis = self.analyze_backtest_results(results)
        
        # Step 3: Export results
        self.export_results(results, analysis, output_dir)
        
        # Step 4: Create visualizations
        self.create_visualizations(results, analysis, output_dir, benchmark_symbol, start_date, end_date)
        
        return {
            'results': results,
            'analysis': analysis,
            'output_dir': output_dir
        }
    
    # New advanced methods that use integrated system
    def run_forecasting_analysis(self, symphony_config: dict, backtest_results: pd.DataFrame,
                               forecast_days: int = 252, output_dir: str = "./forecast_results") -> dict:
        """Run comprehensive forecasting analysis"""
        
        print(f"\n🔮 Running Forecasting Analysis")
        print(f"📊 Symphony: {symphony_config['name']}")
        print(f"📅 Forecast horizon: {forecast_days} days")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run Monte Carlo forecast
            mc_forecast = self.forecaster.forecast_symphony_performance(
                backtest_results, forecast_days, method='monte_carlo'
            )
            
            # Generate forecast report
            forecast_report = {
                'symphony_name': symphony_config['name'],
                'forecast_date': datetime.now().isoformat(),
                'forecast_horizon_days': forecast_days,
                'monte_carlo_results': {
                    'expected_annual_return': mc_forecast.expected_annual_return,
                    'expected_volatility': mc_forecast.expected_volatility,
                    'expected_sharpe': mc_forecast.expected_sharpe,
                    'worst_case_scenario': mc_forecast.worst_case_scenario,
                    'best_case_scenario': mc_forecast.best_case_scenario,
                    'probability_beating_benchmark': mc_forecast.probability_of_beating_benchmark,
                    'stress_test_results': mc_forecast.stress_test_results
                }
            }
            
            # Save forecast report
            report_file = os.path.join(output_dir, "forecast_analysis.json")
            with open(report_file, 'w') as f:
                json.dump(forecast_report, f, indent=2, default=str)
            
            # Create forecast visualization
            self.forecaster.plot_forecast(mc_forecast, symphony_config['name'])
            
            print("✅ Forecasting analysis completed")
            print(f"📄 Report saved: {report_file}")
            
            return {
                'monte_carlo': mc_forecast,
                'report': forecast_report,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Forecasting failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def optimize_symphony(self, symphony_config: dict, market_data: Dict[str, pd.DataFrame],
                         start_date: str, end_date: str, output_dir: str = "./optimization_results") -> dict:
        """Run symphony optimization"""
        
        print(f"\n🔧 Optimizing Symphony Parameters")
        print(f"📊 Symphony: {symphony_config['name']}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Define parameter ranges based on symphony structure
            parameter_ranges = self._generate_optimization_ranges(symphony_config)
            
            if not parameter_ranges:
                print("⚠️ No optimizable parameters found")
                return {'status': 'skipped', 'reason': 'No optimizable parameters'}
            
            # Run optimization
            optimization_result = self.optimizer.optimize_symphony_parameters(
                symphony_config, parameter_ranges, market_data, 
                start_date, end_date, max_combinations=50
            )
            
            # Generate optimization report
            report = self.optimizer.generate_optimization_report(
                optimization_result, os.path.join(output_dir, "optimization_report.json")
            )
            
            print("✅ Optimization completed")
            print(f"🏆 Best Sharpe Ratio: {optimization_result.best_metrics.get('sharpe_ratio', 0):.3f}")
            
            return {
                'result': optimization_result,
                'report': report,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def convert_composer_symphony(self, composer_dsl: str, composer_csv_path: str = None,
                                output_dir: str = "./composer_conversion") -> dict:
        """Convert and validate Composer symphony"""
        
        print(f"\n🔄 Converting Composer Symphony")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Use integrated system for conversion
            result = self.integrated_system.convert_composer_symphony(composer_dsl, composer_csv_path)
            
            if result.get('parsing', {}).get('status') == 'success':
                parse_result = result['parsing']
                
                # Save converted symphony
                converted_file = os.path.join(output_dir, f"{parse_result['name'].replace(' ', '_')}.json")
                with open(converted_file, 'w') as f:
                    json.dump(parse_result['our_format'], f, indent=2)
                
                result['converted_file'] = converted_file
                print(f"✅ Conversion completed: {converted_file}")
                
                if 'reconciliation' in result:
                    reconciliation_status = result['reconciliation']['reconciliation'].get('reconciliation_status', 'Unknown')
                    print(f"📊 Reconciliation Status: {reconciliation_status}")
            
            return result
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def create_spy_plus_strategies(self, market_data: Dict[str, pd.DataFrame], 
                                 start_date: str, end_date: str,
                                 output_dir: str = "./spy_plus_strategies") -> dict:
        """Create and validate SPY-plus strategies"""
        
        print(f"\n🛡️ Creating SPY-Plus Strategies")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Generate SPY-plus strategies
            strategies = self.optimizer.create_spy_plus_strategies(market_data, start_date, end_date)
            
            # Test each strategy and create detailed reports
            strategy_results = []
            
            for i, strategy in enumerate(strategies):
                print(f"\n📊 Testing: {strategy['name']}")
                
                # Run full analysis
                results = self.run_full_analysis(
                    strategy, start_date, end_date, 'monthly', 'SPY',
                    os.path.join(output_dir, f"strategy_{i+1}")
                )
                
                strategy_results.append({
                    'strategy': strategy,
                    'analysis_results': results,
                    'recommendation': self._evaluate_spy_plus_strategy(results)
                })
                
                # Save individual strategy
                strategy_file = os.path.join(output_dir, f"{strategy['name'].replace(' ', '_')}.json")
                with open(strategy_file, 'w') as f:
                    json.dump(strategy, f, indent=2)
            
            # Create summary report
            summary = {
                'creation_date': datetime.now().isoformat(),
                'strategies_created': len(strategies),
                'strategies_tested': len(strategy_results),
                'recommended_strategies': [
                    r for r in strategy_results 
                    if r['recommendation']['deploy']
                ],
                'detailed_results': strategy_results
            }
            
            summary_file = os.path.join(output_dir, "spy_plus_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"✅ SPY-plus strategies created: {len(strategies)}")
            print(f"🎯 Recommended for deployment: {len(summary['recommended_strategies'])}")
            
            return {
                'strategies': strategies,
                'results': strategy_results,
                'summary': summary,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ SPY-plus creation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_production_pipeline(self, symphony_config: dict, market_data: Dict[str, pd.DataFrame] = None,
                              risk_level: str = 'low', output_dir: str = "./production_analysis") -> dict:
        """Complete production readiness pipeline"""
        
        print(f"\n🏭 Production Readiness Pipeline")
        print(f"📊 Symphony: {symphony_config['name']}")
        print(f"🛡️ Risk Level: {risk_level.upper()}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Prepare market data if not provided
            if market_data is None:
                market_data = self.data_manager.prepare_symphony_data(symphony_config)
            
            # Use integrated system for complete analysis
            pipeline_results = self.integrated_system.full_symphony_development_pipeline(
                symphony_config, market_data, '2022-01-01', '2024-12-31'
            )
            
            # Save comprehensive results
            results_file = os.path.join(output_dir, "production_pipeline_results.json")
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            # Generate production recommendation
            deployment_rec = pipeline_results.get('deployment_recommendation', {})
            
            print(f"\n🎯 Production Recommendation: {deployment_rec.get('recommendation', 'UNKNOWN')}")
            print(f"🎯 Confidence: {deployment_rec.get('confidence', 0):.2%}")
            
            return {
                'pipeline_results': pipeline_results,
                'recommendation': deployment_rec,
                'results_file': results_file,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Production pipeline failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_optimization_ranges(self, symphony_config: dict) -> Dict[str, List]:
        """Generate parameter ranges for optimization based on symphony structure"""
        
        ranges = {}
        
        # Check for lookback days parameters
        conditions = symphony_config.get('logic', {}).get('conditions', [])
        for condition in conditions:
            if 'condition' in condition and 'lookback_days' in condition['condition']:
                current_lookback = condition['condition']['lookback_days']
                ranges['lookback_days'] = [
                    max(10, current_lookback - 30),
                    current_lookback,
                    current_lookback + 30,
                    current_lookback + 60
                ]
                break
        
        # Check for threshold parameters
        for condition in conditions:
            if ('condition' in condition and 'asset_2' in condition['condition'] and
                isinstance(condition['condition']['asset_2'], dict) and
                condition['condition']['asset_2'].get('type') == 'fixed_value'):
                
                current_threshold = condition['condition']['asset_2']['value']
                ranges['threshold'] = [
                    current_threshold * 0.5,
                    current_threshold * 0.75,
                    current_threshold,
                    current_threshold * 1.5,
                    current_threshold * 2.0
                ]
                break
        
        return ranges
    
    def _evaluate_spy_plus_strategy(self, analysis_results: dict) -> dict:
        """Evaluate SPY-plus strategy for deployment recommendation"""
        
        recommendation = {
            'deploy': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        if analysis_results.get('analysis'):
            analysis = analysis_results['analysis']
            
            annual_return = analysis.get('annual_return', 0)
            sharpe_ratio = analysis.get('sharpe_ratio', 0)
            max_drawdown = analysis.get('max_drawdown', 0)
            
            # SPY-plus criteria (conservative)
            if annual_return > 0.09 and sharpe_ratio > 0.8 and max_drawdown > -0.25:
                recommendation['deploy'] = True
                recommendation['confidence'] = 0.8
                recommendation['reasons'].append("Beats SPY with reasonable risk")
            elif annual_return > 0.08 and sharpe_ratio > 0.6:
                recommendation['deploy'] = True
                recommendation['confidence'] = 0.6
                recommendation['reasons'].append("Modest improvement over SPY")
            else:
                recommendation['reasons'].append("Insufficient improvement over SPY")
        
        return recommendation


def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description="Symphony Trading System Runner - Complete Suite")
    
    # Basic operations
    parser.add_argument('--config', '-c', type=str, help='Path to symphony configuration file')
    parser.add_argument('--create-sample', action='store_true', help='Create sample symphony configuration')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--execute', action='store_true', help='Run single execution')
    parser.add_argument('--full-analysis', action='store_true', help='Run complete analysis with visualizations')
    
    # Advanced features
    parser.add_argument('--forecast', action='store_true', help='Run forecasting analysis')
    parser.add_argument('--optimize', action='store_true', help='Optimize symphony parameters')
    parser.add_argument('--create-spy-plus', action='store_true', help='Create SPY-plus strategies')
    parser.add_argument('--production-pipeline', action='store_true', help='Run production readiness pipeline')
    
    # Composer integration
    parser.add_argument('--convert-composer', type=str, help='Convert Composer DSL file')
    parser.add_argument('--composer-csv', type=str, help='Composer backtest CSV for validation')
    
    # Configuration parameters
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--frequency', type=str, default='monthly', choices=['daily', 'weekly', 'monthly'], 
                       help='Rebalance frequency')
    parser.add_argument('--benchmark', type=str, default='SPY', help='Benchmark symbol')
    parser.add_argument('--forecast-days', type=int, default=252, help='Number of days to forecast')
    parser.add_argument('--risk-level', type=str, default='low', choices=['low', 'medium', 'high'],
                       help='Risk level for production pipeline')
    
    # Output control
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--visualize-only', action='store_true', help='Create visualizations from existing results')
    
    args = parser.parse_args()
    
    runner = SymphonyRunner()
    
    try:
        # Create sample configuration if requested
        if args.create_sample:
            runner.create_sample_symphony()
            print("✅ Sample symphony created.")
            print("💡 Quick start commands:")
            print("   Full analysis:      python symphony_runner.py --config sample_symphony_v2.json --full-analysis")
            print("   Forecast:          python symphony_runner.py --config sample_symphony_v2.json --forecast")
            print("   Optimize:          python symphony_runner.py --config sample_symphony_v2.json --optimize")
            print("   SPY-plus creation: python symphony_runner.py --create-spy-plus")
            print("   Production ready:  python symphony_runner.py --config sample_symphony_v2.json --production-pipeline")
            return
        
        # Convert Composer symphony
        if args.convert_composer:
            if not os.path.exists(args.convert_composer):
                print(f"❌ Composer DSL file not found: {args.convert_composer}")
                return
            
            with open(args.convert_composer, 'r') as f:
                composer_dsl = f.read()
            
            result = runner.convert_composer_symphony(composer_dsl, args.composer_csv, args.output_dir)
            
            if result.get('parsing', {}).get('status') == 'success':
                print(f"✅ Conversion completed: {result.get('converted_file', 'converted symphony')}")
                if 'reconciliation' in result:
                    reconciliation_status = result['reconciliation']['reconciliation'].get('reconciliation_status', 'Unknown')
                    print(f"📊 Reconciliation: {reconciliation_status}")
            return
        
        # Create SPY-plus strategies
        if args.create_spy_plus:
            print("📊 Loading market data for SPY-plus strategies...")
            
            # Load market data for common assets
            symbols = ['SPY', 'QQQ', 'TLT', 'XLK', 'XLF', 'XLE', 'IWM']
            
            market_data = runner.data_pipeline.get_multiple_symbols(
                symbols, 'daily', start_date=args.start_date, 
                end_date=args.end_date, use_cache=True
            )
            
            if market_data:
                result = runner.create_spy_plus_strategies(
                    market_data, args.start_date, args.end_date or datetime.now().strftime('%Y-%m-%d'),
                    args.output_dir
                )
                
                if result.get('status') == 'success':
                    print(f"✅ Created {len(result['strategies'])} SPY-plus strategies")
                    print(f"🎯 Recommended: {len(result['summary']['recommended_strategies'])}")
            else:
                print("❌ Could not load market data for SPY-plus strategies")
            return
        
        # Handle visualization-only mode
        if args.visualize_only:
            results_file = os.path.join(args.output_dir, "backtest_results.csv")
            analysis_file = os.path.join(args.output_dir, "performance_analysis.json")
            
            if not os.path.exists(results_file) or not os.path.exists(analysis_file):
                print(f"❌ Results files not found in {args.output_dir}")
                print("💡 Run a backtest first, then use --visualize-only")
                return
            
            # Load existing results
            results = pd.read_csv(results_file, parse_dates=['date'])
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            # Convert allocation strings back to dictionaries
            results['allocation'] = results['allocation'].apply(eval)
            
            runner.create_visualizations(results, analysis, args.output_dir, args.benchmark, 
                                       args.start_date, args.end_date)
            return
        
        # Load symphony configuration for operations that need it
        if not args.config:
            print("❌ Please specify a symphony configuration file with --config")
            print("💡 Use --create-sample to create a sample configuration")
            return
        
        symphony_config = runner.load_symphony_config(args.config)
        
        if not runner.validate_symphony(symphony_config):
            print("❌ Symphony configuration validation failed")
            return
        
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Production pipeline (comprehensive analysis)
        if args.production_pipeline:
            result = runner.run_production_pipeline(
                symphony_config, None, args.risk_level, args.output_dir
            )
            
            if result.get('status') == 'success':
                rec = result['recommendation']
                print(f"\n🎉 Production Pipeline Completed!")
                print(f"📋 Recommendation: {rec.get('recommendation', 'UNKNOWN')}")
                print(f"🎯 Confidence: {rec.get('confidence', 0):.1%}")
                print(f"💡 Next Steps: {rec.get('next_steps', [])}")
            return
        
        # Forecasting analysis
        if args.forecast:
            # First run backtest to get data for forecasting
            print("📊 Running backtest for forecasting base...")
            backtest_results = runner.run_backtest(symphony_config, args.start_date, end_date, args.frequency)
            
            forecast_result = runner.run_forecasting_analysis(
                symphony_config, backtest_results, args.forecast_days, args.output_dir
            )
            
            if forecast_result.get('status') == 'success':
                mc_forecast = forecast_result['monte_carlo']
                print(f"\n🔮 Forecast Results:")
                print(f"Expected Annual Return: {mc_forecast.expected_annual_return:.2%}")
                print(f"Expected Sharpe Ratio:  {mc_forecast.expected_sharpe:.2f}")
                print(f"Probability Beat Bench: {mc_forecast.probability_of_beating_benchmark:.1%}")
            return
        
        # Parameter optimization
        if args.optimize:
            # Prepare market data
            market_data = runner.data_manager.prepare_symphony_data(symphony_config, args.start_date, end_date)
            
            optimization_result = runner.optimize_symphony(
                symphony_config, market_data, args.start_date, end_date, args.output_dir
            )
            
            if optimization_result.get('status') == 'success':
                best_metrics = optimization_result['result'].best_metrics
                print(f"\n🔧 Optimization Results:")
                print(f"Best Sharpe Ratio:    {best_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"Best Annual Return:   {best_metrics.get('annual_return', 0):.2%}")
                print(f"Optimized symphony saved in results")
            return
        
        # Full analysis (backtest + visualizations)
        if args.full_analysis:
            full_results = runner.run_full_analysis(
                symphony_config, args.start_date, end_date, args.frequency, 
                args.benchmark, args.output_dir
            )
            
            print(f"\n🎉 Full analysis completed!")
            print(f"📁 Results saved to: {args.output_dir}")
            print(f"🌐 Open {args.output_dir}/charts/interactive_dashboard.html for interactive analysis")
        
        # Standard backtest
        elif args.backtest:
            results = runner.run_backtest(symphony_config, args.start_date, end_date, args.frequency)
            analysis = runner.analyze_backtest_results(results)
            runner.export_results(results, analysis, args.output_dir)
            
            # Create visualizations unless disabled
            if not args.no_charts:
                runner.create_visualizations(results, analysis, args.output_dir, 
                                          args.benchmark, args.start_date, end_date)
        
        # Single execution
        elif args.execute:
            execution_result = runner.run_single_execution(symphony_config)
            
            # Save execution result
            os.makedirs(args.output_dir, exist_ok=True)
            execution_file = os.path.join(args.output_dir, "latest_execution.json")
            with open(execution_file, 'w') as f:
                json.dump(execution_result, f, indent=2, default=str)
            print(f"💾 Execution result saved: {execution_file}")
        
        else:
            print("❌ Please specify an operation:")
            print("   --full-analysis, --backtest, --execute, --forecast, --optimize")
            print("   --create-spy-plus, --production-pipeline, --convert-composer")
            print("💡 Use --help for detailed usage information")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
