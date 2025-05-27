#!/usr/bin/env python3
"""
Symphony Runner - Main Integration Script

Ties together all components to run symphony backtests and live execution.
This is your main entry point for testing the new symphony system.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Import our custom modules
from data_pipeline import MarketDataPipeline, SymphonyDataManager, DataConfig
from symphony_engine import SymphonyEngine, SymphonyBacktester
from metrics_module import SymphonyMetrics
from symphony_visualizer import SymphonyVisualizer, fetch_benchmark_data

class SymphonyRunner:
    """Main orchestrator for symphony operations"""
    
    def __init__(self, config_path: str = None):
        # Initialize components
        self.data_config = DataConfig()
        self.data_pipeline = MarketDataPipeline(self.data_config)
        self.data_manager = SymphonyDataManager(self.data_pipeline)
        self.engine = SymphonyEngine()
        self.backtester = SymphonyBacktester()
        self.visualizer = SymphonyVisualizer()
        
        print("🎵 Symphony Runner initialized")
        print(f"📊 Data source: {self.data_config.source}")
        print(f"💾 Cache directory: {self.data_config.cache_dir}")
        print(f"🎨 Visualization enabled")
    
    def load_symphony_config(self, config_path: str) -> dict:
        """Load symphony configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Loaded symphony: {config.get('name', 'Unknown')}")
            return config
        except Exception as e:
            raise ValueError(f"Error loading symphony config: {e}")
    
    def create_sample_symphony(self, output_path: str = "sample_symphony_v2.json"):
        """Create a sample symphony configuration file"""
        
        sample_config = {
            "name": "Momentum Quality Strategy",
            "description": "Buy top 3 momentum stocks when market is bullish, defensive when bearish",
            "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "TLT"],
            "rebalance_frequency": "monthly",
            
            "logic": {
                "conditions": [
                    {
                        "id": "market_momentum_check",
                        "type": "if_statement", 
                        "condition": {
                            "metric": "cumulative_return",
                            "asset_1": "SPY",
                            "operator": "greater_than",
                            "asset_2": {"type": "fixed_value", "value": 0.0},
                            "lookback_days": 60
                        },
                        "if_true": "momentum_allocation",
                        "if_false": "defensive_allocation"
                    }
                ],
                
                "allocations": {
                    "momentum_allocation": {
                        "type": "sort_and_weight",
                        "sort": {
                            "metric": "cumulative_return", 
                            "lookback_days": 90,
                            "direction": "top",
                            "count": 3
                        },
                        "weighting": {
                            "method": "inverse_volatility",
                            "lookback_days": 30
                        }
                    },
                    
                    "defensive_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "TLT": 0.7,
                            "SPY": 0.3
                        }
                    }
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"📝 Created sample symphony: {output_path}")
        return sample_config
    
    def validate_symphony(self, symphony_config: dict) -> bool:
        """Validate symphony configuration"""
        
        print("🔍 Validating symphony configuration...")
        
        required_fields = ['name', 'universe', 'logic']
        for field in required_fields:
            if field not in symphony_config:
                print(f"❌ Missing required field: {field}")
                return False
        
        universe = symphony_config['universe']
        if not isinstance(universe, list) or len(universe) == 0:
            print("❌ Universe must be a non-empty list")
            return False
        
        logic = symphony_config['logic']
        if 'allocations' not in logic:
            print("❌ No allocations defined in logic")
            return False
        
        print("✅ Symphony configuration is valid")
        return True
    
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


def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description="Symphony Trading System Runner")
    parser.add_argument('--config', '-c', type=str, help='Path to symphony configuration file')
    parser.add_argument('--create-sample', action='store_true', help='Create sample symphony configuration')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--execute', action='store_true', help='Run single execution')
    parser.add_argument('--full-analysis', action='store_true', help='Run complete analysis with visualizations')
    parser.add_argument('--visualize-only', action='store_true', help='Create visualizations from existing results')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--frequency', type=str, default='monthly', choices=['daily', 'weekly', 'monthly'], 
                       help='Rebalance frequency')
    parser.add_argument('--benchmark', type=str, default='SPY', help='Benchmark symbol for comparison')
    parser.add_argument('--output-dir', type=str, default='./backtest_results', help='Output directory for results')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    
    args = parser.parse_args()
    
    runner = SymphonyRunner()
    
    try:
        # Create sample configuration if requested
        if args.create_sample:
            runner.create_sample_symphony()
            print("✅ Sample symphony created.")
            print("💡 Quick start commands:")
            print("   Full analysis:  python symphony_runner.py --config sample_symphony_v2.json --full-analysis")
            print("   Just backtest:  python symphony_runner.py --config sample_symphony_v2.json --backtest")
            return
        
        # Load symphony configuration
        if not args.config and not args.visualize_only:
            print("❌ Please specify a symphony configuration file with --config")
            print("💡 Use --create-sample to create a sample configuration")
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
        
        symphony_config = runner.load_symphony_config(args.config)
        
        if not runner.validate_symphony(symphony_config):
            print("❌ Symphony configuration validation failed")
            return
        
        # Run full analysis (backtest + visualizations)
        if args.full_analysis:
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
            
            full_results = runner.run_full_analysis(
                symphony_config, args.start_date, end_date, args.frequency, 
                args.benchmark, args.output_dir
            )
            
            print(f"\n🎉 Full analysis completed!")
            print(f"📁 Results saved to: {args.output_dir}")
            print(f"🌐 Open {args.output_dir}/charts/interactive_dashboard.html for interactive analysis")
        
        # Run backtest only
        elif args.backtest:
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
            
            results = runner.run_backtest(
                symphony_config, args.start_date, end_date, args.frequency
            )
            
            analysis = runner.analyze_backtest_results(results)
            runner.export_results(results, analysis, args.output_dir)
            
            # Create visualizations unless disabled
            if not args.no_charts:
                runner.create_visualizations(results, analysis, args.output_dir, 
                                          args.benchmark, args.start_date, end_date)
        
        # Run single execution
        elif args.execute:
            execution_result = runner.run_single_execution(symphony_config)
            
            # Save execution result
            os.makedirs(args.output_dir, exist_ok=True)
            execution_file = os.path.join(args.output_dir, "latest_execution.json")
            with open(execution_file, 'w') as f:
                json.dump(execution_result, f, indent=2, default=str)
            print(f"💾 Execution result saved: {execution_file}")
        
        else:
            print("❌ Please specify one of: --full-analysis, --backtest, --execute, or --visualize-only")
            print("💡 Use --help for usage information")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
