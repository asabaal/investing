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

class SymphonyRunner:
    """Main orchestrator for symphony operations"""
    
    def __init__(self, config_path: str = None):
        # Initialize components
        self.data_config = DataConfig()
        self.data_pipeline = MarketDataPipeline(self.data_config)
        self.data_manager = SymphonyDataManager(self.data_pipeline)
        self.engine = SymphonyEngine()
        self.backtester = SymphonyBacktester()
        
        print("🎵 Symphony Runner initialized")
        print(f"📊 Data source: {self.data_config.source}")
        print(f"💾 Cache directory: {self.data_config.cache_dir}")
    
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


def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description="Symphony Trading System Runner")
    parser.add_argument('--config', '-c', type=str, help='Path to symphony configuration file')
    parser.add_argument('--create-sample', action='store_true', help='Create sample symphony configuration')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--execute', action='store_true', help='Run single execution')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--frequency', type=str, default='monthly', choices=['daily', 'weekly', 'monthly'], 
                       help='Rebalance frequency')
    parser.add_argument('--output-dir', type=str, default='./backtest_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    runner = SymphonyRunner()
    
    try:
        # Create sample configuration if requested
        if args.create_sample:
            runner.create_sample_symphony()
            print("✅ Sample symphony created. You can now run: python symphony_runner.py --config sample_symphony_v2.json --backtest")
            return
        
        # Load symphony configuration
        if not args.config:
            print("❌ Please specify a symphony configuration file with --config")
            print("💡 Use --create-sample to create a sample configuration")
            return
        
        symphony_config = runner.load_symphony_config(args.config)
        
        if not runner.validate_symphony(symphony_config):
            print("❌ Symphony configuration validation failed")
            return
        
        # Run backtest
        if args.backtest:
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
            
            results = runner.run_backtest(
                symphony_config, args.start_date, end_date, args.frequency
            )
            
            analysis = runner.analyze_backtest_results(results)
            runner.export_results(results, analysis, args.output_dir)
        
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
            print("❌ Please specify either --backtest or --execute")
            print("💡 Use --help for usage information")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
