"""
Symphony CLI - Command Line Interface for Symphony Analysis

This script provides a command-line interface for analyzing, testing,
and monitoring Composer symphonies using the components we've built.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from alpha_vantage_api import AlphaVantageClient
from composer_symphony import Symphony, SymbolList
from symphony_analyzer import SymphonyAnalyzer
from symphony_simulation import SymphonySimulator
from prophet_forecasting import StockForecast, ProphetEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("symphony_cli")

def load_symphony(file_path: str) -> Symphony:
    """Load a symphony from a file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        symphony = Symphony.from_dict(data)
        logger.info(f"Loaded symphony: {symphony.name}")
        return symphony
    except Exception as e:
        logger.error(f"Failed to load symphony from {file_path}: {str(e)}")
        raise

def save_symphony(symphony: Symphony, file_path: str):
    """Save a symphony to a file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(symphony.to_dict(), f, indent=2)
        logger.info(f"Saved symphony to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save symphony to {file_path}: {str(e)}")
        raise

def save_results(results: Dict, file_path: str):
    """Save analysis results to a file."""
    try:
        # Convert any non-serializable objects
        def convert_for_json(obj):
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert_for_json)
        logger.info(f"Saved results to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {file_path}: {str(e)}")
        raise

def analyze_command(args):
    """Run the analyze command."""
    # Initialize client
    client = AlphaVantageClient(api_key=args.api_key)
    
    # Load symphony
    symphony = load_symphony(args.symphony)
    
    # Create analyzer
    analyzer = SymphonyAnalyzer(client)
    
    # Set date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start_date
    if not start_date:
        # Default to 1 year ago
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Run analysis
    logger.info(f"Analyzing symphony {symphony.name} from {start_date} to {end_date}")
    results = analyzer.analyze_symphony(
        symphony,
        start_date,
        end_date,
        scenarios=args.scenarios.split(',') if args.scenarios else None,
        forecast_days=args.forecast_days,
        benchmark_symbol=args.benchmark
    )
    
    # Save results if output file specified
    if args.output:
        save_results(results, args.output)
    
    # Print summary
    print(f"\nAnalysis Results for {symphony.name}")
    print("=" * 50)
    
    # Print backtest results
    if 'backtest_results' in results and results['backtest_results'].get('success', False):
        summary = results['backtest_results']['backtest_summary']
        print("\nBacktest Summary:")
        print(f"  Total Return: {summary['total_return']:.2%}")
        print(f"  Annual Return: {summary['annual_return']:.2%}")
        print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
    
    # Print forecast summary
    if 'forecasts' in results and 'weighted_symphony_forecast' in results['forecasts']:
        forecast = results['forecasts']['weighted_symphony_forecast']
        print("\nForecast Summary:")
        for period, data in forecast.items():
            print(f"  {period}: {data.get('weighted_percent_change', 0):.2%}")
    
    # Print benchmark comparison
    if 'benchmark_comparison' in results and 'error' not in results['benchmark_comparison']:
        comparison = results['benchmark_comparison']
        print(f"\nBenchmark Comparison ({comparison['benchmark_symbol']}):")
        print(f"  Benchmark Return: {comparison['benchmark_return']:.2%}")
        print(f"  Excess Return: {comparison['excess_return']:.2%}")
        
        if comparison['beta'] is not None:
            print(f"  Beta: {comparison['beta']:.2f}")
        
        if comparison['information_ratio'] is not None:
            print(f"  Information Ratio: {comparison['information_ratio']:.2f}")
    
    # Print risk analysis
    if 'risk_analysis' in results:
        risk = results['risk_analysis']
        print("\nRisk Analysis:")
        print(f"  Volatility: {risk.get('volatility', 0):.2%}")
        print(f"  Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}")
        print(f"  Concentration: {risk.get('concentration', 0):.2f}")
    
    return results

def optimize_command(args):
    """Run the optimize command."""
    # Initialize client
    client = AlphaVantageClient(api_key=args.api_key)
    
    # Load symphony
    symphony = load_symphony(args.symphony)
    
    # Create simulator
    simulator = SymphonySimulator(client)
    
    # Set date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start_date
    if not start_date:
        # Default to 1 year ago
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Generate parameter space or load from file
    if args.param_space:
        with open(args.param_space, 'r') as f:
            parameter_space = json.load(f)
    else:
        parameter_space = simulator.variation_generator.create_parameter_space(symphony)
    
    # Run variation test
    logger.info(f"Running variation test for {symphony.name} with {args.variations} variations")
    results = simulator.run_variation_test(
        symphony,
        start_date,
        end_date,
        num_variations=args.variations,
        rebalance_frequency=args.rebalance
    )
    
    # Find best variation
    best_variation = None
    best_metric = -float('inf')
    metric = args.metric
    
    for var in results.get('variations', []):
        if metric in var and var[metric] > best_metric:
            best_metric = var[metric]
            best_variation = var
    
    # Save optimized symphony if output path provided
    if args.output and best_variation:
        # Find the Symphony object for the best variation
        best_symphony_name = best_variation['variation']
        best_symphony = None
        
        for i, var_result in enumerate(results.get('variations', [])):
            if var_result['variation'] == best_symphony_name:
                if i < len(simulator.variation_generator.variations):
                    best_symphony = simulator.variation_generator.variations[i]
                break
        
        if best_symphony:
            save_symphony(best_symphony, args.output)
    
    # Print summary
    print(f"\nOptimization Results for {symphony.name}")
    print("=" * 50)
    
    print(f"\nTested {len(results.get('variations', []))} variations")
    print(f"Optimization metric: {metric}")
    
    if best_variation:
        print(f"\nBest Variation: {best_variation['variation']}")
        print(f"  {metric}: {best_variation[metric]:.4f}")
        print(f"  Total Return: {best_variation['total_return']:.2%}")
        print(f"  Annual Return: {best_variation['annual_return']:.2%}")
        print(f"  Sharpe Ratio: {best_variation['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {best_variation['max_drawdown']:.2%}")
    
    # Print parameter sensitivity
    print("\nParameter Sensitivity:")
    for op_type, params in results.get('parameter_sensitivity', {}).get('operators', {}).items():
        for param, sensitivity in params.items():
            print(f"  {op_type}.{param}: {sensitivity:.4f}")
    
    return results

def test_command(args):
    """Run the test command."""
    # Initialize client
    client = AlphaVantageClient(api_key=args.api_key)
    
    # Load symphony
    symphony = load_symphony(args.symphony)
    
    # Create simulator
    simulator = SymphonySimulator(client)
    
    # Run scenario test
    logger.info(f"Running scenario test for {symphony.name}")
    results = simulator.run_scenario_test(symphony)
    
    # Save results if output file specified
    if args.output:
        save_results(results, args.output)
    
    # Print summary
    print(f"\nScenario Test Results for {symphony.name}")
    print("=" * 50)
    
    # Print scenario results
    for scenario in results.get('scenario_results', []):
        print(f"\nScenario: {scenario['scenario']}")
        print(f"  Description: {scenario.get('description', 'N/A')}")
        
        if 'error' in scenario:
            print(f"  Error: {scenario['error']}")
            continue
        
        print(f"  Total Return: {scenario.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {scenario.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {scenario.get('max_drawdown', 0):.2%}")
    
    return results

def forecast_command(args):
    """Run the forecast command."""
    # Initialize client
    client = AlphaVantageClient(api_key=args.api_key)
    
    # Load symbols from file or args
    if args.symbols_file:
        with open(args.symbols_file, 'r') as f:
            symbols = json.load(f)
    elif args.symbols:
        symbols = args.symbols.split(',')
    else:
        logger.error("No symbols provided. Use --symbols or --symbols-file")
        sys.exit(1)
    
    # Create forecaster
    if args.ensemble:
        forecaster = ProphetEnsemble(client)
        
        # Generate forecasts
        logger.info(f"Generating ensemble forecasts for {len(symbols)} symbols")
        results = {}
        
        for symbol in symbols:
            try:
                result = forecaster.forecast_ensemble(
                    symbol,
                    days=args.days,
                    num_models=args.models
                )
                summary = forecaster.get_ensemble_forecast_summary(result)
                results[symbol] = summary
            except Exception as e:
                logger.error(f"Failed to forecast {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
    else:
        forecaster = StockForecast(client)
        
        # Generate forecasts
        logger.info(f"Generating forecasts for {len(symbols)} symbols")
        results = {}
        
        for symbol in symbols:
            try:
                df, forecast, model = forecaster.forecast(
                    symbol,
                    days=args.days
                )
                
                summary = forecaster.get_forecast_summary(forecast)
                results[symbol] = summary
            except Exception as e:
                logger.error(f"Failed to forecast {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
    
    # Save results if output file specified
    if args.output:
        save_results(results, args.output)
    
    # Print summary
    print(f"\nForecast Summary ({args.days} days)")
    print("=" * 50)
    
    for symbol, forecast in results.items():
        print(f"\nSymbol: {symbol}")
        
        if 'error' in forecast:
            print(f"  Error: {forecast['error']}")
            continue
        
        for period in [f"{args.days}_day"]:
            if period in forecast:
                data = forecast[period]
                print(f"  {period.replace('_', ' ')}:")
                print(f"    Date: {data.get('date', 'N/A')}")
                print(f"    Forecast: {data.get('forecast', 0):.2f}")
                print(f"    Change: {data.get('percent_change', 0):.2%}")
                print(f"    Lower Bound: {data.get('lower_bound', 0):.2f}")
                print(f"    Upper Bound: {data.get('upper_bound', 0):.2f}")
    
    return results

def create_command(args):
    """Run the create command."""
    # Create basic symphony
    universe = SymbolList(args.universe.split(','))
    symphony = Symphony(args.name, args.description, universe)
    
    # Save symphony
    save_symphony(symphony, args.output)
    
    print(f"Created symphony {args.name} with {len(universe)} symbols")
    print(f"Saved to {args.output}")
    
    return symphony

def watchlist_command(args):
    """Run the watchlist command."""
    print("Starting Symphony Watchlist GUI...")
    
    # This will be launched in a separate window
    from symphony_watchlist import SymphonyWatchlistApp
    
    # Initialize client
    client = AlphaVantageClient(api_key=args.api_key)
    
    # Create and run app
    app = SymphonyWatchlistApp(client)
    app.run()

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Symphony CLI - Composer Symphony Analysis Tool")
    
    parser.add_argument('--api-key', default=os.environ.get('ALPHA_VANTAGE_API_KEY'),
                       help="Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a symphony')
    analyze_parser.add_argument('symphony', help='Symphony JSON file')
    analyze_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    analyze_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    analyze_parser.add_argument('--benchmark', default='SPY', help='Benchmark symbol')
    analyze_parser.add_argument('--scenarios', help='Comma-separated list of scenarios')
    analyze_parser.add_argument('--forecast-days', type=int, default=30, help='Days to forecast')
    analyze_parser.add_argument('--output', help='Output JSON file for results')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize a symphony')
    optimize_parser.add_argument('symphony', help='Symphony JSON file')
    optimize_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    optimize_parser.add_argument('--variations', type=int, default=10, help='Number of variations to generate')
    optimize_parser.add_argument('--metric', default='sharpe_ratio', 
                              choices=['sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown'],
                              help='Metric to optimize')
    optimize_parser.add_argument('--param-space', help='Parameter space JSON file')
    optimize_parser.add_argument('--rebalance', default='monthly', 
                              choices=['daily', 'weekly', 'monthly', 'quarterly'],
                              help='Rebalance frequency')
    optimize_parser.add_argument('--output', help='Output JSON file for optimized symphony')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a symphony under different scenarios')
    test_parser.add_argument('symphony', help='Symphony JSON file')
    test_parser.add_argument('--scenarios', help='Scenarios JSON file')
    test_parser.add_argument('--output', help='Output JSON file for results')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate forecasts for symbols')
    forecast_parser.add_argument('--symbols', help='Comma-separated list of symbols')
    forecast_parser.add_argument('--symbols-file', help='JSON file with symbols list')
    forecast_parser.add_argument('--days', type=int, default=30, help='Days to forecast')
    forecast_parser.add_argument('--ensemble', action='store_true', help='Use ensemble forecasting')
    forecast_parser.add_argument('--models', type=int, default=5, help='Number of models for ensemble')
    forecast_parser.add_argument('--output', help='Output JSON file for results')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new symphony')
    create_parser.add_argument('name', help='Symphony name')
    create_parser.add_argument('--description', default='', help='Symphony description')
    create_parser.add_argument('--universe', required=True, help='Comma-separated list of symbols')
    create_parser.add_argument('--output', required=True, help='Output JSON file')
    
    # Watchlist command
    watchlist_parser = subparsers.add_parser('watchlist', help='Launch the Symphony Watchlist GUI')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check API key
    if not args.api_key and args.command != 'create':
        print("Error: Alpha Vantage API key is required")
        print("Set it with --api-key or ALPHA_VANTAGE_API_KEY environment variable")
        sys.exit(1)
    
    # Run command
    if args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'optimize':
        optimize_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'forecast':
        forecast_command(args)
    elif args.command == 'create':
        create_command(args)
    elif args.command == 'watchlist':
        watchlist_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
