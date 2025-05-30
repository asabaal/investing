#!/usr/bin/env python3
"""
Symphony Backtester - Integration script for Symphony Trading System

This script provides a command-line interface to perform comprehensive analysis of
symphony trading strategies, including backtesting, benchmark comparison, forecasting,
and risk analysis.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# Add the current directory to Python's path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import local modules
from symphony_analyzer import SymphonyAnalyzer
from prophet_forecasting import StockForecast
from alpha_vantage_api import AlphaVantageClient
from report_generator import ReportGenerator

# Configure logging with improved setup
def setup_logging():
    """Set up logging configuration with proper file and console output"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/symphony_backtester_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging with both file and console handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    
    # Set formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set log level and add handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Return logger for this module
    return logging.getLogger(__name__)

# Set up logging immediately
logger = setup_logging()

def run_full_analysis(
    symphony_file, 
    benchmark_symbol="SPY", 
    forecast_periods=30,
    start_date=None,
    end_date=None,
    output_dir="symphony_analysis_results",
    save_plots=True
):
    """
    Run a complete analysis of a symphony:
    - Load and validate the symphony
    - Run a backtest
    - Compare to benchmark
    - Generate a forecast
    
    Args:
        symphony_file (str): Path to symphony JSON file
        benchmark_symbol (str): Symbol for benchmark comparison
        forecast_periods (int): Number of days to forecast
        start_date (str): Start date for analysis (YYYY-MM-DD format)
        end_date (str): End date for analysis (YYYY-MM-DD format)
        output_dir (str): Directory to save output files
        save_plots (bool): Whether to save visualization plots
        
    Returns:
        dict: Results from all analyses
    """
    logger.info(f"Starting full analysis for symphony: {symphony_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract symphony name from filename
    symphony_name = os.path.splitext(os.path.basename(symphony_file))[0]
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 1 year lookback
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Load symphony
    logger.info(f"Loading symphony from {symphony_file}")
    analyzer = SymphonyAnalyzer(symphony_file)
    
    try:
        # Run analysis
        logger.info(f"Running analysis with start date {start_date} and end date {end_date}")
        results = analyzer.analyze_symphony(
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol=benchmark_symbol,
            forecast_days=forecast_periods
        )
        
        # Save results to JSON
        results_file = os.path.join(output_dir, f"{symphony_name}_analysis.json")
        with open(results_file, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json.dump(clean_for_json(results), f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {results_file}")
        return results
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}

def clean_for_json(obj):
    """
    Convert objects to JSON-serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd and hasattr(pd, 'Series') and isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    elif pd and hasattr(pd, 'DataFrame') and isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict(orient='records'))
    else:
        return obj

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Symphony Trading System Backtester')
    
    parser.add_argument('symphony_file', help='Path to the symphony JSON file')
    parser.add_argument('--benchmark', '-b', default='SPY', help='Benchmark symbol (default: SPY)')
    parser.add_argument('--forecast', '-f', type=int, default=30, help='Number of days to forecast (default: 30)')
    parser.add_argument('--start-date', '-s', help='Start date for backtest (YYYY-MM-DD format)')
    parser.add_argument('--end-date', '-e', help='End date for backtest (YYYY-MM-DD format)')
    parser.add_argument('--output-dir', '-o', default='symphony_analysis_results', help='Output directory for results')
    parser.add_argument('--html-report', '-r', action='store_true', help='Generate HTML report')
    parser.add_argument('--enhanced-report', '-er', action='store_true', help='Generate enhanced HTML report with visualizations')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main function to run the symphony backtester"""
    # Parse command line arguments
    args = parse_args()
    
    # Adjust log level if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Symphony Backtester starting - Version 1.2.0")
    logger.info(f"Command line arguments: {args}")
    
    if not os.path.exists(args.symphony_file):
        logger.error(f"Symphony file not found: {args.symphony_file}")
        sys.exit(1)
    
    # Run analysis
    try:
        logger.info("Starting full analysis...")
        results = run_full_analysis(
            symphony_file=args.symphony_file,
            benchmark_symbol=args.benchmark,
            forecast_periods=args.forecast,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        
        # Generate HTML report if requested
        if (args.html_report or args.enhanced_report) and results and "error" not in results:
            report_generator = ReportGenerator(args.output_dir)
            
            if args.enhanced_report:
                report_file = report_generator.generate_enhanced_report(results)
                logger.info(f"Enhanced HTML report generated: {report_file}")
                print(f"\nEnhanced HTML report generated: {report_file}")
            else:
                report_file = report_generator.generate_basic_report(results)
                logger.info(f"Basic HTML report generated: {report_file}")
                print(f"\nHTML report generated: {report_file}")
        
        # Print a summary to console
        print("\n" + "="*50)
        print(f"SYMPHONY ANALYSIS SUMMARY: {results.get('symphony_info', {}).get('name', 'Unknown')}")
        print("="*50)
        
        if 'backtest_results' in results and 'backtest_summary' in results['backtest_results']:
            summary = results['backtest_results']['backtest_summary']
            print("\nBACKTEST RESULTS:")
            print(f"Total Return: {summary.get('total_return', 0)*100:.2f}%")
            print(f"Annualized Return: {summary.get('annualized_return', 0)*100:.2f}%")
            print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {summary.get('max_drawdown', 0)*100:.2f}%")
        
        if 'benchmark_comparison' in results and 'benchmark_symbol' in results['benchmark_comparison']:
            benchmark = results['benchmark_comparison']
            print(f"\nBENCHMARK COMPARISON ({args.benchmark}):")
            print(f"Benchmark Return: {benchmark.get('benchmark_return', 0)*100:.2f}%")
            print(f"Portfolio Return: {benchmark.get('portfolio_return', 0)*100:.2f}%")
            print(f"Excess Return: {benchmark.get('excess_return', 0)*100:.2f}%")
            print(f"Beta: {benchmark.get('beta', 0):.2f}")
            print(f"Information Ratio: {benchmark.get('information_ratio', 0):.2f}")
        
        if 'forecasts' in results and 'average_30d_forecast' in results['forecasts']:
            print("\nFORECAST SUMMARY:")
            print(f"30-Day Average: {results['forecasts']['average_30d_forecast']:.2f}%")
            print(f"Sentiment: {results['forecasts'].get('forecast_sentiment', 'unknown').title()}")
        
        logger.info("Analysis complete. Results saved to the specified output directory.")
        print("\nAnalysis complete. Results saved to the specified output directory.")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        print(f"\nError during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
