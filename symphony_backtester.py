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

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/symphony_backtester_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def generate_html_report(results, output_dir="symphony_analysis_results"):
    """
    Generate an HTML report from the analysis results.
    
    Args:
        results (dict): Analysis results from run_full_analysis
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated HTML report
    """
    symphony_name = results['symphony_info']['name']
    file_name = results['symphony_info']['file']
    
    # Create HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symphony Analysis: {symphony_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ background-color: #f5f5f5; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin-right: 30px; margin-bottom: 15px; }}
        .metric .value {{ font-size: 24px; font-weight: bold; }}
        .metric .label {{ font-size: 14px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; }}
        table, th, td {{ border: 1px solid #ddd; }}
        th, td {{ padding: 10px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .chart {{ margin-top: 20px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .neutral {{ color: gray; }}
        footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Symphony Analysis Report</h1>
            <p>Symphony: <strong>{symphony_name}</strong></p>
            <p>File: {file_name}</p>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
    
    # Symphony Info Section
    html += f"""
        <div class="section">
            <h2>Symphony Information</h2>
            <div class="metric">
                <div class="label">Name</div>
                <div class="value">{symphony_name}</div>
            </div>
            <div class="metric">
                <div class="label">Symbols</div>
                <div class="value">{len(results['symphony_info']['symbols'])}</div>
            </div>
            <h3>Universe</h3>
            <p>{', '.join(results['symphony_info']['symbols'])}</p>
        </div>
"""
    
    # Backtest Results Section
    if 'backtest_results' in results and 'backtest_summary' in results['backtest_results']:
        summary = results['backtest_results']['backtest_summary']
        total_return = summary.get('total_return', 0) * 100
        annual_return = summary.get('annualized_return', 0) * 100
        max_drawdown = summary.get('max_drawdown', 0) * 100
        
        html += f"""
        <div class="section">
            <h2>Backtest Results</h2>
            <div class="metric">
                <div class="label">Total Return</div>
                <div class="value {get_color_class(total_return)}">{total_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Annualized Return</div>
                <div class="value {get_color_class(annual_return)}">{annual_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Max Drawdown</div>
                <div class="value {get_color_class(-max_drawdown)}">{max_drawdown:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Sharpe Ratio</div>
                <div class="value {get_color_class(summary.get('sharpe_ratio', 0))}">{summary.get('sharpe_ratio', 0):.2f}</div>
            </div>
        </div>
"""
    
    # Benchmark Comparison Section
    if 'benchmark_comparison' in results and 'benchmark_symbol' in results['benchmark_comparison']:
        benchmark = results['benchmark_comparison']
        benchmark_symbol = benchmark['benchmark_symbol']
        benchmark_return = benchmark.get('benchmark_return', 0) * 100
        portfolio_return = benchmark.get('portfolio_return', 0) * 100
        excess_return = benchmark.get('excess_return', 0) * 100
        
        html += f"""
        <div class="section">
            <h2>Benchmark Comparison ({benchmark_symbol})</h2>
            <div class="metric">
                <div class="label">Benchmark Return</div>
                <div class="value {get_color_class(benchmark_return)}">{benchmark_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Portfolio Return</div>
                <div class="value {get_color_class(portfolio_return)}">{portfolio_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Excess Return</div>
                <div class="value {get_color_class(excess_return)}">{excess_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Beta</div>
                <div class="value">{benchmark.get('beta', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="label">Correlation</div>
                <div class="value">{benchmark.get('correlation', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="label">Information Ratio</div>
                <div class="value {get_color_class(benchmark.get('information_ratio', 0))}">{benchmark.get('information_ratio', 0):.2f}</div>
            </div>
        </div>
"""
    
    # Forecast Section
    if 'forecasts' in results:
        forecasts = results['forecasts']
        if 'average_30d_forecast' in forecasts:
            forecast_30d = forecasts['average_30d_forecast']
            sentiment = forecasts.get('forecast_sentiment', 'unknown')
            
            html += f"""
        <div class="section">
            <h2>Forecast</h2>
            <div class="metric">
                <div class="label">30-Day Average Forecast</div>
                <div class="value {get_color_class(forecast_30d)}">{forecast_30d:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Forecast Sentiment</div>
                <div class="value {sentiment}">{sentiment.title()}</div>
            </div>
            <h3>Symbol Forecasts</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>7-Day Forecast</th>
                    <th>30-Day Forecast</th>
                </tr>
"""
            
            for symbol in results['symphony_info']['symbols']:
                if symbol in forecasts:
                    symbol_forecast = forecasts[symbol]
                    forecast_7d = symbol_forecast.get('7_day', {}).get('percent_change', 0)
                    forecast_30d = symbol_forecast.get('30_day', {}).get('percent_change', 0)
                    
                    html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td class="{get_color_class(forecast_7d)}">{forecast_7d:.2f}%</td>
                    <td class="{get_color_class(forecast_30d)}">{forecast_30d:.2f}%</td>
                </tr>
"""
            
            html += """
            </table>
        </div>
"""
    
    # Risk Analysis Section
    if 'risk_analysis' in results:
        risk = results['risk_analysis']
        volatility = risk.get('volatility', 0) * 100
        max_drawdown = risk.get('max_drawdown', 0) * 100
        
        html += f"""
        <div class="section">
            <h2>Risk Analysis</h2>
            <div class="metric">
                <div class="label">Volatility</div>
                <div class="value {get_color_class(-volatility)}">{volatility:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Max Drawdown</div>
                <div class="value {get_color_class(-max_drawdown)}">{max_drawdown:.2f}%</div>
            </div>
            <div class="metric">
                <div class="label">Sortino Ratio</div>
                <div class="value {get_color_class(risk.get('sortino_ratio', 0))}">{risk.get('sortino_ratio', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="label">Downside Deviation</div>
                <div class="value {get_color_class(-risk.get('downside_deviation', 0) * 100)}">{risk.get('downside_deviation', 0) * 100:.2f}%</div>
            </div>
        </div>
"""
    
    # Footer
    html += f"""
        <footer>
            <p>Generated by Symphony Trading System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    report_file = os.path.join(output_dir, f"{symphony_name}_report.html")
    with open(report_file, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {report_file}")
    return report_file

def get_color_class(value):
    """Helper function to determine CSS class based on value"""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"

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
    
    return parser.parse_args()

def main():
    """Main function to run the symphony backtester"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    if not os.path.exists(args.symphony_file):
        logger.error(f"Symphony file not found: {args.symphony_file}")
        sys.exit(1)
    
    # Run analysis
    results = run_full_analysis(
        symphony_file=args.symphony_file,
        benchmark_symbol=args.benchmark,
        forecast_periods=args.forecast,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir
    )
    
    # Generate HTML report if requested
    if args.html_report:
        report_file = generate_html_report(results, args.output_dir)
        print(f"\nHTML report generated: {report_file}")
    
    # Print a summary to console
    print("\n" + "="*50)
    print(f"SYMPHONY ANALYSIS SUMMARY: {results['symphony_info']['name']}")
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
    
    print("\nAnalysis complete. Results saved to the specified output directory.")

if __name__ == "__main__":
    main()
