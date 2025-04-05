"""
Test script for the benchmark comparison function

This script helps test the benchmark comparison functionality
by providing a standalone way to compare portfolio performance
to benchmarks.
"""

import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from alpha_vantage_api import AlphaVantageClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compare_to_benchmark(
    portfolio_history: list,
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
    client: AlphaVantageClient
) -> dict:
    """
    Compare portfolio performance to a benchmark.
    
    Args:
        portfolio_history: List of portfolio snapshots
        benchmark_symbol: Symbol to use as benchmark
        start_date: Start date for comparison
        end_date: End date for comparison
        client: Alpha Vantage client
        
    Returns:
        Dictionary with comparison metrics
    """
    # Check if we have portfolio history
    if not portfolio_history:
        return {'error': 'No portfolio history provided'}
    
    try:
        # Get benchmark data for the same period
        benchmark_data = client.get_daily(benchmark_symbol)
        
        # Parse date strings to datetime
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        
        # Extract benchmark data for the period
        benchmark_data = benchmark_data.loc[
            (benchmark_data.index >= start_date_pd) & 
            (benchmark_data.index <= end_date_pd)
        ]
        
        if benchmark_data.empty:
            return {'error': f'No benchmark data available for {benchmark_symbol}'}
        
        # Calculate benchmark performance
        benchmark_start = benchmark_data['adjusted_close'].iloc[0]
        benchmark_end = benchmark_data['adjusted_close'].iloc[-1]
        benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
        
        # Calculate benchmark daily returns
        benchmark_returns = benchmark_data['adjusted_close'].pct_change().dropna()
        
        # Get portfolio values and dates
        portfolio_dates = [entry['date'] for entry in portfolio_history]
        portfolio_values = [entry['portfolio_value'] for entry in portfolio_history]
        
        # Convert to pandas series for easier handling
        portfolio_series = pd.Series(
            data=portfolio_values,
            index=pd.to_datetime(portfolio_dates)
        )
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        # Verify we have data
        if len(portfolio_returns) == 0:
            return {
                'benchmark_symbol': benchmark_symbol,
                'benchmark_return': benchmark_return,
                'error': 'Not enough portfolio data points to calculate returns'
            }
        
        # Calculate portfolio performance
        portfolio_start = portfolio_values[0]
        portfolio_end = portfolio_values[-1]
        portfolio_return = (portfolio_end - portfolio_start) / portfolio_start
        
        # Calculate excess return
        excess_return = portfolio_return - benchmark_return
        
        # Calculate correlation and beta
        # First, align the returns by date
        # Create a DataFrame with both return series
        returns_df = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        })
        
        # Debug prints to help diagnose alignment issues
        logger.debug(f"Portfolio returns shape: {portfolio_returns.shape}")
        logger.debug(f"Benchmark returns shape: {benchmark_returns.shape}")
        logger.debug(f"Portfolio returns index: {portfolio_returns.index}")
        logger.debug(f"Benchmark returns index: {benchmark_returns.index}")
        
        # Only use dates that are in both series
        common_returns = returns_df.dropna()
        
        # Debug prints for aligned data
        logger.debug(f"Common dates count: {len(common_returns)}")
        
        # Initialize correlation and beta
        correlation = None
        beta = None
        information_ratio = None
        
        # Calculate correlation and beta if we have enough common data points
        if len(common_returns) > 1:
            correlation = common_returns['portfolio'].corr(common_returns['benchmark'])
            
            # Calculate beta (portfolio return to benchmark return)
            cov_matrix = common_returns.cov()
            if cov_matrix.loc['benchmark', 'benchmark'] > 0:
                beta = cov_matrix.loc['portfolio', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            else:
                beta = 0
            
            # Calculate tracking error and information ratio
            if beta is not None:
                # Tracking error is the standard deviation of the active returns
                active_returns = common_returns['portfolio'] - beta * common_returns['benchmark']
                tracking_error = np.std(active_returns) * np.sqrt(252)  # Annualize
                
                if tracking_error > 0:
                    information_ratio = excess_return / tracking_error
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = None
        benchmark_sharpe = None
        
        if len(portfolio_returns) > 1:
            # Assuming 3% annual risk-free rate
            risk_free_rate = 0.03 / 252  # Daily rate
            
            # Portfolio Sharpe
            portfolio_excess_return = portfolio_returns - risk_free_rate
            sharpe_ratio = (portfolio_excess_return.mean() / portfolio_excess_return.std()) * np.sqrt(252)
            
            # Benchmark Sharpe
            if len(benchmark_returns) > 1:
                benchmark_excess_return = benchmark_returns - risk_free_rate
                benchmark_sharpe = (benchmark_excess_return.mean() / benchmark_excess_return.std()) * np.sqrt(252)
        
        return {
            'benchmark_symbol': benchmark_symbol,
            'benchmark_return': benchmark_return,
            'portfolio_return': portfolio_return,
            'excess_return': excess_return,
            'correlation': correlation,
            'beta': beta,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'information_ratio': information_ratio,
            'common_data_points': len(common_returns) if common_returns is not None else 0
        }
        
    except Exception as e:
        logger.error(f"Benchmark comparison error: {str(e)}")
        return {'error': str(e)}

def plot_comparison(
    portfolio_history: list,
    benchmark_data: pd.DataFrame,
    benchmark_symbol: str,
    title: str = None
):
    """
    Plot portfolio performance against benchmark.
    
    Args:
        portfolio_history: List of portfolio snapshots
        benchmark_data: Benchmark price data
        benchmark_symbol: Benchmark symbol
        title: Plot title
    """
    # Extract portfolio dates and values
    portfolio_dates = [pd.to_datetime(entry['date']) for entry in portfolio_history]
    portfolio_values = [entry['portfolio_value'] for entry in portfolio_history]
    
    # Create a Series for easier indexing
    portfolio_series = pd.Series(
        data=portfolio_values,
        index=portfolio_dates
    )
    
    # Normalize both series to start at 100
    portfolio_normalized = portfolio_series / portfolio_series.iloc[0] * 100
    benchmark_normalized = benchmark_data['adjusted_close'] / benchmark_data['adjusted_close'].iloc[0] * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(portfolio_normalized.index, portfolio_normalized.values, label='Portfolio')
    plt.plot(benchmark_normalized.index, benchmark_normalized.values, label=benchmark_symbol)
    
    plt.title(title or f'Portfolio vs {benchmark_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Start = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test benchmark comparison."""
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
        return
    
    # Create API client
    client = AlphaVantageClient(api_key=api_key)
    
    # Create sample portfolio history
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get benchmark data for sample period
    benchmark_symbol = 'SPY'
    try:
        benchmark_data = client.get_daily(benchmark_symbol)
        
        # Parse date strings to datetime
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        
        benchmark_data = benchmark_data.loc[
            (benchmark_data.index >= start_date_pd) & 
            (benchmark_data.index <= end_date_pd)
        ]
        
        if benchmark_data.empty:
            print(f"No benchmark data available for {benchmark_symbol}")
            return
        
        # Create a portfolio that outperforms the benchmark by 5%
        benchmark_dates = benchmark_data.index
        benchmark_prices = benchmark_data['adjusted_close']
        
        # Generate some portfolio values that follow the benchmark plus some alpha
        portfolio_values = []
        
        for date, price in zip(benchmark_dates, benchmark_prices):
            # Start with benchmark price
            value = price
            
            # Add some alpha (outperformance)
            # This is very simplified and just for testing
            alpha_factor = 1.05  # 5% outperformance
            value = value * alpha_factor
            
            # Add to list
            portfolio_values.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': value
            })
        
        # Run comparison
        results = compare_to_benchmark(
            portfolio_values,
            benchmark_symbol,
            start_date,
            end_date,
            client
        )
        
        # Display results
        print("\nBenchmark Comparison Results:")
        print(json.dumps(
            {k: v if not isinstance(v, float) else round(v, 4) for k, v in results.items()},
            indent=2
        ))
        
        # Plot comparison
        plot_comparison(
            portfolio_values,
            benchmark_data,
            benchmark_symbol,
            'Sample Portfolio vs S&P 500 ETF'
        )
    
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main()
