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

def compare_portfolio_to_benchmark(
    portfolio_returns,
    benchmark_symbol="SPY",
    start_date=None,
    end_date=None,
    client=None,
    save_plot=False,
    plot_filename=None
):
    """
    Compare portfolio returns to a benchmark.
    
    Args:
        portfolio_returns (pd.Series): Portfolio returns series
        benchmark_symbol (str): Symbol for benchmark
        start_date (str): Start date (optional)
        end_date (str): End date (optional)
        client (AlphaVantageClient): Alpha Vantage client (optional)
        save_plot (bool): Whether to save the plot to file
        plot_filename (str): Filename to save the plot (default: benchmark_comparison.png)
        
    Returns:
        dict: Comparison metrics between portfolio and benchmark
    """
    # Initialize API client if not provided
    if client is None:
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return {'error': 'No API key provided or found in environment'}
        client = AlphaVantageClient(api_key=api_key)
    
    try:
        # Get benchmark data
        benchmark_data = client.get_daily(benchmark_symbol)
        
        # Filter by date range if provided
        if start_date and end_date:
            start_date_pd = pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date)
            benchmark_data = benchmark_data.loc[(benchmark_data.index >= start_date_pd) & 
                                             (benchmark_data.index <= end_date_pd)]
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_data['adjusted_close'].pct_change().dropna()
        
        # Ensure portfolio_returns is a pandas Series with datetime index
        if not isinstance(portfolio_returns, pd.Series):
            try:
                portfolio_returns = pd.Series(portfolio_returns)
            except:
                return {'error': 'Portfolio returns must be a pandas Series or convertible to one'}
        
        # Create DataFrames for alignment
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
        
        if len(common_returns) == 0:
            return {'error': 'No common dates between portfolio and benchmark returns'}
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + common_returns['portfolio']).cumprod()
        benchmark_cumulative = (1 + common_returns['benchmark']).cumprod()
        
        # Calculate total returns
        portfolio_return = portfolio_cumulative.iloc[-1] - 1
        benchmark_return = benchmark_cumulative.iloc[-1] - 1
        
        # Calculate excess return
        excess_return = portfolio_return - benchmark_return
        
        # Calculate correlation
        correlation = common_returns['portfolio'].corr(common_returns['benchmark'])
        
        # Calculate beta
        cov_matrix = common_returns.cov()
        beta = cov_matrix.loc['portfolio', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
        
        # Calculate tracking error and information ratio
        active_returns = common_returns['portfolio'] - beta * common_returns['benchmark']
        tracking_error = np.std(active_returns) * np.sqrt(252)  # Annualize
        
        if tracking_error > 0:
            information_ratio = excess_return / tracking_error
        else:
            information_ratio = 0
        
        # Calculate risk-adjusted metrics
        # Assuming 3% annual risk-free rate
        risk_free_rate = 0.03 / 252  # Daily rate
        
        # Portfolio Sharpe
        portfolio_excess_return = common_returns['portfolio'] - risk_free_rate
        sharpe_ratio = (portfolio_excess_return.mean() / portfolio_excess_return.std()) * np.sqrt(252)
        
        # Benchmark Sharpe
        benchmark_excess_return = common_returns['benchmark'] - risk_free_rate
        benchmark_sharpe = (benchmark_excess_return.mean() / benchmark_excess_return.std()) * np.sqrt(252)
        
        # Generate visualization
        plot_comparison(portfolio_cumulative, benchmark_cumulative, benchmark_symbol, 
                      save_plot=save_plot, plot_filename=plot_filename)
        
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
            'common_data_points': len(common_returns)
        }
    
    except Exception as e:
        logger.error(f"Benchmark comparison error: {str(e)}")
        return {'error': str(e)}

def plot_comparison(
    portfolio_cumulative, 
    benchmark_cumulative, 
    benchmark_symbol, 
    save_plot=False, 
    plot_filename=None
):
    """
    Plot the portfolio vs benchmark performance.
    
    Args:
        portfolio_cumulative (pd.Series): Portfolio cumulative returns
        benchmark_cumulative (pd.Series): Benchmark cumulative returns
        benchmark_symbol (str): Symbol of the benchmark
        save_plot (bool): Whether to save the plot to file
        plot_filename (str): Filename to save the plot (default: benchmark_comparison.png)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative, label='Portfolio')
    plt.plot(benchmark_cumulative, label=f'Benchmark ({benchmark_symbol})')
    plt.title('Portfolio vs Benchmark Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    if save_plot:
        filename = plot_filename or "benchmark_comparison.png"
        plt.savefig(filename)
        logger.info(f"Saved benchmark comparison plot to {filename}")
    else:
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"Could not display plot: {e}. Try using save_plot=True instead.")

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

def plot_comparison_interactive(
    portfolio_history: list,
    benchmark_data: pd.DataFrame,
    benchmark_symbol: str,
    title: str = None
):
    """
    Plot portfolio performance against benchmark with Plotly.
    
    Args:
        portfolio_history: List of portfolio snapshots
        benchmark_data: Benchmark price data
        benchmark_symbol: Benchmark symbol
        title: Plot title
    """
    try:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not installed. Using static plot instead.")
        plot_comparison(portfolio_history, benchmark_data, benchmark_symbol, title)
        return
    
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
    
    # Create the figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=portfolio_normalized.index, y=portfolio_normalized.values,
                 name='Portfolio'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=benchmark_normalized.index, y=benchmark_normalized.values,
                 name=benchmark_symbol),
        secondary_y=False,
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Normalized Value (Start = 100)", secondary_y=False)
    
    # Set title
    fig.update_layout(title_text=title or f'Portfolio vs {benchmark_symbol}')
    
    # Show the figure
    fig.show()

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
        
        # Create a portfolio that equals the benchmark (identical performance)
        benchmark_dates = benchmark_data.index
        benchmark_prices = benchmark_data['adjusted_close']
        
        # Generate portfolio values that exactly match the benchmark
        portfolio_values = []
        
        for date, price in zip(benchmark_dates, benchmark_prices):
            portfolio_values.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': price
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
            pd.Series([v['portfolio_value'] for v in portfolio_values], index=benchmark_dates),
            benchmark_data['adjusted_close'],
            benchmark_symbol
        )
        
        # Try the new portfolio comparison function
        portfolio_returns = pd.Series(
            data=[0.001] * len(benchmark_data.index),  # 0.1% daily returns
            index=benchmark_data.index
        )
        
        results2 = compare_portfolio_to_benchmark(
            portfolio_returns=portfolio_returns,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            client=client,
            save_plot=True,
            plot_filename="test_benchmark_comparison.png"
        )
        
        print("\nNew Benchmark Comparison Results:")
        print(json.dumps(
            {k: v if not isinstance(v, float) else round(v, 4) for k, v in results2.items()},
            indent=2
        ))
    
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main()
