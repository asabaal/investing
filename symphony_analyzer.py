"""
Symphony Analyzer Module

This module provides advanced analytics for Composer symphonies, including
backtesting under different market conditions, forecasting using Prophet,
and watchlist functionality for monitoring symphony health.

It builds on the existing composer_symphony.py module and integrates with
the Prophet forecasting capabilities.
"""

import json
import logging
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from composer_symphony import Symphony, SymphonyBacktester, SymbolList
from prophet_forecasting import StockForecast, ProphetEnsemble
from alpha_vantage_api import AlphaVantageClient
from compare_benchmarks import compare_portfolio_to_benchmark

# Configure logging
logger = logging.getLogger(__name__)

class MarketScenario:
    """
    A market scenario for testing symphony performance under specific conditions.
    
    This class defines a market scenario with specific characteristics like trend,
    volatility, and correlation structure for backtesting symphonies under
    different market conditions.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        trend: str = 'neutral',
        volatility: str = 'normal',
        correlation: str = 'normal',
        duration_days: int = 60
    ):
        """
        Initialize a market scenario.
        
        Args:
            name: Scenario name
            description: Detailed description of the scenario
            trend: Market trend ('bullish', 'bearish', 'neutral', 'choppy')
            volatility: Volatility level ('low', 'normal', 'high', 'extreme')
            correlation: Correlation structure ('low', 'normal', 'high', 'inverse')
            duration_days: Duration of the scenario in days
        """
        self.name = name
        self.description = description
        self.trend = trend
        self.volatility = volatility
        self.correlation = correlation
        self.duration_days = duration_days
    
    def __repr__(self) -> str:
        return (f"MarketScenario('{self.name}', trend='{self.trend}', "
                f"volatility='{self.volatility}', correlation='{self.correlation}')")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'trend': self.trend,
            'volatility': self.volatility,
            'correlation': self.correlation,
            'duration_days': self.duration_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketScenario':
        """Create a MarketScenario from a dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            trend=data['trend'],
            volatility=data['volatility'],
            correlation=data['correlation'],
            duration_days=data['duration_days']
        )
    
    @classmethod
    def bull_market(cls) -> 'MarketScenario':
        """Create a standard bull market scenario."""
        return cls(
            name="Bull Market",
            description="Steady upward trend with normal volatility",
            trend='bullish',
            volatility='normal',
            correlation='normal',
            duration_days=90
        )
    
    @classmethod
    def bear_market(cls) -> 'MarketScenario':
        """Create a standard bear market scenario."""
        return cls(
            name="Bear Market",
            description="Steady downward trend with elevated volatility",
            trend='bearish',
            volatility='high',
            correlation='high',
            duration_days=90
        )
    
    @classmethod
    def market_crash(cls) -> 'MarketScenario':
        """Create a market crash scenario."""
        return cls(
            name="Market Crash",
            description="Sharp downward trend with extreme volatility and high correlations",
            trend='bearish',
            volatility='extreme',
            correlation='high',
            duration_days=30
        )
    
    @classmethod
    def sideways_market(cls) -> 'MarketScenario':
        """Create a sideways market scenario."""
        return cls(
            name="Sideways Market",
            description="Choppy market with no clear direction",
            trend='neutral',
            volatility='normal',
            correlation='low',
            duration_days=60
        )

class SymphonyAnalyzer:
    """
    Advanced analyzer for Composer symphonies.
    
    This class provides tools for detailed analysis, testing, and optimization
    of Composer symphonies. It integrates the backtesting engine with
    forecasting capabilities and provides tools for symphony variation.
    """
    
    def __init__(
        self,
        symphony_file: str,
        client: Optional[AlphaVantageClient] = None,
        cache_dir: str = './cache',
        default_scenarios: bool = True
    ):
        """
        Initialize a symphony analyzer.
        
        Args:
            symphony_file: Path to symphony JSON file
            client: Alpha Vantage client for market data
            cache_dir: Directory to store cached data
            default_scenarios: Whether to initialize with default market scenarios
        """
        # Initialize client if not provided
        if client is None:
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise ValueError("No Alpha Vantage API key provided or found in environment")
            self.client = AlphaVantageClient(api_key=api_key)
        else:
            self.client = client
        
        # Load symphony from file
        with open(symphony_file, 'r') as f:
            self.symphony_data = json.load(f)
        
        self.symphony_name = self.symphony_data.get('name', 'Unnamed Symphony')
        self.symphony_file = symphony_file
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize backtest engine
        self.backtester = SymphonyBacktester(self.client)
        
        # Initialize forecast engines
        self.stock_forecaster = StockForecast(self.client, cache_dir)
        self.ensemble_forecaster = ProphetEnsemble(self.client, cache_dir)
        
        # Market scenarios for testing
        self.scenarios = {}
        if default_scenarios:
            self._initialize_default_scenarios()
        
        # Store backtest results
        self.backtest_results = None
        
        # Create Symphony object for backtesting
        self.symphony = self._create_symphony_object()
    
    def _create_symphony_object(self):
        """Create a Symphony object from the loaded JSON data"""
        try:
            # Extract universe symbols - handle both formats
            if 'universe' in self.symphony_data:
                universe_data = self.symphony_data['universe']
                if isinstance(universe_data, list):
                    # Direct list of symbols
                    symbols = universe_data
                elif isinstance(universe_data, dict) and 'symbols' in universe_data:
                    # Dictionary with 'symbols' key
                    symbols = universe_data['symbols']
                else:
                    symbols = []
            else:
                symbols = []
            
            logger.info(f"Creating Symphony with {len(symbols)} symbols")
            
            # Create SymbolList
            universe = SymbolList(symbols)
            
            # Create Symphony
            symphony = Symphony(
                name=self.symphony_data.get('name', 'Unnamed Symphony'),
                description=self.symphony_data.get('description', ''),
                universe=universe
            )
            
            # Add operators if present in the data
            operators = self.symphony_data.get('operators', [])
            for operator_data in operators:
                # This is simplified - in a real implementation, you'd need to 
                # instantiate the correct operator type based on the data
                logger.info(f"Adding operator: {operator_data.get('name', 'Unnamed')}")
            
            # Set allocator if present in the data
            allocator_data = self.symphony_data.get('allocator')
            if allocator_data:
                # This is simplified - in a real implementation, you'd need to
                # instantiate the correct allocator type based on the data
                logger.info(f"Setting allocator: {allocator_data.get('name', 'Default')}")
            
            return symphony
            
        except Exception as e:
            logger.error(f"Error creating Symphony object: {str(e)}")
            # Return a minimal Symphony object with default empty list
            return Symphony(
                name=self.symphony_name,
                description="Extracted from JSON",
                universe=SymbolList([])
            )
    
    def _initialize_default_scenarios(self):
        """Initialize default market scenarios for testing."""
        default_scenarios = [
            MarketScenario.bull_market(),
            MarketScenario.bear_market(),
            MarketScenario.market_crash(),
            MarketScenario.sideways_market(),
            MarketScenario(
                name="Sector Rotation",
                description="Market characterized by shifting performance across sectors",
                trend='neutral',
                volatility='normal',
                correlation='low',
                duration_days=90
            ),
            MarketScenario(
                name="Rising Rate Environment",
                description="Market adjusting to rising interest rates",
                trend='bearish',
                volatility='high',
                correlation='normal',
                duration_days=120
            ),
            MarketScenario(
                name="Recovery",
                description="Market recovering from a significant downturn",
                trend='bullish',
                volatility='high',
                correlation='decreasing',
                duration_days=90
            )
        ]
        
        for scenario in default_scenarios:
            self.scenarios[scenario.name] = scenario
    
    def add_scenario(self, scenario: MarketScenario):
        """
        Add a custom market scenario.
        
        Args:
            scenario: Market scenario to add
        """
        self.scenarios[scenario.name] = scenario
    
    def get_symbols(self) -> List[str]:
        """
        Get list of symbols used in the symphony.
        
        Returns:
            List of symbol strings
        """
        # Handle both direct list and dictionary with 'symbols' key
        if 'universe' in self.symphony_data:
            universe_data = self.symphony_data['universe']
            if isinstance(universe_data, list):
                return universe_data
            elif isinstance(universe_data, dict) and 'symbols' in universe_data:
                return universe_data['symbols']
        return []
    
    def backtest(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_frequency: str = 'monthly'
    ) -> Dict:
        """
        Run a backtest of the symphony.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            rebalance_frequency: Rebalance frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary containing backtest results
        """
        # Use default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Default to 1 year lookback
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Run backtest using the Symphony object instead of the raw dictionary
        try:
            logger.info(f"Running backtest from {start_date} to {end_date} with {rebalance_frequency} rebalancing")
            results = self.backtester.backtest(
                self.symphony,  # Use Symphony object instead of dictionary
                start_date,
                end_date,
                rebalance_frequency=rebalance_frequency
            )
            
            # Store results for later use
            self.backtest_results = results
            
            return results
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            # Return empty results with error
            return {
                'success': False,
                'error': str(e),
                'portfolio_history': [],
                'backtest_summary': {
                    'total_return': 0,
                    'annualized_return': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
            }
    
    def compare_to_benchmark(
        self, 
        benchmark_symbol: str = "SPY", 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_plot: bool = True,
        plot_filename: Optional[str] = None
    ) -> Dict:
        """
        Compare symphony performance to a benchmark.
        
        Args:
            benchmark_symbol: Symbol for benchmark (default: SPY)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_plot: Whether to save the plot to file
            plot_filename: Filename to save the plot (default: symphony_name_vs_benchmark.png)
            
        Returns:
            Dictionary with comparison metrics
        """
        # Run backtest if we don't have results yet
        if not self.backtest_results:
            self.backtest(start_date, end_date)
        
        # Extract portfolio returns from backtest results
        if 'portfolio_history' not in self.backtest_results or not self.backtest_results['portfolio_history']:
            logger.warning("No portfolio history found in backtest results")
            return {
                "error": "No portfolio history found in backtest results",
                "benchmark_symbol": benchmark_symbol,
                "benchmark_return": 0,
                "portfolio_return": 0,
                "excess_return": 0
            }
        
        try:
            # Create portfolio returns series
            portfolio_history = self.backtest_results['portfolio_history']
            portfolio_dates = [pd.to_datetime(entry['date']) for entry in portfolio_history]
            portfolio_values = [entry['portfolio_value'] for entry in portfolio_history]
            
            portfolio_series = pd.Series(
                data=portfolio_values,
                index=portfolio_dates
            )
            
            # Calculate daily returns
            portfolio_returns = portfolio_series.pct_change().dropna()
            
            # Set filename if not provided
            if save_plot and plot_filename is None:
                # Clean up symphony name for filename
                clean_name = self.symphony_name.replace(' ', '_').replace('/', '_').lower()
                plot_filename = f"{clean_name}_vs_{benchmark_symbol}.png"
            
            # Run comparison
            comparison_results = compare_portfolio_to_benchmark(
                portfolio_returns=portfolio_returns,
                benchmark_symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                client=self.client,
                save_plot=save_plot,
                plot_filename=plot_filename
            )
            
            return comparison_results
        except Exception as e:
            logger.error(f"Benchmark comparison error: {str(e)}")
            return {
                "error": str(e),
                "benchmark_symbol": benchmark_symbol
            }
    
    def forecast_symphony(
        self,
        days: int = 30,
        use_ensemble: bool = True,
        num_models: int = 5,
        save_plots: bool = True
    ) -> Dict:
        """
        Generate forecasts for all symbols in the symphony.
        
        Args:
            days: Number of days to forecast
            use_ensemble: Whether to use ensemble forecasting
            num_models: Number of models for ensemble (if use_ensemble=True)
            save_plots: Whether to save forecast plots
            
        Returns:
            Dictionary with forecast results
        """
        symbols = self.get_symbols()
        if not symbols:
            return {"error": "No symbols found in symphony"}
        
        results = {}
        
        # Create plots directory if saving plots
        if save_plots:
            plots_dir = os.path.join(self.cache_dir, 'forecast_plots')
            os.makedirs(plots_dir, exist_ok=True)
        
        for symbol in symbols:
            try:
                logger.info(f"Forecasting {symbol}")
                
                if use_ensemble:
                    # Use ensemble forecasting
                    result = self.ensemble_forecaster.forecast_ensemble(
                        symbol,
                        days=days,
                        num_models=num_models
                    )
                    
                    forecast = result['forecast']
                    models = result['models']
                    
                    # Get summary
                    summary = self.ensemble_forecaster.get_ensemble_forecast_summary(result)
                    
                    if save_plots:
                        # Create visualization
                        plt.figure(figsize=(12, 6))
                        
                        # Plot historical data
                        historical_dates = pd.to_datetime(result['df'].index)
                        historical_prices = result['df']['adjusted_close']
                        plt.scatter(historical_dates, historical_prices, s=10, c='black', alpha=0.5, label='Historical')
                        
                        # Plot ensemble forecast
                        plt.plot(forecast['ds'], forecast['yhat'], 'b-', linewidth=2, label='Ensemble Forecast')
                        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                                       alpha=0.2, color='blue', label='95% Confidence Interval')
                        
                        # Add labels and legend
                        plt.title(f"{symbol} Forecast - Next {days} Days")
                        plt.xlabel("Date")
                        plt.ylabel("Price")
                        plt.legend()
                        plt.grid(True)
                        
                        # Save plot
                        filename = os.path.join(self.cache_dir, 'forecast_plots', f"{symbol}_forecast.png")
                        plt.savefig(filename)
                        plt.close()
                        
                        summary['plot_filename'] = filename
                    
                    results[symbol] = summary
                    
                else:
                    # Use single model forecasting
                    df, forecast, model = self.stock_forecaster.forecast(
                        symbol,
                        days=days
                    )
                    
                    # Get summary
                    summary = self.stock_forecaster.get_forecast_summary(forecast)
                    
                    if save_plots:
                        # Create visualization
                        plt.figure(figsize=(12, 6))
                        plt.plot(df.index, df['adjusted_close'], 'k.', label='Historical')
                        plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
                        
                        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                                           color='blue', alpha=0.2, label='Prediction Interval')
                        
                        plt.title(f"{symbol} Forecast - Next {days} Days")
                        plt.xlabel("Date")
                        plt.ylabel("Price")
                        plt.legend()
                        plt.grid(True)
                        
                        # Save plot
                        filename = os.path.join(self.cache_dir, 'forecast_plots', f"{symbol}_forecast.png")
                        plt.savefig(filename)
                        plt.close()
                        
                        summary['plot_filename'] = filename
                    
                    results[symbol] = summary
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        # Calculate aggregated forecasts
        if results:
            # Gather forecast metrics
            forecast_7d = []
            forecast_30d = []
            
            for symbol, result in results.items():
                if '7_day' in result and 'percent_change' in result['7_day']:
                    forecast_7d.append(result['7_day']['percent_change'])
                
                if '30_day' in result and 'percent_change' in result['30_day']:
                    forecast_30d.append(result['30_day']['percent_change'])
            
            # Calculate averages
            if forecast_7d:
                results['average_7d_forecast'] = sum(forecast_7d) / len(forecast_7d)
            
            if forecast_30d:
                results['average_30d_forecast'] = sum(forecast_30d) / len(forecast_30d)
            
            # Overall forecast sentiment
            if 'average_30d_forecast' in results:
                if results['average_30d_forecast'] > 5:
                    results['forecast_sentiment'] = 'bullish'
                elif results['average_30d_forecast'] < -5:
                    results['forecast_sentiment'] = 'bearish'
                else:
                    results['forecast_sentiment'] = 'neutral'
        
        return results
    
    def analyze_symphony_risk(self) -> Dict:
        """
        Analyze risk characteristics of the symphony.
        
        Returns:
            Dictionary with risk analysis metrics
        """
        # Run backtest if we don't have results yet
        if not self.backtest_results:
            self.backtest()
        
        try:
            # Check if we have portfolio history
            if 'portfolio_history' not in self.backtest_results or not self.backtest_results['portfolio_history']:
                return {
                    'error': 'No portfolio history available',
                    'volatility': 0,
                    'max_drawdown': 0,
                    'downside_deviation': 0,
                    'sortino_ratio': 0
                }
            
            # Extract performance data from backtest
            portfolio_history = self.backtest_results['portfolio_history']
            values = [entry['portfolio_value'] for entry in portfolio_history]
            
            # Calculate returns
            returns = []
            for i in range(1, len(values)):
                returns.append((values[i] - values[i-1]) / values[i-1])
            
            returns_array = np.array(returns)
            
            # Calculate volatility (annualized)
            volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 0 else 0
            
            # Calculate drawdowns
            drawdowns = []
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns) if drawdowns else 0
            
            # Calculate downside deviation (semi-deviation of negative returns)
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Calculate Sortino ratio
            avg_return = np.mean(returns_array) * 252 if len(returns_array) > 0 else 0
            risk_free_rate = 0.03  # Assuming 3% annual risk-free rate
            sortino_ratio = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Analyze allocations
            latest_allocation = portfolio_history[-1].get('allocations', {}) if portfolio_history else {}
            
            # Calculate concentration metrics
            concentration = 0
            if latest_allocation:
                weights = list(latest_allocation.values())
                concentration = sum(w**2 for w in weights)  # Herfindahl-Hirschman Index
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'downside_deviation': downside_deviation,
                'sortino_ratio': sortino_ratio,
                'concentration': concentration,
                'allocation_count': len(latest_allocation),
                'avg_allocation': np.mean(list(latest_allocation.values())) if latest_allocation else 0,
                'max_allocation': max(latest_allocation.values()) if latest_allocation else 0
            }
            
        except Exception as e:
            logger.error(f"Risk analysis error: {str(e)}")
            return {'error': str(e)}
    
    def analyze_symphony(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        benchmark_symbol: str = 'SPY',
        forecast_days: int = 30,
        scenarios: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform comprehensive analysis of the symphony.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            benchmark_symbol: Symbol to use as benchmark
            forecast_days: Number of days to forecast
            scenarios: List of scenario names to test (or None for standard backtest)
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'symphony_info': {
                'name': self.symphony_name,
                'file': self.symphony_file,
                'symbols': self.get_symbols()
            },
            'backtest_results': {},
            'benchmark_comparison': {},
            'forecasts': {},
            'risk_analysis': {},
            'scenario_results': {}
        }
        
        # Run backtest
        logger.info(f"Running backtest for symphony: {self.symphony_name}")
        backtest_results = self.backtest(start_date, end_date)
        results['backtest_results'] = backtest_results
        
        # Compare to benchmark
        logger.info(f"Comparing symphony to benchmark: {benchmark_symbol}")
        benchmark_comparison = self.compare_to_benchmark(benchmark_symbol, start_date, end_date)
        results['benchmark_comparison'] = benchmark_comparison
        
        # Generate forecasts
        logger.info(f"Generating {forecast_days}-day forecast")
        forecasts = self.forecast_symphony(days=forecast_days)
        results['forecasts'] = forecasts
        
        # Analyze risk
        logger.info("Analyzing symphony risk")
        risk_analysis = self.analyze_symphony_risk()
        results['risk_analysis'] = risk_analysis
        
        # Run scenario analysis if requested
        if scenarios:
            for scenario_name in scenarios:
                if scenario_name in self.scenarios:
                    logger.info(f"Running scenario: {scenario_name}")
                    # TODO: Implement scenario testing
                    results['scenario_results'][scenario_name] = {
                        'note': 'Scenario testing not yet implemented'
                    }
        
        return results

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if a symphony file is provided as argument
    import sys
    if len(sys.argv) > 1:
        symphony_file = sys.argv[1]
    else:
        symphony_file = 'sample_symphony.json'
    
    # Create symphony analyzer
    analyzer = SymphonyAnalyzer(symphony_file)
    
    # Analyze symphony
    results = analyzer.analyze_symphony()
    
    # Display results summary
    print(f"\nSymphony: {results['symphony_info']['name']}")
    print(f"Symbols: {', '.join(results['symphony_info']['symbols'])}")
    
    if 'backtest_summary' in results['backtest_results']:
        summary = results['backtest_results']['backtest_summary']
        print(f"\nBacktest Results:")
        print(f"  Total Return: {summary.get('total_return', 0)*100:.2f}%")
        print(f"  Annualized Return: {summary.get('annualized_return', 0)*100:.2f}%")
        print(f"  Max Drawdown: {summary.get('max_drawdown', 0)*100:.2f}%")
    
    if 'benchmark_symbol' in results['benchmark_comparison']:
        benchmark = results['benchmark_comparison']
        print(f"\nBenchmark Comparison ({benchmark['benchmark_symbol']}):")
        print(f"  Benchmark Return: {benchmark.get('benchmark_return', 0)*100:.2f}%")
        print(f"  Portfolio Return: {benchmark.get('portfolio_return', 0)*100:.2f}%")
        print(f"  Excess Return: {benchmark.get('excess_return', 0)*100:.2f}%")
        print(f"  Beta: {benchmark.get('beta', 0):.2f}")
    
    if 'average_30d_forecast' in results['forecasts']:
        print(f"\nForecast:")
        print(f"  30-Day Average: {results['forecasts']['average_30d_forecast']:.2f}%")
        print(f"  Sentiment: {results['forecasts'].get('forecast_sentiment', 'unknown')}")
    
    if 'volatility' in results['risk_analysis']:
        risk = results['risk_analysis']
        print(f"\nRisk Analysis:")
        print(f"  Volatility: {risk.get('volatility', 0)*100:.2f}%")
        print(f"  Max Drawdown: {risk.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}")
