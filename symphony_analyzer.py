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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from composer_symphony import Symphony, SymphonyBacktester, SymbolList
from prophet_forecasting import StockForecast, ProphetEnsemble
from alpha_vantage_api import AlphaVantageClient
from trading_system import TradingAnalytics

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
        client: AlphaVantageClient,
        cache_dir: str = './cache',
        default_scenarios: bool = True
    ):
        """
        Initialize a symphony analyzer.
        
        Args:
            client: Alpha Vantage client for market data
            cache_dir: Directory to store cached data
            default_scenarios: Whether to initialize with default market scenarios
        """
        self.client = client
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize backtest engine
        self.backtester = SymphonyBacktester(client)
        
        # Initialize forecast engines
        self.stock_forecaster = StockForecast(client, cache_dir)
        self.ensemble_forecaster = ProphetEnsemble(client, cache_dir)
        
        # Market scenarios for testing
        self.scenarios = {}
        if default_scenarios:
            self._initialize_default_scenarios()
    
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
    
    def analyze_symphony(
        self, 
        symphony: Symphony,
        start_date: str,
        end_date: str,
        scenarios: Optional[List[str]] = None,
        forecast_days: int = 30,
        benchmark_symbol: str = 'SPY'
    ) -> Dict:
        """
        Perform comprehensive analysis of a symphony.
        
        Args:
            symphony: Symphony to analyze
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            scenarios: List of scenario names to test (or None for standard backtest)
            forecast_days: Number of days to forecast
            benchmark_symbol: Symbol to use as benchmark
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'symphony_info': {
                'name': symphony.name,
                'description': symphony.description,
                'universe_size': len(symphony.universe),
                'operator_count': len(symphony.operators),
                'allocator_type': symphony.allocator.__class__.__name__
            },
            'backtest_results': {},
            'forecasts': {},
            'scenario_results': {},
            'benchmark_comparison': {},
            'risk_analysis': {}
        }
        
        # Perform standard backtest
        standard_backtest = self.backtester.backtest(
            symphony, 
            start_date, 
            end_date, 
            rebalance_frequency='monthly'
        )
        results['backtest_results'] = standard_backtest
        
        # Generate forecasts for securities in the symphony
        forecasts = self._forecast_symphony_securities(symphony, forecast_days)
        results['forecasts'] = forecasts
        
        # Perform scenario testing if requested
        if scenarios:
            for scenario_name in scenarios:
                if scenario_name in self.scenarios:
                    scenario_results = self._run_scenario_test(
                        symphony, 
                        self.scenarios[scenario_name],
                        end_date
                    )
                    results['scenario_results'][scenario_name] = scenario_results
        
        # Compare to benchmark
        benchmark_comparison = self._compare_to_benchmark(
            standard_backtest, 
            benchmark_symbol,
            start_date,
            end_date
        )
        results['benchmark_comparison'] = benchmark_comparison
        
        # Perform risk analysis
        risk_analysis = self._analyze_symphony_risk(
            symphony, 
            standard_backtest
        )
        results['risk_analysis'] = risk_analysis
        
        return results
    
    def _forecast_symphony_securities(
        self, 
        symphony: Symphony, 
        forecast_days: int = 30
    ) -> Dict:
        """
        Generate forecasts for all securities in a symphony.
        
        Args:
            symphony: Symphony to forecast securities for
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary mapping symbols to forecast data
        """
        forecasts = {}
        
        # Generate forecasts for each symbol in the universe
        for symbol in symphony.universe.symbols:
            try:
                # Use ensemble forecasting for better accuracy
                result = self.ensemble_forecaster.forecast_ensemble(
                    symbol, 
                    days=forecast_days, 
                    num_models=5
                )
                
                summary = self.ensemble_forecaster.get_ensemble_forecast_summary(
                    result
                )
                
                forecasts[symbol] = summary
            except Exception as e:
                logger.warning(f"Failed to forecast {symbol}: {str(e)}")
                forecasts[symbol] = {'error': str(e)}
        
        # Add aggregate forecast based on symphony weights
        try:
            # Execute symphony to get current allocations
            allocations = symphony.execute(self.client)
            
            # Weight the forecasts based on allocations
            weighted_forecasts = {}
            for time_horizon in ['7_day', '30_day']:
                if time_horizon in forecasts.get(list(forecasts.keys())[0], {}):
                    weighted_change = 0
                    total_weight = 0
                    
                    for symbol, weight in allocations.items():
                        if symbol in forecasts and time_horizon in forecasts[symbol]:
                            symbol_forecast = forecasts[symbol][time_horizon]
                            weighted_change += symbol_forecast.get('percent_change', 0) * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_forecasts[time_horizon] = {
                            'weighted_percent_change': weighted_change / total_weight
                        }
            
            forecasts['weighted_symphony_forecast'] = weighted_forecasts
            
        except Exception as e:
            logger.warning(f"Failed to generate weighted forecast: {str(e)}")
        
        return forecasts
    
    def _run_scenario_test(
        self, 
        symphony: Symphony, 
        scenario: MarketScenario,
        base_date: str
    ) -> Dict:
        """
        Run a symphony through a specific market scenario.
        
        Args:
            symphony: Symphony to test
            scenario: Market scenario to test against
            base_date: Base date for scenario (YYYY-MM-DD)
            
        Returns:
            Dictionary with scenario test results
        """
        # Create date range for scenario
        base_date_dt = datetime.strptime(base_date, '%Y-%m-%d')
        start_date = base_date_dt - timedelta(days=scenario.duration_days)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Get historical data for symbols
        historical_data = {}
        for symbol in symphony.universe.symbols:
            try:
                df = self.client.get_daily(symbol, outputsize='full')
                historical_data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {str(e)}")
        
        # Modify the data based on scenario parameters
        scenario_data = self._generate_scenario_data(
            historical_data,
            scenario,
            start_date_str,
            base_date
        )
        
        # Create mock client with scenario data
        from composer_symphony import MockAlphaVantageClient
        mock_client = MockAlphaVantageClient(scenario_data)
        
        # Create backtester with mock client
        scenario_backtester = SymphonyBacktester(mock_client)
        
        # Run backtest
        results = scenario_backtester.backtest(
            symphony,
            start_date_str,
            base_date,
            rebalance_frequency='monthly'
        )
        
        results['scenario_info'] = scenario.to_dict()
        
        return results
    
    def _generate_scenario_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        scenario: MarketScenario,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate modified market data for a specific scenario.
        
        Args:
            historical_data: Dictionary of historical dataframes by symbol
            scenario: Market scenario to simulate
            start_date: Start date for scenario
            end_date: End date for scenario
            
        Returns:
            Dictionary of modified dataframes by symbol
        """
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        scenario_data = {}
        
        # Base parameters for scenario types
        trend_params = {
            'bullish': {'drift': 0.001},  # +0.1% per day drift (~25% annual)
            'bearish': {'drift': -0.001},  # -0.1% per day drift (~-25% annual)
            'neutral': {'drift': 0.0},     # No drift
            'choppy': {'drift': 0.0, 'reversal_prob': 0.4}  # Frequent reversals
        }
        
        volatility_params = {
            'low': {'vol_multiplier': 0.5},
            'normal': {'vol_multiplier': 1.0},
            'high': {'vol_multiplier': 2.0},
            'extreme': {'vol_multiplier': 3.0}
        }
        
        correlation_params = {
            'low': {'corr_reduction': 0.5},
            'normal': {'corr_reduction': 0.0},
            'high': {'corr_reduction': -0.5},  # Increase correlations
            'inverse': {'correlation_flip': True}
        }
        
        # Get parameters for this scenario
        t_params = trend_params.get(scenario.trend, trend_params['neutral'])
        v_params = volatility_params.get(scenario.volatility, volatility_params['normal'])
        c_params = correlation_params.get(scenario.correlation, correlation_params['normal'])
        
        # Calculate returns for all symbols in the historical data
        returns_data = {}
        for symbol, df in historical_data.items():
            if len(df) > 0:
                df_in_range = df.loc[(df.index >= start_date_pd) & (df.index <= end_date_pd)].copy()
                if len(df_in_range) > 0:
                    df_in_range['returns'] = df_in_range['adjusted_close'].pct_change().fillna(0)
                    returns_data[symbol] = df_in_range
        
        if not returns_data:
            logger.warning("No data available in the specified date range for any symbol")
            return historical_data
        
        # Calculate a "market return" as average of all symbols
        all_returns = pd.DataFrame({
            symbol: df['returns'] for symbol, df in returns_data.items()
        })
        market_return = all_returns.mean(axis=1)
        
        # Generate scenario-specific returns and prices
        for symbol, df_original in historical_data.items():
            df = df_original.copy()
            df_in_range = df.loc[(df.index >= start_date_pd) & (df.index <= end_date_pd)].copy()
            
            if len(df_in_range) > 0:
                # Start with original returns
                if symbol in returns_data:
                    original_returns = returns_data[symbol]['returns']
                    
                    # Decompose returns into market and specific components
                    market_beta = np.cov(original_returns, market_return)[0, 1] / np.var(market_return) if np.var(market_return) > 0 else 1.0
                    specific_returns = original_returns - market_beta * market_return
                    
                    # Apply scenario adjustments
                    # 1. Trend adjustment
                    modified_market = market_return + t_params.get('drift', 0.0)
                    
                    if t_params.get('reversal_prob', 0) > 0:
                        # Create more choppy market with potential reversals
                        reversal = np.random.random(len(modified_market)) < t_params['reversal_prob']
                        modified_market[reversal] = -modified_market[reversal]
                    
                    # 2. Volatility adjustment
                    vol_mul = v_params.get('vol_multiplier', 1.0)
                    modified_market = modified_market * vol_mul
                    specific_returns = specific_returns * vol_mul
                    
                    # 3. Correlation adjustment
                    corr_reduction = c_params.get('corr_reduction', 0.0)
                    reduced_beta = market_beta * (1.0 - corr_reduction)
                    
                    if c_params.get('correlation_flip', False):
                        reduced_beta = -reduced_beta  # Inverse correlation
                    
                    # Combine components to get scenario returns
                    scenario_returns = (reduced_beta * modified_market) + specific_returns
                    
                    # Generate new prices based on scenario returns
                    start_price = df_in_range['adjusted_close'].iloc[0]
                    scenario_prices = start_price * (1 + scenario_returns).cumprod()
                    
                    # Update price columns
                    df_in_range['adjusted_close'] = scenario_prices
                    df_in_range['close'] = scenario_prices
                    
                    # Update other price columns (approximate scaling)
                    for col in ['open', 'high', 'low']:
                        if col in df_in_range.columns:
                            ratio = df_in_range[col] / df_original.loc[df_in_range.index, 'adjusted_close']
                            df_in_range[col] = scenario_prices * ratio
                    
                    # Update the original slice with the scenario data
                    df.loc[(df.index >= start_date_pd) & (df.index <= end_date_pd)] = df_in_range
                    
                    scenario_data[symbol] = df
                else:
                    scenario_data[symbol] = df
            else:
                scenario_data[symbol] = df
        
        return scenario_data
    
    def _compare_to_benchmark(
        self, 
        backtest_results: Dict,
        benchmark_symbol: str,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Compare symphony performance to a benchmark.
        
        Args:
            backtest_results: Results from backtest
            benchmark_symbol: Symbol to use as benchmark
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        if not backtest_results.get('success', False):
            return {'error': 'Backtest was not successful'}
        
        try:
            # Get benchmark data for the same period
            benchmark_data = self.client.get_daily(benchmark_symbol)
            start_date_pd = pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date)
            
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
            
            # Get symphony performance
            summary = backtest_results['backtest_summary']
            symphony_return = summary['total_return']
            
            # Get symphony equity curve
            portfolio_history = backtest_results['portfolio_history']
            symphony_dates = [entry['date'] for entry in portfolio_history]
            symphony_values = [entry['portfolio_value'] for entry in portfolio_history]
            
            # Calculate excess return
            excess_return = symphony_return - benchmark_return
            
            # Calculate correlation if possible
            correlation = None
            beta = None
            
            if len(symphony_values) > 1:
                # Calculate symphony returns
                symphony_returns = np.diff(symphony_values) / symphony_values[:-1]
                
                # Create dataframe of symphony and benchmark returns
                symphony_index = pd.DatetimeIndex([pd.Timestamp(d) for d in symphony_dates[1:]])
                symphony_returns_series = pd.Series(symphony_returns, index=symphony_index)
                
                # Align benchmark returns with symphony dates
                aligned_returns = pd.DataFrame({
                    'symphony': symphony_returns_series,
                    'benchmark': benchmark_returns
                })
                aligned_returns = aligned_returns.dropna()
                
                # Make sure both series have the same length after alignment
                if len(aligned_returns) > 1:
                    correlation = aligned_returns['symphony'].corr(aligned_returns['benchmark'])
                    # Calculate beta (symphony return to benchmark return)
                    cov_matrix = aligned_returns.cov()
                    beta = cov_matrix.loc['symphony', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            
            # Calculate risk-adjusted metrics
            sharpe_ratio = summary.get('sharpe_ratio', 0)
            
            # Calculate benchmark Sharpe
            benchmark_sharpe = None
            if len(benchmark_returns) > 1:
                risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
                benchmark_excess_return = benchmark_returns - risk_free_rate
                benchmark_sharpe = (benchmark_excess_return.mean() / benchmark_excess_return.std()) * np.sqrt(252)
            
            # Information ratio
            information_ratio = None
            if correlation is not None and beta is not None and len(symphony_returns) > 0 and len(benchmark_returns) > 0:
                # Need to re-align the returns again for accurate tracking error calculation
                aligned_returns = pd.DataFrame({
                    'symphony': symphony_returns_series,
                    'benchmark': benchmark_returns
                }).dropna()
                
                if len(aligned_returns) > 1:
                    symphony_aligned = aligned_returns['symphony'].values
                    benchmark_aligned = aligned_returns['benchmark'].values
                    
                    tracking_error = np.std(symphony_aligned - (beta * benchmark_aligned)) * np.sqrt(252)
                    if tracking_error > 0:
                        information_ratio = excess_return / tracking_error
            
            return {
                'benchmark_symbol': benchmark_symbol,
                'benchmark_return': benchmark_return,
                'symphony_return': symphony_return,
                'excess_return': excess_return,
                'correlation': correlation,
                'beta': beta,
                'sharpe_ratio': sharpe_ratio,
                'benchmark_sharpe': benchmark_sharpe,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            logger.error(f"Benchmark comparison error: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_symphony_risk(self, symphony: Symphony, backtest_results: Dict) -> Dict:
        """
        Analyze risk characteristics of a symphony.
        
        Args:
            symphony: Symphony to analyze
            backtest_results: Results from backtest
            
        Returns:
            Dictionary with risk analysis
        """
        if not backtest_results.get('success', False):
            return {'error': 'Backtest was not successful'}
        
        try:
            # Extract performance data from backtest
            portfolio_history = backtest_results['portfolio_history']
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
            latest_allocation = portfolio_history[-1]['allocations'] if portfolio_history else {}
            
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
    
    def generate_symphony_variations(
        self, 
        base_symphony: Symphony,
        parameter_space: Dict[str, List],
        n_variations: int = 5
    ) -> List[Symphony]:
        """
        Generate variations of a symphony by perturbing parameters.
        
        Args:
            base_symphony: Base symphony to vary
            parameter_space: Dictionary mapping parameter names to possible values
            n_variations: Number of variations to generate
            
        Returns:
            List of symphony variations
        """
        variations = []
        
        # Convert base symphony to dict for easier modification
        base_dict = base_symphony.to_dict()
        
        # Generate variations
        for i in range(n_variations):
            # Create a copy of the base symphony dict
            variation_dict = json.loads(json.dumps(base_dict))
            variation_dict['name'] = f"{base_symphony.name} (Variation {i+1})"
            
            # Modify operators based on parameter space
            if 'operators' in parameter_space and variation_dict.get('operators'):
                for j, op in enumerate(variation_dict['operators']):
                    op_type = op['type']
                    
                    # Modify parameters for this operator type
                    if op_type in parameter_space.get('operators', {}):
                        op_params = parameter_space['operators'][op_type]
                        
                        # Apply parameter variations
                        for param, values in op_params.items():
                            if param in op.get('condition', {}):
                                # Randomly select a parameter value
                                random_value = np.random.choice(values)
                                op['condition'][param] = random_value
            
            # Modify allocator parameters
            if 'allocator' in parameter_space and variation_dict.get('allocator'):
                allocator_type = variation_dict['allocator']['type']
                
                if allocator_type in parameter_space.get('allocator', {}):
                    allocator_params = parameter_space['allocator'][allocator_type]
                    
                    # Apply parameter variations
                    for param, values in allocator_params.items():
                        if param in variation_dict['allocator']:
                            # Randomly select a parameter value
                            random_value = np.random.choice(values)
                            variation_dict['allocator'][param] = random_value
            
            # Create symphony from modified dict
            variation = Symphony.from_dict(variation_dict)
            variations.append(variation)
        
        return variations
    
    def optimize_symphony(
        self, 
        base_symphony: Symphony,
        parameter_space: Dict[str, List],
        start_date: str,
        end_date: str,
        n_iterations: int = 10,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[Symphony, Dict]:
        """
        Optimize a symphony by testing variations and selecting the best.
        
        Args:
            base_symphony: Base symphony to optimize
            parameter_space: Dictionary mapping parameter names to possible values
            start_date: Start date for testing
            end_date: End date for testing
            n_iterations: Number of iterations for optimization
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Tuple of (best symphony, performance details)
        """
        best_symphony = base_symphony
        best_performance = -float('inf')
        best_results = None
        
        for iteration in range(n_iterations):
            # Generate variations for this iteration
            variations = self.generate_symphony_variations(
                base_symphony,
                parameter_space,
                n_variations=3
            )
            
            # Test each variation
            for variation in variations:
                try:
                    # Backtest the variation
                    results = self.backtester.backtest(
                        variation,
                        start_date,
                        end_date,
                        rebalance_frequency='monthly'
                    )
                    
                    if results.get('success', False):
                        # Extract the performance metric
                        performance = 0
                        
                        if metric == 'sharpe_ratio':
                            # Calculate Sharpe ratio from returns
                            returns = []
                            portfolio_history = results['portfolio_history']
                            values = [entry['portfolio_value'] for entry in portfolio_history]
                            
                            for i in range(1, len(values)):
                                returns.append((values[i] - values[i-1]) / values[i-1])
                            
                            returns_array = np.array(returns)
                            avg_return = np.mean(returns_array) * 252 if len(returns_array) > 0 else 0
                            volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 0 else 1
                            
                            risk_free_rate = 0.03  # Assuming 3% annual risk-free rate
                            performance = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
                            
                        elif metric == 'total_return':
                            performance = results['backtest_summary']['total_return']
                            
                        elif metric == 'max_drawdown':
                            # For drawdown, smaller is better, so negate
                            performance = -results['backtest_summary']['max_drawdown']
                        
                        # Check if this is the best so far
                        if performance > best_performance:
                            best_performance = performance
                            best_symphony = variation
                            best_results = results
                
                except Exception as e:
                    logger.warning(f"Error testing variation: {str(e)}")
        
        # Return the best symphony
        return best_symphony, {
            'metric': metric,
            'value': best_performance,
            'results': best_results
        }

    def create_symphony_watchlist(self, symphonies: List[Symphony]) -> Dict:
        """
        Create a watchlist of all securities across multiple symphonies.
        
        Args:
            symphonies: List of symphonies to include in watchlist
            
        Returns:
            Dictionary with watchlist data
        """
        # Collect all symbols from all symphonies
        all_symbols = set()
        for symphony in symphonies:
            all_symbols.update(symphony.universe.symbols)
        
        # Create lookup of which symbols are in which symphonies
        symbol_to_symphonies = {symbol: [] for symbol in all_symbols}
        for symphony in symphonies:
            for symbol in symphony.universe.symbols:
                symbol_to_symphonies[symbol].append(symphony.name)
        
        # Get latest data for each symbol
        symbol_data = {}
        for symbol in all_symbols:
            try:
                # Get latest quote
                quote = self.client.get_quote(symbol)
                
                # Get historical data for trend analysis
                history = self.client.get_daily(symbol)
                
                # Calculate 5-day and 20-day percentage change
                if len(history) >= 5:
                    pct_change_5d = (history['adjusted_close'].iloc[-1] - history['adjusted_close'].iloc[-5]) / history['adjusted_close'].iloc[-5]
                else:
                    pct_change_5d = None
                    
                if len(history) >= 20:
                    pct_change_20d = (history['adjusted_close'].iloc[-1] - history['adjusted_close'].iloc[-20]) / history['adjusted_close'].iloc[-20]
                else:
                    pct_change_20d = None
                
                # Get latest technical indicators
                try:
                    rsi_data = self.client.get_rsi(symbol)
                    rsi = rsi_data.iloc[-1]['RSI'] if not rsi_data.empty else None
                except Exception as e:
                    logger.warning(f"Failed to get RSI for {symbol}: {str(e)}")
                    rsi = None
                
                try:
                    macd_data = self.client.get_macd(symbol)
                    macd = macd_data.iloc[-1]['MACD'] if not macd_data.empty else None
                    macd_signal = macd_data.iloc[-1]['MACD_Signal'] if not macd_data.empty and 'MACD_Signal' in macd_data.columns else None
                except Exception as e:
                    logger.warning(f"Failed to get MACD for {symbol}: {str(e)}")
                    macd = None
                    macd_signal = None
                
                # Generate forecasts
                try:
                    result = self.ensemble_forecaster.forecast_ensemble(symbol, days=30, num_models=3)
                    forecast = self.ensemble_forecaster.get_ensemble_forecast_summary(result)
                except Exception as e:
                    logger.warning(f"Failed to forecast {symbol}: {str(e)}")
                    forecast = None
                
                # Add to symbol data
                symbol_data[symbol] = {
                    'quote': quote,
                    'trend': {
                        'pct_change_5d': pct_change_5d,
                        'pct_change_20d': pct_change_20d
                    },
                    'technicals': {
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal
                    },
                    'forecast': forecast,
                    'symphonies': symbol_to_symphonies[symbol]
                }
                
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                symbol_data[symbol] = {'error': str(e)}
        
        # Create symphony health indicators
        symphony_health = {}
        for symphony in symphonies:
            health_score = self._calculate_symphony_health(symphony, symbol_data)
            symphony_health[symphony.name] = health_score
        
        return {
            'last_updated': datetime.now().isoformat(),
            'symbols': symbol_data,
            'symphony_health': symphony_health
        }
    
    def _calculate_symphony_health(
        self, 
        symphony: Symphony,
        symbol_data: Dict
    ) -> Dict:
        """
        Calculate health indicators for a symphony.
        
        Args:
            symphony: Symphony to analyze
            symbol_data: Dictionary of symbol data
            
        Returns:
            Dictionary with health metrics
        """
        # Count symbols with positive/negative trends
        trend_positive = 0
        trend_negative = 0
        
        # Count symbols with bullish/bearish technicals
        tech_bullish = 0
        tech_bearish = 0
        
        # Count symbols with positive/negative forecasts
        forecast_positive = 0
        forecast_negative = 0
        
        # Track symbols with potential issues
        symbols_at_risk = []
        
        for symbol in symphony.universe.symbols:
            if symbol in symbol_data:
                data = symbol_data[symbol]
                
                # Skip symbols with errors
                if 'error' in data:
                    continue
                
                # Check trend
                trend = data.get('trend', {})
                if trend.get('pct_change_5d') is not None:
                    if trend['pct_change_5d'] > 0:
                        trend_positive += 1
                    else:
                        trend_negative += 1
                
                # Check technicals
                tech = data.get('technicals', {})
                # RSI > 50 and MACD > Signal are bullish
                if tech.get('rsi') is not None and tech.get('macd') is not None and tech.get('macd_signal') is not None:
                    if tech['rsi'] > 50 and tech['macd'] > tech['macd_signal']:
                        tech_bullish += 1
                    elif tech['rsi'] < 50 and tech['macd'] < tech['macd_signal']:
                        tech_bearish += 1
                
                # Check forecast
                forecast = data.get('forecast', {})
                if forecast and '30_day' in forecast:
                    day30 = forecast['30_day']
                    if day30.get('percent_change', 0) > 0:
                        forecast_positive += 1
                    else:
                        forecast_negative += 1
                
                # Check for potential issues
                if (trend.get('pct_change_5d', 0) < -0.05 and  # 5% drop in 5 days
                    tech.get('rsi', 70) < 30 and               # Oversold
                    forecast and '30_day' in forecast and
                    forecast['30_day'].get('percent_change', 0) < -0.05):  # Forecast 5% drop
                    
                    symbols_at_risk.append({
                        'symbol': symbol,
                        'reasons': [
                            'Recent downtrend',
                            'Oversold technicals',
                            'Negative forecast'
                        ]
                    })
        
        # Calculate overall health score
        total_symbols = len(symphony.universe.symbols)
        if total_symbols > 0:
            trend_score = (trend_positive - trend_negative) / total_symbols
            tech_score = (tech_bullish - tech_bearish) / total_symbols
            forecast_score = (forecast_positive - forecast_negative) / total_symbols
            
            # Weighted average of scores
            health_score = 0.3 * trend_score + 0.3 * tech_score + 0.4 * forecast_score
            
            # Scale to 0-100
            health_score = 50 + (health_score * 50)
        else:
            health_score = 50  # Neutral
        
        return {
            'health_score': health_score,
            'symbols_at_risk': symbols_at_risk,
            'trend_positive': trend_positive,
            'trend_negative': trend_negative,
            'tech_bullish': tech_bullish,
            'tech_bearish': tech_bearish,
            'forecast_positive': forecast_positive,
            'forecast_negative': forecast_negative
        }

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Alpha Vantage client
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
    else:
        client = AlphaVantageClient(api_key=api_key)
        
        # Create symphony analyzer
        analyzer = SymphonyAnalyzer(client)
        
        # Create a test symphony
        universe = SymbolList(['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'LQD', 'HYG'])
        symphony = Symphony('Test Symphony', 'A simple test symphony', universe)
        
        # Add some operators
        from composer_symphony import Momentum, RSIFilter
        symphony.add_operator(Momentum('Momentum Filter', lookback_days=90, top_n=3))
        symphony.add_operator(RSIFilter('RSI Oversold', threshold=30, condition='below'))
        
        # Set an allocator
        from composer_symphony import InverseVolatilityAllocator
        symphony.set_allocator(InverseVolatilityAllocator(lookback_days=30))
        
        # Analyze the symphony
        today = datetime.now().strftime('%Y-%m-%d')
        lookback = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        results = analyzer.analyze_symphony(
            symphony,
            lookback,
            today,
            scenarios=['Bull Market', 'Bear Market'],
            forecast_days=30
        )
        
        # Create variations
        param_space = {
            'operators': {
                'Momentum': {
                    'lookback_days': [30, 60, 90, 120],
                    'top_n': [2, 3, 4, 5]
                },
                'RSIFilter': {
                    'threshold': [20, 25, 30, 35],
                    'condition': ['below', 'above']
                }
            },
            'allocator': {
                'InverseVolatilityAllocator': {
                    'lookback_days': [21, 30, 60]
                }
            }
        }
        
        variations = analyzer.generate_symphony_variations(
            symphony,
            param_space,
            n_variations=3
        )
        
        # Create watchlist
        watchlist = analyzer.create_symphony_watchlist([symphony] + variations)
        
        print(json.dumps(watchlist, indent=2))
