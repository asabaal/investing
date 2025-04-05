"""
Symphony Simulation Module

This module provides tools for simulating Composer symphonies under different
market conditions and generating variations to test strategy robustness.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from composer_symphony import Symphony, SymphonyBacktester, SymbolList
from alpha_vantage_api import AlphaVantageClient
from symphony_analyzer import MarketScenario

# Configure logging
logger = logging.getLogger(__name__)

class SymphonyVariationGenerator:
    """
    Generate variations of a symphony for robustness testing.
    
    This class creates systematic variations of a symphony by modifying
    parameters of operators and allocators to assess how sensitive the
    strategy is to parameter changes.
    """
    
    def __init__(self, client: AlphaVantageClient):
        """
        Initialize the variation generator.
        
        Args:
            client: Alpha Vantage client for market data
        """
        self.client = client
    
    def generate_variations(
        self, 
        base_symphony: Symphony,
        parameter_space: Dict[str, Any],
        num_variations: int = 5,
        random_seed: Optional[int] = None
    ) -> List[Symphony]:
        """
        Generate variations of a symphony by perturbing parameters.
        
        Args:
            base_symphony: Base symphony to vary
            parameter_space: Parameter space to explore
            num_variations: Number of variations to generate
            random_seed: Optional seed for reproducibility
            
        Returns:
            List of symphony variations
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        variations = []
        
        # Generate variations
        for i in range(num_variations):
            # Create a copy of the base symphony
            variation_dict = base_symphony.to_dict()
            variation_dict['name'] = f"{base_symphony.name} (Variation {i+1})"
            
            # Modify operators
            if 'operators' in parameter_space and variation_dict.get('operators'):
                for op in variation_dict['operators']:
                    op_type = op['type']
                    if op_type in parameter_space.get('operators', {}):
                        op_params = parameter_space['operators'][op_type]
                        
                        # Apply random parameter values
                        for param, values in op_params.items():
                            if param in op.get('condition', {}):
                                # Select random value
                                random_value = np.random.choice(values)
                                op['condition'][param] = random_value
            
            # Modify allocator
            if 'allocator' in parameter_space and variation_dict.get('allocator'):
                allocator_type = variation_dict['allocator']['type']
                if allocator_type in parameter_space.get('allocator', {}):
                    allocator_params = parameter_space['allocator'][allocator_type]
                    
                    # Apply random parameter values
                    for param, values in allocator_params.items():
                        if param in variation_dict['allocator']:
                            # Select random value
                            random_value = np.random.choice(values)
                            variation_dict['allocator'][param] = random_value
            
            # Create symphony from modified dict
            try:
                variation = Symphony.from_dict(variation_dict)
                variations.append(variation)
            except Exception as e:
                logger.warning(f"Failed to create variation {i+1}: {str(e)}")
        
        return variations
    
    def create_parameter_space(self, base_symphony: Symphony) -> Dict[str, Any]:
        """
        Create a parameter space based on the structure of a symphony.
        
        Args:
            base_symphony: Symphony to analyze
            
        Returns:
            Dictionary defining the parameter space for variations
        """
        parameter_space = {'operators': {}, 'allocator': {}}
        
        # Extract operator types and parameters
        base_dict = base_symphony.to_dict()
        for op in base_dict.get('operators', []):
            op_type = op['type']
            
            # Define parameter variations based on operator type
            if op_type == 'Momentum':
                parameter_space['operators'][op_type] = {
                    'lookback_days': [30, 60, 90, 120, 180],
                    'top_n': [2, 3, 4, 5, 6]
                }
            elif op_type == 'RSIFilter':
                parameter_space['operators'][op_type] = {
                    'threshold': [20, 25, 30, 35, 40],
                    'condition': ['below', 'above']
                }
            elif op_type == 'MovingAverageCrossover':
                parameter_space['operators'][op_type] = {
                    'fast_period': [20, 50, 100],
                    'slow_period': [50, 100, 200],
                    'condition': ['golden_cross', 'death_cross']
                }
        
        # Define allocator variations
        allocator = base_dict.get('allocator', {})
        allocator_type = allocator.get('type')
        
        if allocator_type == 'EqualWeightAllocator':
            parameter_space['allocator'][allocator_type] = {}
        elif allocator_type == 'InverseVolatilityAllocator':
            parameter_space['allocator'][allocator_type] = {
                'lookback_days': [21, 30, 60, 90]
            }
        
        return parameter_space

class MarketScenarioGenerator:
    """
    Generate market scenarios for testing symphony performance.
    
    This class creates realistic market scenarios based on historical 
    patterns or synthetic data for stress testing trading strategies.
    """
    
    def __init__(self, client: AlphaVantageClient):
        """
        Initialize the scenario generator.
        
        Args:
            client: Alpha Vantage client for market data
        """
        self.client = client
    
    def generate_default_scenarios(self) -> List[MarketScenario]:
        """
        Generate a set of default market scenarios for testing.
        
        Returns:
            List of market scenarios
        """
        scenarios = [
            MarketScenario(
                name="Bull Market",
                description="Steady upward trend with normal volatility",
                trend='bullish',
                volatility='normal',
                correlation='normal',
                duration_days=90
            ),
            MarketScenario(
                name="Bear Market",
                description="Steady downward trend with elevated volatility",
                trend='bearish',
                volatility='high',
                correlation='high',
                duration_days=90
            ),
            MarketScenario(
                name="Market Crash",
                description="Sharp downward trend with extreme volatility",
                trend='bearish',
                volatility='extreme',
                correlation='high',
                duration_days=30
            ),
            MarketScenario(
                name="Sideways Market",
                description="Choppy market with no clear direction",
                trend='neutral',
                volatility='normal',
                correlation='low',
                duration_days=60
            ),
            MarketScenario(
                name="Recovery",
                description="Market recovering from a significant downturn",
                trend='bullish',
                volatility='high',
                correlation='decreasing',
                duration_days=90
            ),
            MarketScenario(
                name="Sector Rotation",
                description="Market characterized by shifting sector performance",
                trend='neutral',
                volatility='normal',
                correlation='low',
                duration_days=60
            ),
            MarketScenario(
                name="Rising Rate Environment",
                description="Market adjusting to rising interest rates",
                trend='bearish',
                volatility='high',
                correlation='normal',
                duration_days=90
            )
        ]
        
        return scenarios
    
    def generate_historical_scenarios(
        self,
        start_date: str = '2000-01-01',
        symbols: List[str] = ['SPY', 'QQQ', 'IWM']
    ) -> List[MarketScenario]:
        """
        Generate scenarios based on historical market periods.
        
        Args:
            start_date: Start date for historical analysis
            symbols: Symbols to analyze for detecting scenarios
            
        Returns:
            List of market scenarios based on historical periods
        """
        scenarios = []
        
        try:
            # Get historical data
            data = {}
            for symbol in symbols:
                df = self.client.get_daily(symbol, outputsize='full')
                if start_date:
                    df = df[df.index >= start_date]
                data[symbol] = df
            
            # Find bull markets (20%+ gains over 60+ days)
            # Find bear markets (20%+ losses over 60+ days)
            # Find crash periods (15%+ losses over <= 30 days)
            # Find sideways periods (< 5% change over 60+ days)
            
            # Example: Detect bull markets for SPY
            if 'SPY' in data:
                spy_data = data['SPY']
                spy_data['rolling_60d_return'] = spy_data['adjusted_close'].pct_change(60)
                
                # Find periods of 20%+ returns over 60 days
                bull_periods = spy_data[spy_data['rolling_60d_return'] >= 0.2]
                
                # Extract non-overlapping periods
                bull_start_dates = []
                last_end_date = None
                
                for date in bull_periods.index:
                    if last_end_date is None or date > last_end_date + timedelta(days=30):
                        period_start = date - timedelta(days=60)
                        period_end = date
                        
                        # Calculate statistics for this period
                        period_data = spy_data[(spy_data.index >= period_start) & 
                                            (spy_data.index <= period_end)]
                        
                        if len(period_data) > 30:
                            returns = period_data['adjusted_close'].pct_change().dropna()
                            volatility = returns.std() * np.sqrt(252)  # Annualized
                            
                            # Determine volatility level
                            vol_level = 'normal'
                            if volatility > 0.25:
                                vol_level = 'high'
                            elif volatility < 0.15:
                                vol_level = 'low'
                            
                            # Create scenario
                            scenario = MarketScenario(
                                name=f"Bull Market {period_start.strftime('%Y-%m')}",
                                description=f"Historical bull market from {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                                trend='bullish',
                                volatility=vol_level,
                                correlation='normal',
                                duration_days=(period_end - period_start).days
                            )
                            
                            scenarios.append(scenario)
                            bull_start_dates.append(period_start)
                            last_end_date = period_end
                
                # Similar logic can be applied for bear markets, crashes, etc.
                
        except Exception as e:
            logger.error(f"Error generating historical scenarios: {str(e)}")
        
        # If no historical scenarios found, return default ones
        if not scenarios:
            logger.warning("No historical scenarios found, returning defaults")
            return self.generate_default_scenarios()
        
        return scenarios
    
    def generate_synthetic_scenario(self, scenario: MarketScenario, base_date: str) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic market data for a specific scenario.
        
        Args:
            scenario: Market scenario to simulate
            base_date: Base date for scenario
            
        Returns:
            Dictionary of synthetic DataFrames by symbol
        """
        # This is a placeholder that would be implemented with more
        # sophisticated time series generation in a real application
        
        # Parameters based on scenario type
        trend_params = {
            'bullish': {'drift': 0.0005, 'trend_strength': 0.8},  # ~13% annual
            'bearish': {'drift': -0.0005, 'trend_strength': 0.8},  # ~-13% annual
            'neutral': {'drift': 0.0001, 'trend_strength': 0.2},  # ~2.5% annual
            'choppy': {'drift': 0.0, 'trend_strength': 0.1, 'reversal_prob': 0.4}
        }
        
        volatility_params = {
            'low': {'volatility': 0.008},  # ~13% annualized
            'normal': {'volatility': 0.012},  # ~19% annualized
            'high': {'volatility': 0.018},  # ~28% annualized
            'extreme': {'volatility': 0.025}  # ~40% annualized
        }
        
        correlation_params = {
            'low': {'correlation': 0.3},
            'normal': {'correlation': 0.7},
            'high': {'correlation': 0.9},
            'inverse': {'correlation': -0.5},
            'decreasing': {'correlation_start': 0.9, 'correlation_end': 0.3}
        }
        
        # Get appropriate parameters for this scenario
        t_params = trend_params.get(scenario.trend, trend_params['neutral'])
        v_params = volatility_params.get(scenario.volatility, volatility_params['normal'])
        c_params = correlation_params.get(scenario.correlation, correlation_params['normal'])
        
        # Synthetic data generation would go here
        # For now, return an empty dictionary
        return {}

class SymphonySimulator:
    """
    Simulate and test symphonies under different conditions.
    
    This class provides tools for comprehensive testing of symphonies
    under various market conditions and for different parameter variations.
    """
    
    def __init__(self, client: AlphaVantageClient):
        """
        Initialize the simulator.
        
        Args:
            client: Alpha Vantage client for market data
        """
        self.client = client
        self.backtester = SymphonyBacktester(client)
        self.variation_generator = SymphonyVariationGenerator(client)
        self.scenario_generator = MarketScenarioGenerator(client)
    
    def run_variation_test(
        self,
        base_symphony: Symphony,
        start_date: str,
        end_date: str,
        num_variations: int = 5,
        rebalance_frequency: str = 'monthly'
    ) -> Dict:
        """
        Test variations of a symphony to assess parameter sensitivity.
        
        Args:
            base_symphony: Base symphony to test
            start_date: Start date for testing
            end_date: End date for testing
            num_variations: Number of variations to generate
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Dictionary of test results
        """
        # Generate parameter space
        parameter_space = self.variation_generator.create_parameter_space(base_symphony)
        
        # Generate variations
        variations = self.variation_generator.generate_variations(
            base_symphony,
            parameter_space,
            num_variations
        )
        
        # Run backtest on base symphony
        base_results = self.backtester.backtest(
            base_symphony,
            start_date,
            end_date,
            rebalance_frequency=rebalance_frequency
        )
        
        # Run backtest on variations
        variation_results = []
        
        for i, variation in enumerate(variations):
            try:
                results = self.backtester.backtest(
                    variation,
                    start_date,
                    end_date,
                    rebalance_frequency=rebalance_frequency
                )
                
                if results.get('success', False):
                    # Extract key metrics
                    summary = results['backtest_summary']
                    
                    variation_results.append({
                        'variation': variation.name,
                        'parameters': variation.to_dict(),
                        'total_return': summary['total_return'],
                        'annual_return': summary['annual_return'],
                        'sharpe_ratio': summary['sharpe_ratio'],
                        'max_drawdown': summary['max_drawdown'],
                        'results': results
                    })
            except Exception as e:
                logger.warning(f"Failed to backtest variation {i+1}: {str(e)}")
        
        # Calculate parameter sensitivity
        sensitivity = self._calculate_parameter_sensitivity(
            base_symphony,
            variations,
            variation_results
        )
        
        return {
            'base_symphony': base_symphony.name,
            'base_results': base_results,
            'variations': variation_results,
            'parameter_sensitivity': sensitivity
        }
    
    def _calculate_parameter_sensitivity(
        self,
        base_symphony: Symphony,
        variations: List[Symphony],
        variation_results: List[Dict]
    ) -> Dict:
        """
        Calculate sensitivity to parameter changes.
        
        Args:
            base_symphony: Base symphony
            variations: List of symphony variations
            variation_results: List of variation backtest results
            
        Returns:
            Dictionary of parameter sensitivities
        """
        # This is a simplified implementation
        sensitivity = {'operators': {}, 'allocator': {}}
        
        # Extract base performance metrics
        base_dict = base_symphony.to_dict()
        base_metrics = None
        
        if 'base_results' in variation_results and variation_results['base_results'].get('success', False):
            base_summary = variation_results['base_results']['backtest_summary']
            base_metrics = {
                'total_return': base_summary['total_return'],
                'annual_return': base_summary['annual_return'],
                'sharpe_ratio': base_summary['sharpe_ratio'],
                'max_drawdown': base_summary['max_drawdown']
            }
        
        # Cannot calculate sensitivity without base metrics
        if base_metrics is None:
            return sensitivity
        
        # Collect parameter values and corresponding performance
        param_performances = {}
        
        for var_result in variation_results:
            # Skip failed backtests
            if 'parameters' not in var_result:
                continue
                
            # Extract parameters and performance
            var_params = var_result['parameters']
            var_metrics = {
                'total_return': var_result['total_return'],
                'annual_return': var_result['annual_return'],
                'sharpe_ratio': var_result['sharpe_ratio'],
                'max_drawdown': var_result['max_drawdown']
            }
            
            # Analyze operator parameters
            for op_idx, op in enumerate(var_params.get('operators', [])):
                op_type = op['type']
                
                if op_type not in param_performances:
                    param_performances[op_type] = {}
                
                # Check each parameter
                for param, value in op.get('condition', {}).items():
                    param_key = f"{op_type}_{param}"
                    
                    if param_key not in param_performances:
                        param_performances[param_key] = {}
                    
                    # Convert value to string for dictionary key
                    value_str = str(value)
                    
                    if value_str not in param_performances[param_key]:
                        param_performances[param_key][value_str] = []
                    
                    # Add performance for this parameter value
                    param_performances[param_key][value_str].append(var_metrics)
        
        # Calculate sensitivity for each parameter
        for param_key, value_performances in param_performances.items():
            # Skip parameters with insufficient data
            if len(value_performances) < 2:
                continue
            
            # Calculate variance in performance across parameter values
            metric = 'sharpe_ratio'  # Use Sharpe ratio for sensitivity
            
            param_values = []
            metric_values = []
            
            for value_str, performances in value_performances.items():
                if len(performances) > 0:
                    avg_metric = np.mean([p[metric] for p in performances])
                    param_values.append(value_str)
                    metric_values.append(avg_metric)
            
            # Calculate the range of metric values
            if len(metric_values) > 1:
                metric_range = max(metric_values) - min(metric_values)
                
                # Determine parameter type and name
                param_parts = param_key.split('_')
                op_type = param_parts[0]
                param_name = '_'.join(param_parts[1:])
                
                # Store sensitivity
                if op_type not in sensitivity['operators']:
                    sensitivity['operators'][op_type] = {}
                
                sensitivity['operators'][op_type][param_name] = metric_range
        
        return sensitivity
    
    def run_scenario_test(
        self,
        symphony: Symphony,
        scenarios: Optional[List[MarketScenario]] = None,
        base_date: Optional[str] = None
    ) -> Dict:
        """
        Test a symphony under different market scenarios.
        
        Args:
            symphony: Symphony to test
            scenarios: List of scenarios to test (or None for defaults)
            base_date: Base date for scenarios (or None for current date)
            
        Returns:
            Dictionary of scenario test results
        """
        # Use default scenarios if none provided
        if scenarios is None:
            scenarios = self.scenario_generator.generate_default_scenarios()
        
        # Use current date if none provided
        if base_date is None:
            base_date = datetime.now().strftime('%Y-%m-%d')
        
        # Run tests for each scenario
        scenario_results = []
        
        for scenario in scenarios:
            try:
                # Generate synthetic data for this scenario
                synthetic_data = self.scenario_generator.generate_synthetic_scenario(
                    scenario,
                    base_date
                )
                
                # A real implementation would use the synthetic data
                # For now, we'll return placeholder results
                
                scenario_results.append({
                    'scenario': scenario.name,
                    'description': scenario.description,
                    'trend': scenario.trend,
                    'volatility': scenario.volatility,
                    'correlation': scenario.correlation,
                    'duration_days': scenario.duration_days,
                    'total_return': np.random.uniform(-0.2, 0.3),  # Placeholder
                    'sharpe_ratio': np.random.uniform(-1, 2),  # Placeholder
                    'max_drawdown': np.random.uniform(0, 0.25)  # Placeholder
                })
                
            except Exception as e:
                logger.warning(f"Failed to test scenario {scenario.name}: {str(e)}")
                scenario_results.append({
                    'scenario': scenario.name,
                    'error': str(e)
                })
        
        return {
            'symphony': symphony.name,
            'base_date': base_date,
            'scenario_results': scenario_results
        }
    
    def visualize_variation_results(self, results: Dict, metric: str = 'sharpe_ratio') -> plt.Figure:
        """
        Visualize parameter sensitivity from variation tests.
        
        Args:
            results: Results from run_variation_test
            metric: Performance metric to visualize
            
        Returns:
            Matplotlib Figure
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Variation performance
        var_names = []
        var_metrics = []
        
        # Add base symphony
        if 'base_results' in results and results['base_results'].get('success', False):
            base_summary = results['base_results']['backtest_summary']
            var_names.append(results['base_symphony'])
            
            if metric == 'total_return':
                var_metrics.append(base_summary['total_return'])
            elif metric == 'annual_return':
                var_metrics.append(base_summary['annual_return'])
            elif metric == 'sharpe_ratio':
                var_metrics.append(base_summary['sharpe_ratio'])
            elif metric == 'max_drawdown':
                var_metrics.append(base_summary['max_drawdown'])
        
        # Add variations
        for var_result in results.get('variations', []):
            var_names.append(var_result['variation'])
            var_metrics.append(var_result[metric])
        
        # Sort by metric value
        sorted_indices = np.argsort(var_metrics)
        sorted_names = [var_names[i] for i in sorted_indices]
        sorted_metrics = [var_metrics[i] for i in sorted_indices]
        
        # Plot variation performance
        axs[0].barh(sorted_names, sorted_metrics)
        axs[0].set_title(f"Symphony Variations - {metric.replace('_', ' ').title()}")
        axs[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axs[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Parameter sensitivity
        sensitivities = []
        param_names = []
        
        # Extract sensitivities
        for op_type, params in results.get('parameter_sensitivity', {}).get('operators', {}).items():
            for param, sensitivity in params.items():
                param_names.append(f"{op_type}_{param}")
                sensitivities.append(sensitivity)
        
        # Sort by sensitivity
        if sensitivities:
            sorted_indices = np.argsort(sensitivities)
            sorted_params = [param_names[i] for i in sorted_indices]
            sorted_sensitivities = [sensitivities[i] for i in sorted_indices]
            
            # Plot parameter sensitivity
            axs[1].barh(sorted_params, sorted_sensitivities)
            axs[1].set_title("Parameter Sensitivity")
            axs[1].grid(axis='x', alpha=0.3)
        else:
            axs[1].text(0.5, 0.5, "Insufficient data for sensitivity analysis",
                      ha='center', va='center', transform=axs[1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def visualize_scenario_results(self, results: Dict) -> plt.Figure:
        """
        Visualize symphony performance across different scenarios.
        
        Args:
            results: Results from run_scenario_test
            
        Returns:
            Matplotlib Figure
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        scenario_results = results.get('scenario_results', [])
        
        if not scenario_results:
            for ax in axs:
                ax.text(0.5, 0.5, "No scenario results available",
                      ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract data
        scenarios = [r['scenario'] for r in scenario_results if 'error' not in r]
        returns = [r['total_return'] for r in scenario_results if 'error' not in r]
        sharpes = [r['sharpe_ratio'] for r in scenario_results if 'error' not in r]
        drawdowns = [r['max_drawdown'] for r in scenario_results if 'error' not in r]
        
        # Plot 1: Returns by scenario
        axs[0].bar(scenarios, returns)
        axs[0].set_title("Total Return by Scenario")
        axs[0].set_ylabel("Return")
        axs[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0].set_xticklabels(scenarios, rotation=45, ha='right')
        axs[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Sharpe ratios by scenario
        axs[1].bar(scenarios, sharpes)
        axs[1].set_title("Sharpe Ratio by Scenario")
        axs[1].set_ylabel("Sharpe Ratio")
        axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1].set_xticklabels(scenarios, rotation=45, ha='right')
        axs[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Max drawdowns by scenario
        axs[2].bar(scenarios, drawdowns)
        axs[2].set_title("Maximum Drawdown by Scenario")
        axs[2].set_ylabel("Drawdown")
        axs[2].set_xticklabels(scenarios, rotation=45, ha='right')
        axs[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig

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
        
        # Create simulator
        simulator = SymphonySimulator(client)
        
        # Create test symphony
        universe = SymbolList(['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'LQD', 'HYG'])
        symphony = Symphony('Test Symphony', 'A simple test symphony', universe)
        
        # Create dates for testing
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Run variation test
        print("Running variation test...")
        var_results = simulator.run_variation_test(
            symphony,
            start_date,
            end_date,
            num_variations=3
        )
        
        # Run scenario test
        print("Running scenario test...")
        scenario_results = simulator.run_scenario_test(
            symphony
        )
        
        print("Tests complete.")
