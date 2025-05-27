"""
Symphony Logic Engine

Executes symphony trading logic based on configuration and market data.
Handles conditional statements, sorting, and weight allocation.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from metrics_module import SymphonyMetrics, WeightingStrategies

@dataclass
class SymphonyResult:
    """Results of symphony execution"""
    allocation: Dict[str, float]  # Symbol -> weight
    triggered_condition: str      # Which condition was triggered
    metrics_used: Dict[str, Any]  # Metrics calculated during execution
    execution_date: str           # When this was calculated

class SymphonyEngine:
    """Execute symphony trading logic"""
    
    def __init__(self):
        self.metrics_calculator = SymphonyMetrics()
        self.weighting = WeightingStrategies()
        
    def load_symphony(self, config_path: str) -> dict:
        """Load symphony configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['name', 'universe', 'logic']
        symphony_config = config.get('symphony', {})
        
        for field in required_fields:
            if field not in symphony_config:
                raise ValueError(f"Missing required field: {field}")
        
        return symphony_config
    
    def execute_symphony(self, config: dict, market_data: Dict[str, pd.DataFrame], 
                        execution_date: str = None) -> SymphonyResult:
        """
        Execute symphony logic and return portfolio allocation
        
        Args:
            config: Symphony configuration dictionary
            market_data: {symbol: DataFrame} with OHLCV data for each symbol
            execution_date: Date of execution (for records)
            
        Returns:
            SymphonyResult with allocation and metadata
        """
        if execution_date is None:
            execution_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        universe = config['universe']
        logic = config['logic']
        
        # Validate all universe symbols have data
        missing_data = [symbol for symbol in universe if symbol not in market_data]
        if missing_data:
            raise ValueError(f"Missing market data for symbols: {missing_data}")
        
        metrics_used = {}
        
        # Evaluate conditions
        triggered_condition = self._evaluate_conditions(logic, market_data, metrics_used)
        
        # Get allocation based on triggered condition
        allocation = self._execute_allocation(
            logic['allocations'][triggered_condition], 
            universe, 
            market_data, 
            metrics_used
        )
        
        return SymphonyResult(
            allocation=allocation,
            triggered_condition=triggered_condition,
            metrics_used=metrics_used,
            execution_date=execution_date
        )
    
    def _evaluate_conditions(self, logic: dict, market_data: Dict[str, pd.DataFrame], 
                           metrics_used: dict) -> str:
        """Evaluate conditional statements and return which allocation to use"""
        
        conditions = logic.get('conditions', [])
        
        # If no conditions, use the first allocation
        if not conditions:
            allocations = list(logic['allocations'].keys())
            return allocations[0] if allocations else 'default'
        
        # Evaluate each condition
        for condition in conditions:
            condition_result = self._evaluate_single_condition(
                condition, market_data, metrics_used
            )
            
            if condition_result:
                return condition['if_true']
            else:
                return condition['if_false']
        
        # Fallback
        return list(logic['allocations'].keys())[0]
    
    def _evaluate_single_condition(self, condition: dict, market_data: Dict[str, pd.DataFrame],
                                 metrics_used: dict) -> bool:
        """Evaluate a single if-statement condition"""
        
        condition_def = condition['condition']
        metric = condition_def['metric']
        asset_1 = condition_def['asset_1']
        operator = condition_def['operator']
        asset_2 = condition_def['asset_2']
        lookback_days = condition_def.get('lookback_days', 20)
        
        # Calculate metric for asset_1
        if asset_1 not in market_data:
            raise ValueError(f"Asset {asset_1} not found in market data")
        
        value_1 = self.metrics_calculator.calculate_metric(
            market_data[asset_1], metric, lookback_days
        )
        
        # Store metric for tracking
        metrics_used[f"{asset_1}_{metric}_{lookback_days}d"] = value_1
        
        # Calculate value for asset_2 (either another asset or fixed value)
        if isinstance(asset_2, dict) and asset_2.get('type') == 'fixed_value':
            value_2 = asset_2['value']
        else:
            # asset_2 is another symbol
            if asset_2 not in market_data:
                raise ValueError(f"Asset {asset_2} not found in market data")
            
            value_2 = self.metrics_calculator.calculate_metric(
                market_data[asset_2], metric, lookback_days
            )
            metrics_used[f"{asset_2}_{metric}_{lookback_days}d"] = value_2
        
        # Evaluate condition
        return self._compare_values(value_1, operator, value_2)
    
    def _compare_values(self, value_1: float, operator: str, value_2: float) -> bool:
        """Compare two values using the specified operator"""
        
        operators = {
            'greater_than': lambda x, y: x > y,
            'less_than': lambda x, y: x < y,
            'greater_than_or_equal': lambda x, y: x >= y,
            'less_than_or_equal': lambda x, y: x <= y,
            'equal': lambda x, y: abs(x - y) < 1e-6,  # Floating point comparison
            'not_equal': lambda x, y: abs(x - y) >= 1e-6
        }
        
        if operator not in operators:
            raise ValueError(f"Unknown operator: {operator}")
        
        return operators[operator](value_1, value_2)
    
    def _execute_allocation(self, allocation_config: dict, universe: List[str], 
                          market_data: Dict[str, pd.DataFrame], metrics_used: dict) -> Dict[str, float]:
        """Execute allocation logic based on configuration"""
        
        allocation_type = allocation_config['type']
        
        if allocation_type == 'fixed_allocation':
            return allocation_config['weights']
        
        elif allocation_type == 'sort_and_weight':
            return self._execute_sort_and_weight(allocation_config, universe, market_data, metrics_used)
        
        else:
            raise ValueError(f"Unknown allocation type: {allocation_type}")
    
    def _execute_sort_and_weight(self, config: dict, universe: List[str], 
                               market_data: Dict[str, pd.DataFrame], metrics_used: dict) -> Dict[str, float]:
        """Execute sort and weight allocation strategy"""
        
        sort_config = config['sort']
        weight_config = config['weighting']
        
        # Calculate sort metric for all universe symbols
        metric = sort_config['metric']
        lookback_days = sort_config.get('lookback_days', 20)
        direction = sort_config['direction']  # 'top' or 'bottom'
        count = sort_config['count']
        
        symbol_metrics = {}
        for symbol in universe:
            if symbol in market_data:
                metric_value = self.metrics_calculator.calculate_metric(
                    market_data[symbol], metric, lookback_days
                )
                symbol_metrics[symbol] = metric_value
                metrics_used[f"{symbol}_{metric}_{lookback_days}d"] = metric_value
        
        # Sort symbols by metric
        sorted_symbols = sorted(
            symbol_metrics.items(), 
            key=lambda x: x[1], 
            reverse=(direction == 'top')
        )
        
        # Select top/bottom N symbols
        selected_symbols = [symbol for symbol, _ in sorted_symbols[:count]]
        
        # Apply weighting strategy
        weighting_method = weight_config['method']
        
        if weighting_method == 'equal_weight':
            weights = self.weighting.equal_weight(selected_symbols)
        
        elif weighting_method == 'inverse_volatility':
            lookback = weight_config.get('lookback_days', 30)
            # Filter market data for selected symbols only
            selected_data = {s: market_data[s] for s in selected_symbols if s in market_data}
            weights = self.weighting.inverse_volatility_weight(selected_symbols, selected_data, lookback)
        
        elif weighting_method == 'specified_weight':
            weights = self.weighting.specified_weight(weight_config['weights'])
        
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")
        
        return weights


# Backtesting Engine
class SymphonyBacktester:
    """Backtest symphony strategies over historical data"""
    
    def __init__(self):
        self.engine = SymphonyEngine()
        self.results = []
    
    def backtest(self, symphony_config: dict, market_data: Dict[str, pd.DataFrame], 
                start_date: str, end_date: str, rebalance_frequency: str = 'monthly') -> pd.DataFrame:
        """
        Run backtest over specified period
        
        Args:
            symphony_config: Symphony configuration
            market_data: Historical market data for all symbols
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
            DataFrame with backtest results
        """
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date, rebalance_frequency)
        
        backtest_results = []
        current_allocation = {}
        
        for date in rebalance_dates:
            try:
                # Get market data up to this date for all symbols
                date_market_data = {}
                for symbol, data in market_data.items():
                    symbol_data = data[data.index <= date]
                    if len(symbol_data) > 0:
                        date_market_data[symbol] = symbol_data
                
                # Execute symphony for this date
                result = self.engine.execute_symphony(
                    symphony_config, date_market_data, date.strftime('%Y-%m-%d')
                )
                
                # Calculate portfolio performance
                portfolio_return = self._calculate_portfolio_return(
                    current_allocation, result.allocation, date_market_data, date
                )
                
                backtest_results.append({
                    'date': date,
                    'allocation': result.allocation.copy(),
                    'triggered_condition': result.triggered_condition,
                    'portfolio_return': portfolio_return,
                    'metrics': result.metrics_used.copy()
                })
                
                current_allocation = result.allocation.copy()
                
            except Exception as e:
                print(f"Error on {date}: {e}")
                continue
        
        return pd.DataFrame(backtest_results)
    
    def _generate_rebalance_dates(self, start_date: str, end_date: str, frequency: str) -> List[pd.Timestamp]:
        """Generate list of rebalance dates"""
        
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        
        if frequency not in freq_map:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq_map[frequency])
        return dates.tolist()
    
    def _calculate_portfolio_return(self, old_allocation: dict, new_allocation: dict, 
                                  market_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """Calculate portfolio return between rebalances"""
        
        if not old_allocation:
            return 0.0
        
        total_return = 0.0
        
        for symbol, weight in old_allocation.items():
            if symbol in market_data and len(market_data[symbol]) >= 2:
                # Get price data
                symbol_data = market_data[symbol]
                if len(symbol_data) >= 2:
                    # Calculate return since last rebalance
                    current_price = symbol_data['Close'].iloc[-1]
                    previous_price = symbol_data['Close'].iloc[-2]
                    symbol_return = (current_price - previous_price) / previous_price
                    
                    # Weight by portfolio allocation
                    total_return += weight * symbol_return
        
        return total_return


# Example usage
if __name__ == "__main__":
    
    # Example: Create sample symphony configuration
    sample_symphony = {
        'name': 'Simple Momentum Strategy',
        'universe': ['AAPL', 'MSFT', 'GOOGL'],
        'logic': {
            'conditions': [
                {
                    'id': 'market_momentum',
                    'type': 'if_statement',
                    'condition': {
                        'metric': 'cumulative_return',
                        'asset_1': 'AAPL',
                        'operator': 'greater_than',
                        'asset_2': {'type': 'fixed_value', 'value': 0.05},
                        'lookback_days': 30
                    },
                    'if_true': 'momentum_allocation',
                    'if_false': 'defensive_allocation'
                }
            ],
            'allocations': {
                'momentum_allocation': {
                    'type': 'sort_and_weight', 
                    'sort': {
                        'metric': 'cumulative_return',
                        'lookback_days': 60,
                        'direction': 'top',
                        'count': 2
                    },
                    'weighting': {
                        'method': 'equal_weight'
                    }
                },
                'defensive_allocation': {
                    'type': 'fixed_allocation',
                    'weights': {
                        'MSFT': 1.0
                    }
                }
            }
        }
    }
    
    print("Symphony Engine Test")  
    print("=" * 40)
    
    # Create sample market data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    market_data = {}
    
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    # Test symphony execution
    engine = SymphonyEngine()
    
    try:
        result = engine.execute_symphony(sample_symphony, market_data)
        
        print(f"Symphony: {sample_symphony['name']}")
        print(f"Triggered Condition: {result.triggered_condition}")
        print(f"Allocation: {result.allocation}")
        print(f"Execution Date: {result.execution_date}")
        print("\nMetrics Used:")
        for metric, value in result.metrics_used.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error executing symphony: {e}")
        import traceback
        traceback.print_exc()
