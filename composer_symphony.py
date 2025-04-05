"""
Composer Symphony Module

This module provides functionality for parsing, evaluating, and modifying
Composer symphonies. It includes tools for analyzing symphony performance
under various market conditions and generating variations of symphonies
for optimization.

Composer symphonies are trading strategies defined as a tree of operators that
combine various trading signals and filters to produce a portfolio allocation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from alpha_vantage_api import AlphaVantageClient

# Configure logging
logger = logging.getLogger(__name__)

class SymbolList:
    """
    A container for handling lists of symbols in a symphony.
    
    This class helps manage collections of symbols that are frequently used
    together in symphonies, providing utilities for validation and manipulation.
    """
    
    def __init__(self, symbols: List[str], name: Optional[str] = None):
        """
        Initialize a symbol list.
        
        Args:
            symbols: List of stock/ETF symbols
            name: Optional name for this list
        """
        self.symbols = [s.upper() for s in symbols]
        self.name = name
        
    def __repr__(self) -> str:
        if self.name:
            return f"SymbolList({self.name}: {', '.join(self.symbols)})"
        return f"SymbolList({', '.join(self.symbols)})"
        
    def __len__(self) -> int:
        return len(self.symbols)
        
    def __iter__(self):
        return iter(self.symbols)
        
    def __getitem__(self, index):
        return self.symbols[index]
        
    def validate_symbols(self, client: AlphaVantageClient) -> Dict[str, bool]:
        """
        Validate that all symbols exist and have data available.
        
        Args:
            client: Alpha Vantage client to use for validation
            
        Returns:
            Dictionary mapping symbols to boolean validity
        """
        result = {}
        
        for symbol in self.symbols:
            try:
                # Attempt to get a quote for the symbol
                client.get_quote(symbol)
                result[symbol] = True
            except Exception as e:
                logger.warning(f"Symbol {symbol} validation failed: {str(e)}")
                result[symbol] = False
                
        return result
        
    def add_symbol(self, symbol: str):
        """Add a symbol to the list."""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            
    def remove_symbol(self, symbol: str):
        """Remove a symbol from the list."""
        symbol = symbol.upper()
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            
    def intersect(self, other: 'SymbolList') -> 'SymbolList':
        """Return a new SymbolList with symbols in both lists."""
        common_symbols = [s for s in self.symbols if s in other.symbols]
        name = f"Intersection({self.name or 'unnamed'}, {other.name or 'unnamed'})"
        return SymbolList(common_symbols, name)
        
    def union(self, other: 'SymbolList') -> 'SymbolList':
        """Return a new SymbolList with symbols in either list."""
        all_symbols = list(set(self.symbols) | set(other.symbols))
        name = f"Union({self.name or 'unnamed'}, {other.name or 'unnamed'})"
        return SymbolList(all_symbols, name)

class SymbologyOperator:
    """
    Base class for operators that manipulate lists of symbols.
    
    This serves as the abstract base class for all operators that can
    be used in a symphony to filter, combine, or transform symbol lists.
    """
    
    def __init__(self, name: str):
        """
        Initialize an operator.
        
        Args:
            name: Operator name
        """
        self.name = name
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
        
    def execute(self, inputs: List[Any]) -> Any:
        """
        Execute the operator on the given inputs.
        
        Args:
            inputs: List of inputs to the operator
            
        Returns:
            Result of the operation
        """
        raise NotImplementedError("Subclasses must implement execute method")

class Filter(SymbologyOperator):
    """
    An operator that filters symbols based on some condition.
    """
    
    def __init__(self, name: str, condition: Dict[str, Any]):
        """
        Initialize a filter.
        
        Args:
            name: Filter name
            condition: Dictionary defining the filter condition
        """
        super().__init__(name)
        self.condition = condition
        
    def execute(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame]) -> SymbolList:
        """
        Apply the filter to a list of symbols.
        
        Args:
            symbols: Input symbol list
            market_data: Dictionary mapping symbols to market data
            
        Returns:
            Filtered symbol list
        """
        raise NotImplementedError("Specific filter types must implement this method")

class Momentum(Filter):
    """
    Filter that selects symbols based on momentum over a period.
    """
    
    def __init__(self, name: str, lookback_days: int = 30, top_n: int = 5):
        """
        Initialize a momentum filter.
        
        Args:
            name: Filter name
            lookback_days: Period to measure momentum
            top_n: Number of top symbols to select
        """
        condition = {'lookback_days': lookback_days, 'top_n': top_n}
        super().__init__(name, condition)
        self.lookback_days = lookback_days
        self.top_n = top_n
        
    def execute(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame]) -> SymbolList:
        """
        Select top N symbols by momentum.
        
        Args:
            symbols: Input symbol list
            market_data: Dictionary mapping symbols to market data
            
        Returns:
            Symbol list of top performers
        """
        momentum_values = {}
        
        for symbol in symbols:
            if symbol in market_data:
                df = market_data[symbol]
                
                # Calculate momentum as percentage change over lookback period
                if len(df) > self.lookback_days:
                    start_price = df['adjusted_close'].iloc[-self.lookback_days-1]
                    end_price = df['adjusted_close'].iloc[-1]
                    momentum = (end_price - start_price) / start_price
                    momentum_values[symbol] = momentum
        
        # Sort symbols by momentum and take top N
        sorted_symbols = sorted(momentum_values.keys(), 
                               key=lambda s: momentum_values[s], 
                               reverse=True)
        
        top_symbols = sorted_symbols[:min(self.top_n, len(sorted_symbols))]
        return SymbolList(top_symbols, f"Top_{self.top_n}_Momentum_{self.lookback_days}d")

class RSIFilter(Filter):
    """
    Filter that selects symbols based on RSI values.
    """
    
    def __init__(self, name: str, threshold: float = 30.0, condition: str = 'below'):
        """
        Initialize an RSI filter.
        
        Args:
            name: Filter name
            threshold: RSI threshold value
            condition: 'below' or 'above'
        """
        condition_dict = {'threshold': threshold, 'condition': condition}
        super().__init__(name, condition_dict)
        self.threshold = threshold
        self.condition = condition
        
    def execute(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame],
              technical_data: Dict[str, pd.DataFrame]) -> SymbolList:
        """
        Select symbols based on RSI condition.
        
        Args:
            symbols: Input symbol list
            market_data: Dictionary mapping symbols to market data
            technical_data: Dictionary mapping symbols to technical indicators
            
        Returns:
            Filtered symbol list
        """
        selected_symbols = []
        
        for symbol in symbols:
            if symbol in technical_data and 'RSI' in technical_data[symbol].columns:
                rsi_value = technical_data[symbol]['RSI'].iloc[-1]
                
                if (self.condition == 'below' and rsi_value < self.threshold) or \
                   (self.condition == 'above' and rsi_value > self.threshold):
                    selected_symbols.append(symbol)
        
        return SymbolList(selected_symbols, f"RSI_{self.condition}_{self.threshold}")

class MovingAverageCrossover(Filter):
    """
    Filter that selects symbols based on moving average crossovers.
    """
    
    def __init__(self, name: str, fast_period: int = 50, slow_period: int = 200,
               condition: str = 'golden_cross'):
        """
        Initialize a moving average crossover filter.
        
        Args:
            name: Filter name
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            condition: 'golden_cross' or 'death_cross'
        """
        condition_dict = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'condition': condition
        }
        super().__init__(name, condition_dict)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.condition = condition
        
    def execute(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame]) -> SymbolList:
        """
        Select symbols based on moving average crossover.
        
        Args:
            symbols: Input symbol list
            market_data: Dictionary mapping symbols to market data
            
        Returns:
            Filtered symbol list
        """
        selected_symbols = []
        
        for symbol in symbols:
            if symbol in market_data and len(market_data[symbol]) >= self.slow_period:
                df = market_data[symbol].copy()
                
                # Calculate moving averages
                df[f'MA_{self.fast_period}'] = df['adjusted_close'].rolling(window=self.fast_period).mean()
                df[f'MA_{self.slow_period}'] = df['adjusted_close'].rolling(window=self.slow_period).mean()
                
                # Check for crossover in the most recent data
                if len(df) > self.slow_period + 1:
                    prev_row = df.iloc[-2]
                    curr_row = df.iloc[-1]
                    
                    prev_fast = prev_row[f'MA_{self.fast_period}']
                    prev_slow = prev_row[f'MA_{self.slow_period}']
                    curr_fast = curr_row[f'MA_{self.fast_period}']
                    curr_slow = curr_row[f'MA_{self.slow_period}']
                    
                    # Golden cross: fast MA crosses above slow MA
                    golden_cross = prev_fast < prev_slow and curr_fast > curr_slow
                    
                    # Death cross: fast MA crosses below slow MA
                    death_cross = prev_fast > prev_slow and curr_fast < curr_slow
                    
                    if (self.condition == 'golden_cross' and golden_cross) or \
                       (self.condition == 'death_cross' and death_cross):
                        selected_symbols.append(symbol)
        
        crossover_type = "Golden" if self.condition == 'golden_cross' else "Death"
        return SymbolList(selected_symbols, f"{crossover_type}_Cross_{self.fast_period}_{self.slow_period}")

class Allocator(SymbologyOperator):
    """
    An operator that determines allocation weights for symbols.
    """
    
    def __init__(self, name: str, allocation_method: str):
        """
        Initialize an allocator.
        
        Args:
            name: Allocator name
            allocation_method: Method for determining weights
        """
        super().__init__(name)
        self.allocation_method = allocation_method
        
    def execute(self, symbols: SymbolList) -> Dict[str, float]:
        """
        Calculate allocation weights for symbols.
        
        Args:
            symbols: Symbol list to allocate
            
        Returns:
            Dictionary mapping symbols to weights
        """
        raise NotImplementedError("Specific allocator types must implement this method")

class EqualWeightAllocator(Allocator):
    """
    Allocates equal weight to all symbols.
    """
    
    def __init__(self, name: str = "Equal Weight"):
        """Initialize an equal weight allocator."""
        super().__init__(name, "equal_weight")
        
    def execute(self, symbols: SymbolList) -> Dict[str, float]:
        """
        Calculate equal weights for all symbols.
        
        Args:
            symbols: Symbol list to allocate
            
        Returns:
            Dictionary mapping symbols to equal weights
        """
        if not symbols or len(symbols) == 0:
            return {}
            
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

class InverseVolatilityAllocator(Allocator):
    """
    Allocates weights inversely proportional to volatility.
    """
    
    def __init__(self, name: str = "Inverse Volatility", lookback_days: int = 63):
        """
        Initialize an inverse volatility allocator.
        
        Args:
            name: Allocator name
            lookback_days: Period for volatility calculation
        """
        super().__init__(name, "inverse_volatility")
        self.lookback_days = lookback_days
        
    def execute(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate weights inversely proportional to volatility.
        
        Args:
            symbols: Symbol list to allocate
            market_data: Dictionary mapping symbols to market data
            
        Returns:
            Dictionary mapping symbols to weights
        """
        volatilities = {}
        
        for symbol in symbols:
            if symbol in market_data and len(market_data[symbol]) > self.lookback_days:
                df = market_data[symbol]
                
                # Calculate daily returns
                df = df.iloc[-self.lookback_days:]
                returns = df['adjusted_close'].pct_change().dropna()
                
                # Calculate volatility
                volatility = returns.std()
                if volatility > 0:  # Avoid division by zero
                    volatilities[symbol] = 1.0 / volatility
        
        # Normalize weights to sum to 1
        total_inverse_vol = sum(volatilities.values())
        if total_inverse_vol > 0:
            return {symbol: volatilities[symbol] / total_inverse_vol for symbol in volatilities}
        
        # Fallback to equal weight if volatility calculation fails
        return EqualWeightAllocator().execute(SymbolList(list(volatilities.keys())))

class Symphony:
    """
    A complete trading strategy defined by a tree of operators.
    
    A Symphony represents a trading strategy in Composer format, with
    a tree of operators that filter symbols and determine allocation.
    """
    
    def __init__(self, name: str, description: str, universe: SymbolList):
        """
        Initialize a Symphony.
        
        Args:
            name: Symphony name
            description: Description of the symphony
            universe: Universe of symbols for this symphony
        """
        self.name = name
        self.description = description
        self.universe = universe
        self.operators = []
        self.allocator = EqualWeightAllocator()
        
    def add_operator(self, operator: SymbologyOperator):
        """Add an operator to the symphony."""
        self.operators.append(operator)
        
    def set_allocator(self, allocator: Allocator):
        """Set the allocator for the symphony."""
        self.allocator = allocator
        
    def execute(self, client: AlphaVantageClient) -> Dict[str, float]:
        """
        Execute the symphony to generate allocations.
        
        Args:
            client: Alpha Vantage client for market data
            
        Returns:
            Dictionary mapping symbols to allocation weights
        """
        # Fetch market data for universe
        market_data = {}
        technical_data = {}
        
        for symbol in self.universe:
            try:
                # Get daily price data
                market_data[symbol] = client.get_daily(symbol)
                
                # Get RSI technical indicator
                technical_data[symbol] = client.get_rsi(symbol)
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {str(e)}")
        
        # Apply operators in sequence
        current_symbols = self.universe
        
        for operator in self.operators:
            if isinstance(operator, Filter):
                if isinstance(operator, RSIFilter):
                    current_symbols = operator.execute(current_symbols, market_data, technical_data)
                else:
                    current_symbols = operator.execute(current_symbols, market_data)
        
        # Apply allocator to final symbol list
        if isinstance(self.allocator, EqualWeightAllocator):
            allocations = self.allocator.execute(current_symbols)
        else:
            allocations = self.allocator.execute(current_symbols, market_data)
        
        return allocations
    
    def to_dict(self) -> Dict:
        """
        Convert symphony to a dictionary representation.
        
        Returns:
            Dictionary representation of the symphony
        """
        operators_list = []
        for op in self.operators:
            if isinstance(op, Filter):
                operators_list.append({
                    'type': op.__class__.__name__,
                    'name': op.name,
                    'condition': op.condition
                })
        
        return {
            'name': self.name,
            'description': self.description,
            'universe': self.universe.symbols,
            'operators': operators_list,
            'allocator': {
                'type': self.allocator.__class__.__name__,
                'name': self.allocator.name,
                'method': self.allocator.allocation_method
            }
        }
    
    def to_json(self) -> str:
        """
        Convert symphony to JSON string.
        
        Returns:
            JSON string representation of the symphony
        """
        return json.dumps(self.to_dict(), indent=4)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Symphony':
        """
        Create a Symphony from a dictionary representation.
        
        Args:
            data: Dictionary representing a symphony
            
        Returns:
            Symphony object
        """
        universe = SymbolList(data['universe'])
        symphony = cls(data['name'], data['description'], universe)
        
        # Add operators
        for op_data in data.get('operators', []):
            op_type = op_data['type']
            
            if op_type == 'Momentum':
                condition = op_data['condition']
                operator = Momentum(
                    op_data['name'],
                    lookback_days=condition.get('lookback_days', 30),
                    top_n=condition.get('top_n', 5)
                )
                symphony.add_operator(operator)
                
            elif op_type == 'RSIFilter':
                condition = op_data['condition']
                operator = RSIFilter(
                    op_data['name'],
                    threshold=condition.get('threshold', 30.0),
                    condition=condition.get('condition', 'below')
                )
                symphony.add_operator(operator)
                
            elif op_type == 'MovingAverageCrossover':
                condition = op_data['condition']
                operator = MovingAverageCrossover(
                    op_data['name'],
                    fast_period=condition.get('fast_period', 50),
                    slow_period=condition.get('slow_period', 200),
                    condition=condition.get('condition', 'golden_cross')
                )
                symphony.add_operator(operator)
        
        # Add allocator
        allocator_data = data.get('allocator', {})
        allocator_type = allocator_data.get('type', 'EqualWeightAllocator')
        
        if allocator_type == 'EqualWeightAllocator':
            symphony.set_allocator(EqualWeightAllocator(allocator_data.get('name', 'Equal Weight')))
            
        elif allocator_type == 'InverseVolatilityAllocator':
            symphony.set_allocator(InverseVolatilityAllocator(
                allocator_data.get('name', 'Inverse Volatility'),
                lookback_days=allocator_data.get('lookback_days', 63)
            ))
        
        return symphony
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Symphony':
        """
        Create a Symphony from a JSON string.
        
        Args:
            json_str: JSON string representing a symphony
            
        Returns:
            Symphony object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

class SymphonyBacktester:
    """
    Class for backtesting Composer symphonies.
    
    This class allows you to backtest a symphony over historical data
    and generate performance metrics and visualizations.
    """
    
    def __init__(self, client: AlphaVantageClient):
        """
        Initialize a backtester.
        
        Args:
            client: Alpha Vantage client for market data
        """
        self.client = client
        
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for a list of symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Get full history to ensure coverage of date range
                df = self.client.get_daily(symbol, outputsize='full')
                
                # Filter to requested date range
                start_date_pd = pd.to_datetime(start_date)
                end_date_pd = pd.to_datetime(end_date)
                
                df = df[(df.index >= start_date_pd) & (df.index <= end_date_pd)]
                
                if not df.empty:
                    historical_data[symbol] = df
                else:
                    logger.warning(f"No data available for {symbol} in date range {start_date} to {end_date}")
            except Exception as e:
                logger.warning(f"Failed to get historical data for {symbol}: {str(e)}")
        
        return historical_data
    
    def backtest(self, symphony: Symphony, start_date: str, end_date: str, 
               rebalance_frequency: str = 'monthly', initial_capital: float = 10000.0) -> Dict:
        """
        Backtest a symphony over a historical period.
        
        Args:
            symphony: Symphony to backtest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            rebalance_frequency: 'daily', 'weekly', 'monthly', or 'quarterly'
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary of backtest results
        """
        # Get historical data for all symbols in the universe
        historical_data = self.get_historical_data(symphony.universe.symbols, start_date, end_date)
        
        if not historical_data:
            logger.error("No historical data available for backtest")
            return {
                'success': False,
                'error': 'No historical data available for backtest'
            }
        
        # Create a date range for the backtest
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        
        # Determine rebalance dates
        if rebalance_frequency == 'daily':
            # Create business day date range
            date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='B')
        elif rebalance_frequency == 'weekly':
            # Weekly on Fridays
            date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='W-FRI')
        elif rebalance_frequency == 'monthly':
            # Monthly on last business day
            date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='BM')
        elif rebalance_frequency == 'quarterly':
            # Quarterly on last business day of quarter
            date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='BQ')
        else:
            logger.error(f"Invalid rebalance frequency: {rebalance_frequency}")
            return {
                'success': False,
                'error': f"Invalid rebalance frequency: {rebalance_frequency}"
            }
        
        # Keep only dates that exist in our data
        all_dates = set()
        for symbol, df in historical_data.items():
            all_dates.update(df.index.tolist())
        
        valid_dates = [date for date in date_range if date in all_dates]
        
        if not valid_dates:
            logger.error("No valid rebalance dates in data range")
            return {
                'success': False,
                'error': 'No valid rebalance dates in data range'
            }
        
        # Set up portfolio tracking
        portfolio_history = []
        current_positions = {}
        current_cash = initial_capital
        
        # Run backtest
        for rebalance_date in valid_dates:
            rebalance_date_str = rebalance_date.strftime('%Y-%m-%d')
            logger.info(f"Rebalancing on {rebalance_date_str}")
            
            # Get data up to rebalance date for symphony execution
            data_for_execution = {}
            for symbol, df in historical_data.items():
                df_to_date = df[df.index <= rebalance_date].copy()
                if not df_to_date.empty:
                    data_for_execution[symbol] = df_to_date
            
            # Execute symphony with data up to rebalance date
            # We'll mock the execution by using a simplified approach
            # In a real system, we would re-run the symphony logic with historical data
            mock_client = MockAlphaVantageClient(data_for_execution)
            allocations = self._mock_execute_symphony(symphony, mock_client, rebalance_date)
            
            # Calculate portfolio value before rebalance
            portfolio_value = current_cash
            for symbol, shares in current_positions.items():
                if symbol in data_for_execution and not data_for_execution[symbol].empty:
                    price = data_for_execution[symbol].loc[rebalance_date, 'adjusted_close']
                    portfolio_value += price * shares
            
            # Rebalance portfolio
            new_positions = {}
            for symbol, weight in allocations.items():
                if symbol in data_for_execution and not data_for_execution[symbol].empty:
                    price = data_for_execution[symbol].loc[rebalance_date, 'adjusted_close']
                    target_value = portfolio_value * weight
                    new_positions[symbol] = target_value / price
            
            # Record portfolio state
            positions_value = sum(
                new_positions.get(symbol, 0) * data_for_execution[symbol].loc[rebalance_date, 'adjusted_close']
                for symbol in new_positions
                if symbol in data_for_execution and not data_for_execution[symbol].empty
            )
            
            current_cash = portfolio_value - positions_value
            current_positions = new_positions
            
            portfolio_history.append({
                'date': rebalance_date,
                'portfolio_value': portfolio_value,
                'cash': current_cash,
                'positions': {s: p for s, p in new_positions.items()},
                'allocations': allocations
            })
        
        # Calculate performance metrics
        if len(portfolio_history) < 2:
            logger.error("Insufficient data for performance calculation")
            return {
                'success': False,
                'error': 'Insufficient data for performance calculation'
            }
        
        # Extract portfolio values
        dates = [entry['date'] for entry in portfolio_history]
        values = [entry['portfolio_value'] for entry in portfolio_history]
        
        # Calculate returns
        returns = []
        for i in range(1, len(values)):
            returns.append((values[i] - values[i-1]) / values[i-1])
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0]
        annual_return = (1 + total_return) ** (252 / len(dates)) - 1
        
        std_dev = np.std(returns)
        sharpe_ratio = np.mean(returns) / std_dev if std_dev != 0 else 0
        
        # Calculate drawdowns
        drawdowns = []
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Prepare results
        return {
            'success': True,
            'backtest_summary': {
                'initial_capital': initial_capital,
                'final_value': values[-1],
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'portfolio_history': portfolio_history
        }
    
    def _mock_execute_symphony(self, symphony: Symphony, mock_client: 'MockAlphaVantageClient', 
                            as_of_date: pd.Timestamp) -> Dict[str, float]:
        """
        Execute a symphony using mocked historical data.
        
        Args:
            symphony: Symphony to execute
            mock_client: Mock Alpha Vantage client with historical data
            as_of_date: Date to execute as of
            
        Returns:
            Dictionary of allocations
        """
        # This is a simplified approach - in a real implementation, we would
        # re-run the full symphony logic with the historical data
        
        # For now, we'll use a basic approach:
        # 1. For momentum filters, calculate lookback period momentum
        # 2. For RSI filters, calculate RSI values
        # 3. For MA crossovers, check for crossovers
        
        filtered_symbols = symphony.universe.symbols
        
        for operator in symphony.operators:
            if isinstance(operator, Momentum):
                filtered_symbols = self._apply_momentum_filter(
                    SymbolList(filtered_symbols),
                    mock_client.data,
                    operator.lookback_days,
                    operator.top_n,
                    as_of_date
                )
            elif isinstance(operator, RSIFilter):
                # Calculate RSI for each symbol
                rsi_data = {}
                for symbol in filtered_symbols:
                    if symbol in mock_client.data:
                        df = mock_client.data[symbol].copy()
                        df = df[df.index <= as_of_date]
                        if len(df) > 14:  # Need at least 14 days for RSI
                            df['return'] = df['adjusted_close'].pct_change()
                            df['gain'] = np.where(df['return'] > 0, df['return'], 0)
                            df['loss'] = np.where(df['return'] < 0, -df['return'], 0)
                            df['avg_gain'] = df['gain'].rolling(window=14).mean()
                            df['avg_loss'] = df['loss'].rolling(window=14).mean()
                            df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, 0.001)
                            df['rsi'] = 100 - (100 / (1 + df['rs']))
                            rsi_data[symbol] = df
                
                # Apply RSI filter
                filtered_symbols = [
                    symbol for symbol in filtered_symbols
                    if symbol in rsi_data and not rsi_data[symbol].empty and
                    ((operator.condition == 'below' and rsi_data[symbol]['rsi'].iloc[-1] < operator.threshold) or
                     (operator.condition == 'above' and rsi_data[symbol]['rsi'].iloc[-1] > operator.threshold))
                ]
            
            elif isinstance(operator, MovingAverageCrossover):
                filtered_symbols = self._apply_ma_crossover_filter(
                    SymbolList(filtered_symbols),
                    mock_client.data,
                    operator.fast_period,
                    operator.slow_period,
                    operator.condition,
                    as_of_date
                )
        
        # Apply allocator
        if isinstance(symphony.allocator, EqualWeightAllocator):
            if filtered_symbols:
                weight = 1.0 / len(filtered_symbols)
                return {symbol: weight for symbol in filtered_symbols}
            return {}
        
        elif isinstance(symphony.allocator, InverseVolatilityAllocator):
            # Calculate volatilities
            volatilities = {}
            for symbol in filtered_symbols:
                if symbol in mock_client.data:
                    df = mock_client.data[symbol].copy()
                    df = df[df.index <= as_of_date]
                    if len(df) > symphony.allocator.lookback_days:
                        returns = df['adjusted_close'].pct_change().dropna()
                        vol = returns.std()
                        if vol > 0:
                            volatilities[symbol] = 1.0 / vol
            
            # Normalize weights
            total = sum(volatilities.values())
            if total > 0:
                return {symbol: volatilities[symbol] / total for symbol in volatilities}
            
            # Fallback to equal weight
            if filtered_symbols:
                weight = 1.0 / len(filtered_symbols)
                return {symbol: weight for symbol in filtered_symbols}
            
        return {}
    
    def _apply_momentum_filter(self, symbols: SymbolList, data: Dict[str, pd.DataFrame],
                             lookback_days: int, top_n: int, as_of_date: pd.Timestamp) -> List[str]:
        """Apply momentum filter to historical data."""
        momentum_values = {}
        
        for symbol in symbols:
            if symbol in data:
                df = data[symbol].copy()
                df = df[df.index <= as_of_date]
                
                if len(df) > lookback_days:
                    start_price = df['adjusted_close'].iloc[-lookback_days-1]
                    end_price = df['adjusted_close'].iloc[-1]
                    momentum = (end_price - start_price) / start_price
                    momentum_values[symbol] = momentum
        
        # Sort symbols by momentum and take top N
        sorted_symbols = sorted(momentum_values.keys(), 
                               key=lambda s: momentum_values[s], 
                               reverse=True)
        
        return sorted_symbols[:min(top_n, len(sorted_symbols))]
    
    def _apply_ma_crossover_filter(self, symbols: SymbolList, data: Dict[str, pd.DataFrame],
                                 fast_period: int, slow_period: int, condition: str,
                                 as_of_date: pd.Timestamp) -> List[str]:
        """Apply moving average crossover filter to historical data."""
        selected_symbols = []
        
        for symbol in symbols:
            if symbol in data:
                df = data[symbol].copy()
                df = df[df.index <= as_of_date]
                
                if len(df) >= slow_period + 1:
                    # Calculate moving averages
                    df[f'fast_ma'] = df['adjusted_close'].rolling(window=fast_period).mean()
                    df[f'slow_ma'] = df['adjusted_close'].rolling(window=slow_period).mean()
                    
                    # Check for crossover
                    if len(df) > slow_period + 1:
                        prev_row = df.iloc[-2]
                        curr_row = df.iloc[-1]
                        
                        prev_fast = prev_row['fast_ma']
                        prev_slow = prev_row['slow_ma']
                        curr_fast = curr_row['fast_ma']
                        curr_slow = curr_row['slow_ma']
                        
                        # Golden cross: fast MA crosses above slow MA
                        golden_cross = prev_fast < prev_slow and curr_fast > curr_slow
                        
                        # Death cross: fast MA crosses below slow MA
                        death_cross = prev_fast > prev_slow and curr_fast < curr_slow
                        
                        if (condition == 'golden_cross' and golden_cross) or \
                           (condition == 'death_cross' and death_cross):
                            selected_symbols.append(symbol)
        
        return selected_symbols
    
    def plot_backtest_results(self, results: Dict, benchmark_symbol: Optional[str] = None):
        """
        Plot backtest results with optional benchmark comparison.
        
        Args:
            results: Results from backtest method
            benchmark_symbol: Optional symbol to use as benchmark
        """
        if not results['success']:
            logger.error(f"Cannot plot failed backtest: {results.get('error', 'Unknown error')}")
            return
        
        history = results['portfolio_history']
        dates = [entry['date'] for entry in history]
        values = [entry['portfolio_value'] for entry in history]
        
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(dates, values, label='Portfolio Value')
        
        # Add benchmark if provided
        if benchmark_symbol:
            try:
                # Get benchmark data for the same period
                start_date = min(dates)
                end_date = max(dates)
                
                benchmark_data = self.client.get_daily(benchmark_symbol)
                benchmark_data = benchmark_data.loc[
                    (benchmark_data.index >= start_date) & 
                    (benchmark_data.index <= end_date)
                ]
                
                # Scale benchmark to same starting value
                if not benchmark_data.empty:
                    benchmark_values = benchmark_data['adjusted_close']
                    scale_factor = values[0] / benchmark_values.iloc[0]
                    scaled_benchmark = benchmark_values * scale_factor
                    
                    plt.plot(benchmark_data.index, scaled_benchmark, label=f'Benchmark ({benchmark_symbol})')
            except Exception as e:
                logger.warning(f"Failed to plot benchmark {benchmark_symbol}: {str(e)}")
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        
        # Calculate drawdowns
        peak = values[0]
        drawdowns = []
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        plt.plot(dates, drawdowns)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.ylim(0, max(0.01, max(drawdowns) * 1.1))  # Ensure y-axis shows drawdowns properly
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

class MockAlphaVantageClient:
    """
    Mock Alpha Vantage client for backtesting.
    
    This class mimics the interface of AlphaVantageClient but uses
    pre-loaded historical data instead of making API calls.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize a mock client.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with historical data
        """
        self.data = data
    
    def get_daily(self, symbol: str, **kwargs) -> pd.DataFrame:
        """
        Get daily data for a symbol from the pre-loaded data.
        
        Args:
            symbol: Stock symbol
            **kwargs: Ignored parameters
            
        Returns:
            DataFrame with daily data
        """
        if symbol in self.data:
            return self.data[symbol].copy()
        else:
            raise RuntimeError(f"Symbol {symbol} not found in mock data")
    
    def get_rsi(self, symbol: str, **kwargs) -> pd.DataFrame:
        """
        Get RSI data for a symbol, calculated from the pre-loaded data.
        
        Args:
            symbol: Stock symbol
            **kwargs: Ignored parameters
            
        Returns:
            DataFrame with RSI data
        """
        if symbol in self.data:
            df = self.data[symbol].copy()
            
            # Calculate RSI
            df['return'] = df['adjusted_close'].pct_change()
            df['gain'] = np.where(df['return'] > 0, df['return'], 0)
            df['loss'] = np.where(df['return'] < 0, -df['return'], 0)
            df['avg_gain'] = df['gain'].rolling(window=14).mean()
            df['avg_loss'] = df['loss'].rolling(window=14).mean()
            df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, 0.001)
            df['RSI'] = 100 - (100 / (1 + df['rs']))
            
            # Return only the RSI column
            result = pd.DataFrame({'RSI': df['RSI']})
            result.index = df.index
            return result
        else:
            raise RuntimeError(f"Symbol {symbol} not found in mock data")
    
    def get_sma(self, symbol: str, time_period: int = 20, **kwargs) -> pd.DataFrame:
        """
        Get SMA data for a symbol, calculated from the pre-loaded data.
        
        Args:
            symbol: Stock symbol
            time_period: SMA period
            **kwargs: Ignored parameters
            
        Returns:
            DataFrame with SMA data
        """
        if symbol in self.data:
            df = self.data[symbol].copy()
            
            # Calculate SMA
            df[f'SMA_{time_period}'] = df['adjusted_close'].rolling(window=time_period).mean()
            
            # Return only the SMA column
            result = pd.DataFrame({f'SMA': df[f'SMA_{time_period}']})
            result.index = df.index
            return result
        else:
            raise RuntimeError(f"Symbol {symbol} not found in mock data")
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get latest quote for a symbol from the pre-loaded data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        if symbol in self.data:
            df = self.data[symbol]
            
            if not df.empty:
                latest_row = df.iloc[-1]
                
                return {
                    'symbol': symbol,
                    'open': float(latest_row['open']),
                    'high': float(latest_row['high']),
                    'low': float(latest_row['low']),
                    'price': float(latest_row['adjusted_close']),
                    'volume': int(latest_row['volume']),
                    'latest trading day': str(latest_row.name.date())
                }
        
        raise RuntimeError(f"Symbol {symbol} not found in mock data or has no data")

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a test symphony
    universe = SymbolList(['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'LQD', 'HYG'])
    symphony = Symphony('Test Symphony', 'A simple test symphony', universe)
    
    # Add some operators
    symphony.add_operator(Momentum('Momentum Filter', lookback_days=90, top_n=3))
    symphony.add_operator(RSIFilter('RSI Oversold', threshold=30, condition='below'))
    
    # Set an allocator
    symphony.set_allocator(InverseVolatilityAllocator(lookback_days=30))
    
    # Print the symphony definition
    print(symphony.to_json())
    
    # Note: To run a backtest, you would need an Alpha Vantage API key
    # client = AlphaVantageClient(api_key='YOUR_API_KEY')
    # backtester = SymphonyBacktester(client)
    # results = backtester.backtest(symphony, '2020-01-01', '2023-01-01')
    # backtester.plot_backtest_results(results, benchmark_symbol='SPY')
