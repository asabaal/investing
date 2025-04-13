"""
Filter Types Module

This module provides specific filter implementations for common technical indicators.
These filters extend the base filter classes to provide ready-to-use filters for
trading strategies.
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd

from technical_indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_momentum,
    calculate_volatility,
    calculate_drawdown,
    calculate_returns
)

from filter_base import (
    ThresholdFilter,
    RangeFilter,
    CrossoverFilter,
    RankFilter,
    PercentileFilter
)

logger = logging.getLogger(__name__)


class RSIFilter(ThresholdFilter):
    """
    Filter that selects symbols based on RSI values.
    """
    
    def __init__(self, name: str, threshold: float = 30.0, condition: str = 'below',
                method: str = 'last', lookback_days: int = 1, rsi_period: int = 14):
        """
        Initialize an RSI filter.
        
        Args:
            name: Filter name
            threshold: RSI threshold value (default: 30.0)
            condition: 'below' or 'above' (default: 'below')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back for RSI evaluation (default: 1)
            rsi_period: Period for RSI calculation (default: 14)
        """
        super().__init__(
            name=name,
            indicator_func=calculate_rsi,
            threshold=threshold,
            condition=condition,
            method=method,
            lookback_days=lookback_days,
            indicator_params={'period': rsi_period}
        )


class MovingAverageFilter(ThresholdFilter):
    """
    Filter that selects symbols based on their price relative to a moving average.
    """
    
    def __init__(self, name: str, ma_type: str = 'sma', ma_period: int = 50, 
                condition: str = 'above', method: str = 'last', lookback_days: int = 1):
        """
        Initialize a moving average filter.
        
        Args:
            name: Filter name
            ma_type: 'sma' or 'ema' (default: 'sma')
            ma_period: Moving average period (default: 50)
            condition: 'above' or 'below' (default: 'above')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back (default: 1)
        """
        indicator_func = calculate_sma if ma_type.lower() == 'sma' else calculate_ema
        
        # Instead of comparing the indicator to a threshold, we need to compare
        # the price to the moving average. To do this, we'll override execute().
        super().__init__(
            name=name,
            indicator_func=indicator_func,
            threshold=0.0,  # Placeholder, not used
            condition=condition,
            method=method,
            lookback_days=lookback_days,
            indicator_params={'period': ma_period}
        )
        self.ma_type = ma_type
        self.ma_period = ma_period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'ma_type': self.ma_type,
            'ma_period': self.ma_period
        })
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
        This override calculates the moving average and then compares prices to it.
        
        Args:
            symbols: List of symbols to filter
            market_data: Dictionary mapping symbols to market data
            technical_data: Optional dictionary mapping symbols to technical data
            
        Returns:
            Filtered list of symbols
        """
        selected_symbols = []
        
        for symbol in symbols:
            if symbol not in market_data:
                logger.warning(f"No market data for symbol {symbol}")
                continue
            
            try:
                # Get price data
                prices = market_data[symbol]['adjusted_close'].values
                
                # Calculate moving average
                if self.ma_type.lower() == 'sma':
                    ma_values = calculate_sma(prices, period=self.ma_period)
                else:  # 'ema'
                    ma_values = calculate_ema(prices, period=self.ma_period)
                
                # For the comparison, we'll create a "difference" series (price - MA)
                diff_values = prices - ma_values
                
                # If condition is 'above', we want diff_values > 0
                # If condition is 'below', we want diff_values < 0
                threshold = 0.0
                diff_condition = 'above' if self.condition == 'above' else 'below'
                
                # Now we can use the standard threshold evaluation
                meets_condition = self._evaluate_threshold(
                    diff_values, 
                    threshold, 
                    diff_condition, 
                    self.method, 
                    self.lookback_days
                )
                
                if meets_condition:
                    selected_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error applying MA filter to {symbol}: {str(e)}")
        
        return selected_symbols
    
    def _evaluate_threshold(self, values, threshold, condition, method, lookback):
        """Helper method to evaluate threshold conditions."""
        import numpy as np
        
        # Handle NaN values
        valid_values = ~np.isnan(values)
        if np.sum(valid_values) == 0:
            return False
        
        # Get the values to evaluate based on lookback
        if len(values) <= lookback:
            lookback = len(values)
        
        lookback_values = values[-lookback:]
        lookback_valid = ~np.isnan(lookback_values)
        
        # If all values in lookback period are NaN, return False
        if np.sum(lookback_valid) == 0:
            return False
        
        # Filter out NaN values
        lookback_values = lookback_values[lookback_valid]
        
        # Evaluate based on method
        if method == 'last':
            # Only evaluate the most recent valid value
            if condition == 'above':
                return lookback_values[-1] > threshold
            else:  # 'below'
                return lookback_values[-1] < threshold
                
        elif method == 'any':
            # Return True if any value meets the condition
            if condition == 'above':
                return np.any(lookback_values > threshold)
            else:  # 'below'
                return np.any(lookback_values < threshold)
                
        elif method == 'all':
            # Return True if all values meet the condition
            if condition == 'above':
                return np.all(lookback_values > threshold)
            else:  # 'below'
                return np.all(lookback_values < threshold)
                
        elif method == 'average':
            # Evaluate the average of values
            avg_value = np.mean(lookback_values)
            if condition == 'above':
                return avg_value > threshold
            else:  # 'below'
                return avg_value < threshold
        
        return False


class MACrossoverFilter(CrossoverFilter):
    """
    Filter that selects symbols based on moving average crossovers.
    """
    
    def __init__(self, name: str, fast_period: int = 50, slow_period: int = 200,
                condition: str = 'above', lookback_days: int = 5, ma_type: str = 'sma'):
        """
        Initialize a moving average crossover filter.
        
        Args:
            name: Filter name
            fast_period: Fast moving average period (default: 50)
            slow_period: Slow moving average period (default: 200)
            condition: 'above' (golden cross) or 'below' (death cross) (default: 'above')
            lookback_days: Number of days to look back for crossover (default: 5)
            ma_type: 'sma' or 'ema' (default: 'sma')
        """
        # Select the appropriate MA function
        ma_func = calculate_sma if ma_type.lower() == 'sma' else calculate_ema
        
        super().__init__(
            name=name,
            fast_func=ma_func,
            slow_func=ma_func,
            condition=condition,
            lookback_days=lookback_days,
            fast_params={'period': fast_period},
            slow_params={'period': slow_period}
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'ma_type': self.ma_type
        })
        return result


class MomentumFilter(RankFilter):
    """
    Filter that selects symbols based on momentum ranking.
    """
    
    def __init__(self, name: str, period: int = 30, top_n: int = 5):
        """
        Initialize a momentum filter.
        
        Args:
            name: Filter name
            period: Momentum calculation period (default: 30)
            top_n: Number of top momentum symbols to select (default: 5)
        """
        super().__init__(
            name=name,
            indicator_func=calculate_momentum,
            rank_type='top',
            count=top_n,
            indicator_params={'period': period}
        )
        self.period = period
        self.top_n = top_n
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'period': self.period,
            'top_n': self.top_n
        })
        return result


class VolatilityFilter(PercentileFilter):
    """
    Filter that selects symbols based on volatility ranking.
    """
    
    def __init__(self, name: str, period: int = 63, percentile: float = 0.20, rank_type: str = 'bottom'):
        """
        Initialize a volatility filter.
        
        Args:
            name: Filter name
            period: Volatility calculation period (default: 63)
            percentile: Percentile threshold (0.0 to 1.0) (default: 0.20)
            rank_type: 'top' (high volatility) or 'bottom' (low volatility) (default: 'bottom')
        """
        super().__init__(
            name=name,
            indicator_func=calculate_volatility,
            percentile=percentile,
            rank_type=rank_type,
            indicator_params={'period': period}
        )
        self.period = period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'period': self.period
        })
        return result


class DrawdownFilter(ThresholdFilter):
    """
    Filter that selects symbols based on drawdown.
    """
    
    def __init__(self, name: str, threshold: float = 0.10, condition: str = 'below',
                method: str = 'last', lookback_days: int = 1):
        """
        Initialize a drawdown filter.
        
        Args:
            name: Filter name
            threshold: Drawdown threshold value (0.0 to 1.0) (default: 0.10)
            condition: 'below' or 'above' (default: 'below')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back (default: 1)
        """
        super().__init__(
            name=name,
            indicator_func=self._get_drawdown,
            threshold=threshold,
            condition=condition,
            method=method,
            lookback_days=lookback_days
        )
    
    def _get_drawdown(self, prices):
        """Helper function to get current drawdown values."""
        drawdown, _ = calculate_drawdown(prices)
        return drawdown


class ReturnFilter(ThresholdFilter):
    """
    Filter that selects symbols based on returns over a period.
    """
    
    def __init__(self, name: str, period: int = 30, threshold: float = 0.0, 
                condition: str = 'above', method: str = 'last', lookback_days: int = 1):
        """
        Initialize a return filter.
        
        Args:
            name: Filter name
            period: Return calculation period (default: 30)
            threshold: Return threshold value (default: 0.0)
            condition: 'above' or 'below' (default: 'above')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back (default: 1)
        """
        super().__init__(
            name=name,
            indicator_func=calculate_returns,
            threshold=threshold,
            condition=condition,
            method=method,
            lookback_days=lookback_days,
            indicator_params={'period': period}
        )
        self.period = period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'period': self.period
        })
        return result
