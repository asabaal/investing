"""
Filter Base Module

This module provides the base classes for building filters in the filtering system.
These classes form the third layer of the filtering system architecture and are
designed to be extended by specific filter implementations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable

from evaluation_functions import (
    evaluate_threshold,
    evaluate_range,
    evaluate_crossover,
    evaluate_lookback,
    evaluate_persistence
)

logger = logging.getLogger(__name__)


class FilterBase:
    """
    Base class for all filters.
    
    This abstract base class defines the interface that all filters must implement.
    """
    
    def __init__(self, name: str, filter_type: str):
        """
        Initialize a filter.
        
        Args:
            name: Filter name
            filter_type: Type of filter
        """
        self.name = name
        self.filter_type = filter_type
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.filter_type}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert filter to a dictionary representation.
        
        Returns:
            Dictionary representation of the filter
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'filter_type': self.filter_type
        }
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
        Args:
            symbols: List of symbols to filter
            market_data: Dictionary mapping symbols to market data
            technical_data: Optional dictionary mapping symbols to technical data
            
        Returns:
            Filtered list of symbols
        """
        raise NotImplementedError("Subclasses must implement execute method")


class ThresholdFilter(FilterBase):
    """
    Filter that selects symbols based on whether values meet a threshold condition.
    """
    
    def __init__(self, name: str, indicator_func: Callable, threshold: float, condition: str = 'above',
                method: str = 'last', lookback_days: int = 1, indicator_params: Optional[Dict] = None):
        """
        Initialize a threshold filter.
        
        Args:
            name: Filter name
            indicator_func: Function to calculate the indicator values
            threshold: Threshold value
            condition: 'above' or 'below' (default: 'above')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back (default: 1)
            indicator_params: Additional parameters to pass to the indicator function
        """
        super().__init__(name, 'threshold')
        self.indicator_func = indicator_func
        self.threshold = threshold
        self.condition = condition
        self.method = method
        self.lookback_days = lookback_days
        self.indicator_params = indicator_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'threshold': self.threshold,
            'condition': self.condition,
            'method': self.method,
            'lookback_days': self.lookback_days,
            'indicator_params': self.indicator_params
        })
        # Note: indicator_func is not included as it's not JSON serializable
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
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
            
            # Calculate indicator values if not provided in technical_data
            indicator_values = None
            if technical_data is not None and symbol in technical_data:
                # Try to get indicator values from technical_data
                # This assumes technical_data may have different column names for different indicators
                for col in technical_data[symbol].columns:
                    if col.upper() in self.name.upper():
                        indicator_values = technical_data[symbol][col].values
                        break
            
            if indicator_values is None:
                # Calculate indicator values using the provided function
                try:
                    prices = market_data[symbol]['adjusted_close'].values
                    indicator_values = self.indicator_func(prices, **self.indicator_params)
                except Exception as e:
                    logger.error(f"Error calculating indicator for {symbol}: {str(e)}")
                    continue
            
            # Evaluate the indicator values against the threshold
            try:
                meets_condition = evaluate_threshold(
                    indicator_values, 
                    self.threshold, 
                    self.condition, 
                    self.method, 
                    self.lookback_days
                )
                
                if meets_condition:
                    selected_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error evaluating threshold for {symbol}: {str(e)}")
        
        return selected_symbols


class RangeFilter(FilterBase):
    """
    Filter that selects symbols based on whether values fall within a range.
    """
    
    def __init__(self, name: str, indicator_func: Callable, lower_bound: float, upper_bound: float,
                condition: str = 'inside', method: str = 'last', lookback_days: int = 1,
                indicator_params: Optional[Dict] = None):
        """
        Initialize a range filter.
        
        Args:
            name: Filter name
            indicator_func: Function to calculate the indicator values
            lower_bound: Lower bound of the range
            upper_bound: Upper bound of the range
            condition: 'inside' or 'outside' (default: 'inside')
            method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
            lookback_days: Number of days to look back (default: 1)
            indicator_params: Additional parameters to pass to the indicator function
        """
        super().__init__(name, 'range')
        self.indicator_func = indicator_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.condition = condition
        self.method = method
        self.lookback_days = lookback_days
        self.indicator_params = indicator_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'condition': self.condition,
            'method': self.method,
            'lookback_days': self.lookback_days,
            'indicator_params': self.indicator_params
        })
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
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
            
            # Calculate indicator values if not provided in technical_data
            indicator_values = None
            if technical_data is not None and symbol in technical_data:
                # Try to get indicator values from technical_data
                for col in technical_data[symbol].columns:
                    if col.upper() in self.name.upper():
                        indicator_values = technical_data[symbol][col].values
                        break
            
            if indicator_values is None:
                # Calculate indicator values using the provided function
                try:
                    prices = market_data[symbol]['adjusted_close'].values
                    indicator_values = self.indicator_func(prices, **self.indicator_params)
                except Exception as e:
                    logger.error(f"Error calculating indicator for {symbol}: {str(e)}")
                    continue
            
            # Evaluate the indicator values against the range
            try:
                meets_condition = evaluate_range(
                    indicator_values, 
                    self.lower_bound,
                    self.upper_bound, 
                    self.condition, 
                    self.method, 
                    self.lookback_days
                )
                
                if meets_condition:
                    selected_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error evaluating range for {symbol}: {str(e)}")
        
        return selected_symbols


class CrossoverFilter(FilterBase):
    """
    Filter that selects symbols based on indicator crossovers.
    """
    
    def __init__(self, name: str, fast_func: Callable, slow_func: Callable, condition: str = 'above',
                lookback_days: int = 5, fast_params: Optional[Dict] = None, slow_params: Optional[Dict] = None):
        """
        Initialize a crossover filter.
        
        Args:
            name: Filter name
            fast_func: Function to calculate the fast indicator values
            slow_func: Function to calculate the slow indicator values
            condition: 'above' (fast crosses above slow) or 'below' (fast crosses below slow)
                      (default: 'above')
            lookback_days: Number of days to look back (default: 5)
            fast_params: Additional parameters to pass to the fast indicator function
            slow_params: Additional parameters to pass to the slow indicator function
        """
        super().__init__(name, 'crossover')
        self.fast_func = fast_func
        self.slow_func = slow_func
        self.condition = condition
        self.lookback_days = lookback_days
        self.fast_params = fast_params or {}
        self.slow_params = slow_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'condition': self.condition,
            'lookback_days': self.lookback_days,
            'fast_params': self.fast_params,
            'slow_params': self.slow_params
        })
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
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
            
            # Calculate fast and slow indicator values
            try:
                prices = market_data[symbol]['adjusted_close'].values
                fast_values = self.fast_func(prices, **self.fast_params)
                slow_values = self.slow_func(prices, **self.slow_params)
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                continue
            
            # Evaluate the crossover
            try:
                has_crossover = evaluate_crossover(
                    fast_values, 
                    slow_values, 
                    self.condition, 
                    self.lookback_days
                )
                
                if has_crossover:
                    selected_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error evaluating crossover for {symbol}: {str(e)}")
        
        return selected_symbols


class RankFilter(FilterBase):
    """
    Filter that selects symbols based on ranking by an indicator.
    """
    
    def __init__(self, name: str, indicator_func: Callable, rank_type: str = 'top', 
                count: int = 5, indicator_params: Optional[Dict] = None):
        """
        Initialize a rank filter.
        
        Args:
            name: Filter name
            indicator_func: Function to calculate the indicator values
            rank_type: 'top' or 'bottom' (default: 'top')
            count: Number of symbols to select (default: 5)
            indicator_params: Additional parameters to pass to the indicator function
        """
        super().__init__(name, 'rank')
        self.indicator_func = indicator_func
        self.rank_type = rank_type
        self.count = count
        self.indicator_params = indicator_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'rank_type': self.rank_type,
            'count': self.count,
            'indicator_params': self.indicator_params
        })
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
        Args:
            symbols: List of symbols to filter
            market_data: Dictionary mapping symbols to market data
            technical_data: Optional dictionary mapping symbols to technical data
            
        Returns:
            Filtered list of symbols
        """
        if not symbols:
            logger.warning("Empty symbol list provided to RankFilter")
            return []
        
        # Calculate indicator values for each symbol
        indicator_dict = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                logger.warning(f"No market data for symbol {symbol}")
                continue
            
            # Calculate indicator values
            try:
                prices = market_data[symbol]['adjusted_close'].values
                indicator_values = self.indicator_func(prices, **self.indicator_params)
                
                # Use the last valid (non-NaN) value
                valid_indices = ~np.isnan(indicator_values)
                if np.any(valid_indices):
                    last_valid_idx = np.where(valid_indices)[0][-1]
                    indicator_dict[symbol] = indicator_values[last_valid_idx]
            except Exception as e:
                logger.error(f"Error calculating indicator for {symbol}: {str(e)}")
        
        # Rank symbols by indicator value
        if not indicator_dict:
            logger.warning("No valid indicators calculated for any symbol")
            return []
        
        # Sort symbols based on rank_type
        if self.rank_type == 'top':
            # Higher values are better
            sorted_symbols = sorted(indicator_dict.keys(), 
                                   key=lambda s: indicator_dict[s], 
                                   reverse=True)
        else:  # 'bottom'
            # Lower values are better
            sorted_symbols = sorted(indicator_dict.keys(), 
                                   key=lambda s: indicator_dict[s])
        
        # Select the top/bottom N symbols
        return sorted_symbols[:min(self.count, len(sorted_symbols))]


class PercentileFilter(FilterBase):
    """
    Filter that selects symbols based on percentile ranking by an indicator.
    """
    
    def __init__(self, name: str, indicator_func: Callable, percentile: float = 0.20, 
                rank_type: str = 'top', indicator_params: Optional[Dict] = None):
        """
        Initialize a percentile filter.
        
        Args:
            name: Filter name
            indicator_func: Function to calculate the indicator values
            percentile: Percentile threshold (0.0 to 1.0) (default: 0.20)
            rank_type: 'top' or 'bottom' (default: 'top')
            indicator_params: Additional parameters to pass to the indicator function
        """
        super().__init__(name, 'percentile')
        self.indicator_func = indicator_func
        self.percentile = percentile
        self.rank_type = rank_type
        self.indicator_params = indicator_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary representation."""
        result = super().to_dict()
        result.update({
            'percentile': self.percentile,
            'rank_type': self.rank_type,
            'indicator_params': self.indicator_params
        })
        return result
    
    def execute(self, symbols: List[str], market_data: Dict[str, pd.DataFrame], 
               technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Execute the filter on a list of symbols.
        
        Args:
            symbols: List of symbols to filter
            market_data: Dictionary mapping symbols to market data
            technical_data: Optional dictionary mapping symbols to technical data
            
        Returns:
            Filtered list of symbols
        """
        if not symbols:
            logger.warning("Empty symbol list provided to PercentileFilter")
            return []
        
        # Calculate indicator values for each symbol
        indicator_dict = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                logger.warning(f"No market data for symbol {symbol}")
                continue
            
            # Calculate indicator values
            try:
                prices = market_data[symbol]['adjusted_close'].values
                indicator_values = self.indicator_func(prices, **self.indicator_params)
                
                # Use the last valid (non-NaN) value
                valid_indices = ~np.isnan(indicator_values)
                if np.any(valid_indices):
                    last_valid_idx = np.where(valid_indices)[0][-1]
                    indicator_dict[symbol] = indicator_values[last_valid_idx]
            except Exception as e:
                logger.error(f"Error calculating indicator for {symbol}: {str(e)}")
        
        # Rank symbols by indicator value
        if not indicator_dict:
            logger.warning("No valid indicators calculated for any symbol")
            return []
        
        # Calculate number of symbols to select
        count = max(1, int(len(indicator_dict) * self.percentile))
        
        # Sort symbols based on rank_type
        if self.rank_type == 'top':
            # Higher values are better
            sorted_symbols = sorted(indicator_dict.keys(), 
                                   key=lambda s: indicator_dict[s], 
                                   reverse=True)
        else:  # 'bottom'
            # Lower values are better
            sorted_symbols = sorted(indicator_dict.keys(), 
                                   key=lambda s: indicator_dict[s])
        
        # Select the top/bottom percentile of symbols
        return sorted_symbols[:count]
