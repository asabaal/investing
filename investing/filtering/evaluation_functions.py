"""
Evaluation Functions Module

This module provides functions for evaluating technical indicators against criteria.
These functions form the second layer of the filtering system architecture and are
used by filter classes to determine whether symbols meet specific conditions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Callable, List, Tuple, Dict


def evaluate_threshold(values: Union[pd.Series, np.ndarray], 
                      threshold: float, 
                      condition: str = 'above',
                      method: str = 'last',
                      lookback: int = 1) -> Union[bool, np.ndarray]:
    """
    Evaluate whether values meet a threshold condition.
    
    Args:
        values: Series of values to evaluate
        threshold: Threshold value for comparison
        condition: 'above' or 'below' (default: 'above')
        method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
        lookback: Number of periods to look back (default: 1)
        
    Returns:
        Boolean result of evaluation or boolean array if input is a Series/array
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(values, pd.Series):
        values = values.values
    
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
    
    else:
        raise ValueError(f"Unknown evaluation method: {method}")


def evaluate_range(values: Union[pd.Series, np.ndarray],
                  lower_bound: float,
                  upper_bound: float,
                  condition: str = 'inside',
                  method: str = 'last',
                  lookback: int = 1) -> Union[bool, np.ndarray]:
    """
    Evaluate whether values fall within a specified range.
    
    Args:
        values: Series of values to evaluate
        lower_bound: Lower bound of the range
        upper_bound: Upper bound of the range
        condition: 'inside' or 'outside' (default: 'inside')
        method: Evaluation method - 'last', 'any', 'all', or 'average' (default: 'last')
        lookback: Number of periods to look back (default: 1)
        
    Returns:
        Boolean result of evaluation or boolean array if input is a Series/array
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(values, pd.Series):
        values = values.values
    
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
        value = lookback_values[-1]
        if condition == 'inside':
            return lower_bound <= value <= upper_bound
        else:  # 'outside'
            return value < lower_bound or value > upper_bound
            
    elif method == 'any':
        # Return True if any value meets the condition
        if condition == 'inside':
            return np.any((lookback_values >= lower_bound) & (lookback_values <= upper_bound))
        else:  # 'outside'
            return np.any((lookback_values < lower_bound) | (lookback_values > upper_bound))
            
    elif method == 'all':
        # Return True if all values meet the condition
        if condition == 'inside':
            return np.all((lookback_values >= lower_bound) & (lookback_values <= upper_bound))
        else:  # 'outside'
            return np.all((lookback_values < lower_bound) | (lookback_values > upper_bound))
            
    elif method == 'average':
        # Evaluate the average of values
        avg_value = np.mean(lookback_values)
        if condition == 'inside':
            return lower_bound <= avg_value <= upper_bound
        else:  # 'outside'
            return avg_value < lower_bound or avg_value > upper_bound
    
    else:
        raise ValueError(f"Unknown evaluation method: {method}")


def evaluate_crossover(fast_values: Union[pd.Series, np.ndarray],
                      slow_values: Union[pd.Series, np.ndarray],
                      condition: str = 'above',
                      lookback: int = 1) -> bool:
    """
    Evaluate whether one series crosses another.
    
    Args:
        fast_values: Faster-moving indicator series
        slow_values: Slower-moving indicator series
        condition: 'above' (fast crosses above slow) or 'below' (fast crosses below slow)
                  (default: 'above')
        lookback: Number of periods to look back for crossover (default: 1)
        
    Returns:
        Boolean indicating if a crossover occurred
    """
    # Convert inputs to numpy arrays if they are pandas Series
    if isinstance(fast_values, pd.Series):
        fast_values = fast_values.values
    if isinstance(slow_values, pd.Series):
        slow_values = slow_values.values
    
    # Ensure series are the same length
    if len(fast_values) != len(slow_values):
        raise ValueError("Fast and slow value series must be the same length")
    
    # Need at least 2 points to check for a crossover
    if len(fast_values) < 2:
        return False
    
    # Handle lookback - only look at the last 'lookback' periods
    if lookback > len(fast_values) - 1:
        lookback = len(fast_values) - 1
    
    # Get only the last 'lookback+1' elements to check for crossovers
    # We need lookback+1 elements to check for 'lookback' transitions
    fast_window = fast_values[-(lookback+1):]
    slow_window = slow_values[-(lookback+1):]
    
    # Loop through the window to check for crossovers
    for i in range(1, len(fast_window)):
        # Skip if any value is NaN
        if (np.isnan(fast_window[i]) or np.isnan(slow_window[i]) or 
            np.isnan(fast_window[i-1]) or np.isnan(slow_window[i-1])):
            continue
        
        if condition == 'above':
            # Check for fast crossing above slow
            if fast_window[i-1] <= slow_window[i-1] and fast_window[i] > slow_window[i]:
                return True
        else:  # 'below'
            # Check for fast crossing below slow
            if fast_window[i-1] >= slow_window[i-1] and fast_window[i] < slow_window[i]:
                return True
    
    return False


def evaluate_lookback(values: Union[pd.Series, np.ndarray],
                     evaluation_func: Callable,
                     lookback_days: int = 1,
                     **func_args) -> Union[bool, int, float, List]:
    """
    Apply a custom evaluation function to a series of values over a lookback period.
    
    Args:
        values: Series of values to evaluate
        evaluation_func: Function to apply to values
        lookback_days: Number of days to look back
        **func_args: Additional arguments to pass to the evaluation function
        
    Returns:
        Result of applying the evaluation function to values in the lookback period
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(values, pd.Series):
        values = values.values
    
    # Get the values to evaluate based on lookback
    if len(values) <= lookback_days:
        lookback_days = len(values)
    
    lookback_values = values[-lookback_days:]
    
    # Apply the evaluation function
    return evaluation_func(lookback_values, **func_args)


def evaluate_persistence(boolean_series: Union[pd.Series, np.ndarray, List[bool]],
                        days_required: int,
                        condition: str = 'consecutive') -> bool:
    """
    Evaluate whether a condition persists for a specified number of days.
    
    Args:
        boolean_series: Series of boolean values indicating whether a condition is met
        days_required: Number of days the condition must be met
        condition: 'consecutive' or 'total' (default: 'consecutive')
        
    Returns:
        Boolean indicating if the persistence condition is met
    """
    # Convert input to numpy array if it's a pandas Series or list
    if isinstance(boolean_series, (pd.Series, list)):
        boolean_series = np.array(boolean_series)
    
    # Handle empty series
    if len(boolean_series) == 0:
        return False
    
    # Ensure the series contains boolean values
    if boolean_series.dtype != bool:
        boolean_series = boolean_series.astype(bool)
    
    if condition == 'consecutive':
        # Check for consecutive True values
        if days_required > len(boolean_series):
            return False
        
        # Find longest streak of consecutive True values
        # Convert to integers for easier computation
        int_series = boolean_series.astype(int)
        
        # Find lengths of all consecutive True sequences
        # First, identify where transitions occur
        transitions = np.diff(np.hstack(([0], int_series, [0])))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        # Calculate lengths of True sequences
        lengths = ends - starts
        
        # Check if any sequence meets the required length
        return np.any(lengths >= days_required)
        
    elif condition == 'total':
        # Check if the total number of True values meets the requirement
        return np.sum(boolean_series) >= days_required
    
    else:
        raise ValueError(f"Unknown persistence condition: {condition}")
