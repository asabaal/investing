import warnings

import numpy as np
import pandas as pd
from typing import Callable, Optional, Union

warnings.filterwarnings('ignore')

def rolling_growing_window(
    data: Union[pd.Series, np.ndarray],
    window: int,
    func: Callable,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Apply a rolling window function with growing window size for initial periods.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute the rolling function on
        window (int): Maximum size of the rolling window
        func (Callable): Function to apply to each window (e.g., np.mean, np.std)
        min_periods (Optional[int], optional): Minimum number of observations required
            for calculation. If None, defaults to 1.

    Returns:
        np.ndarray: Array of same length as input with rolling calculation results

    Example:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> rolling_growing_window(data, window=3, func=np.mean)
        array([1., 1.5, 2., 3., 4.])

        >>> # Custom function example
        >>> def weighted_mean(x):
        ...     weights = np.arange(1, len(x) + 1)
        ...     return np.average(x, weights=weights)
        >>> rolling_growing_window(data, window=3, func=weighted_mean)

    Notes:
        The function handles the initial periods (i < window) by using all available
        data up to that point, creating a growing window effect. After reaching the
        window size, it uses a standard rolling window.
    """
    # Convert input to numpy array if it's a Series
    if isinstance(data, pd.Series):
        data = data.values
    
    # Set default min_periods
    if min_periods is None:
        min_periods = 1
    
    # Initialize output array
    result = np.zeros(len(data))
    
    # Handle each position
    for i in range(len(data)):
        if i < window:
            # Growing window phase
            if i + 1 >= min_periods:
                result[i] = func(data[:i+1])
            else:
                result[i] = np.nan
        else:
            # Standard rolling window
            result[i] = func(data[i-window+1:i+1])
    
    return result

# Example specialized functions using the growing window
def growing_mean(
    data: Union[pd.Series, np.ndarray],
    window: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate rolling mean with growing initial window.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute rolling mean
        window (int): Maximum size of the rolling window
        min_periods (Optional[int], optional): Minimum observations required. 
            Defaults to None.

    Returns:
        np.ndarray: Array of rolling means with growing window for initial periods
    """
    return rolling_growing_window(data, window, np.mean, min_periods)

def growing_std(
    data: Union[pd.Series, np.ndarray],
    window: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate rolling standard deviation with growing initial window.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute rolling standard deviation
        window (int): Maximum size of the rolling window
        min_periods (Optional[int], optional): Minimum observations required. 
            Defaults to None.

    Returns:
        np.ndarray: Array of rolling standard deviations with growing window for initial periods
    """
    return rolling_growing_window(data, window, np.std, min_periods)

def growing_sum(
    data: Union[pd.Series, np.ndarray],
    window: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate rolling sum with growing initial window.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute rolling sum
        window (int): Maximum size of the rolling window
        min_periods (Optional[int], optional): Minimum observations required. 
            Defaults to None.

    Returns:
        np.ndarray: Array of rolling sums with growing window for initial periods
    """
    return rolling_growing_window(data, window, np.sum, min_periods)

def growing_max(
    data: Union[pd.Series, np.ndarray],
    window: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate rolling max with growing initial window.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute rolling max
        window (int): Maximum size of the rolling window
        min_periods (Optional[int], optional): Minimum observations required. 
            Defaults to None.

    Returns:
        np.ndarray: Array of rolling maxs with growing window for initial periods
    """
    return rolling_growing_window(data, window, np.max, min_periods)

def growing_min(
    data: Union[pd.Series, np.ndarray],
    window: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate rolling min with growing initial window.

    Args:
        data (Union[pd.Series, np.ndarray]): Input data to compute rolling min
        window (int): Maximum size of the rolling window
        min_periods (Optional[int], optional): Minimum observations required. 
            Defaults to None.

    Returns:
        np.ndarray: Array of rolling mins with growing window for initial periods
    """
    return rolling_growing_window(data, window, np.min, min_periods)        