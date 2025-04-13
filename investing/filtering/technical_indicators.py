"""
Technical Indicators Module

This module provides functions to calculate various technical indicators used in
financial analysis and trading strategies. Each function is designed to perform
a single, specific calculation with limited complexity.

The functions in this module serve as the foundation for the filtering system
in the Symphony trading platform.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.
    
    Args:
        prices: Price series (usually closing prices)
        period: RSI calculation period (default: 14)
        
    Returns:
        Array of RSI values (same length as input, with the first period values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calculate price changes
    price_diff = np.diff(prices, prepend=prices[0])
    
    # Separate gains and losses
    gains = np.where(price_diff > 0, price_diff, 0)
    losses = np.where(price_diff < 0, -price_diff, 0)
    
    # Initialize arrays for average gains and losses
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    # Calculate first average gain and loss
    avg_gains[period] = np.mean(gains[1:period+1])
    avg_losses[period] = np.mean(losses[1:period+1])
    
    # Calculate subsequent average gains and losses
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Set NaN for the first period values
    rsi[:period] = np.nan
    
    return rsi


def calculate_sma(prices: Union[pd.Series, np.ndarray], period: int = 20) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA) for a series of prices.
    
    Args:
        prices: Price series
        period: SMA period (default: 20)
        
    Returns:
        Array of SMA values (same length as input, with the first period-1 values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Initialize the output array
    sma = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate SMA for each window
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    
    return sma


def calculate_ema(prices: Union[pd.Series, np.ndarray], period: int = 20, 
                 smoothing: float = 2.0) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) for a series of prices.
    
    Args:
        prices: Price series
        period: EMA period (default: 20)
        smoothing: Smoothing factor (default: 2.0)
        
    Returns:
        Array of EMA values (same length as input, with the first period-1 values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Initialize the output array
    ema = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate the multiplier
    multiplier = smoothing / (period + 1)
    
    # Use SMA as the initial EMA value
    ema[period-1] = np.mean(prices[:period])
    
    # Calculate EMA for the rest of the series
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_momentum(prices: Union[pd.Series, np.ndarray], period: int = 30) -> np.ndarray:
    """
    Calculate price momentum over a period.
    
    Momentum is calculated as the percentage change over the specified period.
    
    Args:
        prices: Price series
        period: Momentum period (default: 30)
        
    Returns:
        Array of momentum values (same length as input, with the first period values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Initialize the output array
    momentum = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate momentum for each point
    for i in range(period, len(prices)):
        momentum[i] = (prices[i] - prices[i - period]) / prices[i - period]
    
    return momentum


def calculate_volatility(prices: Union[pd.Series, np.ndarray], period: int = 63) -> np.ndarray:
    """
    Calculate price volatility over a period.
    
    Volatility is calculated as the standard deviation of returns over the specified period.
    
    Args:
        prices: Price series
        period: Volatility period (default: 63)
        
    Returns:
        Array of volatility values (same length as input, with the first period values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)  # Add a 0 at the beginning to maintain length
    
    # Initialize the output array
    volatility = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate volatility for each window
    for i in range(period, len(returns)):
        volatility[i] = np.std(returns[i-period+1:i+1])
    
    return volatility


def calculate_drawdown(prices: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate drawdown and maximum drawdown for a series of prices.
    
    Drawdown measures the decline from a peak in the price series.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (drawdown, max_drawdown) arrays
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Initialize arrays
    drawdown = np.zeros_like(prices, dtype=float)
    max_drawdown = np.zeros_like(prices, dtype=float)
    peak = prices[0]
    
    # Calculate drawdown at each point
    for i in range(len(prices)):
        if prices[i] > peak:
            peak = prices[i]
        
        if peak != 0:  # Avoid division by zero
            drawdown[i] = (peak - prices[i]) / peak
        
        # Update max drawdown
        max_drawdown[i] = np.max(drawdown[:i+1])
    
    return drawdown, max_drawdown


def calculate_returns(prices: Union[pd.Series, np.ndarray], period: int = 30) -> np.ndarray:
    """
    Calculate returns over a specified period.
    
    Args:
        prices: Price series
        period: Return calculation period (default: 30)
        
    Returns:
        Array of return values (same length as input, with the first period values as NaN)
    """
    # Convert input to numpy array if it's a pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Initialize the output array
    returns = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate returns for each point
    for i in range(period, len(prices)):
        returns[i] = (prices[i] - prices[i - period]) / prices[i - period]
    
    return returns
