"""
Filtering System Module

This package provides a comprehensive filtering system for Symphony trading strategies.
It includes technical indicators, evaluation functions, filter base classes, and
specific filter implementations for common trading indicators.
"""

from investing.filtering.technical_indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_momentum,
    calculate_volatility,
    calculate_drawdown,
    calculate_returns
)

from investing.filtering.evaluation_functions import (
    evaluate_threshold,
    evaluate_range,
    evaluate_crossover,
    evaluate_lookback,
    evaluate_persistence
)

'''
from investing.filtering.filter_base import (
    FilterBase,
    ThresholdFilter,
    RangeFilter,
    CrossoverFilter,
    RankFilter,
    PercentileFilter
)

from investing.filtering.filter_types import (
    RSIFilter,
    MovingAverageFilter,
    MACrossoverFilter,
    MomentumFilter,
    VolatilityFilter,
    DrawdownFilter,
    ReturnFilter
)
'''
__all__ = [
    # Technical indicators
    'calculate_rsi',
    'calculate_sma',
    'calculate_ema',
    'calculate_momentum',
    'calculate_volatility',
    'calculate_drawdown',
    'calculate_returns',
    
    # Evaluation functions
    'evaluate_threshold',
    'evaluate_range',
    'evaluate_crossover',
    'evaluate_lookback',
    'evaluate_persistence',
    
    # Filter base classes
    'FilterBase',
    'ThresholdFilter',
    'RangeFilter',
    'CrossoverFilter',
    'RankFilter',
    'PercentileFilter',
    
    # Specific filter types
    'RSIFilter',
    'MovingAverageFilter',
    'MACrossoverFilter',
    'MomentumFilter',
    'VolatilityFilter',
    'DrawdownFilter',
    'ReturnFilter'
]
