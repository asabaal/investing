"""
Trade Likelihood Estimator

A comprehensive system for calculating trading probabilities and expected returns
based on geometric Brownian motion and first passage time theory.

Main Components:
- TradeAnalyzer: Primary interface for paper trading decisions
- TradeSetup: Configuration for trade parameters
- Market data management and parameter estimation
- Probability calculations and return rate analysis

Quick Start:
    from trade_likelihood import TradeAnalyzer
    
    analyzer = TradeAnalyzer()
    analyzer.load_price_data(your_price_series)
    
    # Analyze a simple setup
    setups = analyzer.create_simple_setups(current_price=100.0)
    analysis = analyzer.analyze_trade_setup(setups['bidirectional'])
    
    print(f"Return rate: {analysis.return_rate_total:.3f}")
    print(f"Attractive setup: {analysis.is_attractive_setup()}")
"""

from .trade_analyzer import TradeAnalyzer, TradeAnalysisError
from .data_structures import (
    TradeSetup, 
    MarketParameters, 
    TradeAnalysis, 
    BacktestResult,
    create_simple_long_setup,
    create_simple_short_setup,
    create_bidirectional_setup
)
from .market_data import MarketDataManager, DataQualityError
from .parameter_estimation import (
    ParameterEstimator, 
    EstimationConfig, 
    ParameterEstimationError
)
from .probability_calculator import ProbabilityCalculator, CalculationError
from .core_math import MathematicalError

# Version info
__version__ = "1.0.0"
__author__ = "Trade Likelihood Team"
__description__ = "Trading probability estimation based on geometric Brownian motion"

# Main exports for easy access
__all__ = [
    # Main interface
    'TradeAnalyzer',
    
    # Core data structures
    'TradeSetup',
    'TradeAnalysis', 
    'MarketParameters',
    'BacktestResult',
    
    # Setup creation helpers
    'create_simple_long_setup',
    'create_simple_short_setup', 
    'create_bidirectional_setup',
    
    # Component classes (for advanced use)
    'MarketDataManager',
    'ParameterEstimator',
    'ProbabilityCalculator',
    'EstimationConfig',
    
    # Exceptions
    'TradeAnalysisError',
    'DataQualityError',
    'ParameterEstimationError', 
    'CalculationError',
    'MathematicalError',
    
    # Metadata
    '__version__'
]


def quick_analysis(prices, current_price, entry_pct=0.02, stop_pct=0.01, target_pct=0.03):
    """
    Quick analysis function for immediate trade likelihood assessment.
    
    Perfect for paper trading decisions!
    
    Parameters:
    -----------
    prices : list or pd.Series
        Historical price data (at least 30+ points recommended)
    current_price : float
        Current asset price
    entry_pct : float
        Entry distance as % of current price (default 2%)
    stop_pct : float  
        Stop distance as % of entry price (default 1%)
    target_pct : float
        Target distance as % of entry price (default 3%)
        
    Returns:
    --------
    dict
        Quick analysis summary with key metrics
        
    Example:
    --------
    >>> prices = [98, 99, 100, 101, 99.5, 100.2, ...]  # Your price history
    >>> result = quick_analysis(prices, current_price=100.0)
    >>> print(f"Best direction: {result['best_direction']}")
    >>> print(f"Return rate: {result['return_rate']:.3f} per day")
    >>> print(f"Attractive setup: {result['is_attractive']}")
    """
    try:
        # Initialize analyzer
        analyzer = TradeAnalyzer()
        
        # Load data
        analyzer.load_price_data(prices)
        
        # Create bidirectional setup
        setup = create_bidirectional_setup(
            current_price=current_price,
            entry_pct=entry_pct,
            stop_pct=stop_pct, 
            target_pct=target_pct
        )
        
        # Analyze
        analysis = analyzer.analyze_trade_setup(setup)
        
        # Return summary
        return {
            'setup_id': analysis.setup_id,
            'current_price': current_price,
            'best_direction': analysis.get_best_direction(),
            'return_rate': analysis.return_rate_total,
            'expected_value': analysis.expected_value_total,
            'combined_entry_prob': analysis.get_combined_entry_probability(),
            'is_attractive': analysis.is_attractive_setup(),
            'market_regime': analysis.market_params.regime,
            'volatility': f"{analysis.market_params.volatility:.1%}",
            'drift': f"{analysis.market_params.drift:.1%}",
            'confidence': {
                'drift': analysis.market_params.drift_confidence,
                'volatility': analysis.market_params.volatility_confidence
            },
            'warnings': analysis.warnings[:3],  # Top 3 warnings
            'recommendation': _get_recommendation(analysis)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'recommendation': 'Unable to analyze - check data quality and try again'
        }


def _get_recommendation(analysis):
    """Generate trading recommendation based on analysis"""
    
    if analysis.return_rate_total is None or analysis.return_rate_total <= 0:
        return "AVOID - Negative expected return"
    
    if analysis.return_rate_total > 0.1:  # >10% per day
        return "STRONG BUY - Very attractive setup"
    elif analysis.return_rate_total > 0.05:  # >5% per day  
        return "BUY - Good setup"
    elif analysis.return_rate_total > 0.01:  # >1% per day
        return "WEAK BUY - Modest opportunity"
    else:
        return "NEUTRAL - Low expected return"


# Convenience imports for common workflows
def create_analyzer(**kwargs):
    """Create a TradeAnalyzer with optional configuration"""
    return TradeAnalyzer(**kwargs)


# Configuration helpers
def get_conservative_config():
    """Get configuration for conservative parameter estimation"""
    return EstimationConfig(
        volatility_lambda=0.97,  # More weight on historical data
        drift_alpha=0.05,        # Less weight on recent returns  
        min_periods=50,          # More data required
        confidence_threshold=0.8  # Higher confidence threshold
    )


def get_aggressive_config():
    """Get configuration for aggressive parameter estimation"""
    return EstimationConfig(
        volatility_lambda=0.90,  # More weight on recent data
        drift_alpha=0.15,        # More weight on recent returns
        min_periods=20,          # Less data required  
        confidence_threshold=0.6  # Lower confidence threshold
    )