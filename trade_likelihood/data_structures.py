"""
Data structures for the trading probability model.

Core classes for trade setups, market parameters, and analysis results.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np


@dataclass
class TradeSetup:
    """Configuration for a single trade setup (long, short, or bidirectional)"""
    current_price: float
    entry_long: Optional[float] = None
    entry_short: Optional[float] = None
    stop_long: Optional[float] = None
    stop_short: Optional[float] = None
    target_long: Optional[float] = None
    target_short: Optional[float] = None
    max_time_window: float = 1.0  # in days
    
    def __post_init__(self):
        """Validate trade setup parameters after initialization"""
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate trade setup parameters"""
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
        
        if self.max_time_window <= 0:
            raise ValueError("Max time window must be positive")
        
        # Validate long setup if specified
        if self.entry_long is not None:
            # Allow both breakout (entry > current) and pullback (entry < current) strategies
            # The key validation is the order: target > entry > stop
            
            if self.stop_long is not None and self.stop_long >= self.entry_long:
                raise ValueError("Long stop must be below long entry")
            
            if self.target_long is not None and self.target_long <= self.entry_long:
                raise ValueError("Long target must be above long entry")
        
        # Validate short setup if specified
        if self.entry_short is not None:
            # Short entries can be above OR below current price depending on strategy
            # Above current = short breakout (sell-stop), Below current = short pullback (limit)
            
            if self.stop_short is not None and self.stop_short <= self.entry_short:
                raise ValueError("Short stop must be above short entry")
            
            if self.target_short is not None and self.target_short >= self.entry_short:
                raise ValueError("Short target must be below short entry")
    
    def has_long_setup(self) -> bool:
        """Check if long trade setup is complete"""
        return all([
            self.entry_long is not None,
            self.stop_long is not None,
            self.target_long is not None
        ])
    
    def has_short_setup(self) -> bool:
        """Check if short trade setup is complete"""
        return all([
            self.entry_short is not None,
            self.stop_short is not None,
            self.target_short is not None
        ])
    
    def get_long_risk_reward(self) -> Optional[float]:
        """Calculate risk-reward ratio for long setup"""
        if not self.has_long_setup():
            return None
        reward = self.target_long - self.entry_long
        risk = self.entry_long - self.stop_long
        return reward / risk if risk > 0 else None
    
    def get_short_risk_reward(self) -> Optional[float]:
        """Calculate risk-reward ratio for short setup"""
        if not self.has_short_setup():
            return None
        reward = self.entry_short - self.target_short
        risk = self.stop_short - self.entry_short
        return reward / risk if risk > 0 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'current_price': self.current_price,
            'entry_long': self.entry_long,
            'entry_short': self.entry_short,
            'stop_long': self.stop_long,
            'stop_short': self.stop_short,
            'target_long': self.target_long,
            'target_short': self.target_short,
            'max_time_window': self.max_time_window,
            'long_risk_reward': self.get_long_risk_reward(),
            'short_risk_reward': self.get_short_risk_reward()
        }


@dataclass
class MarketParameters:
    """Current market parameter estimates with confidence metrics"""
    drift: float
    volatility: float
    drift_confidence: float
    volatility_confidence: float
    regime: str  # 'low_vol', 'high_vol', 'trending', 'ranging'
    last_updated: pd.Timestamp
    estimation_window: int  # number of periods used
    
    def __post_init__(self):
        """Validate market parameters"""
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        
        if not 0 <= self.drift_confidence <= 1:
            raise ValueError("Drift confidence must be between 0 and 1")
        
        if not 0 <= self.volatility_confidence <= 1:
            raise ValueError("Volatility confidence must be between 0 and 1")
        
        valid_regimes = ['low_vol', 'high_vol', 'trending', 'ranging', 'unknown']
        if self.regime not in valid_regimes:
            raise ValueError(f"Regime must be one of: {valid_regimes}")
        
        if self.estimation_window <= 0:
            raise ValueError("Estimation window must be positive")
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if parameters have high confidence"""
        return (self.drift_confidence >= threshold and 
                self.volatility_confidence >= threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'drift': self.drift,
            'volatility': self.volatility,
            'drift_confidence': self.drift_confidence,
            'volatility_confidence': self.volatility_confidence,
            'regime': self.regime,
            'last_updated': self.last_updated.isoformat(),
            'estimation_window': self.estimation_window,
            'high_confidence': self.is_high_confidence()
        }


@dataclass
class TradeAnalysis:
    """Complete analysis results for a trade setup"""
    
    # Basic setup info
    setup_id: str
    timestamp: pd.Timestamp
    current_price: float
    
    # Market parameters used
    market_params: MarketParameters
    
    # Entry probabilities
    prob_entry_long: Optional[float] = None
    prob_entry_short: Optional[float] = None
    
    # Win probabilities given entry
    prob_win_long: Optional[float] = None
    prob_win_short: Optional[float] = None
    
    # Expected times (in days)
    expected_entry_time_long: Optional[float] = None
    expected_entry_time_short: Optional[float] = None
    expected_trade_duration_long: Optional[float] = None
    expected_trade_duration_short: Optional[float] = None
    
    # Expected values (risk-adjusted returns)
    expected_value_long: Optional[float] = None
    expected_value_short: Optional[float] = None
    expected_value_total: Optional[float] = None
    
    # Return rates (key metrics) - expected value per unit time
    return_rate_long: Optional[float] = None
    return_rate_short: Optional[float] = None
    return_rate_total: Optional[float] = None
    
    # Risk metrics
    risk_reward_long: Optional[float] = None
    risk_reward_short: Optional[float] = None
    correlation_risk: Optional[float] = None
    
    # Confidence intervals (95% by default)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Additional metadata
    calculation_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_best_direction(self) -> Optional[str]:
        """Get the direction with highest return rate"""
        rates = {}
        
        if self.return_rate_long is not None:
            rates['long'] = self.return_rate_long
            
        if self.return_rate_short is not None:
            rates['short'] = self.return_rate_short
        
        if not rates:
            return None
        
        return max(rates, key=rates.get)
    
    def get_combined_entry_probability(self) -> Optional[float]:
        """Get probability of at least one entry occurring"""
        if self.prob_entry_long is None and self.prob_entry_short is None:
            return None
        
        prob_long = self.prob_entry_long or 0.0
        prob_short = self.prob_entry_short or 0.0
        
        # P(A or B) = P(A) + P(B) - P(A and B)
        # Assuming independence for now (can be refined)
        return prob_long + prob_short - (prob_long * prob_short)
    
    def is_attractive_setup(self, min_return_rate: float = 0.1, 
                          min_entry_prob: float = 0.3) -> bool:
        """Determine if this is an attractive trading setup"""
        # Check return rate threshold
        if self.return_rate_total is None:
            return False
        
        if self.return_rate_total < min_return_rate:
            return False
        
        # Check entry probability threshold
        combined_prob = self.get_combined_entry_probability()
        if combined_prob is None or combined_prob < min_entry_prob:
            return False
        
        return True
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get key summary statistics"""
        return {
            'setup_id': self.setup_id,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'best_direction': self.get_best_direction(),
            'combined_entry_prob': self.get_combined_entry_probability(),
            'total_return_rate': self.return_rate_total,
            'total_expected_value': self.expected_value_total,
            'is_attractive': self.is_attractive_setup(),
            'market_regime': self.market_params.regime,
            'parameter_confidence': self.market_params.is_high_confidence()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'setup_id': self.setup_id,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'market_params': self.market_params.to_dict(),
            
            # Probabilities
            'prob_entry_long': self.prob_entry_long,
            'prob_entry_short': self.prob_entry_short,
            'prob_win_long': self.prob_win_long,
            'prob_win_short': self.prob_win_short,
            
            # Times
            'expected_entry_time_long': self.expected_entry_time_long,
            'expected_entry_time_short': self.expected_entry_time_short,
            'expected_trade_duration_long': self.expected_trade_duration_long,
            'expected_trade_duration_short': self.expected_trade_duration_short,
            
            # Values and rates
            'expected_value_long': self.expected_value_long,
            'expected_value_short': self.expected_value_short,
            'expected_value_total': self.expected_value_total,
            'return_rate_long': self.return_rate_long,
            'return_rate_short': self.return_rate_short,
            'return_rate_total': self.return_rate_total,
            
            # Risk metrics
            'risk_reward_long': self.risk_reward_long,
            'risk_reward_short': self.risk_reward_short,
            'correlation_risk': self.correlation_risk,
            
            # Confidence intervals
            'confidence_intervals': self.confidence_intervals,
            
            # Summary
            'summary': self.get_summary_stats(),
            
            # Metadata
            'calculation_notes': self.calculation_notes,
            'warnings': self.warnings
        }
        
        return result


@dataclass
class BacktestResult:
    """Results from backtesting the trading strategy"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    successful_trades: int
    
    # Detailed trade history
    trade_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Performance by regime
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'total_return': f"{self.total_return:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'volatility': f"{self.volatility:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'avg_trade_duration': f"{self.avg_trade_duration:.1f} days",
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'regime_performance': self.regime_performance
        }


# Utility functions for data structure operations

def create_simple_long_setup(current_price: float, entry_pct: float = 0.02,
                            stop_pct: float = 0.01, target_pct: float = 0.03,
                            max_time_days: float = 1.0) -> TradeSetup:
    """Create a simple long trade setup with percentage-based levels"""
    return TradeSetup(
        current_price=current_price,
        entry_long=current_price * (1 + entry_pct),
        stop_long=current_price * (1 + entry_pct) * (1 - stop_pct),
        target_long=current_price * (1 + entry_pct) * (1 + target_pct),
        max_time_window=max_time_days
    )


def create_simple_short_setup(current_price: float, entry_pct: float = 0.02,
                             stop_pct: float = 0.01, target_pct: float = 0.03,
                             max_time_days: float = 1.0) -> TradeSetup:
    """Create a simple short trade setup with percentage-based levels"""
    return TradeSetup(
        current_price=current_price,
        entry_short=current_price * (1 - entry_pct),
        stop_short=current_price * (1 - entry_pct) * (1 + stop_pct),
        target_short=current_price * (1 - entry_pct) * (1 - target_pct),
        max_time_window=max_time_days
    )


def create_bidirectional_setup(current_price: float, entry_pct: float = 0.02,
                              stop_pct: float = 0.01, target_pct: float = 0.03,
                              max_time_days: float = 1.0) -> TradeSetup:
    """Create a bidirectional trade setup with symmetric levels"""
    return TradeSetup(
        current_price=current_price,
        entry_long=current_price * (1 + entry_pct),
        entry_short=current_price * (1 - entry_pct),
        stop_long=current_price * (1 + entry_pct) * (1 - stop_pct),
        stop_short=current_price * (1 - entry_pct) * (1 + stop_pct),
        target_long=current_price * (1 + entry_pct) * (1 + target_pct),
        target_short=current_price * (1 - entry_pct) * (1 - target_pct),
        max_time_window=max_time_days
    )