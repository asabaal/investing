"""
Parameter estimation for trading probability calculations.

Implements EWMA volatility estimation, adaptive drift estimation, 
and regime detection for market parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings
from dataclasses import dataclass

from .data_structures import MarketParameters


class ParameterEstimationError(Exception):
    """Exception for parameter estimation errors"""
    pass


@dataclass
class EstimationConfig:
    """Configuration for parameter estimation"""
    volatility_lambda: float = 0.94  # EWMA decay factor
    drift_alpha: float = 0.1  # Adaptive drift decay
    min_periods: int = 30  # Minimum data points
    confidence_threshold: float = 0.7  # Minimum confidence for reliability
    regime_window: int = 50  # Window for regime detection
    vol_threshold_low: float = 0.15  # Annual volatility threshold for low vol
    vol_threshold_high: float = 0.35  # Annual volatility threshold for high vol


class ParameterEstimator:
    """Estimates and updates market parameters dynamically"""
    
    def __init__(self, config: Optional[EstimationConfig] = None):
        """
        Initialize parameter estimator.
        
        Parameters:
        -----------
        config : EstimationConfig, optional
            Configuration parameters (uses defaults if None)
        """
        self.config = config or EstimationConfig()
        self.current_params: Optional[MarketParameters] = None
        self.estimation_history: List[MarketParameters] = []
        
        # Internal state for EWMA calculations
        self._ewma_var: Optional[float] = None
        self._drift_ewma: Optional[float] = None
        self._last_returns: Optional[pd.Series] = None
    
    def update_parameters(self, returns: pd.Series) -> MarketParameters:
        """
        Update market parameters with new return data.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of log returns
            
        Returns:
        --------
        MarketParameters
            Updated parameter estimates
            
        Raises:
        -------
        ParameterEstimationError
            If estimation fails or data is insufficient
        """
        if len(returns) < self.config.min_periods:
            raise ParameterEstimationError(
                f"Insufficient data: {len(returns)} < {self.config.min_periods}"
            )
        
        try:
            # Estimate volatility using EWMA
            volatility, vol_confidence = self.estimate_volatility_ewma(returns)
            
            # Estimate drift using adaptive method
            drift, drift_confidence = self.estimate_drift_adaptive(returns)
            
            # Detect current market regime
            regime = self.detect_regime(returns)
            
            # Create parameter object
            params = MarketParameters(
                drift=drift,
                volatility=volatility,
                drift_confidence=drift_confidence,
                volatility_confidence=vol_confidence,
                regime=regime,
                last_updated=pd.Timestamp.now(),
                estimation_window=len(returns)
            )
            
            # Update internal state
            self.current_params = params
            self.estimation_history.append(params)
            self._last_returns = returns.copy()
            
            return params
            
        except Exception as e:
            raise ParameterEstimationError(f"Parameter estimation failed: {e}")
    
    def estimate_volatility_ewma(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate EWMA volatility estimate with confidence score.
        
        Uses exponentially weighted moving average for volatility estimation:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of log returns
            
        Returns:
        --------
        Tuple[float, float]
            (annualized_volatility, confidence_score)
        """
        if len(returns) < 2:
            raise ParameterEstimationError("Need at least 2 returns for volatility estimation")
        
        lambda_decay = self.config.volatility_lambda
        
        try:
            # Convert to numpy for faster computation
            returns_array = returns.dropna().values
            n = len(returns_array)
            
            if n < self.config.min_periods:
                warnings.warn(f"Using {n} periods for volatility (< {self.config.min_periods})")
            
            # Initialize EWMA variance
            if self._ewma_var is None or len(returns) != len(self._last_returns):
                # Cold start: use sample variance for first estimate
                self._ewma_var = np.var(returns_array, ddof=1)
            
            # Update EWMA variance iteratively
            squared_returns = returns_array ** 2
            
            # For efficiency, calculate EWMA in batch
            ewma_var = self._ewma_var
            for r_squared in squared_returns[-min(100, len(squared_returns)):]:  # Last 100 points
                ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * r_squared
            
            self._ewma_var = ewma_var
            
            # Convert to annualized volatility (assuming daily returns)
            # For other frequencies, adjust the scaling factor
            annualized_vol = np.sqrt(ewma_var * 252)  # 252 trading days per year
            
            # Calculate confidence based on data quantity and stability
            confidence = self._calculate_volatility_confidence(returns, annualized_vol)
            
            return max(0.01, annualized_vol), confidence  # Minimum vol of 1%
            
        except Exception as e:
            raise ParameterEstimationError(f"EWMA volatility calculation failed: {e}")
    
    def estimate_drift_adaptive(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate adaptive drift estimate with exponential weighting.
        
        Uses exponentially weighted average of recent returns:
        μ_t = Σ w_i * r_{t-i}, where w_i = exp(-α*i) / Σ exp(-α*j)
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of log returns
            
        Returns:
        --------
        Tuple[float, float]
            (annualized_drift, confidence_score)
        """
        if len(returns) < 2:
            raise ParameterEstimationError("Need at least 2 returns for drift estimation")
        
        alpha = self.config.drift_alpha
        
        try:
            returns_array = returns.dropna().values
            n = len(returns_array)
            
            # Create exponential weights (most recent gets highest weight)
            weights = np.exp(-alpha * np.arange(n))
            weights = weights / weights.sum()  # Normalize to sum to 1
            
            # Calculate weighted average (reverse order so newest is first)
            weighted_mean = np.dot(weights[::-1], returns_array)
            
            # Annualize the drift (assuming daily returns)
            annualized_drift = weighted_mean * 252
            
            # Calculate confidence based on consistency and sample size
            confidence = self._calculate_drift_confidence(returns, weighted_mean)
            
            return annualized_drift, confidence
            
        except Exception as e:
            raise ParameterEstimationError(f"Adaptive drift calculation failed: {e}")
    
    def detect_regime(self, returns: pd.Series) -> str:
        """
        Detect current market regime based on volatility and trend characteristics.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of log returns
            
        Returns:
        --------
        str
            Regime classification: 'low_vol', 'high_vol', 'trending', 'ranging'
        """
        if len(returns) < self.config.regime_window:
            return 'unknown'
        
        try:
            # Use recent window for regime detection
            recent_returns = returns.iloc[-self.config.regime_window:]
            
            # Calculate rolling volatility
            rolling_vol = recent_returns.rolling(window=min(20, len(recent_returns))).std()
            current_vol = rolling_vol.iloc[-1] * np.sqrt(252)  # Annualize
            
            # Volatility-based regime classification
            if current_vol < self.config.vol_threshold_low:
                vol_regime = 'low_vol'
            elif current_vol > self.config.vol_threshold_high:
                vol_regime = 'high_vol'
            else:
                vol_regime = 'normal_vol'
            
            # Trend detection using linear regression
            x = np.arange(len(recent_returns))
            slope, _, r_value, _, _ = stats.linregress(x, recent_returns.cumsum())
            
            # Strong trend if R² > 0.3 and significant slope
            trend_strength = r_value ** 2
            is_trending = trend_strength > 0.3 and abs(slope) > 0.001
            
            # Combine volatility and trend information
            if is_trending:
                return 'trending'
            elif vol_regime == 'low_vol':
                return 'ranging'  # Low vol + no trend = ranging
            elif vol_regime == 'high_vol':
                return 'high_vol'
            else:
                return 'ranging'  # Default for normal volatility with no trend
            
        except Exception as e:
            warnings.warn(f"Regime detection failed: {e}")
            return 'unknown'
    
    def _calculate_volatility_confidence(self, returns: pd.Series, estimated_vol: float) -> float:
        """Calculate confidence score for volatility estimate"""
        try:
            n = len(returns)
            
            # Base confidence on sample size
            size_confidence = min(1.0, n / (2 * self.config.min_periods))
            
            # Stability check: compare recent vs historical volatility
            if n >= 60:  # Need enough data for comparison
                recent_vol = returns.iloc[-30:].std() * np.sqrt(252)
                historical_vol = returns.iloc[:-30].std() * np.sqrt(252)
                
                # Penalize large differences between recent and historical
                vol_diff = abs(recent_vol - historical_vol) / historical_vol
                stability_confidence = max(0.1, 1.0 - vol_diff)
            else:
                stability_confidence = 0.7  # Default for small samples
            
            # Check for extreme values
            extreme_penalty = 1.0
            if estimated_vol > 2.0:  # Very high volatility (>200%)
                extreme_penalty = 0.5
            elif estimated_vol < 0.05:  # Very low volatility (<5%)
                extreme_penalty = 0.7
            
            # Combined confidence
            confidence = size_confidence * stability_confidence * extreme_penalty
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence
    
    def _calculate_drift_confidence(self, returns: pd.Series, estimated_drift: float) -> float:
        """Calculate confidence score for drift estimate"""
        try:
            n = len(returns)
            
            # Base confidence on sample size
            size_confidence = min(1.0, n / self.config.min_periods)
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            significance_confidence = max(0.2, 1.0 - p_value)
            
            # Consistency check: standard error relative to estimate
            std_error = returns.std() / np.sqrt(n)
            if abs(estimated_drift) > 0:
                consistency = min(1.0, abs(estimated_drift) / (std_error * 252))
                consistency_confidence = min(1.0, consistency / 2.0)  # Scale down
            else:
                consistency_confidence = 0.5
            
            # Combined confidence (drift is generally less reliable than volatility)
            confidence = (size_confidence * 0.4 + 
                         significance_confidence * 0.3 + 
                         consistency_confidence * 0.3)
            
            return max(0.1, min(0.8, confidence))  # Cap at 80% for drift
            
        except Exception:
            return 0.3  # Default low confidence for drift
    
    def get_parameter_confidence(self) -> Dict[str, float]:
        """Get confidence levels for current parameters"""
        if self.current_params is None:
            return {'drift': 0.0, 'volatility': 0.0, 'overall': 0.0}
        
        overall = (self.current_params.drift_confidence + 
                  self.current_params.volatility_confidence) / 2
        
        return {
            'drift': self.current_params.drift_confidence,
            'volatility': self.current_params.volatility_confidence,
            'overall': overall
        }
    
    def estimate_parameter_uncertainty(self, returns: pd.Series) -> Dict[str, Tuple[float, float]]:
        """
        Estimate confidence intervals for parameters.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        Dict[str, Tuple[float, float]]
            Confidence intervals for drift and volatility (95% by default)
        """
        if self.current_params is None:
            raise ParameterEstimationError("No current parameters available")
        
        try:
            n = len(returns)
            
            # Volatility confidence interval using chi-square distribution
            vol_point = self.current_params.volatility
            sample_var = returns.var()
            
            # Chi-square confidence interval for variance
            alpha = 0.05  # 95% confidence
            chi2_lower = stats.chi2.ppf(alpha/2, n-1)
            chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)
            
            var_lower = (n-1) * sample_var / chi2_upper
            var_upper = (n-1) * sample_var / chi2_lower
            
            vol_lower = np.sqrt(var_lower * 252)
            vol_upper = np.sqrt(var_upper * 252)
            
            # Drift confidence interval using t-distribution
            drift_point = self.current_params.drift
            std_error = returns.std() / np.sqrt(n)
            t_critical = stats.t.ppf(1-alpha/2, n-1)
            
            drift_margin = t_critical * std_error * 252
            drift_lower = drift_point - drift_margin
            drift_upper = drift_point + drift_margin
            
            return {
                'volatility': (max(0.01, vol_lower), vol_upper),
                'drift': (drift_lower, drift_upper)
            }
            
        except Exception as e:
            warnings.warn(f"Could not calculate confidence intervals: {e}")
            # Return wide intervals if calculation fails
            vol = self.current_params.volatility
            drift = self.current_params.drift
            
            return {
                'volatility': (vol * 0.5, vol * 1.5),
                'drift': (drift - 0.2, drift + 0.2)
            }
    
    def reset(self) -> None:
        """Reset estimator state"""
        self.current_params = None
        self.estimation_history = []
        self._ewma_var = None
        self._drift_ewma = None
        self._last_returns = None
    
    def get_estimation_summary(self) -> Dict[str, any]:
        """Get summary of estimation process and results"""
        if self.current_params is None:
            return {'status': 'No parameters estimated'}
        
        return {
            'status': 'Active',
            'last_updated': self.current_params.last_updated.isoformat(),
            'drift': f"{self.current_params.drift:.2%}",
            'volatility': f"{self.current_params.volatility:.2%}",
            'regime': self.current_params.regime,
            'drift_confidence': f"{self.current_params.drift_confidence:.1%}",
            'volatility_confidence': f"{self.current_params.volatility_confidence:.1%}",
            'estimation_window': self.current_params.estimation_window,
            'total_updates': len(self.estimation_history),
            'config': {
                'volatility_lambda': self.config.volatility_lambda,
                'drift_alpha': self.config.drift_alpha,
                'min_periods': self.config.min_periods
            }
        }