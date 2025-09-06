"""
Core mathematical functions for trading probability calculations.

Based on geometric Brownian motion (GBM) and first passage time theory.
All functions use annualized parameters (drift and volatility).
"""

import numpy as np
from scipy import stats
from typing import Tuple, Union
import warnings


class MathematicalError(Exception):
    """Custom exception for mathematical calculation errors"""
    pass


def first_passage_probability(S0: float, barrier: float, mu: float, sigma: float, T: float) -> float:
    """
    Calculate probability of hitting barrier before time T using geometric Brownian motion.
    
    For GBM: dS_t = μ S_t dt + σ S_t dW_t
    
    Uses the analytical solution:
    P(τ_B ≤ T) = Φ(d1) + (S0/B)^(2α) * Φ(d2)
    
    Parameters:
    -----------
    S0 : float
        Current price (must be positive)
    barrier : float  
        Price level to hit (must be positive, different from S0)
    mu : float
        Drift rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Time window in years (must be positive)
        
    Returns:
    --------
    float
        Probability between 0 and 1
        
    Raises:
    -------
    MathematicalError
        If parameters are invalid or calculation fails
    """
    # Input validation
    if S0 <= 0 or barrier <= 0:
        raise MathematicalError("Prices must be positive")
    if sigma <= 0:
        raise MathematicalError("Volatility must be positive")
    if T <= 0:
        raise MathematicalError("Time must be positive")
    if S0 == barrier:
        return 1.0  # Already at barrier
    
    try:
        # Calculate parameters
        alpha = mu / (sigma**2) - 0.5
        ln_ratio = np.log(S0 / barrier)
        sigma_sqrt_T = sigma * np.sqrt(T)
        
        # Calculate d1 and d2
        d1 = (ln_ratio + (mu + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = (ln_ratio + (mu - 0.5 * sigma**2) * T) / sigma_sqrt_T
        
        # Handle numerical issues
        if abs(d1) > 10:  # Extreme values
            return 1.0 if d1 > 0 else 0.0
        if abs(d2) > 10:
            d2 = np.sign(d2) * 10
            
        # Calculate probability components
        phi_d1 = stats.norm.cdf(d1)
        reflection_term = (S0 / barrier) ** (2 * alpha) * stats.norm.cdf(d2)
        
        probability = phi_d1 + reflection_term
        
        # Ensure valid probability
        return max(0.0, min(1.0, probability))
        
    except (OverflowError, UnderflowError, FloatingPointError) as e:
        raise MathematicalError(f"Numerical error in probability calculation: {e}")


def barrier_race_probability(entry: float, take_profit: float, stop_loss: float, 
                           mu: float, sigma: float) -> float:
    """
    Calculate probability of hitting take profit before stop loss.
    
    For a position entered at 'entry', calculates P(hit TP before SL).
    Uses the barrier race formula for GBM.
    
    Parameters:
    -----------
    entry : float
        Entry price (must be positive)
    take_profit : float
        Take profit level (must be positive)
    stop_loss : float
        Stop loss level (must be positive)  
    mu : float
        Drift rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
        
    Returns:
    --------
    float
        Win probability between 0 and 1
        
    Raises:
    -------
    MathematicalError
        If parameters are invalid or inconsistent
    """
    # Input validation
    if entry <= 0 or take_profit <= 0 or stop_loss <= 0:
        raise MathematicalError("All price levels must be positive")
    if sigma <= 0:
        raise MathematicalError("Volatility must be positive")
    
    # Validate trade setup logic
    if entry == take_profit or entry == stop_loss:
        raise MathematicalError("Entry price cannot equal TP or SL")
    if take_profit == stop_loss:
        raise MathematicalError("Take profit and stop loss cannot be equal")
    
    # Determine trade direction and validate setup
    is_long = take_profit > entry
    is_short = take_profit < entry
    
    if is_long and stop_loss >= entry:
        raise MathematicalError("Long trade: stop loss must be below entry")
    if is_short and stop_loss <= entry:
        raise MathematicalError("Short trade: stop loss must be above entry")
    
    try:
        # For zero drift case (more stable numerically)
        if abs(mu) < 1e-8:
            ln_entry_sl = np.log(entry / stop_loss)
            ln_tp_sl = np.log(take_profit / stop_loss)
            return ln_entry_sl / ln_tp_sl
        
        # General case with drift
        alpha = mu / (sigma**2) - 0.5
        
        # Calculate ratios
        sl_entry_ratio = stop_loss / entry
        sl_tp_ratio = stop_loss / take_profit
        
        # Handle potential numerical issues
        if abs(2 * alpha) > 50:  # Very large alpha
            # Use limiting behavior
            if alpha > 0 and is_long:
                return 1.0
            elif alpha < 0 and is_short:
                return 1.0
            else:
                return 0.0
        
        # Calculate probability using barrier race formula
        numerator = 1 - sl_entry_ratio ** (2 * alpha)
        denominator = 1 - sl_tp_ratio ** (2 * alpha)
        
        if abs(denominator) < 1e-12:
            # Handle degenerate case
            return 0.5
            
        probability = numerator / denominator
        
        # Ensure valid probability
        return max(0.0, min(1.0, probability))
        
    except (OverflowError, UnderflowError, FloatingPointError) as e:
        raise MathematicalError(f"Numerical error in win probability calculation: {e}")


def expected_first_passage_time(S0: float, barrier: float, mu: float, sigma: float) -> float:
    """
    Calculate expected time to hit barrier starting from S0.
    
    For geometric Brownian motion, the expected first passage time depends
    on whether the drift is toward or away from the barrier.
    
    Parameters:
    -----------
    S0 : float
        Current price (must be positive)
    barrier : float
        Price level to hit (must be positive)
    mu : float
        Drift rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
        
    Returns:
    --------
    float
        Expected time in years (np.inf if drift opposes hitting barrier)
        
    Raises:
    -------
    MathematicalError
        If parameters are invalid
    """
    # Input validation
    if S0 <= 0 or barrier <= 0:
        raise MathematicalError("Prices must be positive")
    if sigma <= 0:
        raise MathematicalError("Volatility must be positive")
    if S0 == barrier:
        return 0.0  # Already at barrier
    
    try:
        ln_ratio = np.log(barrier / S0)
        
        # Zero drift case
        if abs(mu) < 1e-8:
            return ln_ratio**2 / (sigma**2)
        
        # Check if drift is toward barrier
        drift_toward_barrier = (barrier > S0 and mu > 0) or (barrier < S0 and mu < 0)
        
        if not drift_toward_barrier:
            return np.inf  # Drift opposes hitting barrier
        
        # Calculate expected time
        expected_time = ln_ratio / mu
        
        return max(0.0, expected_time)
        
    except (OverflowError, UnderflowError) as e:
        raise MathematicalError(f"Numerical error in expected time calculation: {e}")


def expected_exit_time(entry: float, take_profit: float, stop_loss: float,
                      mu: float, sigma: float) -> float:
    """
    Calculate expected time to exit position (hit either TP or SL).
    
    Uses the formula for expected exit time from a double barrier.
    Handles both long and short trades automatically.
    
    Parameters:
    -----------
    entry : float
        Entry price (must be positive)
    take_profit : float
        Take profit level (must be positive) 
    stop_loss : float
        Stop loss level (must be positive)
    mu : float
        Drift rate (annualized)  
    sigma : float
        Volatility (annualized, must be positive)
        
    Returns:
    --------
    float
        Expected exit time in years
        
    Raises:
    -------
    MathematicalError
        If parameters are invalid
    """
    # Input validation
    if entry <= 0 or take_profit <= 0 or stop_loss <= 0:
        raise MathematicalError("All price levels must be positive")
    if sigma <= 0:
        raise MathematicalError("Volatility must be positive")
    if entry == take_profit or entry == stop_loss:
        return 0.0  # Already at exit level
    
    # Validate trade setup
    if take_profit == stop_loss:
        raise MathematicalError("Take profit and stop loss cannot be equal")
    
    # Determine trade direction and normalize to long trade format
    # The math formula requires upper_barrier > entry > lower_barrier
    if take_profit > stop_loss:
        # Long trade: TP > entry > SL (normal case)
        upper_barrier = take_profit
        lower_barrier = stop_loss
        trade_mu = mu  # Use drift as-is for long trades
    else:
        # Short trade: SL > entry > TP, need to normalize
        upper_barrier = stop_loss  
        lower_barrier = take_profit
        trade_mu = -mu  # Flip drift for short trades (short benefits from downward movement)
    
    try:
        # Zero drift case (more stable numerically)
        if abs(trade_mu) < 1e-8:
            upper_entry_diff = upper_barrier - entry
            entry_lower_diff = entry - lower_barrier
            total_diff = upper_barrier - lower_barrier
            
            return (2 / (sigma**2)) * (upper_entry_diff * entry_lower_diff) / total_diff
        
        # General case with drift
        mu_squared_plus_half_sigma_squared = trade_mu**2 + 0.5 * sigma**2
        
        if abs(mu_squared_plus_half_sigma_squared) < 1e-12:
            # Degenerate case, fall back to zero drift
            upper_entry_diff = upper_barrier - entry
            entry_lower_diff = entry - lower_barrier
            total_diff = upper_barrier - lower_barrier
            
            return (2 / (sigma**2)) * (upper_entry_diff * entry_lower_diff) / total_diff
        
        # Calculate logarithmic terms using normalized barriers
        ln_upper_entry = np.log(upper_barrier / entry)
        ln_entry_lower = np.log(entry / lower_barrier)
        total_diff = upper_barrier - lower_barrier
        upper_entry_diff = upper_barrier - entry
        entry_lower_diff = entry - lower_barrier
        
        # Calculate expected time components
        term1 = (upper_entry_diff / total_diff) * ln_upper_entry
        term2 = (entry_lower_diff / total_diff) * ln_entry_lower
        
        expected_time = (term1 + term2) / mu_squared_plus_half_sigma_squared
        
        return max(0.0, expected_time)
        
    except (OverflowError, UnderflowError, FloatingPointError) as e:
        raise MathematicalError(f"Numerical error in exit time calculation: {e}")


def calculate_risk_reward_ratio(entry: float, take_profit: float, stop_loss: float) -> float:
    """
    Calculate risk-reward ratio for a trade setup.
    
    Parameters:
    -----------
    entry : float
        Entry price
    take_profit : float
        Take profit level
    stop_loss : float
        Stop loss level
        
    Returns:
    --------
    float
        Risk-reward ratio (reward/risk)
    """
    if entry <= 0 or take_profit <= 0 or stop_loss <= 0:
        raise MathematicalError("All price levels must be positive")
    
    reward = abs(take_profit - entry)
    risk = abs(entry - stop_loss)
    
    if risk == 0:
        return np.inf
    
    return reward / risk


def validate_trade_setup_math(entry: float, take_profit: float, stop_loss: float) -> Tuple[bool, str]:
    """
    Validate that trade setup prices are mathematically consistent.
    
    Parameters:
    -----------
    entry : float
        Entry price
    take_profit : float
        Take profit level
    stop_loss : float
        Stop loss level
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        if entry <= 0 or take_profit <= 0 or stop_loss <= 0:
            return False, "All prices must be positive"
        
        if entry == take_profit:
            return False, "Entry and take profit cannot be equal"
        
        if entry == stop_loss:
            return False, "Entry and stop loss cannot be equal"
        
        if take_profit == stop_loss:
            return False, "Take profit and stop loss cannot be equal"
        
        # Check if it's a valid long or short setup
        is_long = take_profit > entry
        is_short = take_profit < entry
        
        if is_long and stop_loss >= entry:
            return False, "Long setup: stop loss must be below entry price"
        
        if is_short and stop_loss <= entry:
            return False, "Short setup: stop loss must be above entry price"
        
        return True, "Valid trade setup"
        
    except Exception as e:
        return False, f"Validation error: {e}"