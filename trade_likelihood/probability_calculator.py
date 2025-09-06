"""
Probability calculator for trading setups.

High-level interface for calculating trade probabilities, expected times,
and return rates using the core mathematical functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

from .core_math import (
    first_passage_probability,
    barrier_race_probability, 
    expected_first_passage_time,
    expected_exit_time,
    calculate_risk_reward_ratio,
    validate_trade_setup_math,
    MathematicalError
)
from .data_structures import TradeSetup, MarketParameters, TradeAnalysis


class CalculationError(Exception):
    """Exception for probability calculation errors"""
    pass


@dataclass
class CalculationResult:
    """Result container for individual calculations"""
    value: float
    confidence: float
    notes: List[str]
    warnings: List[str]
    
    def is_reliable(self, min_confidence: float = 0.7) -> bool:
        """Check if result is reliable based on confidence"""
        return self.confidence >= min_confidence


class ProbabilityCalculator:
    """
    High-level interface for trading probability calculations.
    
    Combines the core mathematical functions with practical trading logic,
    error handling, and confidence assessment.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 max_time_horizon: float = 30.0,  # days
                 min_probability: float = 0.01):
        """
        Initialize probability calculator.
        
        Parameters:
        -----------
        confidence_threshold : float
            Minimum confidence for reliable calculations
        max_time_horizon : float
            Maximum time horizon for calculations (days)
        min_probability : float
            Minimum probability threshold
        """
        self.confidence_threshold = confidence_threshold
        self.max_time_horizon = max_time_horizon
        self.min_probability = min_probability
    
    def calculate_entry_probability(self, 
                                  current_price: float,
                                  entry_level: float,
                                  market_params: MarketParameters,
                                  time_window: float) -> CalculationResult:
        """
        Calculate probability of hitting entry level within time window.
        
        Parameters:
        -----------
        current_price : float
            Current asset price
        entry_level : float
            Target entry price level
        market_params : MarketParameters
            Current market parameter estimates
        time_window : float
            Time window in days
            
        Returns:
        --------
        CalculationResult
            Probability calculation with confidence and metadata
        """
        notes = []
        warnings_list = []
        
        try:
            # Validate inputs
            if time_window > self.max_time_horizon:
                warnings_list.append(f"Time window ({time_window:.1f}) exceeds max ({self.max_time_horizon})")
                time_window = self.max_time_horizon
            
            # Convert time to years for calculations
            time_years = time_window / 365.25
            
            # Calculate probability
            prob = first_passage_probability(
                S0=current_price,
                barrier=entry_level,
                mu=market_params.drift,
                sigma=market_params.volatility,
                T=time_years
            )
            
            # Assess confidence based on market parameter confidence
            confidence = (market_params.drift_confidence + 
                         market_params.volatility_confidence) / 2
            
            # Adjust confidence based on calculation specifics
            if abs(entry_level - current_price) / current_price < 0.005:  # Very close to current price
                confidence *= 0.8
                notes.append("Entry level very close to current price")
            
            if time_window < 0.1:  # Less than 2.4 hours
                confidence *= 0.7
                warnings_list.append("Very short time window may be unreliable")
            
            # Check for extreme parameter values
            if market_params.volatility > 1.0:  # > 100% annual volatility
                confidence *= 0.6
                warnings_list.append("High volatility may affect calculation accuracy")
            
            notes.append(f"Using {market_params.regime} regime parameters")
            
            return CalculationResult(
                value=max(self.min_probability, prob),
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except MathematicalError as e:
            raise CalculationError(f"Entry probability calculation failed: {e}")
        except Exception as e:
            raise CalculationError(f"Unexpected error in entry probability: {e}")
    
    def calculate_win_probability(self,
                                entry_price: float,
                                take_profit: float,
                                stop_loss: float,
                                market_params: MarketParameters) -> CalculationResult:
        """
        Calculate probability of hitting take profit before stop loss.
        
        Parameters:
        -----------
        entry_price : float
            Entry price level
        take_profit : float
            Take profit level
        stop_loss : float
            Stop loss level
        market_params : MarketParameters
            Current market parameter estimates
            
        Returns:
        --------
        CalculationResult
            Win probability with confidence assessment
        """
        notes = []
        warnings_list = []
        
        try:
            # Validate trade setup
            is_valid, error_msg = validate_trade_setup_math(entry_price, take_profit, stop_loss)
            if not is_valid:
                raise CalculationError(f"Invalid trade setup: {error_msg}")
            
            # Calculate win probability
            prob = barrier_race_probability(
                entry=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mu=market_params.drift,
                sigma=market_params.volatility
            )
            
            # Calculate risk-reward ratio for context
            risk_reward = calculate_risk_reward_ratio(entry_price, take_profit, stop_loss)
            
            # Assess confidence
            confidence = market_params.volatility_confidence  # Win prob depends more on vol than drift
            
            # Adjust confidence based on trade characteristics
            if risk_reward < 0.5:  # Very poor risk-reward
                confidence *= 0.7
                warnings_list.append(f"Poor risk-reward ratio: {risk_reward:.2f}")
            elif risk_reward > 5.0:  # Very high risk-reward
                confidence *= 0.8
                notes.append(f"High risk-reward ratio: {risk_reward:.2f}")
            
            # Check drift impact
            is_long = take_profit > entry_price
            drift_helps = (is_long and market_params.drift > 0) or (not is_long and market_params.drift < 0)
            
            if drift_helps:
                notes.append("Market drift favors this trade direction")
            else:
                notes.append("Market drift opposes this trade direction")
                confidence *= 0.9  # Slight penalty for opposing drift
            
            notes.append(f"Risk-reward ratio: {risk_reward:.2f}")
            
            return CalculationResult(
                value=max(self.min_probability, min(1.0 - self.min_probability, prob)),
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except MathematicalError as e:
            raise CalculationError(f"Win probability calculation failed: {e}")
        except Exception as e:
            raise CalculationError(f"Unexpected error in win probability: {e}")
    
    def calculate_expected_entry_time(self,
                                    current_price: float,
                                    entry_level: float,
                                    market_params: MarketParameters) -> CalculationResult:
        """
        Calculate expected time to reach entry level.
        
        Parameters:
        -----------
        current_price : float
            Current asset price
        entry_level : float
            Target entry level
        market_params : MarketParameters
            Market parameter estimates
            
        Returns:
        --------
        CalculationResult
            Expected time in days with confidence
        """
        notes = []
        warnings_list = []
        
        try:
            # Calculate expected time in years
            time_years = expected_first_passage_time(
                S0=current_price,
                barrier=entry_level,
                mu=market_params.drift,
                sigma=market_params.volatility
            )
            
            # Convert to days
            time_days = time_years * 365.25
            
            # Handle infinite times (drift opposes movement)
            if np.isinf(time_days):
                time_days = self.max_time_horizon * 10  # Very large but finite
                warnings_list.append("Expected time is infinite (drift opposes reaching entry)")
                confidence = 0.1
            else:
                # Base confidence on drift reliability (entry time depends heavily on drift)
                confidence = market_params.drift_confidence * 0.8  # Lower than probability confidence
                
                if time_days > self.max_time_horizon:
                    warnings_list.append(f"Expected time ({time_days:.1f} days) exceeds max horizon")
                    confidence *= 0.6
            
            # Distance-based confidence adjustment
            price_distance = abs(entry_level - current_price) / current_price
            if price_distance < 0.01:  # Less than 1%
                confidence *= 1.1  # Boost confidence for nearby levels
                notes.append("Entry level is nearby")
            elif price_distance > 0.1:  # More than 10%
                confidence *= 0.8
                notes.append("Entry level is distant")
            
            confidence = max(0.1, min(1.0, confidence))
            
            return CalculationResult(
                value=time_days,
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except MathematicalError as e:
            raise CalculationError(f"Expected entry time calculation failed: {e}")
        except Exception as e:
            raise CalculationError(f"Unexpected error in entry time: {e}")
    
    def calculate_expected_trade_duration(self,
                                        entry_price: float,
                                        take_profit: float,
                                        stop_loss: float,
                                        market_params: MarketParameters) -> CalculationResult:
        """
        Calculate expected time to exit trade (hit either TP or SL).
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        take_profit : float
            Take profit level
        stop_loss : float  
            Stop loss level
        market_params : MarketParameters
            Market parameters
            
        Returns:
        --------
        CalculationResult
            Expected duration in days
        """
        notes = []
        warnings_list = []
        
        try:
            # Validate setup
            is_valid, error_msg = validate_trade_setup_math(entry_price, take_profit, stop_loss)
            if not is_valid:
                raise CalculationError(f"Invalid trade setup: {error_msg}")
            
            # Calculate expected exit time in years
            time_years = expected_exit_time(
                entry=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mu=market_params.drift,
                sigma=market_params.volatility
            )
            
            # Convert to days
            time_days = time_years * 365.25
            
            # Assess confidence (exit time depends on both drift and volatility)
            confidence = (market_params.drift_confidence + market_params.volatility_confidence) / 2
            
            # Adjust based on trade setup characteristics
            risk_reward = calculate_risk_reward_ratio(entry_price, take_profit, stop_loss)
            
            if risk_reward > 3.0:  # High RR trades may take longer
                confidence *= 0.9
                notes.append("High risk-reward may extend trade duration")
            
            if time_days > self.max_time_horizon:
                warnings_list.append(f"Expected duration ({time_days:.1f} days) is very long")
                confidence *= 0.7
            
            # Very short durations may be unreliable
            if time_days < 0.1:  # Less than 2.4 hours
                warnings_list.append("Very short expected duration")
                confidence *= 0.8
            
            notes.append(f"Based on risk-reward ratio: {risk_reward:.2f}")
            
            return CalculationResult(
                value=time_days,
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except MathematicalError as e:
            raise CalculationError(f"Trade duration calculation failed: {e}")
        except Exception as e:
            raise CalculationError(f"Unexpected error in trade duration: {e}")
    
    def calculate_expected_value(self,
                               entry_probability: float,
                               win_probability: float,
                               risk_reward_ratio: float) -> CalculationResult:
        """
        Calculate expected value for a trade setup.
        
        EV = P(entry) × [P(win|entry) × RR - (1 - P(win|entry)) × 1]
        
        Parameters:
        -----------
        entry_probability : float
            Probability of trade entry
        win_probability : float
            Probability of win given entry
        risk_reward_ratio : float
            Risk-reward ratio
            
        Returns:
        --------
        CalculationResult
            Expected value calculation
        """
        notes = []
        warnings_list = []
        
        try:
            # Calculate expected value
            expected_value = entry_probability * (
                win_probability * risk_reward_ratio - 
                (1 - win_probability) * 1.0
            )
            
            # Confidence based on input reliability (this is a simple calculation)
            confidence = 0.9  # High confidence in the math itself
            
            # Add context notes
            if expected_value > 0:
                notes.append("Positive expected value - favorable setup")
            else:
                notes.append("Negative expected value - unfavorable setup")
            
            if entry_probability < 0.3:
                warnings_list.append("Low entry probability reduces overall expected value")
            
            if risk_reward_ratio < 1.0:
                warnings_list.append("Risk-reward ratio < 1.0 is generally unfavorable")
            
            notes.append(f"Entry prob: {entry_probability:.2%}, Win prob: {win_probability:.2%}")
            
            return CalculationResult(
                value=expected_value,
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except Exception as e:
            raise CalculationError(f"Expected value calculation failed: {e}")
    
    def calculate_return_rate(self,
                            expected_value: float,
                            expected_entry_time: float,
                            expected_trade_duration: float,
                            entry_probability: float) -> CalculationResult:
        """
        Calculate return rate (expected value per unit time).
        
        This is the key innovation: RR = EV / (Entry_Time + P(entry) × Trade_Duration)
        
        Parameters:
        -----------
        expected_value : float
            Expected value of the trade
        expected_entry_time : float
            Expected time to entry (days)
        expected_trade_duration : float
            Expected trade duration (days)
        entry_probability : float
            Probability of trade entry
            
        Returns:
        --------
        CalculationResult
            Return rate per day
        """
        notes = []
        warnings_list = []
        
        try:
            # Calculate total expected time investment
            total_time = expected_entry_time + entry_probability * expected_trade_duration
            
            if total_time <= 0:
                raise CalculationError("Total expected time must be positive")
            
            # Calculate return rate
            return_rate = expected_value / total_time
            
            # Confidence assessment
            confidence = 0.7  # Moderate confidence as this combines multiple estimates
            
            # Adjust confidence based on time estimates
            if expected_entry_time > self.max_time_horizon:
                confidence *= 0.6
                warnings_list.append("Very long entry time reduces confidence")
            
            if expected_trade_duration > self.max_time_horizon:
                confidence *= 0.8
                warnings_list.append("Very long trade duration")
            
            # Context notes
            if return_rate > 0.1:  # >10% per day
                notes.append("High return rate - very attractive setup")
            elif return_rate > 0.01:  # >1% per day
                notes.append("Moderate return rate - decent setup")
            elif return_rate > 0:
                notes.append("Low positive return rate")
            else:
                notes.append("Negative return rate - avoid this setup")
            
            notes.append(f"Total expected time: {total_time:.2f} days")
            
            return CalculationResult(
                value=return_rate,
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except Exception as e:
            raise CalculationError(f"Return rate calculation failed: {e}")
    
    def calculate_correlation_risk(self,
                                 long_entry_prob: Optional[float],
                                 short_entry_prob: Optional[float]) -> CalculationResult:
        """
        Estimate correlation risk for bidirectional setups.
        
        Parameters:
        -----------
        long_entry_prob : float, optional
            Long entry probability
        short_entry_prob : float, optional
            Short entry probability
            
        Returns:
        --------
        CalculationResult
            Correlation risk assessment
        """
        notes = []
        warnings_list = []
        
        try:
            if long_entry_prob is None or short_entry_prob is None:
                return CalculationResult(
                    value=0.0,
                    confidence=1.0,
                    notes=["No correlation risk - unidirectional setup"],
                    warnings=[]
                )
            
            # Simple correlation risk model
            # Higher when both probabilities are high (both trades likely to trigger)
            correlation_risk = long_entry_prob * short_entry_prob
            
            confidence = 0.6  # Moderate confidence in this simple model
            
            if correlation_risk > 0.3:
                warnings_list.append("High correlation risk - both trades may trigger")
            elif correlation_risk > 0.1:
                notes.append("Moderate correlation risk")
            else:
                notes.append("Low correlation risk")
            
            notes.append(f"Long prob: {long_entry_prob:.2%}, Short prob: {short_entry_prob:.2%}")
            
            return CalculationResult(
                value=correlation_risk,
                confidence=confidence,
                notes=notes,
                warnings=warnings_list
            )
            
        except Exception as e:
            raise CalculationError(f"Correlation risk calculation failed: {e}")
    
    def validate_calculation_inputs(self,
                                  trade_setup: TradeSetup,
                                  market_params: MarketParameters) -> Tuple[bool, List[str]]:
        """
        Validate inputs before running calculations.
        
        Parameters:
        -----------
        trade_setup : TradeSetup
            Trade setup configuration
        market_params : MarketParameters
            Market parameters
            
        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check market parameters
            if market_params.volatility <= 0:
                errors.append("Volatility must be positive")
            
            if market_params.volatility > 3.0:  # >300% annual vol
                errors.append("Volatility is extremely high and may cause numerical issues")
            
            if not market_params.is_high_confidence(self.confidence_threshold):
                errors.append(f"Market parameters have low confidence (< {self.confidence_threshold:.0%})")
            
            # Check trade setup
            if trade_setup.max_time_window <= 0:
                errors.append("Time window must be positive")
            
            if trade_setup.max_time_window > self.max_time_horizon:
                errors.append(f"Time window exceeds maximum ({self.max_time_horizon} days)")
            
            # Validate individual setups
            if trade_setup.has_long_setup():
                is_valid, msg = validate_trade_setup_math(
                    trade_setup.entry_long, 
                    trade_setup.target_long, 
                    trade_setup.stop_long
                )
                if not is_valid:
                    errors.append(f"Long setup invalid: {msg}")
            
            if trade_setup.has_short_setup():
                is_valid, msg = validate_trade_setup_math(
                    trade_setup.entry_short,
                    trade_setup.target_short,
                    trade_setup.stop_short
                )
                if not is_valid:
                    errors.append(f"Short setup invalid: {msg}")
            
            if not trade_setup.has_long_setup() and not trade_setup.has_short_setup():
                errors.append("No valid trade setup (neither long nor short)")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors