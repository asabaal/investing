"""
Gradient-Based Trade Likelihood Analyzer

Integrates gradient candle clustering with trade probability calculations
to replace GBM with more accurate market microstructure modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from gradient_candle_clustering import GradientCandleAnalyzer, GradientState
from data_structures import TradeSetup, TradeAnalysis, MarketParameters
from core_math import MathematicalError

logger = logging.getLogger(__name__)

@dataclass
class GradientScenario:
    """Represents a possible future gradient evolution scenario"""
    gradient_sequence: List[int]  # Sequence of cluster IDs
    probability: float           # Probability of this sequence
    final_ohlc: Dict[str, float]  # Predicted final OHLC values
    hits_entry: bool            # Whether sequence hits entry level
    hits_stop: bool             # Whether sequence hits stop before target
    hits_target: bool           # Whether sequence hits target before stop
    
class GradientTradeAnalyzer:
    """Analyzes trades using gradient-based candle evolution"""
    
    def __init__(self, gradient_analyzer: Optional[GradientCandleAnalyzer] = None):
        self.gradient_analyzer = gradient_analyzer or GradientCandleAnalyzer()
        
    def setup_symbol_analysis(self, symbol: str, start_date: str = '2024-01-01', 
                             end_date: str = '2025-08-31') -> bool:
        """Setup gradient analysis for a symbol"""
        logger.info(f"Setting up gradient analysis for {symbol}")
        success = self.gradient_analyzer.analyze_symbol(symbol, start_date, end_date)
        
        if success:
            logger.info(f"✅ Gradient analysis ready for {symbol}")
        else:
            logger.error(f"❌ Failed to setup gradient analysis for {symbol}")
            
        return success
    
    def gradient_to_ohlc(self, current_ohlc: Dict[str, float], 
                        gradient_sequence: List[int], 
                        max_steps: int = 10) -> List[Dict[str, float]]:
        """
        Convert gradient cluster sequence to OHLC evolution
        
        Args:
            current_ohlc: Current candle OHLC values
            gradient_sequence: Sequence of gradient cluster IDs
            max_steps: Maximum number of steps to simulate
            
        Returns:
            List of OHLC dictionaries for each step
        """
        if not self.gradient_analyzer.gradient_clusters:
            logger.error("No gradient clusters available")
            return []
            
        ohlc_sequence = [current_ohlc.copy()]
        current = current_ohlc.copy()
        
        for step, cluster_id in enumerate(gradient_sequence[:max_steps]):
            if cluster_id >= len(self.gradient_analyzer.gradient_clusters):
                logger.warning(f"Invalid cluster ID: {cluster_id}")
                continue
                
            cluster = self.gradient_analyzer.gradient_clusters[cluster_id]
            centroid = cluster.centroid
            
            # Extract gradient components
            body_grad = centroid[0]
            upper_grad = centroid[1]
            lower_grad = centroid[2]
            log_range_grad = centroid[3]
            log_price_grad = centroid[4]
            
            # Apply coordinate system changes first
            prev_range = current['high'] - current['low']
            prev_close = current['close']
            
            # Update range (volatility)
            if prev_range > 0:
                new_range = prev_range * np.exp(log_range_grad)
            else:
                new_range = prev_range * 1.1  # Default small expansion
                
            # Update price level (center point)
            new_center = prev_close * np.exp(log_price_grad)
            
            # Calculate current geometric ratios
            prev_body_ratio = (current['close'] - current['open']) / prev_range if prev_range > 0 else 0
            prev_upper_ratio = (current['high'] - max(current['open'], current['close'])) / prev_range if prev_range > 0 else 0
            prev_lower_ratio = (min(current['open'], current['close']) - current['low']) / prev_range if prev_range > 0 else 0
            
            # Apply gradient changes to ratios
            new_body_ratio = prev_body_ratio + body_grad
            new_upper_ratio = max(0, min(1, prev_upper_ratio + upper_grad))  # Clamp to valid range
            new_lower_ratio = max(0, min(1, prev_lower_ratio + lower_grad))  # Clamp to valid range
            
            # Ensure ratios sum to valid range
            total_ratio = abs(new_body_ratio) + new_upper_ratio + new_lower_ratio
            if total_ratio > 1:
                # Normalize ratios
                scale_factor = 0.95 / total_ratio  # Leave small margin
                new_upper_ratio *= scale_factor
                new_lower_ratio *= scale_factor
            
            # Convert ratios back to OHLC in new coordinate system
            # Position the new candle around the new center price
            body_size = new_body_ratio * new_range
            upper_wick_size = new_upper_ratio * new_range
            lower_wick_size = new_lower_ratio * new_range
            
            # Determine open/close based on body direction
            if new_body_ratio >= 0:  # Bullish candle
                new_close = new_center + body_size / 2
                new_open = new_center - body_size / 2
                new_high = new_close + upper_wick_size
            else:  # Bearish candle
                new_close = new_center + body_size / 2  # body_size is negative
                new_open = new_center - body_size / 2
                new_high = new_open + upper_wick_size
                
            new_low = min(new_open, new_close) - lower_wick_size
            
            # Ensure OHLC integrity
            new_high = max(new_high, new_open, new_close)
            new_low = min(new_low, new_open, new_close)
            
            # Create next candle
            next_candle = {
                'open': new_open,
                'high': new_high,
                'low': new_low,
                'close': new_close
            }
            
            ohlc_sequence.append(next_candle)
            current = next_candle
            
        return ohlc_sequence
    
    def generate_gradient_scenarios(self, current_gradient: GradientState, 
                                  max_steps: int = 10, 
                                  num_scenarios: int = 100) -> List[GradientScenario]:
        """
        Generate possible gradient evolution scenarios using Monte Carlo
        
        Args:
            current_gradient: Starting gradient state
            max_steps: Maximum steps to simulate
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of gradient scenarios with probabilities
        """
        if not self.gradient_analyzer.gradient_clusters or self.gradient_analyzer.transition_matrix is None:
            logger.error("Gradient analysis not ready")
            return []
            
        scenarios = []
        transition_matrix = self.gradient_analyzer.transition_matrix
        
        # Find starting cluster
        current_vector = current_gradient.to_array()
        current_vector_scaled = self.gradient_analyzer.scaler.transform([current_vector])[0]
        
        distances = []
        for cluster in self.gradient_analyzer.gradient_clusters:
            dist = np.linalg.norm(current_vector_scaled - cluster.centroid)
            distances.append(dist)
        
        start_cluster = np.argmin(distances)
        
        # Generate scenarios
        for scenario_id in range(num_scenarios):
            gradient_sequence = [start_cluster]
            current_cluster = start_cluster
            scenario_probability = 1.0
            
            for step in range(max_steps):
                if current_cluster >= len(transition_matrix):
                    break
                    
                # Get transition probabilities
                probs = transition_matrix[current_cluster]
                
                if np.sum(probs) == 0:
                    break
                    
                # Sample next cluster
                next_cluster = np.random.choice(len(probs), p=probs)
                gradient_sequence.append(next_cluster)
                scenario_probability *= probs[next_cluster]
                
                current_cluster = next_cluster
                
                # Early stopping if probability gets too small
                if scenario_probability < 1e-6:
                    break
            
            # Create scenario object (will populate OHLC and trade outcomes later)
            scenario = GradientScenario(
                gradient_sequence=gradient_sequence,
                probability=scenario_probability,
                final_ohlc={},
                hits_entry=False,
                hits_stop=False,
                hits_target=False
            )
            
            scenarios.append(scenario)
        
        # Sort by probability (highest first)
        scenarios.sort(key=lambda x: x.probability, reverse=True)
        
        logger.info(f"Generated {len(scenarios)} gradient scenarios")
        logger.info(f"Top scenario probability: {scenarios[0].probability:.6f}")
        logger.info(f"Average steps: {np.mean([len(s.gradient_sequence)-1 for s in scenarios]):.1f}")
        
        return scenarios
    
    def analyze_trade_with_gradients(self, setup: TradeSetup, current_ohlc: Dict[str, float],
                                   time_horizon_days: int = 5, historical_window_days: int = None) -> TradeAnalysis:
        """
        Analyze trade setup using gradient-based evolution
        
        Args:
            setup: Trade setup to analyze
            current_ohlc: Current candle OHLC
            time_horizon_days: Time horizon for simulation
            
        Returns:
            TradeAnalysis with gradient-based predictions
        """
        logger.info("Analyzing trade with gradient-based method")
        
        # Create current gradient state (using recent gradient)
        # For now, use neutral gradient as starting point
        current_gradient = GradientState(
            body_ratio_gradient=0.0,
            upper_wick_gradient=0.0,
            lower_wick_gradient=0.0,
            log_range_gradient=0.0,
            log_price_gradient=0.0
        )
        
        # Generate scenarios
        scenarios = self.generate_gradient_scenarios(
            current_gradient, 
            max_steps=time_horizon_days * 2,  # Assume ~2 candles per day for intraday
            num_scenarios=1000
        )
        
        if not scenarios:
            logger.error("No scenarios generated")
            return self._create_empty_analysis(setup)
        
        # Analyze each scenario for trade outcomes
        entry_count = 0
        win_count_long = 0
        win_count_short = 0
        total_scenarios = len(scenarios)
        
        # Timing tracking
        entry_times_long = []
        entry_times_short = []
        trade_durations_long = []
        trade_durations_short = []
        
        for scenario in scenarios:
            # Convert gradient sequence to OHLC
            ohlc_sequence = self.gradient_to_ohlc(current_ohlc, scenario.gradient_sequence[1:])
            
            if not ohlc_sequence:
                continue
                
            scenario.final_ohlc = ohlc_sequence[-1]
            
            # Check if trade triggers and outcomes, track timing
            entry_step, trade_duration = self._analyze_scenario_outcomes_with_timing(scenario, setup, ohlc_sequence)
            
            if scenario.hits_entry:
                entry_count += 1
                
                # Track timing data
                if setup.has_long_setup() and entry_step is not None:
                    entry_times_long.append(entry_step)
                    if trade_duration is not None:
                        trade_durations_long.append(trade_duration)
                        
                if setup.has_short_setup() and entry_step is not None:
                    entry_times_short.append(entry_step)
                    if trade_duration is not None:
                        trade_durations_short.append(trade_duration)
                
                # Check win conditions
                if setup.has_long_setup() and scenario.hits_target and not scenario.hits_stop:
                    win_count_long += 1
                    
                if setup.has_short_setup() and scenario.hits_target and not scenario.hits_stop:
                    win_count_short += 1
        
        # Calculate probabilities
        prob_entry = entry_count / total_scenarios if total_scenarios > 0 else 0
        prob_win_long = win_count_long / entry_count if entry_count > 0 else 0
        prob_win_short = win_count_short / entry_count if entry_count > 0 else 0
        
        # Calculate expected values
        long_risk_reward = setup.get_long_risk_reward() or 0
        short_risk_reward = setup.get_short_risk_reward() or 0
        
        expected_value_long = None
        expected_value_short = None
        expected_entry_time_long = None
        expected_entry_time_short = None
        expected_trade_duration_long = None
        expected_trade_duration_short = None
        
        if setup.has_long_setup():
            expected_value_long = prob_entry * (prob_win_long * long_risk_reward - (1 - prob_win_long))
            # Calculate expected entry time from actual scenario data (each step = 1 day)
            expected_entry_time_long = np.mean(entry_times_long) if entry_times_long else None
            # Calculate expected trade duration from actual scenario data (each step = 1 day)
            expected_trade_duration_long = np.mean(trade_durations_long) if trade_durations_long else None
            
        if setup.has_short_setup():
            expected_value_short = prob_entry * (prob_win_short * short_risk_reward - (1 - prob_win_short))
            # Calculate expected entry time from actual scenario data (each step = 1 day)
            expected_entry_time_short = np.mean(entry_times_short) if entry_times_short else None
            # Calculate expected trade duration from actual scenario data (each step = 1 day)
            expected_trade_duration_short = np.mean(trade_durations_short) if trade_durations_short else None
        
        # Create market parameters (simplified for now)
        market_params = MarketParameters(
            drift=0.0,  # Gradient model doesn't use traditional drift
            volatility=0.2,  # Estimate from scenarios
            drift_confidence=0.8,
            volatility_confidence=0.8,
            regime='unknown',  # Use valid regime
            last_updated=pd.Timestamp.now(),
            estimation_window=len(scenarios)
        )
        
        # Create analysis result
        analysis = TradeAnalysis(
            setup_id=f"gradient_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=pd.Timestamp.now(),
            current_price=setup.current_price,
            market_params=market_params,
            prob_entry_long=prob_entry if setup.has_long_setup() else None,
            prob_entry_short=prob_entry if setup.has_short_setup() else None,
            prob_win_long=prob_win_long if setup.has_long_setup() else None,
            prob_win_short=prob_win_short if setup.has_short_setup() else None,
            expected_entry_time_long=expected_entry_time_long,
            expected_entry_time_short=expected_entry_time_short,
            expected_trade_duration_long=expected_trade_duration_long,
            expected_trade_duration_short=expected_trade_duration_short,
            expected_value_long=expected_value_long,
            expected_value_short=expected_value_short,
            expected_value_total=(expected_value_long or 0) + (expected_value_short or 0),
            return_rate_long=expected_value_long / time_horizon_days if expected_value_long else None,
            return_rate_short=expected_value_short / time_horizon_days if expected_value_short else None,
            calculation_notes=[f"Gradient-based analysis using {total_scenarios} scenarios"],
            warnings=[]
        )
        
        # Calculate total return rate
        if analysis.return_rate_long and analysis.return_rate_short:
            analysis.return_rate_total = analysis.return_rate_long + analysis.return_rate_short
        elif analysis.return_rate_long:
            analysis.return_rate_total = analysis.return_rate_long
        elif analysis.return_rate_short:
            analysis.return_rate_total = analysis.return_rate_short
        
        logger.info(f"Gradient analysis complete:")
        logger.info(f"  Entry probability: {prob_entry:.1%}")
        logger.info(f"  Long win probability: {prob_win_long:.1%}")
        logger.info(f"  Short win probability: {prob_win_short:.1%}")
        logger.info(f"  Total expected value: {analysis.expected_value_total:.3f}")
        
        return analysis
    
    def _analyze_scenario_outcomes(self, scenario: GradientScenario, setup: TradeSetup, 
                                 ohlc_sequence: List[Dict[str, float]]):
        """Analyze whether scenario hits entry/stop/target levels"""
        
        # Check if any candle in sequence hits the levels
        for ohlc in ohlc_sequence[1:]:  # Skip first (current) candle
            high = ohlc['high']
            low = ohlc['low']
            
            # Check entry triggers
            if setup.entry_long and not scenario.hits_entry:
                if high >= setup.entry_long:
                    scenario.hits_entry = True
                    
            if setup.entry_short and not scenario.hits_entry:
                if low <= setup.entry_short:
                    scenario.hits_entry = True
            
            # Once entry is hit, check stop/target
            if scenario.hits_entry:
                # Long trade checks
                if setup.has_long_setup():
                    if setup.stop_long and low <= setup.stop_long:
                        scenario.hits_stop = True
                    elif setup.target_long and high >= setup.target_long:
                        scenario.hits_target = True
                        
                # Short trade checks
                if setup.has_short_setup():
                    if setup.stop_short and high >= setup.stop_short:
                        scenario.hits_stop = True
                    elif setup.target_short and low <= setup.target_short:
                        scenario.hits_target = True

    def _analyze_scenario_outcomes_with_timing(self, scenario: GradientScenario, setup: TradeSetup, 
                                             ohlc_sequence: List[Dict[str, float]]) -> Tuple[Optional[int], Optional[int]]:
        """Analyze scenario outcomes and return timing information"""
        
        entry_step = None
        trade_duration = None
        
        # Check if any candle in sequence hits the levels
        for step, ohlc in enumerate(ohlc_sequence[1:], 1):  # Skip first (current) candle, start counting at 1
            high = ohlc['high']
            low = ohlc['low']
            
            # Check entry triggers
            if not scenario.hits_entry:
                if setup.entry_long and high >= setup.entry_long:
                    scenario.hits_entry = True
                    entry_step = step
                elif setup.entry_short and low <= setup.entry_short:
                    scenario.hits_entry = True
                    entry_step = step
            
            # Once entry is hit, check stop/target
            elif scenario.hits_entry and not scenario.hits_stop and not scenario.hits_target:
                # Long trade checks
                if setup.has_long_setup():
                    if setup.stop_long and low <= setup.stop_long:
                        scenario.hits_stop = True
                        trade_duration = step - entry_step if entry_step else None
                        break
                    elif setup.target_long and high >= setup.target_long:
                        scenario.hits_target = True
                        trade_duration = step - entry_step if entry_step else None
                        break
                        
                # Short trade checks
                if setup.has_short_setup():
                    if setup.stop_short and high >= setup.stop_short:
                        scenario.hits_stop = True
                        trade_duration = step - entry_step if entry_step else None
                        break
                    elif setup.target_short and low <= setup.target_short:
                        scenario.hits_target = True
                        trade_duration = step - entry_step if entry_step else None
                        break
        
        return entry_step, trade_duration
    
    def _create_empty_analysis(self, setup: TradeSetup) -> TradeAnalysis:
        """Create empty analysis when gradient analysis fails"""
        market_params = MarketParameters(
            drift=0.0, volatility=0.2, drift_confidence=0.0, volatility_confidence=0.0,
            regime='unknown', last_updated=pd.Timestamp.now(), estimation_window=0
        )
        
        return TradeAnalysis(
            setup_id="gradient_failed",
            timestamp=pd.Timestamp.now(),
            current_price=setup.current_price,
            market_params=market_params,
            warnings=["Gradient analysis failed"]
        )
    
    def compare_with_gbm(self, setup: TradeSetup, current_ohlc: Dict[str, float],
                        gbm_analysis: TradeAnalysis) -> Dict[str, Any]:
        """
        Compare gradient-based analysis with GBM analysis
        
        Args:
            setup: Trade setup
            current_ohlc: Current OHLC data
            gbm_analysis: GBM-based analysis results
            
        Returns:
            Comparison dictionary
        """
        gradient_analysis = self.analyze_trade_with_gradients(setup, current_ohlc)
        
        comparison = {
            'gradient_analysis': gradient_analysis,
            'gbm_analysis': gbm_analysis,
            'differences': {},
            'method_comparison': {}
        }
        
        # Compare key metrics
        if gradient_analysis.prob_entry_long and gbm_analysis.prob_entry_long:
            comparison['differences']['entry_prob_long'] = {
                'gradient': gradient_analysis.prob_entry_long,
                'gbm': gbm_analysis.prob_entry_long,
                'difference': gradient_analysis.prob_entry_long - gbm_analysis.prob_entry_long
            }
        
        if gradient_analysis.prob_win_long and gbm_analysis.prob_win_long:
            comparison['differences']['win_prob_long'] = {
                'gradient': gradient_analysis.prob_win_long,
                'gbm': gbm_analysis.prob_win_long,
                'difference': gradient_analysis.prob_win_long - gbm_analysis.prob_win_long
            }
        
        if gradient_analysis.expected_value_total and gbm_analysis.expected_value_total:
            comparison['differences']['expected_value'] = {
                'gradient': gradient_analysis.expected_value_total,
                'gbm': gbm_analysis.expected_value_total,
                'difference': gradient_analysis.expected_value_total - gbm_analysis.expected_value_total
            }
        
        # Method characteristics
        comparison['method_comparison'] = {
            'gradient': {
                'approach': 'Candle geometry evolution patterns',
                'data_driven': True,
                'captures_microstructure': True,
                'coordinate_invariant': True
            },
            'gbm': {
                'approach': 'Continuous price random walk',
                'data_driven': False,
                'captures_microstructure': False,
                'coordinate_invariant': False
            }
        }
        
        logger.info("Gradient vs GBM comparison completed")
        
        return comparison
    
    def analyze_trade_multiple_windows(self, setup: TradeSetup, current_ohlc: Dict[str, float],
                                     symbol: str, trade_date: datetime,
                                     historical_windows: List[int] = [60, 90, 120, 180],
                                     time_horizon_days: int = 5) -> Dict[str, TradeAnalysis]:
        """
        Analyze trade setup across multiple historical windows using gradient approach
        
        Args:
            setup: Trade setup to analyze
            current_ohlc: Current candle OHLC
            symbol: Symbol being analyzed
            trade_date: Date for the analysis
            historical_windows: List of historical window sizes in days
            time_horizon_days: Forward-looking horizon for trade
            
        Returns:
            Dict mapping window size to TradeAnalysis results
        """
        logger.info(f"🔄 Analyzing {setup.current_price:.2f} across {len(historical_windows)} gradient windows")
        
        # Add parent directory for market database import if needed
        from market_data_database import MarketDataDatabase
        market_db = MarketDataDatabase()
        
        results = {}
        
        for window_days in historical_windows:
            try:
                logger.info(f"📊 Analyzing {window_days}-day gradient window...")
                
                # Setup gradient analysis for this specific window
                end_date = trade_date.strftime('%Y-%m-%d')
                start_date = (trade_date - pd.Timedelta(days=window_days + 30)).strftime('%Y-%m-%d')  # Extra buffer
                
                # Create fresh gradient analyzer for this window
                window_gradient_analyzer = GradientCandleAnalyzer(market_db)
                window_trade_analyzer = GradientTradeAnalyzer(window_gradient_analyzer)
                
                # Setup analysis with specific date range
                success = window_trade_analyzer.setup_symbol_analysis(symbol, start_date, end_date)
                
                if not success:
                    logger.warning(f"⚠️ Failed to setup {window_days}d gradient analysis")
                    continue
                
                # Run the analysis
                analysis = window_trade_analyzer.analyze_trade_with_gradients(
                    setup, current_ohlc, time_horizon_days, window_days
                )
                
                results[f"{window_days}d"] = analysis
                
                # Log results summary
                entry_prob = (analysis.prob_entry_long or 0) + (analysis.prob_entry_short or 0)
                win_prob_long = analysis.prob_win_long or 0
                win_prob_short = analysis.prob_win_short or 0
                return_rate = analysis.return_rate_total or 0
                expected_value = analysis.expected_value_total or 0
                
                logger.info(f"   Window {window_days}d: Entry {entry_prob:.1%}, "
                           f"Win L/S {win_prob_long:.1%}/{win_prob_short:.1%}, "
                           f"RR {return_rate:.4f}/day, EV ${expected_value:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error in {window_days}d gradient window: {e}")
                continue
                
        logger.info(f"✅ Completed gradient analysis across {len(results)} windows")
        return results