"""
Main interface for trading probability analysis.

This is the primary user interface for analyzing trade setups and making
paper trading decisions based on probability calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime
import uuid

from .market_data import MarketDataManager, DataQualityError
from .parameter_estimation import ParameterEstimator, EstimationConfig, ParameterEstimationError
from .probability_calculator import ProbabilityCalculator, CalculationError
from .data_structures import (
    TradeSetup, MarketParameters, TradeAnalysis, BacktestResult,
    create_simple_long_setup, create_simple_short_setup, create_bidirectional_setup
)


class TradeAnalysisError(Exception):
    """Exception for trade analysis errors"""
    pass


class TradeAnalyzer:
    """
    Main interface for trade analysis and optimization.
    
    This is your primary tool for paper trading decisions. It combines
    market data management, parameter estimation, and probability calculations
    into a single, easy-to-use interface.
    """
    
    def __init__(self, 
                 estimation_config: Optional[EstimationConfig] = None,
                 max_history: int = 1000):
        """
        Initialize the trade analyzer.
        
        Parameters:
        -----------
        estimation_config : EstimationConfig, optional
            Configuration for parameter estimation
        max_history : int
            Maximum price history to maintain
        """
        # Core components
        self.data_manager = MarketDataManager(max_history=max_history)
        self.param_estimator = ParameterEstimator(estimation_config)
        self.prob_calculator = ProbabilityCalculator()
        
        # State tracking
        self.current_parameters: Optional[MarketParameters] = None
        self.analysis_history: List[TradeAnalysis] = []
        self.last_analysis_id: Optional[str] = None
        
        # Configuration
        self.min_data_quality_score = 0.8
        self.min_parameter_confidence = 0.6
        
    def load_price_data(self, 
                       prices: Union[pd.Series, List[float], np.ndarray],
                       timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                       validate: bool = True) -> Dict[str, Any]:
        """
        Load historical price data for analysis.
        
        Parameters:
        -----------
        prices : pd.Series, List[float], or np.ndarray
            Historical price data
        timestamps : pd.DatetimeIndex or List[datetime], optional
            Corresponding timestamps
        validate : bool
            Whether to validate data quality
            
        Returns:
        --------
        Dict[str, Any]
            Data loading summary and quality report
        """
        try:
            # Load data into manager
            self.data_manager.add_prices(prices, timestamps)
            
            # Validate data quality if requested
            quality_report = None
            if validate:
                quality_report = self.data_manager.validate_data_quality()
                
                if not quality_report.is_acceptable(self.min_data_quality_score):
                    warnings.warn(f"Data quality score ({quality_report.quality_score:.2f}) "
                                f"below threshold ({self.min_data_quality_score})")
            
            # Update market parameters with new data
            returns = self.data_manager.get_returns()
            if len(returns) >= self.param_estimator.config.min_periods:
                self.current_parameters = self.param_estimator.update_parameters(returns)
            
            # Return summary
            summary = {
                'status': 'success',
                'data_points': len(self.data_manager.price_history),
                'price_range': [
                    self.data_manager.price_history.min(),
                    self.data_manager.price_history.max()
                ],
                'date_range': [
                    self.data_manager.price_history.index[0],
                    self.data_manager.price_history.index[-1]
                ],
                'parameters_estimated': self.current_parameters is not None,
                'data_quality': quality_report.to_dict() if quality_report else None
            }
            
            return summary
            
        except Exception as e:
            raise TradeAnalysisError(f"Failed to load price data: {e}")
    
    def add_new_price(self, 
                     price: float,
                     timestamp: Optional[pd.Timestamp] = None,
                     update_parameters: bool = True) -> Optional[MarketParameters]:
        """
        Add a single new price point and optionally update parameters.
        
        Parameters:
        -----------
        price : float
            New price value
        timestamp : pd.Timestamp, optional
            Timestamp for the price
        update_parameters : bool
            Whether to recalculate market parameters
            
        Returns:
        --------
        MarketParameters, optional
            Updated parameters if recalculated
        """
        try:
            # Add price to data manager
            self.data_manager.add_price(price, timestamp)
            
            # Update parameters if requested
            if update_parameters:
                returns = self.data_manager.get_returns()
                if len(returns) >= self.param_estimator.config.min_periods:
                    self.current_parameters = self.param_estimator.update_parameters(returns)
                    return self.current_parameters
            
            return None
            
        except Exception as e:
            raise TradeAnalysisError(f"Failed to add new price: {e}")
    
    def analyze_trade_setup(self, 
                           trade_setup: TradeSetup,
                           setup_id: Optional[str] = None,
                           update_parameters: bool = False) -> TradeAnalysis:
        """
        Perform complete analysis of a trade setup.
        
        This is the main function you'll use for paper trading decisions!
        
        Parameters:
        -----------
        trade_setup : TradeSetup
            Trade setup configuration to analyze
        setup_id : str, optional
            Identifier for this analysis (auto-generated if None)
        update_parameters : bool
            Whether to recalculate parameters before analysis
            
        Returns:
        --------
        TradeAnalysis
            Complete analysis results with probabilities, times, and return rates
        """
        if setup_id is None:
            setup_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        try:
            # Update parameters if requested
            if update_parameters:
                returns = self.data_manager.get_returns()
                if len(returns) >= self.param_estimator.config.min_periods:
                    self.current_parameters = self.param_estimator.update_parameters(returns)
            
            # Check if we have parameters
            if self.current_parameters is None:
                raise TradeAnalysisError("No market parameters available - load price data first")
            
            # Validate inputs
            is_valid, errors = self.prob_calculator.validate_calculation_inputs(
                trade_setup, self.current_parameters
            )
            
            calculation_notes = []
            warnings_list = []
            
            if not is_valid:
                warnings_list.extend(errors)
                # Continue with analysis but flag warnings
            
            # Initialize analysis results
            analysis = TradeAnalysis(
                setup_id=setup_id,
                timestamp=pd.Timestamp.now(),
                current_price=trade_setup.current_price,
                market_params=self.current_parameters
            )
            
            # Analyze long setup if present
            if trade_setup.has_long_setup():
                long_results = self._analyze_single_direction(
                    trade_setup.current_price,
                    trade_setup.entry_long,
                    trade_setup.target_long,
                    trade_setup.stop_long,
                    trade_setup.max_time_window,
                    'long'
                )
                
                # Store long results
                analysis.prob_entry_long = long_results['entry_probability']
                analysis.prob_win_long = long_results['win_probability']
                analysis.expected_entry_time_long = long_results['entry_time']
                analysis.expected_trade_duration_long = long_results['trade_duration']
                analysis.expected_value_long = long_results['expected_value']
                analysis.return_rate_long = long_results['return_rate']
                analysis.risk_reward_long = long_results['risk_reward']
                
                # Collect notes and warnings
                calculation_notes.extend(long_results['notes'])
                warnings_list.extend(long_results['warnings'])
            
            # Analyze short setup if present
            if trade_setup.has_short_setup():
                short_results = self._analyze_single_direction(
                    trade_setup.current_price,
                    trade_setup.entry_short,
                    trade_setup.target_short,
                    trade_setup.stop_short,
                    trade_setup.max_time_window,
                    'short'
                )
                
                # Store short results
                analysis.prob_entry_short = short_results['entry_probability']
                analysis.prob_win_short = short_results['win_probability']
                analysis.expected_entry_time_short = short_results['entry_time']
                analysis.expected_trade_duration_short = short_results['trade_duration']
                analysis.expected_value_short = short_results['expected_value']
                analysis.return_rate_short = short_results['return_rate']
                analysis.risk_reward_short = short_results['risk_reward']
                
                # Collect notes and warnings
                calculation_notes.extend(short_results['notes'])
                warnings_list.extend(short_results['warnings'])
            
            # Calculate combined metrics
            self._calculate_combined_metrics(analysis)
            
            # Calculate correlation risk for bidirectional setups
            if trade_setup.has_long_setup() and trade_setup.has_short_setup():
                corr_risk = self.prob_calculator.calculate_correlation_risk(
                    analysis.prob_entry_long,
                    analysis.prob_entry_short
                )
                analysis.correlation_risk = corr_risk.value
                calculation_notes.extend(corr_risk.notes)
                warnings_list.extend(corr_risk.warnings)
            
            # Store metadata
            analysis.calculation_notes = calculation_notes
            analysis.warnings = list(set(warnings_list))  # Remove duplicates
            
            # Add to history
            self.analysis_history.append(analysis)
            self.last_analysis_id = setup_id
            
            return analysis
            
        except Exception as e:
            raise TradeAnalysisError(f"Trade analysis failed: {e}")
    
    def _analyze_single_direction(self,
                                current_price: float,
                                entry_level: float,
                                target_level: float,
                                stop_level: float,
                                time_window: float,
                                direction: str) -> Dict[str, Any]:
        """Analyze a single trade direction (long or short)"""
        
        try:
            # Calculate entry probability
            entry_result = self.prob_calculator.calculate_entry_probability(
                current_price, entry_level, self.current_parameters, time_window
            )
            
            # Calculate win probability given entry
            win_result = self.prob_calculator.calculate_win_probability(
                entry_level, target_level, stop_level, self.current_parameters
            )
            
            # Calculate expected entry time
            entry_time_result = self.prob_calculator.calculate_expected_entry_time(
                current_price, entry_level, self.current_parameters
            )
            
            # Calculate expected trade duration
            trade_duration_result = self.prob_calculator.calculate_expected_trade_duration(
                entry_level, target_level, stop_level, self.current_parameters
            )
            
            # Calculate risk-reward ratio
            from .core_math import calculate_risk_reward_ratio
            risk_reward = calculate_risk_reward_ratio(entry_level, target_level, stop_level)
            
            # Calculate expected value
            ev_result = self.prob_calculator.calculate_expected_value(
                entry_result.value, win_result.value, risk_reward
            )
            
            # Calculate return rate (the key metric!)
            rr_result = self.prob_calculator.calculate_return_rate(
                ev_result.value,
                entry_time_result.value,
                trade_duration_result.value,
                entry_result.value
            )
            
            # Collect all notes and warnings
            all_notes = []
            all_warnings = []
            
            for result in [entry_result, win_result, entry_time_result, 
                          trade_duration_result, ev_result, rr_result]:
                all_notes.extend([f"{direction.title()}: {note}" for note in result.notes])
                all_warnings.extend([f"{direction.title()}: {warn}" for warn in result.warnings])
            
            return {
                'entry_probability': entry_result.value,
                'win_probability': win_result.value,
                'entry_time': entry_time_result.value,
                'trade_duration': trade_duration_result.value,
                'expected_value': ev_result.value,
                'return_rate': rr_result.value,
                'risk_reward': risk_reward,
                'notes': all_notes,
                'warnings': all_warnings
            }
            
        except Exception as e:
            raise CalculationError(f"Failed to analyze {direction} direction: {e}")
    
    def _calculate_combined_metrics(self, analysis: TradeAnalysis) -> None:
        """Calculate combined metrics for bidirectional setups"""
        
        # Combined expected value
        ev_long = analysis.expected_value_long or 0.0
        ev_short = analysis.expected_value_short or 0.0
        analysis.expected_value_total = ev_long + ev_short
        
        # Combined return rate (weighted by probabilities)
        rr_long = analysis.return_rate_long or 0.0
        rr_short = analysis.return_rate_short or 0.0
        prob_long = analysis.prob_entry_long or 0.0
        prob_short = analysis.prob_entry_short or 0.0
        
        total_prob = prob_long + prob_short
        if total_prob > 0:
            analysis.return_rate_total = (
                (prob_long * rr_long + prob_short * rr_short) / total_prob
            )
        else:
            analysis.return_rate_total = 0.0
    
    def create_simple_setups(self, 
                           current_price: float,
                           entry_pct: float = 0.02,
                           stop_pct: float = 0.01,
                           target_pct: float = 0.03,
                           max_time_days: float = 1.0) -> Dict[str, TradeSetup]:
        """
        Create simple trade setups for quick analysis.
        
        Perfect for paper trading - creates standard setups with percentage-based levels.
        
        Parameters:
        -----------
        current_price : float
            Current asset price
        entry_pct : float
            Entry distance as percentage of current price (default 2%)
        stop_pct : float
            Stop distance as percentage of entry price (default 1%)
        target_pct : float
            Target distance as percentage of entry price (default 3%)
        max_time_days : float
            Maximum time window in days (default 1 day)
            
        Returns:
        --------
        Dict[str, TradeSetup]
            Dictionary with 'long', 'short', and 'bidirectional' setups
        """
        try:
            setups = {
                'long': create_simple_long_setup(
                    current_price, entry_pct, stop_pct, target_pct, max_time_days
                ),
                'short': create_simple_short_setup(
                    current_price, entry_pct, stop_pct, target_pct, max_time_days
                ),
                'bidirectional': create_bidirectional_setup(
                    current_price, entry_pct, stop_pct, target_pct, max_time_days
                )
            }
            
            return setups
            
        except Exception as e:
            raise TradeAnalysisError(f"Failed to create simple setups: {e}")
    
    def find_best_setup(self, 
                       setups: List[TradeSetup],
                       criterion: str = 'return_rate_total') -> Tuple[TradeSetup, TradeAnalysis]:
        """
        Find the best setup from a list based on specified criterion.
        
        Parameters:
        -----------
        setups : List[TradeSetup]
            List of setups to compare
        criterion : str
            Comparison criterion ('return_rate_total', 'expected_value_total', etc.)
            
        Returns:
        --------
        Tuple[TradeSetup, TradeAnalysis]
            Best setup and its analysis
        """
        if not setups:
            raise ValueError("No setups provided")
        
        best_setup = None
        best_analysis = None
        best_value = float('-inf')
        
        try:
            for setup in setups:
                analysis = self.analyze_trade_setup(setup)
                
                # Get comparison value
                value = getattr(analysis, criterion, None)
                if value is not None and value > best_value:
                    best_value = value
                    best_setup = setup
                    best_analysis = analysis
            
            if best_setup is None:
                raise TradeAnalysisError(f"No valid setups found for criterion '{criterion}'")
            
            return best_setup, best_analysis
            
        except Exception as e:
            raise TradeAnalysisError(f"Failed to find best setup: {e}")
    
    def compare_setups(self, 
                      setups: List[TradeSetup],
                      setup_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple trade setups in a convenient table format.
        
        Great for paper trading - helps you choose between different setups!
        
        Parameters:
        -----------
        setups : List[TradeSetup]
            List of trade setups to compare
        setup_names : List[str], optional
            Names for the setups (auto-generated if None)
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with key metrics
        """
        if not setups:
            return pd.DataFrame()
        
        if setup_names is None:
            setup_names = [f"Setup_{i+1}" for i in range(len(setups))]
        
        try:
            results = []
            
            for i, setup in enumerate(setups):
                analysis = self.analyze_trade_setup(setup, f"compare_{i}")
                
                # Extract key metrics
                row = {
                    'Setup': setup_names[i],
                    'Current_Price': setup.current_price,
                    'Entry_Long': setup.entry_long,
                    'Entry_Short': setup.entry_short,
                    'Prob_Entry_Long': analysis.prob_entry_long,
                    'Prob_Entry_Short': analysis.prob_entry_short,
                    'Win_Prob_Long': analysis.prob_win_long,
                    'Win_Prob_Short': analysis.prob_win_short,
                    'RR_Long': analysis.risk_reward_long,
                    'RR_Short': analysis.risk_reward_short,
                    'EV_Total': analysis.expected_value_total,
                    'Return_Rate_Total': analysis.return_rate_total,
                    'Best_Direction': analysis.get_best_direction(),
                    'Is_Attractive': analysis.is_attractive_setup()
                }
                
                results.append(row)
            
            df = pd.DataFrame(results)
            
            # Sort by return rate (best first)
            df = df.sort_values('Return_Rate_Total', ascending=False, na_position='last')
            
            return df
            
        except Exception as e:
            raise TradeAnalysisError(f"Failed to compare setups: {e}")
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market parameter estimates and data status"""
        
        status = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_points': len(self.data_manager.price_history),
            'current_price': self.data_manager.get_current_price(),
            'parameters_available': self.current_parameters is not None
        }
        
        if self.current_parameters:
            status.update({
                'drift': f"{self.current_parameters.drift:.2%}",
                'volatility': f"{self.current_parameters.volatility:.2%}",
                'regime': self.current_parameters.regime,
                'drift_confidence': f"{self.current_parameters.drift_confidence:.1%}",
                'volatility_confidence': f"{self.current_parameters.volatility_confidence:.1%}",
                'parameters_updated': self.current_parameters.last_updated.isoformat()
            })
        
        # Data quality info
        if self.data_manager.quality_report:
            status['data_quality_score'] = f"{self.data_manager.quality_report.quality_score:.2f}"
        
        return status
    
    def get_analysis_summary(self, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of analysis results"""
        
        if analysis_id is None:
            analysis_id = self.last_analysis_id
        
        if analysis_id is None:
            return {'error': 'No analysis available'}
        
        # Find analysis
        analysis = None
        for a in self.analysis_history:
            if a.setup_id == analysis_id:
                analysis = a
                break
        
        if analysis is None:
            return {'error': f'Analysis {analysis_id} not found'}
        
        return analysis.get_summary_stats()
    
    def export_analysis(self, 
                       analysis_id: Optional[str] = None,
                       filepath: Optional[str] = None) -> str:
        """Export analysis results to JSON file"""
        
        if analysis_id is None:
            analysis_id = self.last_analysis_id
        
        if analysis_id is None:
            raise TradeAnalysisError("No analysis to export")
        
        # Find analysis
        analysis = None
        for a in self.analysis_history:
            if a.setup_id == analysis_id:
                analysis = a
                break
        
        if analysis is None:
            raise TradeAnalysisError(f"Analysis {analysis_id} not found")
        
        # Generate filepath if not provided
        if filepath is None:
            timestamp = analysis.timestamp.strftime("%Y%m%d_%H%M%S")
            filepath = f"trade_analysis_{analysis_id}_{timestamp}.json"
        
        # Export to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    def reset(self) -> None:
        """Reset analyzer state (clear data and parameters)"""
        self.data_manager.reset()
        self.param_estimator.reset()
        self.current_parameters = None
        self.analysis_history = []
        self.last_analysis_id = None