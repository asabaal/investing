"""
Bidirectional Trading Strategy Analyzer

Analyzes strategies that involve simultaneous long and short setups to capture
movement in either direction. Calculates combined probabilities, expected values,
and overall strategy performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .data_structures import TradeSetup, TradeAnalysis, MarketParameters
from .backtest_system import PaperTrade, PaperTradeBacktester, BacktestResult
from .trade_setup_optimizer import TradeSetupOptimizer, OptimizationConstraints


@dataclass
class BidirectionalSetup:
    """Complete bidirectional trading strategy setup"""
    symbol: str
    trade_date: datetime
    current_price: float
    
    # Long setup
    long_entry: float
    long_stop: float
    long_target: float
    
    # Short setup  
    short_entry: float
    short_stop: float
    short_target: float
    
    max_time_window: float = 5.0  # Days to hold both sides
    
    def get_long_rr(self) -> float:
        """Get long trade risk-reward ratio"""
        risk = self.long_entry - self.long_stop
        reward = self.long_target - self.long_entry
        return reward / risk if risk > 0 else 0
    
    def get_short_rr(self) -> float:
        """Get short trade risk-reward ratio"""  
        risk = self.short_stop - self.short_entry
        reward = self.short_entry - self.short_target
        return reward / risk if risk > 0 else 0
    
    def to_long_trade_setup(self, current_price: float = None) -> TradeSetup:
        """Convert to TradeSetup for long side"""
        price = current_price or self.current_price
        return TradeSetup(
            current_price=price,
            entry_long=self.long_entry,
            stop_long=self.long_stop,
            target_long=self.long_target,
            max_time_window=self.max_time_window
        )
    
    def to_short_trade_setup(self, current_price: float = None) -> TradeSetup:
        """Convert to TradeSetup for short side"""
        price = current_price or self.current_price
        return TradeSetup(
            current_price=price,
            entry_short=self.short_entry,
            stop_short=self.short_stop,
            target_short=self.short_target,
            max_time_window=self.max_time_window
        )


@dataclass 
class BidirectionalAnalysis:
    """Complete analysis of bidirectional strategy"""
    setup: BidirectionalSetup
    
    # Individual analyses
    long_analysis: Dict[str, TradeAnalysis]  # By historical window
    short_analysis: Dict[str, TradeAnalysis]  # By historical window
    
    # Combined probabilities
    prob_long_entry: Optional[float] = None
    prob_short_entry: Optional[float] = None
    prob_any_entry: Optional[float] = None
    prob_both_entries: Optional[float] = None
    
    # Strategy outcomes
    prob_net_win: Optional[float] = None  # Probability of overall positive outcome
    expected_net_return: Optional[float] = None
    expected_strategy_duration: Optional[float] = None
    
    # Risk metrics
    max_loss_scenario: Optional[float] = None
    max_win_scenario: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    def calculate_combined_metrics(self) -> None:
        """Calculate combined strategy metrics from individual analyses"""
        
        if not self.long_analysis or not self.short_analysis:
            return
            
        # Use best window for calculations (highest combined return rate)
        best_window = None
        best_combined_return = float('-inf')
        
        for window in self.long_analysis.keys():
            if window in self.short_analysis:
                long_return = self.long_analysis[window].return_rate_total or 0
                short_return = self.short_analysis[window].return_rate_total or 0
                combined_return = long_return + short_return
                
                if combined_return > best_combined_return:
                    best_combined_return = combined_return
                    best_window = window
        
        if not best_window:
            return
            
        # Get best analyses
        long_best = self.long_analysis[best_window]
        short_best = self.short_analysis[best_window]
        
        # Entry probabilities
        self.prob_long_entry = long_best.get_combined_entry_probability() or 0
        self.prob_short_entry = short_best.get_combined_entry_probability() or 0
        
        # Combined entry probabilities (assuming independence)
        self.prob_any_entry = (self.prob_long_entry + self.prob_short_entry - 
                              self.prob_long_entry * self.prob_short_entry)
        self.prob_both_entries = self.prob_long_entry * self.prob_short_entry
        
        # Win probabilities for each side
        long_win_prob = (long_best.prob_win_long or 0) if long_best.prob_win_long else 0
        short_win_prob = (short_best.prob_win_short or 0) if short_best.prob_win_short else 0
        
        # Strategy outcomes (considering all scenarios)
        self._calculate_strategy_outcomes(long_win_prob, short_win_prob)
        
        # Risk/reward scenarios
        self._calculate_risk_scenarios()
    
    def _calculate_strategy_outcomes(self, long_win_prob: float, short_win_prob: float):
        """Calculate net strategy win probability and expected return"""
        
        # Scenario probabilities
        p_long_only = self.prob_long_entry * (1 - self.prob_short_entry)
        p_short_only = self.prob_short_entry * (1 - self.prob_long_entry)  
        p_both = self.prob_both_entries
        p_neither = (1 - self.prob_long_entry) * (1 - self.prob_short_entry)
        
        # Calculate expected returns for each scenario
        long_risk = self.setup.long_entry - self.setup.long_stop
        long_reward = self.setup.long_target - self.setup.long_entry
        short_risk = self.setup.short_stop - self.setup.short_entry
        short_reward = self.setup.short_entry - self.setup.short_target
        
        expected_return = 0.0
        
        # Long only scenario
        long_only_return = long_win_prob * long_reward - (1 - long_win_prob) * long_risk
        expected_return += p_long_only * long_only_return
        
        # Short only scenario  
        short_only_return = short_win_prob * short_reward - (1 - short_win_prob) * short_risk
        expected_return += p_short_only * short_only_return
        
        # Both entries scenario (complex - need to consider correlation)
        both_return = self._calculate_both_entries_return(long_win_prob, short_win_prob,
                                                        long_reward, long_risk,
                                                        short_reward, short_risk)
        expected_return += p_both * both_return
        
        # Neither entry scenario (no cost, no gain)
        expected_return += p_neither * 0
        
        self.expected_net_return = expected_return
        self.prob_net_win = 1.0 if expected_return > 0 else expected_return / (abs(expected_return) + 0.01)
    
    def _calculate_both_entries_return(self, long_win_prob: float, short_win_prob: float,
                                     long_reward: float, long_risk: float,
                                     short_reward: float, short_risk: float) -> float:
        """Calculate expected return when both trades are entered"""
        
        # If both entries trigger, we have 4 possible outcomes:
        # 1. Both win: long_reward + short_reward
        # 2. Long wins, short loses: long_reward - short_risk  
        # 3. Long loses, short wins: -long_risk + short_reward
        # 4. Both lose: -long_risk - short_risk
        
        # Assuming independence of outcomes (conservative estimate)
        p_both_win = long_win_prob * short_win_prob
        p_long_win_short_lose = long_win_prob * (1 - short_win_prob)
        p_long_lose_short_win = (1 - long_win_prob) * short_win_prob
        p_both_lose = (1 - long_win_prob) * (1 - short_win_prob)
        
        return (p_both_win * (long_reward + short_reward) +
                p_long_win_short_lose * (long_reward - short_risk) +
                p_long_lose_short_win * (-long_risk + short_reward) +
                p_both_lose * (-long_risk - short_risk))
    
    def _calculate_risk_scenarios(self):
        """Calculate maximum win/loss scenarios"""
        
        long_risk = self.setup.long_entry - self.setup.long_stop
        long_reward = self.setup.long_target - self.setup.long_entry
        short_risk = self.setup.short_stop - self.setup.short_entry
        short_reward = self.setup.short_entry - self.setup.short_target
        
        # Best case: both trades win
        self.max_win_scenario = long_reward + short_reward
        
        # Worst case: both trades lose
        self.max_loss_scenario = long_risk + short_risk
        
        # Overall risk-reward ratio
        if self.max_loss_scenario > 0:
            self.risk_reward_ratio = self.max_win_scenario / self.max_loss_scenario


class BidirectionalTradeAnalyzer:
    """Analyzer for bidirectional trading strategies"""
    
    def __init__(self):
        self.backtester = PaperTradeBacktester()
    
    def analyze_bidirectional_setup(self, setup: BidirectionalSetup, 
                                   historical_windows: List[int] = [60, 90, 120]) -> BidirectionalAnalysis:
        """
        Analyze a complete bidirectional setup
        
        Args:
            setup: BidirectionalSetup with both long and short configurations
            historical_windows: List of historical data windows to test (in days)
            
        Returns:
            Complete BidirectionalAnalysis with combined metrics
        """
        
        print(f"🔄 Analyzing bidirectional strategy for {setup.symbol}")
        print(f"   Long:  Entry ${setup.long_entry:.2f}, Stop ${setup.long_stop:.2f}, Target ${setup.long_target:.2f} ({setup.get_long_rr():.2f}:1)")
        print(f"   Short: Entry ${setup.short_entry:.2f}, Stop ${setup.short_stop:.2f}, Target ${setup.short_target:.2f} ({setup.get_short_rr():.2f}:1)")
        
        long_analyses = {}
        short_analyses = {}
        
        # Analyze each historical window
        for window_days in historical_windows:
            try:
                # Load historical data
                prices = self.backtester.load_market_data_from_your_db(
                    setup.symbol, setup.trade_date, window_days
                )
                
                if len(prices) < 20:
                    print(f"⚠️ Insufficient data for {window_days}-day window")
                    continue
                
                print(f"📊 Analyzing {window_days}-day window...")
                
                # Create analyzer
                from .trade_analyzer import TradeAnalyzer
                analyzer = TradeAnalyzer()
                analyzer.load_price_data(prices.values, prices.index)
                
                # Analyze long setup
                long_setup = setup.to_long_trade_setup(prices.iloc[-1])
                long_analysis = analyzer.analyze_trade_setup(long_setup, f"long_{window_days}d")
                long_analyses[f"{window_days}d"] = long_analysis
                
                # Analyze short setup  
                short_setup = setup.to_short_trade_setup(prices.iloc[-1])
                short_analysis = analyzer.analyze_trade_setup(short_setup, f"short_{window_days}d")
                short_analyses[f"{window_days}d"] = short_analysis
                
                print(f"   Long:  Return rate {long_analysis.return_rate_total:.4f}, Win prob {(long_analysis.prob_win_long or 0):.1%}")
                print(f"   Short: Return rate {short_analysis.return_rate_total:.4f}, Win prob {(short_analysis.prob_win_short or 0):.1%}")
                
            except Exception as e:
                print(f"❌ Error analyzing {window_days}-day window: {e}")
                continue
        
        # Create combined analysis
        analysis = BidirectionalAnalysis(
            setup=setup,
            long_analysis=long_analyses,
            short_analysis=short_analyses
        )
        
        # Calculate combined metrics
        analysis.calculate_combined_metrics()
        
        return analysis
    
    def optimize_bidirectional_setup(self, setup: BidirectionalSetup, 
                                    analysis: BidirectionalAnalysis,
                                    constraints: OptimizationConstraints = None) -> Tuple[Any, Any]:
        """
        Optimize both sides of a bidirectional setup
        
        Args:
            setup: Original bidirectional setup
            analysis: Current analysis of the setup  
            constraints: Optimization constraints
            
        Returns:
            Tuple of (optimized_long_result, optimized_short_result)
        """
        
        if not analysis.long_analysis or not analysis.short_analysis:
            print("⚠️ Cannot optimize without analysis data")
            return None, None
            
        # Find best window for optimization
        best_window = None
        best_combined_return = float('-inf')
        
        for window in analysis.long_analysis.keys():
            if window in analysis.short_analysis:
                long_return = analysis.long_analysis[window].return_rate_total or 0
                short_return = analysis.short_analysis[window].return_rate_total or 0
                combined = long_return + short_return
                
                if combined > best_combined_return:
                    best_combined_return = combined
                    best_window = window
        
        if not best_window:
            print("⚠️ No valid window for optimization")
            return None, None
            
        print(f"🎯 Optimizing bidirectional setup using {best_window} data...")
        
        # Load data for optimization
        window_days = int(best_window.replace('d', ''))
        prices = self.backtester.load_market_data_from_your_db(
            setup.symbol, setup.trade_date, window_days
        )
        
        if len(prices) < 20:
            print("⚠️ Insufficient data for optimization")
            return None, None
        
        # Create fresh analyzer
        from .trade_analyzer import TradeAnalyzer
        analyzer = TradeAnalyzer()
        analyzer.load_price_data(prices.values, prices.index)
        
        # Set up constraints
        if constraints is None:
            constraints = OptimizationConstraints(
                min_risk_reward=3.0,
                max_risk_reward=None,  # Will use original R:R
                preserve_direction=True
            )
        
        # Create optimizer
        optimizer = TradeSetupOptimizer(analyzer)
        
        # Optimize long setup
        long_setup = setup.to_long_trade_setup(prices.iloc[-1])
        long_constraints = OptimizationConstraints(
            min_risk_reward=constraints.min_risk_reward,
            max_risk_reward=setup.get_long_rr(),
            preserve_direction=True
        )
        
        print("🔍 Optimizing long setup...")
        long_result = optimizer.optimize_setup(
            original_setup=long_setup,
            constraints=long_constraints,
            objective="return_rate_total",
            method="differential_evolution",
            max_iterations=500
        )
        
        # Optimize short setup
        short_setup = setup.to_short_trade_setup(prices.iloc[-1])
        short_constraints = OptimizationConstraints(
            min_risk_reward=constraints.min_risk_reward,
            max_risk_reward=setup.get_short_rr(),
            preserve_direction=True
        )
        
        print("🔍 Optimizing short setup...")
        short_result = optimizer.optimize_setup(
            original_setup=short_setup,
            constraints=short_constraints,
            objective="return_rate_total",
            method="differential_evolution", 
            max_iterations=500
        )
        
        return long_result, short_result


def quick_dpz_bidirectional_test():
    """Quick test of bidirectional analysis with DPZ setups"""
    
    print("🔄 TESTING BIDIRECTIONAL DPZ STRATEGY")
    print("=" * 50)
    
    # Create bidirectional setup
    setup = BidirectionalSetup(
        symbol="DPZ",
        trade_date=datetime(2025, 9, 1),
        current_price=458.30,  # Estimate between long and short entries
        
        # Long setup: Entry 441, Stop 435, Target 464
        long_entry=441.00,
        long_stop=435.00,
        long_target=464.00,
        
        # Short setup: Entry 464.80, Stop 468.46, Target 441.47  
        short_entry=464.80,
        short_stop=468.46,
        short_target=441.47,
        
        max_time_window=5.0
    )
    
    print(f"Setup created:")
    print(f"  Long R:R:  {setup.get_long_rr():.2f}:1") 
    print(f"  Short R:R: {setup.get_short_rr():.2f}:1")
    
    # Analyze strategy
    analyzer = BidirectionalTradeAnalyzer()
    analysis = analyzer.analyze_bidirectional_setup(setup)
    
    # Print results
    print(f"\n🎯 BIDIRECTIONAL ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Long Entry Probability:  {(analysis.prob_long_entry or 0):.1%}")
    print(f"Short Entry Probability: {(analysis.prob_short_entry or 0):.1%}")
    print(f"Any Entry Probability:   {(analysis.prob_any_entry or 0):.1%}")
    print(f"Both Entries Probability: {(analysis.prob_both_entries or 0):.1%}")
    print(f"Net Win Probability:     {(analysis.prob_net_win or 0):.1%}")
    print(f"Expected Net Return:     ${(analysis.expected_net_return or 0):.2f}")
    print(f"Max Win Scenario:        ${(analysis.max_win_scenario or 0):.2f}")
    print(f"Max Loss Scenario:       ${(analysis.max_loss_scenario or 0):.2f}")
    print(f"Strategy R:R Ratio:      {(analysis.risk_reward_ratio or 0):.2f}:1")
    
    # Run optimization
    print(f"\n🎯 RUNNING BIDIRECTIONAL OPTIMIZATION...")
    long_opt, short_opt = analyzer.optimize_bidirectional_setup(setup, analysis)
    
    if long_opt and short_opt:
        print(f"\n✅ OPTIMIZATION RESULTS:")
        print(f"Long Optimization:")
        print(f"  Original: Entry ${setup.long_entry:.2f}, Stop ${setup.long_stop:.2f}, Target ${setup.long_target:.2f}")
        print(f"  Optimized: Entry ${long_opt.optimized_setup.entry_long:.2f}, Stop ${long_opt.optimized_setup.stop_long:.2f}, Target ${long_opt.optimized_setup.target_long:.2f}")
        print(f"Short Optimization:")  
        print(f"  Original: Entry ${setup.short_entry:.2f}, Stop ${setup.short_stop:.2f}, Target ${setup.short_target:.2f}")
        print(f"  Optimized: Entry ${short_opt.optimized_setup.entry_short:.2f}, Stop ${short_opt.optimized_setup.stop_short:.2f}, Target ${short_opt.optimized_setup.target_short:.2f}")
    
    return analysis, long_opt, short_opt


if __name__ == "__main__":
    quick_dpz_bidirectional_test()