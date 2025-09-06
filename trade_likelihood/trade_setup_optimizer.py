"""
Trade Setup Optimizer

Intelligently optimizes trade setups by perturbing entry, stop, and target levels
while maintaining risk-reward ratio constraints. Search bounds are mathematically
derived from R:R constraints rather than arbitrary limits.

Key Innovation: Given original setup and R:R bounds, calculates the exact feasible
parameter space for each variable while keeping others constant.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import warnings
from datetime import datetime

from .data_structures import TradeSetup, TradeAnalysis
from .trade_analyzer import TradeAnalyzer


@dataclass
class OptimizationConstraints:
    """Constraints for trade setup optimization"""
    min_risk_reward: float = 3.0
    max_risk_reward: Optional[float] = None  # Will use original setup R:R if None
    preserve_direction: bool = True  # Keep long/short direction
    min_price_increment: float = 0.01  # Minimum price movement (e.g., penny increments)


@dataclass
class OptimizationResult:
    """Result from trade setup optimization"""
    original_setup: TradeSetup
    optimized_setup: TradeSetup
    original_analysis: TradeAnalysis
    optimized_analysis: TradeAnalysis
    improvement_metrics: Dict[str, float]
    optimization_path: List[Dict[str, Any]] = field(default_factory=list)
    constraints_used: OptimizationConstraints = None
    objective_function: str = "return_rate_total"


class TradeSetupOptimizer:
    """
    Optimizes trade setups by intelligently adjusting entry, stop, and target levels
    while respecting risk-reward ratio constraints.
    """
    
    def __init__(self, trade_analyzer: TradeAnalyzer):
        """
        Initialize optimizer with a configured trade analyzer
        
        Args:
            trade_analyzer: Configured TradeAnalyzer instance with market data loaded
        """
        self.analyzer = trade_analyzer
        self.optimization_history = []
        
    def calculate_feasible_bounds(self, 
                                 original_entry: float,
                                 original_stop: float, 
                                 original_target: float,
                                 direction: str,
                                 constraints: OptimizationConstraints) -> Dict[str, Tuple[float, float]]:
        """
        Calculate mathematically feasible bounds for each parameter given R:R constraints.
        
        For a trade with Entry (E), Stop (S), Target (T):
        - Long: Risk = E - S, Reward = T - E, R:R = (T - E) / (E - S)
        - Short: Risk = S - E, Reward = E - T, R:R = (E - T) / (S - E)
        
        Given min/max R:R ratios, we solve for the feasible ranges.
        """
        
        # Calculate original R:R ratio
        if direction == 'long':
            original_risk = original_entry - original_stop
            original_reward = original_target - original_entry
        else:  # short
            original_risk = original_stop - original_entry  
            original_reward = original_entry - original_target
            
        original_rr = original_reward / original_risk if original_risk > 0 else 0
        
        # Set max R:R to original if not specified
        max_rr = constraints.max_risk_reward or original_rr
        min_rr = constraints.min_risk_reward
        
        # Ensure constraints are sensible
        if max_rr < min_rr:
            max_rr = min_rr
            
        bounds = {}
        
        if direction == 'long':
            # Long trade bounds calculation
            bounds.update(self._calculate_long_bounds(
                original_entry, original_stop, original_target,
                min_rr, max_rr, constraints.min_price_increment
            ))
        else:
            # Short trade bounds calculation  
            bounds.update(self._calculate_short_bounds(
                original_entry, original_stop, original_target,
                min_rr, max_rr, constraints.min_price_increment
            ))
            
        return bounds
    
    def _calculate_long_bounds(self, entry: float, stop: float, target: float,
                              min_rr: float, max_rr: float, increment: float) -> Dict[str, Tuple[float, float]]:
        """Calculate bounds for long trade setup"""
        
        # Entry bounds: Keep stop and target fixed, vary entry
        # R:R = (T - E) / (E - S) where T > E > S
        # Solving for E: E = (T + R:R * S) / (1 + R:R)
        
        entry_min = (target + max_rr * stop) / (1 + max_rr)  # Higher R:R = lower entry
        entry_max = (target + min_rr * stop) / (1 + min_rr)  # Lower R:R = higher entry
        
        # Ensure entry stays below target and above stop
        entry_min = max(entry_min, stop + increment)
        entry_max = min(entry_max, target - increment)
        
        # Stop bounds: Keep entry and target fixed, vary stop  
        # R:R = (T - E) / (E - S), solving for S: S = E - (T - E) / R:R
        
        stop_min = entry - (target - entry) / min_rr  # Lower R:R = higher stop (tighter)
        stop_max = entry - (target - entry) / max_rr  # Higher R:R = lower stop (looser)
        
        # Ensure stop stays below entry
        stop_min = max(stop_min, 0)  # Can't be negative
        stop_max = min(stop_max, entry - increment)
        
        # Target bounds: Keep entry and stop fixed, vary target
        # R:R = (T - E) / (E - S), solving for T: T = E + R:R * (E - S)
        
        target_min = entry + min_rr * (entry - stop)  # Lower R:R = closer target
        target_max = entry + max_rr * (entry - stop)  # Higher R:R = farther target
        
        # Ensure target stays above entry
        target_min = max(target_min, entry + increment)
        
        return {
            'entry': (round(entry_min / increment) * increment, round(entry_max / increment) * increment),
            'stop': (round(stop_min / increment) * increment, round(stop_max / increment) * increment), 
            'target': (round(target_min / increment) * increment, round(target_max / increment) * increment)
        }
    
    def _calculate_short_bounds(self, entry: float, stop: float, target: float,
                               min_rr: float, max_rr: float, increment: float) -> Dict[str, Tuple[float, float]]:
        """Calculate bounds for short trade setup"""
        
        # Short trade: Risk = S - E, Reward = E - T, R:R = (E - T) / (S - E)
        # Where S > E > T (stop above entry, target below entry)
        
        # Entry bounds: Keep stop and target fixed
        # R:R = (E - T) / (S - E), solving for E: E = (R:R * S + T) / (1 + R:R)
        
        entry_min = (min_rr * stop + target) / (1 + min_rr)  # Lower R:R = lower entry
        entry_max = (max_rr * stop + target) / (1 + max_rr)  # Higher R:R = higher entry
        
        # Ensure entry stays between target and stop
        entry_min = max(entry_min, target + increment)
        entry_max = min(entry_max, stop - increment)
        
        # Stop bounds: Keep entry and target fixed
        # R:R = (E - T) / (S - E), solving for S: S = E + (E - T) / R:R
        
        stop_min = entry + (entry - target) / max_rr  # Higher R:R = tighter stop
        stop_max = entry + (entry - target) / min_rr  # Lower R:R = looser stop
        
        # Ensure stop stays above entry
        stop_min = max(stop_min, entry + increment)
        
        # Target bounds: Keep entry and stop fixed
        # R:R = (E - T) / (S - E), solving for T: T = E - R:R * (S - E)
        
        target_max = entry - min_rr * (stop - entry)  # Lower R:R = farther target
        target_min = entry - max_rr * (stop - entry)  # Higher R:R = closer target
        
        # Ensure target stays below entry
        target_min = max(target_min, 0)  # Can't be negative
        target_max = min(target_max, entry - increment)
        
        return {
            'entry': (round(entry_min / increment) * increment, round(entry_max / increment) * increment),
            'stop': (round(stop_min / increment) * increment, round(stop_max / increment) * increment),
            'target': (round(target_min / increment) * increment, round(target_max / increment) * increment)
        }
    
    def optimize_setup(self, 
                      original_setup: TradeSetup,
                      constraints: OptimizationConstraints = None,
                      objective: str = "return_rate_total",
                      method: str = "differential_evolution",
                      max_iterations: int = 1000) -> OptimizationResult:
        """
        Optimize a trade setup to maximize the specified objective.
        
        Args:
            original_setup: Starting trade setup to optimize
            constraints: Optimization constraints (R:R limits, etc.)
            objective: What to optimize ("return_rate_total", "prob_win", "expected_value_total") 
            method: Optimization algorithm ("differential_evolution", "minimize", "grid_search")
            max_iterations: Maximum optimization iterations
            
        Returns:
            OptimizationResult with original vs optimized setups and analysis
        """
        
        if constraints is None:
            constraints = OptimizationConstraints()
            
        # Determine trade direction and get parameters
        if original_setup.has_long_setup():
            direction = "long"
            entry = original_setup.entry_long
            stop = original_setup.stop_long  
            target = original_setup.target_long
        elif original_setup.has_short_setup():
            direction = "short"
            entry = original_setup.entry_short
            stop = original_setup.stop_short
            target = original_setup.target_short
        else:
            raise ValueError("Setup must have either long or short configuration")
            
        # Analyze original setup
        original_analysis = self.analyzer.analyze_trade_setup(original_setup, "original")
        
        # Calculate feasible parameter bounds
        bounds = self.calculate_feasible_bounds(entry, stop, target, direction, constraints)
        
        print(f"🎯 Optimizing {direction} setup:")
        print(f"   Original: Entry ${entry:.2f}, Stop ${stop:.2f}, Target ${target:.2f}")
        print(f"   Entry bounds: ${bounds['entry'][0]:.2f} - ${bounds['entry'][1]:.2f}")
        print(f"   Stop bounds: ${bounds['stop'][0]:.2f} - ${bounds['stop'][1]:.2f}")  
        print(f"   Target bounds: ${bounds['target'][0]:.2f} - ${bounds['target'][1]:.2f}")
        
        # Create objective function
        objective_func = self._create_objective_function(original_setup, direction, objective, constraints)
        
        # Perform optimization
        if method == "differential_evolution":
            result = self._optimize_differential_evolution(
                objective_func, bounds, max_iterations, direction, constraints
            )
        elif method == "grid_search":
            result = self._optimize_grid_search(
                objective_func, bounds, max_iterations
            )
        else:
            result = self._optimize_scipy_minimize(
                objective_func, bounds, (entry, stop, target), max_iterations
            )
        
        # Create optimized setup
        opt_entry, opt_stop, opt_target = result['optimal_params']
        optimized_setup = self._create_optimized_setup(
            original_setup, direction, opt_entry, opt_stop, opt_target
        )
        
        # Analyze optimized setup
        optimized_analysis = self.analyzer.analyze_trade_setup(optimized_setup, "optimized")
        
        # Calculate improvements
        improvements = self._calculate_improvements(original_analysis, optimized_analysis)
        
        optimization_result = OptimizationResult(
            original_setup=original_setup,
            optimized_setup=optimized_setup,
            original_analysis=original_analysis,
            optimized_analysis=optimized_analysis,
            improvement_metrics=improvements,
            optimization_path=result.get('path', []),
            constraints_used=constraints,
            objective_function=objective
        )
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def _create_objective_function(self, original_setup: TradeSetup, direction: str, 
                                  objective: str, constraints: OptimizationConstraints) -> Callable:
        """Create objective function for optimization with R:R ratio validation"""
        
        def objective_func(params):
            """Objective function to minimize (negative because we maximize)"""
            entry, stop, target = params
            
            try:
                # CRITICAL: Validate R:R ratio constraints first
                if direction == "long":
                    risk = entry - stop
                    reward = target - entry
                else:  # short
                    risk = stop - entry
                    reward = entry - target
                
                if risk <= 0:
                    return 999  # Invalid setup
                
                rr_ratio = reward / risk
                
                # Enforce R:R constraints strictly
                if rr_ratio < constraints.min_risk_reward:
                    return 999  # Penalty for violating min R:R
                
                max_rr = constraints.max_risk_reward
                if max_rr is not None and rr_ratio > max_rr:
                    return 999  # Penalty for violating max R:R
                
                # Create temporary setup only after validation
                temp_setup = self._create_optimized_setup(original_setup, direction, entry, stop, target)
                
                # Analyze setup
                analysis = self.analyzer.analyze_trade_setup(temp_setup, f"temp_{hash(params)}")
                
                # Return negative of objective (since we minimize)
                if objective == "return_rate_total":
                    value = analysis.return_rate_total or -999
                elif objective == "expected_value_total":
                    value = analysis.expected_value_total or -999
                elif objective == "prob_win":
                    if direction == "long":
                        value = analysis.prob_win_long or 0
                    else:
                        value = analysis.prob_win_short or 0
                elif objective == "combined_entry_prob":
                    value = analysis.get_combined_entry_probability() or 0
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                
                return -value  # Negative because we minimize
                
            except Exception as e:
                # Return large penalty for invalid setups
                return 999
        
        return objective_func
    
    def _optimize_differential_evolution(self, objective_func: Callable, 
                                       bounds: Dict[str, Tuple[float, float]],
                                       max_iterations: int,
                                       direction: str,
                                       constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Optimize using differential evolution (global optimizer)"""
        
        # Convert bounds to scipy format
        scipy_bounds = [
            bounds['entry'],
            bounds['stop'], 
            bounds['target']
        ]
        
        # Create constraint function for R:R ratio
        def rr_constraint(params):
            """Constraint function: returns 0 when constraint is satisfied, positive when violated"""
            entry, stop, target = params
            
            if direction == "long":
                risk = entry - stop
                reward = target - entry
            else:  # short
                risk = stop - entry
                reward = entry - target
            
            if risk <= 0:
                return 999  # Invalid setup
            
            rr_ratio = reward / risk
            
            # Return violation amount (0 = satisfied, positive = violated)
            min_violation = max(0, constraints.min_risk_reward - rr_ratio)
            max_violation = 0
            if constraints.max_risk_reward is not None:
                max_violation = max(0, rr_ratio - constraints.max_risk_reward)
            
            return min_violation + max_violation
        
        # Use constrained optimization with penalty method
        def constrained_objective(params):
            constraint_violation = rr_constraint(params)
            if constraint_violation > 0:
                return 999 + constraint_violation * 1000  # Heavy penalty for constraint violation
            return objective_func(params)
        
        result = differential_evolution(
            constrained_objective,
            scipy_bounds,
            maxiter=max_iterations // 10,  # DE is expensive
            seed=42,
            polish=True,
            atol=1e-6
        )
        
        return {
            'optimal_params': result.x,
            'optimal_value': -result.fun if result.fun < 900 else -999,  # Handle penalties
            'success': result.success and result.fun < 900,  # Success only if no constraint violation
            'iterations': result.nit,
            'path': []  # DE doesn't provide path
        }
    
    def _optimize_grid_search(self, objective_func: Callable,
                            bounds: Dict[str, Tuple[float, float]], 
                            max_iterations: int) -> Dict[str, Any]:
        """Optimize using grid search (thorough but expensive)"""
        
        # Create grids (cube root to get reasonable 3D grid)
        n_points = int(max_iterations ** (1/3))
        
        entry_grid = np.linspace(bounds['entry'][0], bounds['entry'][1], n_points)
        stop_grid = np.linspace(bounds['stop'][0], bounds['stop'][1], n_points)
        target_grid = np.linspace(bounds['target'][0], bounds['target'][1], n_points)
        
        best_params = None
        best_value = np.inf
        path = []
        
        total_combinations = len(entry_grid) * len(stop_grid) * len(target_grid)
        print(f"🔍 Grid search: {total_combinations} combinations")
        
        for i, entry in enumerate(entry_grid):
            for j, stop in enumerate(stop_grid):
                for k, target in enumerate(target_grid):
                    params = [entry, stop, target]
                    value = objective_func(params)
                    
                    path.append({
                        'params': params.copy(),
                        'value': -value,
                        'iteration': len(path)
                    })
                    
                    if value < best_value:
                        best_value = value
                        best_params = params.copy()
        
        return {
            'optimal_params': best_params,
            'optimal_value': -best_value,
            'success': True,
            'iterations': total_combinations,
            'path': path
        }
    
    def _optimize_scipy_minimize(self, objective_func: Callable,
                               bounds: Dict[str, Tuple[float, float]],
                               initial_guess: Tuple[float, float, float],
                               max_iterations: int) -> Dict[str, Any]:
        """Optimize using scipy minimize (local optimizer)"""
        
        scipy_bounds = [
            bounds['entry'],
            bounds['stop'],
            bounds['target']
        ]
        
        result = minimize(
            objective_func,
            initial_guess,
            method='L-BFGS-B',
            bounds=scipy_bounds,
            options={'maxiter': max_iterations}
        )
        
        return {
            'optimal_params': result.x,
            'optimal_value': -result.fun,
            'success': result.success, 
            'iterations': result.nit,
            'path': []  # L-BFGS-B doesn't provide detailed path
        }
    
    def _create_optimized_setup(self, original_setup: TradeSetup, direction: str,
                              entry: float, stop: float, target: float) -> TradeSetup:
        """Create new TradeSetup with optimized parameters"""
        
        if direction == "long":
            return TradeSetup(
                current_price=original_setup.current_price,
                entry_long=entry,
                stop_long=stop,
                target_long=target,
                max_time_window=original_setup.max_time_window
            )
        else:  # short
            return TradeSetup(
                current_price=original_setup.current_price,
                entry_short=entry,
                stop_short=stop, 
                target_short=target,
                max_time_window=original_setup.max_time_window
            )
    
    def _calculate_improvements(self, original: TradeAnalysis, optimized: TradeAnalysis) -> Dict[str, float]:
        """Calculate improvement metrics between original and optimized setups"""
        
        improvements = {}
        
        # Return rate improvement
        orig_rr = original.return_rate_total or 0
        opt_rr = optimized.return_rate_total or 0
        improvements['return_rate_improvement'] = opt_rr - orig_rr
        improvements['return_rate_pct_change'] = ((opt_rr - orig_rr) / abs(orig_rr)) * 100 if orig_rr != 0 else 0
        
        # Win probability improvement
        orig_win = (original.prob_win_long or 0) + (original.prob_win_short or 0)
        opt_win = (optimized.prob_win_long or 0) + (optimized.prob_win_short or 0)
        improvements['win_prob_improvement'] = opt_win - orig_win
        improvements['win_prob_pct_change'] = ((opt_win - orig_win) / orig_win) * 100 if orig_win != 0 else 0
        
        # Expected value improvement
        orig_ev = original.expected_value_total or 0
        opt_ev = optimized.expected_value_total or 0
        improvements['expected_value_improvement'] = opt_ev - orig_ev
        improvements['expected_value_pct_change'] = ((opt_ev - orig_ev) / abs(orig_ev)) * 100 if orig_ev != 0 else 0
        
        # Entry probability improvement
        orig_entry = original.get_combined_entry_probability() or 0
        opt_entry = optimized.get_combined_entry_probability() or 0
        improvements['entry_prob_improvement'] = opt_entry - orig_entry
        improvements['entry_prob_pct_change'] = ((opt_entry - orig_entry) / orig_entry) * 100 if orig_entry != 0 else 0
        
        return improvements
    
    def print_optimization_summary(self, result: OptimizationResult) -> None:
        """Print comprehensive optimization summary"""
        
        print(f"\n🎯 TRADE SETUP OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Original setup
        orig = result.original_setup
        opt = result.optimized_setup
        
        if orig.has_long_setup():
            direction = "LONG"
            print(f"Direction: {direction}")
            print(f"Original:  Entry ${orig.entry_long:.2f}, Stop ${orig.stop_long:.2f}, Target ${orig.target_long:.2f}")
            print(f"Optimized: Entry ${opt.entry_long:.2f}, Stop ${opt.stop_long:.2f}, Target ${opt.target_long:.2f}")
            
            orig_rr = orig.get_long_risk_reward() or 0
            opt_rr = opt.get_long_risk_reward() or 0
        else:
            direction = "SHORT"
            print(f"Direction: {direction}")
            print(f"Original:  Entry ${orig.entry_short:.2f}, Stop ${orig.stop_short:.2f}, Target ${orig.target_short:.2f}")
            print(f"Optimized: Entry ${opt.entry_short:.2f}, Stop ${opt.stop_short:.2f}, Target ${opt.target_short:.2f}")
            
            orig_rr = orig.get_short_risk_reward() or 0
            opt_rr = opt.get_short_risk_reward() or 0
        
        print(f"R:R Ratio: {orig_rr:.2f}:1 → {opt_rr:.2f}:1")
        
        # Analysis comparison
        orig_analysis = result.original_analysis
        opt_analysis = result.optimized_analysis
        
        print(f"\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        
        # Return rate
        orig_return = orig_analysis.return_rate_total or 0
        opt_return = opt_analysis.return_rate_total or 0
        return_change = result.improvement_metrics.get('return_rate_pct_change', 0)
        print(f"Return Rate/Day: {orig_return:.4f} → {opt_return:.4f} ({return_change:+.1f}%)")
        
        # Win probability
        orig_win = (orig_analysis.prob_win_long or 0) + (orig_analysis.prob_win_short or 0)
        opt_win = (opt_analysis.prob_win_long or 0) + (opt_analysis.prob_win_short or 0)
        win_change = result.improvement_metrics.get('win_prob_pct_change', 0)
        print(f"Win Probability: {orig_win:.1%} → {opt_win:.1%} ({win_change:+.1f}%)")
        
        # Entry probability
        orig_entry = orig_analysis.get_combined_entry_probability() or 0
        opt_entry = opt_analysis.get_combined_entry_probability() or 0
        entry_change = result.improvement_metrics.get('entry_prob_pct_change', 0)
        print(f"Entry Probability: {orig_entry:.1%} → {opt_entry:.1%} ({entry_change:+.1f}%)")
        
        # Expected value
        orig_ev = orig_analysis.expected_value_total or 0
        opt_ev = opt_analysis.expected_value_total or 0
        ev_change = result.improvement_metrics.get('expected_value_pct_change', 0)
        print(f"Expected Value: {orig_ev:.4f} → {opt_ev:.4f} ({ev_change:+.1f}%)")
        
        # Attractiveness
        orig_attractive = orig_analysis.is_attractive_setup()
        opt_attractive = opt_analysis.is_attractive_setup()
        print(f"Attractive Setup: {'✅' if orig_attractive else '❌'} → {'✅' if opt_attractive else '❌'}")
        
        print(f"\n💡 Optimized for: {result.objective_function}")
        print(f"🔧 Constraints: R:R {result.constraints_used.min_risk_reward:.1f}:1 - {result.constraints_used.max_risk_reward:.1f}:1")


def quick_optimize_dpz():
    """Quick test with DPZ trade optimization"""
    
    print("🎯 OPTIMIZING DPZ TRADE SETUP")
    print("=" * 50)
    
    # Load analyzer with DPZ data (reuse existing logic)
    from trade_likelihood.backtest_system import PaperTradeBacktester
    from trade_likelihood.trade_analyzer import TradeAnalyzer
    
    backtester = PaperTradeBacktester()
    
    # Load DPZ data up to trade date
    prices = backtester.load_market_data_from_your_db('DPZ', datetime(2025, 9, 1), 90)
    
    if len(prices) < 20:
        print("❌ Insufficient DPZ data for optimization")
        return None
    
    # Create analyzer and load data
    analyzer = TradeAnalyzer()
    analyzer.load_price_data(prices.values, prices.index)
    
    # Create original DPZ setup
    current_price = prices.iloc[-1]  # Price on Aug 31
    original_setup = TradeSetup(
        current_price=current_price,
        entry_short=464.80,
        stop_short=468.46,
        target_short=441.47,
        max_time_window=5.0
    )
    
    # Initialize optimizer
    optimizer = TradeSetupOptimizer(analyzer)
    
    # Set optimization constraints
    constraints = OptimizationConstraints(
        min_risk_reward=3.0,  # Minimum 3:1
        max_risk_reward=6.37,  # Don't exceed original
        preserve_direction=True
    )
    
    # Optimize for different objectives
    objectives = ["return_rate_total", "prob_win", "expected_value_total"]
    
    results = {}
    for objective in objectives:
        print(f"\n🔍 Optimizing for: {objective}")
        result = optimizer.optimize_setup(
            original_setup=original_setup,
            constraints=constraints,
            objective=objective,
            method="differential_evolution",
            max_iterations=1000
        )
        results[objective] = result
        optimizer.print_optimization_summary(result)
    
    return results


if __name__ == "__main__":
    results = quick_optimize_dpz()