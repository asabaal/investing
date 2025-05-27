"""
Symphony Optimization & Benchmark Comparison System

Creates optimal symphonies by:
1. Systematic parameter optimization
2. Benchmark comparison (SPY, QQQ, etc.)
3. Risk-adjusted performance analysis
4. Low-risk strategy generation
5. Performance attribution analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Import our symphony components
from symphony_engine import SymphonyEngine, SymphonyBacktester
from data_pipeline import SymphonyDataManager
from symphony_forecaster import SymphonyForecaster, ForecastResult

@dataclass
class OptimizationResult:
    """Results from symphony optimization"""
    best_symphony: dict
    best_metrics: dict
    all_results: pd.DataFrame
    benchmark_comparison: dict
    optimization_summary: dict

@dataclass
class BenchmarkComparison:
    """Benchmark comparison results"""
    symphony_metrics: dict
    benchmark_metrics: dict
    excess_return: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    correlation: float
    recommendation: str

class SymphonyOptimizer:
    """Optimize symphonies and compare against benchmarks"""
    
    def __init__(self):
        self.engine = SymphonyEngine()
        self.backtester = SymphonyBacktester()
        self.forecaster = SymphonyForecaster()
        
    def optimize_symphony_parameters(self, 
                                   base_symphony: dict,
                                   parameter_ranges: Dict[str, List],
                                   market_data: Dict[str, pd.DataFrame],
                                   start_date: str,
                                   end_date: str,
                                   benchmark_data: pd.DataFrame = None,
                                   optimization_metric: str = 'sharpe_ratio',
                                   max_combinations: int = 100) -> OptimizationResult:
        """
        Optimize symphony parameters through systematic testing
        
        Args:
            base_symphony: Base symphony configuration
            parameter_ranges: Dictionary of parameters to optimize with their ranges
            market_data: Historical market data
            start_date: Optimization start date
            end_date: Optimization end date
            benchmark_data: Benchmark for comparison
            optimization_metric: Metric to optimize ('sharpe_ratio', 'annual_return', 'calmar_ratio')
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            OptimizationResult with best symphony and analysis
        """
        
        print(f"🔧 Optimizing symphony: {base_symphony['name']}")
        print(f"📊 Optimization metric: {optimization_metric}")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges, max_combinations)
        print(f"🎯 Testing {len(param_combinations)} parameter combinations...")
        
        # Test each combination
        results = []
        
        for i, params in enumerate(param_combinations):
            try:
                # Create modified symphony
                test_symphony = self._apply_parameters(base_symphony.copy(), params)
                
                # Run backtest
                backtest_results = self.backtester.backtest(
                    test_symphony, market_data, start_date, end_date, 'monthly'
                )
                
                if backtest_results.empty:
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(backtest_results, benchmark_data)
                
                # Store result
                result = {
                    'combination_id': i,
                    'parameters': params,
                    **metrics
                }
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  ✓ Completed {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                print(f"  ❌ Error with combination {i}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful optimization runs completed")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find best combination
        best_idx = results_df[optimization_metric].idxmax()
        best_result = results_df.loc[best_idx]
        best_symphony = self._apply_parameters(base_symphony.copy(), best_result['parameters'])
        
        # Benchmark comparison
        benchmark_comparison = self._compare_with_benchmark(
            results_df.loc[best_idx], benchmark_data
        )
        
        # Optimization summary
        optimization_summary = {
            'total_combinations_tested': len(results),
            'best_metric_value': best_result[optimization_metric],
            'improvement_over_base': self._calculate_improvement(base_symphony, best_symphony, market_data, start_date, end_date),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(results_df, parameter_ranges)
        }
        
        print(f"✅ Optimization completed!")
        print(f"🏆 Best {optimization_metric}: {best_result[optimization_metric]:.4f}")
        
        return OptimizationResult(
            best_symphony=best_symphony,
            best_metrics=dict(best_result),
            all_results=results_df,
            benchmark_comparison=benchmark_comparison,
            optimization_summary=optimization_summary
        )
    
    def create_spy_plus_strategies(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 start_date: str,
                                 end_date: str) -> List[dict]:
        """
        Create low-risk strategies that aim to beat SPY with minimal additional risk
        
        Returns:
            List of conservative symphony configurations
        """
        
        print("🛡️ Creating SPY-plus (low-risk) strategies...")
        
        strategies = []
        
        # Strategy 1: SPY with momentum overlay
        spy_momentum = {
            "name": "SPY Momentum Plus",
            "description": "SPY with momentum-based tactical allocation",
            "universe": ["SPY", "QQQ", "TLT"],
            "rebalance_frequency": "monthly",
            "logic": {
                "conditions": [
                    {
                        "id": "spy_momentum_check",
                        "type": "if_statement",
                        "condition": {
                            "metric": "cumulative_return",
                            "asset_1": "SPY",
                            "operator": "greater_than",
                            "asset_2": {"type": "fixed_value", "value": 0.05},  # 5% threshold
                            "lookback_days": 90
                        },
                        "if_true": "momentum_allocation",
                        "if_false": "base_allocation"
                    }
                ],
                "allocations": {
                    "momentum_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 0.8,
                            "QQQ": 0.2
                        }
                    },
                    "base_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 0.9,
                            "TLT": 0.1
                        }
                    }
                }
            }
        }
        strategies.append(spy_momentum)
        
        # Strategy 2: SPY with volatility protection
        spy_volatility = {
            "name": "SPY Volatility Shield",
            "description": "SPY with bonds during high volatility periods",
            "universe": ["SPY", "TLT", "VIX"],
            "rebalance_frequency": "monthly",
            "logic": {
                "conditions": [
                    {
                        "id": "volatility_check",
                        "type": "if_statement",
                        "condition": {
                            "metric": "standard_deviation_return",
                            "asset_1": "SPY",
                            "operator": "less_than",
                            "asset_2": {"type": "fixed_value", "value": 0.02},  # 2% daily vol threshold
                            "lookback_days": 30
                        },
                        "if_true": "low_vol_allocation",
                        "if_false": "high_vol_allocation"
                    }
                ],
                "allocations": {
                    "low_vol_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 1.0
                        }
                    },
                    "high_vol_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 0.7,
                            "TLT": 0.3
                        }
                    }
                }
            }
        }
        strategies.append(spy_volatility)
        
        # Strategy 3: SPY with sector rotation
        spy_sector = {
            "name": "SPY Sector Rotation",
            "description": "SPY with tactical sector allocation",
            "universe": ["SPY", "XLK", "XLF", "XLE", "TLT"],
            "rebalance_frequency": "monthly",
            "logic": {
                "conditions": [
                    {
                        "id": "sector_momentum",
                        "type": "if_statement",
                        "condition": {
                            "metric": "cumulative_return",
                            "asset_1": "XLK",  # Technology
                            "operator": "greater_than",
                            "asset_2": {"type": "fixed_value", "value": 0.03},
                            "lookback_days": 60
                        },
                        "if_true": "tech_allocation",
                        "if_false": "balanced_allocation"
                    }
                ],
                "allocations": {
                    "tech_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 0.7,
                            "XLK": 0.3
                        }
                    },
                    "balanced_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "SPY": 0.8,
                            "XLF": 0.1,
                            "XLE": 0.1
                        }
                    }
                }
            }
        }
        strategies.append(spy_sector)
        
        # Test each strategy
        print("📊 Backtesting SPY-plus strategies...")
        
        validated_strategies = []
        for strategy in strategies:
            try:
                # Quick backtest to validate
                results = self.backtester.backtest(
                    strategy, market_data, start_date, end_date, 'monthly'
                )
                
                if not results.empty:
                    metrics = self._calculate_metrics(results)
                    
                    # Only include strategies that beat SPY
                    if metrics.get('annual_return', 0) > 0.08:  # Reasonable SPY proxy
                        validated_strategies.append(strategy)
                        print(f"  ✅ {strategy['name']}: {metrics['annual_return']:.2%} annual return")
                    else:
                        print(f"  ❌ {strategy['name']}: Underperformed threshold")
                        
            except Exception as e:
                print(f"  ❌ Error testing {strategy['name']}: {e}")
        
        print(f"🎯 Created {len(validated_strategies)} validated SPY-plus strategies")
        return validated_strategies
    
    def comprehensive_benchmark_comparison(self, 
                                         symphony_results: pd.DataFrame,
                                         benchmark_symbols: List[str] = ['SPY', 'QQQ', 'IWM'],
                                         market_data: Dict[str, pd.DataFrame] = None) -> Dict[str, BenchmarkComparison]:
        """
        Compare symphony against multiple benchmarks
        
        Returns:
            Dictionary of benchmark comparisons
        """
        
        print("📊 Running comprehensive benchmark comparison...")
        
        comparisons = {}
        
        # Calculate symphony metrics
        symphony_metrics = self._calculate_metrics(symphony_results)
        symphony_returns = symphony_results['portfolio_return']
        
        for benchmark in benchmark_symbols:
            try:
                if market_data and benchmark in market_data:
                    # Calculate benchmark returns
                    benchmark_data = market_data[benchmark]
                    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                    
                    # Align dates with symphony results
                    benchmark_aligned = self._align_returns(symphony_returns, benchmark_returns, symphony_results['date'])
                    
                    if benchmark_aligned is not None:
                        benchmark_metrics = self._calculate_benchmark_metrics(benchmark_aligned)
                        
                        # Calculate comparison metrics
                        comparison = self._detailed_benchmark_comparison(
                            symphony_returns, benchmark_aligned, symphony_metrics, benchmark_metrics
                        )
                        
                        comparisons[benchmark] = comparison
                        
                        print(f"  ✅ {benchmark}: Excess return = {comparison.excess_return:.2%}")
                    
            except Exception as e:
                print(f"  ❌ Error comparing with {benchmark}: {e}")
        
        return comparisons
    
    def generate_optimization_report(self, optimization_result: OptimizationResult, 
                                   output_path: str = "optimization_report.json") -> dict:
        """Generate comprehensive optimization report"""
        
        report = {
            "optimization_summary": {
                "symphony_name": optimization_result.best_symphony['name'],
                "optimization_date": pd.Timestamp.now().isoformat(),
                "total_combinations_tested": len(optimization_result.all_results),
                "best_metrics": optimization_result.best_metrics,
                "improvement_summary": optimization_result.optimization_summary
            },
            
            "best_symphony_config": optimization_result.best_symphony,
            
            "benchmark_comparison": optimization_result.benchmark_comparison,
            
            "parameter_analysis": {
                "parameter_ranges_tested": optimization_result.optimization_summary.get('parameter_sensitivity', {}),
                "top_10_combinations": optimization_result.all_results.nlargest(10, 'sharpe_ratio').to_dict('records')
            },
            
            "recommendations": self._generate_recommendations(optimization_result)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Optimization report saved: {output_path}")
        return report
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List], max_combinations: int) -> List[dict]:
        """Generate parameter combinations for testing"""
        
        # Get all parameter names and their ranges
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in selected_indices]
        
        # Convert to list of dictionaries
        combinations = []
        for combo in all_combinations:
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _apply_parameters(self, symphony: dict, parameters: dict) -> dict:
        """Apply parameter values to symphony configuration"""
        
        symphony_copy = symphony.copy()
        
        # Apply parameters to different parts of the symphony
        for param_name, param_value in parameters.items():
            if 'lookback_days' in param_name:
                # Find the condition and update lookback days
                conditions = symphony_copy['logic'].get('conditions', [])
                for condition in conditions:
                    if 'condition' in condition:
                        condition['condition']['lookback_days'] = param_value
            
            elif 'threshold' in param_name:
                # Update threshold values
                conditions = symphony_copy['logic'].get('conditions', [])
                for condition in conditions:
                    if 'condition' in condition and 'asset_2' in condition['condition']:
                        if isinstance(condition['condition']['asset_2'], dict) and condition['condition']['asset_2'].get('type') == 'fixed_value':
                            condition['condition']['asset_2']['value'] = param_value
            
            elif 'count' in param_name:
                # Update sort count
                allocations = symphony_copy['logic'].get('allocations', {})
                for allocation in allocations.values():
                    if allocation.get('type') == 'sort_and_weight' and 'sort' in allocation:
                        allocation['sort']['count'] = param_value
        
        return symphony_copy
    
    def _calculate_metrics(self, backtest_results: pd.DataFrame, benchmark_data: pd.DataFrame = None) -> dict:
        """Calculate comprehensive performance metrics"""
        
        returns = backtest_results['portfolio_return']
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_periods': len(returns)
        }
        
        return metrics
    
    def _compare_with_benchmark(self, symphony_metrics: dict, benchmark_data: pd.DataFrame) -> dict:
        """Compare symphony with benchmark"""
        
        if benchmark_data is None or benchmark_data.empty:
            return {"error": "No benchmark data available"}
        
        # Calculate benchmark metrics (simplified)
        benchmark_returns = benchmark_data.get('return', pd.Series([0.08/252]*252))  # Default 8% annual
        benchmark_annual = benchmark_returns.mean() * 252
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        comparison = {
            'symphony_annual_return': symphony_metrics.get('annual_return', 0),
            'benchmark_annual_return': benchmark_annual,
            'excess_return': symphony_metrics.get('annual_return', 0) - benchmark_annual,
            'symphony_sharpe': symphony_metrics.get('sharpe_ratio', 0),
            'benchmark_sharpe': benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0,
            'beats_benchmark': symphony_metrics.get('annual_return', 0) > benchmark_annual
        }
        
        return comparison
    
    def _calculate_improvement(self, base_symphony: dict, optimized_symphony: dict, 
                             market_data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> dict:
        """Calculate improvement from optimization"""
        
        try:
            # Test base symphony
            base_results = self.backtester.backtest(base_symphony, market_data, start_date, end_date, 'monthly')
            base_metrics = self._calculate_metrics(base_results)
            
            # Test optimized symphony
            opt_results = self.backtester.backtest(optimized_symphony, market_data, start_date, end_date, 'monthly')
            opt_metrics = self._calculate_metrics(opt_results)
            
            improvement = {
                'annual_return_improvement': opt_metrics['annual_return'] - base_metrics['annual_return'],
                'sharpe_improvement': opt_metrics['sharpe_ratio'] - base_metrics['sharpe_ratio'],
                'max_drawdown_improvement': opt_metrics['max_drawdown'] - base_metrics['max_drawdown'],
                'improvement_percentage': (opt_metrics['sharpe_ratio'] / base_metrics['sharpe_ratio'] - 1) * 100 if base_metrics['sharpe_ratio'] > 0 else 0
            }
            
            return improvement
            
        except Exception as e:
            return {"error": f"Could not calculate improvement: {e}"}
    
    def _analyze_parameter_sensitivity(self, results_df: pd.DataFrame, parameter_ranges: Dict[str, List]) -> dict:
        """Analyze which parameters have the most impact on performance"""
        
        sensitivity = {}
        
        for param_name in parameter_ranges.keys():
            if param_name in results_df.columns:
                # Calculate correlation with performance metrics
                param_values = pd.to_numeric(results_df[param_name], errors='coerce')
                sharpe_corr = param_values.corr(results_df['sharpe_ratio'])
                return_corr = param_values.corr(results_df['annual_return'])
                
                sensitivity[param_name] = {
                    'sharpe_correlation': sharpe_corr,
                    'return_correlation': return_corr,
                    'optimal_value': results_df.loc[results_df['sharpe_ratio'].idxmax(), param_name],
                    'range_tested': parameter_ranges[param_name]
                }
        
        return sensitivity
    
    def _detailed_benchmark_comparison(self, symphony_returns: pd.Series, benchmark_returns: pd.Series,
                                     symphony_metrics: dict, benchmark_metrics: dict) -> BenchmarkComparison:
        """Detailed comparison between symphony and benchmark"""
        
        # Calculate additional comparison metrics
        excess_returns = symphony_returns - benchmark_returns
        excess_return = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Beta and Alpha
        covariance = np.cov(symphony_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        alpha = symphony_metrics['annual_return'] - beta * benchmark_metrics['annual_return']
        correlation = np.corrcoef(symphony_returns, benchmark_returns)[0, 1]
        
        # Generate recommendation
        recommendation = self._generate_benchmark_recommendation(
            excess_return, information_ratio, alpha, beta, correlation
        )
        
        return BenchmarkComparison(
            symphony_metrics=symphony_metrics,
            benchmark_metrics=benchmark_metrics,
            excess_return=excess_return,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            recommendation=recommendation
        )
    
    def _generate_benchmark_recommendation(self, excess_return: float, information_ratio: float,
                                         alpha: float, beta: float, correlation: float) -> str:
        """Generate recommendation based on benchmark comparison"""
        
        if excess_return > 0.02 and information_ratio > 0.5 and alpha > 0.01:
            return "STRONG BUY - Significantly outperforms benchmark with good risk control"
        elif excess_return > 0.01 and information_ratio > 0.3:
            return "BUY - Outperforms benchmark with reasonable risk"
        elif excess_return > 0 and information_ratio > 0.1:
            return "HOLD - Slight outperformance, monitor closely"
        elif excess_return > -0.01:
            return "CAUTION - Marginal performance vs benchmark"
        else:
            return "AVOID - Underperforms benchmark significantly"
    
    def _align_returns(self, symphony_returns: pd.Series, benchmark_returns: pd.Series, dates: pd.Series) -> pd.Series:
        """Align benchmark returns with symphony dates"""
        
        try:
            # Create benchmark series with same dates
            aligned_benchmark = pd.Series(index=pd.to_datetime(dates), dtype=float)
            
            # Fill with closest benchmark returns
            for date in aligned_benchmark.index:
                closest_date = benchmark_returns.index[benchmark_returns.index <= date]
                if len(closest_date) > 0:
                    aligned_benchmark[date] = benchmark_returns[closest_date[-1]]
                else:
                    aligned_benchmark[date] = 0
            
            return aligned_benchmark.fillna(0)
            
        except Exception as e:
            print(f"Error aligning returns: {e}")
            return None
    
    def _calculate_benchmark_metrics(self, benchmark_returns: pd.Series) -> dict:
        """Calculate metrics for benchmark returns"""
        
        annual_return = benchmark_returns.mean() * 252
        volatility = benchmark_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        cumulative = (1 + benchmark_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _generate_recommendations(self, optimization_result: OptimizationResult) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance recommendations
        best_metrics = optimization_result.best_metrics
        
        if best_metrics.get('sharpe_ratio', 0) > 1.0:
            recommendations.append("Excellent risk-adjusted performance - consider for production deployment")
        elif best_metrics.get('sharpe_ratio', 0) > 0.5:
            recommendations.append("Good performance - monitor for consistency before deployment")
        else:
            recommendations.append("Performance needs improvement - consider further optimization")
        
        # Risk recommendations
        if best_metrics.get('max_drawdown', 0) < -0.20:
            recommendations.append("High drawdown risk - consider adding defensive components")
        
        # Benchmark recommendations
        benchmark_comp = optimization_result.benchmark_comparison
        if isinstance(benchmark_comp, dict) and benchmark_comp.get('beats_benchmark', False):
            recommendations.append("Beats benchmark - suitable for active strategy allocation")
        else:
            recommendations.append("Consider index fund allocation instead of active strategy")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    
    print("🔧 Symphony Optimizer Test")
    print("=" * 50)
    
    # This would normally use real market data and symphony configurations
    print("Note: This is a demonstration. In practice, you would:")
    print("1. Load real market data")
    print("2. Define parameter ranges to optimize")
    print("3. Run comprehensive optimization")
    print("4. Compare against benchmarks")
    print("5. Generate low-risk SPY-plus strategies")
    
    # Example parameter ranges for optimization
    example_parameter_ranges = {
        'lookback_days': [30, 60, 90, 120],
        'threshold': [0.01, 0.02, 0.03, 0.05],
        'sort_count': [2, 3, 4, 5]
    }
    
    print(f"\nExample parameter ranges:")
    for param, values in example_parameter_ranges.items():
        print(f"  {param}: {values}")
    
    print(f"\nTotal combinations: {np.prod([len(v) for v in example_parameter_ranges.values()])}")
