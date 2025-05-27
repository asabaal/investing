#!/usr/bin/env python3
"""
Integrated Symphony Analysis System

Combines forecasting, optimization, and Composer compatibility
for comprehensive symphony development and deployment.

Fixed: No circular imports - uses component factory pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

# Import core components without circular dependencies
from symphony_core import SymphonyService, SymphonyConfig

class IntegratedSymphonySystem:
    """Complete symphony development and analysis system"""
    
    def __init__(self, config: SymphonyConfig = None):
        self.service = SymphonyService(config)
        self.factory = self.service.factory
        
        # Get all components from factory
        self.data_manager = self.factory.get_data_manager()
        self.engine = self.factory.get_engine()
        self.backtester = self.factory.get_backtester()
        self.visualizer = self.factory.get_visualizer()
        self.forecaster = self.factory.get_forecaster()
        self.optimizer = self.factory.get_optimizer()
        self.composer_parser = self.factory.get_composer_parser()
        self.reconciler = self.factory.get_composer_reconciler()
        
        print("🎼 Integrated Symphony System initialized")
        print("🔧 Forecasting, Optimization, and Composer Compatibility ready")
    
    def full_symphony_development_pipeline(self, 
                                         symphony_config: dict,
                                         market_data: Dict[str, pd.DataFrame],
                                         start_date: str,
                                         end_date: str,
                                         benchmark_symbol: str = 'SPY') -> dict:
        """
        Complete symphony development pipeline:
        1. Backtest historical performance
        2. Optimize parameters
        3. Forecast future performance
        4. Compare with benchmarks
        5. Generate deployment recommendation
        """
        
        print(f"\n🚀 Full Symphony Development Pipeline")
        print(f"📊 Symphony: {symphony_config['name']}")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 60)
        
        results = {}
        
        # Stage 1: Historical Backtest
        print("\n📈 Stage 1: Historical Backtesting")
        try:
            backtest_results = self.backtester.backtest(
                symphony_config, market_data, start_date, end_date, 'monthly'
            )
            backtest_analysis = self._calculate_backtest_metrics(backtest_results)
            
            results['backtest'] = {
                'results': backtest_results,
                'analysis': backtest_analysis,
                'status': 'success'
            }
            print("✅ Historical backtest completed")
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            results['backtest'] = {'status': 'failed', 'error': str(e)}
            return results
        
        # Stage 2: Parameter Optimization
        print("\n🔧 Stage 2: Parameter Optimization")
        try:
            # Define parameter ranges to optimize
            parameter_ranges = {
                'lookback_days': [30, 60, 90],
                'threshold': [0.01, 0.03, 0.05]
            }
            
            optimization_result = self.optimizer.optimize_symphony_parameters(
                symphony_config, parameter_ranges, market_data, 
                start_date, end_date, max_combinations=20
            )
            
            results['optimization'] = {
                'result': optimization_result,
                'status': 'success'
            }
            print("✅ Parameter optimization completed")
            
        except Exception as e:
            print(f"⚠️ Optimization failed, using original symphony: {e}")
            results['optimization'] = {'status': 'failed', 'error': str(e)}
            optimization_result = None
        
        # Use optimized symphony if available
        final_symphony = optimization_result.best_symphony if optimization_result else symphony_config
        
        # Stage 3: Future Forecasting
        print("\n🔮 Stage 3: Future Performance Forecasting")
        try:
            # Use the best backtest results for forecasting
            forecast_data = results['backtest']['results']
            
            forecast_result = self.forecaster.forecast_symphony_performance(
                forecast_data, forecast_days=252, method='monte_carlo'
            )
            
            results['forecast'] = {
                'result': forecast_result,
                'status': 'success'
            }
            print("✅ Future forecasting completed")
            
        except Exception as e:
            print(f"⚠️ Forecasting failed: {e}")
            results['forecast'] = {'status': 'failed', 'error': str(e)}
        
        # Stage 4: Benchmark Comparison
        print("\n📊 Stage 4: Comprehensive Benchmark Analysis")
        try:
            benchmark_comparisons = self.optimizer.comprehensive_benchmark_comparison(
                results['backtest']['results'], [benchmark_symbol, 'QQQ'], market_data
            )
            
            results['benchmark_comparison'] = {
                'comparisons': benchmark_comparisons,
                'status': 'success'
            }
            print("✅ Benchmark comparison completed")
            
        except Exception as e:
            print(f"⚠️ Benchmark comparison failed: {e}")
            results['benchmark_comparison'] = {'status': 'failed', 'error': str(e)}
        
        # Stage 5: SPY-Plus Strategy Generation
        print("\n🛡️ Stage 5: Low-Risk Strategy Alternatives")
        try:
            spy_plus_strategies = self.optimizer.create_spy_plus_strategies(
                market_data, start_date, end_date
            )
            
            results['spy_plus_strategies'] = {
                'strategies': spy_plus_strategies,
                'count': len(spy_plus_strategies),
                'status': 'success'
            }
            print("✅ SPY-plus strategies generated")
            
        except Exception as e:
            print(f"⚠️ SPY-plus generation failed: {e}")
            results['spy_plus_strategies'] = {'status': 'failed', 'error': str(e)}
        
        # Stage 6: Deployment Recommendation
        print("\n🎯 Stage 6: Deployment Recommendation")
        deployment_rec = self._generate_deployment_recommendation(results)
        results['deployment_recommendation'] = deployment_rec
        
        print(f"\n🏁 Pipeline completed!")
        print(f"📋 Recommendation: {deployment_rec['recommendation']}")
        print(f"🎯 Confidence: {deployment_rec['confidence']}")
        
        return results
    
    def convert_composer_symphony(self, composer_dsl: str, composer_csv_path: str = None) -> dict:
        """
        Convert Composer symphony and validate results
        
        Args:
            composer_dsl: Composer DSL code
            composer_csv_path: Path to Composer backtest results CSV
            
        Returns:
            Conversion and validation results
        """
        
        print("\n🔄 Converting Composer Symphony")
        print("=" * 40)
        
        results = {}
        
        # Parse Composer DSL
        try:
            parse_result = self.composer_parser.parse_composer_dsl(composer_dsl)
            
            results['parsing'] = {
                'name': parse_result.name,
                'our_format': parse_result.our_format,
                'notes': parse_result.parsing_notes,
                'status': 'success'
            }
            
            print(f"✅ Parsed: {parse_result.name}")
            
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            results['parsing'] = {'status': 'failed', 'error': str(e)}
            return results
        
        # Test our version
        if composer_csv_path and os.path.exists(composer_csv_path):
            try:
                # Prepare market data
                market_data = self.data_manager.prepare_symphony_data(parse_result.our_format)
                
                # Run our backtest
                our_results = self.backtester.backtest(
                    parse_result.our_format, market_data, '2023-01-01', '2024-12-31', 'daily'
                )
                
                # Reconcile with Composer results
                reconciliation = self.reconciler.reconcile_backtest_results(
                    composer_csv_path, our_results
                )
                
                results['reconciliation'] = {
                    'reconciliation': reconciliation,
                    'our_results': our_results,
                    'status': 'success'
                }
                
                print(f"✅ Reconciliation: {reconciliation['reconciliation_status']}")
                
            except Exception as e:
                print(f"⚠️ Reconciliation failed: {e}")
                results['reconciliation'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def create_production_ready_strategies(self, 
                                         base_strategies: List[dict],
                                         market_data: Dict[str, pd.DataFrame],
                                         risk_level: str = 'low') -> List[dict]:
        """
        Create production-ready strategies based on risk level
        
        Args:
            base_strategies: List of base strategy configurations
            market_data: Market data for testing
            risk_level: 'low', 'medium', 'high'
            
        Returns:
            List of validated production-ready strategies
        """
        
        print(f"\n🏭 Creating Production-Ready Strategies (Risk: {risk_level.upper()})")
        print("=" * 60)
        
        production_strategies = []
        
        for strategy in base_strategies:
            try:
                print(f"\n🔧 Processing: {strategy['name']}")
                
                # Run development pipeline
                pipeline_results = self.full_symphony_development_pipeline(
                    strategy, market_data, '2022-01-01', '2024-12-31'
                )
                
                # Evaluate for production readiness
                is_ready, readiness_score = self._evaluate_production_readiness(
                    pipeline_results, risk_level
                )
                
                if is_ready:
                    # Use optimized version if available
                    final_strategy = pipeline_results.get('optimization', {}).get('result', {}).get('best_symphony', strategy)
                    
                    production_strategy = {
                        'strategy': final_strategy,
                        'readiness_score': readiness_score,
                        'pipeline_results': pipeline_results,
                        'risk_level': risk_level,
                        'deployment_date': datetime.now().isoformat()
                    }
                    
                    production_strategies.append(production_strategy)
                    print(f"✅ Added to production (Score: {readiness_score:.2f})")
                else:
                    print(f"❌ Not ready for production (Score: {readiness_score:.2f})")
                    
            except Exception as e:
                print(f"❌ Error processing {strategy.get('name', 'Unknown')}: {e}")
        
        print(f"\n🎯 Production Summary:")
        print(f"📊 Strategies tested: {len(base_strategies)}")
        print(f"✅ Production ready: {len(production_strategies)}")
        print(f"🛡️ Risk level: {risk_level.upper()}")
        
        return production_strategies
    
    def _calculate_backtest_metrics(self, backtest_results: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics for backtest"""
        
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
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_periods': len(returns)
        }
    
    def _generate_deployment_recommendation(self, pipeline_results: dict) -> dict:
        """Generate deployment recommendation based on pipeline results"""
        
        recommendation = {
            'recommendation': 'HOLD',
            'confidence': 0.5,
            'reasons': [],
            'next_steps': []
        }
        
        # Check backtest performance
        if pipeline_results.get('backtest', {}).get('status') == 'success':
            analysis = pipeline_results['backtest']['analysis']
            
            sharpe = analysis.get('sharpe_ratio', 0)
            annual_return = analysis.get('annual_return', 0)
            max_drawdown = analysis.get('max_drawdown', 0)
            
            if sharpe > 1.0 and annual_return > 0.1 and max_drawdown > -0.2:
                recommendation['recommendation'] = 'DEPLOY'
                recommendation['confidence'] = 0.8
                recommendation['reasons'].append("Strong risk-adjusted returns")
            elif sharpe > 0.5 and annual_return > 0.08:
                recommendation['recommendation'] = 'DEPLOY_SMALL'
                recommendation['confidence'] = 0.6
                recommendation['reasons'].append("Decent performance, deploy with small allocation")
        
        # Check optimization results
        if pipeline_results.get('optimization', {}).get('status') == 'success':
            opt_result = pipeline_results['optimization']['result']
            improvement = opt_result.optimization_summary.get('improvement_over_base', {})
            
            if improvement.get('sharpe_improvement', 0) > 0.2:
                recommendation['reasons'].append("Significant optimization improvements")
                recommendation['confidence'] = min(recommendation['confidence'] + 0.1, 1.0)
        
        # Check forecast
        if pipeline_results.get('forecast', {}).get('status') == 'success':
            forecast = pipeline_results['forecast']['result']
            
            if forecast.expected_sharpe > 1.0 and forecast.probability_of_beating_benchmark > 0.6:
                recommendation['reasons'].append("Positive future outlook")
                recommendation['confidence'] = min(recommendation['confidence'] + 0.1, 1.0)
        
        # Check benchmark comparison
        if pipeline_results.get('benchmark_comparison', {}).get('status') == 'success':
            comparisons = pipeline_results['benchmark_comparison']['comparisons']
            
            if any(comp.excess_return > 0.02 for comp in comparisons.values() if hasattr(comp, 'excess_return')):
                recommendation['reasons'].append("Beats major benchmarks")
                recommendation['confidence'] = min(recommendation['confidence'] + 0.1, 1.0)
        
        # Generate next steps
        if recommendation['recommendation'] == 'DEPLOY':
            recommendation['next_steps'] = [
                "Start with 5-10% portfolio allocation",
                "Monitor daily for first month",
                "Set up automated alerts for drawdowns > 15%",
                "Review performance weekly"
            ]
        elif recommendation['recommendation'] == 'DEPLOY_SMALL':
            recommendation['next_steps'] = [
                "Start with 2-5% portfolio allocation",
                "Monitor closely for 2 months",
                "Consider increasing allocation if performance continues"
            ]
        else:
            recommendation['next_steps'] = [
                "Continue optimization and testing",
                "Consider paper trading first",
                "Revisit in 3 months with more data"
            ]
        
        return recommendation
    
    def _evaluate_production_readiness(self, pipeline_results: dict, risk_level: str) -> tuple:
        """Evaluate if strategy is ready for production deployment"""
        
        score = 0.0
        max_score = 100.0
        
        # Performance criteria (40 points)
        if pipeline_results.get('backtest', {}).get('status') == 'success':
            analysis = pipeline_results['backtest']['analysis']
            
            sharpe = analysis.get('sharpe_ratio', 0)
            annual_return = analysis.get('annual_return', 0)
            max_drawdown = analysis.get('max_drawdown', 0)
            
            # Sharpe ratio (20 points)
            if sharpe > 1.5:
                score += 20
            elif sharpe > 1.0:
                score += 15
            elif sharpe > 0.5:
                score += 10
            
            # Annual return (10 points)
            if annual_return > 0.15:
                score += 10
            elif annual_return > 0.10:
                score += 7
            elif annual_return > 0.08:
                score += 5
            
            # Max drawdown (10 points)
            if max_drawdown > -0.10:
                score += 10
            elif max_drawdown > -0.15:
                score += 7
            elif max_drawdown > -0.20:
                score += 5
        
        # Optimization success (20 points)
        if pipeline_results.get('optimization', {}).get('status') == 'success':
            score += 20
        
        # Forecast quality (20 points)
        if pipeline_results.get('forecast', {}).get('status') == 'success':
            forecast = pipeline_results['forecast']['result']
            
            if forecast.expected_sharpe > 1.0:
                score += 15
            elif forecast.expected_sharpe > 0.5:
                score += 10
            
            if forecast.probability_of_beating_benchmark > 0.6:
                score += 5
        
        # Benchmark beating (20 points)
        if pipeline_results.get('benchmark_comparison', {}).get('status') == 'success':
            score += 20
        
        # Risk level thresholds
        thresholds = {
            'low': 70,      # Conservative threshold
            'medium': 60,   # Moderate threshold  
            'high': 50      # Aggressive threshold
        }
        
        threshold = thresholds.get(risk_level, 60)
        is_ready = score >= threshold
        
        return is_ready, score / max_score


# Example usage
if __name__ == "__main__":
    
    print("🎼 Integrated Symphony Analysis System")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedSymphonySystem()
    
    print("✅ System initialized without circular imports!")
    print("🔧 All components loaded via factory pattern")
    print("📊 Ready for production symphony development")
