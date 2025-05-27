"""
Symphony Forecasting System

Forecasts future performance of symphonies using:
1. Monte Carlo simulation
2. Prophet time series forecasting
3. Rolling walk-forward analysis
4. Stress testing scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Try to import Prophet (optional dependency)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

@dataclass
class ForecastResult:
    """Results from symphony forecasting"""
    forecast_returns: pd.Series
    confidence_intervals: Dict[str, pd.Series]
    expected_annual_return: float
    expected_volatility: float
    expected_sharpe: float
    worst_case_scenario: float
    best_case_scenario: float
    probability_of_beating_benchmark: float
    stress_test_results: Dict[str, float]

class SymphonyForecaster:
    """Forecast future symphony performance"""
    
    def __init__(self):
        self.available_methods = [
            'monte_carlo',
            'prophet',  
            'walk_forward',
            'stress_test'
        ]
    
    def forecast_symphony_performance(self, 
                                    backtest_results: pd.DataFrame,
                                    forecast_days: int = 252,
                                    confidence_levels: List[float] = [0.8, 0.95],
                                    benchmark_returns: pd.Series = None,
                                    method: str = 'monte_carlo') -> ForecastResult:
        """
        Forecast future symphony performance
        
        Args:
            backtest_results: Historical backtest results
            forecast_days: Number of days to forecast
            confidence_levels: Confidence intervals for forecast
            benchmark_returns: Benchmark returns for comparison
            method: Forecasting method to use
            
        Returns:
            ForecastResult with projections and analysis
        """
        
        if method not in self.available_methods:
            raise ValueError(f"Unknown method: {method}. Available: {self.available_methods}")
        
        # Ensure we have return data
        if 'portfolio_return' not in backtest_results.columns:
            raise ValueError("backtest_results must contain 'portfolio_return' column")
        
        returns = backtest_results['portfolio_return'].dropna()
        
        if len(returns) < 30:
            warnings.warn("Less than 30 observations for forecasting. Results may be unreliable.")
        
        # Route to specific forecasting method
        if method == 'monte_carlo':
            return self._monte_carlo_forecast(returns, forecast_days, confidence_levels, benchmark_returns)
        elif method == 'prophet' and PROPHET_AVAILABLE:
            return self._prophet_forecast(backtest_results, forecast_days, confidence_levels, benchmark_returns)
        elif method == 'walk_forward':
            return self._walk_forward_forecast(returns, forecast_days, confidence_levels, benchmark_returns)
        elif method == 'stress_test':
            return self._stress_test_forecast(returns, forecast_days, confidence_levels, benchmark_returns)
        else:
            # Fallback to Monte Carlo
            return self._monte_carlo_forecast(returns, forecast_days, confidence_levels, benchmark_returns)
    
    def _monte_carlo_forecast(self, returns: pd.Series, forecast_days: int, 
                            confidence_levels: List[float], benchmark_returns: pd.Series = None) -> ForecastResult:
        """Monte Carlo simulation forecasting"""
        
        # Calculate historical statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Number of simulation paths
        num_simulations = 10000
        
        # Generate random paths
        np.random.seed(42)  # For reproducibility
        
        # Simulate daily returns
        simulated_returns = np.random.normal(
            mean_return, std_return, 
            size=(num_simulations, forecast_days)
        )
        
        # Calculate cumulative returns for each path
        cumulative_paths = (1 + simulated_returns).cumprod(axis=1)
        
        # Calculate final returns for each simulation
        final_returns = cumulative_paths[:, -1] - 1
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            lower_percentile = (1 - conf_level) / 2 * 100
            upper_percentile = (1 + conf_level) / 2 * 100
            
            confidence_intervals[f'{conf_level}_lower'] = pd.Series(
                np.percentile(cumulative_paths, lower_percentile, axis=0)
            )
            confidence_intervals[f'{conf_level}_upper'] = pd.Series(
                np.percentile(cumulative_paths, upper_percentile, axis=0)
            )
        
        # Expected forecast path (median)
        forecast_returns = pd.Series(np.median(cumulative_paths, axis=0))
        
        # Calculate key metrics
        expected_annual_return = np.mean(final_returns) * (252 / forecast_days)
        expected_volatility = np.std(final_returns) * np.sqrt(252 / forecast_days)
        expected_sharpe = expected_annual_return / expected_volatility if expected_volatility > 0 else 0
        
        worst_case_scenario = np.percentile(final_returns, 5)  # 5th percentile
        best_case_scenario = np.percentile(final_returns, 95)  # 95th percentile
        
        # Probability of beating benchmark
        prob_beat_benchmark = 0.5  # Default if no benchmark
        if benchmark_returns is not None:
            benchmark_mean = benchmark_returns.mean()
            benchmark_final = (1 + benchmark_mean) ** forecast_days - 1
            prob_beat_benchmark = np.mean(final_returns > benchmark_final)
        
        # Stress test scenarios
        stress_tests = self._run_stress_tests(returns, forecast_days)
        
        return ForecastResult(
            forecast_returns=forecast_returns,
            confidence_intervals=confidence_intervals,
            expected_annual_return=expected_annual_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            worst_case_scenario=worst_case_scenario,
            best_case_scenario=best_case_scenario,
            probability_of_beating_benchmark=prob_beat_benchmark,
            stress_test_results=stress_tests
        )
    
    def _prophet_forecast(self, backtest_results: pd.DataFrame, forecast_days: int,
                         confidence_levels: List[float], benchmark_returns: pd.Series = None) -> ForecastResult:
        """Prophet time series forecasting"""
        
        if not PROPHET_AVAILABLE:
            print("Prophet not available, falling back to Monte Carlo")
            return self._monte_carlo_forecast(backtest_results['portfolio_return'], 
                                            forecast_days, confidence_levels, benchmark_returns)
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(backtest_results['date']),
            'y': (1 + backtest_results['portfolio_return']).cumprod()
        })
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=max(confidence_levels)
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast returns
        forecast_values = forecast['yhat'].iloc[-forecast_days:]
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            if conf_level == max(confidence_levels):
                confidence_intervals[f'{conf_level}_lower'] = forecast['yhat_lower'].iloc[-forecast_days:]
                confidence_intervals[f'{conf_level}_upper'] = forecast['yhat_upper'].iloc[-forecast_days:]
            else:
                # Approximate other confidence levels
                width_factor = conf_level / max(confidence_levels)
                center = forecast['yhat'].iloc[-forecast_days:]
                width = (forecast['yhat_upper'] - forecast['yhat_lower']).iloc[-forecast_days:] * width_factor / 2
                confidence_intervals[f'{conf_level}_lower'] = center - width
                confidence_intervals[f'{conf_level}_upper'] = center + width
        
        # Calculate metrics
        final_forecast_return = forecast_values.iloc[-1] / forecast_values.iloc[0] - 1
        expected_annual_return = final_forecast_return * (252 / forecast_days)
        
        # Estimate volatility from confidence intervals
        upper_bound = confidence_intervals[f'{max(confidence_levels)}_upper'].iloc[-1]
        lower_bound = confidence_intervals[f'{max(confidence_levels)}_lower'].iloc[-1]
        estimated_vol = (upper_bound - lower_bound) / (2 * 1.96) * np.sqrt(252 / forecast_days)  # Approx
        
        expected_sharpe = expected_annual_return / estimated_vol if estimated_vol > 0 else 0
        
        # Stress tests
        stress_tests = self._run_stress_tests(backtest_results['portfolio_return'], forecast_days)
        
        return ForecastResult(
            forecast_returns=forecast_values,
            confidence_intervals=confidence_intervals,
            expected_annual_return=expected_annual_return,
            expected_volatility=estimated_vol,
            expected_sharpe=expected_sharpe,
            worst_case_scenario=confidence_intervals[f'{max(confidence_levels)}_lower'].iloc[-1] / forecast_values.iloc[0] - 1,
            best_case_scenario=confidence_intervals[f'{max(confidence_levels)}_upper'].iloc[-1] / forecast_values.iloc[0] - 1,
            probability_of_beating_benchmark=0.6,  # Placeholder
            stress_test_results=stress_tests
        )
    
    def _walk_forward_forecast(self, returns: pd.Series, forecast_days: int,
                             confidence_levels: List[float], benchmark_returns: pd.Series = None) -> ForecastResult:
        """Walk-forward analysis forecast"""
        
        # Use expanding window to make predictions
        window_size = min(60, len(returns) // 2)  # Use 60 days or half the data
        
        predictions = []
        
        for i in range(window_size, len(returns)):
            # Use data up to point i
            historical_data = returns.iloc[:i]
            
            # Simple forecast: mean reversion + trend
            recent_mean = historical_data.tail(20).mean()
            long_term_mean = historical_data.mean()
            trend = historical_data.tail(10).mean() - historical_data.tail(20).mean()
            
            # Combine mean reversion and trend
            forecast = 0.6 * recent_mean + 0.3 * long_term_mean + 0.1 * trend
            predictions.append(forecast)
        
        # Use the pattern to forecast forward
        if len(predictions) > 0:
            avg_prediction = np.mean(predictions[-20:])  # Use recent predictions
            prediction_std = np.std(predictions[-20:])
        else:
            avg_prediction = returns.mean()
            prediction_std = returns.std()
        
        # Generate forecast path
        np.random.seed(42)
        forecast_path = []
        current_value = 1.0
        
        for day in range(forecast_days):
            daily_return = np.random.normal(avg_prediction, prediction_std)
            current_value *= (1 + daily_return)
            forecast_path.append(current_value)
        
        forecast_returns = pd.Series(forecast_path)
        
        # Simple confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            z_score = 1.96 if conf_level == 0.95 else 1.28  # Approximate
            error_margin = prediction_std * z_score * np.sqrt(np.arange(1, forecast_days + 1))
            
            confidence_intervals[f'{conf_level}_lower'] = forecast_returns * (1 - error_margin)
            confidence_intervals[f'{conf_level}_upper'] = forecast_returns * (1 + error_margin)
        
        # Calculate metrics
        final_return = forecast_returns.iloc[-1] - 1
        expected_annual_return = final_return * (252 / forecast_days)
        expected_volatility = prediction_std * np.sqrt(252)
        expected_sharpe = expected_annual_return / expected_volatility if expected_volatility > 0 else 0
        
        # Stress tests
        stress_tests = self._run_stress_tests(returns, forecast_days)
        
        return ForecastResult(
            forecast_returns=forecast_returns,
            confidence_intervals=confidence_intervals,
            expected_annual_return=expected_annual_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            worst_case_scenario=final_return * 0.5,  # Rough estimate
            best_case_scenario=final_return * 1.5,   # Rough estimate
            probability_of_beating_benchmark=0.55,   # Placeholder
            stress_test_results=stress_tests
        )
    
    def _run_stress_tests(self, returns: pd.Series, forecast_days: int) -> Dict[str, float]:
        """Run various stress test scenarios"""
        
        stress_scenarios = {
            'market_crash': self._simulate_market_crash(returns, forecast_days),
            'high_volatility': self._simulate_high_volatility(returns, forecast_days),
            'low_return_environment': self._simulate_low_returns(returns, forecast_days),
            'trending_market': self._simulate_trending_market(returns, forecast_days),
            'sideways_market': self._simulate_sideways_market(returns, forecast_days)
        }
        
        return stress_scenarios
    
    def _simulate_market_crash(self, returns: pd.Series, forecast_days: int) -> float:
        """Simulate market crash scenario"""
        # Assume 30% crash over 30 days, then recovery
        crash_days = min(30, forecast_days)
        crash_return_per_day = -0.3 / crash_days
        
        # Recovery phase
        recovery_days = forecast_days - crash_days
        recovery_return_per_day = returns.mean() * 1.2  # Slightly better than historical
        
        total_return = (1 + crash_return_per_day) ** crash_days * (1 + recovery_return_per_day) ** recovery_days - 1
        return total_return
    
    def _simulate_high_volatility(self, returns: pd.Series, forecast_days: int) -> float:
        """Simulate high volatility environment"""
        high_vol_returns = np.random.normal(returns.mean(), returns.std() * 2, forecast_days)
        return (1 + high_vol_returns).prod() - 1
    
    def _simulate_low_returns(self, returns: pd.Series, forecast_days: int) -> float:
        """Simulate low return environment"""
        low_returns = np.random.normal(returns.mean() * 0.3, returns.std() * 0.8, forecast_days)
        return (1 + low_returns).prod() - 1
    
    def _simulate_trending_market(self, returns: pd.Series, forecast_days: int) -> float:
        """Simulate strong trending market"""
        trend_returns = np.random.normal(returns.mean() * 1.5, returns.std(), forecast_days)
        return (1 + trend_returns).prod() - 1
    
    def _simulate_sideways_market(self, returns: pd.Series, forecast_days: int) -> float:
        """Simulate sideways market"""
        sideways_returns = np.random.normal(0, returns.std() * 0.6, forecast_days)
        return (1 + sideways_returns).prod() - 1
    
    def plot_forecast(self, forecast_result: ForecastResult, title: str = "Symphony Forecast"):
        """Plot forecast results with confidence intervals"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Forecast path with confidence intervals
        days = range(len(forecast_result.forecast_returns))
        ax1.plot(days, forecast_result.forecast_returns, 'b-', linewidth=2, label='Expected Path')
        
        # Plot confidence intervals
        for conf_level in [0.8, 0.95]:
            if f'{conf_level}_lower' in forecast_result.confidence_intervals:
                lower = forecast_result.confidence_intervals[f'{conf_level}_lower']
                upper = forecast_result.confidence_intervals[f'{conf_level}_upper']
                alpha = 0.3 if conf_level == 0.95 else 0.5
                ax1.fill_between(days, lower, upper, alpha=alpha, 
                               label=f'{conf_level*100:.0f}% Confidence')
        
        ax1.set_title(f'{title} - Forecast Path')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stress test results
        stress_scenarios = list(forecast_result.stress_test_results.keys())
        stress_returns = [forecast_result.stress_test_results[scenario] * 100 
                         for scenario in stress_scenarios]
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        bars = ax2.bar(stress_scenarios, stress_returns, color=colors[:len(stress_scenarios)])
        
        # Add value labels on bars
        for bar, value in zip(bars, stress_returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        ax2.set_title('Stress Test Scenarios')
        ax2.set_ylabel('Total Return (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def compare_forecasts(self, forecasts: Dict[str, ForecastResult]) -> pd.DataFrame:
        """Compare multiple symphony forecasts"""
        
        comparison_data = []
        
        for name, forecast in forecasts.items():
            comparison_data.append({
                'Symphony': name,
                'Expected Annual Return': f"{forecast.expected_annual_return:.2%}",
                'Expected Volatility': f"{forecast.expected_volatility:.2%}",
                'Expected Sharpe': f"{forecast.expected_sharpe:.2f}",
                'Worst Case': f"{forecast.worst_case_scenario:.2%}",
                'Best Case': f"{forecast.best_case_scenario:.2%}",
                'Prob Beat Benchmark': f"{forecast.probability_of_beating_benchmark:.2%}"
            })
        
        return pd.DataFrame(comparison_data)


# Example usage
if __name__ == "__main__":
    
    print("🔮 Symphony Forecasting System Test")
    print("=" * 50)
    
    # Create sample backtest data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic returns
    returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual return, 30% vol
    
    sample_backtest = pd.DataFrame({
        'date': dates,
        'portfolio_return': returns
    })
    
    # Test forecasting
    forecaster = SymphonyForecaster()
    
    print("Running Monte Carlo forecast...")
    forecast = forecaster.forecast_symphony_performance(
        sample_backtest, 
        forecast_days=252,  # 1 year
        method='monte_carlo'
    )
    
    print(f"\n📈 Forecast Results:")
    print(f"Expected Annual Return: {forecast.expected_annual_return:.2%}")
    print(f"Expected Volatility:    {forecast.expected_volatility:.2%}")
    print(f"Expected Sharpe Ratio:  {forecast.expected_sharpe:.2f}")
    print(f"Worst Case Scenario:    {forecast.worst_case_scenario:.2%}")
    print(f"Best Case Scenario:     {forecast.best_case_scenario:.2%}")
    print(f"Prob Beat Benchmark:    {forecast.probability_of_beating_benchmark:.2%}")
    
    print(f"\n🧪 Stress Test Results:")
    for scenario, result in forecast.stress_test_results.items():
        print(f"{scenario:20}: {result:8.2%}")
    
    # Plot results
    forecaster.plot_forecast(forecast, "Sample Symphony Forecast")
