"""
Symphony Metrics Module

Calculates all metrics needed for symphony conditional logic and sorting.
Each metric function takes price data and returns a single value for comparison.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings

class SymphonyMetrics:
    """Calculate all metrics used in symphony logic"""
    
    def __init__(self):
        self.available_metrics = [
            'current_price',
            'cumulative_return', 
            'ema_price',
            'max_drawdown',
            'moving_average_price',
            'moving_average_return',
            'rsi',
            'standard_deviation_price',
            'standard_deviation_return'
        ]
    
    def calculate_metric(self, data: pd.DataFrame, metric: str, lookback_days: int, **kwargs) -> float:
        """
        Calculate a specific metric for given price data
        
        Args:
            data: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            metric: Name of metric to calculate
            lookback_days: Number of days to look back for calculation
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            float: Calculated metric value
        """
        if metric not in self.available_metrics:
            raise ValueError(f"Unknown metric: {metric}. Available: {self.available_metrics}")
        
        if len(data) < lookback_days:
            warnings.warn(f"Not enough data for {lookback_days} day lookback. Using {len(data)} days.")
            lookback_days = len(data)
        
        # Get the relevant data slice
        recent_data = data.tail(lookback_days).copy()
        
        # Route to appropriate calculation method
        method_name = f"_calculate_{metric}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(recent_data, **kwargs)
        else:
            raise NotImplementedError(f"Metric calculation not implemented: {metric}")
    
    def _calculate_current_price(self, data: pd.DataFrame, **kwargs) -> float:
        """Current closing price"""
        return float(data['Close'].iloc[-1])
    
    def _calculate_cumulative_return(self, data: pd.DataFrame, **kwargs) -> float:
        """Total return over the lookback period"""
        if len(data) < 2:
            return 0.0
        
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        return float((end_price - start_price) / start_price)
    
    def _calculate_ema_price(self, data: pd.DataFrame, **kwargs) -> float:
        """Exponential Moving Average of closing prices"""
        span = kwargs.get('span', len(data))
        ema = data['Close'].ewm(span=span).mean()
        return float(ema.iloc[-1])
    
    def _calculate_max_drawdown(self, data: pd.DataFrame, **kwargs) -> float:
        """Maximum drawdown over the period"""
        prices = data['Close']
        
        # Calculate rolling maximum (peak)
        rolling_max = prices.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (prices - rolling_max) / rolling_max
        
        # Return the maximum (most negative) drawdown
        return float(drawdown.min())
    
    def _calculate_moving_average_price(self, data: pd.DataFrame, **kwargs) -> float:
        """Simple moving average of closing prices"""
        return float(data['Close'].mean())
    
    def _calculate_moving_average_return(self, data: pd.DataFrame, **kwargs) -> float:
        """Moving average of daily returns"""
        if len(data) < 2:
            return 0.0
        
        daily_returns = data['Close'].pct_change().dropna()
        return float(daily_returns.mean())
    
    def _calculate_rsi(self, data: pd.DataFrame, **kwargs) -> float:
        """Relative Strength Index"""
        period = kwargs.get('period', 14)
        
        if len(data) < period + 1:
            period = max(2, len(data) - 1)
        
        prices = data['Close']
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_standard_deviation_price(self, data: pd.DataFrame, **kwargs) -> float:
        """Standard deviation of closing prices"""
        return float(data['Close'].std())
    
    def _calculate_standard_deviation_return(self, data: pd.DataFrame, **kwargs) -> float:
        """Standard deviation of daily returns (volatility)"""
        if len(data) < 2:
            return 0.0
        
        daily_returns = data['Close'].pct_change().dropna()
        return float(daily_returns.std())
    
    def calculate_all_metrics(self, data: pd.DataFrame, lookback_days: int = 30) -> dict:
        """
        Calculate all available metrics for given data
        
        Returns:
            dict: {metric_name: calculated_value}
        """
        results = {}
        
        for metric in self.available_metrics:
            try:
                results[metric] = self.calculate_metric(data, metric, lookback_days)
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                results[metric] = None
        
        return results


class WeightingStrategies:
    """Different weighting methods for portfolio allocation"""
    
    @staticmethod
    def equal_weight(symbols: list, data_dict: dict = None) -> dict:
        """Equal weight allocation across all symbols"""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    @staticmethod
    def specified_weight(weights: dict) -> dict:
        """Use specified weights (must sum to 1.0)"""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            # Normalize if weights don't sum to 1
            return {k: v/total for k, v in weights.items()}
        return weights
    
    @staticmethod 
    def inverse_volatility_weight(symbols: list, data_dict: dict, lookback_days: int = 30) -> dict:
        """Weight inversely proportional to volatility"""
        metrics_calc = SymphonyMetrics()
        volatilities = {}
        
        # Calculate volatility for each symbol
        for symbol in symbols:
            if symbol in data_dict:
                vol = metrics_calc.calculate_metric(
                    data_dict[symbol], 
                    'standard_deviation_return', 
                    lookback_days
                )
                # Avoid division by zero
                volatilities[symbol] = max(vol, 0.001)
            else:
                volatilities[symbol] = 0.1  # Default volatility
        
        # Calculate inverse volatility weights
        inv_vols = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize to sum to 1.0
        weights = {symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_vols.items()}
        
        return weights


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Simulate price data with some trend and volatility
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0.1, 1.5, 100))
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.normal(0, 0.5, 100),
        'High': prices + np.random.normal(1, 0.5, 100),
        'Low': prices + np.random.normal(-1, 0.5, 100), 
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    sample_data.set_index('Date', inplace=True)
    
    # Test metrics calculation
    metrics = SymphonyMetrics()
    
    print("Testing Symphony Metrics Calculator")
    print("=" * 40)
    
    # Test individual metrics
    test_metrics = [
        ('current_price', 1),
        ('cumulative_return', 30),
        ('moving_average_price', 20),
        ('rsi', 14),
        ('standard_deviation_return', 30)
    ]
    
    for metric_name, lookback in test_metrics:
        try:
            value = metrics.calculate_metric(sample_data, metric_name, lookback)
            print(f"{metric_name:25} ({lookback:2d} days): {value:8.4f}")
        except Exception as e:
            print(f"{metric_name:25} ({lookback:2d} days): ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("All metrics for 30-day lookback:")
    all_metrics = metrics.calculate_all_metrics(sample_data, 30)
    for metric, value in all_metrics.items():
        if value is not None:
            print(f"{metric:25}: {value:8.4f}")
    
    # Test weighting strategies
    print("\n" + "=" * 40)
    print("Testing Weighting Strategies:")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    equal_weights = WeightingStrategies.equal_weight(symbols)
    print(f"Equal weights: {equal_weights}")
    
    specified_weights = WeightingStrategies.specified_weight({'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': 0.2})
    print(f"Specified weights: {specified_weights}")
