import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm
from typing import List, Dict

warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """
    Advanced risk analysis including VaR, stress testing, and volatility analysis.
    """
    def __init__(self, returns: pd.Series, prices: pd.Series):
        self.returns = returns
        self.prices = prices
        
    def calculate_var(self, 
                     confidence_level: float = 0.95, 
                     time_horizon: int = 1,
                     method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methods.
        
        Args:
            confidence_level: Confidence level for VaR calculation (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: Method to use ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            Dictionary containing VaR calculations
        """
        results = {}
        
        if time_horizon <= 0:
            raise ValueError

        if method == 'historical' or method == 'all':
            # Historical VaR
            var_percentile = 1 - confidence_level
            historical_var = np.percentile(self.returns, var_percentile * 100) * np.sqrt(time_horizon)
            results['historical_var'] = historical_var
            
        if method == 'parametric' or method == 'all':
            # Parametric VaR (assuming normal distribution)
            mean = self.returns.mean()
            std = self.returns.std()
            z_score = norm.ppf(1 - confidence_level)
            parametric_var = -(mean + z_score * std) * np.sqrt(time_horizon)
            results['parametric_var'] = parametric_var
            
        if method == 'monte_carlo' or method == 'all':
            # Monte Carlo VaR
            mean = self.returns.mean()
            std = self.returns.std()
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            mc_var = np.percentile(simulated_returns, (1 - confidence_level) * 100) * np.sqrt(time_horizon)
            results['monte_carlo_var'] = mc_var
        
        if method not in ["historical", "parametric", "monte_carlo", "all"]:
            raise ValueError

        return results
    
    def calculate_expected_shortfall(self, 
                                   confidence_level: float = 0.95, 
                                   time_horizon: int = 1) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        Args:
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            
        Returns:
            Expected Shortfall value
        """
        var_percentile = 1 - confidence_level
        threshold = np.percentile(self.returns, var_percentile * 100)
        tail_returns = self.returns[self.returns <= threshold]
        return tail_returns.mean() * np.sqrt(time_horizon)
    
    def stress_test(self, 
                   scenarios: Dict[str, float]) -> pd.DataFrame:
        """
        Perform stress testing under different scenarios.
        
        Args:
            scenarios: Dictionary of scenario names and return shocks
            
        Returns:
            DataFrame with stress test results
        """
        current_price = self.prices.iloc[-1]
        results = []
        
        for scenario_name, shock in scenarios.items():
            price_impact = current_price * (1 + shock)
            var = self.calculate_var(method='parametric')['parametric_var']
            stressed_var = var * (1 + abs(shock))  # VaR increases with volatility
            
            results.append({
                'scenario': scenario_name,
                'price_shock': shock,
                'stressed_price': price_impact,
                'price_change': price_impact - current_price,
                'normal_var': var,
                'stressed_var': stressed_var
            })
            
        return pd.DataFrame(results)
    
    def calculate_volatility_surface(self, 
                                windows: List[int] = [5, 21, 63, 252],
                                quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> pd.DataFrame:
        """
        Calculate volatility surface across different time windows and return quantiles.
        
        Step by step:
        1. For each window (e.g., 5 days):
        - Calculate rolling volatility
        - Calculate rolling returns
        - For each quantile:
            - Find that quantile's value in both the volatility and returns series
        
        Args:
            windows: List of rolling windows to calculate volatility
            quantiles: List of return quantiles to calculate
            
        Returns:
            DataFrame with columns: window, quantile, volatility, return
        """
        surface_data = []
        
        for window in windows:
            # Calculate rolling volatility for this window
            rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)  # Annualized
            
            # Calculate rolling returns (not annualized since these are cumulative returns)
            rolling_rets = self.returns.rolling(window).sum()
            
            # For each quantile, find its value in both the volatility and returns series
            for quantile in quantiles:
                vol_at_quantile = rolling_vol.quantile(quantile)
                ret_at_quantile = rolling_rets.quantile(quantile)
                
                surface_data.append({
                    'window': window,
                    'quantile': quantile,
                    'volatility': vol_at_quantile,
                    'return': ret_at_quantile
                })
        
        return pd.DataFrame(surface_data)