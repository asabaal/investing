import warnings

import numpy as np
import pandas as pd

from hmmlearn import hmm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')

class RegimeAnalyzer(BaseEstimator, TransformerMixin):
    """
    Market regime detection using multiple methods including HMM, structural breaks,
    and trend analysis.
    
    Parameters
    ----------
    n_states : int, default=3
        Number of distinct market states to identify
    window : int, default=252
        Window size for structural break and trend detection
    regime_labels : dict, optional
        Dictionary mapping state indices to labels. If not provided,
        will use default labeling based on mean values
    standardize : bool, default=True
        Whether to standardize features before fitting HMM
    hmm_params : dict, optional
        Additional parameters to pass to GaussianHMM
    detection_methods : list of str, default=['hmm', 'breaks', 'trend']
        Which detection methods to use
        
    Attributes
    ----------
    model_ : GaussianHMM
        The fitted Hidden Markov Model (only if 'hmm' in detection_methods)
    scaler_ : StandardScaler
        The fitted scaler if standardize=True
    feature_names_ : list
        Names of features used in fitting
    converged_ : bool
        Whether the HMM model converged
    """
    
    def __init__(
        self,
        n_states: int = 3,
        window: int = 252,
        regime_labels: Optional[Dict[int, str]] = None,
        standardize: bool = True,
        hmm_params: Optional[dict] = None,
        detection_methods: List[str] = ['hmm', 'breaks', 'trend']
    ):
        self.n_states = n_states
        self.window = window
        self.regime_labels = regime_labels
        self.standardize = standardize
        self.hmm_params = hmm_params or {}
        self.detection_methods = detection_methods
    
    def fit(self, X: pd.DataFrame, y=None) -> 'RegimeAnalyzer':
        """
        Fit the regime detection models.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to use for regime detection. Must include required columns
            depending on detection_methods:
            - 'hmm': any features
            - 'breaks': must include 'returns' and 'volume'
            - 'trend': must include 'price'
        y : None
            Ignored, present for sklearn API compatibility
            
        Returns
        -------
        self : MarketRegimeDetector
            The fitted detector
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        # Validate columns for each method
        if 'breaks' in self.detection_methods:
            if not {'returns', 'volume'}.issubset(X.columns):
                raise ValueError(
                    "X must contain 'returns' and 'volume' columns for break detection"
                )
        
        if 'trend' in self.detection_methods:
            if 'price' not in X.columns:
                raise ValueError(
                    "X must contain 'price' column for trend detection"
                )
        
        if 'hmm' in self.detection_methods:
            self._fit_hmm(X)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to detect regimes using all configured methods.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to detect regimes for. Must include same columns as
            used in fit()
            
        Returns
        -------
        pd.DataFrame
            DataFrame with regime indicators from all methods
        """
        results = pd.DataFrame(index=X.index)
        
        if 'hmm' in self.detection_methods:
            hmm_results = self._transform_hmm(X)
            results = pd.concat([results, hmm_results], axis=1)
            
        if 'breaks' in self.detection_methods:
            break_results = self._detect_breaks(X)
            results = pd.concat([results, break_results], axis=1)
            
        if 'trend' in self.detection_methods:
            trend_results = self._detect_trend(X)
            results = pd.concat([results, trend_results], axis=1)
        
        if len(self.detection_methods) > 1:
            results['composite_regime'] = self._compute_composite_regime(results)
        
        return results
    
    def _fit_hmm(self, X: pd.DataFrame):
        """Fit the HMM model to the data."""
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        # Prepare data
        X_prep = X.copy()
        X_prep = X_prep.fillna(method='ffill').fillna(method='bfill')
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_values = self.scaler_.fit_transform(X_prep)
        else:
            X_values = X_prep.values
            
        # Initialize and fit HMM
        hmm_defaults = {
            'n_components': self.n_states,
            'covariance_type': 'full',
            'n_iter': 100
        }
        hmm_params = {**hmm_defaults, **self.hmm_params}
        
        self.model_ = hmm.GaussianHMM(**hmm_params)
        self.model_.fit(X_values)
        
        # Store convergence status
        self.converged_ = self.model_.monitor_.converged
    
    def _transform_hmm(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply HMM to detect regimes."""
        if not hasattr(self, 'model_'):
            raise ValueError("Must fit HMM before transform")
        
        # Prepare data
        X_prep = X.copy()
        X_prep = X_prep.fillna(method='ffill').fillna(method='bfill')
        
        if self.standardize:
            X_values = self.scaler_.transform(X_prep)
        else:
            X_values = X_prep.values
        
        # Get state sequence and probabilities
        hidden_states = self.model_.predict(X_values)
        state_probs = self.model_.predict_proba(X_values)
        
        # Create results DataFrame
        results = pd.DataFrame(index=X.index)
        results['regime'] = hidden_states
        
        # Add probability for each regime
        for i in range(self.n_states):
            results[f'regime_{i}_prob'] = state_probs[:, i]
        
        # Add regime labels
        if not self.converged_:
            results['regime_type'] = 'unknown'
        else:
            if self.regime_labels is None:
                # Default labeling based on mean values
                means = pd.DataFrame(
                    self.model_.means_,
                    columns=self.feature_names_
                )
                state_chars = {}
                
                for state in range(self.n_states):
                    features_above_mean = (
                        means.loc[state] > means.mean()
                    ).sum()
                    
                    if features_above_mean > len(self.feature_names_) / 2:
                        state_chars[state] = 'bullish'
                    elif features_above_mean < len(self.feature_names_) / 2:
                        state_chars[state] = 'bearish'
                    else:
                        state_chars[state] = 'neutral'
            else:
                state_chars = self.regime_labels
            
            results['regime_type'] = results['regime'].map(state_chars)
        
        return results
    
    def _detect_breaks(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect structural breaks using rolling statistical tests with proper initialization."""
        results = pd.DataFrame(index=X.index)
        results['break_chow'] = np.nan
        results['break_volatility'] = np.nan
        results['break_volume'] = np.nan
        
        returns = X['returns']
        volumes = X['volume']
        
        min_required = 3  # Minimum periods needed for each half (pre/post) to calculate mean and std
        
        for i in range(len(returns)):
            if i < min_required:
                # Not enough data for even a minimal comparison
                continue
                
            # Use expanding window for initial periods
            window_size = min(i + 1, self.window)
            start_idx = max(0, i - window_size + 1)
            mid_point = start_idx + window_size // 2
            
            window_returns = returns.iloc[start_idx:i+1]
            window_volumes = volumes.iloc[start_idx:i+1]
            
            # Divide window into pre and post periods
            pre_returns = window_returns[:mid_point-start_idx]
            post_returns = window_returns[mid_point-start_idx:]
            pre_volumes = window_volumes[:mid_point-start_idx]
            post_volumes = window_volumes[mid_point-start_idx:]
            
            # Calculate break statistics using available data
            if len(pre_returns) >= min_required and len(post_returns) >= min_required:
                # Test for breaks in mean using Chow test-like approach
                pre_mean = pre_returns.mean()
                post_mean = post_returns.mean()
                mean_diff = abs(pre_mean - post_mean)
                mean_std = window_returns.std()
                if mean_std > 0:  # Avoid division by zero
                    results.iloc[i, 0] = mean_diff / mean_std
                
                # Test for breaks in volatility
                pre_vol = pre_returns.std()
                post_vol = post_returns.std()
                if min(pre_vol, post_vol) > 0:  # Avoid division by zero
                    vol_ratio = max(pre_vol, post_vol) / min(pre_vol, post_vol)
                    results.iloc[i, 1] = vol_ratio
                
                # Test for breaks in volume
                pre_vol_mean = pre_volumes.mean()
                post_vol_mean = post_volumes.mean()
                if min(pre_vol_mean, post_vol_mean) > 0:  # Avoid division by zero
                    vol_mean_ratio = max(pre_vol_mean, post_vol_mean) / min(pre_vol_mean, post_vol_mean)
                    results.iloc[i, 2] = vol_mean_ratio
        
        # Identify significant breaks
        results['significant_break'] = (
            (results['break_chow'] > 2) |  # 2 std dev threshold
            (results['break_volatility'] > 2) |  # Double volatility
            (results['break_volume'] > 2)  # Double volume
        ).fillna(False)  # Handle NaN values explicitly
        
        return results
    
    def _detect_trend(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect price trends using moving averages."""
        results = pd.DataFrame(index=X.index)
        prices = X['price']
        
        # Initialize arrays for our moving averages
        long_ma = np.zeros(len(prices))
        short_ma = np.zeros(len(prices))
        
        # Calculate moving averages with growing windows for initial periods
        for i in range(len(prices)):
            if i < self.window:
                # Use expanding window for periods less than window size
                long_ma[i] = prices[:i+1].mean()
                short_ma[i] = prices[max(0, i-self.window//2+1):i+1].mean()
            else:
                # Use fixed window once we have enough data
                long_ma[i] = prices[i-self.window+1:i+1].mean()
                short_ma[i] = prices[i-self.window//2+1:i+1].mean()
        
        # Compare short-term MA to long-term MA
        # Short MA > Long MA indicates uptrend
        results['trend'] = np.where(
            short_ma > long_ma,
            'uptrend', 'downtrend'
        )
        
        return results
    
    def _compute_composite_regime(self, results: pd.DataFrame) -> pd.Series:
        """Compute composite regime by combining all indicators."""
        def get_composite_regime(row):
            components = []
            
            if 'regime_type' in row:
                components.append(row['regime_type'])
            
            if 'significant_break' in row and row['significant_break']:
                components.append('transition')
            
            if 'trend' in row:
                components.append(row['trend'])
            
            return '_'.join(components)
        
        return results.apply(get_composite_regime, axis=1)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)