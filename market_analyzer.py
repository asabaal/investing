"""
Claude chat:
https://claude.ai/chat/e57d8498-85ed-478a-9aa4-a5dcba070116
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from arch import arch_model
import warnings
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
import networkx as nx
from hmmlearn import hmm
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

@dataclass
class TechnicalPattern:
    """
    Data class to store information about detected technical patterns in price data.
    
    Technical patterns are specific price formations that traders use to identify
    potential future price movements. Each pattern has characteristics like its
    type, location in the data, and confidence level.

    Attributes:
    -----------
    pattern_type : str
        The type of pattern detected (e.g., "HEAD_AND_SHOULDERS", "DOUBLE_BOTTOM")
    start_idx : int
        Index in the price series where the pattern begins
    end_idx : int
        Index in the price series where the pattern ends
    confidence : float
        A measure between 0 and 1 indicating how well the pattern matches ideal criteria
        Example: 0.8 means the pattern is a strong match, 0.3 suggests a weak match
    price_range : Tuple[float, float]
        The price range covered by the pattern (min_price, max_price)
    failure_reasons : Optional[Dict[str, str]]
        If confidence < 1.0, explains why the pattern isn't perfect
        Keys are check names, values are descriptions of what isn't ideal
    specific_points : Optional[dict]
        Dictionary containing pattern-specific point indices and values
        For example, for head and shoulders, includes shoulder and head points

    Example:
    --------
    >>> pattern = TechnicalPattern(
    ...     pattern_type="DOUBLE_BOTTOM",
    ...     start_idx=100,
    ...     end_idx=150,
    ...     confidence=0.85,
    ...     price_range=(100.0, 110.0)
    ... )
    >>> print(f"Found {pattern.pattern_type} with {pattern.confidence:.1%} confidence")
    "Found DOUBLE_BOTTOM with 85.0% confidence"        
    """
    pattern_type: str
    start_idx: int
    end_idx: int
    price_range: Tuple[float, float]
    failure_reasons: Optional[Dict[str, str]] = None
    specific_points: Optional[dict] = None
    volume_range: Optional[Tuple[float, float]] = None
    sub_classification: Optional[Enum] = None
    confidence: Optional[float] = np.nan


    def __post_init__(self):
        # Convert numpy integers to Python integers
        self.start_idx = int(self.start_idx)
        self.end_idx = int(self.end_idx)
        # Convert numpy numbers in price_range to Python numbers
        self.price_range = (float(self.price_range[0]), float(self.price_range[1]))
        # Convert confidence to float if needed
        self.confidence = float(self.confidence)

@dataclass
class HeadAndShouldersPoints:
    left_shoulder_idx: int
    head_idx: int
    right_shoulder_idx: int
    left_trough_idx: int
    right_trough_idx: int
    
@dataclass
class PatternValidation:
    is_valid: bool
    confidence: float
    failure_reasons: Dict[str, str]  # Key is check name, value is failure description
    price_range: Optional[Tuple[float, float]] = None


class VolumePatternType(Enum):
    DIVERGENCE = "DIVERGENCE"          # Volume actively moving opposite to price
    NON_CONFIRMATION = "NON_CONFIRMATION"  # Volume flat while price moves
    VOLUME_FORCE = "VOLUME_FORCE"      # Volume moving while price is flat
    NEUTRAL = "NEUTRAL"                # Both price and volume are flat
    CONCORDANT = "CONCORDANT"          # Price is moving in same direction as volume

@dataclass
class VolumePattern(TechnicalPattern):
    pattern_type: VolumePatternType
    price_range: Tuple[float, float]
    volume_range: Tuple[float, float]

def validate_head_and_shoulders(
    prices: np.ndarray,
    points: HeadAndShouldersPoints,
    shoulder_height_tolerance: float = 0.02,
    neckline_slope_tolerance: float = 0.02
) -> PatternValidation:
    """
    Validate whether given points form a head and shoulders pattern.
    
    Args:
        prices: Array of price values
        points: HeadAndShouldersPoints containing indices of potential pattern points
        shoulder_height_tolerance: Maximum allowed difference between shoulder heights as percentage
        neckline_slope_tolerance: Maximum allowed neckline slope as percentage
    
    Returns:
        PatternValidation object containing validation results and details
    """
    # Extract prices at pattern points
    left_shoulder = prices[points.left_shoulder_idx]
    head = prices[points.head_idx]
    right_shoulder = prices[points.right_shoulder_idx]
    left_trough = prices[points.left_trough_idx]
    right_trough = prices[points.right_trough_idx]
    
    failure_reasons = {}
    
    # Check 1: Head must be higher than both shoulders
    head_height_valid = head > left_shoulder and head > right_shoulder
    if not head_height_valid:
        failure_reasons['head_height'] = 'Head is not higher than both shoulders'
    
    # Check 2: Shoulders should be at similar heights
    shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
    shoulders_valid = shoulder_diff <= shoulder_height_tolerance
    if not shoulders_valid:
        failure_reasons['shoulder_heights'] = f'Shoulder height difference ({shoulder_diff:.1%}) exceeds tolerance ({shoulder_height_tolerance:.1%})'
    
    # Check 3: Neckline should be roughly horizontal
    neckline_slope = abs(right_trough - left_trough) / left_trough
    neckline_valid = neckline_slope <= neckline_slope_tolerance
    if not neckline_valid:
        failure_reasons['neckline_slope'] = f'Neckline slope ({neckline_slope:.1%}) exceeds tolerance ({neckline_slope_tolerance:.1%})'
    
    # Check 4: Pattern sequence should be valid
    sequence_valid = (points.left_shoulder_idx < points.left_trough_idx < 
                     points.head_idx < points.right_trough_idx < 
                     points.right_shoulder_idx)
    if not sequence_valid:
        failure_reasons['sequence'] = 'Points are not in correct chronological order'
    
    # Calculate confidence based on how well the pattern matches ideal conditions
    confidence_factors = {
        'head_height': 1.0 if head_height_valid else 0.0,
        'shoulder_symmetry': 1.0 - (shoulder_diff / shoulder_height_tolerance) if shoulders_valid else 0.0,
        'neckline': 1.0 - (neckline_slope / neckline_slope_tolerance) if neckline_valid else 0.0,
        'sequence': 1.0 if sequence_valid else 0.0
    }
    confidence = np.mean(list(confidence_factors.values()))
    
    is_valid = len(failure_reasons) == 0
    price_range = (min(left_trough, right_trough), head) if is_valid else None
    
    return PatternValidation(
        is_valid=is_valid,
        confidence=confidence,
        failure_reasons=failure_reasons,
        price_range=price_range
    )

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

class MarketVisualizer:
    """
    Creates interactive visualizations for market analysis results.
    """
    def __init__(self, data: Dict[str, pd.DataFrame], results: Dict[str, Any]):
        self.data = data
        self.results = results

        if "returns" not in results:
            # Calculate returns for all assets
            self.returns = pd.DataFrame({
                symbol: df['close'].pct_change()
                for symbol, df in self.data.items()
            })        
        
    def plot_price_patterns(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> go.Figure:
        """
        Create candlestick chart with pattern annotations.
        
        Args:
            symbol: Stock symbol to plot
            start_date: Start date for plotting
            end_date: End date for plotting
            
        Returns:
            Plotly figure object
        """
        df = self.data[symbol].copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Create candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume'
        ), row=2, col=1)

        # Add patterns if available
        if 'patterns' in self.results:
            patterns = self.results['patterns']
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    # Add pattern annotation
                    fig.add_shape(
                        type="rect",
                        x0=self.data[symbol].index[pattern.start_idx],
                        x1=self.data[symbol].index[pattern.end_idx],
                        y0=pattern.price_range[0],
                        y1=pattern.price_range[1],
                        line=dict(color="rgba(255, 0, 0, 0.3)"),
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        row=1, col=1
                    )
                    
                    # Add pattern label
                    fig.add_annotation(
                        x=df.index[pattern.start_idx],
                        y=pattern.price_range[1],
                        text=pattern_type,
                        showarrow=True,
                        arrowhead=1
                    )
        
        fig.update_layout(
            title=f"{symbol} Price and Volume with Pattern Detection",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume",
            showlegend=True
        )
        
        return fig
    
    def plot_regimes(self, symbol: str, regime_type: str = 'combined') -> go.Figure:
        """
        Visualize regime analysis results.
        
        Args:
            symbol: Stock symbol to plot
            regime_type: Type of regime analysis to visualize
            
        Returns:
            Plotly figure object
        """
        if regime_type not in self.results.get('regimes', {}):
            raise ValueError(f"Regime type {regime_type} not found in results")
  
        regime_data = self.results['regimes'][f'{regime_type}']
        price_data = self.data[symbol]['close']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(x=price_data.index, y=price_data, name="Price"),
            secondary_y=False
        )
        
        # Add regime indicators
        if 'regime' in regime_data.columns:
            regime_numeric = pd.Categorical(regime_data['regime']).codes
            fig.add_trace(
                go.Scatter(x=regime_data.index, y=regime_numeric, 
                          name="Regime", line=dict(dash='dot')),
                secondary_y=True
            )
            
        # Add regime probability if available
        prob_cols = [col for col in regime_data.columns if 'prob' in col]
        for col in prob_cols:
            fig.add_trace(
                go.Scatter(x=regime_data.index, y=regime_data[col],
                          name=col, line=dict(dash='dot')),
                secondary_y=True
            )
            
        fig.update_layout(
            title=f"{symbol} Price and {regime_type.capitalize()} Regimes",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Regime/Probability",
            showlegend=True
        )
        
        return fig
    
    def plot_network(self, min_correlation: float = 0.5) -> go.Figure:
        """
        Create network visualization of stock relationships.
        
        Args:
            min_correlation: Minimum correlation to show relationship
            
        Returns:
            Plotly figure object
        """
        if 'relationship_network' not in self.results:
            raise ValueError("Relationship network not found in results")
            
        G = self.results['relationship_network']
        
        # Get node positions using Fruchterman-Reingold force-directed algorithm
        pos = nx.spring_layout(G)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Correlation: {edge[2]['weight']:.2f}")
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            ))
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
            
        node_trace.marker.color = node_adjacencies
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Stock Relationship Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def plot_lead_lag_heatmap(self) -> go.Figure:
        """
        Create heatmap of lead-lag relationships.
        
        Returns:
            Plotly figure object
        """
        if 'cross_correlations' not in self.results:
            raise ValueError("Cross-correlation results not found")
            
        # Pivot the cross-correlation results
        corr_data = self.results['cross_correlations']
        heatmap_data = pd.pivot_table(
            corr_data,
            values='correlation',
            index='symbol1',
            columns='lag'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(heatmap_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False))
        
        fig.update_layout(
            title='Lead-Lag Correlation Heatmap',
            xaxis_title='Lag',
            yaxis_title='Symbol',
            height=800
        )
        
        return fig
    
    def plot_volatility_surface(self) -> go.Figure:
        """
        Create 3D visualization of volatility surface.
        
        Returns:
            Plotly figure object
        """
        surface_data = self.results['volatility_surface']
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=surface_data['window'].unique(),
            y=surface_data['quantile'].unique(),
            z=surface_data.pivot(index='quantile', columns='window', values='volatility').values,
            colorscale='Viridis'
        )])
        
        # Update layout for better visualization
        fig.update_layout(
            title='Volatility Surface',
            scene=dict(
                xaxis_title='Time Window (days)',
                yaxis_title='Quantile',
                zaxis_title='Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def plot_risk_metrics(self) -> go.Figure:
        """
        Create visualization of risk metrics including VaR and stress tests.
        
        Returns:
            Plotly figure object
        """
        risk_data = self.results.get('risk_metrics', {})
        stress_data = self.results.get('stress_test', pd.DataFrame())
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('VaR Comparison', 'Expected Shortfall', 
                          'Stress Test Scenarios', 'Return Distribution')
        )
        
        # Plot VaR comparison
        if 'var' in risk_data:
            var_methods = list(risk_data['var'].keys())
            var_values = list(risk_data['var'].values())
            fig.add_trace(
                go.Bar(x=var_methods, y=var_values, name='VaR'),
                row=1, col=1
            )
            
        # Plot Expected Shortfall
        if 'expected_shortfall' in risk_data:
            es_data = pd.Series(risk_data['expected_shortfall'])
            fig.add_trace(
                go.Scatter(x=es_data.index, y=es_data.values, 
                          mode='lines+markers', name='ES'),
                row=1, col=2
            )
            
        # Plot stress test results
        if not stress_data.empty:
            fig.add_trace(
                go.Bar(x=stress_data['scenario'], 
                      y=stress_data['price_change'],
                      name='Price Impact'),
                row=2, col=1
            )
            
        # Plot return distribution
        returns = self.data[list(self.data.keys())[0]]['close'].pct_change()
        fig.add_trace(
            go.Histogram(x=returns, name='Returns',
                        nbinsx=50, histnorm='probability'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text="Risk Analysis Dashboard"
        )
        
        return fig    

    def plot_efficient_frontier(self, ef_data: pd.DataFrame) -> go.Figure:
        """
        Plot the efficient frontier.
        
        Args:
            ef_data: DataFrame with efficient frontier data
            
        Returns:
            Plotly figure object
        """
        # Create scatter plot of efficient frontier
        fig = go.Figure()
        
        # Add efficient frontier line
        fig.add_trace(go.Scatter(
            x=ef_data['volatility'],
            y=ef_data['return'],
            mode='lines+markers',
            name='Efficient Frontier',
            marker=dict(
                color=ef_data['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            )
        ))
        
        # Add individual assets
        for symbol in self.returns.columns:
            asset_vol = np.sqrt(self.returns[symbol].var() * 252)
            asset_ret = self.returns[symbol].mean() * 252
            
            fig.add_trace(go.Scatter(
                x=[asset_vol],
                y=[asset_ret],
                mode='markers+text',
                name=symbol,
                text=[symbol],
                textposition="top center",
                marker=dict(size=10)
            ))
            
        # Update layout
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Portfolio Volatility',
            yaxis_title='Portfolio Return',
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig

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
        """Detect structural breaks using rolling statistical tests."""
        results = pd.DataFrame(index=X.index)
        results['break_chow'] = np.nan
        results['break_volatility'] = np.nan
        results['break_volume'] = np.nan
        
        returns = X['returns']
        volumes = X['volume']
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i-self.window:i]
            window_volumes = volumes.iloc[i-self.window:i]
            
            # Test for breaks in mean using Chow test-like approach
            pre_mean = window_returns[:self.window//2].mean()
            post_mean = window_returns[self.window//2:].mean()
            mean_diff = abs(pre_mean - post_mean)
            mean_std = window_returns.std()
            results.iloc[i, 0] = mean_diff / mean_std
            
            # Test for breaks in volatility
            pre_vol = window_returns[:self.window//2].std()
            post_vol = window_returns[self.window//2:].std()
            vol_ratio = max(pre_vol, post_vol) / min(pre_vol, post_vol)
            results.iloc[i, 1] = vol_ratio
            
            # Test for breaks in volume
            pre_vol_mean = window_volumes[:self.window//2].mean()
            post_vol_mean = window_volumes[self.window//2:].mean()
            vol_mean_ratio = max(pre_vol_mean, post_vol_mean) / min(pre_vol_mean, post_vol_mean)
            results.iloc[i, 2] = vol_mean_ratio
        
        # Identify significant breaks
        results['significant_break'] = (
            (results['break_chow'] > 2) |  # 2 std dev threshold
            (results['break_volatility'] > 2) |  # Double volatility
            (results['break_volume'] > 2)  # Double volume
        )
        
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

class LeadLagAnalyzer:
    """
    Analyzes lead-lag relationships between securities using various methods.
    """
    def __init__(self, returns_data: Dict[str, pd.DataFrame]):
        self.returns_data = returns_data
        self.relationships = {}
        
    def calculate_cross_correlations(self, 
                                   symbols: List[str], 
                                   max_lags: int = 5) -> pd.DataFrame:
        """
        Calculate cross-correlations between multiple symbols at different lags.
        
        Args:
            symbols: List of symbols to analyze
            max_lags: Maximum number of lags to consider
            
        Returns:
            DataFrame with cross-correlations at different lags
        """
        results = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:  # Only calculate upper triangle
                    continue
                    
                returns1 = self.returns_data[symbol1]['daily_return'].dropna()
                returns2 = self.returns_data[symbol2]['daily_return'].dropna()
                
                # Align the time series
                common_idx = returns1.index.intersection(returns2.index)
                returns1 = returns1[common_idx]
                returns2 = returns2[common_idx]
                
                # Calculate correlations at different lags
                for lag in range(-max_lags, max_lags + 1):
                    if lag < 0:
                        corr = pearsonr(returns1.iloc[-lag:], returns2.iloc[:lag])[0]
                    elif lag > 0:
                        corr = pearsonr(returns1.iloc[:-lag], returns2.iloc[lag:])[0]
                    else:
                        corr = pearsonr(returns1, returns2)[0]
                        
                    results.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'lag': lag,
                        'correlation': corr
                    })
        
        return pd.DataFrame(results)
    
    def test_granger_causality(self, 
                              symbols: List[str],
                              max_lag: int = 5,
                              significance_level=0.05) -> pd.DataFrame:
        """
        Test for Granger causality between two symbols.

        Imagine you're trying to figure out if rainy weather actually causes people to carry umbrellas. 
        Common sense says yes, but how can we prove it statistically? 
        This is where Granger Causality comes in.
        Granger Causality, developed by Clive Granger, 
        is a statistical concept that helps determine 
        if one time series (let's call it X) 
        helps predict another time series (Y). 
        The key idea is that if X "Granger-causes" Y, 
        then past values of X should contain information that helps predict Y, 
        beyond what we can predict just using past values of Y alone.
        Let's break this down with our rain and umbrellas example:

        First, try to predict umbrella usage using only past umbrella usage data
        Then, try to predict umbrella usage using both past umbrella usage AND past rainfall data
        If adding rainfall data significantly improves our prediction of umbrella usage, we say that rainfall "Granger-causes" umbrella usage

        Here's the catch though - Granger Causality isn't the same as real causation.
        It really just tells us about predictive ability. 
        For instance, dark clouds might Granger-cause rainfall, 
        but they don't directly cause rain - they're both part of the same weather system.        
        
        Args:
            symbols: The symbols to check for Granger Causality
            max_lag: Maximum number of lags to test
            significance_level: default value for "passing" the causality test
            
        Returns:
            Dictionary with test results
        """
        results = []
        
        # Test each possible pair of symbols
        for cause_symbol in symbols:
            for effect_symbol in symbols:
                # Skip self-causation tests
                if cause_symbol == effect_symbol:
                    continue
                    
                # Get the return series
                returns1 = self.returns_data[cause_symbol]['daily_return'].dropna()
                returns2 = self.returns_data[effect_symbol]['daily_return'].dropna()
                
                # Align the time series
                common_idx = returns1.index.intersection(returns2.index)
                returns1 = returns1[common_idx]
                returns2 = returns2[common_idx]
                
                # Create DataFrame for testing
                data = pd.DataFrame({
                    'y': returns2,  # effect
                    'x': returns1   # cause
                })
                
                try:
                    test_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                    
                    # Extract results for each lag
                    for lag in range(1, max_lag + 1):
                        # Get test statistics and coefficients
                        model_results = test_results[lag][1][1]  # unrestricted model
                        
                        # Get coefficient names and values
                        coef_names = [f'y_lag_{i+1}' for i in range(lag)]
                        coef_names.extend([f'x_lag_{i+1}' for i in range(lag)])
                        if model_results.model.k_constant:
                            coef_names.append('const')
                        
                        coeffs = pd.Series(model_results.params, index=coef_names)
                        pvalues = pd.Series(model_results.pvalues, index=coef_names)
                        
                        # Store results
                        row = {
                            'cause': cause_symbol,
                            'effect': effect_symbol,
                            'lag': lag,
                            'ssr_chi2_pvalue': test_results[lag][0]['ssr_chi2test'][1],
                            'ssr_f_pvalue': test_results[lag][0]['ssr_ftest'][1],
                            'r2': model_results.rsquared,
                            'adj_r2': model_results.rsquared_adj
                        }
                        
                        # Add coefficients and their p-values
                        for name in coef_names:
                            row[f'coef_{name}'] = coeffs[name]
                            row[f'pval_{name}'] = pvalues[name]
                        
                        results.append(row)
                        
                except Exception as e:
                    print(f"Error testing {cause_symbol} -> {effect_symbol}: {e}")
                    continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by p-value to highlight most significant relationships
        results_df = results_df.sort_values('ssr_f_pvalue')

        #significant_results = results_df[results_df['ssr_f_pvalue'] < significance_level].copy()
        
        # Add effect size (using R-squared as a simple measure)
        results_df['effect_size'] = results_df['r2']
        
        # Create summary with relevant coefficients
        summaries = []
        for _, row in results_df.iterrows():
            coef_cols = [col for col in row.index if col.startswith('coef_x_lag_')]
            pval_cols = [col for col in row.index if col.startswith('pval_x_lag_')]
            
            # Get significant coefficients
            sig_coeffs = []
            for coef_col, pval_col in zip(coef_cols, pval_cols):
                if row[pval_col] < significance_level:
                    lag_num = coef_col.split('_')[-1]
                    sig_coeffs.append(f"Lag {lag_num}: {row[coef_col]:.4f}")
            
            summaries.append({
                'cause': row['cause'],
                'effect': row['effect'],
                'lag': row['lag'],
                'p_value': row['ssr_f_pvalue'],
                'r2': row['r2'],
                'significant_coefficients': ', '.join(sig_coeffs)
            })
        
        return pd.DataFrame(summaries)
    
    def build_relationship_network(self, 
                                 symbols: List[str], 
                                 threshold: float = 0.5) -> nx.Graph:
        """
        Build a network of relationships between symbols based on correlations.
        
        Args:
            symbols: List of symbols to include in network
            threshold: Minimum absolute correlation to include edge
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(symbols)
        
        # Calculate correlations and add edges
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                returns1 = self.returns_data[symbol1]['daily_return'].dropna()
                returns2 = self.returns_data[symbol2]['daily_return'].dropna()
                
                # Align the time series
                common_idx = returns1.index.intersection(returns2.index)
                if len(common_idx) < 252:  # Require at least 1 year of common data
                    continue
                    
                returns1 = returns1[common_idx]
                returns2 = returns2[common_idx]
                
                corr = pearsonr(returns1, returns2)[0]
                
                if abs(corr) >= threshold:
                    G.add_edge(symbol1, symbol2, weight=corr)
        
        return G
    
    def find_market_leaders(self,
                            symbols: List[str],
                            max_lag: int = 5,
                            significance_level: float = 0.05,
                            use_effect_size: bool = True) -> Dict[str, float]:
        """
        Identify market leaders based on Granger causality relationships.
        
        This function analyzes the Granger causality relationships between all pairs of symbols
        to identify which symbols tend to lead others in price movements. A symbol's leadership
        score increases when it Granger-causes other symbols with statistical significance.
        
        The scoring incorporates both statistical significance (-log(p-value)) and optionally
        the effect size (R²) to provide a robust measure of leadership.
        
        Args:
            symbols: List of symbols to analyze
            max_lag: Maximum number of lags to test
            significance_level: P-value threshold for statistical significance
            use_effect_size: If True, incorporates R² in the leadership scoring
            
        Returns:
            Dictionary mapping each symbol to its normalized leadership score (0 to 1)
        """
        # Get Granger causality test results for all pairs
        results_df = self.test_granger_causality(symbols, max_lag, significance_level)
        
        # Initialize leadership scores
        leadership_scores = {symbol: 0.0 for symbol in symbols}
        
        # Group by cause-effect pairs and get the minimum p-value for each relationship
        pair_results = (results_df.groupby(['cause', 'effect'])
                                .agg({'p_value': 'min',
                                    'r2': 'max'})
                                .reset_index())
        
        # Calculate leadership scores
        for _, row in pair_results.iterrows():
            if row['p_value'] < significance_level:
                # Base score using -log(p-value) to reflect strength of significance
                # Add small epsilon to prevent log(0) for extremely small p-values
                score_increment = -np.log10(row['p_value'] + 1e-300)
                
                # Optionally weight by effect size (R²)
                if use_effect_size:
                    score_increment *= row['r2']
                
                leadership_scores[row['cause']] += score_increment
        
        # Normalize scores
        max_score = max(leadership_scores.values())
        if max_score > 0:
            leadership_scores = {
                k: v/max_score for k, v in leadership_scores.items()
            }
        
        # Sort by score in descending order
        leadership_scores = dict(sorted(leadership_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True))
        
        return leadership_scores

class PatternRecognition:
    """
    A comprehensive framework for detecting and analyzing technical patterns in financial price data.
    
    This class implements various pattern detection algorithms to identify common technical
    trading patterns like head and shoulders, double bottoms, and volume-price divergences.
    These patterns are used by traders to make predictions about future price movements.

    Key Concepts:
    -------------
    1. Swing Points:
       - Local maxima (peaks) and minima (troughs) in price data
       - Form the building blocks of many technical patterns
       Example: In a double bottom pattern, we look for two similar price troughs

    2. Pattern Formation:
       - Specific arrangements of swing points that form recognizable patterns
       - Each pattern has criteria for price levels, timing, and symmetry
       Example: Head & shoulders pattern needs three peaks with the middle one highest

    3. Volume Confirmation:
       - Trading volume often helps confirm pattern validity
       - Volume patterns can diverge from price, signaling potential reversals
       Example: Decreasing volume during price rises might signal weakness

    Parameters:
    -----------
    prices : pd.Series
        Time series of price data (typically closing prices)
    volumes : pd.Series
        Time series of trading volume data, indexed same as prices

    Example:
    --------
    >>> # Initialize with price and volume data
    >>> pattern_finder = PatternRecognition(
    ...     prices=df['close'],
    ...     volumes=df['volume']
    ... )
    >>> 
    >>> # Find head and shoulders patterns
    >>> patterns = pattern_finder.detect_head_and_shoulders()
    >>> 
    >>> # Analyze the patterns
    >>> for pattern in patterns:
    ...     print(f"Found pattern between index {pattern.start_idx} and {pattern.end_idx}")
    ...     print(f"Price range: ${pattern.price_range[0]:.2f} to ${pattern.price_range[1]:.2f}")
    """
    def __init__(self, prices: pd.Series, volumes: pd.Series):
        self.prices = prices
        self.volumes = volumes
        self.patterns = []
    
    def find_swing_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify swing high and low points in the price series using local extrema detection.
        
        What are Swing Points?
        - Swing Highs: Local price peaks (price higher than surrounding prices)
        - Swing Lows: Local price troughs (price lower than surrounding prices)
        - Used as building blocks to identify larger technical patterns
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Two arrays containing indices of swing highs and swing lows
            First array: Indices of swing highs
            Second array: Indices of swing lows

        Example:
        --------
        >>> highs, lows = pattern_finder.find_swing_points(window=10)
        >>> print(f"Found {len(highs)} swing highs and {len(lows)} swing lows")
        >>> 
        >>> # Get the prices at swing points
        >>> swing_high_prices = prices.iloc[highs]
        >>> swing_low_prices = prices.iloc[lows]
        """
        diff = np.diff(self.prices)
        maxima = []
        minima = []
        
        i = 0
        while i < len(diff):
            # Found increasing then decreasing (potential maximum)
            if i > 0 and diff[i-1] > 0 and diff[i] < 0:
                maxima.append(i)
                
            # Found decreasing then increasing (potential minimum)
            elif i > 0 and diff[i-1] < 0 and diff[i] > 0:
                minima.append(i)
                
            # Handle plateau
            elif diff[i] == 0:
                plateau_start = i
                # Find end of plateau
                while i < len(diff) and diff[i] == 0:
                    i += 1
                    
                plateau_end = i
                plateau_center = plateau_start + (plateau_end - plateau_start) // 2
                
                # Check if it's a maximum or minimum plateau
                if plateau_start > 0:
                    if diff[plateau_start-1] > 0 and (i < len(diff) and diff[i] < 0):
                        maxima.append(plateau_center)
                    elif diff[plateau_start-1] < 0 and (i < len(diff) and diff[i] > 0):
                        minima.append(plateau_center)
                
                continue
                
            i += 1
        
        return np.array(maxima), np.array(minima)
    
    def detect_head_and_shoulders(self) -> List[TechnicalPattern]:
        """
        Detect head and shoulders patterns in the price data.
        
        What is a Head and Shoulders Pattern?
        - A reversal pattern suggesting a trend change from up to down
        - Consists of:
          * Left Shoulder: First peak
          * Head: Higher middle peak
          * Right Shoulder: Third peak at similar height to left shoulder
          * Neckline: Support line connecting the troughs between peaks
        
        Pattern Criteria:
        1. Head must be higher than both shoulders
        2. Shoulders should be at similar price levels (within 2%)
        3. Neckline should be roughly horizontal (within 2% slope)
        
        Parameters:
        -----------
        window : int, default=20
            Window size for finding swing points
            Larger values find larger patterns
            Example: window=20 finds patterns lasting about a month

        Returns:
        --------
        List[TechnicalPattern]
            List of detected head and shoulders patterns
            Each pattern includes start/end points, confidence level, and price range

        Example:
        --------
        >>> patterns = pattern_finder.detect_head_and_shoulders()
        >>> for pattern in patterns:
        ...     print(f"Pattern found between days {pattern.start_idx} and {pattern.end_idx}")
        ...     print(f"Price range: ${pattern.price_range[0]:.2f} to ${pattern.price_range[1]:.2f}")
        ...     print(f"Confidence: {pattern.confidence:.1%}")
        """
        highs, lows = self.find_swing_points()
        patterns = []
        
        if len(highs) < 3:
            return patterns
        
        if lows[0] < highs[0]:
            lows = lows[1:]

        for i in range(len(highs) - 3):
            points = HeadAndShouldersPoints(
                left_shoulder_idx=highs[i],
                head_idx=highs[i + 1],
                right_shoulder_idx=highs[i + 2],
                left_trough_idx=lows[i],
                right_trough_idx=lows[i + 1]
            )
            
            validation = validate_head_and_shoulders(self.prices, points)
            if validation.is_valid:
                patterns.append(TechnicalPattern("HEAD_AND_SHOULDERS", highs[i], highs[i+2], price_range=(min(lows[i], lows[i+1]), highs[i+1])))

        return patterns
    
    def detect_double_bottom(self, window: int = 20, tolerance: float = 0.02) -> List[TechnicalPattern]:
        """
        Detect double bottom patterns in the price data.
        
        What is a Double Bottom Pattern?
        - A reversal pattern suggesting a trend change from down to up
        - Consists of:
          * First Bottom: Initial price trough
          * Second Bottom: Similar price trough
          * Peak: Higher price point between bottoms
        
        Pattern Criteria:
        1. Two price troughs at similar levels (within tolerance)
        2. A noticeable peak between the troughs
        3. Second bottom should confirm support level
        
        Parameters:
        -----------
        window : int, default=20
            Window size for finding swing points
            Larger values find larger patterns
            Example: window=20 finds patterns lasting about a month
        tolerance : float, default=0.02
            Maximum allowed difference between bottom prices (as percentage)
            Example: 0.02 means bottoms must be within 2% of each other

        Returns:
        --------
        List[TechnicalPattern]
            List of detected double bottom patterns
            Each pattern includes start/end points, confidence level, and price range

        Example:
        --------
        >>> patterns = pattern_finder.detect_double_bottom(tolerance=0.03)
        >>> if patterns:
        ...     pattern = patterns[0]
        ...     print(f"Double bottom found with bottoms at ${pattern.price_range[0]:.2f}")
        ...     print(f"Pattern confidence: {pattern.confidence:.1%}")
        """
        patterns = []
        _, lows = self.find_swing_points()
        
        for i in range(len(lows) - 1):
            bottom1 = self.prices.iloc[lows[i]]
            bottom2 = self.prices.iloc[lows[i + 1]]
            
            if abs(bottom1 - bottom2) / bottom1 < tolerance:
                middle_idx = slice(lows[i], lows[i + 1])
                middle_high = self.prices.iloc[middle_idx].max()
                
                pattern = TechnicalPattern(
                    pattern_type="DOUBLE_BOTTOM",
                    start_idx=lows[i],
                    end_idx=lows[i + 1],
                    confidence=0.7,
                    price_range=(min(bottom1, bottom2), middle_high)
                )
                patterns.append(pattern)
        
        return patterns
        
    def detect_volume_price_patterns(self,
                                min_price_change: float = 0.002,
                                min_volume_change: float = 0.01,
                                window_size: int = 20,
                                target_length: int = 4,
                                min_pattern_weight: float = 0.6
                                ) -> pd.DataFrame:
        """
        Detect volume-price patterns using weighted pattern detection with temporal decay.
        
        Parameters:
        -----------
        min_price_change : float
            Minimum relative change to consider price as moving
        min_volume_change : float
            Minimum relative change to consider volume as moving
        window_size: int
            Size of rolling window for pattern identification
        target_length: int
            Target number of points for pattern confirmation (used for decay rate)
        min_pattern_weight: float
            Minimum weighted score required to confirm a pattern
        
        Returns:
        --------
        pd.DataFrame with columns:
            - timestamp_idx: index of the time point
            - primary_pattern: dominant pattern type if above threshold, else None
            - pattern_weight: weighted score of dominant pattern
            - pattern: classification if pattern_weight > threshold, else None
            - DIVERGENCE: weighted score for DIVERGENCE pattern
            - NON_CONFIRMATION: weighted score for NON_CONFIRMATION pattern
            - VOLUME_FORCE: weighted score for VOLUME_FORCE pattern
            - NEUTRAL: weighted score for NEUTRAL pattern
            - CONCORDANT: weighted score for CONCORDANT pattern
        """
        def classify_points(prices: pd.Series, volumes: pd.Series, 
                        min_price_change: float = 0.002,
                        min_volume_change: float = 0.01
                        ) -> List[VolumePatternType]:
            # First classify all points
            price_changes = prices.pct_change()
            volume_changes = volumes.pct_change()
            
            point_patterns = [VolumePatternType.NEUTRAL]  # First point has no change
            
            for i in range(1, len(prices)):
                price_moving = abs(price_changes[i]) >= min_price_change
                volume_moving = abs(volume_changes[i]) >= min_volume_change
                
                if not price_moving and not volume_moving:
                    pattern = VolumePatternType.NEUTRAL
                elif price_moving and not volume_moving:
                    pattern = VolumePatternType.NON_CONFIRMATION
                elif not price_moving and volume_moving:
                    pattern = VolumePatternType.VOLUME_FORCE
                else:  # Both moving
                    if np.sign(price_changes[i]) != np.sign(volume_changes[i]):
                        pattern = VolumePatternType.DIVERGENCE
                    else:
                        pattern = VolumePatternType.CONCORDANT
                
                point_patterns.append(pattern)
            
            return point_patterns
        
        # Calculate decay rate based on target length
        decay_rate = np.log(2) / target_length  # Half-life at target length
        
        # Generate weights for the window
        weights = np.exp(-decay_rate * np.arange(window_size))
        weights = np.flip(weights / weights.sum())  # Normalize weights
        
        # First classify all points
        point_patterns = classify_points(self.prices, self.volumes, 
                                    min_price_change=min_price_change,
                                    min_volume_change=min_volume_change)
        
        # Initialize results storage
        results = []

        # Process points with growing/sliding window
        for i in range(len(point_patterns)):
            # For early points, use growing window with adjusted weights
            if i < window_size:
                window = point_patterns[:i+1]
                current_weights = weights[-(i+1):]
                current_weights = current_weights / current_weights.sum()
            else:
                # For later points, use sliding window
                window = point_patterns[i - window_size + 1:i + 1]
                current_weights = weights
            
            # Calculate weighted pattern distribution
            distribution = {}
            for pattern_type in VolumePatternType:
                # Create mask for this pattern type
                pattern_mask = [1 if p == pattern_type else 0 for p in window]
                # Calculate weighted sum
                weighted_sum = np.sum(pattern_mask * current_weights)
                distribution[pattern_type.value] = weighted_sum
            
            # Find primary pattern
            primary_pattern = max(distribution.items(), key=lambda x: x[1])
            
            # Create result entry
            result = {
                'timestamp_idx': i,
                'primary_pattern': primary_pattern[0],
                'pattern_weight': primary_pattern[1],
                'pattern': primary_pattern[0] if primary_pattern[1] >= min_pattern_weight else None
            }
            result.update(distribution)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df

class MarketAnalyzer:
    """
    A comprehensive framework for market analysis that processes financial market data
    and calculates various metrics useful for investment decision-making.

    Key Concepts:
    -------------
    1. Market Data:
       - OHLCV: Open, High, Low, Close prices and Volume
       - Open: First trading price of the day
       - Close: Last trading price of the day
       - High/Low: Highest/lowest prices during the day
       - Volume: Number of shares traded

    2. Returns:
       - Daily Returns: Percentage change in price (e.g., +2% means price increased by 2%)
       - Log Returns: Natural logarithm of price ratios, useful for statistical analysis
       Example:
           If a stock goes from $100 to $102:
           Daily return = ($102 - $100)/$100 = 2%
           Log return = ln($102/$100) ≈ 1.98%

    3. Market Indices:
       - Benchmark indices like S&P 500 (^GSPC) or Dow Jones (^DJI)
       - Used to compare individual stock performance against overall market
       Example:
           If S&P 500 is up 1% and a stock is up 2%, the stock outperformed by 1%

    Parameters:
    -----------
    data : Dict[str, pd.DataFrame]
        Dictionary containing market data for each symbol
        Each DataFrame should have columns: ['open', 'high', 'low', 'close', 'volume']
    market_indices : List[str]
        List of market index symbols (e.g., ['^GSPC', '^DJI'])
    benchmark_index : str, optional
        Primary market index for comparison (default: '^GSPC' - S&P 500)

    Example:
    --------
    >>> data = {
    ...     'AAPL': apple_data_df,  # DataFrame with OHLCV data
    ...     '^GSPC': sp500_data_df  # S&P 500 index data
    ... }
    >>> analyzer = MarketAnalyzer(data=data, market_indices=['^GSPC'])
    >>> features = analyzer.calculate_rolling_features('AAPL')
    """
    def __init__(self, 
                 data: Dict[str, pd.DataFrame],
                 market_indices: List[str] = ['^GSPC']):
        self.data = data
        self.market_indices = market_indices
        self.returns_data = {}
        self._prepare_returns_data()

    def _prepare_returns_data(self):
        """
        Calculate and store various types of returns for all symbols.

        This method computes:
        1. Daily Returns: Simple percentage change
           Formula: (P₁ - P₀)/P₀ where P₁ is current price and P₀ is previous price
           
        2. Log Returns: Natural logarithm of price ratio
           Formula: ln(P₁/P₀)
           Why? Log returns are:
           - More likely to be normally distributed (good for statistics)
           - Additive across time (useful for multi-period analysis)

        Example:
        --------
        If stock price moves: $100 → $102 → $100
        Daily returns: +2%, -1.96%
        Log returns: +1.98%, -1.98% (notice they sum to ~0)
        """
        for symbol, df in self.data.items():
            returns = pd.DataFrame({
                'daily_return': df['close'].pct_change(),
                'log_return': np.log(df['close']).diff(),
                'volume': df['volume']
            })
            returns.index = df.index
            self.returns_data[symbol] = returns

    def calculate_rolling_features(self, 
                                 symbol: str,
                                 windows: List[int] = [5, 21, 63, 252]) -> pd.DataFrame:
        """
        Calculate rolling window features for market analysis.

        Windows Explanation:
        - 5 days: One trading week
        - 21 days: One trading month
        - 63 days: One trading quarter
        - 252 days: One trading year

        Features Calculated:
        1. Return Statistics:
           - Mean: Average return (trend direction)
           - Std: Standard deviation (volatility)
           - Skew: Return distribution asymmetry
           - Kurt: "Fatness" of return tails (extreme event frequency)

        2. Price Statistics:
           - Mean: Average price level
           - Std: Price volatility

        3. Volume Statistics:
           - Mean: Average trading activity
           - Std: Trading activity volatility

        4. Technical Indicators:
           - RSI: Relative Strength Index (momentum indicator)

        Example:
        --------
        >>> features = analyzer.calculate_rolling_features('AAPL', windows=[5, 21])
        >>> # Check if stock is trending up over last month
        >>> if features['return_mean_21d'].iloc[-1] > 0:
        ...     print("Stock has positive momentum")
        """
        df = self.returns_data[symbol].copy()
        price_data = self.data[symbol]['close']
        
        features = pd.DataFrame(index=df.index)
        
        for window in windows:
            # Return-based features
            roll = df['daily_return'].rolling(window=window)
            features[f'return_mean_{window}d'] = roll.mean()
            features[f'return_std_{window}d'] = roll.std()
            features[f'return_skew_{window}d'] = roll.skew()
            features[f'return_kurt_{window}d'] = roll.kurt()
            
            # Price-based features
            roll_price = price_data.rolling(window=window)
            features[f'price_mean_{window}d'] = roll_price.mean()
            features[f'price_std_{window}d'] = roll_price.std()
            
            # Volume-based features
            roll_vol = df['volume'].rolling(window=window)
            features[f'volume_mean_{window}d'] = roll_vol.mean()
            features[f'volume_std_{window}d'] = roll_vol.std()
            
            # Technical indicators
            features[f'RSI_{window}d'] = self._calculate_rsi(price_data, window)
            
        return features

    def detect_volatility_regime(self, 
                               symbol: str,
                               lookback: int = 252) -> pd.DataFrame:
        """
        Detect market volatility regimes using GARCH (Generalized AutoRegressive 
        Conditional Heteroskedasticity) models.

        What is GARCH?
        - A statistical model that describes how volatility changes over time
        - Captures "volatility clustering" (volatile periods tend to persist)
        - Helps predict future volatility levels

        Regime Classifications:
        0: Low Volatility
           - Market is calm
           - Price changes are small
           - Good for steady growth strategies

        1: Normal Volatility
           - Typical market conditions
           - Price changes are moderate
           - Standard trading conditions

        2: High Volatility
           - Market is turbulent
           - Large price swings
           - May need risk management

        Example:
        --------
        >>> regimes = analyzer.detect_volatility_regime('AAPL')
        >>> if regimes['regime'].iloc[-1] == 2:
        ...     print("Market is highly volatile - consider reducing position sizes")
        """
        returns = self.returns_data[symbol]['daily_return'].dropna()
        
        results = pd.DataFrame(index=returns.index)
        results['volatility'] = np.nan
        results['regime'] = np.nan
        
        for i in range(lookback, len(returns)):
            window_returns = returns.iloc[i-lookback:i]
            try:
                model = arch_model(window_returns, vol='Garch', p=1, q=1)
                res = model.fit(disp='off')
                results.iloc[i, 0] = res.conditional_volatility[-1]
                
                vol_mean = res.conditional_volatility.mean()
                vol_std = res.conditional_volatility.std()
                current_vol = res.conditional_volatility[-1]
                
                if current_vol > vol_mean + vol_std:
                    results.iloc[i, 1] = 2  # High volatility
                elif current_vol < vol_mean - vol_std:
                    results.iloc[i, 1] = 0  # Low volatility
                else:
                    results.iloc[i, 1] = 1  # Normal volatility
                    
            except:
                continue
                
        return results

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI), a momentum indicator.

        What is RSI?
        - Measures the speed and magnitude of recent price changes
        - Ranges from 0 to 100
        - Used to identify overbought/oversold conditions

        Interpretation:
        - RSI > 70: Potentially overbought (price might fall)
        - RSI < 30: Potentially oversold (price might rise)
        - RSI = 50: Neutral momentum

        Example:
        --------
        If a stock has had more up days than down days recently:
        - RSI will be high (>70)
        - This might indicate the stock is overbought
        - Traders might consider taking profits

        Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = (Average Gain / Average Loss)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_patterns(self,
                     symbol: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, List[TechnicalPattern]]:
        """
        Analyze price data to detect technical trading patterns over a specified time period.
        This comprehensive analysis looks for multiple pattern types that traders use to
        make predictions about future price movements.

        What are Technical Patterns?
        --------------------------
        Technical patterns are specific price formations that traders believe can indicate
        future market movements. This method detects three key pattern types:

        1. Head and Shoulders
        - Represents potential trend reversal from up to down
        - Three peaks with middle peak (head) highest
        Example: Stock rises to $50 (left shoulder), $55 (head), then $50 (right shoulder)

        2. Double Bottom
        - Represents potential trend reversal from down to up
        - Two similar price lows with higher price between
        Example: Stock drops to $40 twice with rise to $45 between drops

        3. Volume-Price Divergence
        - When price and volume trends differ
        - May signal trend weakness
        Example: Price rising but volume declining suggests weak buying interest

        Parameters:
        -----------
        symbol : str
            The stock symbol to analyze
            Example: 'AAPL' for Apple Inc.

        start_date : str, optional
            Start date for analysis in 'YYYY-MM-DD' format
            Example: '2023-01-01' for January 1st, 2023
            If None, starts from earliest available date

        end_date : str, optional
            End date for analysis in 'YYYY-MM-DD' format
            Example: '2023-12-31' for December 31st, 2023
            If None, continues to latest available date

        Returns:
        --------
        Dict[str, List[TechnicalPattern]]
            Dictionary where:
            - Keys: Pattern types ('head_and_shoulders', 'double_bottom', 'volume_price')
            - Values: Lists of TechnicalPattern objects for each type found

        Example Usage:
        -------------
        >>> # Analyze patterns for Apple stock in 2023
        >>> patterns = analyzer.analyze_patterns(
        ...     symbol='AAPL',
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31'
        ... )
        >>> 
        >>> # Check for head and shoulders patterns
        >>> hs_patterns = patterns['head_and_shoulders']
        >>> if hs_patterns:
        ...     pattern = hs_patterns[0]
        ...     print(f"Found H&S pattern with {pattern.confidence:.1%} confidence")
        ...     print(f"Price range: ${pattern.price_range[0]:.2f} to ${pattern.price_range[1]:.2f}")
        >>> 
        >>> # Look for recent double bottoms
        >>> db_patterns = patterns['double_bottom']
        >>> recent_patterns = [p for p in db_patterns if p.end_idx > len(prices) - 20]
        >>> if recent_patterns:
        ...     print("Found recent double bottom - potential upward reversal")

        Tips for Pattern Analysis:
        ------------------------
        1. Pattern Reliability
        - Higher confidence patterns (>0.8) are more reliable
        - Look for patterns with clear price levels
        - Volume confirmation strengthens pattern signals

        2. Time Horizons
        - Longer patterns (more days between start_idx and end_idx) often more significant
        - Recent patterns more relevant for current trading decisions
        - Compare patterns across different time windows

        3. Pattern Combinations
        - Multiple pattern types confirming same direction = stronger signal
        - Example: Double bottom + positive volume divergence suggests stronger upward move
        - Check for conflicting patterns before making decisions
        """

        # Get symbol data
        df = self.data[symbol]
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Initialize pattern recognition
        pattern_finder = PatternRecognition(df['close'], df['volume'])
        
        # Detect patterns
        patterns = {
            'head_and_shoulders': pattern_finder.detect_head_and_shoulders(),
            'double_bottom': pattern_finder.detect_double_bottom(),
            'volume_price': pattern_finder.detect_volume_price_patterns()
        }
        
        return patterns

    def analyze_market_state(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze overall market conditions using index data to identify market trends,
        volatility levels, and momentum. This analysis helps understand the broader
        market environment for making informed investment decisions.

        Market State Components:
        -----------------------
        1. Trend Analysis (20-day vs 50-day moving average)
        - Compares short-term (20-day) to longer-term (50-day) price averages
        - Trend = 1 (Uptrend): Short-term average > Long-term average
        - Trend = 0 (Downtrend): Short-term average < Long-term average
        
        Example:
            If 20-day avg = $105 and 50-day avg = $100
            → Trend = 1 (uptrend)
            This suggests recent prices are higher than historical prices

        2. Volatility Analysis (20-day standard deviation)
        - Measures how much prices are moving/varying
        - Higher values = More market uncertainty/risk
        - Lower values = More stable/predictable market
        
        Example:
            If 20-day std dev = 2%
            → Daily price typically varies by ±2%
            During calm markets: might be 0.5-1%
            During turbulent markets: might be 2-4%

        3. Momentum Analysis (20-day average return)
        - Measures the recent "strength" of price movements
        - Positive value = Upward momentum (prices trending up)
        - Negative value = Downward momentum (prices trending down)
        
        Example:
            If 20-day avg return = +0.1%
            → Market has been moving up on average
            Strong momentum: > ±0.2% per day
            Weak momentum: < ±0.05% per day

        Parameters:
        -----------
        start_date : str, optional
            Starting date for analysis in 'YYYY-MM-DD' format
            Example: '2023-01-01'
            If None, starts from earliest available date
        
        end_date : str, optional
            Ending date for analysis in 'YYYY-MM-DD' format
            Example: '2023-12-31'
            If None, continues to latest available date

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary where:
            - Keys: Market index symbols (e.g., '^GSPC' for S&P 500)
            - Values: DataFrames containing:
                - trend_20d: Binary indicator (1=uptrend, 0=downtrend)
                - volatility_20d: Rolling 20-day standard deviation
                - momentum_20d: Rolling 20-day average return

        Example Usage:
        -------------
        >>> # Get market state for last quarter of 2023
        >>> market_state = analyzer.analyze_market_state(
        ...     start_date='2023-10-01',
        ...     end_date='2023-12-31'
        ... )
        >>> 
        >>> # Check S&P 500 conditions
        >>> sp500_state = market_state['^GSPC']
        >>> 
        >>> # Get latest readings
        >>> latest = sp500_state.iloc[-1]
        >>> 
        >>> # Example analysis
        >>> if (latest['trend_20d'] == 1 and      # In uptrend
        ...     latest['volatility_20d'] < 0.01 and   # Low volatility
        ...     latest['momentum_20d'] > 0):      # Positive momentum
        ...     print("Market showing healthy uptrend with low risk")
        >>> 
        >>> # Or analyze changing conditions
        >>> if (sp500_state['volatility_20d'].mean() > 0.02 and  # High average volatility
        ...     sp500_state['trend_20d'].sum() < len(sp500_state) * 0.3):  # Mostly downtrend
        ...     print("Market has been unstable and weak")
        """
        market_state = {}
        
        for index in self.market_indices:
            # Skip if index data not available
            if index not in self.data:
                continue
                
            # Get index data for specified date range
            index_data = self.data[index]
            if start_date:
                index_data = index_data[index_data.index >= start_date]
            if end_date:
                index_data = index_data[index_data.index <= end_date]
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=index_data.index)
            
            # 1. Calculate trend indicator
            # Compare 20-day vs 50-day moving averages
            ma_20 = index_data['close'].rolling(20).mean()
            ma_50 = index_data['close'].rolling(50).mean()
            features['trend_20d'] = (ma_20 > ma_50).astype(int)
            
            # 2. Calculate volatility
            # Use log returns for better statistical properties
            log_returns = np.log(index_data['close']).diff()
            features['volatility_20d'] = log_returns.rolling(20).std()
            
            # 3. Calculate momentum
            # Use 20-day average of log returns
            features['momentum_20d'] = log_returns.rolling(20).mean()
            
            market_state[index] = features
            
        return market_state
    
    def analyze_relationships(self,
                              symbols: List[str],
                              max_lags: int = 5,
                              correlation_threshold: float = 0.5) -> Dict[str, Union[pd.DataFrame, nx.Graph, Dict]]:
        """
        Comprehensive analysis of relationships between symbols.
        
        Args:
            symbols: List of symbols to analyze
            max_lags: Maximum number of lags for analysis
            correlation_threshold: Minimum correlation for network relationships
            
        Returns:
            Dictionary containing various relationship analyses
        """
        lead_lag = LeadLagAnalyzer(self.returns_data)
        
        results = {
            'cross_correlations': lead_lag.calculate_cross_correlations(symbols, max_lags),
            'relationship_network': lead_lag.build_relationship_network(symbols, correlation_threshold),
            'market_leaders': lead_lag.find_market_leaders(symbols, max_lags)
        }
        
        # Add pairwise Granger causality for market leaders
        leader_symbols = [s for s, score in results['market_leaders'].items() if score > 0.5]
        granger_results = {}
        
        for symbol1 in leader_symbols:
            for symbol2 in symbols:
                if symbol1 != symbol2:
                    key = f"{symbol1}->{symbol2}"
                    granger_results[key] = lead_lag.test_granger_causality(symbol1, symbol2, max_lags)
        
        results['granger_causality'] = granger_results
        
        return results

    def analyze_regimes(self,
                         symbol: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         hmm_states: int = 3,
                         window: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive regime analysis for a given symbol.
        
        Args:
            symbol: Stock symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            hmm_states: Number of HMM states to detect
            window: Window size for structural break detection
            
        Returns:
            Dictionary containing various regime analyses
        """
        # Get symbol data
        returns = self.returns_data[symbol]['daily_return']
        prices = self.data[symbol]['close']
        volumes = self.data[symbol]['volume']
        
        if start_date:
            returns = returns[returns.index >= start_date]
            prices = prices[prices.index >= start_date]
            volumes = volumes[volumes.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]
            prices = prices[prices.index <= end_date]
            volumes = volumes[volumes.index <= end_date]
            
        # Initialize regime analyzer
        regime_analyzer = RegimeAnalyzer(returns, prices, volumes)
        
        # Get various regime analyses
        results = {
            'hmm_regimes': regime_analyzer.detect_hmm_regimes(hmm_states),
            'structural_breaks': regime_analyzer.detect_structural_breaks(window),
            'combined_regimes': regime_analyzer.detect_combined_regime(hmm_states, window)
        }
        
        # Add volatility regimes from earlier method
        vol_regimes = self.detect_volatility_regime(symbol)
        if start_date:
            vol_regimes = vol_regimes[vol_regimes.index >= start_date]
        if end_date:
            vol_regimes = vol_regimes[vol_regimes.index <= end_date]
        results['volatility_regimes'] = vol_regimes
        
        return results    
    
    def create_visualizations(self,
                              symbol: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, go.Figure]:
        """
        Create comprehensive set of visualizations for analysis results.
        
        Args:
            symbol: Stock symbol to visualize
            start_date: Start date for visualization
            end_date: End date for visualization
            
        Returns:
            Dictionary of Plotly figure objects
        """
        visualizer = MarketVisualizer(self.data, self.results)
        
        figures = {
            'price_patterns': visualizer.plot_price_patterns(symbol, start_date, end_date),
            'combined_regimes': visualizer.plot_regimes(symbol, 'combined'),
            'volatility_regimes': visualizer.plot_regimes(symbol, 'volatility'),
            'relationship_network': visualizer.plot_network(),
            'lead_lag_heatmap': visualizer.plot_lead_lag_heatmap()
        }
        
        return figures