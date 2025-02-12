import warnings

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from arch import arch_model
from plotly.subplots import make_subplots
from typing import Any, List, Dict, Optional, Union

from market_analyzer.pattern_recognition import TechnicalPattern, PatternRecognition, LeadLagAnalyzer
from market_analyzer.regime_analyzer import RegimeAnalyzer

warnings.filterwarnings('ignore')

class MarketAnalyzer:
    """
    A comprehensive framework for market analysis and investment decision-making.

    This class processes financial market data to calculate various metrics and identify 
    patterns useful for trading and investment decisions.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary containing market data for each symbol.
            Each DataFrame should have columns: ['open', 'high', 'low', 'close', 'volume'].
        market_indices (List[str], optional): List of market index symbols to use as benchmarks.
            Defaults to ['^GSPC'] (S&P 500).

    Attributes:
        data (Dict[str, pd.DataFrame]): Market data for each symbol.
        market_indices (List[str]): List of market indices used as benchmarks.
        returns_data (Dict[str, pd.DataFrame]): Calculated returns for each symbol.

    Note:
        Market data terminology:
        - OHLCV: Open, High, Low, Close prices and Volume
        - Open: First trading price of the day
        - Close: Last trading price of the day
        - High/Low: Highest/lowest prices during the day
        - Volume: Number of shares traded

    Example:
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

        Calculates and stores the following returns metrics:
        - Daily Returns: Simple percentage change (P₁ - P₀)/P₀
        - Log Returns: Natural logarithm of price ratio ln(P₁/P₀)
        - Volume: Trading volume

        Note:
            Log returns are used because they are:
            - More likely to be normally distributed
            - Additive across time periods

        Example:
            For a stock price movement $100 → $102 → $100:
            - Daily returns: +2%, -1.96%
            - Log returns: +1.98%, -1.98% (sum to ~0)
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
        """Calculate rolling window features for market analysis.

        Args:
            symbol (str): The stock symbol to analyze.
            windows (List[int], optional): List of rolling window sizes in days.
                Defaults to [5, 21, 63, 252] representing week, month, quarter, year.

        Returns:
            pd.DataFrame: DataFrame containing calculated features with columns:
                - return_mean_{window}d: Average return
                - return_std_{window}d: Return standard deviation
                - return_skew_{window}d: Return skewness
                - return_kurt_{window}d: Return kurtosis
                - price_mean_{window}d: Average price
                - price_std_{window}d: Price standard deviation
                - volume_mean_{window}d: Average volume
                - volume_std_{window}d: Volume standard deviation
                - RSI_{window}d: Relative Strength Index

        Example:
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
        """Detect market volatility regimes using GARCH models.

        Uses GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models 
        to classify market volatility into different regimes.

        Args:
            symbol (str): The stock symbol to analyze.
            lookback (int, optional): Number of days to look back. Defaults to 252 (1 trading year).

        Returns:
            pd.DataFrame: DataFrame with columns:
                - volatility: Conditional volatility from GARCH model
                - regime: Volatility regime classification (0=Low, 1=Normal, 2=High)

        Example:
            >>> regimes = analyzer.detect_volatility_regime('AAPL')
            >>> if regimes['regime'].iloc[-1] == 2:
            ...     print("Market is highly volatile")
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
        Calculate the Relative Strength Index (RSI).

        Args:
            prices (pd.Series): Price series to calculate RSI for.
            window (int, optional): Calculation window. Defaults to 14.

        Returns:
            pd.Series: RSI values ranging from 0 to 100.
            - RSI > 70: Potentially overbought
            - RSI < 30: Potentially oversold
            - RSI = 50: Neutral momentum

        Note:
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
        Analyze price data to detect technical trading patterns.

        Args:
            symbol (str): The stock symbol to analyze.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, List[TechnicalPattern]]: Dictionary of detected patterns:
                - head_and_shoulders: Head and shoulders reversal patterns
                - double_bottom: Double bottom reversal patterns
                - volume_price: Volume-price divergence patterns

        Example:
            >>> patterns = analyzer.analyze_patterns(
            ...     symbol='AAPL',
            ...     start_date='2023-01-01',
            ...     end_date='2023-12-31'
            ... )
            >>> hs_patterns = patterns['head_and_shoulders']
            >>> if hs_patterns:
            ...     print(f"Found pattern with {hs_patterns[0].confidence:.1%} confidence")
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
        Analyze overall market conditions using index data.

        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping index symbols to DataFrames with:
                - trend_20d: Binary indicator (1=uptrend, 0=downtrend)
                - volatility_20d: Rolling 20-day standard deviation
                - momentum_20d: Rolling 20-day average return

        Example:
            >>> market_state = analyzer.analyze_market_state('2023-01-01', '2023-12-31')
            >>> sp500_state = market_state['^GSPC']
            >>> latest = sp500_state.iloc[-1]
            >>> if latest['trend_20d'] == 1 and latest['momentum_20d'] > 0:
            ...     print("Market in positive trend")
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
        Analyze relationships between multiple symbols.

        Args:
            symbols (List[str]): List of symbols to analyze.
            max_lags (int, optional): Maximum number of lags for analysis. Defaults to 5.
            correlation_threshold (float, optional): Minimum correlation threshold. 
                Defaults to 0.5.

        Returns:
            Dict[str, Union[pd.DataFrame, nx.Graph, Dict]]: Dictionary containing:
                - cross_correlations: Cross-correlation analysis
                - relationship_network: NetworkX graph of relationships
                - market_leaders: Dictionary of market leader scores
                - granger_causality: Granger causality test results
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
        Analyze market regimes using multiple methods.

        Args:
            symbol (str): Stock symbol to analyze.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            hmm_states (int, optional): Number of HMM states to detect. Defaults to 3.
            window (int, optional): Window size for structural break detection. 
                Defaults to 252.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing:
                - hmm_regimes: Hidden Markov Model regime classification
                - structural_breaks: Structural break points
                - combined_regimes: Combined regime analysis
                - volatility_regimes: Volatility regime classification
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
        Create a comprehensive set of analysis visualizations.

        Args:
            symbol (str): Stock symbol to visualize.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, go.Figure]: Dictionary of Plotly figures:
                - price_patterns: Price chart with pattern annotations
                - combined_regimes: Combined regime analysis visualization
                - volatility_regimes: Volatility regime visualization
                - relationship_network: Network graph of relationships
                - lead_lag_heatmap: Lead-lag relationship heatmap
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
    
class MarketVisualizer:
    """
    Creates interactive visualizations for market analysis results.

    Args:
        data (Dict[str, pd.DataFrame]): Market data dictionary.
        results (Dict[str, Any]): Analysis results dictionary.

    Attributes:
        data (Dict[str, pd.DataFrame]): Market data for visualization.
        results (Dict[str, Any]): Analysis results for visualization.
        returns (pd.DataFrame): Calculated returns for all assets.
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
            symbol (str): Stock symbol to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            go.Figure: Plotly figure with price and volume subplots.
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
            symbol (str): Stock symbol to plot.
            regime_type (str, optional): Type of regime analysis to visualize. 
                Defaults to 'combined'.

        Returns:
            go.Figure: Plotly figure with price and regime indicators.

        Raises:
            ValueError: If specified regime type not found in results.
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
            min_correlation (float, optional): Minimum correlation threshold. 
                Defaults to 0.5.

        Returns:
            go.Figure: Plotly figure with network graph.

        Raises:
            ValueError: If relationship network not found in results.
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
            go.Figure: Plotly figure with correlation heatmap.

        Raises:
            ValueError: If cross-correlation results not found.
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
            go.Figure: Plotly figure with 3D surface plot.
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
        Create visualization of risk metrics.

        Returns:
            go.Figure: Plotly figure with risk analysis dashboard including:
                - VaR comparison
                - Expected shortfall
                - Stress test scenarios
                - Return distribution
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
            ef_data (pd.DataFrame): DataFrame with efficient frontier data.

        Returns:
            go.Figure: Plotly figure with efficient frontier plot.
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
