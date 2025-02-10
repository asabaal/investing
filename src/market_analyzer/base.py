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
