import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from arch import arch_model
import warnings
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
                 market_indices: List[str],
                 benchmark_index: str = '^GSPC'):
        self.data = data
        self.market_indices = market_indices
        self.benchmark_index = benchmark_index
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