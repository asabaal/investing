"""
Claude chat:
https://claude.ai/chat/e57d8498-85ed-478a-9aa4-a5dcba070116
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from arch import arch_model
import warnings
from scipy.signal import argrelextrema
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
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
    confidence: float
    price_range: Tuple[float, float]
    failure_reasons: Optional[Dict[str, str]] = None
    specific_points: Optional[dict] = None

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
                patterns.append(TechnicalPattern("HEAD_AND_SHOULDERS", highs[i], highs[i+2], 0.8, (min(lows[i], lows[i+1]), highs[i+1])))

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
    
    def detect_volume_price_divergence(self, window: int = 20) -> List[TechnicalPattern]:
        """
        Detect divergences between price and volume trends.
        
        What is Volume-Price Divergence?
        - Occurs when price and volume trends move in opposite directions
        - Often signals potential trend reversals
        - Examples:
          * Prices rising but volume declining (weak uptrend)
          * Prices falling but volume declining (weak downtrend)
        
        Why Important?
        - Volume often confirms price movements
        - Divergences suggest current trend might be weakening
        - Used to identify potential trend reversals
        
        Parameters:
        -----------
        window : int, default=20
            Window size for calculating trends
            Larger windows smooth out noise but lag more
            Example: window=20 looks at monthly trends

        Returns:
        --------
        List[TechnicalPattern]
            List of detected divergence patterns
            Each pattern includes:
            - Start/end points of divergence
            - Confidence (based on degree of divergence)
            - Price range during divergence

        Example:
        --------
        >>> patterns = pattern_finder.detect_volume_price_divergence()
        >>> for pattern in patterns:
        ...     print(f"Divergence found from index {pattern.start_idx} to {pattern.end_idx}")
        ...     print(f"Confidence in signal: {pattern.confidence:.1%}")
        """
        patterns = []
        
        price_trend = self.prices.rolling(window).mean().pct_change()
        volume_trend = self.volumes.rolling(window).mean().pct_change()
        
        for i in range(window, len(self.prices) - window):
            price_direction = np.sign(price_trend.iloc[i])
            volume_direction = np.sign(volume_trend.iloc[i])
            
            if price_direction != volume_direction:
                pattern = TechnicalPattern(
                    pattern_type="VOLUME_PRICE_DIVERGENCE",
                    start_idx=i - window,
                    end_idx=i,
                    confidence=abs(price_trend.iloc[i] - volume_trend.iloc[i]),
                    price_range=(self.prices.iloc[i-window], self.prices.iloc[i])
                )
                patterns.append(pattern)
        
        return patterns

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
            - Keys: Pattern types ('head_and_shoulders', 'double_bottom', 'volume_price_divergence')
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
            'volume_price_divergence': pattern_finder.detect_volume_price_divergence()
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