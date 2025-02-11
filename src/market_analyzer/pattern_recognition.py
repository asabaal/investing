import warnings

import networkx as nx
import numpy as np
import pandas as pd

from dataclasses import dataclass
from enum import Enum
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Optional, Tuple

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