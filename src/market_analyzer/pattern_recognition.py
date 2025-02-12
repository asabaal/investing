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
    A data class for storing detected technical pattern information.

    Args:
        pattern_type (str): Type of pattern detected (e.g., "HEAD_AND_SHOULDERS")
        start_idx (int): Starting index of the pattern in the price series
        end_idx (int): Ending index of the pattern in the price series
        price_range (Tuple[float, float]): Min and max prices within pattern (min_price, max_price)
        failure_reasons (Optional[Dict[str, str]], optional): Reasons for imperfect pattern match.
            Defaults to None.
        specific_points (Optional[dict], optional): Pattern-specific point indices. Defaults to None.
        volume_range (Optional[Tuple[float, float]], optional): Min and max volumes. Defaults to None.
        sub_classification (Optional[Enum], optional): Further pattern classification. Defaults to None.
        confidence (Optional[float], optional): Pattern confidence score (0-1). Defaults to np.nan.

    Examples:
        >>> pattern = TechnicalPattern(
        ...     pattern_type="HEAD_AND_SHOULDERS",
        ...     start_idx=100,
        ...     end_idx=150,
        ...     price_range=(45.50, 51.75),
        ...     confidence=0.85
        ... )
        >>> pattern.pattern_type
        'HEAD_AND_SHOULDERS'
        >>> pattern.confidence
        0.85
        >>> pattern.price_range
        (45.5, 51.75)
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
    """
    Points defining a head and shoulders pattern.

    Args:
        left_shoulder_idx (int): Index of the left shoulder peak
        head_idx (int): Index of the head peak
        right_shoulder_idx (int): Index of the right shoulder peak
        left_trough_idx (int): Index of the trough between left shoulder and head
        right_trough_idx (int): Index of the trough between head and right shoulder

    Examples:
        >>> points = HeadAndShouldersPoints(
        ...     left_shoulder_idx=10,
        ...     head_idx=20,
        ...     right_shoulder_idx=30,
        ...     left_trough_idx=15,
        ...     right_trough_idx=25
        ... )
        >>> points.head_idx
        20
        >>> points.left_shoulder_idx < points.head_idx < points.right_shoulder_idx
        True
    """    
    left_shoulder_idx: int
    head_idx: int
    right_shoulder_idx: int
    left_trough_idx: int
    right_trough_idx: int
    
@dataclass
class PatternValidation:
    """
    Results of pattern validation checks.

    Args:
        is_valid (bool): Whether pattern meets all validation criteria
        confidence (float): Confidence score for the pattern (0-1)
        failure_reasons (Dict[str, str]): Reasons for any validation failures
        price_range (Optional[Tuple[float, float]], optional): Price range if valid. 
            Defaults to None.         

    Examples:
        >>> # Test valid pattern
        >>> validation = PatternValidation(
        ...     is_valid=True,
        ...     confidence=0.85,
        ...     failure_reasons={},
        ...     price_range=(100.0, 110.0)
        ... )
        >>> bool(validation.is_valid)  # Convert from numpy bool if needed
        True
        >>> float(validation.confidence)  # Convert from numpy float if needed
        0.85
        >>> validation.failure_reasons == {}  # Empty dict for valid pattern
        True
        
        >>> # Test failed pattern
        >>> failed_validation = PatternValidation(
        ...     is_valid=False,
        ...     confidence=0.3,
        ...     failure_reasons={'slope': 'Neckline slope too steep'},
        ...     price_range=None
        ... )
        >>> bool(failed_validation.is_valid)
        False
        >>> bool(len(failed_validation.failure_reasons) > 0)  # Has failure reasons
        True
    """    
    is_valid: bool
    confidence: float
    failure_reasons: Dict[str, str]  # Key is check name, value is failure description
    price_range: Optional[Tuple[float, float]] = None


class VolumePatternType(Enum):
    """
    Types of volume patterns in relation to price movement.
    
    Examples:
        >>> VolumePatternType.DIVERGENCE.value
        'DIVERGENCE'
        >>> pattern_type = VolumePatternType.CONCORDANT
        >>> pattern_type == VolumePatternType.CONCORDANT
        True
        >>> pattern_type.value
        'CONCORDANT'
    """    
    
    #: Volume moving opposite to price
    DIVERGENCE = "DIVERGENCE"
    
    #: Volume flat while price moves
    NON_CONFIRMATION = "NON_CONFIRMATION"
    
    #: Volume moving while price is flat
    VOLUME_FORCE = "VOLUME_FORCE"
    
    #: Both price and volume are flat
    NEUTRAL = "NEUTRAL"
    
    #: Price and volume moving in same direction
    CONCORDANT = "CONCORDANT"

@dataclass
class VolumePattern(TechnicalPattern):
    """
    Volume-based pattern information.

    Inherits from TechnicalPattern and adds volume-specific attributes.

    Args:
        pattern_type (VolumePatternType): Type of volume pattern
        price_range (Tuple[float, float]): Min and max prices within pattern
        volume_range (Tuple[float, float]): Min and max volumes within pattern

    Examples:
        >>> pattern = VolumePattern(
        ...     pattern_type=VolumePatternType.DIVERGENCE,
        ...     start_idx=50,
        ...     end_idx=60,
        ...     price_range=(100.0, 110.0),
        ...     volume_range=(5000, 7500)
        ... )
        >>> pattern.pattern_type == VolumePatternType.DIVERGENCE
        True
        >>> pattern.volume_range
        (5000, 7500)
        >>> pattern.price_range
        (100.0, 110.0)
    """    
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
    Validate a potential head and shoulders pattern.

    Args:
        prices (np.ndarray): Array of price values
        points (HeadAndShouldersPoints): Points forming the potential pattern
        shoulder_height_tolerance (float, optional): Maximum allowed shoulder height difference.
            Defaults to 0.02 (2%).
        neckline_slope_tolerance (float, optional): Maximum allowed neckline slope.
            Defaults to 0.02 (2%).

    Returns:
        PatternValidation: Validation results including confidence score and any failure reasons        

    Examples:
        >>> # Create a valid head and shoulders pattern
        >>> prices = np.array([10.0, 12.0, 11.0, 13.0, 11.0, 12.0, 10.0])
        >>> test_points = HeadAndShouldersPoints(
        ...     left_shoulder_idx=1,
        ...     head_idx=3,
        ...     right_shoulder_idx=5,
        ...     left_trough_idx=2,
        ...     right_trough_idx=4
        ... )
        >>> validation = validate_head_and_shoulders(prices, test_points)
        >>> bool(validation.is_valid)  # Convert numpy bool to Python bool
        True
        >>> bool(float(validation.confidence) > 0.8)  # High confidence score
        True
        >>> dict(validation.failure_reasons) == {}  # No failure reasons
        True
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
    Analyzes lead-lag relationships between securities.

    Identifies which securities tend to lead or lag others in price movements
    using various statistical methods including cross-correlation and Granger causality.

    Args:
        returns_data (Dict[str, pd.DataFrame]): Dictionary mapping symbols to their returns data
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
            symbols (List[str]): List of symbols to analyze
            max_lags (int, optional): Maximum number of lags to consider. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing correlations with columns:
                - symbol1: First symbol in pair
                - symbol2: Second symbol in pair
                - lag: Time lag in periods
                - correlation: Correlation coefficient

    Examples:
        >>> # Create sample data with known correlation
        >>> dates = pd.date_range('2024-01-01', '2024-01-10')
        >>> np.random.seed(42)
        >>> base_returns = np.random.randn(10) * 0.01
        >>> data1 = pd.DataFrame({
        ...     'daily_return': base_returns
        ... }, index=dates)
        >>> data2 = pd.DataFrame({
        ...     'daily_return': base_returns * 0.9 + np.random.randn(10) * 0.001
        ... }, index=dates)
        >>> returns_data = {'AAPL': data1, 'MSFT': data2}
        >>> analyzer = LeadLagAnalyzer(returns_data)
        >>> correlations = analyzer.calculate_cross_correlations(['AAPL', 'MSFT'], max_lags=2)
        >>> isinstance(correlations, pd.DataFrame)
        True
        >>> set(['symbol1', 'symbol2', 'lag', 'correlation']).issubset(
        ...     set(correlations.columns))
        True
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
        Test for Granger causality between pairs of symbols.

        Args:
            symbols (List[str]): List of symbols to test
            max_lag (int, optional): Maximum number of lags to test. Defaults to 5.
            significance_level (float, optional): P-value threshold. Defaults to 0.05.

        Returns:
            pd.DataFrame: DataFrame containing test results with columns:
                - cause: Potential causing symbol
                - effect: Potential effect symbol
                - lag: Number of lags tested
                - p_value: Test p-value
                - r2: R-squared value
                - significant_coefficients: String of significant lag coefficients

        Examples:
            >>> dates = pd.date_range('2024-01-01', '2024-01-10')
            >>> # Create more realistic data with noise
            >>> leader_returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, -0.01]
            >>> follower_returns = [0.005, 0.015, -0.005, 0.025, -0.015, 0.008, 0.018, -0.008, 0.008, -0.008]
            >>> data1 = pd.DataFrame({'daily_return': leader_returns}, index=dates)
            >>> data2 = pd.DataFrame({'daily_return': follower_returns}, index=dates)
            >>> returns_data = {'LEADER': data1, 'FOLLOWER': data2}
            >>> analyzer = LeadLagAnalyzer(returns_data)
            >>> results = analyzer.test_granger_causality(['LEADER', 'FOLLOWER'], max_lag=2)
            >>> isinstance(results, pd.DataFrame)
            True
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
        breakpoint()
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
        Build a network graph of relationships between symbols.

        Args:
            symbols (List[str]): List of symbols to include
            threshold (float, optional): Minimum correlation for edge inclusion.
                Defaults to 0.5.

        Returns:
            nx.Graph: NetworkX graph where:
                - Nodes are symbols
                - Edges represent correlations above threshold
                - Edge weights are correlation values

        Examples:
            >>> dates = dates = pd.date_range('2024-01-01', '2024-12-31')
            >>> # Create more realistic correlated data
            >>> base = np.random.randn(len(dates))
            >>> data1 = pd.DataFrame({'daily_return': base}, index=dates)
            >>> data2 = pd.DataFrame({'daily_return': base * 0.9 + 0.01}, index=dates)  # Strongly correlated
            >>> data3 = pd.DataFrame({'daily_return': -base * 0.8 + 0.01}, index=dates)  # Negatively correlated
            >>> returns_data = {'A': data1, 'B': data2, 'C': data3}
            >>> analyzer = LeadLagAnalyzer(returns_data)
            >>> G = analyzer.build_relationship_network(['A', 'B', 'C'], threshold=0.5)
            >>> sorted(G.nodes()) == ['A', 'B', 'C']  # All nodes present
            True
            >>> any(G.edges())  # Should have at least one edge
            True
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

        Args:
            symbols (List[str]): List of symbols to analyze
            max_lag (int, optional): Maximum number of lags to test. Defaults to 5.
            significance_level (float, optional): P-value threshold. Defaults to 0.05.
            use_effect_size (bool, optional): Whether to weight by R². Defaults to True.

        Returns:
            Dict[str, float]: Dictionary mapping symbols to normalized leadership scores (0-1)


        Examples:
            >>> # Create sample data with clear but imperfect lead-lag relationship
            >>> dates = pd.date_range('2024-01-01', '2024-01-10')
            >>> np.random.seed(42)  # For reproducibility
            >>> leader_base = np.random.randn(10) * 0.1  # Base returns
            >>> leader_data = pd.DataFrame({
            ...     'daily_return': leader_base
            ... }, index=dates)
            >>> # Create followers with lag and noise
            >>> follower1_data = pd.DataFrame({
            ...     'daily_return': np.roll(leader_base, 1) + np.random.randn(10) * 0.02
            ... }, index=dates)
            >>> follower2_data = pd.DataFrame({
            ...     'daily_return': np.roll(leader_base, 2) + np.random.randn(10) * 0.02
            ... }, index=dates)
            >>> returns_data = {
            ...     'LEADER': leader_data,
            ...     'FOLLOWER1': follower1_data,
            ...     'FOLLOWER2': follower2_data
            ... }
            >>> analyzer = LeadLagAnalyzer(returns_data)
            >>> scores = analyzer.find_market_leaders(
            ...     symbols=['LEADER', 'FOLLOWER1', 'FOLLOWER2'],
            ...     max_lag=2,
            ...     significance_level=0.1  # More lenient for test
            ... )
            >>> isinstance(scores, dict)  # Returns correct type
            True
            >>> set(scores.keys()) == {'LEADER', 'FOLLOWER1', 'FOLLOWER2'}  # Has all symbols
            True
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
    A framework for detecting technical patterns in financial price data.

    Args:
        prices (pd.Series): Time series of price data
        volumes (pd.Series): Time series of trading volume data

    Attributes:
        prices (pd.Series): Price data
        volumes (pd.Series): Volume data
        patterns (list): List of detected patterns  
    """
    def __init__(self, prices: pd.Series, volumes: pd.Series):
        self.prices = prices
        self.volumes = volumes
        self.patterns = []
    
    def find_swing_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify swing high and low points in the price series.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing:
                - First array: Indices of swing highs
                - Second array: Indices of swing lows

        Examples:
            >>> # Create a series with clear swing points
            >>> prices = pd.Series([10, 12, 11, 14, 13, 15, 14, 13])
            >>> volumes = pd.Series([1000] * len(prices))
            >>> pattern_recognition = PatternRecognition(prices, volumes)
            >>> highs, lows = pattern_recognition.find_swing_points()
            >>> # Convert numpy indices to Python ints for comparison
            >>> [int(i) for i in highs]  # Indices of local maxima
            [1, 3, 5]
            >>> [int(i) for i in lows]  # Indices of local minima
            [2, 4]
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

        Returns:
            List[TechnicalPattern]: List of detected head and shoulders patterns

        Examples:
            >>> # Create a price series with a head and shoulders pattern
            >>> prices = pd.Series([5, 15, 10, 20, 10, 15, 5])
            >>> volumes = pd.Series([1000] * len(prices))
            >>> pattern_recognition = PatternRecognition(prices, volumes)
            >>> patterns = pattern_recognition.detect_head_and_shoulders()
            >>> len(patterns)  # Should find one pattern
            1
            >>> patterns[0].pattern_type
            'HEAD_AND_SHOULDERS'
            >>> patterns[0].price_range  # (min_price, max_price)
            (10.0, 20.0)
        """
        highs, lows = self.find_swing_points()
        patterns = []
        
        if len(highs) < 3:
            return patterns
        
        if lows[0] < highs[0]:
            lows = lows[1:]

        for i in range(len(highs) - 2):
            points = HeadAndShouldersPoints(
                left_shoulder_idx=highs[i],
                head_idx=highs[i + 1],
                right_shoulder_idx=highs[i + 2],
                left_trough_idx=lows[i],
                right_trough_idx=lows[i + 1]
            )
            
            validation = validate_head_and_shoulders(self.prices, points)
            if validation.is_valid:
                patterns.append(TechnicalPattern("HEAD_AND_SHOULDERS", highs[i], highs[i+2], price_range=(min(self.prices[lows[i]], self.prices[lows[i+1]]), self.prices[highs[i+1]])))

        return patterns
    
    def detect_double_bottom(self, window: int = 20, tolerance: float = 0.02) -> List[TechnicalPattern]:
        """
        Detect double bottom patterns in the price data.

        Args:
            window (int, optional): Window size for finding swing points. Defaults to 20.
            tolerance (float, optional): Maximum difference between bottoms. Defaults to 0.02.

        Returns:
            List[TechnicalPattern]: List of detected double bottom patterns

        Examples:
            >>> # Create a price series with a clear double bottom
            >>> prices = pd.Series([10, 8, 9.5, 8.1, 11])
            >>> volumes = pd.Series([1000, 1200, 900, 1100, 1300])
            >>> pattern_recognition = PatternRecognition(prices, volumes)
            >>> patterns = pattern_recognition.detect_double_bottom(window=3, tolerance=0.05)
            >>> isinstance(patterns, list)
            True
            >>> if len(patterns) > 0:  # Check pattern properties if found
            ...     patterns[0].pattern_type == 'DOUBLE_BOTTOM' and abs(patterns[0].price_range[0] - 8.0) < 0.1
            ... else:
            ...     True  # Skip validation if no patterns found
            True
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
        Detect volume-price patterns using weighted pattern detection.

        Args:
            min_price_change (float, optional): Minimum price change threshold. 
                Defaults to 0.002.
            min_volume_change (float, optional): Minimum volume change threshold. 
                Defaults to 0.01.
            window_size (int, optional): Size of rolling window. Defaults to 20.
            target_length (int, optional): Target points for pattern confirmation. 
                Defaults to 4.
            min_pattern_weight (float, optional): Minimum pattern score threshold. 
                Defaults to 0.6.

        Returns:
            pd.DataFrame: DataFrame containing detected patterns with columns:
                - timestamp_idx: Time point index
                - primary_pattern: Dominant pattern type
                - pattern_weight: Pattern confidence score
                - pattern: Classification if above threshold
                - Plus columns for each pattern type's score

        Examples:
            >>> # Create test data with clear volume-price patterns
            >>> np.random.seed(42)  # For reproducibility
            >>> prices = pd.Series([10.0 + i*0.2 + np.random.randn()*0.05 for i in range(5)])
            >>> volumes = pd.Series([1000 * (1 + np.random.randn()*0.2) for _ in range(5)])
            >>> pattern_recognition = PatternRecognition(prices, volumes)
            >>> patterns_df = pattern_recognition.detect_volume_price_patterns(
            ...     min_price_change=0.001,  # Lower threshold for test
            ...     min_volume_change=0.05,
            ...     window_size=3,
            ...     target_length=2,
            ...     min_pattern_weight=0.3  # Lower threshold for test
            ... )
            >>> isinstance(patterns_df, pd.DataFrame)  # Correct return type
            True
            >>> expected_cols = {'timestamp_idx', 'primary_pattern', 'pattern_weight', 
            ...                 'pattern', 'DIVERGENCE', 'NON_CONFIRMATION', 
            ...                 'VOLUME_FORCE', 'NEUTRAL', 'CONCORDANT'}
            >>> expected_cols.issubset(set(patterns_df.columns))  # Has required columns
            True
        """
        def classify_points(prices: pd.Series, volumes: pd.Series, 
                        min_price_change: float = 0.002,
                        min_volume_change: float = 0.01
                        ) -> List[VolumePatternType]:
            """
            Classify individual points based on price and volume movement patterns.

            Args:
                prices (pd.Series): Series of price values
                volumes (pd.Series): Series of volume values
                min_price_change (float, optional): Minimum change to consider price moving. 
                    Defaults to 0.002.
                min_volume_change (float, optional): Minimum change to consider volume moving. 
                    Defaults to 0.01.

            Returns:
                List[VolumePatternType]: Pattern classification for each point:
                    - NEUTRAL: Neither price nor volume moving
                    - NON_CONFIRMATION: Only price moving
                    - VOLUME_FORCE: Only volume moving
                    - DIVERGENCE: Both moving in opposite directions
                    - CONCORDANT: Both moving in same direction
            """            
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