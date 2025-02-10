import numpy as np
import pandas as pd
from market_analyzer import (rolling_growing_window,
                             growing_max,
                             growing_mean,
                             growing_min,
                             growing_std,
                             growing_sum)
from numpy.testing import assert_array_almost_equal

class TestGrowingWindowRolling:
    def test_growing_window_basic(self):
        """Test basic functionality with simple mean calculation"""
        data = np.array([1, 2, 3, 4, 5])
        result = rolling_growing_window(data, window=3, func=np.mean)
        expected = np.array([1., 1.5, 2., 3., 4.])
        assert_array_almost_equal(result, expected)

    def test_growing_window_pandas(self):
        """Test with pandas Series input"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = rolling_growing_window(data, window=3, func=np.mean)
        expected = np.array([1., 1.5, 2., 3., 4.])
        assert_array_almost_equal(result, expected)

    def test_growing_window_min_periods(self):
        """Test minimum periods requirement"""
        data = np.array([1, 2, 3, 4, 5])
        result = rolling_growing_window(data, window=3, func=np.mean, min_periods=2)
        expected = np.array([np.nan, 1.5, 2., 3., 4.])
        assert_array_almost_equal(result, expected)

    def test_growing_std(self):
        """Test growing window standard deviation"""
        data = np.array([1, 1, 1, 2, 2])
        result = growing_std(data, window=3)
        # First value should be 0 (no std for single value)
        # Second value should be 0 (std of [1,1])
        # Third value should be 0 (std of [1,1,1])
        # Fourth and fifth values should have non-zero std
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] > 0
        assert result[4] > 0

    def test_custom_function(self):
        """Test with a custom window function"""
        def weighted_mean(x):
            weights = np.arange(1, len(x) + 1)
            return np.average(x, weights=weights)
        
        data = np.array([1, 2, 3, 4, 5])
        result = rolling_growing_window(data, window=3, func=weighted_mean)
        # Manual calculation for verification
        expected = np.array([
            1.0,  # single value
            1.666666667,  # weighted mean of [1,2]
            2.333333333,  # weighted mean of [1,2,3]
            3.333333333,  # weighted mean of [2,3,4]
            4.333333333   # weighted mean of [3,4,5]
        ])
        assert_array_almost_equal(result, expected, decimal=5)

    def test_edge_cases(self):
        """Test edge cases"""
        # Empty array
        data = np.array([])
        result = rolling_growing_window(data, window=3, func=np.mean)
        assert len(result) == 0
        
        # Window size larger than data
        data = np.array([1, 2, 3])
        result = rolling_growing_window(data, window=5, func=np.mean)
        expected = np.array([1., 1.5, 2.])
        assert_array_almost_equal(result, expected)
        
        # Window size of 1
        data = np.array([1, 2, 3])
        result = rolling_growing_window(data, window=1, func=np.mean)
        assert_array_almost_equal(result, data)

    def test_specialized_functions(self):
        """Test all specialized growing window functions"""
        data = np.array([1, 2, 3, 4, 5])
        
        # Test growing_mean
        mean_result = growing_mean(data, window=3)
        assert_array_almost_equal(
            mean_result,
            np.array([1., 1.5, 2., 3., 4.])
        )
        
        # Test growing_sum
        sum_result = growing_sum(data, window=3)
        assert_array_almost_equal(
            sum_result,
            np.array([1., 3., 6., 9., 12.])
        )
        
        # Test growing_max
        max_result = growing_max(data, window=3)
        assert_array_almost_equal(
            max_result,
            np.array([1., 2., 3., 4., 5.])
        )
        
        # Test growing_min
        min_result = growing_min(data, window=3)
        assert_array_almost_equal(
            min_result,
            np.array([1., 1., 1., 2., 3.])
        )