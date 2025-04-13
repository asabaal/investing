"""
Tests for evaluation functions module.

This file contains unit tests for the evaluation functions to ensure they correctly
evaluate technical indicators against various criteria.
"""

import unittest
import numpy as np
import pandas as pd
from evaluation_functions import (
    evaluate_threshold,
    evaluate_range,
    evaluate_crossover,
    evaluate_lookback,
    evaluate_persistence
)


class TestEvaluationFunctions(unittest.TestCase):
    """Test cases for evaluation functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create various test series
        self.increasing_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        self.decreasing_values = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
        self.oscillating_values = np.array([50.0, 60.0, 70.0, 60.0, 50.0, 40.0, 30.0, 40.0, 50.0, 60.0])
        self.flat_values = np.array([50.0] * 10)
        
        # Create series with NaN values
        self.values_with_nans = np.array([10.0, np.nan, 30.0, np.nan, 50.0, 60.0, np.nan, 80.0, 90.0, np.nan])
        
        # Series for crossover tests
        self.fast_series = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 55.0, 45.0, 50.0, 55.0])
        self.slow_series = np.array([40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 48.0, 45.0, 47.0])
        
        # Series for persistence tests
        self.bool_series_consecutive = np.array([False, False, True, True, True, False, True, True, False, False])
        self.bool_series_scattered = np.array([True, False, True, False, True, False, True, False, True, False])
    
    def test_evaluate_threshold_above(self):
        """Test threshold evaluation with 'above' condition."""
        # Test with 'last' method (default)
        self.assertTrue(evaluate_threshold(self.increasing_values, 50.0, 'above'),
                       "Last value should be above 50")
        self.assertFalse(evaluate_threshold(self.decreasing_values, 50.0, 'above'),
                        "Last value should not be above 50")
        
        # Test with 'any' method
        self.assertTrue(evaluate_threshold(self.increasing_values, 50.0, 'above', 'any', 10),
                       "Some values should be above 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'above', 'any', 10),
                       "Some values should be above 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'above', 'any', 5),
                       "Some values in the first 5 should be above 50")
        self.assertFalse(evaluate_threshold(self.decreasing_values, 50.0, 'above', 'any', 3),
                        "No values in the last 3 should be above 50")
        
        # Test with 'all' method
        self.assertFalse(evaluate_threshold(self.increasing_values, 50.0, 'above', 'all', 10),
                        "Not all values should be above 50")
        self.assertTrue(evaluate_threshold(self.increasing_values, 50.0, 'above', 'all', 5),
                       "All values in the last 5 should be above 50")
        
        # Test with 'average' method
        self.assertTrue(evaluate_threshold(self.increasing_values, 50.0, 'above', 'average', 10),
                       "Average value should be above 50")
        self.assertFalse(evaluate_threshold(self.decreasing_values, 50.0, 'above', 'average', 5),
                        "Average of last 5 values should not be above 50")
    
    def test_evaluate_threshold_below(self):
        """Test threshold evaluation with 'below' condition."""
        # Test with 'last' method (default)
        self.assertFalse(evaluate_threshold(self.increasing_values, 50.0, 'below'),
                        "Last value should not be below 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'below'),
                       "Last value should be below 50")
        
        # Test with 'any' method
        self.assertTrue(evaluate_threshold(self.increasing_values, 50.0, 'below', 'any', 10),
                       "Some values should be below 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'below', 'any', 10),
                       "Some values should be below 50")
        self.assertFalse(evaluate_threshold(self.increasing_values, 50.0, 'below', 'any', 5),
                        "No values in the last 5 should be below 50")
        
        # Test with 'all' method
        self.assertFalse(evaluate_threshold(self.decreasing_values, 50.0, 'below', 'all', 10),
                        "Not all values should be below 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'below', 'all', 5),
                       "All values in the last 5 should be below 50")
        
        # Test with 'average' method
        self.assertFalse(evaluate_threshold(self.increasing_values, 50.0, 'below', 'average', 10),
                        "Average value should not be below 50")
        self.assertTrue(evaluate_threshold(self.decreasing_values, 50.0, 'below', 'average', 5),
                       "Average of last 5 values should be below 50")
    
    def test_evaluate_threshold_with_nans(self):
        """Test threshold evaluation with NaN values."""
        # Last value is NaN, should return False
        self.assertFalse(evaluate_threshold(self.values_with_nans, 50.0, 'above'),
                        "Should return False when last value is NaN")
        
        # Test with 'any' method - should skip NaNs
        self.assertTrue(evaluate_threshold(self.values_with_nans, 50.0, 'above', 'any', 5),
                       "Some non-NaN values in the last 5 should be above 50")
        
        # Test with 'all' method - should only consider non-NaN values
        values = np.array([60.0, 70.0, np.nan, 80.0, 90.0])
        self.assertTrue(evaluate_threshold(values, 50.0, 'above', 'all', 5),
                       "All non-NaN values should be above 50")
        
        # Test with all NaNs
        all_nans = np.array([np.nan, np.nan, np.nan])
        self.assertFalse(evaluate_threshold(all_nans, 50.0, 'above'),
                        "Should return False when all values are NaN")
    
    def test_evaluate_range_inside(self):
        """Test range evaluation with 'inside' condition."""
        # Test with 'last' method (default)
        self.assertTrue(evaluate_range(self.oscillating_values, 50.0, 70.0, 'inside'),
                       "Last value should be inside the range 50-70")
        self.assertFalse(evaluate_range(self.increasing_values, 50.0, 70.0, 'inside'),
                        "Last value should not be inside the range 50-70")
        
        # Test with 'any' method
        self.assertTrue(evaluate_range(self.oscillating_values, 30.0, 40.0, 'inside', 'any', 10),
                       "Some values should be inside the range 30-40")
        self.assertFalse(evaluate_range(self.increasing_values, 0.0, 5.0, 'inside', 'any', 10),
                        "No values should be inside the range 0-5")
        
        # Test with 'all' method
        self.assertFalse(evaluate_range(self.oscillating_values, 50.0, 70.0, 'inside', 'all', 10),
                        "Not all values should be inside the range 50-70")
        self.assertTrue(evaluate_range(self.flat_values, 45.0, 55.0, 'inside', 'all', 10),
                       "All values should be inside the range 45-55")
        
        # Test with 'average' method
        self.assertTrue(evaluate_range(self.oscillating_values, 40.0, 60.0, 'inside', 'average', 10),
                       "Average value should be inside the range 40-60")
        self.assertFalse(evaluate_range(self.increasing_values, 0.0, 40.0, 'inside', 'average', 10),
                        "Average value should not be inside the range 0-40")
    
    def test_evaluate_range_outside(self):
        """Test range evaluation with 'outside' condition."""
        # Test with 'last' method (default)
        self.assertFalse(evaluate_range(self.oscillating_values, 50.0, 70.0, 'outside'),
                        "Last value should not be outside the range 50-70")
        self.assertTrue(evaluate_range(self.increasing_values, 0.0, 50.0, 'outside'),
                       "Last value should be outside the range 0-50")
        
        # Test with 'any' method
        self.assertTrue(evaluate_range(self.oscillating_values, 50.0, 70.0, 'outside', 'any', 10),
                       "Some values should be outside the range 50-70")
        self.assertFalse(evaluate_range(self.flat_values, 45.0, 55.0, 'outside', 'any', 10),
                        "No values should be outside the range 45-55")
        
        # Test with 'all' method
        self.assertFalse(evaluate_range(self.oscillating_values, 0.0, 100.0, 'outside', 'all', 10),
                        "Not all values should be outside the range 0-100")
        self.assertTrue(evaluate_range(self.increasing_values, 0.0, 5.0, 'outside', 'all', 10),
                       "All values should be outside the range 0-5")
        
        # Test with 'average' method
        self.assertFalse(evaluate_range(self.oscillating_values, 40.0, 60.0, 'outside', 'average', 10),
                        "Average value should not be outside the range 40-60")
        self.assertTrue(evaluate_range(self.increasing_values, 0.0, 40.0, 'outside', 'average', 10),
                       "Average value should be outside the range 0-40")
    
    def test_evaluate_crossover(self):
        """Test crossover evaluation."""
        # Fast crosses above slow at index 5-6
        self.assertTrue(evaluate_crossover(self.fast_series, self.slow_series, 'above', 5),
                       "Fast should cross above slow within the lookback period")
        
        # Fast crosses below slow at index 6-7
        self.assertTrue(evaluate_crossover(self.fast_series, self.slow_series, 'below', 5),
                       "Fast should cross below slow within the lookback period")
        
        # Limit lookback to avoid finding the crossover
        self.assertFalse(evaluate_crossover(self.fast_series, self.slow_series, 'above', 2),
                        "Fast should not cross above slow within the limited lookback period")
        
        # Test with no crossover
        no_cross_fast = np.array([60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0])
        no_cross_slow = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
        
        self.assertFalse(evaluate_crossover(no_cross_fast, no_cross_slow, 'above', 10),
                        "Fast should not cross above slow when always above")
        self.assertFalse(evaluate_crossover(no_cross_fast, no_cross_slow, 'below', 10),
                        "Fast should not cross below slow when always above")
    
    def test_evaluate_lookback(self):
        """Test lookback evaluation with custom function."""
        # Custom function to count values above a threshold
        def count_above(values, threshold):
            return np.sum(values > threshold)
        
        # Test with the custom function
        self.assertEqual(evaluate_lookback(self.increasing_values, count_above, 5, threshold=50),
                        3, "Should count 3 values above 50 in the last 5")
        
        self.assertEqual(evaluate_lookback(self.decreasing_values, count_above, 7, threshold=50),
                        2, "Should count 2 values above 50 in the last 7")
        
        # Test with a boolean-returning function
        def any_above(values, threshold):
            return np.any(values > threshold)
        
        self.assertTrue(evaluate_lookback(self.increasing_values, any_above, 5, threshold=50),
                       "Should be True since some values are above 50")
        
        self.assertFalse(evaluate_lookback(self.decreasing_values, any_above, 3, threshold=50),
                        "Should be False since no values in the last 3 are above 50")
    
    def test_evaluate_persistence_consecutive(self):
        """Test persistence evaluation with 'consecutive' condition."""
        # Test with the consecutive boolean series
        self.assertTrue(evaluate_persistence(self.bool_series_consecutive, 3, 'consecutive'),
                       "Should find 3 consecutive True values")
        
        self.assertTrue(evaluate_persistence(self.bool_series_consecutive, 2, 'consecutive'),
                       "Should find 2 consecutive True values")
        
        self.assertFalse(evaluate_persistence(self.bool_series_consecutive, 4, 'consecutive'),
                        "Should not find 4 consecutive True values")
        
        # Test with a series that doesn't have enough consecutive values
        self.assertFalse(evaluate_persistence(self.bool_series_scattered, 2, 'consecutive'),
                        "Should not find 2 consecutive True values")
    
    def test_evaluate_persistence_total(self):
        """Test persistence evaluation with 'total' condition."""
        # Test with the consecutive boolean series
        self.assertTrue(evaluate_persistence(self.bool_series_consecutive, 5, 'total'),
                       "Should find 5 total True values")
        
        self.assertFalse(evaluate_persistence(self.bool_series_consecutive, 6, 'total'),
                        "Should not find 6 total True values")
        
        # Test with the scattered boolean series
        self.assertTrue(evaluate_persistence(self.bool_series_scattered, 5, 'total'),
                       "Should find 5 total True values")


if __name__ == "__main__":
    unittest.main()
