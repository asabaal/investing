#!/usr/bin/env python3
"""
Tests for Symphony.execute() method.

This module contains tests for the Symphony.execute() method in the
composer_symphony.py module. It tests the current implementation to ensure
functionality is preserved during refactoring.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import system modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from composer_symphony import Symphony, SymbolList, Momentum, RSIFilter, EqualWeightAllocator, InverseVolatilityAllocator

class TestSymphonyExecute(unittest.TestCase):
    """Test cases for Symphony.execute() method."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample data
        self.sample_dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create mock client
        self.mock_client = MagicMock()
        
        # Create simple symphony
        self.universe = SymbolList(["SPY", "QQQ", "IWM"])
        self.symphony = Symphony("Test Symphony", "A test symphony", self.universe)
        self.symphony.add_operator(Momentum("Momentum Filter", lookback_days=10, top_n=2))
        self.symphony.set_allocator(EqualWeightAllocator())
        
        # Set up mock data
        self.setup_mock_data()
    
    def setup_mock_data(self):
        """Set up mock data for testing."""
        # Create sample market data
        market_data = {}
        for symbol in self.universe.symbols:
            # Create basic price data with some randomness
            np.random.seed(42)  # For reproducibility
            base_price = 100.0
            daily_returns = np.random.normal(0.0005, 0.01, size=len(self.sample_dates))
            
            # Make SPY and QQQ have positive momentum, IWM negative
            if symbol == "SPY":
                daily_returns += 0.001  # Strong positive trend
            elif symbol == "QQQ":
                daily_returns += 0.0005  # Moderate positive trend
            else:
                daily_returns -= 0.001  # Negative trend
                
            prices = base_price * (1 + np.cumsum(daily_returns))
            
            # Create DataFrame
            market_data[symbol] = pd.DataFrame({
                'open': prices * (1 - np.random.uniform(0, 0.005, size=len(self.sample_dates))),
                'high': prices * (1 + np.random.uniform(0, 0.01, size=len(self.sample_dates))),
                'low': prices * (1 - np.random.uniform(0, 0.01, size=len(self.sample_dates))),
                'close': prices,
                'adjusted_close': prices,
                'volume': np.random.randint(100000, 10000000, size=len(self.sample_dates))
            }, index=self.sample_dates)
        
        # Create sample technical data
        technical_data = {}
        for symbol in self.universe.symbols:
            # Create basic RSI data
            if symbol == "SPY":
                rsi_values = np.linspace(60, 70, len(self.sample_dates))  # Strong
            elif symbol == "QQQ":
                rsi_values = np.linspace(50, 60, len(self.sample_dates))  # Moderate
            else:
                rsi_values = np.linspace(30, 40, len(self.sample_dates))  # Weak
                
            technical_data[symbol] = pd.DataFrame({
                'RSI': rsi_values
            }, index=self.sample_dates)
        
        # Set up the mock client to return our data
        def mock_get_daily(symbol, **kwargs):
            return market_data.get(symbol, pd.DataFrame())
        
        def mock_get_rsi(symbol, **kwargs):
            return technical_data.get(symbol, pd.DataFrame())
        
        self.mock_client.get_daily.side_effect = mock_get_daily
        self.mock_client.get_rsi.side_effect = mock_get_rsi
    
    def test_basic_execution(self):
        """Test basic execution of a symphony."""
        # Execute symphony
        allocations = self.symphony.execute(self.mock_client)
        
        # Assertions
        self.assertIsNotNone(allocations)
        self.assertIsInstance(allocations, dict)
        
        # Check that allocations contain the expected symbols (SPY and QQQ should be selected by momentum)
        self.assertIn("SPY", allocations)
        self.assertIn("QQQ", allocations)
        self.assertNotIn("IWM", allocations)  # IWM should be filtered out
        
        # Check for equal allocation (0.5 each for the 2 symbols)
        self.assertAlmostEqual(allocations["SPY"], 0.5)
        self.assertAlmostEqual(allocations["QQQ"], 0.5)
        
        # Check that the allocations sum to 1.0
        self.assertAlmostEqual(sum(allocations.values()), 1.0)
        
        # Verify that data was fetched for all symbols
        for symbol in self.universe.symbols:
            self.mock_client.get_daily.assert_any_call(symbol)
            self.mock_client.get_rsi.assert_any_call(symbol)
    
    def test_empty_universe(self):
        """Test execution with an empty universe."""
        # Create symphony with empty universe
        empty_universe = SymbolList([])
        empty_symphony = Symphony("Empty Symphony", "A symphony with no symbols", empty_universe)
        empty_symphony.add_operator(Momentum("Momentum Filter", lookback_days=10, top_n=2))
        empty_symphony.set_allocator(EqualWeightAllocator())
        
        # Execute symphony
        allocations = empty_symphony.execute(self.mock_client)
        
        # Assertions
        self.assertIsNotNone(allocations)
        self.assertIsInstance(allocations, dict)
        self.assertEqual(len(allocations), 0)  # No allocations should be made
        
        # Verify that no data was fetched
        self.mock_client.get_daily.assert_not_called()
        self.mock_client.get_rsi.assert_not_called()
    
    def test_data_fetching_failure(self):
        """Test when data fetching fails for some symbols."""
        # Make get_daily fail for QQQ
        original_get_daily = self.mock_client.get_daily.side_effect
        
        def mock_get_daily_with_failure(symbol, **kwargs):
            if symbol == "QQQ":
                raise Exception("Simulated API failure")
            return original_get_daily(symbol, **kwargs)
        
        self.mock_client.get_daily.side_effect = mock_get_daily_with_failure
        
        # Execute symphony
        allocations = self.symphony.execute(self.mock_client)
        
        # Assertions
        self.assertIsNotNone(allocations)
        self.assertIsInstance(allocations, dict)
        
        # QQQ should be excluded due to data fetching failure
        self.assertIn("SPY", allocations)
        self.assertNotIn("QQQ", allocations)
        self.assertNotIn("IWM", allocations)  # Still filtered out by momentum
        
        # SPY should get full allocation
        self.assertAlmostEqual(allocations["SPY"], 1.0)
    
    def test_all_symbols_filtered(self):
        """Test when all symbols are filtered out."""
        # Replace with a strict filter that filters out all symbols
        strict_rsi = RSIFilter("Strict RSI", threshold=80, condition="above")
        self.symphony.operators = [strict_rsi]  # Replace existing operators
        
        # Execute symphony
        allocations = self.symphony.execute(self.mock_client)
        
        # Assertions
        self.assertIsNotNone(allocations)
        self.assertIsInstance(allocations, dict)
        self.assertEqual(len(allocations), 0)  # No allocations since all symbols filtered out
    
    def test_inverse_volatility_allocator(self):
        """Test with InverseVolatilityAllocator."""
        # Replace allocator with InverseVolatilityAllocator
        self.symphony.set_allocator(InverseVolatilityAllocator(lookback_days=10))
        
        # Execute symphony
        allocations = self.symphony.execute(self.mock_client)
        
        # Assertions
        self.assertIsNotNone(allocations)
        self.assertIsInstance(allocations, dict)
        
        # Check that allocations contain the expected symbols
        self.assertIn("SPY", allocations)
        self.assertIn("QQQ", allocations)
        
        # Check that the allocations sum to 1.0
        self.assertAlmostEqual(sum(allocations.values()), 1.0)
        
        # Verify that inverse volatility was used (lower volatility should get higher allocation)
        # Since we used the same random seed, SPY and QQQ should have predictable volatility differences

if __name__ == '__main__':
    unittest.main()
