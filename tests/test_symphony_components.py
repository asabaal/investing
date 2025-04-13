#!/usr/bin/env python3
"""
Tests for Symphony component methods.

This module contains tests for the new methods being added to Symphony class 
for improved modularity and testability.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add parent directory to path to import system modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from composer_symphony import Symphony, SymbolList, Momentum, RSIFilter, EqualWeightAllocator, InverseVolatilityAllocator

class TestSymphonyComponents(unittest.TestCase):
    """Test cases for Symphony component methods."""
    
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
        self.market_data = {}
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
            self.market_data[symbol] = pd.DataFrame({
                'open': prices * (1 - np.random.uniform(0, 0.005, size=len(self.sample_dates))),
                'high': prices * (1 + np.random.uniform(0, 0.01, size=len(self.sample_dates))),
                'low': prices * (1 - np.random.uniform(0, 0.01, size=len(self.sample_dates))),
                'close': prices,
                'adjusted_close': prices,
                'volume': np.random.randint(100000, 10000000, size=len(self.sample_dates))
            }, index=self.sample_dates)
        
        # Create sample technical data
        self.technical_data = {}
        for symbol in self.universe.symbols:
            # Create basic RSI data
            if symbol == "SPY":
                rsi_values = np.linspace(60, 70, len(self.sample_dates))  # Strong
            elif symbol == "QQQ":
                rsi_values = np.linspace(50, 60, len(self.sample_dates))  # Moderate
            else:
                rsi_values = np.linspace(30, 40, len(self.sample_dates))  # Weak
                
            self.technical_data[symbol] = pd.DataFrame({
                'RSI': rsi_values
            }, index=self.sample_dates)
        
        # Set up the mock client to return our data
        def mock_get_daily(symbol, **kwargs):
            return self.market_data.get(symbol, pd.DataFrame())
        
        def mock_get_rsi(symbol, **kwargs):
            return self.technical_data.get(symbol, pd.DataFrame())
        
        self.mock_client.get_daily.side_effect = mock_get_daily
        self.mock_client.get_rsi.side_effect = mock_get_rsi
    
    def test_fetch_market_data_success(self):
        """Test fetch_market_data with successful API calls."""
        # Call the method
        market_data, technical_data = self.symphony.fetch_market_data(self.mock_client)
        
        # Verify data was fetched for all symbols
        for symbol in self.universe.symbols:
            self.assertIn(symbol, market_data)
            self.assertIn(symbol, technical_data)
            
            # Verify dataframes have expected data
            self.assertIn('adjusted_close', market_data[symbol].columns)
            self.assertIn('RSI', technical_data[symbol].columns)
            
            # Verify calls were made to the client
            self.mock_client.get_daily.assert_any_call(symbol)
            self.mock_client.get_rsi.assert_any_call(symbol)
    
    def test_fetch_market_data_partial_failure(self):
        """Test fetch_market_data when some API calls fail."""
        # Make get_daily fail for QQQ
        original_get_daily = self.mock_client.get_daily.side_effect
        
        def mock_get_daily_with_failure(symbol, **kwargs):
            if symbol == "QQQ":
                raise Exception("Simulated API failure")
            return original_get_daily(symbol, **kwargs)
        
        self.mock_client.get_daily.side_effect = mock_get_daily_with_failure
        
        # Call the method
        market_data, technical_data = self.symphony.fetch_market_data(self.mock_client)
        
        # Verify data was fetched for other symbols
        self.assertIn("SPY", market_data)
        self.assertIn("IWM", market_data)
        
        # Verify QQQ was skipped due to error
        self.assertNotIn("QQQ", market_data)
        
        # Verify technical data still fetched for all symbols
        self.assertIn("SPY", technical_data)
        self.assertIn("IWM", technical_data)
        self.assertIn("QQQ", technical_data)
    
    def test_fetch_market_data_empty_universe(self):
        """Test fetch_market_data with empty universe."""
        # Create symphony with empty universe
        empty_universe = SymbolList([])
        empty_symphony = Symphony("Empty Symphony", "A symphony with no symbols", empty_universe)
        
        # Call the method
        market_data, technical_data = empty_symphony.fetch_market_data(self.mock_client)
        
        # Verify empty dictionaries returned
        self.assertEqual(len(market_data), 0)
        self.assertEqual(len(technical_data), 0)
        
        # Verify no API calls were made
        self.mock_client.get_daily.assert_not_called()
        self.mock_client.get_rsi.assert_not_called()
    
    def test_apply_filters_momentum(self):
        """Test apply_filters with Momentum filter."""
        # Call the method
        filtered_symbols = self.symphony.apply_filters(self.universe, self.market_data, self.technical_data)
        
        # Verify filtering worked as expected (should keep SPY and QQQ, filter out IWM based on momentum)
        self.assertEqual(len(filtered_symbols), 2)
        self.assertIn("SPY", filtered_symbols.symbols)
        self.assertIn("QQQ", filtered_symbols.symbols)
        self.assertNotIn("IWM", filtered_symbols.symbols)
    
    def test_apply_filters_rsi(self):
        """Test apply_filters with RSI filter."""
        # Replace momentum filter with RSI filter
        self.symphony.operators = [RSIFilter("RSI Filter", threshold=55, condition="above")]
        
        # Call the method
        filtered_symbols = self.symphony.apply_filters(self.universe, self.market_data, self.technical_data)
        
        # Verify filtering worked as expected (should keep SPY, filter out QQQ and IWM based on RSI)
        self.assertEqual(len(filtered_symbols), 1)
        self.assertIn("SPY", filtered_symbols.symbols)
        self.assertNotIn("QQQ", filtered_symbols.symbols)
        self.assertNotIn("IWM", filtered_symbols.symbols)
    
    def test_apply_filters_no_operators(self):
        """Test apply_filters with no operators."""
        # Remove all operators
        self.symphony.operators = []
        
        # Call the method
        filtered_symbols = self.symphony.apply_filters(self.universe, self.market_data, self.technical_data)
        
        # Verify no filtering occurred (universe unchanged)
        self.assertEqual(len(filtered_symbols), len(self.universe))
        for symbol in self.universe.symbols:
            self.assertIn(symbol, filtered_symbols.symbols)
    
    def test_apply_filters_all_filtered(self):
        """Test apply_filters when all symbols are filtered out."""
        # Add a strict filter that will filter out all symbols
        self.symphony.operators = [RSIFilter("Strict RSI", threshold=80, condition="above")]
        
        # Call the method
        filtered_symbols = self.symphony.apply_filters(self.universe, self.market_data, self.technical_data)
        
        # Verify all symbols were filtered out
        self.assertEqual(len(filtered_symbols), 0)
    
    def test_calculate_allocations_equal_weight(self):
        """Test calculate_allocations with EqualWeightAllocator."""
        # Create symbol list with SPY and QQQ
        symbols = SymbolList(["SPY", "QQQ"])
        
        # Call the method
        allocations = self.symphony.calculate_allocations(symbols, self.market_data)
        
        # Verify equal allocations (0.5 each)
        self.assertEqual(len(allocations), 2)
        self.assertAlmostEqual(allocations["SPY"], 0.5)
        self.assertAlmostEqual(allocations["QQQ"], 0.5)
        self.assertAlmostEqual(sum(allocations.values()), 1.0)
    
    def test_calculate_allocations_inverse_volatility(self):
        """Test calculate_allocations with InverseVolatilityAllocator."""
        # Replace allocator with InverseVolatilityAllocator
        self.symphony.set_allocator(InverseVolatilityAllocator(lookback_days=10))
        
        # Create symbol list with SPY and QQQ
        symbols = SymbolList(["SPY", "QQQ"])
        
        # Call the method
        allocations = self.symphony.calculate_allocations(symbols, self.market_data)
        
        # Verify allocations were made
        self.assertEqual(len(allocations), 2)
        self.assertIn("SPY", allocations)
        self.assertIn("QQQ", allocations)
        self.assertAlmostEqual(sum(allocations.values()), 1.0)
    
    def test_calculate_allocations_empty_symbols(self):
        """Test calculate_allocations with empty symbol list."""
        # Create empty symbol list
        empty_symbols = SymbolList([])
        
        # Call the method
        allocations = self.symphony.calculate_allocations(empty_symbols, self.market_data)
        
        # Verify empty allocations
        self.assertEqual(len(allocations), 0)
    
    def test_calculate_allocations_single_symbol(self):
        """Test calculate_allocations with a single symbol."""
        # Create symbol list with only SPY
        single_symbol = SymbolList(["SPY"])
        
        # Call the method
        allocations = self.symphony.calculate_allocations(single_symbol, self.market_data)
        
        # Verify full allocation to the single symbol
        self.assertEqual(len(allocations), 1)
        self.assertAlmostEqual(allocations["SPY"], 1.0)

if __name__ == '__main__':
    unittest.main()
