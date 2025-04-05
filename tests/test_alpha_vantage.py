import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path

# Import your module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_vantage_api import AlphaVantageClient

class TestAlphaVantageAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test_key"
        self.api = AlphaVantageClient(api_key=self.test_api_key)
        
        # Create sample data fixtures
        dates = pd.date_range(start='2024-01-01', periods=10)
        self.sample_data = pd.DataFrame({
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.rand(10) * 1000000
        }, index=dates)
        
        # Create a sample response for TIME_SERIES_DAILY_ADJUSTED
        self.sample_response = {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": "SPY",
                "3. Last Refreshed": "2025-04-05",
                "4. Output Size": "Full size",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": {
                "2025-04-05": {
                    "1. open": "450.32",
                    "2. high": "452.10",
                    "3. low": "449.25",
                    "4. close": "451.89",
                    "5. adjusted close": "451.89",
                    "6. volume": "75534521",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0"
                }
            }
        }
        
        # Create a sample premium error response
        self.premium_error = {
            'Information': 'Thank you for using Alpha Vantage! This is a premium endpoint. You may subscribe to any of the premium plans at https://www.alphavantage.co/premium/ to instantly unlock all premium endpoints'
        }

    @patch('requests.get')
    def test_api_key_included_in_requests(self, mock_get):
        """Test that API key is included in all requests."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_response
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Call the API
        self.api.get_daily('SPY')
        
        # Verify API key was included
        args, kwargs = mock_get.call_args
        self.assertIn('apikey', kwargs['params'])
        self.assertEqual(kwargs['params']['apikey'], self.test_api_key)
    
    @patch('requests.get')
    def test_handle_premium_error(self, mock_get):
        """Test handling of premium endpoint errors."""
        # Setup mock to return premium error
        mock_response = MagicMock()
        mock_response.json.return_value = self.premium_error
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Call the API and check for proper error handling
        with self.assertRaises(ValueError) as context:
            self.api.get_daily('SPY')
        
        self.assertIn('premium endpoint', str(context.exception))
    
    def test_handle_nan_volume(self):
        """Test handling of NaN values in volume data."""
        # Create test data with NaN in volume
        test_data = self.sample_data.copy()
        test_data.loc[test_data.index[3:5], 'volume'] = np.nan
        
        # Call the clean_data method
        cleaned_data = self.api.clean_data(test_data)
        
        # Verify NaN values were handled
        self.assertFalse(cleaned_data['volume'].isna().any())

if __name__ == '__main__':
    unittest.main()