"""
Manual test script for Alpha Vantage API client.

This script tests the Alpha Vantage API client using your API key.
Run this script to verify your API key is working correctly and that
the data cleaning functions are working as expected.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add parent directory to path so we can import alpha_vantage_api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_vantage_api import AlphaVantageClient

def run_api_test():
    """Run tests for the Alpha Vantage API client."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("API key not found. Please set ALPHA_VANTAGE_API_KEY environment variable.")
        return False
    
    # Create API client
    logger.info("Creating API client with premium subscription")
    client = AlphaVantageClient(api_key=api_key, premium=True)
    
    # Test with sample data to verify data cleaning
    logger.info("Testing data cleaning with sample data")
    dates = pd.date_range(start='2024-01-01', periods=10)
    sample_data = pd.DataFrame({
        'open': np.random.rand(10) * 100,
        'high': np.random.rand(10) * 100,
        'low': np.random.rand(10) * 100,
        'close': np.random.rand(10) * 100,
        'volume': np.random.rand(10) * 1000000
    }, index=dates)
    
    # Add some NaN values to volume
    sample_data.loc[sample_data.index[3:6], 'volume'] = np.nan
    
    logger.info(f"Sample data contains {sample_data['volume'].isna().sum()} NaN values in volume")
    
    # Clean the data
    cleaned_data = client.clean_data(sample_data)
    
    # Check if cleaning worked
    if cleaned_data['volume'].isna().any():
        logger.error("Data cleaning failed! NaN values still present in volume column.")
        return False
    else:
        logger.info("Data cleaning successful! All NaN values were handled.")
    
    # Test API with real data
    try:
        logger.info("Testing API with real data for SPY")
        spy_data = client.get_daily('SPY', outputsize='compact')
        logger.info(f"Retrieved {len(spy_data)} data points for SPY")
        
        # Check for common issues
        if 'volume' in spy_data.columns and spy_data['volume'].isna().any():
            logger.error(f"SPY data contains {spy_data['volume'].isna().sum()} NaN values in volume after cleaning!")
            return False
        
        logger.info("API test successful!")
        return True
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_api_test()
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
