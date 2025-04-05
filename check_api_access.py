"""
Quick script to check Alpha Vantage API key access level.
"""

import os
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_access():
    """
    Check what level of access your Alpha Vantage API key has.
    """
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("API key not found! Please set ALPHA_VANTAGE_API_KEY environment variable.")
        return
    
    logger.info(f"Using API key: {api_key[:4]}...")
    
    # Test a regular endpoint
    standard_endpoint = "https://www.alphavantage.co/query"
    standard_params = {
        'function': 'OVERVIEW',
        'symbol': 'IBM',
        'apikey': api_key
    }
    
    logger.info("Testing standard endpoint (OVERVIEW)...")
    standard_response = requests.get(standard_endpoint, params=standard_params)
    
    if standard_response.status_code == 200:
        data = standard_response.json()
        if 'Error Message' in data:
            logger.error(f"Standard endpoint error: {data['Error Message']}")
        elif 'Information' in data and 'premium' in data['Information']:
            logger.error("Standard endpoint requires premium access!")
            logger.info(f"Response: {data['Information']}")
        else:
            logger.info("Standard endpoint SUCCESS! Basic API access confirmed.")
    else:
        logger.error(f"Standard endpoint failed with status code: {standard_response.status_code}")
    
    # Test a premium endpoint
    premium_endpoint = "https://www.alphavantage.co/query"
    premium_params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': 'SPY',
        'outputsize': 'full',
        'apikey': api_key
    }
    
    logger.info("Testing premium endpoint (TIME_SERIES_DAILY_ADJUSTED with full outputsize)...")
    premium_response = requests.get(premium_endpoint, params=premium_params)
    
    if premium_response.status_code == 200:
        data = premium_response.json()
        if 'Error Message' in data:
            logger.error(f"Premium endpoint error: {data['Error Message']}")
        elif 'Information' in data and 'premium' in data['Information']:
            logger.error("Premium endpoint requires premium subscription!")
            logger.info(f"Response: {data['Information']}")
            logger.info("You need to upgrade your Alpha Vantage subscription to use this feature.")
        else:
            logger.info("Premium endpoint SUCCESS! Premium API access confirmed.")
    else:
        logger.error(f"Premium endpoint failed with status code: {premium_response.status_code}")

if __name__ == "__main__":
    check_api_access()
