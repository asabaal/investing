"""
Test script specifically for premium API key verification.
This makes a direct API call without using our client library
to eliminate any potential issues in our code.
"""

import os
import requests
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_premium_key():
    """Test if the Alpha Vantage API key has premium access."""
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment. Please set ALPHA_VANTAGE_API_KEY.")
        return False
    
    logger.info(f"Using API key: {api_key[:4]}***{api_key[-4:]}")
    
    # Test a premium endpoint
    url = "https://www.alphavantage.co/query"
    premium_params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': 'SPY',
        'outputsize': 'full',
        'apikey': api_key
    }
    
    # Make direct request
    logger.info("Making direct API request to premium endpoint...")
    logger.info(f"URL: {url}")
    logger.info(f"Params: {json.dumps({k: v for k, v in premium_params.items() if k != 'apikey'})}")
    
    response = requests.get(url, params=premium_params)
    
    # Check response status
    logger.info(f"Response status code: {response.status_code}")
    
    # Try to parse response
    try:
        data = response.json()
        
        # Check if we got premium content or an error
        if 'Information' in data and 'premium endpoint' in data['Information']:
            logger.error("Premium access denied! Full response:")
            logger.error(json.dumps(data, indent=2))
            return False
        elif 'Time Series (Daily)' in data:
            num_days = len(data['Time Series (Daily)'])
            logger.info(f"Premium access confirmed! Retrieved {num_days} days of data.")
            
            # Print a sample of the first day's data
            first_day = list(data['Time Series (Daily)'].keys())[0]
            logger.info(f"Sample data for {first_day}:")
            logger.info(json.dumps(data['Time Series (Daily)'][first_day], indent=2))
            
            return True
        else:
            logger.error("Unexpected response format:")
            logger.error(json.dumps(data, indent=2))
            return False
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.error(f"Response text: {response.text[:500]}")
        return False

def main():
    """Run the test and print a summary."""
    print("\n===== ALPHA VANTAGE PREMIUM KEY TEST =====\n")
    
    # Print environment variables related to Alpha Vantage
    env_vars = [var for var in os.environ if 'ALPHA' in var.upper() or 'VANTAGE' in var.upper()]
    if env_vars:
        print("Found environment variables related to Alpha Vantage:")
        for var in env_vars:
            value = os.environ[var]
            # Mask API keys for security
            if 'KEY' in var.upper() and len(value) > 8:
                value = f"{value[:4]}***{value[-4:]}"
            print(f"  {var} = {value}")
    else:
        print("No environment variables related to Alpha Vantage found!")
        
    print("\nYour current API key as set in ALPHA_VANTAGE_API_KEY:")
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'NOT SET')
    if api_key != 'NOT SET' and len(api_key) > 8:
        print(f"  {api_key[:4]}***{api_key[-4:]}")
    else:
        print(f"  {api_key}")
    
    print("\nTesting premium access...")
    result = test_premium_key()
    
    print("\n===== TEST RESULTS =====")
    if result:
        print("✅ SUCCESS: Your API key has premium access!")
        print("\nRecommended next steps:")
        print("1. Make sure your API key is correctly set in your application")
        print("2. Check if any other parameters are incorrect")
    else:
        print("❌ FAILURE: Your API key does not have premium access.")
        print("\nRecommended next steps:")
        print("1. Double-check that you're using the correct API key")
        print("2. Verify your premium subscription status with Alpha Vantage")
        print("3. Make sure ALPHA_VANTAGE_API_KEY environment variable is set correctly")
        print("4. Try adding your API key to your .bashrc or .profile file:")
        print(f"   echo 'export ALPHA_VANTAGE_API_KEY={api_key}' >> ~/.bashrc")
        print("   source ~/.bashrc")
    
    print("\n=================================\n")

if __name__ == "__main__":
    main()
