"""
Debug script to troubleshoot how the API key is being loaded in symphony_cli.py
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_api_key_loading():
    """Debug how the API key is loaded in the application."""
    logger.info("Checking environment variables...")
    alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    if alpha_vantage_key:
        logger.info(f"Found API key in environment: {alpha_vantage_key[:4]}***{alpha_vantage_key[-4:]}")
    else:
        logger.warning("API key not found in environment variable ALPHA_VANTAGE_API_KEY")
    
    # Check .bashrc
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        logger.info(f"Checking {bashrc_path}...")
        with open(bashrc_path, 'r') as f:
            content = f.read()
            if 'ALPHA_VANTAGE_API_KEY' in content:
                logger.info("Found API key reference in .bashrc")
                # Extract the line with the API key
                for line in content.split('\n'):
                    if 'ALPHA_VANTAGE_API_KEY' in line and 'export' in line:
                        logger.info(f"Export line: {line}")
            else:
                logger.warning("No API key reference found in .bashrc")
    
    # Check custom API key file
    api_key_path = "/home/asabaal/api_keys/alpha_vantage.api"
    if os.path.exists(api_key_path):
        logger.info(f"Checking {api_key_path}...")
        with open(api_key_path, 'r') as f:
            content = f.read().strip()
            logger.info(f"API key file content length: {len(content)}")
            if len(content) > 8:
                logger.info(f"API key in file: {content[:4]}***{content[-4:]}")
    else:
        logger.warning(f"API key file not found: {api_key_path}")
    
    # Now try to make a direct API call
    logger.info("Making direct API call with environment API key...")
    if alpha_vantage_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': 'SPY',
                'outputsize': 'compact',  # Use compact for a faster test
                'apikey': alpha_vantage_key
            }
            
            logger.info(f"Request URL: {url}")
            logger.info(f"Request params: {json.dumps({k: v if k != 'apikey' else f'{v[:4]}***{v[-4:]}' for k, v in params.items()})}")
            
            response = requests.get(url, params=params)
            logger.info(f"Response status: {response.status_code}")
            
            data = response.json()
            if 'Time Series (Daily)' in data:
                days = len(data['Time Series (Daily)'])
                logger.info(f"Success! Retrieved {days} days of data.")
                return True
            elif 'Information' in data and 'premium endpoint' in data['Information']:
                logger.error("Premium endpoint error:")
                logger.error(data['Information'])
                return False
            else:
                logger.error("Unexpected response:")
                logger.error(json.dumps(data, indent=2)[:500] + "...")
                return False
        except Exception as e:
            logger.error(f"Exception during API call: {str(e)}")
            return False
    else:
        logger.error("Cannot make API call without API key in environment")
        return False

def debug_symphony_loading():
    """Debug how the symphony file is loaded."""
    # Try to locate and load sample_symphony.json
    logger.info("Looking for sample_symphony.json...")
    
    current_dir = os.getcwd()
    logger.info(f"Current directory: {current_dir}")
    
    sample_path = os.path.join(current_dir, "sample_symphony.json")
    if os.path.exists(sample_path):
        logger.info(f"Found at: {sample_path}")
        try:
            with open(sample_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded JSON with {len(data)} keys")
                if 'name' in data:
                    logger.info(f"Symphony name: {data['name']}")
                if 'universe' in data and 'symbols' in data['universe']:
                    logger.info(f"Universe symbols: {', '.join(data['universe']['symbols'])}")
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
    else:
        logger.warning(f"sample_symphony.json not found in current directory")
        # Try to find it elsewhere
        found = False
        for dirpath, dirnames, filenames in os.walk(os.path.expanduser("~")):
            if "sample_symphony.json" in filenames:
                full_path = os.path.join(dirpath, "sample_symphony.json")
                logger.info(f"Found at: {full_path}")
                found = True
                break
        
        if not found:
            logger.error("Could not find sample_symphony.json anywhere in home directory")

def debug_api_client_import():
    """Debug how the AlphaVantageClient is being imported."""
    logger.info("Checking AlphaVantageClient import...")
    
    try:
        from alpha_vantage_api import AlphaVantageClient
        logger.info("Successfully imported AlphaVantageClient")
        
        # Create a client instance
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        if api_key:
            logger.info(f"Creating client with API key: {api_key[:4]}***{api_key[-4:]}")
            client = AlphaVantageClient(api_key=api_key)
            logger.info(f"Client created successfully: {client}")
            
            # Examine the client attributes
            logger.info(f"Client base_url: {client.base_url}")
            logger.info(f"Client premium flag: {getattr(client, 'premium', 'Not set')}")
            logger.info(f"Client use_fallbacks flag: {getattr(client, 'use_fallbacks', 'Not set')}")
            
            # Try a mock request by inspecting how the client would form a request
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': 'SPY',
                'outputsize': 'compact'
            }
            request_params = params.copy()
            request_params['apikey'] = api_key
            
            logger.info(f"Request would be sent to: {client.base_url}")
            logger.info(f"Request parameters: {json.dumps({k: v if k != 'apikey' else f'{v[:4]}***{v[-4:]}' for k, v in request_params.items()})}")
            
            return True
        else:
            logger.error("Cannot create client without API key")
            return False
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def main():
    """Run all debug functions."""
    print("\n===== SYMPHONY DEBUG TOOL =====\n")
    
    print("1. Debug API Key Loading:")
    debug_api_key_loading()
    
    print("\n2. Debug Symphony File Loading:")
    debug_symphony_loading()
    
    print("\n3. Debug AlphaVantageClient Import:")
    debug_api_client_import()
    
    print("\n===== DEBUG COMPLETE =====\n")

if __name__ == "__main__":
    main()
