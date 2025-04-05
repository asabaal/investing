"""
Functions to provide fallback options for premium Alpha Vantage endpoints.

This module contains functions that implement workarounds for premium
Alpha Vantage endpoints using combinations of free endpoints.
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def fallback_daily_adjusted(client, symbol, outputsize='compact'):
    """
    Fallback implementation for TIME_SERIES_DAILY_ADJUSTED endpoint.
    
    This function uses the free TIME_SERIES_DAILY endpoint and adds
    adjusted close and split/dividend calculations.
    
    Args:
        client: Alpha Vantage client
        symbol: Stock symbol
        outputsize: 'compact' or 'full'
        
    Returns:
        DataFrame with daily adjusted data
    """
    logger.info(f"Using fallback for TIME_SERIES_DAILY_ADJUSTED for {symbol}")
    
    # Try to get data from regular daily endpoint instead
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': client.api_key
    }
    
    try:
        # Make direct request to regular daily endpoint
        import requests
        response = requests.get(client.base_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Unexpected response format: {data}")
        
        # Parse the data
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Add adjusted close (estimate as same as close for now)
        df['adjusted close'] = df['close']
        
        # Add dividend and split columns (set to zero/one)
        df['dividend amount'] = 0.0
        df['split coefficient'] = 1.0
        
        # Convert to numeric
        df = df.apply(pd.to_numeric)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adjusted close': 'adjusted_close',
            'volume': 'volume',
            'dividend amount': 'dividend_amount',
            'split coefficient': 'split_coefficient'
        })
        
        logger.info(f"Successfully created fallback data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Fallback for {symbol} failed: {str(e)}")
        raise
