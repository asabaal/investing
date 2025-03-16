import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os.path
from datetime import datetime, timedelta
import base64
from io import BytesIO, StringIO
import requests
import json
import time
import math  # Added missing import for math.ceil
import random  # For jitter in exponential backoff

# Set page configuration
st.set_page_config(
    page_title="Stock Prophet Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    /* New styles for waiting animations */
    .api-waiting {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #ffeeba;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    .rate-limit-counter {
        font-family: monospace;
        font-size: 1.2em;
        font-weight: bold;
    }
    .active-fetching {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #bee5eb;
    }
    .animate-ellipsis::after {
        content: '';
        animation: ellipsis 1.5s infinite;
    }
    @keyframes ellipsis {
        0% { content: '.'; }
        33% { content: '..'; }
        66% { content: '...'; }
        100% { content: ''; }
    }
</style>
""", unsafe_allow_html=True)

# Function to prepare stock data for Prophet (original function, unchanged)
def prepare_stock_data_for_prophet(data_source, date_col='Date', open_col='Open', 
                                  high_col='High', low_col='Low', close_col='Close',
                                  volume_col=None, window_sizes=[5, 10, 20]):
    """
    Prepare stock data for use with Prophet model and create features from OHLC data.
    
    Parameters:
    -----------
    data_source : str or pandas.DataFrame
        Either a path to a CSV file or a pandas DataFrame containing stock data
    date_col : str, default 'Date'
        Column name for the date
    open_col, high_col, low_col, close_col : str
        Column names for OHLC data
    volume_col : str or None, default None
        Column name for volume data, if available
    window_sizes : list, default [5, 10, 20]
        Window sizes for moving average calculations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for Prophet with date as 'ds', close price as 'y', and additional features
    """
    # Load the data
    if isinstance(data_source, str) and os.path.isfile(data_source):
        df = pd.read_csv(data_source)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        raise ValueError("data_source must be either a valid file path or a pandas DataFrame")
    
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date (ascending) to ensure proper calculations
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Rename columns for Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df[date_col]
    prophet_df['y'] = df[close_col]
    
    # Basic OHLC features
    prophet_df['open'] = df[open_col]
    prophet_df['high'] = df[high_col]
    prophet_df['low'] = df[low_col]
    
    # Feature 1: Daily price range (volatility indicator)
    prophet_df['daily_range'] = df[high_col] - df[low_col]
    prophet_df['daily_range_pct'] = prophet_df['daily_range'] / df[close_col]
    
    # Feature 2: Open-Close spread (intraday movement)
    prophet_df['open_close_spread'] = df[close_col] - df[open_col]
    prophet_df['open_close_spread_pct'] = prophet_df['open_close_spread'] / df[open_col]
    
    # Feature 3: Position of Close within the daily range (0-1 value)
    # Values close to 1 indicate closing near the high (bullish)
    # Values close to 0 indicate closing near the low (bearish)
    denominator = df[high_col] - df[low_col]
    # Avoid division by zero when High = Low
    denominator = denominator.replace(0, np.nan)
    prophet_df['close_position'] = (df[close_col] - df[low_col]) / denominator
    prophet_df['close_position'] = prophet_df['close_position'].fillna(0.5)  # Use 0.5 when High = Low
    
    # Feature 4: Log returns
    prophet_df['log_return'] = np.log(df[close_col]).diff()
    
    # Feature 5: Moving averages and their differences
    for window in window_sizes:
        prophet_df[f'ma_{window}'] = df[close_col].rolling(window=window).mean()
        prophet_df[f'ma_{window}_diff'] = prophet_df['y'] - prophet_df[f'ma_{window}']
        prophet_df[f'ma_{window}_diff_pct'] = prophet_df[f'ma_{window}_diff'] / prophet_df[f'ma_{window}']
    
    # Feature 6: Moving average of daily range (volatility)
    for window in window_sizes:
        prophet_df[f'range_ma_{window}'] = prophet_df['daily_range'].rolling(window=window).mean()
    
    # Feature 7: RSI-like momentum indicators
    for window in window_sizes:
        delta = df[close_col].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        prophet_df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        prophet_df[f'rsi_{window}'] = prophet_df[f'rsi_{window}'].fillna(50)  # Use neutral value when undefined
    
    # Add volume if available
    if volume_col and volume_col in df.columns:
        prophet_df['volume'] = df[volume_col]
        # Volume Moving Average
        for window in window_sizes:
            prophet_df[f'volume_ma_{window}'] = df[volume_col].rolling(window=window).mean()
        # Volume ratio compared to moving average
        prophet_df['volume_ratio'] = df[volume_col] / prophet_df['volume_ma_20'].replace(0, np.nan)
        prophet_df['volume_ratio'] = prophet_df['volume_ratio'].fillna(1)
    
    # Drop NaN values that result from window calculations
    prophet_df = prophet_df.dropna()
    
    return prophet_df

# Store API call tracking data in session state
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = []

# Initialize session state for tracking the API fetch state
if 'fetch_state' not in st.session_state:
    st.session_state.fetch_state = {
        'is_fetching': False,
        'current_month': 0,
        'total_months': 0,
        'waiting_for_rate_limit': False,
        'waiting_until': None,
        'retry_count': 0,
        'last_status_update': None
    }
    
# Function to track API calls for rate limit management
def track_api_call(endpoint="intraday"):
    """Record API call timing to manage rate limits"""
    now = time.time()
    st.session_state.api_calls.append({"time": now, "endpoint": endpoint})
    
    # Remove calls older than 1 minute (AlphaVantage typically has a 5 calls per minute limit)
    st.session_state.api_calls = [call for call in st.session_state.api_calls 
                                  if now - call["time"] < 60]
    
def get_rate_limit_status():
    """Return estimated rate limit status"""
    if len(st.session_state.api_calls) == 0:
        return {"calls_made": 0, "calls_remaining": 5, "reset_in": 0}
    
    now = time.time()
    # Count calls in the last minute
    calls_in_last_minute = len(st.session_state.api_calls)
    
    # Estimate time until a slot frees up (when oldest call is more than 1 minute old)
    if calls_in_last_minute >= 5:  # Assuming 5 calls per minute limit
        oldest_call = min([call["time"] for call in st.session_state.api_calls])
        reset_in = max(0, 60 - (now - oldest_call))
    else:
        reset_in = 0
        
    return {
        "calls_made": calls_in_last_minute,
        "calls_remaining": max(0, 5 - calls_in_last_minute),
        "reset_in": reset_in
    }

# Function to get CSV download link
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Calculate exponential backoff wait time with jitter
def get_backoff_wait_time(retry_count, base_seconds=5, max_seconds=120):
    """Calculate exponential backoff wait time with jitter
    
    Parameters:
    -----------
    retry_count : int
        Number of retries so far
    base_seconds : int, default 5
        Base wait time in seconds
    max_seconds : int, default 120
        Maximum wait time in seconds
        
    Returns:
    --------
    int
        Wait time in seconds with some random jitter
    """
    # Exponential backoff formula: base * 2^retry with jitter
    wait_time = min(base_seconds * (2 ** retry_count), max_seconds)
    
    # Add jitter (±10%) to avoid thundering herd problem
    jitter = random.uniform(-0.1, 0.1) * wait_time
    wait_time = max(1, wait_time + jitter)  # Ensure wait time is at least 1 second
    
    return math.ceil(wait_time)  # Round up to nearest second

# Function to fetch multi-month intraday data in batches
def fetch_multi_month_intraday(ticker, api_key, interval, start_year, start_month, end_year, end_month):
    """
    Fetch multiple months of intraday data by making sequential API calls
    while respecting rate limits
    """
    # Update the session state to indicate fetching has started
    st.session_state.fetch_state['is_fetching'] = True
    
    # Create a progress bar
    st.info(f"Fetching historical intraday data from {start_month}/{start_year} to {end_month}/{end_year}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total number of months to fetch
    total_months = (end_year - start_year) * 12 + end_month - start_month + 1
    
    # Update session state with total months
    st.session_state.fetch_state['total_months'] = total_months
    st.session_state.fetch_state['current_month'] = 0
    st.session_state.fetch_state['retry_count'] = 0  # Reset retry counter
    
    # Initialize list to store data from each month
    all_data = []
    
    # Convert month names to numbers if needed
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
                   
    if isinstance(start_month, str) and start_month in month_names:
        start_month = month_names.index(start_month) + 1
    if isinstance(end_month, str) and end_month in month_names:
        end_month = month_names.index(end_month) + 1
    
    # Get current date for calculating slice parameters
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Process each month
    month_count = 0
    
    try:
        # Loop through each year and month
        for year in range(start_year, end_year + 1):
            # Determine start/end months for this year
            if year == start_year:
                first_month = start_month
            else:
                first_month = 1
                
            if year == end_year:
                last_month = end_month
            else:
                last_month = 12
                
            for month in range(first_month, last_month + 1):
                month_count += 1
                
                # Update session state with current month
                st.session_state.fetch_state['current_month'] = month_count
                
                # Update progress
                progress_percentage = min(1.0, month_count / total_months)
                progress_bar.progress(progress_percentage)
                
                status_text.info(f"Fetching data for {month_names[month-1]} {year} ({month_count}/{total_months})")
                
                # Update status in session state
                st.session_state.fetch_state['last_status_update'] = f"Fetching {month_names[month-1]} {year}"
                
                # Calculate months ago from current date
                months_ago = (current_year - year) * 12 + (current_month - month)
                
                # Calculate slice parameter
                # AlphaVantage uses: year1month1 (most recent), year1month2 (month before), etc.
                if months_ago <= 11:
                    slice_year = 1
                    slice_month = months_ago + 1  # +1 because current month is month1
                else:
                    slice_year = 2
                    slice_month = months_ago - 11
                    
                # Skip if out of range (AlphaVantage only supports 2 years)
                if slice_year > 2 or (slice_year == 2 and slice_month > 12):
                    status_text.warning(f"Skipping {month_names[month-1]} {year} - beyond 2 year AlphaVantage limit")
                    continue
                    
                slice_param = f"year{slice_year}month{slice_month}"
                
                # Check rate limit and wait if necessary
                rate_status = get_rate_limit_status()
                if rate_status["calls_remaining"] == 0:
                    # Use exponential backoff for wait time calculation
                    wait_time = get_backoff_wait_time(st.session_state.fetch_state['retry_count'])
                    
                    # Increment retry counter for exponential backoff
                    st.session_state.fetch_state['retry_count'] += 1
                    
                    waiting_bar = st.progress(0)
                    
                    # Update session state to indicate waiting for rate limit
                    st.session_state.fetch_state['waiting_for_rate_limit'] = True
                    st.session_state.fetch_state['waiting_until'] = time.time() + wait_time
                    
                    status_text.warning(f"Rate limit reached. Waiting {wait_time} seconds with exponential backoff (retry #{st.session_state.fetch_state['retry_count']})...")
                    
                    for i in range(wait_time):
                        time.sleep(1)
                        waiting_bar.progress((i + 1) / wait_time)
                    
                    waiting_bar.empty()
                    
                    # Reset waiting status
                    st.session_state.fetch_state['waiting_for_rate_limit'] = False
                    st.session_state.fetch_state['waiting_until'] = None
                
                # Fetch data for this month
                try:
                    # Make API call
                    track_api_call("intraday_extended")
                    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice_param}&apikey={api_key}'
                    response = requests.get(url)
                    
                    if response.status_code != 200:
                        status_text.error(f"Error fetching data: HTTP {response.status_code}")
                        # Increment retry count on failure
                        st.session_state.fetch_state['retry_count'] += 1
                        continue
                    
                    csv_data = response.text
                    
                    # Check for API limit messages
                    if "Thank you for using Alpha Vantage" in csv_data and "Our standard API" in csv_data:
                        status_text.warning(f"API Limit reached: {csv_data}")
                        
                        # Use exponential backoff for API daily limit
                        wait_time = get_backoff_wait_time(st.session_state.fetch_state['retry_count'], 
                                                          base_seconds=15, 
                                                          max_seconds=300)
                        
                        # Increment retry counter for exponential backoff
                        st.session_state.fetch_state['retry_count'] += 1
                        
                        # Update session state to indicate waiting for rate limit
                        st.session_state.fetch_state['waiting_for_rate_limit'] = True
                        st.session_state.fetch_state['waiting_until'] = time.time() + wait_time
                        
                        status_text.warning(f"API daily limit may have been reached. Waiting {wait_time} seconds with exponential backoff (retry #{st.session_state.fetch_state['retry_count']})...")
                        time.sleep(wait_time)
                        
                        # Reset waiting status
                        st.session_state.fetch_state['waiting_for_rate_limit'] = False
                        st.session_state.fetch_state['waiting_until'] = None
                        
                        continue
                    
                    # Parse the CSV data
                    month_df = pd.read_csv(StringIO(csv_data))
                    
                    # Skip if no data or just headers
                    if len(month_df) <= 1:
                        status_text.info(f"No data available for {month_names[month-1]} {year}")
                        continue
                    
                    # Rename columns to match our standard format
                    month_df = month_df.rename(columns={
                        'time': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    
                    # Convert time to datetime
                    month_df['Date'] = pd.to_datetime(month_df['Date'])
                    
                    # Append to our data collection
                    all_data.append(month_df)
                    
                    # Show data count
                    if len(month_df) > 0:
                        status_text.info(f"Added {len(month_df)} data points for {month_names[month-1]} {year}")
                    
                    # Reset retry counter on success
                    st.session_state.fetch_state['retry_count'] = 0
                    
                    # Add slight delay between calls to avoid overwhelming the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    status_text.error(f"Error processing data for {month_names[month-1]} {year}: {str(e)}")
                    # Increment retry count on failure
                    st.session_state.fetch_state['retry_count'] += 1
                    continue
    
        # Combine all the monthly data
        if len(all_data) > 0:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Final status update
            status_text.success(f"✅ Completed! Fetched data from {len(all_data)} months with {len(combined_df)} total data points")
            
            # Reset session state
            st.session_state.fetch_state['is_fetching'] = False
            st.session_state.fetch_state['retry_count'] = 0
            
            return combined_df
        else:
            status_text.error("No data was retrieved. Please check your selections and try again.")
            
            # Reset session state
            st.session_state.fetch_state['is_fetching'] = False
            st.session_state.fetch_state['retry_count'] = 0
            
            return None
            
    except Exception as e:
        # Handle any unexpected exceptions
        status_text.error(f"Unexpected error during data fetch: {str(e)}")
        
        # Reset session state in case of error
        st.session_state.fetch_state['is_fetching'] = False
        st.session_state.fetch_state['retry_count'] = 0
        
        return None

# Function to add a sidebar fetch status indicator
def render_fetch_status_sidebar():
    if 'fetch_state' in st.session_state and st.session_state.fetch_state['is_fetching']:
        st.sidebar.markdown("""
        <div style="background-color: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h4>📊 Active Data Fetch</h4>
            <p>Currently fetching historical data...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show progress
        current = st.session_state.fetch_state['current_month']
        total = st.session_state.fetch_state['total_months']
        if total > 0:
            progress = current / total
            st.sidebar.progress(progress)
            st.sidebar.text(f"Month {current} of {total} ({progress*100:.1f}%)")
        
        # Show waiting status if applicable
        if st.session_state.fetch_state['waiting_for_rate_limit']:
            waiting_until = st.session_state.fetch_state['waiting_until']
            if waiting_until is not None:
                seconds_left = max(0, int(waiting_until - time.time()))
                st.sidebar.warning(f"Waiting for API limit reset: {seconds_left}s")

# Function to fetch stock data from AlphaVantage
def fetch_stock_data(ticker, api_key, data_type='daily', interval='5min', output_size='full', start_date=None, end_date=None, slice_option=None):
    """
    Fetch stock data from AlphaVantage API
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL' for Apple)
    api_key : str
        AlphaVantage API key
    data_type : str, default 'daily'
        'daily' for daily data, 'intraday' for intraday data
    interval : str, default '5min'
        Interval for intraday data: '1min', '5min', '15min', '30min', '60min'
    output_size : str, default 'full'
        'compact' returns the latest 100 data points
        For daily data: 'full' returns up to 20 years of data
        For intraday data: 'full' returns about 1-2 months of data
    start_date : datetime, default None
        Start date for filtering the data (used for post-processing)
    end_date : datetime, default None
        End date for filtering the data (used for post-processing)
    slice_option : str, default None
        For intraday extended history: year and month, e.g., 'year1month1'
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with OHLCV data if successful, None if failed
    """
    try:
        # Track API call time
        start_time = time.time()
        
        # Choose API endpoint based on data type
        if data_type == 'daily':
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize={output_size}&apikey={api_key}'
            time_series_key = 'Time Series (Daily)'
        else:  # intraday
            # For standard intraday data (recent)
            if slice_option is None:
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&outputsize={output_size}&apikey={api_key}'
                time_series_key = f'Time Series ({interval})'
            else:
                # For extended intraday data (historical monthly slices)
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice_option}&apikey={api_key}'
                # The extended API returns CSV directly, not JSON
                try:
                    response = requests.get(url)
                    
                    # For extended intraday, we get CSV directly
                    if response.status_code == 200:
                        # Parse CSV data
                        csv_data = response.text
                        
                        # Handle potential API error messages that come as text
                        if "Thank you for using Alpha Vantage" in csv_data and "Our standard API" in csv_data:
                            st.warning(f"API Limit: {csv_data}")
                            return None
                            
                        if "Error Message" in csv_data:
                            st.error(f"API Error: {csv_data}")
                            return None
                            
                        # Import StringIO for parsing CSV from string
                        from io import StringIO
                        
                        # Parse the CSV
                        df = pd.read_csv(StringIO(csv_data))
                        
                        # Rename columns to match our standard format
                        df = df.rename(columns={
                            'time': 'Date',
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        })
                        
                        # Convert time to datetime
                        df['Date'] = pd.to_datetime(df['Date'])
                        
                        # Sort and apply date filtering
                        df = df.sort_values('Date').reset_index(drop=True)
                        
                        # Apply date filtering if start_date and/or end_date are provided
                        if start_date is not None:
                            df = df[df['Date'] >= pd.Timestamp(start_date)]
                        
                        if end_date is not None:
                            df = df[df['Date'] <= pd.Timestamp(end_date)]
                        
                        # Calculate API call duration
                        end_time = time.time()
                        st.info(f"API call completed in {end_time - start_time:.2f} seconds")
                        
                        return df
                    else:
                        st.error(f"API request failed with status code: {response.status_code}")
                        return None
                        
                except Exception as e:
                    st.error(f"Error processing extended intraday data: {str(e)}")
                    return None
        
        # Make the request
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        
        # Check for rate limiting
        if 'Note' in data and 'call frequency' in data['Note']:
            st.warning(f"API Rate Limit: {data['Note']}")
            
            # Visual counter for rate limit
            with st.spinner('API rate limit reached. Waiting for reset...'):
                # For demonstration, wait 15 seconds with a progress bar
                # In production, you might want to extract the actual wait time from the API response
                wait_time = 15
                progress_bar = st.progress(0)
                for i in range(wait_time):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / wait_time)
                    
                    # Update counter text
                    st.text(f"Waiting: {wait_time - i - 1} seconds remaining")
                
                # Clear the progress bar after waiting
                progress_bar.empty()
                
                # Try the request again
                st.info("Retrying API request...")
                return fetch_stock_data(ticker, api_key, data_type, interval, output_size, start_date, end_date)
        
        if 'Information' in data and 'Our standard API' in data['Information']:
            st.warning(f"API Limit: {data['Information']}")
        
        # Check if we have time series data
        if time_series_key not in data:
            st.error("No data found. API may be rate-limited or the ticker symbol may be invalid.")
            return None
        
        # Parse the time series data
        time_series = data[time_series_key]
        rows = []
        
        for date, values in time_series.items():
            # For intraday, the keys are different
            if data_type == 'daily':
                row = {
                    'Date': date,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adjusted Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume']),
                    'Dividend Amount': float(values['7. dividend amount']),
                    'Split Coefficient': float(values['8. split coefficient'])
                }
            else:  # intraday
                row = {
                    'Date': date,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                }
            rows.append(row)
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Apply date filtering if start_date and/or end_date are provided
        if start_date is not None:
            df = df[df['Date'] >= start_date]
        
        if end_date is not None:
            df = df[df['Date'] <= end_date]
        
        # Check if we have data after filtering
        if len(df) == 0:
            st.warning(f"No data found for the selected date range: {start_date} to {end_date}")
            return None
        
        # Calculate API call duration
        end_time = time.time()
        st.info(f"API call completed in {end_time - start_time:.2f} seconds")
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to plot monthly forecast summary
def plot_monthly_summary(forecast):
    # Extract year and month from the forecast dates
    forecast['year_month'] = forecast['ds'].dt.to_period('M')
    
    # Group by month and calculate monthly stats
    monthly_forecast = forecast.groupby('year_month').agg({
        'yhat': ['mean', 'first', 'last', 'min', 'max'],
        'yhat_lower': 'min',
        'yhat_upper': 'max'
    }).reset_index()
    
    # Flatten the MultiIndex columns
    monthly_forecast.columns = ['_'.join(col).strip('_') for col in monthly_forecast.columns.values]
    
    # Convert period to datetime for plotting
    monthly_forecast['month'] = monthly_forecast['year_month_'].dt.to_timestamp()
    
    # Create a monthly summary plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the monthly mean forecast
    ax.plot(monthly_forecast['month'], monthly_forecast['yhat_mean'], 'ro-', linewidth=2, label='Monthly Mean')
    
    # Plot the range from min to max
    ax.fill_between(monthly_forecast['month'], monthly_forecast['yhat_min'], 
                    monthly_forecast['yhat_max'], color='orange', alpha=0.2, label='Min-Max Range')
    
    # Plot the confidence interval
    ax.fill_between(monthly_forecast['month'], monthly_forecast['yhat_lower_min'], 
                    monthly_forecast['yhat_upper_max'], color='lightblue', alpha=0.2, label='95% Confidence Interval')
    
    # Add month start and end points
    ax.plot(monthly_forecast['month'], monthly_forecast['yhat_first'], 'b^', label='Month Open')
    ax.plot(monthly_forecast['month'], monthly_forecast['yhat_last'], 'gv', label='Month Close')
    
    # Add labels and title
    ax.set_title('Monthly Forecast Summary', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis to show month names
    plt.xticks(monthly_forecast['month'], monthly_forecast['month'].dt.strftime('%b %Y'), rotation=45)
    plt.tight_layout()
    
    return fig, monthly_forecast

# Function to create plain English analysis
def generate_analysis(data, forecast):
    last_price = data['y'].iloc[-1]
    last_date = data['ds'].iloc[-1].strftime('%Y-%m-%d')
    forecast_end = forecast['ds'].iloc[-1].strftime('%Y-%m-%d')
    forecast_end_price = forecast['yhat'].iloc[-1]
    
    # Calculate price change
    price_change = forecast_end_price - last_price
    pct_change = (price_change / last_price) * 100
    
    # Determine overall trend direction
    if pct_change > 5:
        trend = "strongly positive"
    elif pct_change > 0:
        trend = "slightly positive"
    elif pct_change > -5:
        trend = "slightly negative"
    else:
        trend = "strongly negative"
        
    # Get monthly breakdown
    forecast['year_month'] = forecast['ds'].dt.to_period('M')
    monthly = forecast.groupby('year_month').agg({
        'yhat': ['first', 'last']
    })
    monthly.columns = ['_'.join(col).strip('_') for col in monthly.columns.values]
    monthly['change'] = monthly['yhat_last'] - monthly['yhat_first']
    monthly['pct_change'] = (monthly['change'] / monthly['yhat_first']) * 100
    
    best_month = monthly['pct_change'].idxmax().strftime('%B %Y')
    worst_month = monthly['pct_change'].idxmin().strftime('%B %Y')
    
    # Volatility assessment
    volatility = (forecast['yhat_upper'] - forecast['yhat_lower']).mean() / forecast['yhat'].mean() * 100
    if volatility > 20:
        vol_assessment = "high"
    elif volatility > 10:
        vol_assessment = "moderate"
    else:
        vol_assessment = "low"
    
    analysis = f"""
    ## Forecast Analysis
    
    Based on historical data up to **{last_date}** (last closing price: **${last_price:.2f}**), 
    the forecast through **{forecast_end}** shows an overall **{trend}** trend.
    
    The model predicts a final price of **${forecast_end_price:.2f}**, which represents 
    a **{pct_change:.1f}%** change from the current price.
    
    ### Monthly Breakdown
    - Best expected month: **{best_month}**
    - Most challenging month: **{worst_month}**
    
    ### Confidence & Volatility
    The forecast shows **{vol_assessment}** volatility with an average uncertainty range of **{volatility:.1f}%**.
    
    ### Investment Perspective
    """
    
    if trend == "strongly positive":
        analysis += "This forecast suggests a strong bullish outlook, potentially favorable for long positions."
    elif trend == "slightly positive":
        analysis += "The slight positive trend indicates a moderately bullish outlook, though caution is advised."
    elif trend == "slightly negative":
        analysis += "The slight negative trend suggests a cautious approach, possibly reducing exposure."
    else:
        analysis += "The strongly negative trend indicates bearish conditions, suggesting defensive positioning."
    
    return analysis

# Main application header
st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("Upload your stock data and generate forecasts with confidence intervals")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Data & Model Settings")
    
    # Data source tabs: File Upload or AlphaVantage API
    data_source = st.radio("Choose Data Source", ["Upload CSV", "AlphaVantage API"])
    
    if data_source == "Upload CSV":
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file with stock data", type=["csv"])
        
        st.subheader("Column Names")
        date_col = st.text_input("Date column", "Date")
        open_col = st.text_input("Open column", "Open")
        high_col = st.text_input("High column", "High")
        low_col = st.text_input("Low column", "Low")
        close_col = st.text_input("Close column", "Close")
        
        volume_col = st.text_input("Volume column (optional)", "Volume")
        if volume_col == "":
            volume_col = None
    else:
        # AlphaVantage API inputs
        st.subheader("Stock Selection")
        ticker = st.text_input("Ticker Symbol (e.g., AAPL, MSFT, GOOG)", "AAPL").upper()
        
        # Store API key in session state to persist across reruns
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
            
        api_key = st.text_input("AlphaVantage API Key", 
                                st.session_state.api_key, 
                                type="password", 
                                help="Get a free API key from alphavantage.co")
        st.session_state.api_key = api_key
        
        # Data type selection (daily or intraday)
        data_type = st.radio("Data Type", ["Daily", "Intraday"])
        
        if data_type == "Intraday":
            interval = st.select_slider(
                "Select Interval",
                options=["1min", "5min", "15min", "30min", "60min"],
                value="5min"
            )
            
            # Add intraday data range options
            st.subheader("Intraday Data Range")
            
            # AlphaVantage intraday data explanation
            st.info("""
            **Note about AlphaVantage intraday data:**
            - Free API typically provides recent data for standard intraday calls
            - Historical data can be accessed via monthly slices (up to 2 years back)
            - API is limited to 5 calls per minute and 500 calls per day
            """)
            
            # Show current rate limit status
            rate_status = get_rate_limit_status()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("API Calls Available", f"{rate_status['calls_remaining']}/5 per minute")
            with col2:
                if rate_status['reset_in'] > 0:
                    st.metric("Reset In", f"{int(rate_status['reset_in'])} seconds")
                else:
                    st.metric("Status", "Ready")
            
            # Data range selector
            intraday_range = st.radio(
                "Select data range",
                options=[
                    "Recent trading days",
                    "Custom date range",
                    "Monthly slice",
                    "Multi-month historical data"
                ],
                index=0
            )
            
            if intraday_range == "Recent trading days":
                # Calculate default dates
                today = datetime.now().date()
                
                # Select number of days
                days_back = st.slider("Number of trading days", 1, 30, 5, 
                                     help="Select 1-30 recent trading days")
                
                start_date = today - timedelta(days=days_back + 2)  # Add weekend buffer
                end_date = today
                
                st.info(f"Will fetch approximately {days_back} trading days of data")
                
                # Use compact for small ranges
                output_size = "compact" if days_back <= 7 else "full"
                
                # Set a default value for multi_month_params
                multi_month_params = None
                
            elif intraday_range == "Custom date range":
                # Calculate default dates (last 7 trading days)
                today = datetime.now().date()
                default_start = today - timedelta(days=7)
                
                # Add date pickers
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date", 
                        value=default_start,
                        help="Select the starting date for intraday data"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=today,
                        help="Select the ending date for intraday data"
                    )
                
                # Calculate date difference
                date_diff = (end_date - start_date).days
                
                if date_diff > 30:
                    st.warning(f"""
                    Selected range is {date_diff} days. AlphaVantage may limit intraday data to about 1 month.
                    Consider using 'Multi-month historical data' for longer historical periods.
                    """)
                
                output_size = "compact" if date_diff <= 7 else "full"
                
                # Set a default value for multi_month_params
                multi_month_params = None
                
            elif intraday_range == "Monthly slice":
                st.info("""
                AlphaVantage provides historical intraday data in monthly slices.
                Select a specific year and month to analyze.
                """)
                
                # Year and month selectors
                current_year = datetime.now().year
                current_month = datetime.now().month
                
                # Create year options (current year and 2 years back)
                year_options = list(range(current_year - 2, current_year + 1))
                selected_year = st.selectbox("Year", options=year_options, index=len(year_options)-1)
                
                # Create month options
                month_names = ["January", "February", "March", "April", "May", "June", 
                             "July", "August", "September", "October", "November", "December"]
                
                # Limit current year to current month
                if selected_year == current_year:
                    month_options = month_names[:current_month]
                    month_index = min(current_month - 1, len(month_options) - 1)
                else:
                    month_options = month_names
                    month_index = 0
                
                selected_month_name = st.selectbox("Month", options=month_options, index=month_index)
                selected_month = month_names.index(selected_month_name) + 1
                
                # Set date range based on selected month
                if selected_year == current_year and selected_month == current_month:
                    # For current month, use days up to today
                    start_date = datetime(selected_year, selected_month, 1).date()
                    end_date = datetime.now().date()
                else:
                    # For other months, use full month
                    start_date = datetime(selected_year, selected_month, 1).date()
                    
                    # Calculate last day of month
                    if selected_month == 12:
                        next_month = datetime(selected_year + 1, 1, 1).date()
                    else:
                        next_month = datetime(selected_year, selected_month + 1, 1).date()
                    
                    end_date = next_month - timedelta(days=1)
                
                # For monthly slice, always use full
                output_size = "full"
                
                st.info(f"Will fetch data for {selected_month_name} {selected_year}")
                
                # Set slice option
                slice_param = f"year{1 if selected_year == current_year else 2}month{abs(current_month - selected_month) + 1}"
                
            else:  # Multi-month historical data
                st.info("""
                This option allows fetching data spanning multiple months by making sequential API calls.
                For long date ranges, this may take some time due to API rate limits.
                """)
                
                # Date range selection for multi-month
                col1, col2 = st.columns(2)
                
                # Current date for reference
                current_year = datetime.now().year
                current_month = datetime.now().month
                
                # Start date selection
                with col1:
                    st.subheader("Start Period")
                    
                    # Only allow up to 2 years back due to API limitations
                    start_year_options = list(range(current_year - 2, current_year + 1))
                    start_year = st.selectbox("Start Year", options=start_year_options, 
                                             index=0, key="start_year")
                    
                    # Month options depend on year
                    if start_year == current_year:
                        start_month_options = list(range(1, current_month + 1))
                        start_month_index = 0
                    else:
                        start_month_options = list(range(1, 13))
                        start_month_index = 0
                    
                    start_month = st.selectbox("Start Month", 
                                              options=start_month_options,
                                              index=start_month_index,
                                              format_func=lambda m: month_names[m-1],
                                              key="start_month")
                
                # End date selection
                with col2:
                    st.subheader("End Period")
                    
                    # End year options depend on start year
                    end_year_options = list(range(start_year, current_year + 1))
                    end_year = st.selectbox("End Year", options=end_year_options, 
                                           index=len(end_year_options)-1, key="end_year")
                    
                    # Month options depend on year
                    if end_year == current_year:
                        end_month_options = list(range(1, current_month + 1))
                        end_month_index = len(end_month_options) - 1
                    else:
                        end_month_options = list(range(1, 13))
                        end_month_index = len(end_month_options) - 1
                    
                    # If same year as start, limit months to >= start_month
                    if end_year == start_year:
                        end_month_options = [m for m in end_month_options if m >= start_month]
                        end_month_index = min(len(end_month_options) - 1, 0)
                    
                    end_month = st.selectbox("End Month", 
                                            options=end_month_options,
                                            index=end_month_index,
                                            format_func=lambda m: month_names[m-1],
                                            key="end_month")
                
                # Calculate and display the number of months
                total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
                st.info(f"Selected range: {total_months} months ({month_names[start_month-1]} {start_year} to {month_names[end_month-1]} {end_year})")
                
                # Warning for large ranges
                if total_months > 12:
                    st.warning(f"""
                    You've selected {total_months} months of data. This will require {total_months} API calls,
                    which may take some time due to rate limits. Consider selecting a shorter range if possible.
                    """)
                    
                # Additional filtering options
                with st.expander("Post-Fetch Filtering Options"):
                    st.info("These options let you filter the data after fetching to focus on specific dates/times.")
                    
                    use_date_filter = st.checkbox("Filter by specific dates", value=False)
                    
                    if use_date_filter:
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            filter_start_date = st.date_input("Filter Start Date", 
                                                            value=datetime(start_year, start_month, 1).date())
                        with filter_col2:
                            filter_end_date = st.date_input("Filter End Date", 
                                                          value=datetime(end_year, end_month, 28).date())
                    else:
                        filter_start_date = datetime(start_year, start_month, 1).date()
                        
                        # Calculate end date (last day of end month)
                        if end_month == 12:
                            next_month = datetime(end_year + 1, 1, 1).date()
                        else:
                            next_month = datetime(end_year, end_month + 1, 1).date()
                        
                        filter_end_date = next_month - timedelta(days=1)
                    
                    # Time filtering
                    use_time_filter = st.checkbox("Filter by specific times", value=False)
                    
                    if use_time_filter:
                        time_col1, time_col2 = st.columns(2)
                        with time_col1:
                            start_time = st.time_input("Start Time", 
                                                      datetime.strptime("09:30", "%H:%M").time())
                        with time_col2:
                            end_time = st.time_input("End Time", 
                                                    datetime.strptime("16:00", "%H:%M").time())
                    else:
                        start_time = None
                        end_time = None
                
                # Store parameters for multi-month fetching
                multi_month_params = {
                    "start_year": start_year,
                    "start_month": start_month,
                    "end_year": end_year,
                    "end_month": end_month,
                    "post_filter": use_date_filter or use_time_filter,
                    "filter_start_date": filter_start_date,
                    "filter_end_date": filter_end_date,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                # Special settings for multi-month
                output_size = "full"
                
                # Set date range for display
                start_date = datetime(start_year, start_month, 1).date()
                
                # Calculate last day of end month
                if end_month == 12:
                    next_month = datetime(end_year + 1, 1, 1).date()
                else:
                    next_month = datetime(end_year, end_month + 1, 1).date()
                
                end_date = next_month - timedelta(days=1)
            
            # Add time pickers for more granular control (for non-multi-month options)
            if intraday_range != "Multi-month historical data":
                st.subheader("Time Range (Optional)")
                use_time_filter = st.checkbox("Filter by specific times", value=False)
                
                if use_time_filter:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_time = st.time_input("Start Time", datetime.strptime("09:30", "%H:%M").time())
                    with col2:
                        end_time = st.time_input("End Time", datetime.strptime("16:00", "%H:%M").time())
                else:
                    start_time = None
                    end_time = None
                
                time_filter_msg = f" between {start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')}" if use_time_filter else ""
                st.info(f"Will fetch {interval} intraday data from {start_date} to {end_date}{time_filter_msg}")
        else:
            interval = "5min"  # Default, won't be used for daily data
            data_amount = st.radio("Amount of Data", ["Full History (20+ years)", "Recent (100 days)"], index=1)
            output_size = "full" if data_amount == "Full History (20+ years)" else "compact"
            
            # Set default values for date variables to avoid errors
            start_date = None
            end_date = None
            start_time = None
            end_time = None
            use_time_filter = False
        
        # Fixed column names for AlphaVantage data
        date_col = "Date"
        open_col = "Open"
        high_col = "High"
        low_col = "Low"
        close_col = "Close"
        volume_col = "Volume"
        
        # Add a "Load Data" button for AlphaVantage
        load_data = st.button("Load Stock Data")
    
    st.subheader("Forecast Settings")
    forecast_periods = st.slider("Forecast period (days)", 30, 365, 180)
    
    interval_width = st.slider("Confidence interval width (%)", 50, 95, 80, 5)
    
    st.subheader("Features to Include")
    use_daily_range = st.checkbox("Daily price range", True)
    use_close_position = st.checkbox("Close position in range", True)
    use_rsi = st.checkbox("RSI indicators", False)

# Render fetch status indicator in sidebar
render_fetch_status_sidebar()

# Main content area
stock_data = None

# Load data based on selected source
if data_source == "Upload CSV" and uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner("Loading and processing stock data..."):
        # Read the CSV file
        stock_data = pd.read_csv(uploaded_file)
        
        # Show data info
        st.subheader(f"Data Loaded: CSV File")
        
elif data_source == "AlphaVantage API" and load_data:
    if not api_key:
        st.error("Please enter an AlphaVantage API key")
    elif not ticker:
        st.error("Please enter a ticker symbol")
    else:
        with st.spinner(f"Fetching data for {ticker}..."):
            # Get the data type
            av_data_type = "intraday" if data_type == "Intraday" else "daily"
            
            # Create a container for API status updates
            api_status = st.empty()
            
            # Prepare slice parameter
            slice_param = None
            
            # For multi-month historical data
            if av_data_type == "intraday" and intraday_range == "Multi-month historical data":
                # Create a container for the multi-month fetching status
                multi_month_status = st.empty()
                multi_month_status.info("Ready to fetch multi-month historical data")
                
                # Fetch data using the special multi-month function
                stock_data = fetch_multi_month_intraday(
                    ticker, 
                    api_key, 
                    interval,
                    multi_month_params['start_year'],
                    multi_month_params['start_month'],
                    multi_month_params['end_year'],
                    multi_month_params['end_month']
                )
                
                # Apply post-filtering if requested
                if stock_data is not None and multi_month_params['post_filter']:
                    original_count = len(stock_data)
                    
                    # Filter by date
                    stock_data = stock_data[
                        (stock_data['Date'].dt.date >= multi_month_params['filter_start_date']) & 
                        (stock_data['Date'].dt.date <= multi_month_params['filter_end_date'])
                    ]
                    
                    # Filter by time if requested
                    if multi_month_params['start_time'] is not None and multi_month_params['end_time'] is not None:
                        stock_data['time'] = stock_data['Date'].dt.time
                        stock_data = stock_data[
                            (stock_data['time'] >= multi_month_params['start_time']) & 
                            (stock_data['time'] <= multi_month_params['end_time'])
                        ]
                        stock_data.drop('time', axis=1, inplace=True)
                    
                    filtered_count = len(stock_data)
                    st.info(f"Applied post-filtering: {original_count} → {filtered_count} data points")
            else:
                # For regular data fetching with other options
                if av_data_type == "intraday":
                    # Convert date inputs to datetime objects
                    fetch_start_date = datetime.combine(start_date, datetime.min.time())
                    fetch_end_date = datetime.combine(end_date, datetime.max.time())
                    
                    # Apply time filters if specified
                    if use_time_filter:
                        # Create proper datetime objects combining the date and time
                        fetch_start_date = datetime.combine(start_date, start_time)
                        fetch_end_date = datetime.combine(end_date, end_time)
                    
                    date_range_str = f" from {fetch_start_date.strftime('%Y-%m-%d')} to {fetch_end_date.strftime('%Y-%m-%d')}"
                    if use_time_filter:
                        date_range_str += f" between {start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')}"
                    
                    # Set slice parameter for monthly slice option
                    if intraday_range == "Monthly slice":
                        date_range_str += f" (using slice: {slice_param})"
                else:
                    fetch_start_date = None
                    fetch_end_date = None
                    date_range_str = ""
                
                # Display the API call details
                api_status.info(f"Fetching {av_data_type} data for {ticker} {'with ' + interval + ' interval' if av_data_type == 'intraday' else ''}{date_range_str}")
                
                # Track this API call
                track_api_call(av_data_type)
                
                # Fetch the data
                stock_data = fetch_stock_data(
                    ticker, 
                    api_key, 
                    av_data_type, 
                    interval, 
                    output_size,
                    fetch_start_date if av_data_type == "intraday" else None,
                    fetch_end_date if av_data_type == "intraday" else None,
                    slice_param
                )
            
            # Apply additional time filtering if needed
            if stock_data is not None and av_data_type == "intraday" and use_time_filter:
                # Extract time from datetime for filtering
                stock_data['time'] = stock_data['Date'].dt.time
                
                # Filter by time of day
                stock_data = stock_data[
                    (stock_data['time'] >= start_time) & 
                    (stock_data['time'] <= end_time)
                ]
                
                # Remove the temporary time column
                stock_data.drop('time', axis=1, inplace=True)
                
                # Check if we still have data after time filtering
                if len(stock_data) == 0:
                    st.error(f"No data available for the selected time range: {start_time} to {end_time}")
                    stock_data = None
            
            if stock_data is not None:
                st.success(f"Successfully loaded data for {ticker}")
                st.subheader(f"Data Loaded: {ticker} ({av_data_type.capitalize()}{'/' + interval if av_data_type == 'intraday' else ''})")
                
                # Add metadata about the stock
                earliest_date = stock_data['Date'].min().strftime('%Y-%m-%d %H:%M:%S' if av_data_type == 'intraday' else '%Y-%m-%d')
                latest_date = stock_data['Date'].max().strftime('%Y-%m-%d %H:%M:%S' if av_data_type == 'intraday' else '%Y-%m-%d')
                num_points = len(stock_data)
                
                st.info(f"Period: {earliest_date} to {latest_date} ({num_points} data points)")
                
                # Show current price and basic stats
                current_price = stock_data.iloc[-1]['Close']
                
                if len(stock_data) > 1:
                    previous_price = stock_data.iloc[-2]['Close']
                    price_change = current_price - previous_price
                    pct_change = (price_change / previous_price) * 100
                else:
                    price_change = 0
                    pct_change = 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Last Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
                with col3:
                    # For intraday, show the day's range. For daily, show the 52-week range
                    if av_data_type == 'intraday':
                        # Group by date to get daily stats
                        today_data = stock_data[stock_data['Date'].dt.date == stock_data['Date'].iloc[-1].date()]
                        day_high = today_data['High'].max()
                        day_low = today_data['Low'].min()
                        st.metric("Day's Range", f"${day_low:.2f} - ${day_high:.2f}")
                    else:  # daily
                        if len(stock_data) > 252:  # 252 trading days in a year
                            year_data = stock_data.iloc[-252:]
                            high_52wk = year_data['High'].max()
                            low_52wk = year_data['Low'].min()
                            st.metric("52-Week Range", f"${low_52wk:.2f} - ${high_52wk:.2f}")
                        else:
                            all_high = stock_data['High'].max()
                            all_low = stock_data['Low'].min()
                            st.metric("Price Range", f"${all_low:.2f} - ${all_high:.2f}")

# If we have stock data (from either source), process it
if stock_data is not None:
    # Show raw data preview
    with st.expander("View raw data"):
        st.dataframe(stock_data.head())
    
    # Prepare data for Prophet
    with st.spinner("Preparing data for forecasting..."):
        prophet_data = prepare_stock_data_for_prophet(
            stock_data, 
            date_col=date_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col
        )
        
    # Show Prophet data preview
    with st.expander("View processed data"):
        st.dataframe(prophet_data.head())
        st.markdown(get_csv_download_link(prophet_data, 
                                        "prophet_ready_data.csv", 
                                        "Download processed data"), 
                 unsafe_allow_html=True)
    
    # Model training and forecasting
    if st.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Select features to use as regressors
            features = []
            if use_daily_range:
                features.append('daily_range_pct')
            if use_close_position:
                features.append('close_position')
            if use_rsi:
                features.append('rsi_14')
            
            # Create and train the model
            model = Prophet(interval_width=interval_width/100)
            
            # Add regressors
            for feature in features:
                model.add_regressor(feature)
            
            # Fit the model
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods)
            
            # Add regressor values to future dataframe
            recent_data = prophet_data.iloc[-30:]  # Last 30 days
            
            for feature in features:
                # Initialize with NaN
                future[feature] = np.nan
                
                # For historical dates, use actual values
                historical_values = prophet_data.set_index('ds')[feature]
                future.loc[future['ds'].isin(prophet_data['ds']), feature] = \
                    future.loc[future['ds'].isin(prophet_data['ds']), 'ds'].map(historical_values)
                
                # For future dates, use the mean of recent values
                future.loc[~future['ds'].isin(prophet_data['ds']), feature] = recent_data[feature].mean()
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Show forecast metrics
            st.subheader("Forecast Overview")
            
            # Create two columns for metrics
            col1, col2, col3 = st.columns(3)
            
            # Last historical date and price
            last_date = prophet_data['ds'].max()
            last_price = prophet_data['y'].iloc[-1]
            
            # Final forecast date and price
            final_date = forecast['ds'].max()
            final_price = forecast['yhat'].iloc[-1]
            final_lower = forecast['yhat_lower'].iloc[-1]
            final_upper = forecast['yhat_upper'].iloc[-1]
            
            # Price change
            price_change = final_price - last_price
            pct_change = (price_change / last_price) * 100
            
            # Display metrics
            with col1:
                st.markdown('<div class="metric-card">' +
                            f'<div class="metric-value">${last_price:.2f}</div>' +
                            f'<div class="metric-label">Current Price ({last_date.strftime("%Y-%m-%d")})</div>' +
                            '</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">' +
                            f'<div class="metric-value">${final_price:.2f}</div>' +
                            f'<div class="metric-label">Forecast Price ({final_date.strftime("%Y-%m-%d")})</div>' +
                            '</div>', unsafe_allow_html=True)
            
            with col3:
                color = "green" if pct_change >= 0 else "red"
                st.markdown('<div class="metric-card">' +
                            f'<div class="metric-value" style="color:{color}">{pct_change:+.2f}%</div>' +
                            f'<div class="metric-label">Projected Change</div>' +
                            '</div>', unsafe_allow_html=True)
            
            # Interactive forecast plot
            st.subheader("Interactive Forecast Visualization")
            
            # Create interactive Plotly chart
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['y'],
                mode='markers',
                name='Historical Data',
                marker=dict(color='blue', size=4)
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,182,193,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{interval_width}% Confidence Interval'
            ))
            
            # Add vertical line for forecast start
            fig.add_vline(x=last_date, line_dash="dash", line_color="green", 
                        annotation_text="Forecast Start", annotation_position="top right")
            
            # Update layout
            fig.update_layout(
                title='Stock Price Forecast',
                yaxis_title='Price ($)',
                xaxis_title='Date',
                hovermode='x unified',
                template='plotly_white',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add range slider and selectors
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="todate"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Component breakdown
            with st.expander("Forecast Components"):
                # Plot the components
                components_fig = plot_components_plotly(model, forecast)
                components_fig.update_layout(height=800)
                st.plotly_chart(components_fig, use_container_width=True)
            
            # Monthly summary
            st.subheader("Monthly Forecast Summary")
            
            monthly_fig, monthly_data = plot_monthly_summary(forecast)
            st.pyplot(monthly_fig)
            
            # Table of monthly predictions
            with st.expander("Monthly Price Targets"):
                # Prepare data for the table
                table_data = monthly_data[['year_month_', 'yhat_first', 'yhat_last', 'yhat_mean', 'yhat_lower_min', 'yhat_upper_max']]
                table_data = table_data.rename(columns={
                    'year_month_': 'Month',
                    'yhat_first': 'Open ($)',
                    'yhat_last': 'Close ($)',
                    'yhat_mean': 'Average ($)',
                    'yhat_lower_min': 'Lower Bound ($)',
                    'yhat_upper_max': 'Upper Bound ($)'
                })
                
                # Calculate monthly change
                table_data['Change (%)'] = ((table_data['Close ($)'] - table_data['Open ($)']) / table_data['Open ($)'] * 100).round(2)
                
                # Convert Period to string for the month
                table_data['Month'] = table_data['Month'].astype(str)
                
                # Style the table with green/red colors for positive/negative changes
                def highlight_change(val):
                    if isinstance(val, float):
                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                        return f'color: {color}'
                    return ''
                
                styled_table = table_data.style.format({
                    'Open ($)': '${:.2f}',
                    'Close ($)': '${:.2f}',
                    'Average ($)': '${:.2f}',
                    'Lower Bound ($)': '${:.2f}',
                    'Upper Bound ($)': '${:.2f}',
                    'Change (%)': '{:.2f}%'
                }).applymap(highlight_change, subset=['Change (%)'])
                
                st.dataframe(styled_table)
            
            # Analysis in plain English
            st.markdown(generate_analysis(prophet_data, forecast))
            
            # Download options
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(get_csv_download_link(forecast, 
                                                "forecast_results.csv", 
                                                "Download full forecast data"), 
                          unsafe_allow_html=True)
            
            with col2:
                st.markdown(get_csv_download_link(table_data, 
                                                "monthly_forecast.csv", 
                                                "Download monthly summary"), 
                          unsafe_allow_html=True)
                
else:
    # Display instructions when no data is loaded
    if data_source == "Upload CSV":
        st.info("👈 Please upload a CSV file with stock data using the sidebar options")
        
        # Display example format
        st.subheader("Expected CSV format:")
        
        sample_data = {
            'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'Open': [100.0, 101.5, 103.0],
            'High': [102.0, 104.0, 105.5],
            'Low': [99.0, 100.5, 102.0],
            'Close': [101.5, 103.0, 104.5],
            'Volume': [1000000, 1200000, 950000]
        }
        
        st.dataframe(pd.DataFrame(sample_data))
        
        st.markdown("""
        ### Instructions:
        1. Upload a CSV file with your stock data
        2. Verify the column names match your data
        3. Adjust forecast settings as needed
        4. Click "Generate Forecast" to run the model
        
        Your CSV should contain historical OHLC (Open, High, Low, Close) data with dates.
        Volume is optional but recommended for better forecasts.
        """)
    else:
        st.info("👈 Enter a ticker symbol and API key, then click 'Load Stock Data'")
        
        # Show AlphaVantage info
        st.markdown("""
        ### Using AlphaVantage API:
        1. Get a free API key from [AlphaVantage](https://www.alphavantage.co/support/#api-key)
        2. Enter your ticker symbol (e.g., AAPL, MSFT, GOOG)
        3. Click "Load Stock Data"
        4. Adjust forecast settings as needed
        5. Click "Generate Forecast" to run the model
        
        Free API keys are limited to 5 requests per minute and 500 requests per day.
        
        ### Popular Ticker Symbols:
        - **Tech**: AAPL (Apple), MSFT (Microsoft), GOOG (Google), META (Meta/Facebook), AMZN (Amazon)
        - **Finance**: JPM (JP Morgan), BAC (Bank of America), GS (Goldman Sachs)
        - **Retail**: WMT (Walmart), TGT (Target), COST (Costco)
        - **Energy**: XOM (Exxon Mobil), CVX (Chevron)
        - **Healthcare**: JNJ (Johnson & Johnson), PFE (Pfizer), MRNA (Moderna)
        """)
        
        # Display sample chart
        st.subheader("Sample Visualization:")
        
        # Create a simple example chart
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        sample_df = pd.DataFrame({'Date': dates, 'Close': close_prices})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_df['Date'], y=sample_df['Close'], mode='lines', name='Stock Price'))
        fig.update_layout(title='Example Stock Price Chart', xaxis_title='Date', yaxis_title='Price ($)')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Stock Forecast Dashboard - Created with Streamlit and Prophet")

# Add a brief help section
with st.expander("Help & Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    
    #### AlphaVantage API
    - **API Key Not Working**: Make sure you're using a valid AlphaVantage API key
    - **Rate Limit Exceeded**: Free API keys are limited to 5 requests per minute and 500 per day
    - **Invalid Ticker**: Check that you're using the correct ticker symbol
    - **No Data Found**: For intraday data, try adjusting the date and time range
    
    #### Forecasting
    - **Error in Model Training**: Try using fewer features or a shorter forecast period
    - **Wide Confidence Intervals**: This indicates high uncertainty - consider using more historical data
    - **Poor Forecast Quality**: Stock prices are inherently difficult to predict - use forecasts as just one input for investment decisions
    
    ### Tips for Better Forecasts:
    - Use at least 1 year of historical data (preferably 2+ years)
    - The 'Daily price range' and 'Close position in range' features often provide the best signals
    - Adjust the confidence interval width to see different risk scenarios
    - Monthly forecasts are generally more reliable than daily forecasts
    - For intraday data, focus on specific trading sessions for more accurate forecasts
    
    ### Need Help?
    For any technical issues or questions, please contact support.
    """)

# Add cache cleanup to prevent memory issues on repeated use
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.experimental_rerun()