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

# Function to handle API rate limits with exponential backoff
def handle_rate_limit(status_container, status_text, wait_container, attempt=1, max_attempts=5):
    """
    Handle API rate limits with exponential backoff
    
    Parameters:
    -----------
    status_container : streamlit.container
        Streamlit container to display status messages
    status_text : streamlit.empty
        Streamlit text element to display status messages
    wait_container : streamlit.container
        Streamlit container for wait animations
    attempt : int, default 1
        Current attempt number
    max_attempts : int, default 5
        Maximum number of retry attempts
        
    Returns:
    --------
    bool
        True if should retry, False if max attempts reached
    """
    if attempt > max_attempts:
        status_text.error(f"Maximum retry attempts ({max_attempts}) reached. Please try again later.")
        return False
    
    # Calculate wait time with exponential backoff and jitter
    base_wait = min(60 * (2 ** (attempt - 1)), 300)  # Cap at 5 minutes
    jitter = random.uniform(0, 0.1 * base_wait)  # Add 0-10% jitter
    wait_time = base_wait + jitter
    
    # Update session state to indicate waiting
    st.session_state.fetch_state['waiting_for_rate_limit'] = True
    st.session_state.fetch_state['waiting_until'] = time.time() + wait_time
    st.session_state.fetch_state['retry_count'] = attempt
    
    # Display the warning with animated elements
    with status_container:
        st.markdown(f"""
        <div class='api-waiting'>
            <h4>📡 API Rate Limit Reached</h4>
            <p>Attempt {attempt}/{max_attempts} - Waiting {wait_time:.1f} seconds for API rate limit to reset...</p>
            <p class="rate-limit-counter" id="countdown">Wait time remaining: {int(wait_time)} seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a progress bar
    progress_bar = wait_container.progress(0)
    
    # Wait with animated progress bar
    start_time = time.time()
    end_time = start_time + wait_time
    
    while time.time() < end_time:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / wait_time)
        progress_bar.progress(progress)
        
        # Update countdown every second
        seconds_left = max(0, int(end_time - time.time()))
        status_container.markdown(f"""
        <div class='api-waiting'>
            <h4>📡 API Rate Limit Reached</h4>
            <p>Attempt {attempt}/{max_attempts} - Waiting for API rate limit to reset...</p>
            <p class="rate-limit-counter" id="countdown">Wait time remaining: {seconds_left} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.5)  # Update more frequently for smoother animation
    
    # Clear the wait animation
    progress_bar.empty()
    
    # Reset waiting state
    st.session_state.fetch_state['waiting_for_rate_limit'] = False
    st.session_state.fetch_state['waiting_until'] = None
    
    status_container.markdown(f"""
    <div class='active-fetching'>
        <h4>✅ Rate limit reset complete</h4>
        <p>Resuming data fetching operation...</p>
    </div>
    """, unsafe_allow_html=True)
    
    return True

# Function to save checkpoint data
def save_checkpoint(data, ticker, interval, progress):
    """
    Save fetched data to session state as a checkpoint
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the fetched data so far
    ticker : str
        Ticker symbol
    interval : str
        Time interval
    progress : dict
        Progress information including current year, month and count
    """
    if 'checkpoints' not in st.session_state:
        st.session_state.checkpoints = {}
    
    checkpoint_key = f"{ticker}_{interval}"
    st.session_state.checkpoints[checkpoint_key] = {
        "data": data,
        "progress": progress,
        "last_updated": datetime.now()
    }

# Function to load checkpoint data
def load_checkpoint(ticker, interval):
    """
    Load checkpoint data if available
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    interval : str
        Time interval
        
    Returns:
    --------
    tuple or None
        (DataFrame, progress_dict) if checkpoint exists, None otherwise
    """
    if 'checkpoints' not in st.session_state:
        return None
    
    checkpoint_key = f"{ticker}_{interval}"
    if checkpoint_key in st.session_state.checkpoints:
        checkpoint = st.session_state.checkpoints[checkpoint_key]
        return checkpoint["data"], checkpoint["progress"]
    
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

# The rest of the app.py code will be included in the next chunks
# ...
