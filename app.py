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

# Function to fetch multi-month intraday data in batches
def fetch_multi_month_intraday(ticker, api_key, interval, start_year, start_month, end_year, end_month, 
                               use_checkpoint=True, resume=False):
    """
    Fetch multiple months of intraday data by making sequential API calls
    while respecting rate limits
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    api_key : str
        AlphaVantage API key
    interval : str
        Time interval for intraday data
    start_year : int
        Starting year
    start_month : int
        Starting month
    end_year : int
        Ending year
    end_month : int
        Ending month
    use_checkpoint : bool, default True
        Whether to save checkpoints during fetching
    resume : bool, default False
        Whether to resume from a previous checkpoint
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with combined intraday data if successful, None if failed
    """
    # Mark that we're starting a fetch operation
    st.session_state.fetch_state['is_fetching'] = True
    
    # Create containers for status updates - use containers to keep UI elements in place
    status_container = st.container()
    info_container = status_container.empty()
    status_text = status_container.empty()
    progress_container = st.container()
    wait_container = st.container()  # Separate container for wait animations
    data_preview_container = st.container()  # To show data previews
    
    with status_container:
        st.markdown(f"""
        <div class='active-fetching'>
            <h4>📊 Data Fetch Operation</h4>
            <p>Preparing to fetch historical intraday data from {start_month}/{start_year} to {end_month}/{end_year}</p>
            <p class="animate-ellipsis">Initializing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Convert month names to numbers if needed
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
                   
    if isinstance(start_month, str) and start_month in month_names:
        start_month = month_names.index(start_month) + 1
    if isinstance(end_month, str) and end_month in month_names:
        end_month = month_names.index(end_month) + 1
    
    # Calculate total number of months to fetch
    total_months = (end_year - start_year) * 12 + end_month - start_month + 1
    
    # Update session state
    st.session_state.fetch_state['total_months'] = total_months
    
    # Try to load checkpoint data if resume is True
    checkpoint_data = None
    initial_month_count = 0
    
    if resume:
        checkpoint_result = load_checkpoint(ticker, interval)
        if checkpoint_result:
            checkpoint_data, checkpoint_progress = checkpoint_result
            initial_month_count = checkpoint_progress.get("month_count", 0)
            
            if initial_month_count > 0:
                status_text.info(f"Resuming from checkpoint: {initial_month_count}/{total_months} months already fetched")
                
                # Calculate the starting point for resuming
                months_elapsed = initial_month_count
                current_year = start_year
                current_month = start_month
                
                while months_elapsed > 0:
                    if current_month == 12:
                        current_year += 1
                        current_month = 1
                    else:
                        current_month += 1
                    months_elapsed -= 1
                
                # Update start point
                start_year = current_year
                start_month = current_month
    
    # Initialize with checkpoint data or empty list
    all_data = [checkpoint_data] if checkpoint_data is not None else []
    
    # Create progress bar
    with progress_container:
        progress_bar = st.progress(initial_month_count / total_months if total_months > 0 else 0)
    
    # Get current date for calculating slice parameters
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Process each month
    month_count = initial_month_count
    
    # Loop through each year and month
    try:
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
                st.session_state.fetch_state['current_month'] = month_count
                
                # Update progress
                progress_percentage = min(1.0, month_count / total_months) if total_months > 0 else 1.0
                progress_bar.progress(progress_percentage)
                
                # Update status with active fetching indicator
                status_container.markdown(f"""
                <div class='active-fetching'>
                    <h4>📊 Fetching Data: Month {month_count} of {total_months}</h4>
                    <p>Currently processing: <strong>{month_names[month-1]} {year}</strong></p>
                    <p>Progress: {month_count}/{total_months} months ({(progress_percentage * 100):.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
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
                retry_attempt = 1
                
                while rate_status["calls_remaining"] == 0:
                    if not handle_rate_limit(status_container, status_text, wait_container, retry_attempt):
                        # Save checkpoint before exiting if rate limit handling fails
                        if use_checkpoint and len(all_data) > 0:
                            combined_df = pd.concat(all_data, ignore_index=True)
                            save_checkpoint(combined_df, ticker, interval, {"month_count": month_count - 1})
                            status_text.info(f"Checkpoint saved with {len(combined_df)} data points")
                        
                        # Reset fetching state
                        st.session_state.fetch_state['is_fetching'] = False
                        return None
                    
                    # Update rate status
                    rate_status = get_rate_limit_status()
                    retry_attempt += 1
                
                # Fetch data for this month with retries
                max_api_attempts = 3
                api_attempt = 1
                month_data = None
                
                while api_attempt <= max_api_attempts and month_data is None:
                    try:
                        # Make API call
                        track_api_call("intraday_extended")
                        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice_param}&apikey={api_key}'
                        
                        # Add logging for debugging
                        status_text.info(f"API Call ({api_attempt}/{max_api_attempts}): {ticker}, {interval}, {slice_param}")
                        
                        # Update last status update time
                        st.session_state.fetch_state['last_status_update'] = time.time()
                        
                        response = requests.get(url, timeout=30)  # Add timeout
                        
                        if response.status_code != 200:
                            status_text.error(f"Error fetching data: HTTP {response.status_code}")
                            api_attempt += 1
                            time.sleep(5)  # Wait before retry
                            continue
                        
                        csv_data = response.text
                        
                        # Check for API limit messages
                        if "Thank you for using Alpha Vantage" in csv_data and "Our standard API" in csv_data:
                            # This is likely a rate limit or API limit message
                            status_container.markdown(f"""
                            <div class='api-waiting'>
                                <h4>⚠️ API Limit Message Received</h4>
                                <p>{csv_data}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Check if it's a daily limit message
                            if "standard API call frequency is" in csv_data:
                                status_container.markdown(f"""
                                <div class='api-waiting'>
                                    <h4>⚠️ Daily API Limit Reached</h4>
                                    <p>This appears to be a daily API limit message. The application will wait 60 seconds before retrying.</p>
                                    <p>If this continues, you may need to wait until tomorrow or use a different API key.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Wait with animated countdown
                                wait_time = 60
                                for i in range(wait_time, 0, -1):
                                    status_container.markdown(f"""
                                    <div class='api-waiting'>
                                        <h4>⚠️ Daily API Limit Reached</h4>
                                        <p>Waiting to retry: {i} seconds remaining</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    time.sleep(1)
                            else:
                                # Likely a per-minute limit
                                if not handle_rate_limit(status_container, status_text, wait_container, api_attempt, max_api_attempts):
                                    # Save checkpoint before exiting
                                    if use_checkpoint and len(all_data) > 0:
                                        combined_df = pd.concat(all_data, ignore_index=True)
                                        save_checkpoint(combined_df, ticker, interval, {"month_count": month_count - 1})
                                        status_text.info(f"Checkpoint saved with {len(combined_df)} data points")
                                    
                                    # Reset fetching state
                                    st.session_state.fetch_state['is_fetching'] = False
                                    return None
                            
                            api_attempt += 1
                            continue
                        
                        # Parse the CSV data
                        try:
                            month_df = pd.read_csv(StringIO(csv_data))
                            
                            # Skip if no data or just headers
                            if len(month_df) <= 1:
                                status_text.info(f"No data available for {month_names[month-1]} {year}")
                                month_data = pd.DataFrame()  # Empty but valid DataFrame
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
                            
                            # Data validation
                            if not all(col in month_df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
                                status_text.warning(f"Invalid data format for {month_names[month-1]} {year}. Retrying...")
                                api_attempt += 1
                                time.sleep(5)
                                continue
                            
                            month_data = month_df
                            
                        except Exception as e:
                            status_text.error(f"Error parsing CSV data: {str(e)}")
                            api_attempt += 1
                            time.sleep(5)  # Wait before retry
                            continue
                        
                    except Exception as e:
                        status_text.error(f"Error during API request: {str(e)}")
                        api_attempt += 1
                        time.sleep(5)  # Wait before retry
                        continue
                
                # Check if we got data after all retries
                if month_data is None or api_attempt > max_api_attempts:
                    status_text.error(f"Failed to fetch data for {month_names[month-1]} {year} after {max_api_attempts} attempts")
                    
                    # Save checkpoint before continuing
                    if use_checkpoint and len(all_data) > 0:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        save_checkpoint(combined_df, ticker, interval, {"month_count": month_count - 1})
                        status_text.info(f"Checkpoint saved with {len(combined_df)} data points")
                    
                    # Continue to next month
                    continue
                
                # Append to our data collection if we have data
                if len(month_data) > 0:
                    all_data.append(month_data)
                    
                    # Show data count and preview
                    status_container.markdown(f"""
                    <div class='active-fetching'>
                        <h4>✅ Data Received: {month_names[month-1]} {year}</h4>
                        <p>Added {len(month_data)} data points</p>
                        <p>Total data points so far: {sum(len(df) for df in all_data if df is not None)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show a small preview of the latest data
                    with data_preview_container:
                        with st.expander(f"Preview data for {month_names[month-1]} {year}", expanded=False):
                            st.dataframe(month_data.head())
                    
                    # Save checkpoint every 3 months
                    if use_checkpoint and month_count % 3 == 0:
                        combined_checkpoint = pd.concat(all_data, ignore_index=True)
                        save_checkpoint(combined_checkpoint, ticker, interval, {"month_count": month_count})
                        status_text.info(f"Checkpoint saved with {len(combined_checkpoint)} data points")
                
                # Add slight delay between calls to avoid overwhelming the API
                # Only delay if we're not at the end
                if not (year == end_year and month == last_month):
                    delay_seconds = 1  # Base delay
                    
                    status_container.markdown(f"""
                    <div class='active-fetching'>
                        <h4>🕒 Preparing for next month</h4>
                        <p>Waiting {delay_seconds} seconds before making the next API call...</p>
                        <p>This helps prevent overwhelming the API and improves reliability.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(delay_seconds)
    
    except Exception as e:
        status_text.error(f"Unexpected error: {str(e)}")
        
        # Save checkpoint before exiting
        if use_checkpoint and len(all_data) > 0:
            try:
                combined_df = pd.concat(all_data, ignore_index=True)
                save_checkpoint(combined_df, ticker, interval, {"month_count": month_count})
                status_text.info(f"Emergency checkpoint saved with {len(combined_df)} data points")
            except Exception as save_err:
                status_text.error(f"Error saving checkpoint: {str(save_err)}")
        
        # Reset fetching state
        st.session_state.fetch_state['is_fetching'] = False
        return None
    
    # Combine all the monthly data
    if len(all_data) > 0:
        try:
            # Show processing status
            status_container.markdown(f"""
            <div class='active-fetching'>
                <h4>🔄 Processing Data</h4>
                <p>Combining {len(all_data)} months of data...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Combine all the data frames
            valid_data = [df for df in all_data if df is not None and not df.empty]
            
            if not valid_data:
                status_text.error("No valid data collected during the operation.")
                st.session_state.fetch_state['is_fetching'] = False
                return None
                
            combined_df = pd.concat(valid_data, ignore_index=True)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Remove duplicates that might occur at month boundaries
            original_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset='Date', keep='first')
            deduped_count = len(combined_df)
            
            if original_count > deduped_count:
                status_text.info(f"Removed {original_count - deduped_count} duplicate data points")
            
            # Cleanup checkpoint if successful
            if use_checkpoint:
                checkpoint_key = f"{ticker}_{interval}"
                if 'checkpoints' in st.session_state and checkpoint_key in st.session_state.checkpoints:
                    # Keep checkpoint but mark as complete
                    st.session_state.checkpoints[checkpoint_key]["complete"] = True
            
            # Final status update
            status_container.markdown(f"""
            <div class='active-fetching' style='background-color: #d4edda; color: #155724; border-left-color: #c3e6cb;'>
                <h4>✅ Data Fetch Complete!</h4>
                <p>Successfully fetched data from {len(valid_data)} months with {len(combined_df)} total data points</p>
                <p>Date range: {combined_df['Date'].min().strftime('%Y-%m-%d %H:%M')} to {combined_df['Date'].max().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset fetching state
            st.session_state.fetch_state['is_fetching'] = False
            
            return combined_df
        except Exception as e:
            status_text.error(f"Error combining data: {str(e)}")
            st.session_state.fetch_state['is_fetching'] = False
            return None
    else:
        status_text.error("No data was retrieved. Please check your selections and try again.")
        st.session_state.fetch_state['is_fetching'] = False
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
