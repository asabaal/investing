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

# Rest of the original code remains unchanged
# ...
