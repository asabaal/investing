# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎯 Trading Data Session Manager Demo
#
# Interactive demonstration of the iterative data collection system for trading analysis.
#
# This notebook shows how to build up your dataset step by step, exactly like your trading workflow:
# 1. Start with daily data (full history)
# 2. Add hourly data for specific periods
# 3. Add minute-level data for recent analysis
# 4. Visualize the data growth with beautiful dark mode charts

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up dark theme for all plots
plt.style.use('dark_background')
pio.templates.default = "plotly_dark"

# Custom color palette for dark mode
COLORS = {
    'primary': '#00D4FF',    # Bright cyan
    'secondary': '#FF6B6B',  # Coral red  
    'accent': '#4ECDC4',     # Teal
    'warning': '#FFE66D',    # Golden yellow
    'success': '#95E1D3',    # Mint green
    'background': '#1e1e1e', # Dark gray
    'text': '#ffffff'        # White
}

# Import our SIMPLE data puller - no complex bullshit
from simple_data_puller import SimpleSession

print("🚀 Trading Data Session Demo - SIMPLE VERSION!")
print("📊 Dark mode visualizations enabled")
print("🎯 Let's just get data and analyze it - no complexity!")

# %% [markdown]
# ## 🎯 Step 1: Choose Your Security
#
# Let's start a trading session for a security you're interested in analyzing.

# %%
# Choose your symbol (feel free to change this!)
SYMBOL = 'SPY'  # Change to any symbol you want: AAPL, TSLA, QQQ, etc.

print(f"🎯 Starting SIMPLE data session for {SYMBOL}")
print("=" * 50)

# Create the session - SIMPLE!
session = SimpleSession(SYMBOL)

print(f"✅ Simple session created for {SYMBOL}")
print("📊 Ready to pull data!")

# %% [markdown]
# ## 📊 Step 2: Add Daily Data (Full History)
#
# Just like your workflow - start with the big picture using daily data.

# %%
print("📊 Getting daily data (full history)...")
print("This gives us the big picture view for strategy development")

# Get daily data - SIMPLE!
daily_data = session.get_daily()

if not daily_data.empty:
    print(f"\n✅ Got {len(daily_data)} daily records!")
    print(f"📅 Date range: {daily_data.index.min().strftime('%Y-%m-%d')} to {daily_data.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n📈 Sample of daily data:")
    display(daily_data.tail())
else:
    print("❌ No daily data found")


# %% [markdown]
# ## 🎨 Visualization 1: Daily Data Overview
#
# Beautiful dark mode visualization of your daily data foundation.

# %%
def create_daily_overview(data, symbol):
    """Create beautiful daily data overview"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{symbol} Price History', 
            'Volume Over Time',
            'Price Distribution', 
            'Data Coverage Timeline'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'],
            name='Close Price',
            line=dict(color=COLORS['primary'], width=2),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index[-252:],  # Last year
            y=data['Volume'].iloc[-252:],
            name='Volume',
            marker_color=COLORS['accent'],
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Price distribution
    fig.add_trace(
        go.Histogram(
            x=data['Close'],
            name='Price Distribution',
            marker_color=COLORS['secondary'],
            opacity=0.7,
            nbinsx=50
        ),
        row=2, col=1
    )
    
    # Data coverage timeline
    years = data.index.year.value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=years.index,
            y=years.values,
            name='Records per Year',
            marker_color=COLORS['success'],
            opacity=0.8
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"🎯 {symbol} Daily Data Foundation - {len(data):,} Records",
            font=dict(size=20, color=COLORS['text']),
            x=0.5
        ),
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

if not daily_data.empty:
    fig1 = create_daily_overview(daily_data, SYMBOL)
    fig1.show()
    
    print(f"\n📊 Daily Data Summary:")
    print(f"   Records: {len(daily_data):,}")
    print(f"   Date range: {daily_data.index.min().strftime('%Y-%m-%d')} to {daily_data.index.max().strftime('%Y-%m-%d')}")
    print(f"   Years of data: {(daily_data.index.max() - daily_data.index.min()).days / 365.25:.1f}")

# %% [markdown]
# ## ⏰ Step 3: Add Hourly Data (Specific Time Window)
#
# Now let's add some 60-minute data for more granular analysis - just like you described in your workflow!

# %%
print("⏰ Getting 60-minute data...")
print("This gives us intraday patterns and more precise entry/exit points")

# Get 60min data - SIMPLE!
hourly_data = session.get_intraday('60min')

if not hourly_data.empty:
    print(f"\n✅ Got {len(hourly_data)} 60-minute records!")
    print(f"📅 Date range: {hourly_data.index.min().strftime('%Y-%m-%d %H:%M')} to {hourly_data.index.max().strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n📈 Sample of 60-minute data:")
    display(hourly_data.tail())
else:
    print("❌ No 60-minute data found")


# %% [markdown]
# ## 🎨 Visualization 2: Data Growth Comparison
#
# Show how our dataset is growing with different time resolutions.

# %%
def create_data_growth_viz(session, symbol):
    """Visualize what data we have"""
    
    # Get what we have
    daily_count = len(session.daily) if not session.daily.empty else 0
    intraday_counts = {interval: len(data) for interval, data in session.intraday.items() if not data.empty}
    
    # Prepare data for visualization
    intervals = []
    record_counts = []
    colors = []
    
    # Daily data
    if daily_count > 0:
        intervals.append('Daily')
        record_counts.append(daily_count)
        colors.append(COLORS['primary'])
    
    # Intraday data
    for interval, count in intraday_counts.items():
        intervals.append(interval)
        record_counts.append(count)
        colors.append(COLORS['accent'] if '60min' in interval else COLORS['secondary'])
    
    if not intervals:
        print("No data to visualize yet")
        return None
    
    # Create the visualization
    fig = go.Figure()
    
    # Bar chart of record counts
    fig.add_trace(
        go.Bar(
            x=intervals,
            y=record_counts,
            marker_color=colors,
            text=[f'{count:,}' for count in record_counts],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>Records: %{y:,}<extra></extra>'
        )
    )
    
    total_records = sum(record_counts)
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} Data Portfolio - {total_records:,} Total Records",
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis_title="Data Interval",
        yaxis_title="Number of Records",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    
    return fig

# Show data we have so far
fig2 = create_data_growth_viz(session, SYMBOL)
if fig2:
    fig2.show()

# %% [markdown]
# ## ⚡ Step 4: Add High-Frequency Data (Recent Period)
#
# Let's add some 15-minute data for recent detailed analysis - perfect for fine-tuning entry/exit strategies!

# %%
print("⚡ Getting 15-minute data...")
print("This gives us high-frequency patterns for precise timing")

# Get 15min data - SIMPLE!
minute_data = session.get_intraday('15min')

if not minute_data.empty:
    print(f"\n✅ Got {len(minute_data)} 15-minute records!")
    print(f"📅 Date range: {minute_data.index.min().strftime('%Y-%m-%d %H:%M')} to {minute_data.index.max().strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n📈 Sample of 15-minute data:")
    display(minute_data.tail())
else:
    print("❌ No 15-minute data found")


# %% [markdown]
# ## 🎨 Visualization 3: Multi-Resolution Analysis
#
# Now let's see how different time resolutions give us different perspectives on the same security.

# %%
def create_multi_resolution_viz(session, symbol):
    """Show the same time period across different resolutions"""
    
    # Get recent data from all available intervals
    recent_start = datetime.now() - timedelta(days=30)  # Last month
    recent_start_str = recent_start.strftime('%Y-%m-%d')
    
    # Filter data to recent period
    datasets = []
    
    if not session.daily.empty:
        daily_recent = session.daily[session.daily.index >= recent_start]
        if not daily_recent.empty:
            datasets.append(('Daily', daily_recent))
    
    for interval, data in session.intraday.items():
        if not data.empty:
            recent_data = data[data.index >= recent_start]
            if not recent_data.empty:
                datasets.append((interval, recent_data))
    
    if not datasets:
        print("No recent data available for multi-resolution view")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=len(datasets), cols=1,
        subplot_titles=[f'{name} Resolution ({len(data)} records)' for name, data in datasets],
        vertical_spacing=0.1
    )
    
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary']]
    
    for i, (name, data) in enumerate(datasets):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name=f'{name} Close',
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Close: $%{{y:.2f}}<extra></extra>'
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title=dict(
            text=f"🔍 {symbol} Multi-Resolution Analysis - Recent 30 Days",
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        height=200 * len(datasets) + 100,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    
    return fig

# Show multi-resolution analysis
fig3 = create_multi_resolution_viz(session, SYMBOL)
if fig3:
    fig3.show()

# %% [markdown]
# ## 📊 Step 5: Data Session Summary & Export
#
# Let's see what we've built and save it for future use!

# %%
print("📊 Final Data Summary")
print("=" * 50)

# Show what we collected
session.status()

# Simple export if you want it
print(f"\n💾 Want to export to CSV? Uncomment the lines below:")
print(f"# session.daily.to_csv('{SYMBOL}_daily.csv')")
for interval in session.intraday.keys():
    print(f"# session.intraday['{interval}'].to_csv('{SYMBOL}_{interval}.csv')")

print(f"\n🎯 Data Collection Complete!")
print(f"📊 You now have everything you need for {SYMBOL} analysis")
print(f"💡 Perfect for your trading workflow - fast and simple!")


# %% [markdown]
# ## 🎨 Visualization 4: Final Data Portfolio Dashboard
#
# A comprehensive view of everything we've collected - your trading data portfolio!

# %%
def create_final_dashboard(session, symbol):
    """Create a comprehensive dashboard of all collected data"""
    
    # Get what we have
    daily_count = len(session.daily) if not session.daily.empty else 0
    intraday_counts = {interval: len(data) for interval, data in session.intraday.items() if not data.empty}
    
    # Create a 2x2 dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Data Collection Summary',
            'Record Distribution', 
            'Recent Price Action (All Resolutions)',
            'Trading Hours Heatmap'
        ],
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"type": "scatter"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Data collection summary
    intervals = []
    counts = []
    colors_list = []
    
    if daily_count > 0:
        intervals.append('Daily')
        counts.append(daily_count)
        colors_list.append(COLORS['primary'])
    
    for interval, count in intraday_counts.items():
        intervals.append(interval)
        counts.append(count)
        colors_list.append(COLORS['accent'] if '60min' in interval else COLORS['secondary'])
    
    if intervals:
        fig.add_trace(
            go.Bar(
                x=intervals,
                y=counts,
                marker_color=colors_list,
                text=[f'{c:,}' for c in counts],
                textposition='auto'
            ),
            row=1, col=1
        )
    
    # 2. Pie chart of record distribution
    if intervals:
        fig.add_trace(
            go.Pie(
                labels=intervals,
                values=counts,
                marker_colors=colors_list,
                textinfo='label+percent',
                textfont_size=12
            ),
            row=1, col=2
        )
    
    # 3. Recent price action overlay (last 7 days)
    recent_start = datetime.now() - timedelta(days=7)
    
    # Add daily data
    if not session.daily.empty:
        daily_recent = session.daily[session.daily.index >= recent_start]
        if not daily_recent.empty:
            fig.add_trace(
                go.Scatter(
                    x=daily_recent.index,
                    y=daily_recent['Close'],
                    name='Daily',
                    line=dict(color=COLORS['primary'], width=3),
                    opacity=0.8
                ),
                row=2, col=1
            )
    
    # Add intraday data if available
    for i, (interval, data) in enumerate(session.intraday.items()):
        if not data.empty:
            intraday_recent = data[data.index >= recent_start]
            if not intraday_recent.empty:
                color = COLORS['accent'] if interval == '60min' else COLORS['secondary']
                fig.add_trace(
                    go.Scatter(
                        x=intraday_recent.index,
                        y=intraday_recent['Close'],
                        name=interval,
                        line=dict(color=color, width=1),
                        opacity=0.6
                    ),
                    row=2, col=1
                )
    
    # 4. Trading hours heatmap (if we have intraday data)
    if session.intraday:
        # Use the first intraday dataset for trading hours analysis
        first_intraday = next(iter(session.intraday.values()))
        hours = first_intraday.index.hour
        hour_counts = pd.Series(hours).value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=hour_counts.index,
                y=[1] * len(hour_counts),  # Single row heatmap
                mode='markers',
                marker=dict(
                    size=[count/50 for count in hour_counts.values],  # Scale marker size
                    color=hour_counts.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Records")
                ),
                text=[f'{hour}:00 - {count} records' for hour, count in hour_counts.items()],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    total_records = daily_count + sum(intraday_counts.values())
    fig.update_layout(
        title=dict(
            text=f"🎯 {symbol} Trading Data Dashboard - {total_records:,} Records Ready!",
            font=dict(size=20, color=COLORS['text']),
            x=0.5
        ),
        height=800,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Data Type", row=1, col=1)
    fig.update_yaxes(title_text="Record Count", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
    fig.update_yaxes(title_text="", row=2, col=2)
    
    return fig

# Create and show the final dashboard
fig4 = create_final_dashboard(session, SYMBOL)
if fig4:
    fig4.show()

print(f"\n🎉 SIMPLE Trading Data Demo Complete!")
print(f"🎯 You have a complete dataset for {SYMBOL} ready for analysis")

# Show actual record counts
total_records = len(session.daily) if not session.daily.empty else 0
for interval, data in session.intraday.items():
    total_records += len(data) if not data.empty else 0

print(f"📊 Total records collected: {total_records:,}")
print(f"💡 Fast, simple, and ready for your 10 trades per day!")
print(f"🚀 No complex logic - just pure data access!")


# %% [markdown]
# ## 🚀 Step 6: Your Turn - Try Different Symbols!
#
# Now you can experiment with different symbols and time periods. Just change the SYMBOL variable at the top and run through the cells again!

# %%
# Quick function to test any symbol - SIMPLE VERSION
def quick_symbol_test(symbol, show_viz=True):
    """Quick test of any symbol"""
    print(f"🎯 Quick test for {symbol}")
    print("=" * 40)
    
    # Create simple session
    test_session = SimpleSession(symbol)
    
    # Get all available data
    daily = test_session.get_daily()
    hourly = test_session.get_intraday('60min')
    minute = test_session.get_intraday('15min')
    
    # Show results
    if not daily.empty:
        print(f"✅ {symbol}: {len(daily):,} daily records")
    
    if not hourly.empty:
        print(f"✅ {symbol}: {len(hourly):,} hourly records")
        
    if not minute.empty:
        print(f"✅ {symbol}: {len(minute):,} minute records")
    
    if show_viz and not daily.empty:
        # Quick price chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily['Close'],
                name=f'{symbol} Close',
                line=dict(color=COLORS['primary'], width=2)
            )
        )
        fig.update_layout(
            title=f"📈 {symbol} Price History",
            template="plotly_dark",
            height=400
        )
        fig.show()
    
    return test_session

# Test some popular symbols - FAST AND SIMPLE!
test_symbols = ['TSLA', 'QQQ', 'NVDA']  # Add your favorites here!

print("🧪 Testing other symbols...")
for test_symbol in test_symbols:
    try:
        quick_symbol_test(test_symbol, show_viz=False)
    except Exception as e:
        print(f"❌ {test_symbol}: Error - {e}")
    print()

# %% [markdown]
# ## 💡 Next Steps for Your Trading Workflow
#
# You now have a **SIMPLE** system for data access! Here's how to use it:
#
# ### 🎯 **Daily Trading Workflow - SIMPLE VERSION**
# ```python
# # For each potential trade - JUST 3 LINES:
# from simple_data_puller import SimpleSession
# session = SimpleSession('YOUR_SYMBOL')
# daily = session.get_daily()        # Full history
# hourly = session.get_intraday('60min')  # Recent intraday
# minute = session.get_intraday('15min')  # High frequency
#
# # That's it! Analyze and trade.
# ```
#
# ### 🚀 **Scaling to 10 Trades/Day**
# - **No complex sessions** - just create `SimpleSession('SYMBOL')` each time
# - **Direct data access** - no waiting, no fallbacks, no API management
# - **Use what's available** - system gives you whatever data exists
# - **Fast and predictable** - perfect for rapid trading decisions
#
# ### 📊 **Key Benefits**
# - **SIMPLE**: 3 lines of code, get data, done
# - **FAST**: No complex logic, no redundant calls
# - **RELIABLE**: Just returns what's in the database
# - **SCALABLE**: Perfect for your high-frequency trading workflow
#
# **Stop overthinking. Start trading! 🎯**
