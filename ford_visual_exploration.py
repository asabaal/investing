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
# # 🚗 Ford Motor Company (F) - Visual Market Exploration
#
# **Deep dive into an affordable security with comprehensive data analysis**
#
# - **Symbol**: F (Ford Motor Company)
# - **Price Range**: ~$10-15 (Very affordable!)
# - **Analysis**: 1-minute intraday data with beautiful dark mode visualizations
# - **Goal**: Understand price patterns, volume dynamics, and trading opportunities
#
# **Let's explore Ford's market behavior! 🚀**

# %%
# Setup and imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Dark theme setup
pio.templates.default = "plotly_dark"

# Import our data systems
from direct_api_puller import get_intraday_data_api, get_daily_data_api
from market_data_database import MarketDataDatabase

# Color palette for Ford-themed visualizations
FORD_COLORS = {
    'primary': '#003478',    # Ford Blue
    'secondary': '#FF6B35',  # Ford Orange
    'accent': '#00D4FF',     # Bright Blue
    'success': '#4ECDC4',    # Teal
    'warning': '#FFE66D',    # Yellow
    'danger': '#FF6B6B'      # Red
}

SYMBOL = 'F'
print(f"🚗 Ford Motor Company ({SYMBOL}) - Visual Market Exploration")
print("=" * 60)

# %% [markdown]
# ## 📊 Step 1: Collect Fresh Data

# %%
# Collect comprehensive Ford data
print(f"📡 Collecting Ford ({SYMBOL}) market data...")

# Get daily data for context
print("📅 Getting daily data (full history)...")
daily_data = get_daily_data_api(SYMBOL)
current_price = daily_data['close'].iloc[-1]
print(f"✅ Daily data: {len(daily_data):,} records")
print(f"💰 Current price: ${current_price:.2f}")

# Get recent 1-minute data for detailed analysis
print("\n⏰ Getting 1-minute intraday data (last 30 days)...")
minute_data = get_intraday_data_api(SYMBOL, '1min')
print(f"✅ 1-minute data: {len(minute_data):,} records")

if not minute_data.empty:
    print(f"📅 Range: {minute_data.index.min()} to {minute_data.index.max()}")
    display(minute_data.tail())

# Get additional timeframes for comparison
print("\n📊 Getting 15-minute data for pattern analysis...")
data_15min = get_intraday_data_api(SYMBOL, '15min', '2025-07-01', '2025-08-05')
print(f"✅ 15-minute data: {len(data_15min):,} records")

print(f"\n🎯 Data collection complete! Ready for visual analysis.")


# %% [markdown]
# ## 📈 Step 2: Price Movement Overview

# %%
# Create comprehensive price overview
def create_price_overview(daily, minute, symbol):
    """Create multi-timeframe price overview"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            f'{symbol} Long-term Price History (Daily)', 'Recent Price Action (1-minute)',
            'Volume Analysis', 'Price Distribution',
            'Daily Returns', 'Volatility Patterns'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08
    )
    
    # 1. Long-term daily price history
    recent_daily = daily.tail(252)  # Last year
    fig.add_trace(
        go.Scatter(
            x=recent_daily.index, 
            y=recent_daily['close'],
            name='Daily Close',
            line=dict(color=FORD_COLORS['primary'], width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Recent 1-minute price action
    if not minute.empty:
        recent_minute = minute.tail(1000)  # Last ~16 hours of trading
        fig.add_trace(
            go.Scatter(
                x=recent_minute.index,
                y=recent_minute['close'],
                name='1-min Close',
                line=dict(color=FORD_COLORS['accent'], width=1),
                hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Volume analysis
    fig.add_trace(
        go.Scatter(
            x=recent_daily.index,
            y=recent_daily['volume'],
            name='Daily Volume',
            fill='tonexty',
            line=dict(color=FORD_COLORS['secondary'], width=1),
            hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Price distribution
    fig.add_trace(
        go.Histogram(
            x=recent_daily['close'],
            name='Price Distribution',
            nbinsx=30,
            marker_color=FORD_COLORS['success'],
            opacity=0.7,
            hovertemplate='Price Range: $%{x:.2f}<br>Frequency: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Daily returns
    daily_returns = recent_daily['close'].pct_change().dropna() * 100
    colors = [FORD_COLORS['success'] if x >= 0 else FORD_COLORS['danger'] for x in daily_returns]
    
    fig.add_trace(
        go.Scatter(
            x=daily_returns.index,
            y=daily_returns,
            mode='markers',
            name='Daily Returns',
            marker=dict(color=colors, size=4),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Volatility (20-day rolling std)
    volatility = daily_returns.rolling(20).std()
    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility,
            name='20-day Volatility',
            line=dict(color=FORD_COLORS['warning'], width=2),
            hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"📊 {symbol} - Comprehensive Price Analysis Dashboard",
        height=900,
        showlegend=False,
        template="plotly_dark"
    )
    
    # Add current price annotation
    current_price = daily['close'].iloc[-1]
    fig.add_annotation(
        x=0.02, y=0.98,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=False,
        font=dict(size=16, color=FORD_COLORS['accent']),
        bgcolor="rgba(0,0,0,0.5)",
        xref="paper", yref="paper"
    )
    
    return fig

# Create and show the overview
overview_fig = create_price_overview(daily_data, minute_data, SYMBOL)
overview_fig.show()

# Print key statistics
print(f"\n📊 Key Statistics for {SYMBOL}:")
print(f"Current Price: ${daily_data['close'].iloc[-1]:.2f}")
print(f"52-week High: ${daily_data['high'].tail(252).max():.2f}")
print(f"52-week Low: ${daily_data['low'].tail(252).min():.2f}")
print(f"Average Volume: {daily_data['volume'].tail(30).mean():,.0f}")
returns = daily_data['close'].pct_change().dropna() * 100
print(f"30-day Volatility: {returns.tail(30).std():.2f}%")


# %% [markdown]
# ## 🕐 Step 3: Intraday Trading Patterns

# %%
# Analyze intraday trading patterns
def analyze_intraday_patterns(minute_data, symbol):
    """Deep dive into intraday trading patterns"""
    
    if minute_data.empty:
        print("No minute data available for intraday analysis")
        return
    
    # Prepare data
    df = minute_data.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute']/60
    df['returns'] = df['close'].pct_change() * 10000  # Basis points
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Average Price by Hour', 'Volume by Hour',
            'Volatility Heatmap', 'Recent Candlestick Chart'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "candlestick"}]
        ]
    )
    
    # 1. Average price by hour
    hourly_price = df.groupby('hour')['close'].mean()
    fig.add_trace(
        go.Scatter(
            x=hourly_price.index,
            y=hourly_price.values,
            mode='lines+markers',
            name='Avg Price',
            line=dict(color=FORD_COLORS['primary'], width=3),
            marker=dict(size=8),
            hovertemplate='Hour: %{x}:00<br>Avg Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Volume by hour
    hourly_volume = df.groupby('hour')['volume'].mean()
    fig.add_trace(
        go.Bar(
            x=hourly_volume.index,
            y=hourly_volume.values,
            name='Avg Volume',
            marker_color=FORD_COLORS['secondary'],
            hovertemplate='Hour: %{x}:00<br>Avg Volume: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Volatility heatmap (hour vs day of week)
    df['day_of_week'] = df.index.dayofweek
    df['abs_returns'] = df['returns'].abs()
    
    pivot_volatility = df.groupby(['day_of_week', 'hour'])['abs_returns'].mean().unstack(fill_value=0)
    
    fig.add_trace(
        go.Heatmap(
            z=pivot_volatility.values,
            x=pivot_volatility.columns,
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            colorscale='Viridis',
            name='Volatility',
            hovertemplate='Hour: %{x}:00<br>Day: %{y}<br>Volatility: %{z:.1f}bp<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Recent candlestick chart (last 5 days)
    recent_15min = data_15min.tail(5*26) if not data_15min.empty else df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().tail(5*26)
    
    if not recent_15min.empty:
        fig.add_trace(
            go.Candlestick(
                x=recent_15min.index,
                open=recent_15min['open'],
                high=recent_15min['high'],
                low=recent_15min['low'],
                close=recent_15min['close'],
                name='15-min Candles',
                increasing_line_color=FORD_COLORS['success'],
                decreasing_line_color=FORD_COLORS['danger']
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f"⏰ {symbol} - Intraday Trading Patterns Analysis",
        height=800,
        showlegend=False
    )
    
    # Update x-axis for hours
    fig.update_xaxes(title_text="Trading Hour (EST)", row=1, col=1)
    fig.update_xaxes(title_text="Trading Hour (EST)", row=1, col=2)
    fig.update_xaxes(title_text="Trading Hour (EST)", row=2, col=1)
    
    return fig

# Create intraday analysis
intraday_fig = analyze_intraday_patterns(minute_data, SYMBOL)
if intraday_fig:
    intraday_fig.show()


# %% [markdown]
# ## 📊 Step 4: Technical Analysis Dashboard

# %%
# Create technical analysis dashboard
def create_technical_dashboard(daily_data, symbol):
    """Comprehensive technical analysis with indicators"""
    
    df = daily_data.tail(100).copy()  # Last 100 days
    
    # Calculate technical indicators
    # Moving averages
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(20).mean()
    rolling_std = df['close'].rolling(20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Create subplot figure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f'{symbol} Price with Moving Averages & Bollinger Bands',
            'Volume',
            'RSI (Relative Strength Index)',
            'MACD'
        ],
        row_heights=[0.4, 0.2, 0.2, 0.2],
        vertical_spacing=0.05
    )
    
    # 1. Price chart with indicators
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['close'],
            name='Close Price',
            line=dict(color=FORD_COLORS['primary'], width=2)
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MA_20'],
            name='MA(20)',
            line=dict(color=FORD_COLORS['accent'], width=1)
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MA_50'],
            name='MA(50)',
            line=dict(color=FORD_COLORS['warning'], width=1)
        ), row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='rgba(128,128,128,0.5)', width=1),
            showlegend=False
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['BB_lower'],
            name='BB Lower',
            line=dict(color='rgba(128,128,128,0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ), row=1, col=1
    )
    
    # 2. Volume
    colors = [FORD_COLORS['success'] if df['close'].iloc[i] >= df['close'].iloc[i-1] 
              else FORD_COLORS['danger'] for i in range(1, len(df))]
    colors = ['gray'] + colors  # First bar
    
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1
    )
    
    # 3. RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI',
            line=dict(color=FORD_COLORS['secondary'], width=2)
        ), row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # 4. MACD
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MACD'],
            name='MACD',
            line=dict(color=FORD_COLORS['accent'], width=2)
        ), row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MACD_signal'],
            name='Signal',
            line=dict(color=FORD_COLORS['warning'], width=1)
        ), row=4, col=1
    )
    
    # MACD histogram
    histogram_colors = [FORD_COLORS['success'] if x >= 0 else FORD_COLORS['danger'] 
                       for x in df['MACD_histogram']]
    
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['MACD_histogram'],
            name='MACD Histogram',
            marker_color=histogram_colors,
            showlegend=False
        ), row=4, col=1
    )
    
    fig.update_layout(
        title=f"🔧 {symbol} - Technical Analysis Dashboard",
        height=1000,
        showlegend=True
    )
    
    return fig, df

# Create technical dashboard
tech_fig, tech_data = create_technical_dashboard(daily_data, SYMBOL)
tech_fig.show()

# Print current technical readings
print(f"\n🔧 Current Technical Indicators for {SYMBOL}:")
print(f"RSI: {tech_data['RSI'].iloc[-1]:.1f} ({'Overbought' if tech_data['RSI'].iloc[-1] > 70 else 'Oversold' if tech_data['RSI'].iloc[-1] < 30 else 'Neutral'})")
print(f"MACD: {tech_data['MACD'].iloc[-1]:.4f}")
print(f"Price vs MA(20): {((tech_data['close'].iloc[-1] / tech_data['MA_20'].iloc[-1] - 1) * 100):.1f}%")
print(f"Price vs MA(50): {((tech_data['close'].iloc[-1] / tech_data['MA_50'].iloc[-1] - 1) * 100):.1f}%")


# %% [markdown]
# ## 💡 Step 5: Market Summary & Insights

# %%
# Generate market insights and summary
def generate_market_insights(daily_data, minute_data, symbol):
    """Generate comprehensive market insights"""
    
    current_price = daily_data['close'].iloc[-1]
    prev_close = daily_data['close'].iloc[-2]
    daily_change = ((current_price / prev_close) - 1) * 100
    
    # Calculate various metrics
    week_high = daily_data['high'].tail(5).max()
    week_low = daily_data['low'].tail(5).min()
    month_high = daily_data['high'].tail(21).max()
    month_low = daily_data['low'].tail(21).min()
    year_high = daily_data['high'].tail(252).max()
    year_low = daily_data['low'].tail(252).min()
    
    avg_volume_30d = daily_data['volume'].tail(30).mean()
    recent_volume = daily_data['volume'].iloc[-1]
    volume_ratio = recent_volume / avg_volume_30d
    
    # Volatility metrics
    returns = daily_data['close'].pct_change().dropna() * 100
    volatility_30d = returns.tail(30).std()
    volatility_252d = returns.tail(252).std()
    
    # Create summary dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Price Levels', 'Volume Analysis', 'Volatility Metrics',
            'Performance vs Benchmarks', 'Risk Metrics', 'Trading Signals'
        ],
        specs=[
            [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "indicator"}, {"type": "table"}]
        ]
    )
    
    # 1. Current price indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta+gauge",
            value=current_price,
            delta={'reference': prev_close, 'position': "top"},
            title={"text": f"{symbol} Current Price"},
            number={'prefix': "$"},
            gauge={
                'axis': {'range': [year_low * 0.9, year_high * 1.1]},
                'bar': {'color': FORD_COLORS['primary']},
                'steps': [
                    {'range': [year_low * 0.9, year_low * 1.1], 'color': FORD_COLORS['danger']},
                    {'range': [year_high * 0.9, year_high * 1.1], 'color': FORD_COLORS['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': year_high
                }
            }
        ), row=1, col=1
    )
    
    # 2. Volume comparison
    volume_categories = ['Today', '5-day Avg', '30-day Avg']
    volume_values = [
        recent_volume,
        daily_data['volume'].tail(5).mean(),
        avg_volume_30d
    ]
    
    fig.add_trace(
        go.Bar(
            x=volume_categories,
            y=volume_values,
            name='Volume Comparison',
            marker_color=[FORD_COLORS['primary'], FORD_COLORS['accent'], FORD_COLORS['secondary']],
            text=[f'{v/1e6:.1f}M' for v in volume_values],
            textposition='auto'
        ), row=1, col=2
    )
    
    # 3. Volatility over time
    vol_30d = returns.tail(60).rolling(30).std()
    fig.add_trace(
        go.Scatter(
            x=vol_30d.index,
            y=vol_30d,
            name='30-day Volatility',
            line=dict(color=FORD_COLORS['warning'], width=2)
        ), row=1, col=3
    )
    
    # 4. Performance comparison
    periods = ['1D', '5D', '30D', '90D', '1Y']
    performance = [
        daily_change,
        ((current_price / daily_data['close'].iloc[-6]) - 1) * 100,
        ((current_price / daily_data['close'].iloc[-31]) - 1) * 100,
        ((current_price / daily_data['close'].iloc[-91]) - 1) * 100,
        ((current_price / daily_data['close'].iloc[-253]) - 1) * 100
    ]
    
    colors = [FORD_COLORS['success'] if p >= 0 else FORD_COLORS['danger'] for p in performance]
    
    fig.add_trace(
        go.Bar(
            x=periods,
            y=performance,
            name='Performance',
            marker_color=colors,
            text=[f'{p:+.1f}%' for p in performance],
            textposition='auto'
        ), row=2, col=1
    )
    
    # 5. Risk indicator
    risk_score = min(volatility_30d / volatility_252d * 50, 100)  # Normalized risk score
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ), row=2, col=2
    )
    
    # 6. Key metrics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color=FORD_COLORS['primary'],
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    ['Current Price', '52W High', '52W Low', 'Avg Volume', 'Volatility', 'RSI'],
                    [f'${current_price:.2f}', f'${year_high:.2f}', f'${year_low:.2f}', 
                     f'{avg_volume_30d/1e6:.1f}M', f'{volatility_30d:.1f}%', 
                     f'{tech_data["RSI"].iloc[-1]:.1f}' if not tech_data.empty else 'N/A']
                ],
                fill_color=['lightgray', 'white'],
                font=dict(color='black', size=11)
            )
        ), row=2, col=3
    )
    
    fig.update_layout(
        title=f"💡 {symbol} - Market Summary & Insights Dashboard",
        height=700,
        showlegend=False
    )
    
    return fig, {
        'current_price': current_price,
        'daily_change': daily_change,
        'volume_ratio': volume_ratio,
        'volatility_30d': volatility_30d,
        'risk_score': risk_score,
        'performance': dict(zip(periods, performance))
    }

# Generate insights
insights_fig, insights_data = generate_market_insights(daily_data, minute_data, SYMBOL)
insights_fig.show()

# Print summary
print(f"\n💡 Market Summary for {SYMBOL}:")
print(f"Current Price: ${insights_data['current_price']:.2f} ({insights_data['daily_change']:+.2f}%)")
print(f"Volume vs Average: {insights_data['volume_ratio']:.1f}x normal")
print(f"30-day Volatility: {insights_data['volatility_30d']:.1f}%")
print(f"Risk Score: {insights_data['risk_score']:.0f}/100")
print(f"\n📈 Performance:")
for period, perf in insights_data['performance'].items():
    print(f"  {period}: {perf:+.1f}%")

# %% [markdown]
# ## 🎯 Summary: Ford Visual Analysis Complete!
#
# **What we explored:**
# 1. ✅ **Price Overview**: Long-term trends, recent action, volume patterns
# 2. ✅ **Intraday Patterns**: Hour-by-hour trading behavior and volatility
# 3. ✅ **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
# 4. ✅ **Market Insights**: Performance metrics, risk analysis, key statistics
#
# **Key Benefits of Ford (F):**
# - **Affordable**: Great entry point for new investors
# - **Liquid**: High volume ensures easy trading
# - **Established**: Long history and stable company
# - **Dividend**: Regular dividend payments
#
# **Next Steps:**
# - Monitor technical indicators for entry/exit signals
# - Track volume patterns for unusual activity
# - Use intraday patterns for optimal timing
# - Consider this framework for other securities
#
# **🚀 This analysis framework can be applied to any security in your database!**
