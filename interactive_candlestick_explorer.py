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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📊 Interactive Candlestick Explorer - Algorithm Development Platform
#
# **Professional-grade interactive candlestick charts for algorithm development**
#
# - **Interactive scaling**: Zoom and pan on both time and price axes
# - **Professional styling**: Dark mode with trading platform colors
# - **Multiple timeframes**: Switch between 1min, 5min, 15min, 1hour, daily
# - **Algorithm development**: Point to specific candles to explain trend logic
# - **Real market data**: Use your collected data for algorithm validation
#
# **Perfect for developing supply/demand trading algorithms! 🚀**

# %%
# Setup and imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

# Professional trading platform theme
pio.templates.default = "plotly_dark"

# Import data systems
from direct_api_puller import get_intraday_data_api, get_daily_data_api
from market_data_database import MarketDataDatabase

# Professional trading colors
TRADING_COLORS = {
    'bullish': '#00ff88',      # Bright green for up candles
    'bearish': '#ff4444',      # Bright red for down candles
    'background': '#0e1118',   # Dark background
    'grid': '#2a2e3a',        # Subtle grid
    'text': '#ffffff',        # White text
    'volume_up': 'rgba(0, 255, 136, 0.6)',  # Transparent green
    'volume_down': 'rgba(255, 68, 68, 0.6)', # Transparent red
    'highlight': '#ffd700',    # Gold for highlights
    'support': '#4169e1',      # Blue for support
    'resistance': '#dc143c'    # Dark red for resistance
}

print("📊 Interactive Candlestick Explorer - Ready for Algorithm Development!")
print("=" * 70)


# %%
# Interactive data selection
class CandlestickExplorer:
    def __init__(self):
        self.symbol = 'AAPL'  # Default symbol
        self.timeframe = '15min'  # Default timeframe
        self.data = None
        self.fig = None
        
    def load_data(self, symbol, timeframe='15min', days_back=30):
        """Load data for the specified symbol and timeframe"""
        self.symbol = symbol
        self.timeframe = timeframe
        
        print(f"📡 Loading {symbol} {timeframe} data...")
        
        if timeframe == 'daily':
            # Get daily data
            self.data = get_daily_data_api(symbol)
            if not self.data.empty:
                self.data = self.data.tail(days_back * 2)  # More days for daily
        else:
            # Get intraday data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            self.data = get_intraday_data_api(symbol, timeframe, start_date, end_date)
            
            # If no data with date range, try recent data
            if self.data.empty:
                print("📅 No data for date range, getting recent data...")
                self.data = get_intraday_data_api(symbol, timeframe)
        
        if not self.data.empty:
            print(f"✅ Loaded {len(self.data):,} {timeframe} candles")
            print(f"📅 Range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Handle column name variations
            self.data.columns = [col.lower() for col in self.data.columns]
            
            return True
        else:
            print("❌ No data loaded")
            return False
    
    def create_professional_chart(self, show_volume=True, show_ma=True):
        """Create professional-grade interactive candlestick chart"""
        
        if self.data is None or self.data.empty:
            print("❌ No data available. Load data first.")
            return None
        
        # Determine number of rows based on what we're showing
        rows = 1
        row_heights = [1.0]
        
        if show_volume:
            rows = 2
            row_heights = [0.7, 0.3]
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=[f'{self.symbol} {self.timeframe.upper()} Candlesticks'] + (['Volume'] if show_volume else [])
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                name=f'{self.symbol}',
                increasing_line_color=TRADING_COLORS['bullish'],
                decreasing_line_color=TRADING_COLORS['bearish'],
                increasing_fillcolor=TRADING_COLORS['bullish'],
                decreasing_fillcolor=TRADING_COLORS['bearish']
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma and len(self.data) > 20:
            ma_20 = self.data['close'].rolling(20).mean()
            ma_50 = self.data['close'].rolling(50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=ma_20,
                    name='MA(20)',
                    line=dict(color='#ffd700', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            if len(self.data) > 50:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=ma_50,
                        name='MA(50)',
                        line=dict(color='#ff6b35', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add volume if requested
        if show_volume:
            # Color volume bars based on price movement
            volume_colors = []
            for i in range(len(self.data)):
                if self.data['close'].iloc[i] >= self.data['open'].iloc[i]:
                    volume_colors.append(TRADING_COLORS['volume_up'])
                else:
                    volume_colors.append(TRADING_COLORS['volume_down'])
            
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Professional styling
        fig.update_layout(
            title=f"📊 {self.symbol} - Interactive Candlestick Chart ({self.timeframe})",
            template="plotly_dark",
            height=800 if show_volume else 600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            # Professional margins
            margin=dict(l=60, r=60, t=60, b=60),
            # Grid styling
            plot_bgcolor=TRADING_COLORS['background'],
            paper_bgcolor=TRADING_COLORS['background']
        )
        
        # Update axes for professional look
        fig.update_xaxes(
            gridcolor=TRADING_COLORS['grid'],
            showgrid=True,
            zeroline=False,
            rangeslider_visible=False  # Remove range slider for cleaner look
        )
        
        fig.update_yaxes(
            gridcolor=TRADING_COLORS['grid'],
            showgrid=True,
            zeroline=False,
            side='right'  # Put price axis on right like professional platforms
        )
        
        # Configure interactive features
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date"
            ),
            # Enable crossfilter cursor
            hovermode='x unified'
        )
        
        self.fig = fig
        return fig
    
    def add_trend_annotations(self, annotations):
        """Add trend analysis annotations to the chart"""
        if self.fig is None:
            print("❌ No chart available. Create chart first.")
            return
        
        for annotation in annotations:
            if annotation['type'] == 'support':
                self.fig.add_hline(
                    y=annotation['price'],
                    line_dash="dash",
                    line_color=TRADING_COLORS['support'],
                    annotation_text=f"Support: ${annotation['price']:.2f}",
                    annotation_position="bottom right"
                )
            elif annotation['type'] == 'resistance':
                self.fig.add_hline(
                    y=annotation['price'],
                    line_dash="dash",
                    line_color=TRADING_COLORS['resistance'],
                    annotation_text=f"Resistance: ${annotation['price']:.2f}",
                    annotation_position="top right"
                )
            elif annotation['type'] == 'highlight':
                # Highlight specific candle
                self.fig.add_annotation(
                    x=annotation['timestamp'],
                    y=annotation['price'],
                    text=annotation['text'],
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=TRADING_COLORS['highlight'],
                    bgcolor=TRADING_COLORS['highlight'],
                    bordercolor=TRADING_COLORS['highlight']
                )

# Initialize the explorer
explorer = CandlestickExplorer()
print("✅ Candlestick Explorer initialized and ready!")

# %% [markdown]
# ## 📊 Step 2: Load Data and Create Interactive Chart

# %%
# Choose your symbol and timeframe for analysis
SYMBOL = 'AAPL'  # Change this to any symbol you want
TIMEFRAME = '15min'  # Options: '1min', '5min', '15min', '30min', '60min', 'daily'
DAYS_BACK = 30  # How many days of data to load

print(f"🎯 Loading {SYMBOL} {TIMEFRAME} data for algorithm development...")

# Load the data
success = explorer.load_data(SYMBOL, TIMEFRAME, DAYS_BACK)

if success:
    # Create the professional interactive chart
    print("\n📊 Creating professional interactive candlestick chart...")
    fig = explorer.create_professional_chart(show_volume=True, show_ma=True)
    
    if fig:
        # Show the chart
        fig.show()
        
        print("\n🎯 Interactive Chart Features:")
        print("   📏 Zoom: Click and drag to zoom into specific time ranges")
        print("   🔍 Pan: Shift+drag to pan left/right")
        print("   📊 Hover: Hover over candles for detailed OHLCV data")
        print("   🎚️  Reset: Double-click to reset zoom")
        print("   📈 Scale: Drag price axis to zoom vertically")
        
        print("\n🚀 Ready for Algorithm Development!")
        print("Now you can point to specific candles and explain your trend logic!")
    else:
        print("❌ Failed to create chart")
else:
    print("❌ Failed to load data")


# %% [markdown]
# ## 🎯 Step 3: Algorithm Development Workspace

# %%
# Quick data switching function
def switch_chart(symbol, timeframe='15min', days_back=30):
    """Quickly switch to different symbol/timeframe"""
    print(f"🔄 Switching to {symbol} {timeframe}...")
    
    success = explorer.load_data(symbol, timeframe, days_back)
    if success:
        fig = explorer.create_professional_chart(show_volume=True, show_ma=True)
        if fig:
            fig.show()
            return True
    return False

# Example: Switch to different charts for algorithm development
print("🔧 Quick Chart Switching Examples:")
print("switch_chart('TSLA', '5min', 14)   # Tesla 5-minute, 2 weeks")
print("switch_chart('SPY', '1min', 7)     # SPY 1-minute, 1 week")
print("switch_chart('F', 'daily', 100)    # Ford daily, 100 days")

# Algorithm development helper functions
def identify_supply_demand_zones(data, window=10):
    """Helper function to identify potential supply/demand zones"""
    zones = []
    
    # Simple supply/demand zone detection
    for i in range(window, len(data) - window):
        current_high = data['high'].iloc[i]
        current_low = data['low'].iloc[i]
        
        # Check if current candle is a local high (potential supply)
        if (current_high == data['high'].iloc[i-window:i+window+1].max()):
            zones.append({
                'type': 'supply',
                'timestamp': data.index[i],
                'price': current_high,
                'strength': data['volume'].iloc[i]
            })
        
        # Check if current candle is a local low (potential demand)
        if (current_low == data['low'].iloc[i-window:i+window+1].min()):
            zones.append({
                'type': 'demand',
                'timestamp': data.index[i],
                'price': current_low,
                'strength': data['volume'].iloc[i]
            })
    
    return zones

def add_supply_demand_analysis():
    """Add supply/demand zones to current chart"""
    if explorer.data is None:
        print("❌ No data loaded")
        return
    
    zones = identify_supply_demand_zones(explorer.data)
    
    annotations = []
    for zone in zones[-10:]:  # Show last 10 zones
        if zone['type'] == 'supply':
            annotations.append({
                'type': 'resistance',
                'price': zone['price']
            })
        else:
            annotations.append({
                'type': 'support',
                'price': zone['price']
            })
    
    explorer.add_trend_annotations(annotations)
    explorer.fig.show()
    
    print(f"✅ Added {len(annotations)} supply/demand zones to chart")

print("\n🎯 Algorithm Development Tools Ready:")
print("   📊 switch_chart(symbol, timeframe, days) - Quick chart switching")
print("   🔍 add_supply_demand_analysis() - Add S/D zones to current chart")
print("   📈 explorer.data - Access raw OHLCV data for analysis")
print("\n💡 Now you can point to specific candles and explain your algorithm logic!")

# %% [markdown]
# ## 💡 Step 4: Algorithm Development - Point to Candles!
#
# **Now you can:**
#
# 1. **🔍 Examine specific candles** - Hover and zoom to see exact OHLCV data
# 2. **📏 Identify patterns** - Point to sequences of candles that show supply/demand
# 3. **✏️ Explain logic** - Describe what the algorithm should detect at specific points
# 4. **🎯 Test ideas** - Switch symbols/timeframes to validate concepts
# 5. **📊 Add annotations** - Mark support/resistance and key levels
#
# **Example Algorithm Development Process:**
# - Point to a specific candle: "This candle at 2:15 PM shows strong rejection at $150.25"
# - Explain the pattern: "The long upper wick with high volume indicates supply zone"
# - Define the rule: "Algorithm should flag when price approaches this level again"
# - Test on other timeframes: "Let's see if this pattern holds on 5-minute charts"

# %%
# Ready for your algorithm explanations!
print("🚀 READY FOR ALGORITHM DEVELOPMENT!")
print("=" * 50)
print("Point to specific candles in the chart above and explain:")
print("")
print("📍 What patterns you see")
print("📍 How supply/demand is manifesting")
print("📍 What the algorithm should detect")
print("📍 Entry/exit criteria")
print("📍 Risk management rules")
print("")
print("I'm ready to help implement whatever logic you describe!")

# Show current data summary
if explorer.data is not None and not explorer.data.empty:
    print(f"\n📊 Current Dataset: {explorer.symbol} {explorer.timeframe}")
    print(f"📅 Time Range: {explorer.data.index.min()} to {explorer.data.index.max()}")
    print(f"📈 Price Range: ${explorer.data['low'].min():.2f} - ${explorer.data['high'].max():.2f}")
    print(f"📊 Total Candles: {len(explorer.data):,}")
    print(f"💰 Latest Price: ${explorer.data['close'].iloc[-1]:.2f}")
