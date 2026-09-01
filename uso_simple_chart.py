#!/usr/bin/env python3
"""
Simple USO chart - just the last 100 candles
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from market_data_database import MarketDataDatabase

# Dark theme
pio.templates.default = "plotly_dark"

def create_simple_uso_chart():
    """Create simple chart with last 100 USO candles"""
    
    print("📊 Creating simple USO chart with last 100 candles...")
    
    # Get USO data
    db = MarketDataDatabase()
    uso_data = db._get_intraday_data_direct('USO', interval='1min')
    
    print(f"✅ Loaded {len(uso_data):,} total USO records")
    
    # Filter to regular trading hours only (9:30 AM - 4:00 PM ET)
    # Convert to ET timezone for proper filtering
    uso_data_et = uso_data.copy()
    uso_data_et.index = uso_data_et.index.tz_localize('UTC').tz_convert('US/Eastern')
    
    # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
    regular_hours = uso_data_et.between_time('09:30', '16:00')
    
    print(f"📊 Total records: {len(uso_data):,}")
    print(f"📈 Regular hours records: {len(regular_hours):,}")
    
    # Get last 100 candles from regular trading hours
    last_100 = regular_hours.tail(100)
    print(f"📅 Last 100 regular trading candles: {last_100.index.min()} to {last_100.index.max()}")
    
    # Create subplot with candlestick and volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=last_100.index,
            open=last_100['Open'],
            high=last_100['High'],
            low=last_100['Low'],
            close=last_100['Close'],
            name='USO'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=last_100.index,
            y=last_100['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.6)',
            marker_line_color='rgba(8,48,107,1.0)',
            marker_line_width=1
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="📊 USO - Last 100 Regular Trading Hours Candles with Volume (1-minute)",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # Save as HTML file
    fig.write_html("uso_last_100_candles.html")
    print("💾 Chart saved as: uso_last_100_candles.html")
    
    # Print summary
    print(f"\n📈 Price Summary:")
    print(f"Latest: ${last_100['Close'].iloc[-1]:.2f}")
    print(f"High: ${last_100['High'].max():.2f}")
    print(f"Low: ${last_100['Low'].min():.2f}")
    print(f"Range: ${last_100['High'].max() - last_100['Low'].min():.2f}")
    
    print(f"\n📊 Volume Summary (Last 100 Regular Trading Candles):")
    print(f"Total Volume: {last_100['Volume'].sum():,}")
    print(f"Average Volume: {last_100['Volume'].mean():.0f}")
    print(f"Max Volume: {last_100['Volume'].max():,}")
    print(f"Min Volume: {last_100['Volume'].min():,}")

if __name__ == "__main__":
    create_simple_uso_chart()