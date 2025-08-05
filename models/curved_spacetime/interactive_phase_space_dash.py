#!/usr/bin/env python3
"""
Interactive Phase Space Analysis with Real-Time KDE Recomputation
Uses Plotly Dash for server-side callbacks to recompute gradients when range changes.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curved_candle_geometry import CurvedCandleGeometry, CandleMetric, create_candle_metrics_from_ohlc

def generate_rich_market_data(n_candles: int = 200) -> pd.DataFrame:
    """Generate rich market data with multiple regimes for meaningful gradients."""
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(n_candles)]
    
    np.random.seed(42)  # Reproducible
    price = 100.0
    ohlc_data = []
    volumes = []
    
    for i in range(n_candles):
        # Create distinct market regimes with clustering
        if i < n_candles * 0.2:  # Bullish trending (positive sentiment cluster)
            sentiment_target = 0.4
            uwr_target = 0.3
            volatility = 1.2
            trend = 0.3
        elif i < n_candles * 0.4:  # Bearish correction (negative sentiment cluster)  
            sentiment_target = -0.3
            uwr_target = 0.6
            volatility = 2.0
            trend = -0.2
        elif i < n_candles * 0.6:  # High volatility consolidation (spread across space)
            sentiment_target = 0.0
            uwr_target = 0.5
            volatility = 2.5
            trend = 0.0
        elif i < n_candles * 0.8:  # Recovery phase (moderate positive sentiment)
            sentiment_target = 0.2
            uwr_target = 0.4
            volatility = 1.5
            trend = 0.15
        else:  # Final consolidation (neutral cluster)
            sentiment_target = 0.1
            uwr_target = 0.35
            volatility = 1.0
            trend = 0.05
        
        # Generate OHLC with trend and clustering
        price += trend + np.random.normal(0, volatility * 0.5)
        
        # Generate candle with target clustering
        base_range = volatility * np.random.uniform(0.5, 2.0)
        
        # Bias towards target sentiment and UWR
        sentiment_bias = np.random.normal(sentiment_target, 0.2)
        uwr_bias = np.random.normal(uwr_target, 0.15)
        
        # Generate OHLC to achieve desired sentiment and UWR
        high = price + base_range * (0.5 + abs(sentiment_bias) * 0.3)
        low = high - base_range
        
        if sentiment_bias > 0:  # Bullish
            open_price = low + base_range * np.random.uniform(0.1, 0.4)
            close = low + base_range * np.random.uniform(0.6, 0.9)
        else:  # Bearish
            open_price = low + base_range * np.random.uniform(0.6, 0.9)
            close = low + base_range * np.random.uniform(0.1, 0.4)
        
        # Adjust high based on UWR target
        wick_adjustment = base_range * abs(uwr_bias) * 0.3
        high = max(open_price, close) + wick_adjustment
        
        ohlc_data.append([open_price, high, low, close])
        
        # Volume varies with volatility
        volume_base = 2000 + volatility * 1000
        volume = volume_base * np.random.uniform(0.5, 2.0)
        volumes.append(int(volume))
    
    # Create DataFrame
    price_array = np.array(ohlc_data)
    return pd.DataFrame({
        'open': price_array[:, 0],
        'high': price_array[:, 1],
        'low': price_array[:, 2],
        'close': price_array[:, 3],
        'volume': volumes
    }, index=dates)


def compute_phase_space_gradients(ohlc_data: pd.DataFrame, start_idx: int = 0, end_idx: int = None):
    """Compute phase space gradients for a given range of data."""
    if end_idx is None:
        end_idx = len(ohlc_data) - 1
    
    # Slice the data
    data_slice = ohlc_data.iloc[start_idx:end_idx+1]
    
    if len(data_slice) < 5:  # Need minimum points for meaningful KDE
        return None, None, None, None, None, None
    
    # Create candle metrics for the slice
    candle_metrics = create_candle_metrics_from_ohlc(data_slice)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    sentiments = np.array([c.sentiment for c in candle_metrics])
    uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
    proper_times = geometry.compute_proper_time_series()
    
    # Generate grid for contours
    grid_res = 40
    sentiment_range = np.linspace(-0.99, 0.99, grid_res)
    uwr_range = np.linspace(0.01, 0.99, grid_res)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # 1. Density contours using KDE
    points = np.column_stack([sentiments, uwrs])
    if len(points) >= 2:  # Need at least 2 points for KDE
        kde = gaussian_kde(points.T, bw_method='scott')
        
        density_grid = np.zeros_like(S_grid)
        for i in range(grid_res):
            for j in range(grid_res):
                s, u = S_grid[i, j], U_grid[i, j]
                if abs(s) + u <= 1.0:  # Phase space constraint
                    density_grid[i, j] = kde([s, u])[0]
                else:
                    density_grid[i, j] = np.nan
    else:
        density_grid = np.full_like(S_grid, np.nan)
    
    # 2. Proper time gradients
    if len(points) >= 2:
        time_grid = griddata(
            points, proper_times, (S_grid, U_grid), 
            method='linear', fill_value=np.nan
        )
        
        # Apply constraint
        for i in range(grid_res):
            for j in range(grid_res):
                if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                    time_grid[i, j] = np.nan
    else:
        time_grid = np.full_like(S_grid, np.nan)
    
    # 3. Coordinate time gradients
    coord_times = np.arange(start_idx, start_idx + len(candle_metrics))
    if len(points) >= 2:
        coord_time_grid = griddata(
            points, coord_times, (S_grid, U_grid),
            method='linear', fill_value=np.nan
        )
        
        # Apply constraint
        for i in range(grid_res):
            for j in range(grid_res):
                if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                    coord_time_grid[i, j] = np.nan
    else:
        coord_time_grid = np.full_like(S_grid, np.nan)
    
    return (sentiment_range, uwr_range, density_grid, 
            time_grid, coord_time_grid, sentiments, uwrs, proper_times)


# Initialize the Dash app
app = dash.Dash(__name__)

# Generate the market data
print("📊 Generating market data...")
ohlc_data = generate_rich_market_data(n_candles=200)
print(f"✅ Generated {len(ohlc_data)} candles")

# Create the layout
app.layout = html.Div([
    html.H1("🌌 Interactive Phase Space Analysis with Real-Time KDE", 
            style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#0d1117'}),
    
    html.Div([
        html.P("Select a range on the candlestick chart to see gradients recompute in real-time!", 
               style={'textAlign': 'center', 'color': 'yellow', 'fontSize': '14px'})
    ], style={'backgroundColor': '#0d1117', 'padding': '10px'}),
    
    dcc.Graph(
        id='combined-analysis',
        style={'height': '800px'},
        config={'displayModeBar': True}
    ),
    
    # Store the range selection
    dcc.Store(id='range-store')
], style={'backgroundColor': '#0d1117'})


@callback(
    [Output('combined-analysis', 'figure'),
     Output('range-store', 'data')],
    [Input('combined-analysis', 'relayoutData')]
)
def update_gradients(relayoutData):
    """Update gradients based on range selection."""
    
    # Default to full range
    start_idx = 0
    end_idx = len(ohlc_data) - 1
    range_info = f"Full Dataset (0-{end_idx})"
    
    # Debug: Print all relayout data
    if relayoutData:
        print(f"📊 RelayoutData received: {relayoutData}")
    
    # Check if there's a range selection on the candlestick chart
    if relayoutData:
        # Check multiple possible axis references for the candlestick chart
        range_keys = [
            ('xaxis3.range[0]', 'xaxis3.range[1]'),  # Standard subplot reference
            ('xaxis.range[0]', 'xaxis.range[1]'),    # Sometimes it's xaxis
            ('xaxis2.range[0]', 'xaxis2.range[1]'),  # Or xaxis2
        ]
        
        range_found = False
        for start_key, end_key in range_keys:
            if start_key in relayoutData and end_key in relayoutData:
                start_idx = max(0, int(relayoutData[start_key]))
                end_idx = min(len(ohlc_data) - 1, int(relayoutData[end_key]))
                range_info = f"Range ({start_idx}-{end_idx})"
                print(f"🔄 Recomputing gradients for range {start_idx} to {end_idx} (using {start_key})")
                range_found = True
                break
        
        # Check for autorange reset
        autorange_keys = ['xaxis3.autorange', 'xaxis.autorange', 'xaxis2.autorange']
        for key in autorange_keys:
            if key in relayoutData and relayoutData[key]:
                start_idx = 0
                end_idx = len(ohlc_data) - 1
                range_info = f"Full Dataset (0-{end_idx})"
                print("🔄 Reset to full range")
                range_found = True
                break
        
        if not range_found:
            print(f"⚠️ No range change detected in relayoutData: {relayoutData}")
    
    # Compute gradients for the selected range
    gradient_data = compute_phase_space_gradients(ohlc_data, start_idx, end_idx)
    
    if gradient_data[0] is None:
        # Not enough data, return empty figure
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title="❌ Not enough data points for gradient computation",
            template='plotly_dark'
        )
        return fig, {'start': start_idx, 'end': end_idx}
    
    (sentiment_range, uwr_range, density_grid, 
     time_grid, coord_time_grid, sentiments, uwrs, proper_times) = gradient_data
    
    # Create the combined figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Density Gradient',
            'Proper Time Gradient', 
            'Traditional Candlesticks',
            'Coordinate Time Gradient',
            'Phase Space Trajectory',
            ''  # Empty bottom-right
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, None]  # No bottom-right subplot
        ]
    )
    
    # 1. Density gradient contour
    density_max = np.nanmax(density_grid) if not np.isnan(density_grid).all() else 1
    density_min = np.nanmin(density_grid) if not np.isnan(density_grid).all() else 0
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=density_grid,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(
                title="Density",
                x=0.32,
                len=0.45,
                y=0.775,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=density_min,
                end=density_max,
                size=max((density_max - density_min)/10, 0.001),
                showlines=True,
                coloring='fill'
            ),
            hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Density: %{z:.6f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Proper time gradient contour
    time_max = np.nanmax(time_grid) if not np.isnan(time_grid).all() else 1
    time_min = np.nanmin(time_grid) if not np.isnan(time_grid).all() else 0
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=time_grid,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Proper<br>Time",
                x=0.66,
                len=0.45,
                y=0.775,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=time_min,
                end=time_max,
                size=max((time_max - time_min)/10, 0.1),
                showlines=True,
                coloring='fill'
            ),
            hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Proper Time: %{z:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=list(range(len(ohlc_data))),
            open=ohlc_data['open'].tolist(),
            high=ohlc_data['high'].tolist(),
            low=ohlc_data['low'].tolist(),
            close=ohlc_data['close'].tolist(),
            name='OHLC',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # 4. Coordinate time gradient contour
    coord_time_max = np.nanmax(coord_time_grid) if not np.isnan(coord_time_grid).all() else end_idx
    coord_time_min = np.nanmin(coord_time_grid) if not np.isnan(coord_time_grid).all() else start_idx
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=coord_time_grid,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title="Coordinate<br>Time",
                x=0.32,
                len=0.45,
                y=0.275,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=coord_time_min,
                end=coord_time_max,
                size=max((coord_time_max - coord_time_min)/10, 1),
                showlines=True,
                coloring='fill'
            ),
            hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Coord Time: %{z:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Debug output
    print(f"📊 Generated gradients for {len(sentiments)} points:")
    print(f"   Density range: {density_min:.6f} to {density_max:.6f}")
    print(f"   Proper time range: {time_min:.3f} to {time_max:.3f}")
    print(f"   Coord time range: {coord_time_min:.0f} to {coord_time_max:.0f}")
    
    # 5. Phase space trajectory
    fig.add_trace(
        go.Scatter(
            x=sentiments,
            y=uwrs,
            mode='lines+markers',
            line=dict(color='white', width=3),
            marker=dict(
                size=6,
                color=proper_times,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title="Proper<br>Time",
                    x=0.66,
                    len=0.45,
                    y=0.275,
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                )
            ),
            name='Phase Space Trajectory',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add phase space boundaries
    boundary_s = np.linspace(-0.99, 0.99, 50)
    boundary_u = 1.0 - np.abs(boundary_s)
    
    phase_space_plots = [(1,1), (1,2), (2,1), (2,2)]
    for row, col in phase_space_plots:
        fig.add_trace(
            go.Scatter(
                x=boundary_s,
                y=boundary_u,
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=f'🌌 Real-Time Interactive Analysis: {range_info} → {len(sentiments)} points',
        template='plotly_dark',
        height=800,
        width=1800,
        showlegend=False,
        # Add range selector to candlestick chart
        xaxis3=dict(
            title='Time Index',
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                buttons=list([
                    dict(count=20, label="20", step="all", stepmode="backward"),
                    dict(count=40, label="40", step="all", stepmode="backward"),
                    dict(count=60, label="60", step="all", stepmode="backward"),
                    dict(count=100, label="100", step="all", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                borderwidth=1,
                font=dict(color='white')
            )
        )
    )
    
    # Update axes
    phase_space_plots = [(1,1), (1,2), (2,1), (2,2)]
    for row, col in phase_space_plots:
        fig.update_xaxes(title='Sentiment', range=[-1, 1], row=row, col=col)
        fig.update_yaxes(title='UWR', range=[0, 1], row=row, col=col)
    
    fig.update_yaxes(title='Price', row=1, col=3)
    
    return fig, {'start': start_idx, 'end': end_idx}


if __name__ == '__main__':
    print("🚀 Starting Interactive Phase Space Analysis Server...")
    print("📊 Open http://127.0.0.1:8050 in your browser")
    print("🎯 Select ranges on the candlestick chart to see gradients recompute!")
    app.run(debug=True, host='127.0.0.1', port=8050)