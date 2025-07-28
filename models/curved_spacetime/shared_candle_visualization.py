"""
Shared Candle Visualization Logic

This module contains the EXACT working visualization logic extracted from 
the two_candle_test.py that successfully creates curvature-filled candlesticks.

This ensures both the two-candle test and main example use identical code.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curvature_color_scheme import get_curvature_gradient_color


def add_single_curvature_candle(fig: go.Figure, 
                               idx: datetime, 
                               ohlc_row: pd.Series, 
                               curvature_val: float,
                               candle_index: int,
                               half_width_minutes: int = 20,
                               debug: bool = False,
                               proper_time_x: float = None) -> None:
    """
    Add a single curvature-filled candlestick to the figure.
    
    This is the EXACT working logic from two_candle_test.py.
    
    Args:
        fig: Plotly figure to add candle to
        idx: Datetime index for this candle
        ohlc_row: Pandas series with 'open', 'high', 'low', 'close' values
        curvature_val: Curvature value for this candle
        candle_index: Index number for naming traces
        half_width_minutes: Half-width of candle in minutes
        debug: Whether to print debug messages
        proper_time_x: If provided, use proper time coordinate instead of datetime for x-axis
    """
    O, H, L, C = ohlc_row['open'], ohlc_row['high'], ohlc_row['low'], ohlc_row['close']
    
    if debug:
        print(f"\\nProcessing candle {candle_index}:")
        print(f"  Date: {idx}")
        print(f"  OHLC: O={O}, H={H}, L={L}, C={C}")
    
    # Colors - EXACT from working test
    is_bullish = C >= O
    outline_color = '#00ff00' if is_bullish else '#ff0000'
    
    # Use gradient color scheme
    fill_color, fill_type = get_curvature_gradient_color(curvature_val)
    
    if debug:
        print(f"  Direction: {'Bullish' if is_bullish else 'Bearish'}")
        print(f"  Outline color: {outline_color}")
        print(f"  Curvature: {curvature_val:.4f}")
        print(f"  Fill color: {fill_color} ({fill_type})")
    
    # Body dimensions - EXACT from working test
    body_top = max(O, C)
    body_bottom = min(O, C)
    body_height = body_top - body_bottom
    
    if debug:
        print(f"  Body: {body_bottom} to {body_top} (height: {body_height})")
    
    # Width calculation - use proper time if provided, otherwise use datetime
    if proper_time_x is not None:
        # Use proper time coordinates with adaptive width
        proper_time_half_width = 2.0  # Larger width in proper time units
        left_x = proper_time_x - proper_time_half_width
        right_x = proper_time_x + proper_time_half_width
        center_x = proper_time_x
    else:
        # Use datetime coordinates with timedelta
        candle_half_width = timedelta(minutes=half_width_minutes)
        left_x = idx - candle_half_width
        right_x = idx + candle_half_width
        center_x = idx
    
    if debug:
        print(f"  Width: {left_x} to {right_x}")
    
    # STEP 1: Add curvature fill FIRST - EXACT from working test
    if body_height > 0:
        if debug:
            print("  Adding curvature fill...")
        fig.add_trace(go.Scatter(
            x=[left_x, right_x, right_x, left_x, left_x],  # Rectangle coordinates
            y=[body_bottom, body_bottom, body_top, body_top, body_bottom],  # Close the shape
            fill='toself',
            fillcolor=fill_color,
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),  # Invisible outline for fill
            showlegend=False,
            name=f'Fill {candle_index}',
            hovertemplate=f'Candle {candle_index}<br>OHLC: {O}, {H}, {L}, {C}<br>Curvature: {curvature_val:.4f}<extra></extra>'
        ))
        if debug:
            print("    ✓ Added curvature fill")
    else:
        if debug:
            print("  Adding doji line...")
        fig.add_trace(go.Scatter(
            x=[left_x, right_x],
            y=[O, C],
            mode='lines',
            line=dict(color=fill_color, width=6),
            showlegend=False,
            name=f'Doji {candle_index}'
        ))
        if debug:
            print("    ✓ Added doji line")
    
    # STEP 2: Add wick - EXACT from working test
    if debug:
        print("  Adding wick...")
    fig.add_trace(go.Scatter(
        x=[center_x, center_x],
        y=[L, H],
        mode='lines',
        line=dict(color=outline_color, width=2),
        showlegend=False,
        name=f'Wick {candle_index}'
    ))
    if debug:
        print("    ✓ Added wick")
    
    # STEP 3: Add body outline - EXACT from working test (4 separate lines)
    if body_height > 0:
        if debug:
            print("  Adding body outline (4 edges)...")
        
        # Bottom edge
        fig.add_trace(go.Scatter(
            x=[left_x, right_x],
            y=[body_bottom, body_bottom],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name=f'Bottom {candle_index}'
        ))
        
        # Top edge
        fig.add_trace(go.Scatter(
            x=[left_x, right_x],
            y=[body_top, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name=f'Top {candle_index}'
        ))
        
        # Left edge
        fig.add_trace(go.Scatter(
            x=[left_x, left_x],
            y=[body_bottom, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name=f'Left {candle_index}'
        ))
        
        # Right edge
        fig.add_trace(go.Scatter(
            x=[right_x, right_x],
            y=[body_bottom, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name=f'Right {candle_index}'
        ))
        if debug:
            print("    ✓ Added 4 outline edges")
    
    if debug:
        print(f"  ✅ Completed candle {candle_index}")


def create_curvature_candlestick_chart_shared(ohlc_data: pd.DataFrame,
                                            curvatures: np.ndarray,
                                            title: str = "Curvature-Enhanced Candlestick Chart",
                                            debug: bool = False,
                                            use_proper_time: bool = False,
                                            proper_times: np.ndarray = None) -> go.Figure:
    """
    Create curvature-filled candlestick chart using the EXACT working logic.
    
    Args:
        ohlc_data: DataFrame with datetime index and OHLC columns
        curvatures: Array of curvature values for each candle
        title: Chart title
        debug: Whether to print debug messages
        use_proper_time: If True, use proper time coordinates for x-axis
        proper_times: Array of proper time values (required if use_proper_time=True)
    
    Returns:
        Plotly figure with curvature-filled candlesticks
    """
    
    if debug:
        print("🔧 CREATING CURVATURE CANDLESTICKS WITH SHARED LOGIC")
        print("=" * 60)
        if use_proper_time:
            print("    Using PROPER TIME coordinates!")
    
    # Validate proper time arguments
    if use_proper_time and proper_times is None:
        raise ValueError("proper_times array required when use_proper_time=True")
    
    if use_proper_time and len(proper_times) != len(ohlc_data):
        raise ValueError("proper_times array must have same length as ohlc_data")
    
    # Normalize curvatures
    curvature_normalized = np.clip(curvatures, -1, 1)
    
    # Create figure
    fig = go.Figure()
    
    if debug:
        print(f"Processing {len(ohlc_data)} candles...")
    
    # Process each candle using EXACT working logic
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        proper_time_x = proper_times[i] if use_proper_time else None
        
        add_single_curvature_candle(
            fig=fig,
            idx=idx,
            ohlc_row=row,
            curvature_val=curvature_normalized[i],
            candle_index=i,
            debug=debug and i < 3,  # Only debug first 3 candles
            proper_time_x=proper_time_x
        )
    
    # Layout - EXACT from working test style
    x_axis_title = "Proper Time (τ)" if use_proper_time else "Time"
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title="Price",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=600,
        width=1200
    )
    
    if debug:
        print(f"\\n✅ Created chart with {len(ohlc_data)} candles")
    
    return fig