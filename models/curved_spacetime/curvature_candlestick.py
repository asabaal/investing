"""
Enhanced candlestick charts with curvature-based filling and directional outlines.
Uses shared visualization logic to ensure consistency.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from curved_candle_geometry import CurvedCandleGeometry, CandleMetric
from curvature_color_scheme import get_curvature_gradient_color, get_discrete_curvature_levels
from shared_candle_visualization import create_curvature_candlestick_chart_shared


def create_curvature_candlestick_chart(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    title: str = "Curvature-Enhanced Candlestick Chart"
) -> go.Figure:
    """
    Create curvature-filled candlestick chart using shared visualization logic.
    This ensures EXACT same code as the working two-candle test.
    """
    
    print("🔧 CREATING CURVATURE CANDLESTICKS WITH SHARED LOGIC")
    print("=" * 60)
    
    # Get curvature values
    curvatures = geometry.compute_curvature_series()
    
    # Use the EXACT same shared logic as two-candle test
    fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=curvatures,
        title=title,
        debug=True  # Show debug info for first few candles
    )
    
    return fig


def create_curvature_legend() -> go.Figure:
    """
    Create a separate legend showing the curvature color scheme.
    """
    levels = get_discrete_curvature_levels()
    
    fig = go.Figure()
    
    y_positions = list(range(len(levels)))
    curvature_values = list(levels.keys())
    colors = [levels[k][0] for k in curvature_values]
    labels = [levels[k][1] for k in curvature_values]
    
    # Create color bar
    for i, (curvature, y_pos) in enumerate(zip(curvature_values, y_positions)):
        color, label = levels[curvature]
        
        # Color swatch
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[y_pos-0.4, y_pos-0.4, y_pos+0.4, y_pos+0.4, y_pos-0.4],
            fill='toself',
            fillcolor=color,
            mode='lines',
            line=dict(width=1, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Label
        fig.add_annotation(
            x=1.5,
            y=y_pos,
            text=f"{curvature:+.2f}: {label.split(':')[1].strip()}",
            showarrow=False,
            font=dict(color='white', size=12),
            xanchor='left'
        )
    
    fig.update_layout(
        title="Market Curvature Color Legend",
        xaxis=dict(visible=False, range=[-0.5, 4]),
        yaxis=dict(visible=False, range=[-1, len(levels)]),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=400,
        width=600,
        showlegend=False
    )
    
    return fig


def create_combined_curvature_chart(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    title: str = "Curvature-Enhanced Candlestick Chart with Legend"
) -> go.Figure:
    """
    Create a combined chart with candlesticks and integrated legend.
    """
    # Create main candlestick chart
    main_fig = create_curvature_candlestick_chart(ohlc_data, geometry, title)
    
    # Get curvature data for legend
    curvatures = geometry.compute_curvature_series()
    levels = get_discrete_curvature_levels()
    
    # Add legend as annotations
    legend_x = 0.02  # Left side of chart
    legend_y_start = 0.98
    legend_y_step = 0.12
    
    for i, (curvature, (color, label)) in enumerate(levels.items()):
        y_pos = legend_y_start - i * legend_y_step
        
        # Extract RGB values from rgba string
        rgba_str = color.replace('rgba(', '').replace(')', '')
        r, g, b, a = map(float, rgba_str.split(', '))
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        
        main_fig.add_annotation(
            x=legend_x,
            y=y_pos,
            text=f"● {curvature:+.2f}: {label.split(':')[1].strip()}",
            showarrow=False,
            font=dict(color=hex_color, size=10, family="monospace"),
            xref="paper",
            yref="paper",
            xanchor='left',
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='white',
            borderwidth=1
        )
    
    return main_fig