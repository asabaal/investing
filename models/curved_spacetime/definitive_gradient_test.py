"""
Definitive gradient test - create candles that MUST show different colors if working.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shared_candle_visualization import create_curvature_candlestick_chart_shared

def create_real_gradient_test():
    """Create a test with REAL curvature calculations from designed OHLC data."""
    
    print("🎨 DEFINITIVE GRADIENT TEST - REAL CURVATURE VALUES")
    print("=" * 60)
    
    from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
    
    # Create 10 candles with OHLC data designed to produce diverse curvatures
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(10)]
    
    # Design OHLC data to get a good range of curvatures:
    # - Large range + balanced → positive curvature (trending)
    # - Small range + extreme patterns → negative curvature (volatile)
    # - Mix of different combinations
    
    ohlc_data = pd.DataFrame({
        # Candle 0: Large range, balanced (should be highly trending)
        # Candle 1: Medium range, slightly bullish (should be trending)
        # Candle 2: Medium range, neutral (should be neutral-trending)
        # Candle 3: Small range, extreme bearish (should be volatile)
        # Candle 4: Very small range, extreme bullish (should be highly volatile)
        # Candle 5: Medium range, balanced (should be trending)
        # Candle 6: Large range, extreme (should be mixed)
        # Candle 7: Small range, balanced (should be neutral)
        # Candle 8: Large range, very balanced (should be highly trending)
        # Candle 9: Tiny range, very extreme (should be highly volatile)
        
        'open':  [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
        'high':  [120.0, 108.0, 107.0, 106.5, 108.1, 115.0, 125.0, 115.0, 130.0, 118.05],
        'low':   [80.0,  98.0,  101.0, 105.8, 107.95, 105.0, 100.0, 113.5, 102.0, 117.98],
        'close': [110.0, 105.0, 104.5, 105.9, 107.96, 112.0, 105.0, 114.2, 125.0, 117.99],
        
        # Add volume data (mass) - design for interesting gravitational effects
        'volume': [50000, 10000, 25000, 5000, 2000, 30000, 80000, 8000, 60000, 3000]
        # High-volume candles (0, 6, 8) should create strong gravity wells
        # Low-volume candles (4, 9, 3) should have weak gravitational effects
    }, index=dates)
    
    print("OHLCV data designed for curvature and mass diversity:")
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        range_val = row['high'] - row['low']
        sentiment = (row['close'] - row['open']) / range_val if range_val > 0 else 0
        uwr = (row['high'] - max(row['open'], row['close'])) / range_val if range_val > 0 else 0
        volume = row['volume']
        print(f"  Candle {i}: Range={range_val:.1f}, Volume={volume:,}, Sentiment={sentiment:.3f}, UWR={uwr:.3f}")
    
    # Compute REAL curvatures, proper times, and proper time intervals using our new intrinsic calculation
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    real_curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    proper_time_intervals = geometry.compute_proper_time_intervals()
    
    print("\\nReal computed curvature values, proper times, intervals, and colors:")
    from curvature_color_scheme import get_curvature_gradient_color
    for i, (curvature, proper_time, interval) in enumerate(zip(real_curvatures, proper_times, proper_time_intervals)):
        color, description = get_curvature_gradient_color(curvature)
        print(f"  Candle {i}: τ={proper_time:.2f}, dτ={interval:.2f}, curvature={curvature:+.3f} -> {color} ({description})")
    
    # Create subplot figure with candlesticks on top, curvature and proper time intervals below
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],  # Candlesticks get 60%, curvature and intervals get 20% each
        subplot_titles=("Curvature-Enhanced Candlesticks", "Market Curvature Over Proper Time", "Proper Time Intervals (dτ)"),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Get the candlestick chart data using our working function with PROPER TIME
    candle_fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=real_curvatures,
        title="",  # No title since we use subplot titles
        debug=False,
        use_proper_time=True,
        proper_times=proper_times
    )
    
    # Add all candlestick traces to the top subplot
    for trace in candle_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add curvature line chart to bottom subplot
    from curvature_color_scheme import get_curvature_gradient_color
    
    # Create curvature line with color-coded points using PROPER TIME
    fig.add_trace(
        go.Scatter(
            x=proper_times,
            y=real_curvatures,
            mode='lines+markers',
            line=dict(color='white', width=2),
            marker=dict(
                size=8,
                color=[get_curvature_gradient_color(c)[0].replace('0.8)', '1.0)') for c in real_curvatures],
                line=dict(color='white', width=1)
            ),
            name='Curvature',
            hovertemplate='Proper Time: %{x:.2f}<br>Curvature: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add proper time intervals chart to third subplot
    fig.add_trace(
        go.Scatter(
            x=proper_times,
            y=proper_time_intervals,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(
                size=6,
                color='#ff6b6b',
                line=dict(color='white', width=1)
            ),
            name='Time Intervals',
            hovertemplate='Proper Time: %{x:.2f}<br>dτ: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add horizontal reference lines for curvature interpretation
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="purple", opacity=0.3, row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", opacity=0.3, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,  # Taller to accommodate three subplots
        showlegend=False,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        title="Definitive Gradient Test - Candlesticks with Curvature and Time Analysis (Proper Time)"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Curvature", range=[-1.1, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="Time Interval (dτ)", row=3, col=1)
    fig.update_xaxes(title_text="Proper Time (τ)", row=3, col=1)
    
    # Add curvature legend to the subplot
    from curvature_color_scheme import get_discrete_curvature_levels
    levels = get_discrete_curvature_levels()
    
    legend_y = 0.95
    for curvature, (color, description) in list(levels.items())[:5]:
        rgba_str = color.replace('rgba(', '').replace(')', '')
        r, g, b, a = map(float, rgba_str.split(', '))
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        
        fig.add_annotation(
            x=0.02, y=legend_y,
            text=f"● {curvature:+.1f}: {description.split(':')[1].strip()}",
            showarrow=False,
            font=dict(color=hex_color, size=9),
            xref="paper", yref="paper",
            xanchor='left',
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )
        legend_y -= 0.06
    
    return fig

def create_comparison_chart():
    """Create a comparison showing the old vs new approach."""
    
    print("\\n📊 CREATING COMPARISON CHART")
    print("=" * 40)
    
    from curvature_color_scheme import get_curvature_gradient_color
    
    fig = go.Figure()
    
    # Test curvature values
    test_curvatures = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for i, curvature in enumerate(test_curvatures):
        color, description = get_curvature_gradient_color(curvature)
        
        # Large rectangle showing the color
        fig.add_trace(go.Scatter(
            x=[i*2, i*2+1.5, i*2+1.5, i*2, i*2],
            y=[0, 0, 2, 2, 0],
            fill='toself',
            fillcolor=color,
            mode='lines',
            line=dict(width=2, color='white'),
            showlegend=False,
            hovertemplate=f'Curvature: {curvature}<br>{description}<extra></extra>'
        ))
        
        # Label
        fig.add_annotation(
            x=i*2 + 0.75,
            y=1,
            text=f"{curvature:+.1f}\\n{description.split()[0]}",
            showarrow=False,
            font=dict(color="white", size=12, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )
    
    fig.update_layout(
        title="Curvature Color Comparison - Should Show Clear Gradient",
        xaxis=dict(visible=False, range=[-0.5, len(test_curvatures)*2]),
        yaxis=dict(visible=False, range=[-0.5, 2.5]),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=300,
        width=1000
    )
    
    return fig

def main():
    """Run definitive gradient tests."""
    
    print("🎨 DEFINITIVE GRADIENT TESTING")
    print("=" * 60)
    
    # Test 1: Real curvature values
    fig1 = create_real_gradient_test()
    fig1.write_html("definitive_gradient_test.html")
    print("\\n✅ Saved: definitive_gradient_test.html")
    
    # Test 2: Comparison chart
    fig2 = create_comparison_chart()
    fig2.write_html("gradient_comparison.html")
    print("✅ Saved: gradient_comparison.html")
    
    print("\\n🎯 FINAL TEST:")
    print("Open definitive_gradient_test.html - you MUST see:")
    print("  - 10 candles with colors based on REAL curvature calculations")
    print("  - Annotations showing actual computed curvature values")
    print("  - Gradient colors from volatile (red/yellow) to trending (purple/cyan)")
    print("\\nThis now uses REAL curvature calculations, not forced values!")

if __name__ == "__main__":
    main()