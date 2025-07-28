"""
EXACT COPY of definitive_gradient_test.py but with main example data instead of forced curvatures.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shared_candle_visualization import create_curvature_candlestick_chart_shared
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

def generate_synthetic_market_data(n_candles: int = 200) -> pd.DataFrame:
    """
    Generate synthetic market data - EXACT COPY from example_usage.py
    """
    dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='1H')
    
    # EXACT same random seed and logic as example_usage.py
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_candles):
        # Varying volatility creates curvature changes
        volatility = 0.02 * (1 + 0.5 * np.sin(i * 0.1))
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLC from price series
    ohlc_data = []
    for i in range(n_candles):
        base = prices[i]
        
        # Generate intrabar movement
        movement = np.random.uniform(0.002, 0.01) * base
        
        open_price = base + np.random.uniform(-movement/2, movement/2)
        close_price = base + np.random.uniform(-movement/2, movement/2)
        
        high_price = max(open_price, close_price) + np.random.uniform(0, movement)
        low_price = min(open_price, close_price) - np.random.uniform(0, movement)
        
        ohlc_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    return df

def create_main_example_gradient_test():
    """EXACT COPY of create_forced_gradient_test but with real data and computed curvatures."""
    
    print("🎨 MAIN EXAMPLE GRADIENT TEST - COMPUTED CURVATURE VALUES")
    print("=" * 60)
    
    # Generate main example data instead of simple data
    ohlc_data = generate_synthetic_market_data(n_candles=200)
    
    # Compute curvatures instead of forcing them
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    computed_curvatures = geometry.compute_curvature_series()
    
    print("Computed curvature values and expected colors:")
    from curvature_color_scheme import get_curvature_gradient_color
    for i, curvature in enumerate(computed_curvatures[:10]):  # Show first 10
        color, description = get_curvature_gradient_color(curvature)
        print(f"  Candle {i}: curvature={curvature:+.3f} -> {color} ({description})")
    
    # Create the chart with computed curvatures - EXACT SAME CALL
    fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=computed_curvatures,
        title="Curvature-Enhanced Candlestick Chart - Market Spacetime Visualization",
        debug=False  # Don't spam debug
    )
    
    # Add basic curvature legend showing color mapping
    from curvature_color_scheme import get_discrete_curvature_levels
    levels = get_discrete_curvature_levels()
    
    # Add simple legend
    legend_y = 0.95
    for curvature, (color, description) in list(levels.items())[:5]:  # Show key levels only
        # Extract RGB from rgba
        rgba_str = color.replace('rgba(', '').replace(')', '')
        r, g, b, a = map(float, rgba_str.split(', '))
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        
        fig.add_annotation(
            x=0.02, y=legend_y,
            text=f"● {curvature:+.1f}: {description.split(':')[1].strip()}",
            showarrow=False,
            font=dict(color=hex_color, size=10),
            xref="paper", yref="paper",
            xanchor='left',
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )
        legend_y -= 0.08
    
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
    """Run main example gradient test - EXACT COPY of definitive test main."""
    
    print("🎨 MAIN EXAMPLE GRADIENT TESTING")
    print("=" * 60)
    
    # Test 1: Main example data with computed curvatures
    fig1 = create_main_example_gradient_test()
    fig1.write_html("curvature_candlesticks.html")
    print("\\n✅ Saved: curvature_candlesticks.html")
    
    # Test 2: Comparison chart (same as definitive test)
    fig2 = create_comparison_chart()
    fig2.write_html("gradient_comparison.html")
    print("✅ Saved: gradient_comparison.html")
    
    print("\\n🎯 FINAL TEST:")
    print("Open curvature_candlesticks.html - you MUST see:")
    print("  - 200 candles with different colors based on curvature")
    print("  - Annotations on first 10 candles showing curvature values")
    print("  - Gradient colors from red (volatile) to cyan (trending)")
    print("\\nThis uses EXACT same code as definitive_gradient_test.py that worked!")

if __name__ == "__main__":
    main()