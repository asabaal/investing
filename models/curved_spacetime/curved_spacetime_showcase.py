"""
Curved Spacetime Market Visualization Showcase

A comprehensive script demonstrating all our curved spacetime visualizations:
1. Traditional gradient-filled candlesticks with proper time
2. Market Gravity Wells (3D spacetime landscape)
3. Time River with Volume Dams (temporal flow metaphor)
4. Hybrid Gravity Wells + Geodesic Paths (complete dynamics)

Features:
- Interactive menu to choose visualization type
- Support for both synthetic and real market data
- Volume (mass) effects fully integrated
- Dark mode styling throughout
- Export capabilities for all visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys

# Import our visualization modules
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color
from shared_candle_visualization import create_curvature_candlestick_chart_shared
from market_gravity_wells import create_market_gravity_wells_visualization
from time_river_visualization import create_time_river_visualization
from hybrid_gravity_geodesics import create_hybrid_gravity_geodesics_visualization


def create_synthetic_market_data_with_volume(n_candles: int = 50) -> pd.DataFrame:
    """
    Generate realistic synthetic market data with volume for testing.
    
    Creates data with interesting volume patterns:
    - High volume during volatile periods (market stress)
    - Low volume during consolidation
    - Volume spikes at trend changes
    """
    print(f"🎲 Generating {n_candles} candles of synthetic market data with volume...")
    
    dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='1H')
    
    # Set random seed for reproducible results
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    volumes = []
    
    # Create realistic price and volume patterns
    for i in range(1, n_candles):
        # Market regime changes
        regime_cycle = np.sin(i * 0.1) * 0.5 + 0.5  # 0 to 1
        
        # Volatility varies with regime
        base_volatility = 0.01 + 0.03 * regime_cycle
        volatility = base_volatility * (1 + 0.5 * np.sin(i * 0.05))
        
        # Price change
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Volume correlated with volatility and trend changes
        base_volume = 10000
        volatility_volume = base_volume * (1 + 3 * abs(change) / volatility)  # Higher volume on big moves
        trend_change_volume = base_volume * (1 + 2 * abs(np.sin(i * 0.08)))  # Volume spikes at trend changes
        
        volume = max(1000, volatility_volume + trend_change_volume + np.random.normal(0, base_volume * 0.2))
        volumes.append(volume)
    
    # Add initial volume
    volumes.insert(0, 15000)
    
    # Create OHLCV from price series
    ohlc_data = []
    for i in range(n_candles):
        base = prices[i]
        volume = volumes[i]
        
        # Generate intrabar movement (correlated with volume)
        volume_factor = volume / 20000  # Normalize around 1.0
        movement = np.random.uniform(0.005, 0.02) * base * np.sqrt(volume_factor)
        
        open_price = base + np.random.uniform(-movement/3, movement/3)
        close_price = base + np.random.uniform(-movement/3, movement/3)
        
        high_price = max(open_price, close_price) + np.random.uniform(0, movement)
        low_price = min(open_price, close_price) - np.random.uniform(0, movement)
        
        ohlc_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    
    print(f"   ✓ Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   ✓ Volume range: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
    print(f"   ✓ Average volume: {df['volume'].mean():,.0f}")
    
    return df


def create_test_data_with_extreme_cases() -> pd.DataFrame:
    """
    Create test data specifically designed to showcase extreme spacetime effects.
    """
    print("🧪 Creating test data with extreme spacetime effects...")
    
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(12)]
    
    ohlc_data = pd.DataFrame({
        # Mix of extreme cases to showcase different physics
        'open':  [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122],
        'high':  [140, 108, 107, 106.2, 108.05, 125, 150, 115, 145, 118.02, 125, 130],
        'low':   [60,  98,  101, 105.9, 107.98, 95,  80,  113.8, 90, 117.99, 115, 120],
        'close': [130, 105, 104.1, 106.1, 108.02, 120, 85,  114.1, 140, 118.01, 122, 125],
        
        # Extreme volume variations to create strong gravitational effects
        'volume': [
            100000,  # Massive opening candle (strong gravity well)
            5000,    # Low volume consolidation
            35000,   # Medium volume
            8000,    # Low volume
            3000,    # Very low volume (weak gravity)
            75000,   # High volume breakout
            150000,  # Massive volume spike (strongest gravity)
            12000,   # Medium-low volume
            90000,   # High volume trending
            4000,    # Low volume
            60000,   # High volume
            25000    # Medium volume close
        ]
    }, index=dates)
    
    print(f"   ✓ Extreme price range: ${ohlc_data['low'].min():.2f} - ${ohlc_data['high'].max():.2f}")
    print(f"   ✓ Volume ratio: {ohlc_data['volume'].max() / ohlc_data['volume'].min():.1f}:1")
    
    return ohlc_data


def create_enhanced_gradient_visualization(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Enhanced version of our gradient candlestick visualization with volume info.
    """
    print("🎨 Creating enhanced gradient candlestick visualization...")
    
    # Compute spacetime geometry with volume
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    proper_time_intervals = geometry.compute_proper_time_intervals()
    volumes = [candle.volume for candle in candle_metrics]
    
    # Create 4-subplot layout
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            "Curved Spacetime Candlesticks (Proper Time Coordinates)",
            "Spacetime Curvature Over Time",
            "Proper Time Intervals (Time Dilation)",
            "Trading Volume (Gravitational Mass)"
        ),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Get candlestick chart with proper time
    candle_fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=curvatures,
        title="",
        debug=False,
        use_proper_time=True,
        proper_times=proper_times
    )
    
    # Add candlestick traces
    for trace in candle_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add curvature chart
    fig.add_trace(go.Scatter(
        x=proper_times,
        y=curvatures,
        mode='lines+markers',
        line=dict(color='white', width=2),
        marker=dict(
            size=8,
            color=[get_curvature_gradient_color(c)[0].replace('0.8)', '1.0)') for c in curvatures],
            line=dict(color='white', width=1)
        ),
        name='Spacetime Curvature',
        hovertemplate='Proper Time: %{x:.2f}<br>Curvature: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    # Add proper time intervals chart
    fig.add_trace(go.Scatter(
        x=proper_times,
        y=proper_time_intervals,
        mode='lines+markers',
        line=dict(color='#ff6b6b', width=2),
        marker=dict(size=6, color='#ff6b6b', line=dict(color='white', width=1)),
        name='Time Dilation',
        hovertemplate='Proper Time: %{x:.2f}<br>dτ: %{y:.2f}<extra></extra>'
    ), row=3, col=1)
    
    # Add volume chart (gravitational mass)
    fig.add_trace(go.Bar(
        x=proper_times,
        y=volumes,
        marker=dict(
            color=[get_curvature_gradient_color(c)[0] for c in curvatures],
            line=dict(color='white', width=1)
        ),
        name='Volume (Mass)',
        hovertemplate='Proper Time: %{x:.2f}<br>Volume: %{y:,}<extra></extra>'
    ), row=4, col=1)
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="purple", opacity=0.3, row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", opacity=0.3, row=2, col=1)
    
    # Layout
    fig.update_layout(
        title="Enhanced Curved Spacetime Market Analysis",
        height=1200,
        showlegend=False,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9')
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Curvature", range=[-1.1, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="dτ", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_xaxes(title_text="Proper Time (τ)", row=4, col=1)
    
    return fig


def show_visualization_menu():
    """Display interactive menu for visualization selection."""
    print("\n" + "="*70)
    print("🌌 CURVED SPACETIME MARKET VISUALIZATION SHOWCASE")
    print("="*70)
    print("\nAvailable Visualizations:")
    print("1. Enhanced Gradient Candlesticks (4-panel analysis)")
    print("2. Market Gravity Wells (3D spacetime landscape)")
    print("3. Time River + Volume Dams (temporal flow metaphor)")
    print("4. Hybrid Gravity Wells + Geodesic Paths (complete dynamics)")
    print("5. Generate ALL visualizations")
    print("6. Exit")
    print("\nData Options:")
    print("A. Use synthetic market data (50 candles)")
    print("B. Use extreme test cases (12 candles)")
    print("-"*70)


def main():
    """Main interactive visualization showcase."""
    
    parser = argparse.ArgumentParser(description="Curved Spacetime Market Visualization Showcase")
    parser.add_argument("--auto", action="store_true", help="Generate all visualizations automatically")
    parser.add_argument("--data", choices=["synthetic", "extreme"], default="synthetic", 
                       help="Data type to use")
    args = parser.parse_args()
    
    if args.auto:
        # Automatic mode - generate all visualizations
        print("🚀 AUTO MODE: Generating all visualizations...")
        
        if args.data == "extreme":
            ohlc_data = create_test_data_with_extreme_cases()
        else:
            ohlc_data = create_synthetic_market_data_with_volume(50)
        
        # Generate all visualizations
        visualizations = [
            ("enhanced_gradient", create_enhanced_gradient_visualization),
            ("gravity_wells", create_market_gravity_wells_visualization),
            ("time_river", create_time_river_visualization),
            ("hybrid_geodesics", create_hybrid_gravity_geodesics_visualization)
        ]
        
        print(f"\n📊 Generating {len(visualizations)} visualizations...")
        for name, func in visualizations:
            try:
                print(f"   Creating {name}...")
                fig = func(ohlc_data)
                filename = f"showcase_{name}.html"
                fig.write_html(filename)
                print(f"   ✅ Saved: {filename}")
            except Exception as e:
                print(f"   ❌ Error creating {name}: {e}")
        
        print("\n🎯 All visualizations generated!")
        return
    
    # Interactive mode
    while True:
        show_visualization_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "6":
                print("👋 Goodbye!")
                break
            
            if choice not in ["1", "2", "3", "4", "5"]:
                print("❌ Invalid choice. Please select 1-6.")
                continue
            
            # Select data type
            data_choice = input("\nSelect data type (A/B): ").strip().upper()
            if data_choice == "B":
                ohlc_data = create_test_data_with_extreme_cases()
            else:
                ohlc_data = create_synthetic_market_data_with_volume(50)
            
            if choice == "1":
                print("\n🎨 Creating Enhanced Gradient Candlesticks...")
                fig = create_enhanced_gradient_visualization(ohlc_data)
                fig.write_html("showcase_enhanced_gradient.html")
                print("✅ Saved: showcase_enhanced_gradient.html")
                
            elif choice == "2":
                print("\n🌌 Creating Market Gravity Wells...")
                fig = create_market_gravity_wells_visualization(ohlc_data)
                fig.write_html("showcase_gravity_wells.html")
                print("✅ Saved: showcase_gravity_wells.html")
                
            elif choice == "3":
                print("\n🌊 Creating Time River + Volume Dams...")
                fig = create_time_river_visualization(ohlc_data)
                fig.write_html("showcase_time_river.html")
                print("✅ Saved: showcase_time_river.html")
                
            elif choice == "4":
                print("\n🌌🛤️  Creating Hybrid Gravity Wells + Geodesics...")
                fig = create_hybrid_gravity_geodesics_visualization(ohlc_data)
                fig.write_html("showcase_hybrid_geodesics.html")
                print("✅ Saved: showcase_hybrid_geodesics.html")
                
            elif choice == "5":
                print("\n🚀 Creating ALL visualizations...")
                
                visualizations = [
                    ("enhanced_gradient", create_enhanced_gradient_visualization),
                    ("gravity_wells", create_market_gravity_wells_visualization),
                    ("time_river", create_time_river_visualization),
                    ("hybrid_geodesics", create_hybrid_gravity_geodesics_visualization)
                ]
                
                for name, func in visualizations:
                    try:
                        print(f"   Creating {name}...")
                        fig = func(ohlc_data)
                        filename = f"showcase_{name}.html"
                        fig.write_html(filename)
                        print(f"   ✅ Saved: {filename}")
                    except Exception as e:
                        print(f"   ❌ Error creating {name}: {e}")
            
            print("\n🎯 Visualization(s) created! Open the HTML files to view.")
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()