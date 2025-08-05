"""
Energy Landscape Visualization

Creates interactive visualizations of market energy states including:
- Energy time series with component breakdown
- 3D energy landscape in pattern space
- Energy flow vectors and gradient fields
- Equilibrium points and energy wells
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from market_hamiltonian import MarketHamiltonian, analyze_market_energy
from curved_candle_geometry import create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color


def create_energy_time_series(analysis_results: dict, dates: pd.DatetimeIndex) -> go.Figure:
    """
    Create time series plot showing total energy and components.
    """
    energies = analysis_results['energies']
    components = analysis_results['energy_components']
    
    # Extract component time series
    kinetic = [c['kinetic'] for c in components]
    potential = [c['potential'] for c in components]
    interaction = [c['interaction'] for c in components]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Market Energy States', 'Energy Components')
    )
    
    # Total energy trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=energies,
            mode='lines',
            name='Total Energy (H)',
            line=dict(color='#e0aaff', width=3),
            fill='tozeroy',
            fillcolor='rgba(224, 170, 255, 0.2)'
        ),
        row=1, col=1
    )
    
    # Mark high/low energy periods
    high_energy = analysis_results['high_energy_periods']
    low_energy = analysis_results['low_energy_periods']
    
    if len(high_energy) > 0:
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[i] if hasattr(dates, 'iloc') else dates[i] for i in high_energy],
                y=[energies[i] for i in high_energy],
                mode='markers',
                name='High Energy',
                marker=dict(color='#ff006e', size=10, symbol='triangle-up'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    if len(low_energy) > 0:
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[i] if hasattr(dates, 'iloc') else dates[i] for i in low_energy],
                y=[energies[i] for i in low_energy],
                mode='markers',
                name='Low Energy',
                marker=dict(color='#3a86ff', size=10, symbol='triangle-down'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Component traces
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=kinetic,
            mode='lines',
            name='Kinetic (T)',
            line=dict(color='#ff006e', width=2),
            stackgroup='energy'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=potential,
            mode='lines',
            name='Potential (V)',
            line=dict(color='#3a86ff', width=2),
            stackgroup='energy'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=interaction,
            mode='lines',
            name='Interaction (I)',
            line=dict(color='#f72585', width=2),
            stackgroup='energy'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Market Energy Evolution',
        template='plotly_dark',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Energy', row=1, col=1)
    fig.update_yaxes(title_text='Energy', row=2, col=1)
    
    return fig


def create_energy_landscape_3d(analysis_results: dict) -> go.Figure:
    """
    Create 3D energy landscape in pattern space (sentiment, UWR, energy).
    """
    states = analysis_results['states']
    
    # Extract coordinates and energies
    sentiments = [s.candle.sentiment for s in states]
    uwrs = [s.candle.upper_wick_ratio for s in states]
    energies = analysis_results['energies']
    
    # Create a grid for the energy surface
    sentiment_range = np.linspace(-0.99, 0.99, 50)
    uwr_range = np.linspace(0.01, 0.99, 50)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # Initialize the Hamiltonian (use average values from data)
    hamiltonian = MarketHamiltonian()
    avg_range = np.mean([s.candle.range_value for s in states])
    avg_volume = np.mean([s.candle.volume for s in states])
    
    # Compute energy surface
    E_grid = np.zeros_like(S_grid)
    for i in range(len(sentiment_range)):
        for j in range(len(uwr_range)):
            s, u = S_grid[j, i], U_grid[j, i]
            
            # Skip invalid regions (triangular constraint)
            if abs(s) + u > 1.0:
                E_grid[j, i] = np.nan
                continue
            
            # Create a test state at this point
            from market_hamiltonian import MarketState
            from curved_candle_geometry import CandleMetric
            
            test_candle = CandleMetric(
                range_value=avg_range,
                low_value=100,  # Arbitrary
                sentiment=s,
                upper_wick_ratio=u,
                volume=avg_volume
            )
            
            test_state = MarketState(
                candle=test_candle,
                velocity=np.zeros(2),
                acceleration=np.zeros(2),
                neighboring_states=None
            )
            
            E_grid[j, i] = hamiltonian.total_energy(test_state)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Energy surface
    fig.add_trace(go.Surface(
        x=sentiment_range,
        y=uwr_range,
        z=E_grid,
        colorscale='Viridis',
        opacity=0.8,
        name='Energy Landscape',
        showscale=True,
        colorbar=dict(title='Energy')
    ))
    
    # Actual trajectory
    fig.add_trace(go.Scatter3d(
        x=sentiments,
        y=uwrs,
        z=energies,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=energies,
            colorscale='Hot',
            showscale=False,
            line=dict(color='white', width=2)
        ),
        line=dict(color='white', width=3),
        name='Market Trajectory'
    ))
    
    # Update layout
    fig.update_layout(
        title='Market Energy Landscape',
        scene=dict(
            xaxis_title='Sentiment',
            yaxis_title='Upper Wick Ratio',
            zaxis_title='Energy',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        template='plotly_dark',
        height=700
    )
    
    return fig


def create_energy_flow_vectors(analysis_results: dict) -> go.Figure:
    """
    Create 2D vector field showing energy gradients and flow.
    """
    states = analysis_results['states']
    gradients = analysis_results['gradients']
    
    # Extract coordinates
    sentiments = [s.candle.sentiment for s in states]
    uwrs = [s.candle.upper_wick_ratio for s in states]
    
    # Create figure
    fig = go.Figure()
    
    # Background heatmap of energies
    energies = analysis_results['energies']
    
    # Create a finer grid for interpolated heatmap
    from scipy.interpolate import griddata
    
    sentiment_range = np.linspace(-0.99, 0.99, 100)
    uwr_range = np.linspace(0.01, 0.99, 100)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # Interpolate energies onto grid
    points = np.column_stack((sentiments, uwrs))
    E_grid = griddata(points, energies, (S_grid, U_grid), method='cubic')
    
    # Mask invalid regions
    for i in range(len(sentiment_range)):
        for j in range(len(uwr_range)):
            if abs(S_grid[j, i]) + U_grid[j, i] > 1.0:
                E_grid[j, i] = np.nan
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        x=sentiment_range,
        y=uwr_range,
        z=E_grid,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Energy'),
        hovertemplate='Sentiment: %{x:.2f}<br>UWR: %{y:.2f}<br>Energy: %{z:.2f}<extra></extra>'
    ))
    
    # Add gradient vectors (force field)
    # Sample every few points to avoid clutter
    step = max(1, len(states) // 20)
    
    for i in range(0, len(states), step):
        if gradients[i] is None or np.all(gradients[i] == 0):
            continue
        
        # Force is negative gradient
        force = -gradients[i]
        force_magnitude = np.linalg.norm(force)
        
        if force_magnitude > 0:
            # Normalize and scale for visualization
            force_normalized = force / force_magnitude * 0.05
            
            # Arrow from current position in direction of force
            fig.add_annotation(
                x=sentiments[i] + force_normalized[0],
                y=uwrs[i] + force_normalized[1],
                ax=sentiments[i],
                ay=uwrs[i],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='white',
                opacity=0.8
            )
    
    # Add trajectory
    fig.add_trace(go.Scatter(
        x=sentiments,
        y=uwrs,
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='white', line=dict(color='red', width=2)),
        name='Market Path',
        hovertemplate='Step %{customdata}<br>Sentiment: %{x:.2f}<br>UWR: %{y:.2f}<extra></extra>',
        customdata=list(range(len(sentiments)))
    ))
    
    # Update layout
    fig.update_layout(
        title='Energy Flow Field',
        xaxis_title='Sentiment',
        yaxis_title='Upper Wick Ratio',
        template='plotly_dark',
        height=700,
        width=800,
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_comprehensive_energy_dashboard(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive dashboard with all energy visualizations.
    """
    # Perform energy analysis
    analysis = analyze_market_energy(ohlc_data)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.5, 0.5],
        column_widths=[0.6, 0.4],
        subplot_titles=(
            'Energy Time Series',
            'Energy Landscape 3D',
            'Energy Components',
            'Energy Flow Vectors'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'surface', 'rowspan': 2}],
            [{'type': 'scatter'}, None]
        ]
    )
    
    # Add traces from individual visualizations
    # (This is a simplified version - in practice you'd adapt each viz)
    
    # 1. Energy time series
    energies = analysis['energies']
    dates = ohlc_data.index
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=energies,
            mode='lines',
            name='Total Energy',
            line=dict(color='#e0aaff', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Component breakdown
    components = analysis['energy_components']
    kinetic = [c['kinetic'] for c in components]
    potential = [c['potential'] for c in components]
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=kinetic,
            mode='lines',
            name='Kinetic',
            line=dict(color='#ff006e')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=potential,
            mode='lines',
            name='Potential',
            line=dict(color='#3a86ff')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Market Energy Analysis Dashboard',
        template='plotly_dark',
        height=800,
        showlegend=True
    )
    
    return fig


def main():
    """Test the energy visualization system."""
    
    # Create test data
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(20)]
    
    # Generate data that shows energy transitions
    ohlc_data = pd.DataFrame({
        'open':  [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                  111, 110, 108, 106, 104, 105, 107, 109, 111, 113],
        'high':  [103, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                  113, 112, 110, 108, 106, 107, 109, 111, 113, 115],
        'low':   [99, 101, 100, 102, 104, 103, 105, 107, 106, 108,
                  110, 109, 107, 105, 103, 104, 106, 108, 110, 112],
        'close': [102, 101, 103, 105, 104, 106, 108, 107, 109, 111,
                  110, 108, 106, 104, 105, 107, 109, 111, 113, 114],
        'volume': [10000, 15000, 8000, 20000, 25000, 12000, 30000, 18000, 22000, 35000,
                   28000, 16000, 14000, 11000, 9000, 13000, 17000, 21000, 26000, 32000]
    }, index=dates)
    
    # Perform analysis
    analysis = analyze_market_energy(ohlc_data)
    
    # Create visualizations
    energy_ts = create_energy_time_series(analysis, dates)
    energy_ts.write_html("energy_time_series.html")
    
    energy_landscape = create_energy_landscape_3d(analysis)
    energy_landscape.write_html("energy_landscape_3d.html")
    
    energy_flow = create_energy_flow_vectors(analysis)
    energy_flow.write_html("energy_flow_vectors.html")
    
    print("✅ Energy visualizations created:")
    print("   - energy_time_series.html")
    print("   - energy_landscape_3d.html") 
    print("   - energy_flow_vectors.html")
    print("\n📊 Analysis Summary:")
    print(f"   Mean Energy: {analysis['mean_energy']:.2f}")
    print(f"   High Energy Periods: {len(analysis['high_energy_periods'])}")
    print(f"   Low Energy Periods: {len(analysis['low_energy_periods'])}")


if __name__ == "__main__":
    main()