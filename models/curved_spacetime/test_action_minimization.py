#!/usr/bin/env python3
"""
Test the Market Action Minimization Framework

This script demonstrates the action minimization approach by:
1. Loading existing market data and GMM analysis
2. Creating the action minimization framework
3. Computing action for observed vs optimal paths
4. Visualizing the potential energy landscape
5. Comparing statistical (GMM) vs physical (action-minimized) trajectories
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging

from market_data_database import MarketDataDatabase
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from market_action_minimization import MarketActionMinimization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_action_minimization_framework():
    """Test the complete action minimization framework."""
    
    logger.info("🎯 Testing Market Action Minimization Framework")
    
    # 1. Load market data
    logger.info("📊 Loading market data...")
    db = MarketDataDatabase()
    symbol = "QQQ"  # Use QQQ as our test case
    
    try:
        # Get recent data
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
        data = db.get_data(symbol, start_date=start_date, end_date=end_date)
        logger.info(f"Loaded {len(data)} candles for {symbol}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # 2. Create curved spacetime geometry
    logger.info("🌌 Creating curved spacetime geometry...")
    
    # Map database columns to expected OHLC format
    data_ohlc = pd.DataFrame({
        'open': data['Open'],
        'high': data['High'], 
        'low': data['Low'],
        'close': data['Close'],
        'volume': data['Volume']
    })
    
    candle_metrics = create_candle_metrics_from_ohlc(data_ohlc)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    # 3. Load GMM analysis and create action minimization framework
    logger.info("🎲 Loading GMM analysis and creating action framework...")
    gmm_file = Path("phase_space_analysis/candle_geometry_classification.json")
    
    if not gmm_file.exists():
        logger.error(f"GMM analysis file not found: {gmm_file}")
        logger.info("Please run the candle geometry classifier first!")
        return
    
    action_minimizer = MarketActionMinimization.create_from_gmm_analysis(geometry, str(gmm_file))
    
    # 4. Analyze potential energy landscape
    logger.info("⚡ Analyzing potential energy landscape...")
    
    # Create grid for potential energy visualization
    sentiment_grid = np.linspace(-0.8, 0.8, 100)
    uwr_grid = np.linspace(0.1, 0.9, 100)
    S, U = np.meshgrid(sentiment_grid, uwr_grid)
    
    # Apply triangular constraint
    valid_mask = np.abs(S) + U <= 1.0
    
    # Compute potential energy at each grid point
    potential_energy = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if valid_mask[i, j]:
                potential_energy[i, j] = action_minimizer.potential_energy(S[i, j], U[i, j])
            else:
                potential_energy[i, j] = np.nan
    
    # 5. Compute action for observed trajectory
    logger.info("🛤️ Computing action for observed market trajectory...")
    
    # Extract observed path from market data
    observed_path = []
    proper_times = geometry.compute_proper_time_series()
    
    for candle in geometry.candles:
        observed_path.append(candle.pattern_coordinates)
    
    observed_action = action_minimizer.compute_action_along_path(observed_path, proper_times)
    logger.info(f"Observed trajectory action: {observed_action:.6f}")
    
    # 6. Test action minimization between two points
    logger.info("🎯 Testing action minimization between selected points...")
    
    if len(observed_path) >= 20:
        start_point = observed_path[5]
        end_point = observed_path[15]
        start_time = proper_times[5]
        end_time = proper_times[15]
        
        optimal_path, optimal_action = action_minimizer.find_action_minimizing_path(
            start_point, end_point, start_time, end_time, n_intermediate_points=8
        )
        
        logger.info(f"Optimal path action: {optimal_action:.6f}")
        
        # Compute action for straight line (comparison)
        straight_path = []
        n_points = len(optimal_path)
        for i in range(n_points):
            alpha = i / (n_points - 1)
            point = (1 - alpha) * start_point + alpha * end_point
            straight_path.append(point)
        
        straight_times = np.linspace(start_time, end_time, n_points)
        straight_action = action_minimizer.compute_action_along_path(straight_path, straight_times)
        logger.info(f"Straight line action: {straight_action:.6f}")
        
    else:
        optimal_path = None
        straight_path = None
    
    # 7. Create comprehensive visualization
    logger.info("📊 Creating visualization...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Potential Energy Landscape',
            'Action Comparison: Observed vs Optimal',
            'Potential Wells (GMM Clusters)',
            'Market Trajectory Analysis'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Potential energy landscape
    fig.add_trace(
        go.Heatmap(
            x=sentiment_grid,
            y=uwr_grid,
            z=potential_energy,
            colorscale='RdBu_r',
            name='Potential Energy',
            hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Energy: %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add potential well centers
    for well in action_minimizer.potential_wells:
        fig.add_trace(
            go.Scatter(
                x=[well.center[0]],
                y=[well.center[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='yellow',
                    line=dict(color='black', width=2),
                    symbol='star'
                ),
                name=well.name,
                showlegend=False,
                hovertemplate=f'<b>{well.name}</b><br>Energy Well<br>Depth: {well.depth:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Action comparison
    if optimal_path is not None:
        actions = ['Observed', 'Optimal', 'Straight Line']
        action_values = [
            observed_action / len(observed_path),  # Normalize by path length
            optimal_action,
            straight_action
        ]
        
        fig.add_trace(
            go.Bar(
                x=actions,
                y=action_values,
                marker_color=['blue', 'green', 'red'],
                name='Action Values',
                hovertemplate='%{x}<br>Action: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Potential wells visualization
    well_names = [well.name for well in action_minimizer.potential_wells]
    well_depths = [well.depth for well in action_minimizer.potential_wells]
    well_frequencies = [well.frequency for well in action_minimizer.potential_wells]
    
    fig.add_trace(
        go.Scatter(
            x=well_frequencies,
            y=well_depths,
            mode='markers+text',
            marker=dict(
                size=[f/2 for f in well_frequencies],
                color=well_depths,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=well_names,
            textposition='top center',
            name='Potential Wells',
            hovertemplate='<b>%{text}</b><br>Frequency: %{x:.1f}%<br>Depth: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Market trajectory
    trajectory_s = [pos[0] for pos in observed_path]
    trajectory_u = [pos[1] for pos in observed_path]
    
    fig.add_trace(
        go.Scatter(
            x=trajectory_s,
            y=trajectory_u,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color='blue'),
            name='Observed Trajectory',
            hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add optimal path if computed
    if optimal_path is not None:
        optimal_s = [pos[0] for pos in optimal_path]
        optimal_u = [pos[1] for pos in optimal_path]
        
        fig.add_trace(
            go.Scatter(
                x=optimal_s,
                y=optimal_u,
                mode='lines+markers',
                line=dict(color='green', width=3, dash='dash'),
                marker=dict(size=6, color='green', symbol='diamond'),
                name='Action-Minimized Path',
                hovertemplate='Optimal Path<br>Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"🎯 Market Action Minimization Analysis - {symbol}<br>" +
                 "<sub>Comparing observed market paths with action-minimized trajectories</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1000,
        width=1400,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=1)
    
    fig.update_xaxes(title_text="Path Type", row=1, col=2)
    fig.update_yaxes(title_text="Action Value", row=1, col=2)
    
    fig.update_xaxes(title_text="Statistical Frequency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Potential Well Depth", row=2, col=1)
    
    fig.update_xaxes(title_text="Sentiment", row=2, col=2)
    fig.update_yaxes(title_text="Upper Wick Ratio", row=2, col=2)
    
    # Save visualization
    output_file = Path("phase_space_analysis/market_action_minimization_test.html")
    fig.write_html(str(output_file))
    logger.info(f"📊 Visualization saved to {output_file}")
    
    # 8. Summary report
    print("\n" + "="*80)
    print("🎯 MARKET ACTION MINIMIZATION TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   • Symbol: {symbol}")
    print(f"   • Candles analyzed: {len(geometry.candles)}")
    print(f"   • Potential wells: {len(action_minimizer.potential_wells)}")
    
    print(f"\n⚡ POTENTIAL ENERGY ANALYSIS:")
    for well in action_minimizer.potential_wells:
        print(f"   • {well.name}: depth={well.depth:.3f}, frequency={well.frequency:.1f}%")
    
    print(f"\n🛤️ ACTION ANALYSIS:")
    print(f"   • Observed trajectory action: {observed_action:.6f}")
    if optimal_path is not None:
        print(f"   • Action-minimized path: {optimal_action:.6f}")
        print(f"   • Straight line path: {straight_action:.6f}")
        
        improvement = ((straight_action - optimal_action) / straight_action) * 100
        print(f"   • Action reduction vs straight line: {improvement:.2f}%")
    
    print(f"\n🏆 CONCLUSION:")
    if optimal_path is not None and optimal_action < straight_action:
        print(f"   ✅ Action minimization working correctly!")
        print(f"   ✅ Optimal path has lower action than straight line")
    else:
        print(f"   ⚠️  Need to investigate action minimization behavior")
    
    print("="*80)
    
    return {
        'geometry': geometry,
        'action_minimizer': action_minimizer,
        'observed_action': observed_action,
        'visualization_file': output_file
    }

if __name__ == "__main__":
    results = test_action_minimization_framework()