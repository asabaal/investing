#!/usr/bin/env python3
"""
Debug the non-parametric potential learning to understand why we're getting 0% agreement.

Let's examine what the learned potentials actually look like and diagnose the issues.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from market_data_database import MarketDataDatabase
from nonparametric_potential_learning import NonParametricPotentialLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_potential_functions():
    """Debug what the learned potential functions are actually returning."""
    
    logger.info("🔍 Debugging non-parametric potential functions")
    
    # Load market data
    db = MarketDataDatabase()
    symbol = "QQQ"
    
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    data = db.get_data(symbol, start_date=start_date, end_date=end_date)
    
    market_data = pd.DataFrame({
        'open': data['Open'],
        'high': data['High'],
        'low': data['Low'],
        'close': data['Close'],
        'volume': data['Volume']
    })
    
    # Initialize learner
    learner = NonParametricPotentialLearner(market_data, temperature_kT=0.5)
    
    # Learn potentials
    learned_methods = learner.compare_learning_methods()
    
    # Test evaluation points in valid phase space
    test_points = [
        (0.0, 0.5),    # Center
        (0.3, 0.3),    # Positive sentiment
        (-0.3, 0.3),   # Negative sentiment
        (0.0, 0.8),    # High UWR
        (0.0, 0.1),    # Low UWR
    ]
    
    print("\n" + "="*80)
    print("🔍 POTENTIAL FUNCTION DEBUGGING")
    print("="*80)
    
    for method_name, method_data in learned_methods.items():
        print(f"\n📊 {method_name.upper()}:")
        print(f"   Type: {method_data.get('type', 'unknown')}")
        
        # Get potential function based on method structure
        if 'potential_function' in method_data:
            potential_func = method_data['potential_function']
        elif method_name == 'adaptive_density' and 'grid_data' in method_data:
            # For adaptive density, create interpolation function from grid
            S, U, V = method_data['grid_data']
            def potential_func(s, u):
                s_idx = np.argmin(np.abs(S[0, :] - s))
                u_idx = np.argmin(np.abs(U[:, 0] - u))
                return V[u_idx, s_idx]
        else:
            print(f"   ERROR: No potential function found for {method_name}")
            continue
        
        # Test potential values
        print(f"   Potential values at test points:")
        for s, u in test_points:
            try:
                V = potential_func(s, u)
                print(f"     V({s:+.1f}, {u:.1f}) = {V:.6f}")
            except Exception as e:
                print(f"     V({s:+.1f}, {u:.1f}) = ERROR: {e}")
        
        # Test force field (numerical gradient)
        print(f"   Force field at test points:")
        for s, u in test_points:
            try:
                eps = 1e-4
                V0 = potential_func(s, u)
                V_s = potential_func(s + eps, u)
                V_u = potential_func(s, u + eps)
                
                dV_ds = (V_s - V0) / eps
                dV_du = (V_u - V0) / eps
                
                force = np.array([-dV_ds, -dV_du])
                force_mag = np.linalg.norm(force)
                
                print(f"     F({s:+.1f}, {u:.1f}) = [{force[0]:+.6f}, {force[1]:+.6f}] (|F|={force_mag:.6f})")
            except Exception as e:
                print(f"     F({s:+.1f}, {u:.1f}) = ERROR: {e}")
    
    # Check the actual trajectory data used for learning
    print(f"\n📈 TRAJECTORY DATA ANALYSIS:")
    print(f"   Number of data points: {len(learner.trajectory_points)}")
    print(f"   Sentiment range: [{learner.trajectory_points[:, 0].min():.3f}, {learner.trajectory_points[:, 0].max():.3f}]")
    print(f"   UWR range: [{learner.trajectory_points[:, 1].min():.3f}, {learner.trajectory_points[:, 1].max():.3f}]")
    print(f"   Phase space constraint violations: {np.sum(np.abs(learner.trajectory_points[:, 0]) + learner.trajectory_points[:, 1] > 1.0)}")
    
    # Show some sample trajectory points
    print(f"   Sample trajectory points:")
    for i in range(min(10, len(learner.trajectory_points))):
        s, u = learner.trajectory_points[i]
        print(f"     ({s:+.3f}, {u:.3f})")
    
    print("="*80)
    
    # Create diagnostic visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trajectory Data Distribution', 'Adaptive Density Potential', 
                       'Maximum Entropy Potential', 'Force Field Magnitude'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # 1. Show trajectory data distribution
    fig.add_trace(
        go.Scatter(
            x=learner.trajectory_points[:, 0],
            y=learner.trajectory_points[:, 1],
            mode='markers',
            marker=dict(size=4, color='blue', alpha=0.6),
            name='Trajectory Points'
        ),
        row=1, col=1
    )
    
    # Add phase space boundary
    sentiment_boundary = np.linspace(-1, 1, 100)
    uwr_upper = 1 - np.abs(sentiment_boundary)
    fig.add_trace(
        go.Scatter(
            x=sentiment_boundary,
            y=uwr_upper,  
            mode='lines',
            line=dict(color='red', width=2),
            name='Phase Space Boundary'
        ),
        row=1, col=1
    )
    
    # 2. Adaptive density potential
    if 'adaptive_density' in learned_methods:
        S, U, V = learned_methods['adaptive_density']['grid_data']
        fig.add_trace(
            go.Heatmap(
                x=S[0, :],
                y=U[:, 0],
                z=V,
                colorscale='RdBu_r',
                name='Adaptive Density'
            ),
            row=1, col=2
        )
    
    # 3. Maximum entropy potential (evaluate on grid)
    if 'maximum_entropy' in learned_methods:
        me_func = learned_methods['maximum_entropy']['potential_function']
        s_grid = np.linspace(-0.8, 0.8, 50)
        u_grid = np.linspace(0.1, 0.9, 50)
        S_me, U_me = np.meshgrid(s_grid, u_grid)
        
        V_me = np.zeros_like(S_me)
        for i in range(S_me.shape[0]):
            for j in range(S_me.shape[1]):
                if abs(S_me[i, j]) + U_me[i, j] <= 1.0:
                    try:
                        V_me[i, j] = me_func(S_me[i, j], U_me[i, j])
                    except:
                        V_me[i, j] = np.nan
                else:
                    V_me[i, j] = np.nan
        
        fig.add_trace(
            go.Heatmap(
                x=s_grid,
                y=u_grid,
                z=V_me,
                colorscale='RdBu_r',
                name='Maximum Entropy'
            ),
            row=2, col=1
        )
    
    # 4. Force field magnitude
    force_mag = np.zeros_like(S_me)
    if 'adaptive_density' in learned_methods:
        density_func = learned_methods['adaptive_density']['potential_function']
        
        for i in range(S_me.shape[0]):
            for j in range(S_me.shape[1]):
                if abs(S_me[i, j]) + U_me[i, j] <= 1.0:
                    try:
                        eps = 1e-4
                        s, u = S_me[i, j], U_me[i, j]
                        V0 = density_func(s, u)
                        V_s = density_func(s + eps, u)
                        V_u = density_func(s, u + eps)
                        
                        dV_ds = (V_s - V0) / eps
                        dV_du = (V_u - V0) / eps
                        
                        force_mag[i, j] = np.sqrt(dV_ds**2 + dV_du**2)
                    except:
                        force_mag[i, j] = np.nan
                else:
                    force_mag[i, j] = np.nan
    
    fig.add_trace(
        go.Heatmap(
            x=s_grid,
            y=u_grid,
            z=force_mag,
            colorscale='Viridis',
            name='Force Magnitude'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="🔍 Non-Parametric Potential Learning Diagnostics",
        height=800,
        width=1200,
        template="plotly_dark"
    )
    
    fig.write_html("phase_space_analysis/nonparametric_debug.html")
    logger.info("Saved diagnostic visualization to phase_space_analysis/nonparametric_debug.html")
    
    return learned_methods

if __name__ == "__main__":
    debug_potential_functions()