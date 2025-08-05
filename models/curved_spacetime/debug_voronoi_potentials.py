#!/usr/bin/env python3
"""
Debug the Voronoi well geometry potentials to understand why validation is failing.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from market_data_database import MarketDataDatabase
from voronoi_well_geometry_learning import VoronoiWellGeometryLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_voronoi_potentials():
    """Debug the Voronoi potential functions."""
    
    logger.info("🔍 Debugging Voronoi well geometry potentials")
    
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
    learner = VoronoiWellGeometryLearner(market_data, temperature_kT=0.5)
    
    # Step 1: Detect cluster centers
    cluster_centers, cluster_assignments = learner.detect_cluster_centers_with_gmm(n_clusters=5)
    
    # Step 2: Create Voronoi boundaries
    voronoi_regions = learner.create_true_voronoi_boundaries(cluster_centers)
    
    # Step 3: Learn well geometry
    well_geometries = learner.learn_well_geometry_in_regions(voronoi_regions, cluster_assignments)
    
    # Step 4: Create piecewise potential
    piecewise_potential = learner.create_piecewise_potential_function(well_geometries)
    
    # Test potential evaluation at various points
    test_points = [
        (0.0, 0.5),    # Center
        (0.3, 0.3),    # Positive sentiment
        (-0.3, 0.3),   # Negative sentiment
        (0.0, 0.8),    # High UWR
        (0.0, 0.1),    # Low UWR
        (0.6, 0.2),    # Near cluster 0 center
        (-0.6, 0.1),   # Near cluster 1 center
    ]
    
    print("\n" + "="*80)
    print("🔍 VORONOI POTENTIAL DEBUGGING")
    print("="*80)
    
    print(f"\n📊 LEARNED WELLS: {len(well_geometries)}")
    for cluster_id, geom in well_geometries.items():
        center = geom['center']
        n_points = geom['n_points']
        freq = geom['frequency']
        print(f"   Cluster {cluster_id}: center=({center[0]:.3f}, {center[1]:.3f}), points={n_points}, freq={freq:.1%}")
    
    print(f"\n🎯 TESTING PIECEWISE POTENTIAL:")
    for s, u in test_points:
        if abs(s) + u > 1.0:
            print(f"   V({s:+.1f}, {u:.1f}) = INVALID (outside phase space)")
            continue
            
        try:
            V = piecewise_potential(s, u)
            print(f"   V({s:+.1f}, {u:.1f}) = {V:.6f}")
        except Exception as e:
            print(f"   V({s:+.1f}, {u:.1f}) = ERROR: {e}")
    
    # Test individual well potentials
    print(f"\n🔧 TESTING INDIVIDUAL WELL POTENTIALS:")
    for cluster_id, geom in well_geometries.items():
        print(f"   Cluster {cluster_id}:")
        potential_func = geom['potential_function']
        center = geom['center']
        
        # Test at cluster center
        try:
            V_center = potential_func(center[0], center[1])
            print(f"     V(center) = {V_center:.6f}")
        except Exception as e:
            print(f"     V(center) = ERROR: {e}")
        
        # Test nearby points
        for offset in [0.05, -0.05]:
            try:
                s_test = center[0] + offset
                u_test = center[1]
                if abs(s_test) + u_test <= 1.0:
                    V_test = potential_func(s_test, u_test)
                    print(f"     V({s_test:+.3f}, {u_test:.3f}) = {V_test:.6f}")
            except Exception as e:
                print(f"     V nearby = ERROR: {e}")
    
    # Test force field computation
    print(f"\n⚡ TESTING FORCE FIELD:")
    for s, u in [(0.0, 0.5), (0.3, 0.3)]:
        if abs(s) + u > 1.0:
            continue
            
        try:
            eps = 1e-4
            V0 = piecewise_potential(s, u)
            V_s_plus = piecewise_potential(s + eps, u)
            V_u_plus = piecewise_potential(s, u + eps)
            
            dV_ds = (V_s_plus - V0) / eps
            dV_du = (V_u_plus - V0) / eps
            
            force = np.array([-dV_ds, -dV_du])
            force_mag = np.linalg.norm(force)
            
            print(f"   F({s:+.1f}, {u:.1f}) = [{force[0]:+.6f}, {force[1]:+.6f}] (|F|={force_mag:.6f})")
        except Exception as e:
            print(f"   F({s:+.1f}, {u:.1f}) = ERROR: {e}")
    
    print("="*80)
    
    return {
        'learner': learner,
        'voronoi_regions': voronoi_regions,
        'well_geometries': well_geometries,
        'piecewise_potential': piecewise_potential
    }

if __name__ == "__main__":
    results = debug_voronoi_potentials()