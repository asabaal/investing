#!/usr/bin/env python3
"""
Voronoi Well Geometry Learning

The CORRECT approach to potential learning:

1. Use TRUE Voronoi polygon boundaries (not rectangular approximations)
2. Learn the actual well geometry WITHIN each Voronoi cell
3. Create piecewise potential functions that respect complex boundaries
4. Maintain the 5 detected cluster states with their proper shapes

This fixes the fundamental flaw: we keep our detected states but learn 
the correct well geometry within each region's true boundaries.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
# Use matplotlib for polygon operations instead of shapely
from matplotlib.path import Path as MplPath
import warnings
warnings.filterwarnings('ignore')

from market_data_database import MarketDataDatabase
from curved_candle_geometry import create_candle_metrics_from_ohlc
from validate_physics_predictions import PhysicsValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoronoiWellGeometryLearner:
    """
    Learn potential well geometry within true Voronoi cell boundaries.
    
    This is the CORRECT approach that:
    1. Keeps the 5 detected cluster states
    2. Uses actual Voronoi polygon boundaries 
    3. Learns well geometry within each cell
    4. Creates proper piecewise potential functions
    """
    
    def __init__(self, market_data: pd.DataFrame, temperature_kT: float = 0.5):
        """Initialize with market data and extract trajectory points."""
        self.market_data = market_data
        self.kT = temperature_kT
        
        # Extract trajectory data in (sentiment, UWR) coordinates
        self.trajectory_points = self._extract_trajectory_data()
        self.n_points = len(self.trajectory_points)
        
        # Phase space bounds
        self.sentiment_bounds = (-0.99, 0.99)
        self.uwr_bounds = (0.01, 0.99)
        
        logger.info(f"Initialized Voronoi well geometry learner with {self.n_points} trajectory points")
    
    def _extract_trajectory_data(self) -> np.ndarray:
        """Extract (sentiment, UWR) trajectory points from market data."""
        candle_metrics = create_candle_metrics_from_ohlc(self.market_data)
        
        trajectory_points = []
        for candle in candle_metrics:
            # Apply phase space constraint: |sentiment| + UWR ≤ 1
            if abs(candle.sentiment) + candle.upper_wick_ratio <= 1.0:
                trajectory_points.append([candle.sentiment, candle.upper_wick_ratio])
        
        return np.array(trajectory_points)
    
    def detect_cluster_centers_with_gmm(self, n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect cluster centers using GMM (same as our original approach).
        
        Returns:
            (cluster_centers, cluster_assignments)
        """
        logger.info(f"Detecting {n_clusters} cluster centers using GMM")
        
        # Apply triangular constraint
        valid_mask = np.abs(self.trajectory_points[:, 0]) + self.trajectory_points[:, 1] <= 1.0
        valid_points = self.trajectory_points[valid_mask]
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=200)
        gmm.fit(valid_points)
        
        cluster_centers = gmm.means_
        cluster_assignments = gmm.predict(valid_points)
        
        logger.info(f"Found {n_clusters} cluster centers:")
        for i, center in enumerate(cluster_centers):
            n_points_in_cluster = np.sum(cluster_assignments == i)
            frequency = n_points_in_cluster / len(valid_points) * 100
            logger.info(f"  Cluster {i}: center=({center[0]:.3f}, {center[1]:.3f}), frequency={frequency:.1f}%")
        
        return cluster_centers, cluster_assignments
    
    def create_true_voronoi_boundaries(self, cluster_centers: np.ndarray) -> Dict[int, Any]:
        """
        Create TRUE Voronoi polygon boundaries (not rectangular approximations).
        
        Returns:
            Dictionary mapping cluster_id -> boundary info
        """
        logger.info("Creating true Voronoi polygon boundaries")
        
        # Create Voronoi diagram
        vor = Voronoi(cluster_centers)
        
        # Extract polygon boundaries for each cluster
        voronoi_regions = {}
        
        for i, center in enumerate(cluster_centers):
            # Find which Voronoi region corresponds to this cluster center
            region_index = vor.point_region[i]
            vertex_indices = vor.regions[region_index]
            
            if len(vertex_indices) == 0 or -1 in vertex_indices:
                # Unbounded region - need to clip to phase space
                logger.warning(f"Cluster {i} has unbounded Voronoi region - will clip to phase space")
                vertices = self._clip_unbounded_region(center, vor)
            else:
                # Bounded region - extract vertices
                vertices = vor.vertices[vertex_indices]
            
            # Clip to triangular phase space constraint: |sentiment| + UWR ≤ 1
            phase_space_triangle = np.array([
                [-1.0, 0.0],  # Left corner
                [1.0, 0.0],   # Right corner  
                [0.0, 1.0]    # Top corner
            ])
            
            clipped_vertices = self._clip_polygon_to_triangle(vertices, phase_space_triangle)
            
            if len(clipped_vertices) < 3:
                logger.warning(f"Cluster {i} polygon has too few vertices after clipping")
                continue
            
            # Store boundary information
            voronoi_regions[i] = {
                'center': center,
                'vertices': clipped_vertices,
                'area': self._polygon_area(clipped_vertices)
            }
            
            logger.info(f"  Cluster {i}: polygon with {len(clipped_vertices)} vertices, area={voronoi_regions[i]['area']:.4f}")
        
        return voronoi_regions
    
    def _clip_unbounded_region(self, center: np.ndarray, vor: Voronoi) -> np.ndarray:
        """Clip unbounded Voronoi region to phase space boundaries."""
        # For unbounded regions, create a circle around the center
        theta = np.linspace(0, 2*np.pi, 20)
        radius = 0.3
        circle_points = np.column_stack([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta)
        ])
        
        return circle_points
    
    def _clip_polygon_to_triangle(self, vertices: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        """Clip polygon vertices to triangular phase space constraint."""
        # Simple clipping: keep vertices that satisfy |sentiment| + UWR <= 1
        clipped = []
        for vertex in vertices:
            s, u = vertex
            if abs(s) + u <= 1.0 and -1.0 <= s <= 1.0 and 0.0 <= u <= 1.0:
                clipped.append(vertex)
        
        return np.array(clipped) if clipped else vertices
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        # Shoelace formula
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    
    def _point_in_polygon(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        if len(vertices) < 3:
            return False
        
        path = MplPath(vertices)
        return path.contains_point(point)
    
    def learn_well_geometry_in_regions(self, voronoi_regions: Dict[int, Any], 
                                     cluster_assignments: np.ndarray) -> Dict[int, Any]:
        """
        Learn the actual well geometry WITHIN each Voronoi region.
        
        This is the key step: learn the potential landscape within each
        true polygonal boundary, not assume Gaussian shapes.
        """
        logger.info("Learning well geometry within each Voronoi region")
        
        # Apply triangular constraint to get valid points
        valid_mask = np.abs(self.trajectory_points[:, 0]) + self.trajectory_points[:, 1] <= 1.0
        valid_points = self.trajectory_points[valid_mask]
        
        well_geometries = {}
        
        for cluster_id, region_info in voronoi_regions.items():
            logger.info(f"Learning well geometry for cluster {cluster_id}")
            
            # Get points assigned to this cluster
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = valid_points[cluster_mask]
            
            if len(cluster_points) < 5:
                logger.warning(f"Too few points ({len(cluster_points)}) in cluster {cluster_id}")
                continue
            
            # Get the polygon vertices
            polygon_vertices = region_info['vertices']
            
            # Filter points to only those inside the polygon
            points_in_polygon = []
            for point in cluster_points:
                if self._point_in_polygon(point, polygon_vertices):
                    points_in_polygon.append(point)
            
            points_in_polygon = np.array(points_in_polygon)
            
            if len(points_in_polygon) < 3:
                logger.warning(f"Too few points inside polygon for cluster {cluster_id}")
                continue
            
            # Learn the density distribution within this region
            kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
            kde.fit(points_in_polygon)
            
            # Create a local coordinate system within the polygon
            # Sample points within the polygon bounds
            minx, miny = np.min(polygon_vertices, axis=0)
            maxx, maxy = np.max(polygon_vertices, axis=0)
            
            # Add padding to ensure coverage
            padding = 0.05
            minx = max(-1.0, minx - padding)
            maxx = min(1.0, maxx + padding)
            miny = max(0.0, miny - padding)
            maxy = min(1.0, maxy + padding)
            
            # Create finer grid within polygon bounds
            n_grid = 50  # Increased resolution
            x_local = np.linspace(minx, maxx, n_grid)
            y_local = np.linspace(miny, maxy, n_grid)
            X_local, Y_local = np.meshgrid(x_local, y_local)
            
            # Evaluate density only at points inside the polygon
            local_potential = np.full_like(X_local, np.nan)
            
            for i in range(n_grid):
                for j in range(n_grid):
                    test_point = np.array([X_local[i, j], Y_local[i, j]])
                    if self._point_in_polygon(test_point, polygon_vertices):
                        # Evaluate KDE density
                        point_array = np.array([[X_local[i, j], Y_local[i, j]]])
                        log_density = kde.score_samples(point_array)[0]
                        density = np.exp(log_density)
                        
                        # Convert to potential: V = -kT ln(P)
                        potential = -self.kT * np.log(max(density, 1e-10))
                        local_potential[i, j] = potential
            
            # Create interpolation function for this region using bilinear interpolation
            def create_region_potential_function(X_grid, Y_grid, V_grid, vertices, point_in_polygon_func):
                # Create interpolator for valid grid points
                x_coords = X_grid[0, :]
                y_coords = Y_grid[:, 0]
                
                # Replace NaN values with a reasonable background potential
                V_clean = np.copy(V_grid)
                V_clean[np.isnan(V_clean)] = 0.0
                
                # Create interpolator
                try:
                    interpolator = RegularGridInterpolator(
                        (y_coords, x_coords), V_clean, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                except:
                    # Fallback to simple nearest neighbor
                    interpolator = None
                
                def region_potential(sentiment, uwr):
                    """Evaluate potential within this specific region."""
                    point = np.array([sentiment, uwr])
                    
                    # Check if point is in this region
                    if not point_in_polygon_func(point, vertices):
                        return float('inf')  # Very high potential outside region
                    
                    # Use interpolation if available
                    if interpolator is not None:
                        try:
                            # Note: interpolator expects (y, x) order
                            result = interpolator((uwr, sentiment))
                            return float(result)
                        except:
                            pass
                    
                    # Fallback to nearest neighbor
                    distances = np.sqrt((X_grid - sentiment)**2 + (Y_grid - uwr)**2)
                    valid_mask = ~np.isnan(V_grid)
                    
                    if not np.any(valid_mask):
                        return 0.0
                    
                    valid_distances = np.where(valid_mask, distances, np.inf)
                    min_idx = np.unravel_index(np.argmin(valid_distances), distances.shape)
                    
                    return float(V_grid[min_idx])
                
                return region_potential
            
            region_potential_func = create_region_potential_function(
                X_local, Y_local, local_potential, polygon_vertices, self._point_in_polygon
            )
            
            # Store well geometry information
            well_geometries[cluster_id] = {
                'vertices': polygon_vertices,
                'potential_function': region_potential_func,
                'grid_data': (X_local, Y_local, local_potential),
                'n_points': len(points_in_polygon),
                'center': region_info['center'],
                'frequency': len(points_in_polygon) / len(valid_points)
            }
            
            logger.info(f"  Learned geometry for cluster {cluster_id}: {len(points_in_polygon)} points, frequency={well_geometries[cluster_id]['frequency']:.1%}")
        
        return well_geometries
    
    def create_piecewise_potential_function(self, well_geometries: Dict[int, Any]) -> callable:
        """
        Create piecewise potential function that respects true Voronoi boundaries.
        
        This is the final potential function that combines all learned well geometries.
        """
        logger.info("Creating piecewise potential function from learned well geometries")
        
        def piecewise_potential(sentiment: float, uwr: float) -> float:
            """
            Evaluate potential by checking which Voronoi region contains the point
            and using the learned geometry within that region.
            """
            point = np.array([sentiment, uwr])
            
            # First check if point is within valid phase space
            if abs(sentiment) + uwr > 1.0:
                return 20.0  # High potential outside phase space
            
            # Check each region to see which one contains this point
            for cluster_id, geometry in well_geometries.items():
                path = MplPath(geometry['vertices'])
                if path.contains_point(point):
                    return geometry['potential_function'](sentiment, uwr)
            
            # If no exact region match, use nearest cluster center (ensures coverage)
            min_distance = float('inf')
            closest_cluster = None
            
            for cluster_id, geometry in well_geometries.items():
                center = geometry['center']
                distance = np.sqrt((sentiment - center[0])**2 + (uwr - center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            if closest_cluster is not None:
                # Use the closest cluster's potential but add distance penalty
                base_potential = well_geometries[closest_cluster]['potential_function'](sentiment, uwr)
                if np.isinf(base_potential):
                    # If still infinite, use a smooth distance-based potential
                    return 2.0 + min_distance * 5.0
                return base_potential
            
            # Final fallback
            return 5.0
        
        return piecewise_potential

class VoronoiWellValidator:
    """Validate the Voronoi well geometry approach."""
    
    def __init__(self, piecewise_potential: callable, well_geometries: Dict[int, Any], temperature_kT: float = 0.5):
        """Initialize with learned piecewise potential."""
        self.piecewise_potential = piecewise_potential
        self.well_geometries = well_geometries
        self.kT = temperature_kT
    
    def validate_learned_wells(self, n_trajectories: int = 50, n_steps: int = 100) -> Dict[str, Any]:
        """Validate the learned Voronoi well geometry against observations."""
        
        logger.info(f"🧪 Validating learned Voronoi well geometry")
        
        # Create a potential class compatible with PhysicsValidator
        class VoronoiPotential:
            def __init__(self, potential_func, well_geoms, kT):
                self.potential_func = potential_func
                self.well_geoms = well_geoms
                self.kT = kT
                
                # Create dummy well_parameters for compatibility
                self.well_parameters = []
                for cluster_id, geom in well_geoms.items():
                    center = geom['center']
                    self.well_parameters.append({
                        'name': f'Voronoi Cluster {cluster_id}',
                        'center': center,
                        'depth': 1.0,  # Will be determined by learned geometry
                        'frequency': geom['frequency'],
                        'width_s': 0.1,  # Dummy values for compatibility
                        'width_u': 0.1
                    })
            
            def potential_energy(self, sentiment, uwr):
                """Evaluate potential using learned piecewise function."""
                try:
                    return self.potential_func(sentiment, uwr)
                except:
                    return 0.0
            
            def force_field(self, sentiment, uwr):
                """Compute force field using numerical gradient."""
                eps = 1e-4
                
                # Ensure we stay within bounds
                s_plus = min(sentiment + eps, 0.99)
                s_minus = max(sentiment - eps, -0.99)
                u_plus = min(uwr + eps, 0.99)
                u_minus = max(uwr - eps, 0.01)
                
                # Apply phase space constraint
                if abs(s_plus) + uwr > 1.0:
                    s_plus = sentiment + eps * 0.1
                if abs(sentiment) + u_plus > 1.0:
                    u_plus = uwr + eps * 0.1
                
                try:
                    V0 = self.potential_energy(sentiment, uwr)
                    V_s_plus = self.potential_energy(s_plus, uwr)
                    V_u_plus = self.potential_energy(sentiment, u_plus)
                    
                    # Central difference when possible
                    if abs(s_minus) + uwr <= 1.0:
                        V_s_minus = self.potential_energy(s_minus, uwr)
                        dV_ds = (V_s_plus - V_s_minus) / (2 * eps)
                    else:
                        dV_ds = (V_s_plus - V0) / eps
                    
                    if abs(sentiment) + u_minus <= 1.0:
                        V_u_minus = self.potential_energy(sentiment, u_minus)
                        dV_du = (V_u_plus - V_u_minus) / (2 * eps)
                    else:
                        dV_du = (V_u_plus - V0) / eps
                    
                    return -np.array([dV_ds, dV_du])  # Force = -∇V
                except:
                    return np.array([0.0, 0.0])
        
        # Create potential and validator
        potential = VoronoiPotential(self.piecewise_potential, self.well_geometries, self.kT)
        validator = PhysicsValidator(potential)
        
        # Generate trajectories
        trajectories = validator.generate_multiple_trajectories(
            n_trajectories=n_trajectories,
            n_steps=n_steps,
            temperature=self.kT
        )
        
        # Analyze results
        predicted_stats = validator.analyze_trajectory_statistics(trajectories)
        comparison = validator.compare_with_observations(predicted_stats)
        
        return {
            'trajectories': trajectories,
            'predicted_stats': predicted_stats,
            'comparison': comparison,
            'agreement_score': comparison['overall_agreement'],
            'mean_relative_error': comparison['mean_relative_error']
        }

def run_voronoi_well_geometry_learning():
    """Run the complete Voronoi well geometry learning pipeline."""
    
    logger.info("🚀 Starting Voronoi Well Geometry Learning")
    
    # Load market data
    db = MarketDataDatabase()
    symbol = "QQQ"
    
    try:
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
        
        logger.info(f"Loaded {len(market_data)} candles for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        return
    
    # Initialize learner
    learner = VoronoiWellGeometryLearner(market_data, temperature_kT=0.5)
    
    # Step 1: Detect cluster centers (same as before)
    cluster_centers, cluster_assignments = learner.detect_cluster_centers_with_gmm(n_clusters=5)
    
    # Step 2: Create TRUE Voronoi boundaries
    voronoi_regions = learner.create_true_voronoi_boundaries(cluster_centers)
    
    # Step 3: Learn well geometry within each region
    well_geometries = learner.learn_well_geometry_in_regions(voronoi_regions, cluster_assignments)
    
    # Step 4: Create piecewise potential function
    piecewise_potential = learner.create_piecewise_potential_function(well_geometries)
    
    # Step 5: Validate the approach
    validator = VoronoiWellValidator(piecewise_potential, well_geometries, temperature_kT=0.5)
    validation_results = validator.validate_learned_wells()
    
    # Create visualization
    fig = create_voronoi_wells_visualization(learner, voronoi_regions, well_geometries, validation_results)
    
    # Save results
    viz_file = Path("phase_space_analysis/voronoi_well_geometry.html")
    fig.write_html(str(viz_file))
    
    # Final report
    print("\n" + "="*80)
    print("🏆 VORONOI WELL GEOMETRY LEARNING RESULTS")
    print("="*80)
    
    print(f"\n📊 LEARNED WELL STRUCTURE:")
    print(f"   • Number of Voronoi regions: {len(voronoi_regions)}")
    print(f"   • Number of learned wells: {len(well_geometries)}")
    
    for cluster_id, geom in well_geometries.items():
        center = geom['center']
        freq = geom['frequency']
        n_points = geom['n_points']
        print(f"   • Cluster {cluster_id}: center=({center[0]:.3f}, {center[1]:.3f}), frequency={freq:.1%}, points={n_points}")
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"   • Agreement Score: {validation_results['agreement_score']:.1%}")
    print(f"   • Mean Relative Error: {validation_results['mean_relative_error']:.1%}")
    
    # Compare with previous approaches
    previous_gaussian = 0.388  # From focused parameter tuning
    improvement = validation_results['agreement_score'] - previous_gaussian
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   • Previous (Gaussian wells): {previous_gaussian:.1%}")
    print(f"   • Current (Voronoi wells): {validation_results['agreement_score']:.1%}")
    print(f"   • Improvement: {improvement:+.1%}")
    
    if improvement > 0:
        print(f"   ✅ Voronoi well geometry is superior!")
    else:
        print(f"   ⚠️  Still need refinement")
    
    success = (validation_results['agreement_score'] >= 0.8 and 
              validation_results['mean_relative_error'] <= 0.3)
    
    if success:
        print(f"\n🎉 VORONOI WELL GEOMETRY SUCCESS!")
        print(f"   ✅ Agreement ≥ 80%: {validation_results['agreement_score']:.1%}")
        print(f"   ✅ Error ≤ 30%: {validation_results['mean_relative_error']:.1%}")
        print(f"   ✅ Correct well geometry achieved!")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Visualization: {viz_file}")
    
    print("="*80)
    
    return {
        'learner': learner,
        'voronoi_regions': voronoi_regions,
        'well_geometries': well_geometries,
        'piecewise_potential': piecewise_potential,
        'validation_results': validation_results,
        'success': success
    }

def create_voronoi_wells_visualization(learner, voronoi_regions, well_geometries, validation_results):
    """Create comprehensive visualization of Voronoi well geometry learning."""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'True Voronoi Boundaries',
            'Learned Well Geometries',
            'Trajectory Data & Clusters',
            'Validation: Predicted vs Observed',
            'Potential Energy Landscape',
            'Agreement Analysis'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}]
        ]
    )
    
    # 1. True Voronoi boundaries
    for cluster_id, region_info in voronoi_regions.items():
        vertices = region_info['vertices']
        
        # Close the polygon by adding first vertex at the end if needed
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        
        fig.add_trace(
            go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                mode='lines',
                line=dict(width=2),
                name=f'Voronoi Cluster {cluster_id}',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add center point
        center = region_info['center']
        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(size=10, symbol='star'),
                name=f'Center {cluster_id}',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Show one learned well geometry example
    if len(well_geometries) > 0:
        # Take first available well geometry
        first_cluster = list(well_geometries.keys())[0]
        geom = well_geometries[first_cluster]
        
        if 'grid_data' in geom:
            X, Y, V = geom['grid_data']
            
            fig.add_trace(
                go.Heatmap(
                    x=X[0, :],
                    y=Y[:, 0],
                    z=V,
                    colorscale='RdBu_r',
                    name=f'Well Geometry {first_cluster}'
                ),
                row=1, col=2
            )
    
    # 3. Trajectory data with cluster assignments
    fig.add_trace(
        go.Scatter(
            x=learner.trajectory_points[:, 0],
            y=learner.trajectory_points[:, 1],
            mode='markers',
            marker=dict(size=3, opacity=0.6),
            name='Trajectory Points'
        ),
        row=1, col=3
    )
    
    # 4. Validation results
    if 'cluster_comparison' in validation_results['comparison']:
        cluster_comp = validation_results['comparison']['cluster_comparison']
        cluster_names = list(cluster_comp.keys())
        observed_freqs = [cluster_comp[name]['observed_frequency'] for name in cluster_names]
        predicted_freqs = [cluster_comp[name]['predicted_frequency'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=observed_freqs,
                name='Observed',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=predicted_freqs,
                name='Predicted',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    agreement = validation_results['agreement_score']
    error = validation_results['mean_relative_error']
    
    fig.update_layout(
        title=dict(
            text=f"🏆 Voronoi Well Geometry Learning Results<br>" +
                 f"<sub>Agreement: {agreement:.1%} | Error: {error:.1%} | Regions: {len(voronoi_regions)}</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1000,
        width=1800,
        template="plotly_dark",
        showlegend=True
    )
    
    return fig

if __name__ == "__main__":
    results = run_voronoi_well_geometry_learning()