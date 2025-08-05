#!/usr/bin/env python3
"""
Voronoi Well Geometry Learning Using Real Market Data

Learn the potential landscape directly from real market trajectories across all securities.
No synthetic trajectory generation - use only observed market data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as MplPath

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMarketVoronoiLearner:
    """Learn potential well geometry from real market trajectories across all securities."""
    
    def __init__(self, trajectory_data_path: str = "phase_space_analysis/real_market_trajectories.npz"):
        """Initialize with real market trajectory data."""
        
        # Load real market trajectory data
        data = np.load(trajectory_data_path)
        self.trajectory_points = data['trajectory_points']
        self.symbols = data['symbols']
        self.n_points = len(self.trajectory_points)
        
        logger.info(f"🔍 Loaded real market trajectory data:")
        logger.info(f"   • Securities: {len(self.symbols)}")
        logger.info(f"   • Total trajectory points: {self.n_points}")
        logger.info(f"   • Sentiment range: [{self.trajectory_points[:, 0].min():.3f}, {self.trajectory_points[:, 0].max():.3f}]")
        logger.info(f"   • UWR range: [{self.trajectory_points[:, 1].min():.3f}, {self.trajectory_points[:, 1].max():.3f}]")
        
        # Phase space bounds
        self.sentiment_bounds = (-1.0, 1.0)
        self.uwr_bounds = (0.0, 1.0)
        
        # Temperature for potential calculation
        self.kT = 0.5
    
    def detect_market_cluster_centers(self, n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect cluster centers from real market trajectories using GMM.
        
        Returns:
            (cluster_centers, cluster_assignments)
        """
        logger.info(f"🎯 Detecting {n_clusters} cluster centers from real market data")
        
        # Apply triangular constraint
        valid_mask = np.abs(self.trajectory_points[:, 0]) + self.trajectory_points[:, 1] <= 1.0
        valid_points = self.trajectory_points[valid_mask]
        
        logger.info(f"   • Valid points after constraint: {len(valid_points)}/{len(self.trajectory_points)}")
        
        # Fit GMM to real market data
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=200)
        gmm.fit(valid_points)
        
        cluster_centers = gmm.means_
        cluster_assignments = gmm.predict(valid_points)
        
        logger.info(f"   • GMM converged: {gmm.converged_}")
        logger.info(f"   • Log likelihood: {gmm.score(valid_points):.2f}")
        
        logger.info(f"\\n📊 DETECTED MARKET CLUSTERS:")
        total_valid = len(valid_points)
        for i, center in enumerate(cluster_centers):
            n_points_in_cluster = np.sum(cluster_assignments == i)
            frequency = n_points_in_cluster / total_valid * 100
            logger.info(f"   • Cluster {i}: center=({center[0]:.3f}, {center[1]:.3f}), frequency={frequency:.1f}%, points={n_points_in_cluster}")
        
        # Verify frequencies sum to 100%
        total_frequency = sum(np.sum(cluster_assignments == i) / total_valid * 100 for i in range(n_clusters))
        logger.info(f"   • Total frequency check: {total_frequency:.1f}%")
        
        return cluster_centers, cluster_assignments
    
    def create_market_voronoi_boundaries(self, cluster_centers: np.ndarray) -> Dict[int, Any]:
        """Create Voronoi boundaries based on real market cluster centers."""
        
        logger.info("🔧 Creating Voronoi boundaries from market cluster centers")
        
        # Create Voronoi diagram
        vor = Voronoi(cluster_centers)
        
        # Extract polygon boundaries for each cluster
        voronoi_regions = {}
        
        for i, center in enumerate(cluster_centers):
            # Find which Voronoi region corresponds to this cluster center
            region_index = vor.point_region[i]
            vertex_indices = vor.regions[region_index]
            
            if len(vertex_indices) == 0 or -1 in vertex_indices:
                # Unbounded region - create bounded approximation
                logger.info(f"   • Cluster {i}: unbounded region, creating bounded approximation")
                vertices = self._create_bounded_region(center)
            else:
                # Bounded region - extract vertices
                vertices = vor.vertices[vertex_indices]
            
            # Clip to triangular phase space constraint: |sentiment| + UWR ≤ 1
            clipped_vertices = self._clip_to_phase_space(vertices)
            
            if len(clipped_vertices) < 3:
                logger.warning(f"   • Cluster {i}: too few vertices after clipping, creating fallback region")
                # Create fallback region around center
                clipped_vertices = self._create_bounded_region(center)
                if len(clipped_vertices) < 3:
                    continue
            
            # Store boundary information
            voronoi_regions[i] = {
                'center': center,
                'vertices': clipped_vertices,
                'area': self._polygon_area(clipped_vertices)
            }
            
            logger.info(f"   • Cluster {i}: {len(clipped_vertices)} vertices, area={voronoi_regions[i]['area']:.4f}")
        
        return voronoi_regions
    
    def _create_bounded_region(self, center: np.ndarray) -> np.ndarray:
        """Create bounded region around cluster center for unbounded Voronoi regions."""
        # Use smaller radius and ensure it stays within phase space
        theta = np.linspace(0, 2*np.pi, 12)
        radius = min(0.2, 0.5 - abs(center[0]), 0.5 - center[1])  # Adaptive radius
        radius = max(0.1, radius)  # Minimum radius
        
        circle_points = np.column_stack([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta)
        ])
        
        # Ensure all points are within phase space bounds
        for i, (s, u) in enumerate(circle_points):
            s = max(-0.95, min(0.95, s))
            u = max(0.05, min(0.95, u))
            # Apply constraint |s| + u <= 1
            if abs(s) + u > 1.0:
                if abs(s) > u:
                    s = np.sign(s) * (1.0 - u - 0.01)
                else:
                    u = 1.0 - abs(s) - 0.01
            circle_points[i] = [s, u]
        
        return circle_points
    
    def _clip_to_phase_space(self, vertices: np.ndarray) -> np.ndarray:
        """Clip vertices to triangular phase space constraint."""
        clipped = []
        for vertex in vertices:
            s, u = vertex
            # Ensure within bounds and constraint
            s = max(-1.0, min(1.0, s))
            u = max(0.0, min(1.0, u))
            if abs(s) + u <= 1.0:
                clipped.append([s, u])
        
        return np.array(clipped) if clipped else vertices
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    
    def learn_market_well_geometry(self, voronoi_regions: Dict[int, Any], 
                                 cluster_assignments: np.ndarray) -> Dict[int, Any]:
        """
        Learn well geometry from real market data within each Voronoi region.
        """
        logger.info("🧠 Learning well geometry from real market trajectories")
        
        # Get valid points used for clustering
        valid_mask = np.abs(self.trajectory_points[:, 0]) + self.trajectory_points[:, 1] <= 1.0
        valid_points = self.trajectory_points[valid_mask]
        
        well_geometries = {}
        
        for cluster_id, region_info in voronoi_regions.items():
            logger.info(f"   • Learning geometry for cluster {cluster_id}")
            
            # Get real market points assigned to this cluster
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = valid_points[cluster_mask]
            
            if len(cluster_points) < 5:
                logger.warning(f"     Too few points ({len(cluster_points)}) in cluster {cluster_id}")
                continue
            
            # Get polygon vertices
            polygon_vertices = region_info['vertices']
            
            # Filter points to only those inside the Voronoi polygon
            points_in_region = []
            for point in cluster_points:
                if self._point_in_polygon(point, polygon_vertices):
                    points_in_region.append(point)
            
            points_in_region = np.array(points_in_region)
            
            if len(points_in_region) < 3:
                logger.warning(f"     Too few points inside region for cluster {cluster_id}")
                continue
            
            # Learn density distribution from real market data
            kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
            kde.fit(points_in_region)
            
            # Create potential surface within the region
            minx, miny = np.min(polygon_vertices, axis=0)
            maxx, maxy = np.max(polygon_vertices, axis=0)
            
            # Add padding
            padding = 0.05
            minx = max(-1.0, minx - padding)
            maxx = min(1.0, maxx + padding)
            miny = max(0.0, miny - padding)
            maxy = min(1.0, maxy + padding)
            
            # Create grid
            n_grid = 50
            x_local = np.linspace(minx, maxx, n_grid)
            y_local = np.linspace(miny, maxy, n_grid)
            X_local, Y_local = np.meshgrid(x_local, y_local)
            
            # Evaluate potential on grid
            local_potential = np.full_like(X_local, np.nan)
            
            for i in range(n_grid):
                for j in range(n_grid):
                    test_point = np.array([X_local[i, j], Y_local[i, j]])
                    if self._point_in_polygon(test_point, polygon_vertices):
                        # Evaluate KDE density from real market data
                        point_array = np.array([[X_local[i, j], Y_local[i, j]]])
                        log_density = kde.score_samples(point_array)[0]
                        density = np.exp(log_density)
                        
                        # Convert to potential: V = -kT ln(P)
                        potential = -self.kT * np.log(max(density, 1e-10))
                        local_potential[i, j] = potential
            
            # Create interpolation function
            potential_func = self._create_potential_function(X_local, Y_local, local_potential, polygon_vertices)
            
            # Store well geometry
            well_geometries[cluster_id] = {
                'vertices': polygon_vertices,
                'potential_function': potential_func,
                'grid_data': (X_local, Y_local, local_potential),
                'n_points': len(points_in_region),
                'center': region_info['center'],
                'frequency': len(points_in_region) / len(valid_points)
            }
            
            logger.info(f"     Cluster {cluster_id}: {len(points_in_region)} market points, frequency={well_geometries[cluster_id]['frequency']:.1%}")
        
        return well_geometries
    
    def _point_in_polygon(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """Check if point is inside polygon."""
        if len(vertices) < 3:
            return False
        path = MplPath(vertices)
        return path.contains_point(point)
    
    def _create_potential_function(self, X_grid, Y_grid, V_grid, vertices):
        """Create interpolated potential function for a region."""
        
        # Create interpolator
        x_coords = X_grid[0, :]
        y_coords = Y_grid[:, 0]
        V_clean = np.copy(V_grid)
        V_clean[np.isnan(V_clean)] = 0.0
        
        try:
            interpolator = RegularGridInterpolator(
                (y_coords, x_coords), V_clean, 
                method='linear', bounds_error=False, fill_value=0.0
            )
        except:
            interpolator = None
        
        def region_potential(sentiment, uwr):
            """Evaluate potential using real market-learned geometry."""
            point = np.array([sentiment, uwr])
            
            # Check if point is in this region
            if not self._point_in_polygon(point, vertices):
                return float('inf')
            
            # Use interpolation
            if interpolator is not None:
                try:
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
    
    def create_market_piecewise_potential(self, well_geometries: Dict[int, Any]) -> callable:
        """Create piecewise potential function from real market well geometries."""
        
        logger.info("⚡ Creating piecewise potential function from market data")
        
        def market_piecewise_potential(sentiment: float, uwr: float) -> float:
            """Evaluate potential learned from real market trajectories."""
            point = np.array([sentiment, uwr])
            
            # Check phase space constraint
            if abs(sentiment) + uwr > 1.0:
                return 20.0
            
            # Check each region
            for cluster_id, geometry in well_geometries.items():
                path = MplPath(geometry['vertices'])
                if path.contains_point(point):
                    return geometry['potential_function'](sentiment, uwr)
            
            # Nearest neighbor fallback for complete coverage
            min_distance = float('inf')
            closest_cluster = None
            
            for cluster_id, geometry in well_geometries.items():
                center = geometry['center']
                distance = np.sqrt((sentiment - center[0])**2 + (uwr - center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            if closest_cluster is not None:
                base_potential = well_geometries[closest_cluster]['potential_function'](sentiment, uwr)
                if np.isinf(base_potential):
                    return 2.0 + min_distance * 5.0
                return base_potential
            
            return 5.0
        
        return market_piecewise_potential
    
    def analyze_market_statistics(self, well_geometries: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze the learned market well geometry statistics."""
        
        logger.info("📈 Analyzing learned market well geometry statistics")
        
        # Calculate statistics from real market data
        market_stats = {
            'n_clusters': len(well_geometries),
            'total_market_points': self.n_points,
            'cluster_frequencies': {},
            'cluster_centers': {},
            'phase_space_coverage': 0.0
        }
        
        total_frequency = 0.0
        for cluster_id, geom in well_geometries.items():
            freq = geom['frequency']
            center = geom['center']
            n_points = geom['n_points']
            
            market_stats['cluster_frequencies'][f'Cluster_{cluster_id}'] = freq
            market_stats['cluster_centers'][f'Cluster_{cluster_id}'] = center.tolist()
            total_frequency += freq
            
            logger.info(f"   • Cluster {cluster_id}: {n_points} market points ({freq:.1%})")
        
        market_stats['phase_space_coverage'] = total_frequency
        
        logger.info(f"\\n📊 MARKET STATISTICS SUMMARY:")
        logger.info(f"   • Total market clusters: {market_stats['n_clusters']}")
        logger.info(f"   • Total trajectory points: {market_stats['total_market_points']}")
        logger.info(f"   • Phase space coverage: {total_frequency:.1%}")
        
        return market_stats

def run_real_market_voronoi_learning():
    """Run complete Voronoi well geometry learning on real market data."""
    
    logger.info("🚀 Starting Real Market Voronoi Well Geometry Learning")
    
    # Initialize learner with real market data
    learner = RealMarketVoronoiLearner()
    
    # Step 1: Detect cluster centers from real market trajectories
    cluster_centers, cluster_assignments = learner.detect_market_cluster_centers(n_clusters=5)
    
    # Step 2: Create Voronoi boundaries
    voronoi_regions = learner.create_market_voronoi_boundaries(cluster_centers)
    
    # Step 3: Learn well geometry from real market data
    well_geometries = learner.learn_market_well_geometry(voronoi_regions, cluster_assignments)
    
    # Step 4: Create piecewise potential function
    market_piecewise_potential = learner.create_market_piecewise_potential(well_geometries)
    
    # Step 5: Analyze market statistics
    market_stats = learner.analyze_market_statistics(well_geometries)
    
    # Save results
    output_dir = Path("phase_space_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save learned potential data (without functions which can't be pickled)
    saveable_data = {
        'cluster_centers': cluster_centers,
        'market_stats': market_stats,
        'well_centers': {k: v['center'] for k, v in well_geometries.items()},
        'well_frequencies': {k: v['frequency'] for k, v in well_geometries.items()},
        'well_n_points': {k: v['n_points'] for k, v in well_geometries.items()}
    }
    
    np.savez(output_dir / "real_market_potential.npz", **saveable_data)
    
    print("\\n" + "="*80)
    print("🏆 REAL MARKET VORONOI WELL GEOMETRY LEARNING RESULTS")
    print("="*80)
    
    print(f"\\n📊 LEARNED MARKET WELL STRUCTURE:")
    print(f"   • Number of Voronoi regions: {len(voronoi_regions)}")
    print(f"   • Number of learned wells: {len(well_geometries)}")
    print(f"   • Total market trajectory points: {learner.n_points}")
    
    for cluster_id, geom in well_geometries.items():
        center = geom['center']
        freq = geom['frequency']
        n_points = geom['n_points']
        print(f"   • Cluster {cluster_id}: center=({center[0]:.3f}, {center[1]:.3f}), frequency={freq:.1%}, points={n_points}")
    
    print(f"\\n✅ SUCCESS: Learned potential landscape from real market data")
    print(f"   • Phase space coverage: {market_stats['phase_space_coverage']:.1%}")
    print(f"   • Market securities: {len(learner.symbols)}")
    
    print(f"\\n📁 OUTPUT FILES:")
    print(f"   • Real market potential: {output_dir}/real_market_potential.npz")
    
    print("="*80)
    
    return {
        'learner': learner,
        'voronoi_regions': voronoi_regions,
        'well_geometries': well_geometries,
        'market_piecewise_potential': market_piecewise_potential,
        'market_stats': market_stats
    }

if __name__ == "__main__":
    results = run_real_market_voronoi_learning()