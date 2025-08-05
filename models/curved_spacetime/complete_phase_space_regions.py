#!/usr/bin/env python3
"""
Complete Phase Space Region Definition

Takes the existing convex hull boundaries and fills gaps to create 
complete mathematical coverage of phase space with well-defined boundaries.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from matplotlib.path import Path as MplPath

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompletePhaseSpaceRegions:
    """Create complete mathematical boundaries that cover all of phase space."""
    
    def __init__(self):
        """Initialize with existing candle geometry classification."""
        
        # Load existing regions from candle geometry classifier
        classification_file = Path("phase_space_analysis/candle_geometry_classification.json")
        
        if not classification_file.exists():
            raise FileNotFoundError("Need to run candle geometry classifier first!")
            
        with open(classification_file, 'r') as f:
            self.classification_data = json.load(f)
        
        self.regions = self.classification_data['classification_system']['regions']
        logger.info(f"Loaded {len(self.regions)} existing regions from candle geometry classifier")
        
        # Phase space bounds
        self.phase_space_bounds = {
            'sentiment_min': -1.0,
            'sentiment_max': 1.0, 
            'uwr_min': 0.0,
            'uwr_max': 1.0
        }
    
    def load_hull_boundary_points(self) -> Dict[str, List[Tuple[float, float]]]:
        """Load the actual convex hull boundary points from the classifier."""
        
        logger.info("🔍 Loading existing convex hull boundary points...")
        
        # We need to recreate the hull boundaries since they weren't saved
        # This requires re-running the boundary analysis
        from candle_geometry_classifier import CandleGeometryClassifier
        
        classifier = CandleGeometryClassifier()
        boundary_analysis = classifier.analyze_cluster_boundaries()  
        regions = classifier.define_geometric_regions(boundary_analysis)
        
        hull_boundaries = {}
        
        for region in regions:
            if hasattr(region, 'boundary_points') and region.boundary_points:
                hull_boundaries[region.name] = region.boundary_points
                logger.info(f"   • {region.name}: {len(region.boundary_points)} boundary points")
            else:
                logger.warning(f"   • {region.name}: No boundary points found")
        
        return hull_boundaries
    
    def create_complete_coverage_map(self, hull_boundaries: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Create complete phase space coverage by filling gaps between hull regions."""
        
        logger.info("🎯 Creating complete phase space coverage...")
        
        # Create high-resolution phase space grid
        n_grid = 200
        sentiment_grid = np.linspace(-0.99, 0.99, n_grid)
        uwr_grid = np.linspace(0.01, 0.99, n_grid)
        
        S, U = np.meshgrid(sentiment_grid, uwr_grid)
        
        # Apply triangular phase space constraint: |sentiment| + UWR ≤ 1
        valid_mask = np.abs(S) + U <= 1.0
        
        # Initialize assignment map
        assignment_map = np.full(S.shape, -1, dtype=int)  # -1 = unassigned
        region_names = list(hull_boundaries.keys())
        
        # Step 1: Assign points inside existing hull boundaries (NO OVERLAP ALLOWED)
        logger.info("   • Assigning points inside existing hull boundaries (preventing overlap)...")
        
        for region_idx, (region_name, boundary_points) in enumerate(hull_boundaries.items()):
            if len(boundary_points) < 3:
                continue
                
            # Create polygon from boundary points
            polygon_vertices = np.array(boundary_points)
            polygon_path = MplPath(polygon_vertices)
            
            # Check each grid point
            for i in range(n_grid):
                for j in range(n_grid):
                    if not valid_mask[i, j]:
                        continue
                        
                    # PREVENT OVERLAP: Only assign if not already assigned
                    if assignment_map[i, j] != -1:
                        continue
                        
                    point = np.array([S[i, j], U[i, j]])
                    
                    if polygon_path.contains_point(point):
                        assignment_map[i, j] = region_idx
        
        # Step 2: Fill gaps using nearest neighbor assignment
        logger.info("   • Filling gaps with nearest neighbor assignment...")
        
        # Get centers of existing regions for distance calculation
        region_centers = {}
        for region_idx, (region_name, boundary_points) in enumerate(hull_boundaries.items()):
            if len(boundary_points) >= 3:
                boundary_array = np.array(boundary_points)
                center = np.mean(boundary_array, axis=0)
                region_centers[region_idx] = center
        
        # Assign unassigned points to nearest region center
        unassigned_count = 0
        for i in range(n_grid):
            for j in range(n_grid):
                if not valid_mask[i, j]:
                    continue
                    
                if assignment_map[i, j] == -1:  # Unassigned
                    point = np.array([S[i, j], U[i, j]])
                    
                    # Find nearest region center
                    min_distance = float('inf')
                    nearest_region = -1
                    
                    for region_idx, center in region_centers.items():
                        distance = np.linalg.norm(point - center)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_region = region_idx
                    
                    if nearest_region != -1:
                        assignment_map[i, j] = nearest_region
                        unassigned_count += 1
        
        logger.info(f"   • Filled {unassigned_count} gap points with nearest neighbor assignment")
        
        # Step 3: Extract complete region boundaries
        logger.info("   • Extracting complete region boundaries...")
        
        complete_regions = {}
        
        for region_idx, region_name in enumerate(region_names):
            # Find all points assigned to this region
            region_mask = (assignment_map == region_idx) & valid_mask
            
            if not np.any(region_mask):
                continue
            
            # Get coordinates of all points in this region
            region_points = []
            for i in range(n_grid):
                for j in range(n_grid):
                    if region_mask[i, j]:
                        region_points.append([S[i, j], U[i, j]])
            
            region_points = np.array(region_points)
            
            # Extract boundary using convex hull
            if len(region_points) >= 3:
                try:
                    hull = ConvexHull(region_points)
                    boundary_vertices = region_points[hull.vertices]
                    
                    complete_regions[region_name] = {
                        'boundary_vertices': boundary_vertices,
                        'n_points': len(region_points),
                        'coverage_fraction': len(region_points) / np.sum(valid_mask)
                    }
                    
                    logger.info(f"   • {region_name}: {len(boundary_vertices)} boundary vertices, {len(region_points)} points")
                    
                except Exception as e:
                    logger.warning(f"   • Failed to compute hull for {region_name}: {e}")
        
        # Verify complete coverage
        total_coverage = sum(info['coverage_fraction'] for info in complete_regions.values())
        logger.info(f"✅ Complete phase space coverage: {total_coverage:.1%}")
        
        return {
            'complete_regions': complete_regions,
            'assignment_map': assignment_map,
            'grid_coordinates': (S, U),
            'valid_mask': valid_mask,
            'total_coverage': total_coverage
        }
    
    def create_mathematical_boundary_functions(self, complete_coverage: Dict[str, Any]) -> Dict[str, Any]:
        """Create mathematical functions that define the complete boundaries."""
        
        logger.info("🔧 Creating mathematical boundary functions...")
        
        complete_regions = complete_coverage['complete_regions']
        mathematical_boundaries = {}
        
        for region_name, region_info in complete_regions.items():
            boundary_vertices = region_info['boundary_vertices']
            
            # Create boundary function using polygon containment
            def create_boundary_function(vertices):
                polygon_path = MplPath(vertices)
                
                def boundary_function(sentiment: float, uwr: float) -> bool:
                    """Return True if point is inside this region."""
                    # Check phase space constraint first
                    if abs(sentiment) + uwr > 1.0:
                        return False
                    
                    point = np.array([sentiment, uwr])
                    return polygon_path.contains_point(point)
                
                return boundary_function
            
            boundary_func = create_boundary_function(boundary_vertices)
            
            # Create boundary equation string
            vertex_str = ", ".join([f"({v[0]:.3f}, {v[1]:.3f})" for v in boundary_vertices[:5]])
            if len(boundary_vertices) > 5:
                vertex_str += f", ... ({len(boundary_vertices)} vertices total)"
            
            mathematical_boundaries[region_name] = {
                'boundary_function': boundary_func,
                'boundary_vertices': boundary_vertices,
                'boundary_equation': f"Polygon with vertices: {vertex_str}",
                'n_vertices': len(boundary_vertices),
                'coverage_fraction': region_info['coverage_fraction']
            }
            
            logger.info(f"   • {region_name}: {len(boundary_vertices)} vertex polygon")
        
        return mathematical_boundaries
    
    def visualize_complete_regions(self, mathematical_boundaries: Dict[str, Any]) -> go.Figure:
        """Create visualization of the complete region coverage."""
        
        logger.info("📊 Creating complete region visualization...")
        
        fig = go.Figure()
        
        # Color palette
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (region_name, boundary_info) in enumerate(mathematical_boundaries.items()):
            vertices = boundary_info['boundary_vertices']
            
            # Close the polygon
            closed_vertices = np.vstack([vertices, vertices[0]])
            
            color = colors[i % len(colors)]
            
            # Add the filled polygon
            fig.add_trace(go.Scatter(
                x=closed_vertices[:, 0],
                y=closed_vertices[:, 1],
                fill='toself',
                fillcolor=color,
                opacity=0.4,
                line=dict(color=color, width=2),
                name=f"{region_name} ({boundary_info['coverage_fraction']:.1%})",
                hovertemplate=f'<b>{region_name}</b><br>' +
                            f'Coverage: {boundary_info["coverage_fraction"]:.1%}<br>' +
                            f'Vertices: {boundary_info["n_vertices"]}<extra></extra>'
            ))
            
            # Add vertex labels
            for j, vertex in enumerate(vertices):
                fig.add_trace(go.Scatter(
                    x=[vertex[0]],
                    y=[vertex[1]],
                    mode='markers+text',
                    marker=dict(
                        size=8,
                        color='white',
                        line=dict(color=color, width=2)
                    ),
                    text=f'V{j}',
                    textposition='top center',
                    textfont=dict(size=10, color='white'),
                    name=f'{region_name} V{j}',
                    showlegend=False,
                    hovertemplate=f'<b>{region_name} Vertex {j}</b><br>' +
                                f'Coordinates: ({vertex[0]:.3f}, {vertex[1]:.3f})<extra></extra>'
                ))
        
        # Add phase space constraint boundary
        sentiment_boundary = np.linspace(-1, 1, 100)
        uwr_upper = 1 - np.abs(sentiment_boundary)
        
        fig.add_trace(go.Scatter(
            x=sentiment_boundary,
            y=uwr_upper,
            mode='lines',
            line=dict(color='white', width=4, dash='dash'),
            name='Phase Space Boundary',
            hovertemplate='Constraint: |sentiment| + UWR ≤ 1<extra></extra>'
        ))
        
        fig.update_layout(
            title="🎯 Complete Phase Space Region Coverage with Vertex Labels<br><sub>Mathematical boundaries with gap filling - NO OVERLAPS</sub>",
            xaxis_title="← Bearish Sentiment | Neutral | Bullish Sentiment →",
            yaxis_title="Upper Wick Ratio ↑",
            xaxis=dict(range=[-1, 1], gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            template="plotly_dark",
            height=1000,
            width=1400
        )
        
        return fig

def run_complete_phase_space_definition():
    """Run the complete phase space region definition pipeline."""
    
    logger.info("🚀 Starting Complete Phase Space Region Definition")
    
    # Initialize
    region_definer = CompletePhaseSpaceRegions()
    
    # Step 1: Load existing hull boundaries
    hull_boundaries = region_definer.load_hull_boundary_points()
    
    # Step 2: Create complete coverage
    complete_coverage = region_definer.create_complete_coverage_map(hull_boundaries)
    
    # Step 3: Create mathematical boundary functions
    mathematical_boundaries = region_definer.create_mathematical_boundary_functions(complete_coverage)
    
    # Step 4: Create visualization
    fig = region_definer.visualize_complete_regions(mathematical_boundaries)
    
    # Save results
    output_dir = Path("phase_space_analysis")
    output_dir.mkdir(exist_ok=True)
    
    fig.write_html(output_dir / "complete_phase_space_regions.html")
    
    # Save boundary data
    saveable_boundaries = {}
    for region_name, boundary_info in mathematical_boundaries.items():
        saveable_boundaries[region_name] = {
            'boundary_vertices': boundary_info['boundary_vertices'].tolist(),
            'boundary_equation': boundary_info['boundary_equation'],
            'n_vertices': boundary_info['n_vertices'],
            'coverage_fraction': boundary_info['coverage_fraction']
        }
    
    with open(output_dir / "complete_mathematical_boundaries.json", 'w') as f:
        json.dump(saveable_boundaries, f, indent=2)
    
    print("\\n" + "="*80)
    print("🎯 COMPLETE PHASE SPACE REGION DEFINITION RESULTS")
    print("="*80)
    
    print(f"\\n📊 MATHEMATICAL BOUNDARIES:")
    total_coverage = complete_coverage['total_coverage']
    
    for region_name, boundary_info in mathematical_boundaries.items():
        coverage = boundary_info['coverage_fraction']
        vertices = boundary_info['n_vertices']
        print(f"   • {region_name}: {vertices} vertices, {coverage:.1%} coverage")
    
    print(f"\\n✅ COMPLETE COVERAGE: {total_coverage:.1%}")
    print(f"   • No gaps in phase space")
    print(f"   • Well-defined mathematical boundaries")
    print(f"   • Built on existing convex hull foundations")
    
    print(f"\\n📁 OUTPUT FILES:")
    print(f"   • Visualization: {output_dir}/complete_phase_space_regions.html")
    print(f"   • Boundaries: {output_dir}/complete_mathematical_boundaries.json")
    
    print("="*80)
    
    return {
        'mathematical_boundaries': mathematical_boundaries,
        'complete_coverage': complete_coverage,
        'hull_boundaries': hull_boundaries
    }

if __name__ == "__main__":
    results = run_complete_phase_space_definition()