#!/usr/bin/env python3
"""
Candle Geometry Classifier - Comprehensive analysis of candle types using GMM cluster boundaries
Creates natural geometric cutoffs for classifying market states across securities.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from gmm_cluster_explorer import GMMClusterExplorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CandleGeometryRegion:
    """Definition of a geometric region in phase space."""
    name: str
    description: str
    sentiment_bounds: Tuple[float, float]  # (min, max)
    uwr_bounds: Tuple[float, float]  # (min, max)
    geometric_interpretation: str
    market_behavior: str
    frequency_across_securities: float  # Average percentage across securities
    typical_examples: List[str]  # Example securities where this is common

@dataclass
class CandleGeometryClassification:
    """Complete classification system for candle geometries."""
    regions: List[CandleGeometryRegion]
    boundary_equations: Dict[str, str]  # Mathematical boundary definitions
    classification_rules: Dict[str, str]  # Rule-based classification logic
    cross_security_statistics: Dict[str, Any]

class CandleGeometryClassifier:
    """Classifier for candle geometries based on GMM cluster analysis."""
    
    def __init__(self, output_dir: str = "./phase_space_analysis"):
        """Initialize the candle geometry classifier."""
        self.output_dir = Path(output_dir)
        self.gmm_explorer = GMMClusterExplorer()
        
        # Load cluster analysis data
        report_file = self.output_dir / "gmm_cluster_analysis_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                self.cluster_data = json.load(f)
        else:
            raise FileNotFoundError(f"GMM cluster analysis report not found: {report_file}")
            
        logger.info(f"Loaded cluster data for {len(self.cluster_data['securities'])} securities")
    
    def analyze_cluster_boundaries(self) -> Dict[str, Any]:
        """Analyze GMM cluster boundaries across all securities to define geometric regions."""
        
        logger.info("Analyzing cluster boundaries across all securities...")
        
        # Collect all cluster data points
        all_clusters = []
        security_clusters = {}
        
        for symbol, data in self.cluster_data['securities'].items():
            security_clusters[symbol] = []
            
            for cluster in data['clusters']:
                cluster_info = {
                    'symbol': symbol,
                    'cluster_id': cluster['cluster_id'],
                    'interpretation': cluster['interpretation'],
                    'percentage': cluster['percentage'],
                    'sentiment_mean': cluster['sentiment']['mean'],
                    'sentiment_std': cluster['sentiment']['std'],
                    'sentiment_range': cluster['sentiment']['range'],
                    'uwr_mean': cluster['uwr']['mean'],
                    'uwr_std': cluster['uwr']['std'],
                    'uwr_range': cluster['uwr']['range'],
                    'center': cluster['center']
                }
                all_clusters.append(cluster_info)
                security_clusters[symbol].append(cluster_info)
        
        # Group clusters by interpretation across securities
        interpretation_groups = {}
        for cluster in all_clusters:
            interp = cluster['interpretation']
            if interp not in interpretation_groups:
                interpretation_groups[interp] = []
            interpretation_groups[interp].append(cluster)
        
        # Analyze boundaries for each interpretation type
        boundary_analysis = {}
        
        for interpretation, clusters in interpretation_groups.items():
            if len(clusters) < 2:  # Need at least 2 clusters to define boundaries
                continue
                
            # Extract coordinates
            sentiment_coords = [c['sentiment_mean'] for c in clusters]
            uwr_coords = [c['uwr_mean'] for c in clusters]
            
            # Calculate statistics
            sentiment_stats = {
                'mean': np.mean(sentiment_coords),
                'std': np.std(sentiment_coords),
                'min': np.min(sentiment_coords),
                'max': np.max(sentiment_coords),
                'q25': np.percentile(sentiment_coords, 25),
                'q75': np.percentile(sentiment_coords, 75)
            }
            
            uwr_stats = {
                'mean': np.mean(uwr_coords),
                'std': np.std(uwr_coords),
                'min': np.min(uwr_coords),
                'max': np.max(uwr_coords),
                'q25': np.percentile(uwr_coords, 25),
                'q75': np.percentile(uwr_coords, 75)
            }
            
            # Calculate convex hull for boundary definition
            if len(clusters) >= 3:
                points = np.column_stack([sentiment_coords, uwr_coords])
                try:
                    hull = ConvexHull(points)
                    hull_vertices = points[hull.vertices]
                except:
                    hull_vertices = points  # Fallback if hull calculation fails
            else:
                hull_vertices = np.column_stack([sentiment_coords, uwr_coords])
            
            boundary_analysis[interpretation] = {
                'n_clusters': len(clusters),
                'sentiment_stats': sentiment_stats,
                'uwr_stats': uwr_stats,
                'hull_vertices': hull_vertices.tolist(),
                'frequency': len(clusters) / len(all_clusters) * 100,
                'securities': list(set(c['symbol'] for c in clusters))
            }
        
        return {
            'all_clusters': all_clusters,
            'security_clusters': security_clusters,
            'interpretation_groups': interpretation_groups,
            'boundary_analysis': boundary_analysis
        }
    
    def define_geometric_regions(self, boundary_analysis: Dict[str, Any]) -> List[CandleGeometryRegion]:
        """Define natural geometric regions using sophisticated non-rectangular boundaries."""
        
        logger.info("Defining geometric regions with non-rectangular boundaries...")
        
        # Extract cluster centers for Voronoi tessellation
        cluster_centers = []
        interpretations = []
        analysis_data = []
        
        for interpretation, analysis in boundary_analysis['boundary_analysis'].items():
            center = [analysis['sentiment_stats']['mean'], analysis['uwr_stats']['mean']]
            cluster_centers.append(center)
            interpretations.append(interpretation)
            analysis_data.append(analysis)
        
        if len(cluster_centers) < 2:
            logger.warning("Not enough clusters for Voronoi tessellation, falling back to convex hulls")
            return self._define_regions_convex_hull(boundary_analysis)
        
        # Create Voronoi tessellation
        points = np.array(cluster_centers)
        
        # Generate dense grid for phase space classification
        sentiment_grid = np.linspace(-1, 1, 200)
        uwr_grid = np.linspace(0, 1, 200)
        S, U = np.meshgrid(sentiment_grid, uwr_grid)
        
        # Apply triangular phase space constraint: |sentiment| + UWR <= 1
        valid_mask = np.abs(S) + U <= 1.0
        
        # Flatten valid points
        valid_points = np.column_stack([S[valid_mask], U[valid_mask]])
        
        # Assign each valid point to nearest cluster center
        distances = cdist(valid_points, points)
        assignments = np.argmin(distances, axis=1)
        
        # Create regions based on Voronoi assignments
        regions = []
        
        for i, (interpretation, analysis) in enumerate(zip(interpretations, analysis_data)):
            # Get all points assigned to this region
            region_points = valid_points[assignments == i]
            
            if len(region_points) == 0:
                continue
            
            # Calculate natural boundaries from assigned points
            sentiment_coords = region_points[:, 0]
            uwr_coords = region_points[:, 1]
            
            # Use more intelligent boundary calculation
            sentiment_bounds = (
                np.percentile(sentiment_coords, 5),  # 5th percentile
                np.percentile(sentiment_coords, 95)  # 95th percentile
            )
            
            uwr_bounds = (
                np.percentile(uwr_coords, 5),
                np.percentile(uwr_coords, 95)
            )
            
            # Ensure bounds are within valid ranges
            sentiment_bounds = (
                max(-1.0, sentiment_bounds[0]),
                min(1.0, sentiment_bounds[1])
            )
            
            uwr_bounds = (
                max(0.0, uwr_bounds[0]),
                min(1.0, uwr_bounds[1])
            )
            
            # Create geometric interpretation and market behavior descriptions
            geometric_interp, market_behavior = self._interpret_geometry(
                analysis['sentiment_stats']['mean'],
                analysis['uwr_stats']['mean']
            )
            
            # Store region boundary points for visualization
            region_boundary_points = self._extract_boundary_points(region_points)
            
            region = CandleGeometryRegion(
                name=self._create_geometric_name(interpretation),
                description=f"{interpretation} - {geometric_interp}",
                sentiment_bounds=sentiment_bounds,
                uwr_bounds=uwr_bounds,
                geometric_interpretation=geometric_interp,
                market_behavior=market_behavior,
                frequency_across_securities=analysis['frequency'],
                typical_examples=analysis['securities'][:3]
            )
            
            # Add boundary points as additional attribute for visualization
            region.boundary_points = region_boundary_points
            
            regions.append(region)
        
        # Sort regions by frequency (most common first)
        regions.sort(key=lambda r: r.frequency_across_securities, reverse=True)
        
        # Verify complete coverage
        total_coverage = sum(r.frequency_across_securities for r in regions)
        logger.info(f"Phase space coverage with Voronoi boundaries: {total_coverage:.1f}%")
        
        return regions
    
    def _extract_boundary_points(self, region_points: np.ndarray) -> List[Tuple[float, float]]:
        """Extract boundary points for visualization of non-rectangular regions."""
        
        if len(region_points) < 3:
            return [(float(p[0]), float(p[1])) for p in region_points]
        
        try:
            # Calculate convex hull of region points for boundary
            hull = ConvexHull(region_points)
            boundary_points = region_points[hull.vertices]
            return [(float(p[0]), float(p[1])) for p in boundary_points]
        except:
            # Fallback: return points sorted by angle from centroid
            centroid = np.mean(region_points, axis=0)
            centered_points = region_points - centroid
            angles = np.arctan2(centered_points[:, 1], centered_points[:, 0])
            sorted_indices = np.argsort(angles)
            
            # Sample boundary points
            n_boundary = min(20, len(region_points))
            step = len(sorted_indices) // n_boundary
            boundary_indices = sorted_indices[::step]
            
            return [(float(region_points[i][0]), float(region_points[i][1])) 
                   for i in boundary_indices]
    
    def _define_regions_convex_hull(self, boundary_analysis: Dict[str, Any]) -> List[CandleGeometryRegion]:
        """Fallback method using convex hulls when Voronoi tessellation is not feasible."""
        
        logger.info("Using convex hull fallback for region definition...")
        
        regions = []
        
        for interpretation, analysis in boundary_analysis['boundary_analysis'].items():
            # Use hull vertices for boundary definition
            hull_vertices = np.array(analysis['hull_vertices'])
            
            if len(hull_vertices) > 0:
                sentiment_coords = hull_vertices[:, 0]
                uwr_coords = hull_vertices[:, 1]
                
                sentiment_bounds = (np.min(sentiment_coords), np.max(sentiment_coords))
                uwr_bounds = (np.min(uwr_coords), np.max(uwr_coords))
            else:
                # Fallback to statistical bounds
                sentiment_bounds = (
                    analysis['sentiment_stats']['min'],
                    analysis['sentiment_stats']['max']
                )
                uwr_bounds = (
                    analysis['uwr_stats']['min'],
                    analysis['uwr_stats']['max']
                )
            
            # Create geometric interpretation and market behavior descriptions
            geometric_interp, market_behavior = self._interpret_geometry(
                analysis['sentiment_stats']['mean'],
                analysis['uwr_stats']['mean']
            )
            
            region = CandleGeometryRegion(
                name=self._create_geometric_name(interpretation),
                description=f"{interpretation} - {geometric_interp}",
                sentiment_bounds=sentiment_bounds,
                uwr_bounds=uwr_bounds,
                geometric_interpretation=geometric_interp,
                market_behavior=market_behavior,
                frequency_across_securities=analysis['frequency'],
                typical_examples=analysis['securities'][:3]
            )
            
            # Add boundary points for visualization
            region.boundary_points = [(float(v[0]), float(v[1])) for v in hull_vertices]
            
            regions.append(region)
        
        # Sort regions by frequency (most common first)
        regions.sort(key=lambda r: r.frequency_across_securities, reverse=True)
        
        return regions
    
    def _interpret_geometry(self, sentiment_mean: float, uwr_mean: float) -> Tuple[str, str]:
        """Interpret the geometric meaning of sentiment/UWR coordinates."""
        
        if sentiment_mean > 0.4 and uwr_mean < 0.2:
            geometric = "Strong Positive Body, Minimal Upper Wick"
            behavior = "Sustained buying pressure with little resistance"
        elif sentiment_mean > 0.2 and uwr_mean < 0.3:
            geometric = "Positive Body, Low Upper Wick"
            behavior = "Moderate bullish momentum with controlled volatility"
        elif sentiment_mean < -0.4 and uwr_mean < 0.2:
            geometric = "Strong Negative Body, Minimal Upper Wick"
            behavior = "Heavy selling pressure with decisive price action"
        elif sentiment_mean < -0.2 and uwr_mean < 0.3:
            geometric = "Negative Body, Low Upper Wick"
            behavior = "Moderate bearish pressure with limited rejection"
        elif abs(sentiment_mean) < 0.1 and uwr_mean > 0.5:
            geometric = "Small Body, Large Upper Wick"
            behavior = "High volatility with price rejection at highs"
        elif abs(sentiment_mean) < 0.2 and uwr_mean > 0.4:
            geometric = "Small Body, Moderate Upper Wick"
            behavior = "Indecision with some upside rejection"
        elif abs(sentiment_mean) < 0.2 and uwr_mean < 0.3:
            geometric = "Small Body, Small Upper Wick"
            behavior = "Low volatility consolidation pattern"
        else:
            geometric = "Mixed Body/Wick Proportions"
            behavior = "Transitional market state with varied dynamics"
            
        return geometric, behavior
    
    def _create_geometric_name(self, interpretation: str) -> str:
        """Create natural geometric names for candle types."""
        
        name_mapping = {
            'Strong Bullish Trend': 'Momentum Climber',
            'Moderate Bullish': 'Steady Riser', 
            'Strong Bearish Pressure': 'Pressure Diver',
            'Moderate Bearish': 'Gentle Decliner',
            'High Volatility': 'Rejection Hammer',
            'Extreme Volatility': 'Chaos Spinner',
            'Consolidation': 'Range Binder',
            'Mixed Regime': 'Transition Doji'
        }
        
        return name_mapping.get(interpretation, interpretation.replace(' ', '_').title())
    
    def create_geometric_boundary_visualization(self, regions: List[CandleGeometryRegion]) -> go.Figure:
        """Create visualization showing non-rectangular geometric boundaries in phase space."""
        
        fig = go.Figure()
        
        # Define colors for each region
        colors = px.colors.qualitative.Set3
        
        # Plot each region using boundary points if available
        for i, region in enumerate(regions):
            color = colors[i % len(colors)]
            
            # Use boundary points if available (non-rectangular)
            if hasattr(region, 'boundary_points') and region.boundary_points:
                boundary_points = region.boundary_points
                
                # Close the boundary by adding first point at the end
                if boundary_points[0] != boundary_points[-1]:
                    boundary_points = boundary_points + [boundary_points[0]]
                
                sentiment_coords = [p[0] for p in boundary_points]
                uwr_coords = [p[1] for p in boundary_points]
                
                # Ensure all points are within phase space constraints
                valid_points = []
                for s, u in zip(sentiment_coords, uwr_coords):
                    s = max(-1.0, min(1.0, s))
                    u = max(0.0, min(1.0, u))
                    # Apply triangular constraint
                    if abs(s) + u > 1.0:
                        u = max(0.0, 1.0 - abs(s))
                    valid_points.append((s, u))
                
                sentiment_coords = [p[0] for p in valid_points]
                uwr_coords = [p[1] for p in valid_points]
                
            else:
                # Fallback to rectangular boundaries
                sentiment_min, sentiment_max = region.sentiment_bounds
                uwr_min, uwr_max = region.uwr_bounds
                
                # Ensure bounds are within phase space constraints
                sentiment_min = max(-1.0, sentiment_min)
                sentiment_max = min(1.0, sentiment_max)
                uwr_min = max(0.0, uwr_min)
                uwr_max = min(1.0, uwr_max)
                
                # Apply triangular constraint
                if abs(sentiment_min) + uwr_max > 1.0:
                    uwr_max = min(uwr_max, 1.0 - abs(sentiment_min))
                if abs(sentiment_max) + uwr_max > 1.0:
                    uwr_max = min(uwr_max, 1.0 - abs(sentiment_max))
                
                sentiment_coords = [sentiment_min, sentiment_max, sentiment_max, sentiment_min, sentiment_min]
                uwr_coords = [uwr_min, uwr_min, uwr_max, uwr_max, uwr_min]
            
            fig.add_trace(go.Scatter(
                x=sentiment_coords,
                y=uwr_coords,
                fill='toself',
                fillcolor=color,
                opacity=0.4,
                line=dict(color=color, width=3),
                name=region.name,
                hovertemplate=f'<b>{region.name}</b><br>' +
                            f'{region.description}<br>' +
                            f'Frequency: {region.frequency_across_securities:.1f}%<br>' +
                            f'Market Behavior: {region.market_behavior}<extra></extra>'
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
        
        # Add cluster centers as points
        for i, region in enumerate(regions):
            sentiment_center = sum(region.sentiment_bounds) / 2
            uwr_center = sum(region.uwr_bounds) / 2
            
            fig.add_trace(go.Scatter(
                x=[sentiment_center],
                y=[uwr_center],
                mode='markers',
                marker=dict(
                    size=12,
                    color='white',
                    line=dict(color=colors[i % len(colors)], width=3),
                    symbol='circle'
                ),
                name=f'{region.name} Center',
                showlegend=False,
                hovertemplate=f'<b>{region.name} Center</b><br>' +
                            f'Sentiment: {sentiment_center:.3f}<br>' +
                            f'UWR: {uwr_center:.3f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎯 Non-Rectangular Candle Geometry Classification Regions<br>" +
                     "<sub>Sophisticated boundaries using Voronoi tessellation and triangular phase space constraint</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="← Bearish Sentiment | Neutral | Bullish Sentiment →",
            yaxis_title="Upper Wick Ratio ↑",
            xaxis=dict(range=[-1, 1], gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            template="plotly_dark",
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def create_cross_security_distribution_analysis(self, boundary_analysis: Dict[str, Any]) -> go.Figure:
        """Create comprehensive analysis of geometric distributions across securities."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Geometric Region Frequency Distribution',
                'Sentiment vs UWR Distribution by Region',
                'Region Prevalence by Security Type',
                'Geometric Diversity Index'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ]
        )
        
        # 1. Frequency distribution
        interpretations = list(boundary_analysis['boundary_analysis'].keys())
        frequencies = [boundary_analysis['boundary_analysis'][interp]['frequency'] 
                      for interp in interpretations]
        
        fig.add_trace(
            go.Bar(
                x=interpretations,
                y=frequencies,
                name='Frequency',
                marker_color='skyblue',
                hovertemplate='%{x}<br>Frequency: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Sentiment vs UWR scatter
        colors = px.colors.qualitative.Set1
        for i, (interp, analysis) in enumerate(boundary_analysis['boundary_analysis'].items()):
            fig.add_trace(
                go.Scatter(
                    x=[analysis['sentiment_stats']['mean']],
                    y=[analysis['uwr_stats']['mean']],
                    mode='markers',
                    marker=dict(
                        size=analysis['frequency'],
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    name=interp,
                    hovertemplate=f'{interp}<br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<br>Frequency: {analysis["frequency"]:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Security type analysis (simplified for now)
        security_symbols = list(boundary_analysis['security_clusters'].keys())
        region_counts = {}
        
        for symbol in security_symbols:
            clusters = boundary_analysis['security_clusters'][symbol]
            for cluster in clusters:
                interp = cluster['interpretation']
                if interp not in region_counts:
                    region_counts[interp] = {}
                if symbol not in region_counts[interp]:
                    region_counts[interp][symbol] = 0
                region_counts[interp][symbol] += 1
        
        # Create heatmap data
        heatmap_data = []
        for interp in interpretations:
            row = []
            for symbol in security_symbols:
                count = region_counts.get(interp, {}).get(symbol, 0)
                row.append(count)
            heatmap_data.append(row)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=security_symbols,
                y=interpretations,
                colorscale='Blues',
                showscale=False,
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Diversity index (number of unique regions per security)
        diversity_scores = []
        for symbol in security_symbols:
            clusters = boundary_analysis['security_clusters'][symbol]
            unique_interpretations = set(c['interpretation'] for c in clusters)
            diversity_scores.append(len(unique_interpretations))
        
        fig.add_trace(
            go.Bar(
                x=security_symbols,
                y=diversity_scores,
                name='Diversity',
                marker_color='lightcoral',
                hovertemplate='%{x}<br>Unique Regions: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="📈 Cross-Security Geometric Distribution Analysis<br>" +
                     "<sub>Comprehensive analysis of candle geometry patterns across markets</sub>",
                x=0.5,
                xanchor='center'
            ),
            height=800,
            width=1400,
            template="plotly_dark",
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Geometric Region", row=1, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Frequency (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Sentiment", range=[-1, 1], row=1, col=2)
        fig.update_yaxes(title_text="UWR", range=[0, 1], row=1, col=2)
        
        fig.update_xaxes(title_text="Security", row=2, col=1)
        fig.update_yaxes(title_text="Geometric Region", row=2, col=1)
        
        fig.update_xaxes(title_text="Security", row=2, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Geometric Diversity", row=2, col=2)
        
        return fig
    
    def create_candle_type_classifier(self, regions: List[CandleGeometryRegion]) -> Dict[str, Any]:
        """Create a rule-based classifier for candle types based on geometric cutoffs."""
        
        logger.info("Creating candle type classifier based on geometric regions...")
        
        classification_rules = {}
        boundary_equations = {}
        
        for region in regions:
            # Create classification rule
            sentiment_min, sentiment_max = region.sentiment_bounds
            uwr_min, uwr_max = region.uwr_bounds
            
            rule = f"({sentiment_min:.3f} <= sentiment <= {sentiment_max:.3f}) AND ({uwr_min:.3f} <= uwr <= {uwr_max:.3f})"
            classification_rules[region.name] = rule
            
            # Create boundary equation
            boundary_eq = f"S ∈ [{sentiment_min:.3f}, {sentiment_max:.3f}], U ∈ [{uwr_min:.3f}, {uwr_max:.3f}]"
            boundary_equations[region.name] = boundary_eq
        
        # Calculate cross-security statistics
        cross_security_stats = {
            'total_regions': len(regions),
            'most_common_region': max(regions, key=lambda r: r.frequency_across_securities).name,
            'average_frequency': np.mean([r.frequency_across_securities for r in regions]),
            'frequency_std': np.std([r.frequency_across_securities for r in regions]),
            'region_coverage': sum(r.frequency_across_securities for r in regions)
        }
        
        classification_system = CandleGeometryClassification(
            regions=regions,
            boundary_equations=boundary_equations,
            classification_rules=classification_rules,
            cross_security_statistics=cross_security_stats
        )
        
        return classification_system
    
    def classify_candle(self, sentiment: float, uwr: float, regions: List[CandleGeometryRegion]) -> Optional[str]:
        """Classify a single candle using nearest-neighbor assignment for complete phase space coverage."""
        
        # Check if point is within phase space constraint
        if abs(sentiment) + uwr > 1.0:
            return "Invalid_Phase_Space"
        
        # First try rectangular bounds for efficiency
        for region in regions:
            sentiment_min, sentiment_max = region.sentiment_bounds
            uwr_min, uwr_max = region.uwr_bounds
            
            if (sentiment_min <= sentiment <= sentiment_max and 
                uwr_min <= uwr <= uwr_max):
                return region.name
        
        # If no rectangular match, use nearest neighbor to cluster centers
        # This ensures complete phase space coverage
        min_distance = float('inf')
        closest_region = None
        
        for region in regions:
            # Calculate distance to region center
            sentiment_center = sum(region.sentiment_bounds) / 2
            uwr_center = sum(region.uwr_bounds) / 2
            
            distance = np.sqrt((sentiment - sentiment_center)**2 + (uwr - uwr_center)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_region = region.name
        
        return closest_region if closest_region else "Unclassified"
    
    def export_classification_system(self, classification_system: CandleGeometryClassification, 
                                   filename: str = "candle_geometry_classification.json") -> None:
        """Export the complete classification system to JSON."""
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'classification_system': {
                'regions': [
                    {
                        'name': region.name,
                        'description': region.description,
                        'sentiment_bounds': region.sentiment_bounds,
                        'uwr_bounds': region.uwr_bounds,
                        'geometric_interpretation': region.geometric_interpretation,
                        'market_behavior': region.market_behavior,
                        'frequency_across_securities': region.frequency_across_securities,
                        'typical_examples': region.typical_examples
                    }
                    for region in classification_system.regions
                ],
                'boundary_equations': classification_system.boundary_equations,
                'classification_rules': classification_system.classification_rules,
                'cross_security_statistics': classification_system.cross_security_statistics
            }
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported classification system to {output_path}")


def main():
    """Main function to run the complete candle geometry classification analysis."""
    
    classifier = CandleGeometryClassifier()
    
    # Analyze cluster boundaries
    logger.info("🔍 Analyzing cluster boundaries...")
    boundary_analysis = classifier.analyze_cluster_boundaries()
    
    # Define geometric regions
    logger.info("📐 Defining geometric regions...")
    regions = classifier.define_geometric_regions(boundary_analysis)
    
    logger.info(f"✅ Identified {len(regions)} geometric regions:")
    for region in regions:
        logger.info(f"   • {region.name}: {region.frequency_across_securities:.1f}% frequency")
    
    # Create visualizations
    logger.info("📊 Creating boundary visualization...")
    boundary_fig = classifier.create_geometric_boundary_visualization(regions)
    boundary_file = classifier.output_dir / "candle_geometry_boundaries.html"
    boundary_fig.write_html(str(boundary_file))
    logger.info(f"   Saved to {boundary_file}")
    
    logger.info("📈 Creating distribution analysis...")
    distribution_fig = classifier.create_cross_security_distribution_analysis(boundary_analysis)
    distribution_file = classifier.output_dir / "candle_geometry_distributions.html"
    distribution_fig.write_html(str(distribution_file))
    logger.info(f"   Saved to {distribution_file}")
    
    # Create classification system
    logger.info("🎯 Creating classification system...")
    classification_system = classifier.create_candle_type_classifier(regions)
    
    # Export results
    classifier.export_classification_system(classification_system)
    
    logger.info("✅ Candle geometry classification analysis complete!")
    logger.info(f"📁 Results saved to: {classifier.output_dir}")


if __name__ == "__main__":
    main()