#!/usr/bin/env python3
"""
Test phase space coverage with the new non-rectangular boundary system.
Verify that the improved boundaries provide complete coverage of the triangular phase space.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from candle_geometry_classifier import CandleGeometryClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase_space_coverage():
    """Test the coverage of the new classification system."""
    
    logger.info("🧪 Testing phase space coverage with new boundary system...")
    
    # Load the classification system
    classification_file = Path("phase_space_analysis/candle_geometry_classification.json")
    with open(classification_file, 'r') as f:
        data = json.load(f)
    
    regions_data = data['classification_system']['regions']
    
    # Create test grid covering the entire triangular phase space
    n_points = 1000
    sentiment_range = np.linspace(-1, 1, n_points)
    uwr_range = np.linspace(0, 1, n_points)
    
    # Generate test points within triangular constraint
    test_points = []
    for s in sentiment_range:
        for u in uwr_range:
            if abs(s) + u <= 1.0:  # Triangular constraint
                test_points.append((s, u))
    
    logger.info(f"Generated {len(test_points)} test points within triangular phase space")
    
    # Initialize classifier
    classifier = CandleGeometryClassifier()
    
    # Create region objects for classification
    from candle_geometry_classifier import CandleGeometryRegion
    regions = []
    for region_data in regions_data:
        region = CandleGeometryRegion(
            name=region_data['name'],
            description=region_data['description'],
            sentiment_bounds=tuple(region_data['sentiment_bounds']),
            uwr_bounds=tuple(region_data['uwr_bounds']),
            geometric_interpretation=region_data['geometric_interpretation'],
            market_behavior=region_data['market_behavior'],
            frequency_across_securities=region_data['frequency_across_securities'],
            typical_examples=region_data['typical_examples']
        )
        regions.append(region)
    
    # Test classification coverage
    classification_results = []
    coverage_stats = {}
    
    for s, u in test_points:
        classification = classifier.classify_candle(s, u, regions)
        classification_results.append({
            'sentiment': s,
            'uwr': u,
            'classification': classification
        })
        
        if classification not in coverage_stats:
            coverage_stats[classification] = 0
        coverage_stats[classification] += 1
    
    # Calculate coverage statistics
    total_points = len(test_points)
    unclassified_count = coverage_stats.get('Unclassified', 0)
    coverage_percentage = (total_points - unclassified_count) / total_points * 100
    
    logger.info(f"📊 Coverage Analysis Results:")
    logger.info(f"   • Total test points: {total_points:,}")
    logger.info(f"   • Classified points: {total_points - unclassified_count:,}")
    logger.info(f"   • Unclassified points: {unclassified_count:,}")
    logger.info(f"   • Coverage percentage: {coverage_percentage:.2f}%")
    
    print("\n🎯 REGION COVERAGE BREAKDOWN:")
    print("=" * 50)
    for region_name, count in sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_points * 100
        print(f"{region_name:20}: {count:6,} points ({percentage:5.2f}%)")
    
    # Create coverage visualization
    df = pd.DataFrame(classification_results)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Phase Space Coverage Test', 'Coverage Density Analysis'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Scatter plot of classifications
    colors = px.colors.qualitative.Set3
    color_map = {region: colors[i % len(colors)] for i, region in enumerate(coverage_stats.keys())}
    
    for region_name in coverage_stats.keys():
        region_data = df[df['classification'] == region_name]
        
        fig.add_trace(
            go.Scatter(
                x=region_data['sentiment'],
                y=region_data['uwr'],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_map[region_name],
                    opacity=0.6
                ),
                name=region_name,
                hovertemplate=f'{region_name}<br>S: %{{x:.3f}}<br>U: %{{y:.3f}}<extra></extra>'
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
            line=dict(color='white', width=3, dash='dash'),
            name='Phase Space Boundary',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Coverage density bar chart
    regions_list = list(coverage_stats.keys())
    counts = list(coverage_stats.values())
    percentages = [c / total_points * 100 for c in counts]
    
    fig.add_trace(
        go.Bar(
            x=regions_list,
            y=percentages,
            marker_color=[color_map[region] for region in regions_list],
            name='Coverage',
            showlegend=False,
            hovertemplate='%{x}<br>Coverage: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"🧪 Phase Space Coverage Test Results<br>" +
                 f"<sub>Coverage: {coverage_percentage:.2f}% | Test Points: {total_points:,}</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=600,
        width=1400,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sentiment", range=[-1, 1], row=1, col=1)
    fig.update_yaxes(title_text="Upper Wick Ratio", range=[0, 1], row=1, col=1)
    
    fig.update_xaxes(title_text="Classification Region", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=2)
    
    # Save visualization
    output_file = Path("phase_space_analysis/phase_space_coverage_test.html")
    fig.write_html(str(output_file))
    logger.info(f"📊 Coverage test visualization saved to {output_file}")
    
    # Create detailed coverage report
    report = {
        'test_summary': {
            'total_test_points': total_points,
            'classified_points': total_points - unclassified_count,
            'unclassified_points': unclassified_count,
            'coverage_percentage': coverage_percentage,
            'improvement_vs_rectangular': coverage_percentage - 97.4  # Previous rectangular coverage
        },
        'region_coverage': {
            region: {
                'point_count': count,
                'percentage': count / total_points * 100
            }
            for region, count in coverage_stats.items()
        },
        'boundary_effectiveness': {
            'triangular_constraint_respected': True,
            'complete_phase_space_covered': unclassified_count == 0,
            'voronoi_tessellation_success': coverage_percentage > 99.0
        }
    }
    
    # Save detailed report
    report_file = Path("phase_space_analysis/coverage_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📋 Detailed coverage report saved to {report_file}")
    
    # Summary assessment
    print("\n🏆 COVERAGE ASSESSMENT:")
    print("=" * 50)
    if coverage_percentage >= 99.9:
        print("✅ EXCELLENT: Near-complete phase space coverage achieved!")
    elif coverage_percentage >= 99.0:
        print("✅ VERY GOOD: High phase space coverage with minimal gaps")
    elif coverage_percentage >= 95.0:
        print("⚠️  GOOD: Acceptable coverage but room for improvement")
    else:
        print("❌ POOR: Significant coverage gaps detected")
    
    print(f"📈 Improvement over rectangular boundaries: {coverage_percentage - 97.4:.2f} percentage points")
    
    return report

if __name__ == "__main__":
    test_phase_space_coverage()