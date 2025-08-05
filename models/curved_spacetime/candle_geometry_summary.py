#!/usr/bin/env python3
"""
Generate a comprehensive summary of the candle geometry classification system.
"""

import json
from pathlib import Path
import pandas as pd

def generate_summary_report():
    """Generate a comprehensive summary of the classification system."""
    
    # Load the classification data
    classification_file = Path("phase_space_analysis/candle_geometry_classification.json")
    with open(classification_file, 'r') as f:
        data = json.load(f)
    
    regions = data['classification_system']['regions']
    stats = data['classification_system']['cross_security_statistics']
    
    print("🎯 NATURAL CANDLE GEOMETRY CLASSIFICATION SYSTEM")
    print("=" * 60)
    print(f"Generated: {data['generated_at']}")
    print(f"Total Geometric Regions: {stats['total_regions']}")
    print(f"Coverage: {stats['region_coverage']:.1f}% of all market states")
    print(f"Most Common Pattern: {stats['most_common_region']}")
    print()
    
    print("📐 GEOMETRIC REGIONS WITH NATURAL CUTOFFS")
    print("=" * 60)
    
    for i, region in enumerate(regions, 1):
        print(f"{i}. {region['name'].upper()}")
        print(f"   🔸 Description: {region['description']}")
        print(f"   🔸 Frequency: {region['frequency_across_securities']:.1f}% across securities")
        print(f"   🔸 Geometric Form: {region['geometric_interpretation']}")
        print(f"   🔸 Market Behavior: {region['market_behavior']}")
        
        # Parse bounds
        s_min, s_max = region['sentiment_bounds']
        u_min, u_max = region['uwr_bounds']
        
        print(f"   🔸 Sentiment Range: [{s_min:.3f}, {s_max:.3f}]")
        print(f"   🔸 Upper Wick Ratio: [{u_min:.3f}, {u_max:.3f}]")
        print(f"   🔸 Common in: {', '.join(region['typical_examples'])}")
        print()
    
    print("🧮 CLASSIFICATION RULES")
    print("=" * 60)
    
    rules = data['classification_system']['classification_rules']
    boundaries = data['classification_system']['boundary_equations']
    
    for region_name in rules:
        print(f"{region_name}:")
        print(f"   Rule: {rules[region_name]}")
        print(f"   Boundary: {boundaries[region_name]}")
        print()
    
    print("📊 CROSS-SECURITY ANALYSIS INSIGHTS")
    print("=" * 60)
    
    # Calculate insights
    frequencies = [r['frequency_across_securities'] for r in regions]
    
    print(f"• Market coverage: {sum(frequencies):.1f}% of all candlestick patterns")
    print(f"• Average region frequency: {stats['average_frequency']:.1f}%")
    print(f"• Frequency standard deviation: {stats['frequency_std']:.1f}%")
    print()
    
    print("🎨 GEOMETRIC INTERPRETATION")
    print("=" * 60)
    
    print("The classification system identifies 5 natural geometric regions:")
    print()
    
    # Organize by geometric characteristics
    strong_patterns = [r for r in regions if 'Strong' in r['geometric_interpretation']]
    volatility_patterns = [r for r in regions if 'Volatility' in r['description']]
    transition_patterns = [r for r in regions if 'Mixed' in r['geometric_interpretation'] or 'Transition' in r['name']]
    
    if strong_patterns:
        print("DECISIVE PATTERNS (Strong Body Formation):")
        for region in strong_patterns:
            s_center = sum(region['sentiment_bounds']) / 2
            direction = "Bullish" if s_center > 0 else "Bearish"
            print(f"  • {region['name']}: {direction} with minimal rejection")
        print()
    
    if volatility_patterns:
        print("VOLATILITY PATTERNS (Large Wick Formation):")
        for region in volatility_patterns:
            print(f"  • {region['name']}: High price rejection and indecision")
        print()
    
    if transition_patterns:
        print("TRANSITION PATTERNS (Mixed Proportions):")
        for region in transition_patterns:
            print(f"  • {region['name']}: Evolving market dynamics")
        print()
    
    print("🔬 USAGE GUIDELINES")
    print("=" * 60)
    
    print("To classify a candle with sentiment S and upper wick ratio U:")
    print()
    for region in regions:
        s_min, s_max = region['sentiment_bounds']
        u_min, u_max = region['uwr_bounds']
        print(f"• {region['name']}: ")
        print(f"  if ({s_min:.3f} ≤ S ≤ {s_max:.3f}) AND ({u_min:.3f} ≤ U ≤ {u_max:.3f})")
    
    print()
    print(f"📁 Detailed visualizations available in:")
    print(f"   • candle_geometry_boundaries.html - Phase space boundary visualization")
    print(f"   • candle_geometry_distributions.html - Cross-security distribution analysis")
    print(f"   • candle_geometry_classification.json - Complete classification data")


if __name__ == "__main__":
    generate_summary_report()