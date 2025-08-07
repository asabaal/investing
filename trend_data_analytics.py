#!/usr/bin/env python3
"""
Comprehensive Trend Data Analytics
Analyzes all trend formations with detailed metrics:
- Number of candles in trend
- Aggregated candles between swings 
- Range covered by trend
- Percent change between extremes
- Number of swings in trend
- Duration statistics
- Price movement efficiency

Outputs both CSV files and interactive visualizations
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from proper_trend_explorer import find_all_proper_trends, find_all_proper_terminations, filter_overlapping_sideways_trends_with_terminations
from swing_point_detector import SwingPointDetector
from uso_supply_demand_visualizer import SupplyDemandVisualizer
import numpy as np
from datetime import datetime
import json

# Dark theme
pio.templates.default = "plotly_dark"

def calculate_trend_metrics(df, formations, all_swing_points):
    """
    Calculate comprehensive metrics for all trend formations
    """
    
    print(f"📊 Calculating comprehensive metrics for {len(formations)} trends...")
    
    trend_metrics = []
    
    for i, formation in enumerate(formations):
        trend_id = i + 1
        trend_type = formation['type']
        
        # Basic info
        formation_date = formation['formation_date']
        
        if trend_type in ['UPTREND', 'DOWNTREND']:
            # Directional trends
            start_idx = formation['breakout']['idx'] if 'breakout' in formation else 0
            
            # Find termination or use end of data
            end_idx = len(df) - 1  # Default to end
            for term_key in ['violation_idx', 'termination_idx', 'end_idx']:
                if term_key in formation:
                    end_idx = formation[term_key]
                    break
            
            # Get swing points during trend
            trend_swings = get_swings_in_range(all_swing_points, start_idx, end_idx)
            
            # Calculate metrics
            metrics = {
                'trend_id': trend_id,
                'trend_type': trend_type,
                'formation_date': formation_date.strftime('%Y-%m-%d') if hasattr(formation_date, 'strftime') else str(formation_date),
                'start_candle': start_idx,
                'end_candle': end_idx,
                'duration_candles': end_idx - start_idx + 1,
                'duration_days': (df.iloc[end_idx]['datetime'] - df.iloc[start_idx]['datetime']).days if 'datetime' in df.columns else 0,
                
                # Price metrics
                'start_price': df.iloc[start_idx]['close'],
                'end_price': df.iloc[end_idx]['close'],
                'highest_price': df.iloc[start_idx:end_idx+1]['high'].max(),
                'lowest_price': df.iloc[start_idx:end_idx+1]['low'].min(),
                'price_range': df.iloc[start_idx:end_idx+1]['high'].max() - df.iloc[start_idx:end_idx+1]['low'].min(),
                'total_percent_change': ((df.iloc[end_idx]['close'] - df.iloc[start_idx]['close']) / df.iloc[start_idx]['close']) * 100,
                'max_favorable_move': calculate_max_favorable_move(df, start_idx, end_idx, trend_type),
                'max_adverse_move': calculate_max_adverse_move(df, start_idx, end_idx, trend_type),
                
                # Swing metrics
                'num_swings': len(trend_swings),
                'num_swing_highs': len([s for s in trend_swings if s['type'] == 'HIGH']),
                'num_swing_lows': len([s for s in trend_swings if s['type'] == 'LOW']),
                'avg_candles_between_swings': calculate_avg_candles_between_swings(trend_swings),
                'swing_density': len(trend_swings) / (end_idx - start_idx + 1) if (end_idx - start_idx + 1) > 0 else 0,
                
                # Efficiency metrics
                'price_efficiency': calculate_price_efficiency(df, start_idx, end_idx),
                'trend_strength': abs(((df.iloc[end_idx]['close'] - df.iloc[start_idx]['close']) / df.iloc[start_idx]['close']) * 100) / (end_idx - start_idx + 1),
                
                # Formation-specific metrics
                'sl1_price': formation.get('sl1', {}).get('price', 0),
                'sh1_price': formation.get('sh1', {}).get('price', 0),
                'sl2_price': formation.get('sl2', {}).get('price', 0),
                'sh2_price': formation.get('sh2', {}).get('price', 0),
                'breakout_price': formation.get('breakout', {}).get('price', 0),
            }
            
        else:  # SIDEWAYS
            start_idx = formation['start_swing']['idx']
            end_idx = formation['end_swing']['idx']
            trend_swings = get_swings_in_range(all_swing_points, start_idx, end_idx)
            
            metrics = {
                'trend_id': trend_id,
                'trend_type': trend_type,
                'formation_date': formation_date.strftime('%Y-%m-%d') if hasattr(formation_date, 'strftime') else str(formation_date),
                'start_candle': start_idx,
                'end_candle': end_idx,
                'duration_candles': end_idx - start_idx + 1,
                'duration_days': (df.iloc[end_idx]['datetime'] - df.iloc[start_idx]['datetime']).days if 'datetime' in df.columns else 0,
                
                # Price metrics
                'start_price': df.iloc[start_idx]['close'],
                'end_price': df.iloc[end_idx]['close'],
                'highest_price': formation.get('high_level', df.iloc[start_idx:end_idx+1]['high'].max()),
                'lowest_price': formation.get('low_level', df.iloc[start_idx:end_idx+1]['low'].min()),
                'price_range': formation.get('range_size', df.iloc[start_idx:end_idx+1]['high'].max() - df.iloc[start_idx:end_idx+1]['low'].min()),
                'total_percent_change': ((df.iloc[end_idx]['close'] - df.iloc[start_idx]['close']) / df.iloc[start_idx]['close']) * 100,
                'max_favorable_move': 0,  # Sideways has no clear direction
                'max_adverse_move': formation.get('range_size', 0) / 2,  # Half the range as typical adverse move
                
                # Swing metrics
                'num_swings': formation.get('swing_count', len(trend_swings)),
                'num_swing_highs': len([s for s in trend_swings if s['type'] == 'HIGH']),
                'num_swing_lows': len([s for s in trend_swings if s['type'] == 'LOW']),
                'avg_candles_between_swings': calculate_avg_candles_between_swings(trend_swings),
                'swing_density': len(trend_swings) / (end_idx - start_idx + 1) if (end_idx - start_idx + 1) > 0 else 0,
                
                # Efficiency metrics  
                'price_efficiency': 0,  # Sideways by definition has low efficiency
                'trend_strength': formation.get('range_size', 0) / (end_idx - start_idx + 1),
                
                # Formation-specific metrics
                'sl1_price': 0,
                'sh1_price': 0, 
                'sl2_price': 0,
                'sh2_price': 0,
                'breakout_price': 0,
            }
        
        trend_metrics.append(metrics)
    
    return pd.DataFrame(trend_metrics)

def get_swings_in_range(all_swing_points, start_idx, end_idx):
    """Get swing points within a candle range"""
    return [sp for sp in all_swing_points if start_idx <= sp['index'] <= end_idx]

def calculate_max_favorable_move(df, start_idx, end_idx, trend_type):
    """Calculate the maximum favorable move during the trend"""
    start_price = df.iloc[start_idx]['close']
    
    if trend_type == 'UPTREND':
        max_price = df.iloc[start_idx:end_idx+1]['high'].max()
        return ((max_price - start_price) / start_price) * 100
    else:  # DOWNTREND
        min_price = df.iloc[start_idx:end_idx+1]['low'].min()
        return ((start_price - min_price) / start_price) * 100

def calculate_max_adverse_move(df, start_idx, end_idx, trend_type):
    """Calculate the maximum adverse move during the trend"""
    start_price = df.iloc[start_idx]['close']
    
    if trend_type == 'UPTREND':
        min_price = df.iloc[start_idx:end_idx+1]['low'].min()
        return abs(((min_price - start_price) / start_price) * 100)
    else:  # DOWNTREND
        max_price = df.iloc[start_idx:end_idx+1]['high'].max()
        return abs(((max_price - start_price) / start_price) * 100)

def calculate_avg_candles_between_swings(trend_swings):
    """Calculate average number of candles between swing points"""
    if len(trend_swings) < 2:
        return 0
    
    distances = []
    for i in range(1, len(trend_swings)):
        distances.append(trend_swings[i]['index'] - trend_swings[i-1]['index'])
    
    return np.mean(distances) if distances else 0

def calculate_price_efficiency(df, start_idx, end_idx):
    """
    Calculate price efficiency: straight-line move vs actual path taken
    Higher values = more efficient trending
    """
    if start_idx == end_idx:
        return 0
        
    start_price = df.iloc[start_idx]['close']
    end_price = df.iloc[end_idx]['close']
    straight_line_move = abs(end_price - start_price)
    
    if straight_line_move == 0:
        return 0
    
    # Calculate actual path (sum of absolute price changes)
    actual_path = 0
    for i in range(start_idx + 1, end_idx + 1):
        actual_path += abs(df.iloc[i]['close'] - df.iloc[i-1]['close'])
    
    if actual_path == 0:
        return 0
    
    return straight_line_move / actual_path

def create_distribution_visualizations(trend_df):
    """Create comprehensive distribution visualizations"""
    
    print(f"📈 Creating distribution visualizations for {len(trend_df)} trends...")
    
    # Filter out sideways trends as requested
    trend_df_filtered = trend_df[trend_df['trend_type'].isin(['UPTREND', 'DOWNTREND'])].copy()
    print(f"📊 Focusing on {len(trend_df_filtered)} directional trends (excluding sideways)")
    
    # Create subplots for multiple distributions
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Duration Distribution (Candles)", "Price Range Distribution", "Percent Change Distribution",
            "Number of Swings Distribution", "Swing Density Distribution", "Price Efficiency Distribution", 
            "Trend Strength Distribution", "Max Favorable Move", "Max Adverse Move"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Color mapping for trend types
    colors = {'UPTREND': '#00ff00', 'DOWNTREND': '#ff0000'}
    
    for trend_type in ['UPTREND', 'DOWNTREND']:
        subset = trend_df_filtered[trend_df_filtered['trend_type'] == trend_type]
        color = colors[trend_type]
        
        if len(subset) == 0:
            continue
        
        # Duration Distribution
        fig.add_trace(
            go.Histogram(x=subset['duration_candles'], name=f'{trend_type} Duration', 
                        marker_color=color, opacity=0.7, nbinsx=30),
            row=1, col=1
        )
        
        # Price Range Distribution  
        fig.add_trace(
            go.Histogram(x=subset['price_range'], name=f'{trend_type} Range',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=1, col=2
        )
        
        # Percent Change Distribution
        fig.add_trace(
            go.Histogram(x=subset['total_percent_change'], name=f'{trend_type} % Change',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=1, col=3
        )
        
        # Number of Swings
        fig.add_trace(
            go.Histogram(x=subset['num_swings'], name=f'{trend_type} Swings',
                        marker_color=color, opacity=0.7, nbinsx=20, showlegend=False),
            row=2, col=1
        )
        
        # Swing Density
        fig.add_trace(
            go.Histogram(x=subset['swing_density'], name=f'{trend_type} Density',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=2
        )
        
        # Price Efficiency
        fig.add_trace(
            go.Histogram(x=subset['price_efficiency'], name=f'{trend_type} Efficiency',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=3
        )
        
        # Trend Strength
        fig.add_trace(
            go.Histogram(x=subset['trend_strength'], name=f'{trend_type} Strength',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=3, col=1
        )
        
        # Max Favorable Move
        fig.add_trace(
            go.Histogram(x=subset['max_favorable_move'], name=f'{trend_type} Favorable',
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=3, col=2
        )
        
        # Max Adverse Move
        fig.add_trace(
            go.Histogram(x=subset['max_adverse_move'], name=f'{trend_type} Adverse', 
                        marker_color=color, opacity=0.7, nbinsx=30, showlegend=False),
            row=3, col=3
        )
    
    fig.update_layout(
        title="Directional Trend Metrics Distributions (Up/Down Only)",
        showlegend=True,
        height=1200,
        template='plotly_dark'
    )
    
    # Save distribution chart
    fig.write_html('directional_trend_distributions.html')
    print(f"✅ Directional trend distribution chart saved as: directional_trend_distributions.html")
    
    return fig

def create_summary_statistics(trend_df):
    """Create summary statistics by trend type"""
    
    print(f"📊 Creating summary statistics...")
    
    # Key metrics to summarize
    metrics = [
        'duration_candles', 'duration_days', 'price_range', 'total_percent_change',
        'num_swings', 'swing_density', 'price_efficiency', 'trend_strength',
        'max_favorable_move', 'max_adverse_move', 'avg_candles_between_swings'
    ]
    
    summary_stats = []
    
    for trend_type in ['UPTREND', 'DOWNTREND', 'SIDEWAYS']:
        subset = trend_df[trend_df['trend_type'] == trend_type]
        
        if len(subset) == 0:
            continue
            
        for metric in metrics:
            if metric in subset.columns:
                stats = {
                    'trend_type': trend_type,
                    'metric': metric,
                    'count': len(subset),
                    'mean': subset[metric].mean(),
                    'std': subset[metric].std(),
                    'min': subset[metric].min(),
                    'q25': subset[metric].quantile(0.25),
                    'median': subset[metric].median(),
                    'q75': subset[metric].quantile(0.75),
                    'max': subset[metric].max()
                }
                summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary statistics
    summary_df.to_csv('trend_summary_statistics.csv', index=False)
    print(f"✅ Summary statistics saved as: trend_summary_statistics.csv")
    
    return summary_df

def main():
    """Run comprehensive trend data analytics"""
    
    print("📊 COMPREHENSIVE TREND DATA ANALYTICS")
    print("Analyzing all trend formations with detailed metrics")
    print("=" * 70)
    
    # Get the data and trends
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Get swing points using the working system
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    print(f"📊 Dataset: {len(df)} candles, {len(swing_points)} swing points")
    
    # Find all trend formations
    formations = find_all_proper_trends(df, swing_points)
    terminations = find_all_proper_terminations(df, formations, swing_points)
    
    # Apply improved filtering AFTER we have termination data (same as trend explorer)
    original_count = len(formations)
    formations = filter_overlapping_sideways_trends_with_terminations(formations, terminations)
    filtered_count = original_count - len(formations)
    if filtered_count > 0:
        print(f"🔄 Filtered out {filtered_count} overlapping sideways trends")
    
    print(f"✅ Found {len(formations)} trend formations")
    print(f"   • {len([f for f in formations if f['type'] == 'UPTREND'])} uptrends")
    print(f"   • {len([f for f in formations if f['type'] == 'DOWNTREND'])} downtrends") 
    print(f"   • {len([f for f in formations if f['type'] == 'SIDEWAYS'])} sideways trends")
    
    # Calculate comprehensive metrics
    trend_df = calculate_trend_metrics(df, formations, swing_points)
    
    # Save detailed CSV
    trend_df.to_csv('trend_detailed_metrics.csv', index=False)
    print(f"✅ Detailed trend metrics saved as: trend_detailed_metrics.csv")
    
    # Create summary statistics
    summary_df = create_summary_statistics(trend_df)
    
    # Create distribution visualizations (directional only)
    dist_fig = create_distribution_visualizations(trend_df)
    
    # Create individual trend type visualizations
    create_individual_trend_visualizations(trend_df)
    
    # Create correlation analysis (directional only)
    create_correlation_analysis(trend_df)
    
    print(f"\n🎯 TREND DATA ANALYTICS COMPLETE!")
    print(f"   📁 Files created:")
    print(f"     • trend_detailed_metrics.csv - All individual trend data")
    print(f"     • trend_summary_statistics.csv - Statistical summaries by type")
    print(f"     • directional_trend_distributions.html - Combined up/down distributions")
    print(f"     • uptrend_analytics.html - Dedicated uptrend visualizations")
    print(f"     • downtrend_analytics.html - Dedicated downtrend visualizations")
    print(f"     • directional_trend_correlations.html - Up/down correlation analysis")
    print(f"   📊 Total trends analyzed: {len(trend_df)} (focusing on directional trends)")
    
    return trend_df, summary_df

def create_individual_trend_visualizations(trend_df):
    """Create separate visualizations for each trend type"""
    
    print(f"📊 Creating individual visualizations for each trend type...")
    
    # Filter to directional trends only
    directional_trends = trend_df[trend_df['trend_type'].isin(['UPTREND', 'DOWNTREND'])]
    
    for trend_type in ['UPTREND', 'DOWNTREND']:
        subset = directional_trends[directional_trends['trend_type'] == trend_type]
        
        if len(subset) == 0:
            continue
            
        print(f"  📈 Creating {trend_type.lower()} analytics ({len(subset)} trends)...")
        
        # Create individual visualization for this trend type
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                f"{trend_type} Duration (Candles)", f"{trend_type} Price Range", f"{trend_type} Percent Change",
                f"{trend_type} Swing Count", f"{trend_type} Swing Density", f"{trend_type} Price Efficiency", 
                f"{trend_type} Trend Strength", f"{trend_type} Max Favorable", f"{trend_type} Max Adverse"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        color = '#00ff00' if trend_type == 'UPTREND' else '#ff0000'
        
        # Duration Distribution
        fig.add_trace(go.Histogram(x=subset['duration_candles'], marker_color=color, opacity=0.7, nbinsx=25), row=1, col=1)
        
        # Price Range Distribution  
        fig.add_trace(go.Histogram(x=subset['price_range'], marker_color=color, opacity=0.7, nbinsx=25), row=1, col=2)
        
        # Percent Change Distribution
        fig.add_trace(go.Histogram(x=subset['total_percent_change'], marker_color=color, opacity=0.7, nbinsx=25), row=1, col=3)
        
        # Number of Swings
        fig.add_trace(go.Histogram(x=subset['num_swings'], marker_color=color, opacity=0.7, nbinsx=20), row=2, col=1)
        
        # Swing Density
        fig.add_trace(go.Histogram(x=subset['swing_density'], marker_color=color, opacity=0.7, nbinsx=25), row=2, col=2)
        
        # Price Efficiency
        fig.add_trace(go.Histogram(x=subset['price_efficiency'], marker_color=color, opacity=0.7, nbinsx=25), row=2, col=3)
        
        # Trend Strength
        fig.add_trace(go.Histogram(x=subset['trend_strength'], marker_color=color, opacity=0.7, nbinsx=25), row=3, col=1)
        
        # Max Favorable Move
        fig.add_trace(go.Histogram(x=subset['max_favorable_move'], marker_color=color, opacity=0.7, nbinsx=25), row=3, col=2)
        
        # Max Adverse Move
        fig.add_trace(go.Histogram(x=subset['max_adverse_move'], marker_color=color, opacity=0.7, nbinsx=25), row=3, col=3)
        
        fig.update_layout(
            title=f"{trend_type} Analytics - {len(subset)} Trends",
            showlegend=False,
            height=1200,
            template='plotly_dark'
        )
        
        filename = f"{trend_type.lower()}_analytics.html"
        fig.write_html(filename)
        print(f"  ✅ {trend_type} analytics saved as: {filename}")

def create_correlation_analysis(trend_df):
    """Create correlation analysis between trend metrics for directional trends"""
    
    print(f"🔗 Creating correlation analysis...")
    
    # Filter to directional trends only
    directional_trends = trend_df[trend_df['trend_type'].isin(['UPTREND', 'DOWNTREND'])]
    
    # Select numeric columns for correlation
    numeric_cols = [
        'duration_candles', 'price_range', 'total_percent_change', 'num_swings',
        'swing_density', 'price_efficiency', 'trend_strength', 'max_favorable_move', 
        'max_adverse_move', 'avg_candles_between_swings'
    ]
    
    # Create correlation matrix by trend type
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Uptrend Correlations", "Downtrend Correlations"],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    for i, trend_type in enumerate(['UPTREND', 'DOWNTREND']):
        subset = directional_trends[directional_trends['trend_type'] == trend_type]
        
        if len(subset) > 1:
            corr_matrix = subset[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=(i == 1),  # Only show scale on last plot
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title="Directional Trend Metrics Correlation Analysis",
        height=600,
        template='plotly_dark'
    )
    
    fig.write_html('directional_trend_correlations.html')
    print(f"✅ Directional correlation analysis saved as: directional_trend_correlations.html")

if __name__ == "__main__":
    main()