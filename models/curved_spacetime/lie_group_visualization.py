"""
Lie Group Transformation Visualization

Demonstrates the ML(2) market transformation group with:
- Interactive transformation explorer
- Pattern classification demo
- Invariant feature display
- Transformation sequences
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from market_lie_group import *
from curved_candle_geometry import CandleMetric
from curvature_color_scheme import get_curvature_gradient_color


def create_pattern_transformation_explorer(base_candle: CandleMetric) -> go.Figure:
    """
    Create an interactive explorer showing how transformations affect patterns.
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Original Pattern',
            'Sentiment Shift',
            'Wick Rotation',
            'Volatility Scaling',
            'Pattern Inversion',
            'Composite Transform'
        ),
        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
               [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]]
    )
    
    # Helper function to create polar plot for a candle
    def add_candle_polar(candle: CandleMetric, row: int, col: int, name: str):
        # Convert to polar coordinates
        sentiment = candle.sentiment
        uwr = candle.upper_wick_ratio
        
        # Radius is distance from origin in pattern space
        r = np.sqrt(sentiment**2 + uwr**2)
        
        # Angle encodes the ratio of sentiment to uwr
        theta = np.arctan2(uwr, sentiment) * 180 / np.pi
        
        # Size represents range
        size = 20 + 30 * np.tanh(candle.range_value)
        
        # Color represents volume
        color_intensity = np.tanh(np.log1p(candle.volume))
        
        fig.add_trace(
            go.Scatterpolar(
                r=[r],
                theta=[theta],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=[color_intensity],
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(width=2, color='white')
                ),
                text=[f'S:{sentiment:.2f}<br>U:{uwr:.2f}<br>R:{candle.range_value:.2f}'],
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add coordinate grid
        fig.add_trace(
            go.Scatterpolar(
                r=[0, 1],
                theta=[0, 0],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ),
            row=row, col=col
        )
    
    # 1. Original pattern
    add_candle_polar(base_candle, 1, 1, 'Original')
    
    # 2. Sentiment shift
    sentiment_transform = SentimentShift(0.3)
    shifted_candle = sentiment_transform.apply(base_candle)
    add_candle_polar(shifted_candle, 1, 2, 'Shifted +0.3')
    
    # 3. Wick rotation
    wick_transform = WickRotation(np.pi/6)  # 30 degrees
    rotated_candle = wick_transform.apply(base_candle)
    add_candle_polar(rotated_candle, 1, 3, 'Rotated 30°')
    
    # 4. Volatility scaling
    vol_transform = VolatilityScaling(2.0)
    scaled_candle = vol_transform.apply(base_candle)
    add_candle_polar(scaled_candle, 2, 1, 'Scaled 2x')
    
    # 5. Pattern inversion
    inv_transform = PatternInversion()
    inverted_candle = inv_transform.apply(base_candle)
    add_candle_polar(inverted_candle, 2, 2, 'Inverted')
    
    # 6. Composite transformation
    composite = CompositeTransformation([
        SentimentShift(0.2),
        WickRotation(np.pi/8),
        VolatilityScaling(1.5)
    ])
    composite_candle = composite.apply(base_candle)
    add_candle_polar(composite_candle, 2, 3, 'Composite')
    
    # Update layout
    fig.update_layout(
        title='Market Pattern Transformations Explorer',
        showlegend=False,
        template='plotly_dark',
        height=800,
        polar=dict(
            radialaxis=dict(range=[0, 1]),
            angularaxis=dict(direction='counterclockwise')
        )
    )
    
    # Update all polar axes
    for i in range(1, 7):
        row = (i-1) // 3 + 1
        col = (i-1) % 3 + 1
        fig.update_polars(
            radialaxis_range=[0, 1],
            row=row, col=col
        )
    
    return fig


def create_invariant_features_display(candles: List[CandleMetric]) -> go.Figure:
    """
    Display transformation-invariant features for a series of candles.
    """
    group = MarketLieGroup()
    
    # Compute invariants for each candle
    invariants_list = [group.compute_invariants(candle) for candle in candles]
    
    # Extract time series
    normalized_volumes = [inv['normalized_volume'] for inv in invariants_list]
    wick_asymmetries = [inv['wick_asymmetry'] for inv in invariants_list]
    pattern_energies = [inv['pattern_energy'] for inv in invariants_list]
    curvatures = [inv['intrinsic_curvature'] for inv in invariants_list]
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            'Normalized Volume (Range-Invariant)',
            'Wick Asymmetry (Sentiment-Invariant)',
            'Pattern Energy (Isometry-Invariant)',
            'Intrinsic Curvature (Geometric Invariant)'
        ),
        vertical_spacing=0.05
    )
    
    x = list(range(len(candles)))
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=normalized_volumes, mode='lines+markers',
                   name='Normalized Volume', line=dict(color='#00b4d8')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x, y=wick_asymmetries, mode='lines+markers',
                   name='Wick Asymmetry', line=dict(color='#f72585')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x, y=pattern_energies, mode='lines+markers',
                   name='Pattern Energy', line=dict(color='#7209b7')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x, y=curvatures, mode='lines+markers',
                   name='Curvature', line=dict(color='#560bad')),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Transformation-Invariant Features',
        template='plotly_dark',
        height=800,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Candle Index', row=4, col=1)
    
    return fig


def create_pattern_classification_demo(test_candles: List[CandleMetric]) -> go.Figure:
    """
    Demonstrate pattern classification using transformations.
    """
    # Define reference patterns
    reference_patterns = {
        'Doji': CandleMetric(
            range_value=0.5,
            low_value=100,
            sentiment=0.0,  # Close = Open
            upper_wick_ratio=0.5,  # Symmetric wicks
            volume=1000
        ),
        'Hammer': CandleMetric(
            range_value=1.0,
            low_value=100,
            sentiment=0.3,  # Bullish
            upper_wick_ratio=0.1,  # Small upper wick
            volume=1500
        ),
        'Shooting Star': CandleMetric(
            range_value=1.0,
            low_value=100,
            sentiment=-0.3,  # Bearish
            upper_wick_ratio=0.7,  # Large upper wick
            volume=1500
        ),
        'Marubozu': CandleMetric(
            range_value=2.0,
            low_value=100,
            sentiment=0.9,  # Strong bullish
            upper_wick_ratio=0.05,  # Tiny wicks
            volume=2000
        )
    }
    
    # Classify each test candle
    classifications = []
    for candle in test_candles:
        pattern_name, transform = classify_pattern_via_transformations(
            candle, reference_patterns
        )
        classifications.append({
            'candle': candle,
            'pattern': pattern_name,
            'transform': transform
        })
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Pattern Classification Results', 'Transformation Parameters')
    )
    
    # Plot classifications
    x = list(range(len(test_candles)))
    patterns = [c['pattern'] for c in classifications]
    pattern_types = list(set(patterns))
    colors = ['#e63946', '#f1faee', '#a8dadc', '#457b9d', '#1d3557']
    
    for i, pattern_type in enumerate(pattern_types):
        indices = [j for j, p in enumerate(patterns) if p == pattern_type]
        if indices:
            fig.add_trace(
                go.Scatter(
                    x=[x[j] for j in indices],
                    y=[1] * len(indices),
                    mode='markers',
                    marker=dict(size=15, color=colors[i % len(colors)]),
                    name=pattern_type
                ),
                row=1, col=1
            )
    
    # Add reference patterns as horizontal lines
    fig.add_trace(
        go.Scatter(
            x=[0, len(test_candles)-1],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Plot transformation magnitudes
    transform_magnitudes = []
    for c in classifications:
        if c['transform'] is not None:
            params = c['transform'].get_parameters()
            # Simple magnitude calculation
            if params['type'] == 'sentiment_shift':
                magnitude = abs(params['amount'])
            elif params['type'] == 'composite':
                magnitude = len(params['components'])
            else:
                magnitude = 0.5
        else:
            magnitude = 0
        transform_magnitudes.append(magnitude)
    
    fig.add_trace(
        go.Bar(
            x=x,
            y=transform_magnitudes,
            name='Transform Magnitude',
            marker_color='#9d4edd'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Pattern Classification via Group Transformations',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    fig.update_yaxes(title_text='Pattern Type', row=1, col=1)
    fig.update_yaxes(title_text='Transform Magnitude', row=2, col=1)
    fig.update_xaxes(title_text='Candle Index', row=2, col=1)
    
    return fig


def create_transformation_sequence_animation(base_candle: CandleMetric,
                                           n_steps: int = 20) -> go.Figure:
    """
    Create an animation showing a sequence of transformations.
    """
    # Generate a smooth transformation sequence
    frames = []
    
    # Parameters for smooth animation
    max_sentiment = 0.5
    max_rotation = np.pi/2
    max_scale = 2.0
    
    for i in range(n_steps):
        t = i / (n_steps - 1)  # Parameter from 0 to 1
        
        # Smooth transformation parameters
        sentiment_shift = max_sentiment * np.sin(2 * np.pi * t)
        wick_rotation = max_rotation * np.sin(4 * np.pi * t)
        volatility_scale = 1 + (max_scale - 1) * (0.5 + 0.5 * np.cos(2 * np.pi * t))
        
        # Create composite transformation
        transform = CompositeTransformation([
            SentimentShift(sentiment_shift),
            WickRotation(wick_rotation),
            VolatilityScaling(volatility_scale)
        ])
        
        # Apply transformation
        transformed = transform.apply(base_candle)
        
        # Create frame data
        frame_data = go.Frame(
            data=[
                go.Scatter(
                    x=[transformed.sentiment],
                    y=[transformed.upper_wick_ratio],
                    mode='markers',
                    marker=dict(
                        size=20 + 20 * transformed.range_value,
                        color='#e0aaff',
                        line=dict(width=2, color='white')
                    ),
                    text=f'Step {i}<br>S:{transformed.sentiment:.2f}<br>U:{transformed.upper_wick_ratio:.2f}'
                )
            ],
            name=f'frame_{i}'
        )
        frames.append(frame_data)
    
    # Create initial figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[base_candle.sentiment],
                y=[base_candle.upper_wick_ratio],
                mode='markers',
                marker=dict(
                    size=20 + 20 * base_candle.range_value,
                    color='#e0aaff',
                    line=dict(width=2, color='white')
                )
            )
        ],
        frames=frames
    )
    
    # Add play/pause buttons
    fig.update_layout(
        title='Transformation Sequence Animation',
        xaxis=dict(title='Sentiment', range=[-1, 1]),
        yaxis=dict(title='Upper Wick Ratio', range=[0, 1]),
        template='plotly_dark',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'label': f'Step {i}',
                    'method': 'animate',
                    'args': [[f'frame_{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for i in range(n_steps)
            ]
        }]
    )
    
    return fig


def main():
    """Test the Lie group visualization system."""
    
    # Create test candles
    test_candles = [
        CandleMetric(range_value=1.0, low_value=100, sentiment=0.2, 
                    upper_wick_ratio=0.3, volume=1000),
        CandleMetric(range_value=0.5, low_value=101, sentiment=-0.1, 
                    upper_wick_ratio=0.5, volume=800),
        CandleMetric(range_value=1.5, low_value=100.5, sentiment=0.5, 
                    upper_wick_ratio=0.2, volume=1500),
        CandleMetric(range_value=0.8, low_value=102, sentiment=-0.3, 
                    upper_wick_ratio=0.6, volume=900),
        CandleMetric(range_value=2.0, low_value=99, sentiment=0.7, 
                    upper_wick_ratio=0.1, volume=2000),
    ]
    
    # Create visualizations
    print("Creating transformation explorer...")
    explorer = create_pattern_transformation_explorer(test_candles[0])
    explorer.write_html("lie_group_explorer.html")
    
    print("Creating invariant features display...")
    invariants = create_invariant_features_display(test_candles)
    invariants.write_html("lie_group_invariants.html")
    
    print("Creating pattern classification demo...")
    classification = create_pattern_classification_demo(test_candles)
    classification.write_html("lie_group_classification.html")
    
    print("Creating transformation animation...")
    animation = create_transformation_sequence_animation(test_candles[0])
    animation.write_html("lie_group_animation.html")
    
    print("\n✅ Lie group visualizations created:")
    print("   - lie_group_explorer.html")
    print("   - lie_group_invariants.html")
    print("   - lie_group_classification.html")
    print("   - lie_group_animation.html")


if __name__ == "__main__":
    main()