#!/usr/bin/env python3
"""
Simple Swing Point Examples
Create clear, simple examples of different candle geometries to identify proper swing point rules
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Dark theme
pio.templates.default = "plotly_dark"

def create_swing_point_examples():
    """Create simple examples of different swing point scenarios"""
    
    examples = []
    
    # Example 1: Clear swing high - high exceeds neighboring candle ranges
    examples.append({
        'title': 'EXAMPLE 1: Clear Swing High',
        'description': 'High at candle 2 exceeds the entire range of candles 1 and 3',
        'candles': [
            {'O': 50, 'H': 52, 'L': 48, 'C': 51},  # Candle 0
            {'O': 51, 'H': 53, 'L': 49, 'C': 50},  # Candle 1  
            {'O': 50, 'H': 58, 'L': 49, 'C': 52},  # Candle 2 - SWING HIGH (58 > 53, 58 > 54)
            {'O': 52, 'H': 54, 'L': 50, 'C': 51},  # Candle 3
            {'O': 51, 'H': 53, 'L': 48, 'C': 49},  # Candle 4
        ],
        'swing_points': [{'idx': 2, 'type': 'HIGH', 'price': 58, 'reason': 'High 58 > neighbor ranges (48-53, 50-54)'}]
    })
    
    # Example 2: NOT a swing high - high within neighboring ranges
    examples.append({
        'title': 'EXAMPLE 2: NOT a Swing High',
        'description': 'High at candle 2 is contained within neighboring candle ranges',
        'candles': [
            {'O': 50, 'H': 55, 'L': 48, 'C': 51},  # Candle 0
            {'O': 51, 'H': 56, 'L': 49, 'C': 50},  # Candle 1  
            {'O': 50, 'H': 54, 'L': 49, 'C': 52},  # Candle 2 - NOT SWING HIGH (54 < 56)
            {'O': 52, 'H': 57, 'L': 50, 'C': 51},  # Candle 3
            {'O': 51, 'H': 53, 'L': 48, 'C': 49},  # Candle 4
        ],
        'swing_points': [{'idx': 3, 'type': 'HIGH', 'price': 57, 'reason': 'High 57 > neighbor ranges'}]
    })
    
    # Example 3: Swing low - low below neighboring ranges
    examples.append({
        'title': 'EXAMPLE 3: Clear Swing Low',
        'description': 'Low at candle 2 falls below neighboring candle ranges',
        'candles': [
            {'O': 50, 'H': 52, 'L': 48, 'C': 51},  # Candle 0
            {'O': 51, 'H': 53, 'L': 49, 'C': 50},  # Candle 1  
            {'O': 50, 'H': 51, 'L': 42, 'C': 49},  # Candle 2 - SWING LOW (42 < 48, 42 < 47)
            {'O': 49, 'H': 52, 'L': 47, 'C': 51},  # Candle 3
            {'O': 51, 'H': 53, 'L': 48, 'C': 49},  # Candle 4
        ],
        'swing_points': [{'idx': 2, 'type': 'LOW', 'price': 42, 'reason': 'Low 42 < neighbor ranges (48-53, 47-52)'}]
    })
    
    # Example 4: Contained candle - entirely within previous range
    examples.append({
        'title': 'EXAMPLE 4: Contained Candle',
        'description': 'Candle 2 is entirely contained within candle 1 range - no swing point',
        'candles': [
            {'O': 50, 'H': 52, 'L': 48, 'C': 51},  # Candle 0
            {'O': 51, 'H': 58, 'L': 44, 'C': 50},  # Candle 1 - Large range
            {'O': 50, 'H': 55, 'L': 47, 'C': 52},  # Candle 2 - CONTAINED (55 < 58, 47 > 44)
            {'O': 52, 'H': 54, 'L': 50, 'C': 51},  # Candle 3
            {'O': 51, 'H': 53, 'L': 48, 'C': 49},  # Candle 4
        ],
        'swing_points': [{'idx': 1, 'type': 'HIGH', 'price': 58, 'reason': 'High 58 exceeds neighbors'}, 
                        {'idx': 1, 'type': 'LOW', 'price': 44, 'reason': 'Low 44 below neighbors'}]
    })
    
    # Example 5: Multiple swing points in sequence
    examples.append({
        'title': 'EXAMPLE 5: Multiple Swing Points',
        'description': 'Clear alternating swing highs and lows',
        'candles': [
            {'O': 50, 'H': 52, 'L': 48, 'C': 49},  # Candle 0
            {'O': 49, 'H': 60, 'L': 48, 'C': 58},  # Candle 1 - SWING HIGH
            {'O': 58, 'H': 59, 'L': 52, 'C': 53},  # Candle 2
            {'O': 53, 'H': 55, 'L': 40, 'C': 42},  # Candle 3 - SWING LOW
            {'O': 42, 'H': 45, 'L': 41, 'C': 44},  # Candle 4
            {'O': 44, 'H': 62, 'L': 43, 'C': 60},  # Candle 5 - SWING HIGH
        ],
        'swing_points': [
            {'idx': 1, 'type': 'HIGH', 'price': 60, 'reason': 'High 60 > neighbor ranges'},
            {'idx': 3, 'type': 'LOW', 'price': 40, 'reason': 'Low 40 < neighbor ranges'},
            {'idx': 5, 'type': 'HIGH', 'price': 62, 'reason': 'High 62 > neighbor ranges'}
        ]
    })
    
    # Example 6: Edge case - touching but not exceeding
    examples.append({
        'title': 'EXAMPLE 6: Edge Case - Equal Levels',
        'description': 'What happens when highs/lows are equal to neighboring ranges?',
        'candles': [
            {'O': 50, 'H': 55, 'L': 48, 'C': 51},  # Candle 0
            {'O': 51, 'H': 53, 'L': 49, 'C': 50},  # Candle 1  
            {'O': 50, 'H': 55, 'L': 49, 'C': 52},  # Candle 2 - HIGH = 55 (same as candle 0)
            {'O': 52, 'H': 54, 'L': 50, 'C': 51},  # Candle 3
            {'O': 51, 'H': 53, 'L': 48, 'C': 49},  # Candle 4
        ],
        'swing_points': []  # Question: Is this a swing point or not?
    })
    
    return examples

def create_visual_examples(examples):
    """Create visual charts for each example"""
    
    # Create one large chart with all examples
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[ex['title'] for ex in examples],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, example in enumerate(examples):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        candles = example['candles']
        x_values = list(range(len(candles)))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=[c['O'] for c in candles],
                high=[c['H'] for c in candles],
                low=[c['L'] for c in candles],
                close=[c['C'] for c in candles],
                name=f"Example {idx+1}",
                showlegend=False,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=row, col=col
        )
        
        # Mark swing points
        for swing in example['swing_points']:
            color = '#ffaa00' if swing['type'] == 'HIGH' else '#00aaff'
            fig.add_trace(
                go.Scatter(
                    x=[swing['idx']],
                    y=[swing['price']],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='diamond'),
                    name=f"Swing {swing['type']}",
                    showlegend=False,
                    hovertemplate=f"{swing['type']}: {swing['price']}<br>{swing['reason']}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # Add range boxes to show the logic
        for i, candle in enumerate(candles):
            # Light box showing each candle's range
            fig.add_shape(
                type="rect",
                x0=i-0.3, y0=candle['L'],
                x1=i+0.3, y1=candle['H'],
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                fillcolor="rgba(255,255,255,0.05)",
                layer="below",
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Swing Point Detection Examples - Range-Based Logic",
            font=dict(size=16, color='white'),
            x=0.5
        ),
        height=1000,
        paper_bgcolor='rgba(15,15,15,1)',
        plot_bgcolor='rgba(25,25,25,1)',
        font=dict(color='white', size=10),
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    # Update axes
    for i in range(1, 7):
        row = ((i-1) // 2) + 1
        col = ((i-1) % 2) + 1
        fig.update_xaxes(title_text="Candle Index", gridcolor='rgba(100,100,100,0.2)', row=row, col=col)
        fig.update_yaxes(title_text="Price", gridcolor='rgba(100,100,100,0.2)', row=row, col=col)
    
    return fig

def print_example_analysis(examples):
    """Print detailed analysis of each example"""
    
    print("🎯 SWING POINT DETECTION EXAMPLES")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{example['title']}")
        print("-" * len(example['title']))
        print(f"📝 {example['description']}")
        
        print(f"\n📊 Candles:")
        for j, candle in enumerate(example['candles']):
            range_size = candle['H'] - candle['L']
            print(f"  {j}: O={candle['O']} H={candle['H']} L={candle['L']} C={candle['C']} | Range={range_size}")
        
        if example['swing_points']:
            print(f"\n🎯 Detected Swing Points:")
            for swing in example['swing_points']:
                print(f"  • Candle {swing['idx']}: {swing['type']} at {swing['price']} - {swing['reason']}")
        else:
            print(f"\n❌ No swing points detected")
        
        print()

def main():
    """Generate swing point examples"""
    
    print("🚀 Swing Point Detection Examples")
    print("Simple geometries to identify proper swing point rules")
    print("=" * 60)
    
    # Create examples
    examples = create_swing_point_examples()
    
    # Print analysis
    print_example_analysis(examples)
    
    # Create visual
    fig = create_visual_examples(examples)
    
    # Save chart
    filename = 'swing_point_examples.html'
    fig.write_html(filename)
    
    print(f"✅ Visual examples saved as: {filename}")
    print(f"\n🎯 Key Questions:")
    print("1. Do you agree with the swing points marked in each example?")
    print("2. Should 'equal levels' (Example 6) count as swing points?")
    print("3. Should we require exceeding by a minimum amount or percentage?")
    print("4. How many neighboring candles should we check (1 each side, 2 each side)?")
    print("\nOnce we agree on these examples, we can code the precise rules!")

if __name__ == "__main__":
    main()