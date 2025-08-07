#!/usr/bin/env python3
"""
Clean Formation Examples - 4 Formation Types Only
Creates exactly 4 visual examples showing the corrected formation detection logic:
RBR, DBD, RBD, DBR
"""

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_rbr_data():
    """Rally-Base-Rally formation"""
    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    ohlc_data = [
        # RALLY IN (0-3)
        [100, 102, 99, 101],   # 0
        [101, 104, 100, 103],  # 1  
        [103, 106, 102, 105],  # 2
        [105, 108, 104, 107],  # 3 - rally ends at 107
        
        # BASE (4-9) - consolidation around 106-109
        [107, 109, 105, 106],  # 4 - base start
        [106, 108, 105, 107],  # 5 - base
        [107, 109, 106, 108],  # 6 - base
        [108, 109, 106, 107],  # 7 - base  
        [107, 109, 106, 108],  # 8 - base
        [108, 110, 107, 109],  # 9 - base end
        
        # RALLY OUT (10-14)
        [109, 112, 108, 111],  # 10 - breakout!
        [111, 114, 110, 113],  # 11
        [113, 116, 112, 115],  # 12  
        [115, 118, 114, 117],  # 13
        [117, 120, 116, 119],  # 14 - rally continues
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_dbd_data():
    """Drop-Base-Drop formation"""  
    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    ohlc_data = [
        # DROP IN (0-3)
        [120, 121, 117, 119],  # 0
        [119, 120, 115, 117],  # 1
        [117, 118, 113, 115],  # 2
        [115, 116, 111, 113],  # 3 - drop ends at 113
        
        # BASE (4-9) - consolidation around 112-115
        [113, 115, 111, 114],  # 4 - base start
        [114, 116, 113, 115],  # 5 - base
        [115, 116, 112, 113],  # 6 - base
        [113, 115, 112, 114],  # 7 - base
        [114, 116, 113, 115],  # 8 - base
        [115, 116, 112, 113],  # 9 - base end
        
        # DROP OUT (10-14)
        [113, 114, 110, 111],  # 10 - breakdown!
        [111, 112, 108, 109],  # 11
        [109, 110, 106, 107],  # 12
        [107, 108, 104, 105],  # 13
        [105, 106, 102, 103],  # 14 - drop continues
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_rbd_data():
    """Rally-Base-Drop formation (reversal)"""
    dates = pd.date_range('2024-01-01', periods=15, freq='D') 
    ohlc_data = [
        # RALLY IN (0-3)
        [100, 102, 99, 101],   # 0
        [101, 104, 100, 103],  # 1
        [103, 106, 102, 105],  # 2
        [105, 108, 104, 107],  # 3 - rally ends
        
        # BASE (4-9) - consolidation around 106-109
        [107, 109, 105, 106],  # 4 - base start
        [106, 108, 105, 107],  # 5 - base
        [107, 109, 106, 108],  # 6 - base
        [108, 109, 106, 107],  # 7 - base
        [107, 109, 106, 108],  # 8 - base
        [108, 110, 107, 109],  # 9 - base end
        
        # DROP OUT (10-14) - reversal!
        [109, 110, 106, 107],  # 10 - breakdown starts
        [107, 108, 104, 105],  # 11
        [105, 106, 102, 103],  # 12
        [103, 104, 100, 101],  # 13
        [101, 102, 98, 99],    # 14 - drop continues
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_dbr_data():
    """Drop-Base-Rally formation (reversal)"""
    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    ohlc_data = [
        # DROP IN (0-3)  
        [120, 121, 117, 119],  # 0
        [119, 120, 115, 117],  # 1
        [117, 118, 113, 115],  # 2
        [115, 116, 111, 113],  # 3 - drop ends
        
        # BASE (4-9) - consolidation around 112-115
        [113, 115, 111, 114],  # 4 - base start
        [114, 116, 113, 115],  # 5 - base
        [115, 116, 112, 113],  # 6 - base
        [113, 115, 112, 114],  # 7 - base
        [114, 116, 113, 115],  # 8 - base  
        [115, 116, 112, 113],  # 9 - base end
        
        # RALLY OUT (10-14) - reversal!
        [113, 116, 112, 115],  # 10 - breakout starts
        [115, 118, 114, 117],  # 11
        [117, 120, 116, 119],  # 12
        [119, 122, 118, 121],  # 13
        [121, 124, 120, 123],  # 14 - rally continues
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_formation_visualization(df, dates, formation_type, leg_in_range, base_range, leg_out_range):
    """Create visualization for a specific formation type"""
    
    df['datetime'] = dates[:len(df)]
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Color coding for different segments
    colors = {
        'LEG_IN': 'rgba(0, 255, 136, 0.3)',   # Green for rally in
        'BASE': 'rgba(255, 170, 0, 0.4)',     # Orange for base
        'LEG_OUT': 'rgba(0, 170, 255, 0.3)'   # Blue for leg out
    }
    
    if formation_type.startswith('D'):  # Drop in
        colors['LEG_IN'] = 'rgba(255, 68, 68, 0.3)'  # Red for drop in
    
    if formation_type.endswith('D'):  # Drop out
        colors['LEG_OUT'] = 'rgba(255, 68, 68, 0.3)'  # Red for drop out
    
    # Highlight LEG IN (0-3)
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[leg_in_range[0]], 
        x1=df['datetime'].iloc[leg_in_range[1]],
        y0=df['low'].min() * 0.98, 
        y1=df['high'].max() * 1.02,
        fillcolor=colors['LEG_IN'],
        opacity=0.6,
        line=dict(width=2, color='white'),
    )
    
    # Highlight BASE (4-9) 
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[base_range[0]], 
        x1=df['datetime'].iloc[base_range[1]],
        y0=df['low'].min() * 0.98, 
        y1=df['high'].max() * 1.02,
        fillcolor=colors['BASE'],
        opacity=0.7,
        line=dict(width=2, color='orange'),
    )
    
    # Highlight LEG OUT (10-14)
    fig.add_shape(
        type="rect", 
        x0=df['datetime'].iloc[leg_out_range[0]], 
        x1=df['datetime'].iloc[leg_out_range[1]],
        y0=df['low'].min() * 0.98, 
        y1=df['high'].max() * 1.02,
        fillcolor=colors['LEG_OUT'],
        opacity=0.6,
        line=dict(width=2, color='cyan'),
    )
    
    # Calculate actual ranges for validation
    leg_in_move = abs(df.iloc[leg_in_range[1]]['close'] - df.iloc[leg_in_range[0]]['close'])
    base_consolidation_range = df.iloc[base_range[0]:base_range[1]+1]['high'].max() - df.iloc[base_range[0]:base_range[1]+1]['low'].min()
    leg_out_move = abs(df.iloc[leg_out_range[1]]['close'] - df.iloc[leg_out_range[0]]['close'])
    
    valid = leg_in_move > base_consolidation_range and leg_out_move > base_consolidation_range
    
    # Add labels
    fig.add_annotation(
        x=df['datetime'].iloc[2],
        y=df['high'].max() * 1.01,
        text=f"<b>LEG IN</b><br>Move: {leg_in_move:.1f}",
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
    )
    
    fig.add_annotation(
        x=df['datetime'].iloc[6],
        y=df['low'].min() * 0.97,
        text=f"<b>BASE</b><br>Range: {base_consolidation_range:.1f}",
        showarrow=False,
        font=dict(size=10, color='orange'),
        bgcolor='rgba(0,0,0,0.7)',
    )
    
    fig.add_annotation(
        x=df['datetime'].iloc[12],
        y=df['high'].max() * 1.01,
        text=f"<b>LEG OUT</b><br>Move: {leg_out_move:.1f}",
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
    )
    
    # Formation title and validation
    status = '✅ VALID' if valid else '❌ INVALID'
    fig.add_annotation(
        x=df['datetime'].iloc[7],
        y=df['high'].max() * 1.03,
        text=f"<b>{formation_type} FORMATION {status}</b><br>" +
             f"Validation: L1({leg_in_move:.1f}) > B({base_consolidation_range:.1f}) < L2({leg_out_move:.1f})",
        showarrow=False,
        font=dict(size=12, color='cyan'),
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='cyan',
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'{formation_type} Formation - {status}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        font=dict(color='white'),
        height=600,
        showlegend=False
    )
    
    return fig

def main():
    """Generate the 4 clean formation examples"""
    
    print("🎯 GENERATING 4 CLEAN FORMATION EXAMPLES")
    print("=" * 50)
    
    formations = [
        ("RBR", create_rbr_data()),
        ("DBD", create_dbd_data()), 
        ("RBD", create_rbd_data()),
        ("DBR", create_dbr_data())
    ]
    
    for formation_type, (df, dates) in formations:
        print(f"\n📊 Creating {formation_type} formation example...")
        
        # Define the segment ranges for each formation type
        leg_in_range = (0, 3)   # First 4 candles
        base_range = (4, 9)     # Middle 6 candles  
        leg_out_range = (10, 14) # Last 5 candles
        
        fig = create_formation_visualization(
            df, dates, formation_type, 
            leg_in_range, base_range, leg_out_range
        )
        
        filename = f"{formation_type.lower()}_formation_example.html"
        fig.write_html(filename)
        print(f"   ✅ Saved: {filename}")
    
    print(f"\n🎯 COMPLETE! Generated 4 clean formation examples:")
    print(f"   • rbr_formation_example.html - Rally-Base-Rally")  
    print(f"   • dbd_formation_example.html - Drop-Base-Drop")
    print(f"   • rbd_formation_example.html - Rally-Base-Drop (reversal)")
    print(f"   • dbr_formation_example.html - Drop-Base-Rally (reversal)")
    print(f"\nThese show the correct formation structure for your educational program!")

if __name__ == "__main__":
    main()