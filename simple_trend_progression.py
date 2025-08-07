#!/usr/bin/env python3
"""
Simple Trend Formation Progression - Series of Static Images
Just create clear, simple images showing each stage of trend formation
"""

import plotly.graph_objects as go
import plotly.io as pio
from daily_trend_examples import analyze_daily_uso_trends
import os

def find_first_breakout_candle(df, start_idx, breakout_level, direction='above'):
    """
    Find the first candle after start_idx that breaks above/below the breakout_level
    This is where trend formation actually occurs!
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Index to start searching from (typically after SL2)
        breakout_level: Price level to break (typically SH1 high for uptrend)
        direction: 'above' for uptrend breakout, 'below' for downtrend breakout
    
    Returns:
        int: Index of first breakout candle, or None if no breakout found
    """
    
    for i in range(start_idx + 1, len(df)):
        candle = df.iloc[i]
        
        if direction == 'above':
            # Uptrend breakout: high must exceed breakout_level
            if candle['high'] > breakout_level:
                return i
        else:
            # Downtrend breakout: low must break below breakout_level  
            if candle['low'] < breakout_level:
                return i
    
    return None  # No breakout found

# Dark theme
pio.templates.default = "plotly_dark"

def write_html_with_navigation(fig, filepath, stage_num, total_stages):
    """Write HTML file with navigation buttons"""
    
    # Generate the base HTML from plotly
    html_string = fig.to_html(include_plotlyjs=True)
    
    # Define stage filenames
    stage_files = {
        1: "stage1_genesis.html",
        2: "stage2_first_rally.html", 
        3: "stage3_setup_complete.html",
        4: "stage4_approaching_breakout.html",
        5: "stage5_breakout_confirmed.html",
        6: "stage6_trend_active.html"
    }
    
    prev_file = stage_files.get(stage_num-1, "#")
    next_file = stage_files.get(stage_num+1, "#")
    
    # Create navigation buttons HTML
    nav_html = f"""
    <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                background: rgba(0,0,0,0.9); padding: 15px; border-radius: 10px; z-index: 1000;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3); border: 2px solid #333;">
        <button onclick="window.location.href='{prev_file}'" 
                {'disabled' if stage_num == 1 else ''}
                style="padding: 12px 24px; margin: 0 8px; background: {'#555' if stage_num == 1 else '#007bff'}; 
                       color: white; border: none; border-radius: 6px; cursor: {'not-allowed' if stage_num == 1 else 'pointer'};
                       font-weight: bold; font-size: 14px;">← Previous</button>
        <span style="color: white; margin: 0 20px; font-weight: bold; font-size: 16px;">
            Stage {stage_num} of {total_stages}
        </span>
        <button onclick="window.location.href='{next_file}'" 
                {'disabled' if stage_num == total_stages else ''}
                style="padding: 12px 24px; margin: 0 8px; background: {'#555' if stage_num == total_stages else '#007bff'}; 
                       color: white; border: none; border-radius: 6px; cursor: {'not-allowed' if stage_num == total_stages else 'pointer'};
                       font-weight: bold; font-size: 14px;">Next →</button>
        <button onclick="window.location.href='progression.html'" 
                style="padding: 12px 24px; margin: 0 8px; background: #28a745; color: white; 
                       border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px;">
            📋 Overview
        </button>
    </div>
    """
    
    # Insert navigation before closing body tag
    html_with_nav = html_string.replace('</body>', f'{nav_html}</body>')
    
    # Write the modified HTML
    with open(filepath, 'w') as f:
        f.write(html_with_nav)

def create_simple_progression_images():
    """Create a series of simple, clear images showing trend formation step by step"""
    
    print("🎯 Creating Simple Trend Formation Progression Images")
    print("=" * 60)
    
    # Get data
    df, swing_points = analyze_daily_uso_trends()
    
    # Our concrete example - USING PROPER BREAKOUT ORIGIN DETECTION
    sl1_idx = 36  # May 5: $61.75 - GENESIS POINT
    sh1_idx = 37  # May 6: $65.41 - FIRST HIGH  
    sl2_idx = 38  # May 7: $63.26 - HIGHER LOW
    
    # CRITICAL: Find the FIRST candle that breaks above SH1 (trend formation occurs here!)
    breakout_level = df.iloc[sh1_idx]['high']  # $65.41
    breakout_idx = find_first_breakout_candle(df, sl2_idx, breakout_level, 'above')
    
    print(f"✅ CORRECTED BREAKOUT ORIGIN DETECTION:")
    print(f"   Breakout level: ${breakout_level:.2f} (SH1)")
    print(f"   First breakout: Candle {breakout_idx} (${df.iloc[breakout_idx]['high']:.2f})")
    print(f"   Date: {df.iloc[breakout_idx]['datetime'].strftime('%m/%d/%Y')}")
    print(f"   🎯 TREND FORMS HERE - at origin, not at end of move!")
    
    # CLEAR window: 8 before SL1 + formation + 7 after breakout origin
    start_idx = sl1_idx - 8  # Candle 28
    end_idx = breakout_idx + 7   # Candle 46 (same window, but now relative to origin)
    
    window_df = df.iloc[start_idx:end_idx+1].copy()
    window_df = window_df.reset_index(drop=True)
    
    print(f"📊 Creating {6} progression images...")
    
    # Create output directory
    os.makedirs('trend_progression', exist_ok=True)
    
    # STAGE 1: Genesis Point (SL1)
    stage1_end = (sl1_idx - start_idx) + 1
    fig1 = create_simple_chart(
        window_df.iloc[:stage1_end],
        "STAGE 1: Genesis Point - First Swing Low (SL1)",
        swing_points=[(sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff')],
        annotations=[
            "🎯 TREND FORMATION BEGINS",
            "",
            "SL1: $61.75 - Genesis swing low",
            "This is our starting reference point",
            "",
            "NEXT: Need swing high above SL1"
        ],
        stage_num=1
    )
    write_html_with_navigation(fig1, 'trend_progression/stage1_genesis.html', 1, 6)
    print("   ✅ Stage 1: Genesis Point")
    
    # STAGE 2: First Rally (SL1 → SH1)
    stage2_end = (sh1_idx - start_idx) + 1
    fig2 = create_simple_chart(
        window_df.iloc[:stage2_end],
        "STAGE 2: First Rally - SL1 → SH1",
        swing_points=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], 'SH1\n$65.41', '#ffaa00')
        ],
        annotations=[
            "📈 FIRST RALLY COMPLETE",
            "",
            "SL1: $61.75 → SH1: $65.41",
            "Rally: +$3.66 (+5.9%)",
            "",
            "NEXT: Need higher low (SL2 > $61.75)"
        ],
        trend_lines=[(sl1_idx - start_idx, df.iloc[sl1_idx]['low'], sh1_idx - start_idx, df.iloc[sh1_idx]['high'], '#888888', 'dot')],
        stage_num=2
    )
    write_html_with_navigation(fig2, 'trend_progression/stage2_first_rally.html', 2, 6)
    print("   ✅ Stage 2: First Rally")
    
    # STAGE 3: Setup Complete (SL1 → SH1 → SL2)
    stage3_end = (sl2_idx - start_idx) + 1
    fig3 = create_simple_chart(
        window_df.iloc[:stage3_end],
        "STAGE 3: Setup Complete - Higher Low Confirmed",
        swing_points=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], 'SH1\n$65.41', '#ffaa00'),
            (sl2_idx - start_idx, df.iloc[sl2_idx]['low'], 'SL2\n$63.26', '#00aaff')
        ],
        annotations=[
            "✅ 3 SWING POINTS COMPLETE!",
            "",
            "SL1: $61.75 → SH1: $65.41 → SL2: $63.26",
            "Higher Low: SL2 ($63.26) > SL1 ($61.75) ✓",
            "Uptrend setup is READY",
            "WAITING: Breakout above SH1 ($65.41)"
        ],
        trend_lines=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], sh1_idx - start_idx, df.iloc[sh1_idx]['high'], '#888888', 'dot'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], sl2_idx - start_idx, df.iloc[sl2_idx]['low'], '#888888', 'dot'),
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], sl2_idx - start_idx, df.iloc[sl2_idx]['low'], '#ffaa00', 'dash')
        ],
        breakout_level=df.iloc[sh1_idx]['high'],
        stage_num=3
    )
    write_html_with_navigation(fig3, 'trend_progression/stage3_setup_complete.html', 3, 6)
    print("   ✅ Stage 3: Setup Complete")
    
    # STAGE 4: Approaching Breakout
    stage4_end = (breakout_idx - start_idx)
    fig4 = create_simple_chart(
        window_df.iloc[:stage4_end],
        "STAGE 4: Approaching Breakout Level",
        swing_points=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], 'SH1\n$65.41', '#ffaa00'),
            (sl2_idx - start_idx, df.iloc[sl2_idx]['low'], 'SL2\n$63.26', '#00aaff')
        ],
        annotations=[
            "⏳ APPROACHING BREAKOUT",
            "",
            "Setup: SL1 → SH1 → SL2 complete",
            "Breakout level: $65.41 (SH1)",
            f"Current high: ${window_df.iloc[stage4_end-1]['high']:.2f}",
            "Waiting for break above $65.41..."
        ],
        breakout_level=df.iloc[sh1_idx]['high'],
        stage_num=4
    )
    write_html_with_navigation(fig4, 'trend_progression/stage4_approaching_breakout.html', 4, 6)
    print("   ✅ Stage 4: Approaching Breakout")
    
    # STAGE 5: BREAKOUT ORIGIN!
    stage5_end = (breakout_idx - start_idx) + 1
    fig5 = create_simple_chart(
        window_df.iloc[:stage5_end],
        "STAGE 5: 🚀 BREAKOUT ORIGIN! UPTREND CONFIRMED",
        swing_points=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], 'SH1\n$65.41', '#ffaa00'),
            (sl2_idx - start_idx, df.iloc[sl2_idx]['low'], 'SL2\n$63.26', '#00aaff')
        ],
        annotations=[
            "🎯 UPTREND CONFIRMED AT ORIGIN!",
            "",
            f"BREAKOUT ORIGIN: ${df.iloc[breakout_idx]['high']:.2f} > $65.41 ✓",
            "3 swings + breakout origin = TREND FORMED",
            "Controlling swing: SL1 ($61.75)",
            "Trend active while price > $61.75"
        ],
        breakout_candle=(breakout_idx - start_idx, df.iloc[breakout_idx]['high']),
        trend_lines=[(sl1_idx - start_idx, df.iloc[sl1_idx]['low'], sl2_idx - start_idx, df.iloc[sl2_idx]['low'], '#00ff00', 'solid')],
        stage_num=5
    )
    write_html_with_navigation(fig5, 'trend_progression/stage5_breakout_confirmed.html', 5, 6)
    print("   ✅ Stage 5: BREAKOUT Confirmed")
    
    # STAGE 6: Trend Active
    fig6 = create_simple_chart(
        window_df,
        "STAGE 6: UPTREND ACTIVE - Trend Continuation",
        swing_points=[
            (sl1_idx - start_idx, df.iloc[sl1_idx]['low'], 'SL1\n$61.75', '#00aaff'),
            (sh1_idx - start_idx, df.iloc[sh1_idx]['high'], 'SH1\n$65.41', '#ffaa00'),
            (sl2_idx - start_idx, df.iloc[sl2_idx]['low'], 'SL2\n$63.26', '#00aaff')
        ],
        annotations=[
            "✅ UPTREND IS ACTIVE",
            "",
            "Formation: 3 swings + breakout ✓",
            "Controlling swing: SL1 ($61.75)",
            "Trend continues...",
            "Watch for violation below $61.75"
        ],
        trend_lines=[(sl1_idx - start_idx, df.iloc[sl1_idx]['low'], sl2_idx - start_idx, df.iloc[sl2_idx]['low'], '#00ff00', 'solid')],
        controlling_level=df.iloc[sl1_idx]['low'],
        stage_num=6
    )
    write_html_with_navigation(fig6, 'trend_progression/stage6_trend_active.html', 6, 6)
    print("   ✅ Stage 6: Trend Active")
    
    print(f"\n✅ All 6 progression images saved in 'trend_progression/' directory")
    print(f"🎬 Images show clear step-by-step trend formation:")
    print(f"   stage1_genesis.png - Genesis point (SL1)")
    print(f"   stage2_first_rally.png - First rally (SL1→SH1)")
    print(f"   stage3_setup_complete.png - 3 swing setup complete")
    print(f"   stage4_approaching_breakout.png - Building to breakout")
    print(f"   stage5_breakout_confirmed.png - BREAKOUT occurs")
    print(f"   stage6_trend_active.png - Uptrend confirmed and active")
    
    # Optional: Create a simple HTML page showing all images
    create_progression_html()

def create_simple_chart(df, title, swing_points=[], annotations=[], trend_lines=[], breakout_level=None, breakout_candle=None, controlling_level=None, stage_num=None, total_stages=6):
    """Create a simple, clear chart"""
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='USO Daily',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        showlegend=False
    ))
    
    # Add swing points
    for idx, price, label, color in swing_points:
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[price],
            mode='markers+text',
            marker=dict(color=color, size=16, symbol='diamond', line=dict(color='white', width=2)),
            text=[label],
            textposition='top center' if 'SH' in label else 'bottom center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            showlegend=False
        ))
    
    # Add trend lines
    for x1, y1, x2, y2, color, dash in trend_lines:
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color=color, width=3, dash=dash),
            showlegend=False
        ))
    
    # Add breakout level
    if breakout_level:
        fig.add_trace(go.Scatter(
            x=[0, len(df)-1],
            y=[breakout_level, breakout_level],
            mode='lines',
            line=dict(color='#ff9900', width=2, dash='dot'),
            showlegend=False
        ))
    
    # Add breakout marker
    if breakout_candle:
        idx, price = breakout_candle
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[price],
            mode='markers+text',
            marker=dict(color='#ff0000', size=20, symbol='star'),
            text=['ORIGIN!'],
            textposition='top center',
            textfont=dict(color='#ff0000', size=14, family='Arial Black'),
            showlegend=False
        ))
    
    # Add controlling level
    if controlling_level:
        fig.add_trace(go.Scatter(
            x=[0, len(df)-1],
            y=[controlling_level, controlling_level],
            mode='lines',
            line=dict(color='#00ff00', width=3, dash='solid'),
            showlegend=False
        ))
    

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white'), x=0.5),
        xaxis=dict(title='Candle Index', gridcolor='rgba(100,100,100,0.2)', showticklabels=True),
        yaxis=dict(title='Price ($)', gridcolor='rgba(100,100,100,0.2)'),
        template='plotly_dark',
        height=800,
        width=1200,
        showlegend=False,
        annotations=[
            dict(
                text='<br>'.join(annotations),
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color='white', size=14),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='white',
                borderwidth=2,
                align='left'
            )
        ]
    )
    
    
    return fig

def create_progression_html():
    """Create simple HTML page showing all progression images"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>USO Trend Formation Progression</title>
        <style>
            body { background-color: #1e1e1e; color: white; font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; color: #00ff88; }
            h2 { color: #ffaa00; }
            .stage { margin: 40px 0; text-align: center; position: relative; }
            .stage iframe { max-width: 100%; height: auto; border: 2px solid #333; }
            .description { margin: 20px 0; font-size: 16px; }
            .stage-nav { margin: 10px 0; }
            .stage-button { 
                background: #007bff; color: white; border: none; padding: 10px 20px; 
                border-radius: 5px; cursor: pointer; margin: 0 5px; font-weight: bold;
                transition: background 0.3s;
            }
            .stage-button:hover { background: #0056b3; }
            .nav-bar {
                text-align: center; margin: 30px 0; padding: 20px;
                background: rgba(0,0,0,0.3); border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <h1>🎯 USO Uptrend Formation: Step-by-Step Progression</h1>
        
        <div class="nav-bar">
            <h3>Quick Navigation:</h3>
            <button class="stage-button" onclick="location.href='stage1_genesis.html'">Stage 1: Genesis</button>
            <button class="stage-button" onclick="location.href='stage2_first_rally.html'">Stage 2: Rally</button>
            <button class="stage-button" onclick="location.href='stage3_setup_complete.html'">Stage 3: Setup</button>
            <button class="stage-button" onclick="location.href='stage4_approaching_breakout.html'">Stage 4: Approach</button>
            <button class="stage-button" onclick="location.href='stage5_breakout_confirmed.html'">Stage 5: Breakout</button>
            <button class="stage-button" onclick="location.href='stage6_trend_active.html'">Stage 6: Active</button>
        </div>
        
        <div class="stage">
            <h2>Stage 1: Genesis Point</h2>
            <img src="stage1_genesis.png" alt="Stage 1: Genesis Point">
            <div class="description">First swing low (SL1) at $61.75 - our starting reference point</div>
        </div>
        
        <div class="stage">
            <h2>Stage 2: First Rally</h2>
            <img src="stage2_first_rally.png" alt="Stage 2: First Rally">
            <div class="description">Rally from SL1 ($61.75) to SH1 ($65.41) - first leg complete</div>
        </div>
        
        <div class="stage">
            <h2>Stage 3: Setup Complete</h2>
            <img src="stage3_setup_complete.png" alt="Stage 3: Setup Complete">
            <div class="description">Higher low SL2 ($63.26) > SL1 ($61.75) ✓ - 3 swing points ready for breakout</div>
        </div>
        
        <div class="stage">
            <h2>Stage 4: Approaching Breakout</h2>
            <img src="stage4_approaching_breakout.png" alt="Stage 4: Approaching Breakout">
            <div class="description">Price building toward breakout level of $65.41 (SH1)</div>
        </div>
        
        <div class="stage">
            <h2>Stage 5: BREAKOUT Confirmed</h2>
            <img src="stage5_breakout_confirmed.png" alt="Stage 5: BREAKOUT">
            <div class="description">🚀 Price breaks above $65.41 - UPTREND CONFIRMED!</div>
        </div>
        
        <div class="stage">
            <h2>Stage 6: Trend Active</h2>
            <img src="stage6_trend_active.png" alt="Stage 6: Trend Active">
            <div class="description">✅ Uptrend is now active with controlling swing at $61.75</div>
        </div>
        
        <div style="text-align: center; margin-top: 50px; color: #888;">
            <p><strong>Key Concept:</strong> Trend formation requires 3 swing points + breakout candle</p>
            <p><strong>Formation:</strong> SL1 → SH1 → SL2 (higher low) → Breakout above SH1</p>
        </div>
    </body>
    </html>
    """
    
    with open('trend_progression/progression.html', 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML overview saved as: trend_progression/progression.html")

def main():
    """Create the simple progression images"""
    create_simple_progression_images()

if __name__ == "__main__":
    main()