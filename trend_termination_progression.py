#!/usr/bin/env python3
"""
Trend Termination Progression - Step-by-step visualization
Shows how trends terminate through violation of controlling swing point
Uses the USO June 2025 crash as a perfect example
"""

import plotly.graph_objects as go
import plotly.io as pio
from daily_trend_examples import analyze_daily_uso_trends
import os

def find_first_violation_candle(df, start_idx, controlling_level, direction='below'):
    """
    Find the first candle after start_idx that violates the controlling level
    This is where trend termination actually occurs!
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Index to start searching from 
        controlling_level: Price level that protects trend
        direction: 'below' for uptrend violation, 'above' for downtrend violation
    
    Returns:
        int: Index of first violation candle, or None if no violation found
    """
    
    for i in range(start_idx + 1, len(df)):
        candle = df.iloc[i]
        
        if direction == 'below':
            # Uptrend violation: low must break below controlling_level
            if candle['low'] < controlling_level:
                return i
        else:
            # Downtrend violation: high must break above controlling_level  
            if candle['high'] > controlling_level:
                return i
    
    return None  # No violation found

def write_html_with_navigation(fig, filepath, stage_num, total_stages):
    """Write HTML file with navigation buttons"""
    
    # Generate the base HTML from plotly
    html_string = fig.to_html(include_plotlyjs=True)
    
    # Define stage filenames for termination
    stage_files = {
        1: "term_stage1_active.html",
        2: "term_stage2_peak.html", 
        3: "term_stage3_approaching.html",
        4: "term_stage4_violation.html",
        5: "term_stage5_terminated.html"
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
        <button onclick="window.location.href='termination_overview.html'" 
                style="padding: 12px 24px; margin: 0 8px; background: #dc3545; color: white; 
                       border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px;">
            📋 Overview
        </button>
        <button onclick="window.location.href='progression.html'" 
                style="padding: 12px 24px; margin: 0 8px; background: #28a745; color: white; 
                       border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px;">
            📈 Formation
        </button>
    </div>
    """
    
    # Insert navigation before closing body tag
    html_with_nav = html_string.replace('</body>', f'{nav_html}</body>')
    
    # Write the modified HTML
    with open(filepath, 'w') as f:
        f.write(html_with_nav)

# Dark theme
pio.templates.default = "plotly_dark"

def create_trend_termination_progression():
    """Create step-by-step trend termination visualization"""
    
    print("🛑 Creating Trend Termination Progression")
    print("=" * 60)
    
    # Get data
    df, swing_points = analyze_daily_uso_trends()
    
    # Perfect termination example: USO June 2025 crash
    controlling_idx = 65  # June 16: $76.25 - controlling swing low
    peak_idx = 69         # June 23: $83.57 - peak before crash  
    violation_idx = 70    # June 24: $71.97 - violation
    
    controlling_level = df.iloc[controlling_idx]['low']
    
    print(f"🎯 USO JUNE 2025 CRASH - TREND TERMINATION EXAMPLE")
    print(f"   Controlling swing: ${controlling_level:.2f} (protects uptrend)")  
    print(f"   Peak: ${df.iloc[peak_idx]['high']:.2f}")
    print(f"   Violation: ${df.iloc[violation_idx]['low']:.2f}")
    print(f"   Termination confirmed!")
    
    # Create visualization window: 8 candles before controlling + termination + 3 after
    start_idx = controlling_idx - 8  # Candle 57
    end_idx = violation_idx + 3       # Candle 73
    
    window_df = df.iloc[start_idx:end_idx+1].copy()
    window_df = window_df.reset_index(drop=True)
    
    print(f"📊 Creating 5 termination progression images...")
    
    # Create output directory
    os.makedirs('trend_progression', exist_ok=True)
    
    # STAGE 1: Active Uptrend - Show controlling swing protection
    stage1_end = (controlling_idx - start_idx) + 1
    fig1 = create_termination_chart(
        window_df.iloc[:stage1_end],
        "STAGE 1: Active Uptrend - Controlling Swing Protection",
        controlling_swing=[(controlling_idx - start_idx, df.iloc[controlling_idx]['low'], 'Control\\n$76.25', '#00aaff')],
        annotations=[
            "🛡️ UPTREND ACTIVE & PROTECTED",
            "",
            "Controlling swing: $76.25 (SL)",
            "Trend stays active while price > $76.25", 
            "This level must hold for uptrend to continue",
            "Watch for violation below $76.25"
        ],
        controlling_level=controlling_level,
        stage_num=1
    )
    write_html_with_navigation(fig1, 'trend_progression/term_stage1_active.html', 1, 5)
    print("   ✅ Stage 1: Active Uptrend")
    
    # STAGE 2: Peak Before Crash - Show the high before termination
    stage2_end = (peak_idx - start_idx) + 1
    fig2 = create_termination_chart(
        window_df.iloc[:stage2_end],
        "STAGE 2: Peak Before Crash - Last High",
        controlling_swing=[(controlling_idx - start_idx, df.iloc[controlling_idx]['low'], 'Control\\n$76.25', '#00aaff')],
        peak_marker=[(peak_idx - start_idx, df.iloc[peak_idx]['high'], 'PEAK\\n$83.57', '#ffaa00')],
        annotations=[
            "⛰️ PEAK REACHED: $83.57",
            "",
            f"High: ${df.iloc[peak_idx]['high']:.2f} on {df.iloc[peak_idx]['datetime'].strftime('%m/%d')}",
            "Controlling swing: $76.25 still holding",
            "Uptrend technically still active",
            "⚠️ Watch for breakdown below $76.25"
        ],
        controlling_level=controlling_level,
        stage_num=2
    )
    write_html_with_navigation(fig2, 'trend_progression/term_stage2_peak.html', 2, 5)
    print("   ✅ Stage 2: Peak Before Crash")
    
    # STAGE 3: Approaching Violation - Show price falling toward violation
    stage3_end = (violation_idx - start_idx)  # Just before violation
    fig3 = create_termination_chart(
        window_df.iloc[:stage3_end],
        "STAGE 3: Approaching Violation Level",
        controlling_swing=[(controlling_idx - start_idx, df.iloc[controlling_idx]['low'], 'Control\\n$76.25', '#00aaff')],
        peak_marker=[(peak_idx - start_idx, df.iloc[peak_idx]['high'], 'PEAK\\n$83.57', '#ffaa00')],
        annotations=[
            "⚠️ APPROACHING VIOLATION LEVEL",
            "",
            "Price falling from peak $83.57",
            "Controlling swing: $76.25", 
            f"Current low: ${window_df.iloc[stage3_end-1]['low']:.2f}",
            "DANGER: Getting close to $76.25!"
        ],
        controlling_level=controlling_level,
        stage_num=3
    )
    write_html_with_navigation(fig3, 'trend_progression/term_stage3_approaching.html', 3, 5)
    print("   ✅ Stage 3: Approaching Violation")
    
    # STAGE 4: VIOLATION! - Show the exact moment of termination  
    stage4_end = (violation_idx - start_idx) + 1
    fig4 = create_termination_chart(
        window_df.iloc[:stage4_end],
        "STAGE 4: 🛑 VIOLATION! UPTREND TERMINATED",
        controlling_swing=[(controlling_idx - start_idx, df.iloc[controlling_idx]['low'], 'Control\\n$76.25', '#00aaff')],
        peak_marker=[(peak_idx - start_idx, df.iloc[peak_idx]['high'], 'PEAK\\n$83.57', '#ffaa00')],
        violation_marker=[(violation_idx - start_idx, df.iloc[violation_idx]['low'], 'VIOLATION!\\n$71.97', '#ff0000')],
        annotations=[
            "🛑 UPTREND TERMINATED!",
            "",
            f"VIOLATION: ${df.iloc[violation_idx]['low']:.2f} < $76.25 ✓",
            "Controlling swing broken",
            f"Violation size: ${controlling_level - df.iloc[violation_idx]['low']:.2f}",
            "Trend is now terminated!"
        ],
        controlling_level=controlling_level,
        show_violation=True,
        stage_num=4
    )
    write_html_with_navigation(fig4, 'trend_progression/term_stage4_violation.html', 4, 5)
    print("   ✅ Stage 4: VIOLATION!")
    
    # STAGE 5: Terminated - Show aftermath
    fig5 = create_termination_chart(
        window_df,
        "STAGE 5: Uptrend Terminated - No Longer in Trend",
        controlling_swing=[(controlling_idx - start_idx, df.iloc[controlling_idx]['low'], 'Control\\n$76.25', '#00aaff')],
        peak_marker=[(peak_idx - start_idx, df.iloc[peak_idx]['high'], 'PEAK\\n$83.57', '#ffaa00')],
        violation_marker=[(violation_idx - start_idx, df.iloc[violation_idx]['low'], 'VIOLATION!\\n$71.97', '#ff0000')],
        annotations=[
            "❌ UPTREND NO LONGER ACTIVE",
            "",
            "Violation confirmed at $71.97",
            "Controlling swing ($76.25) broken", 
            "Price action now bearish/neutral",
            "Look for new trend formation signals"
        ],
        controlling_level=controlling_level,
        show_violation=True,
        show_violation_zone=True,
        stage_num=5
    )
    write_html_with_navigation(fig5, 'trend_progression/term_stage5_terminated.html', 5, 5)
    print("   ✅ Stage 5: Terminated")
    
    print(f"\n✅ All 5 termination progression HTML files saved!")
    print(f"🎬 HTML files show step-by-step trend termination:")
    print(f"   term_stage1_active.html - Active uptrend protection")
    print(f"   term_stage2_peak.html - Peak before crash") 
    print(f"   term_stage3_approaching.html - Approaching violation")
    print(f"   term_stage4_violation.html - VIOLATION occurs")
    print(f"   term_stage5_terminated.html - Trend terminated")
    print(f"")
    print(f"🔧 KEY CONCEPT: Trends terminate at VIOLATION of controlling swing")
    print(f"   Not at the peak, but when support breaks!")
    
    # Create termination overview page
    create_termination_overview()

def create_termination_chart(df, title, controlling_swing=[], peak_marker=[], violation_marker=[], annotations=[], 
                           controlling_level=None, show_violation=False, show_violation_zone=False, stage_num=None):
    """Create a termination chart"""
    
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
    
    # Add controlling swing marker
    for idx, price, label, color in controlling_swing:
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[price],
            mode='markers+text',
            marker=dict(color=color, size=16, symbol='diamond', line=dict(color='white', width=2)),
            text=[label],
            textposition='bottom center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            showlegend=False
        ))
    
    # Add peak marker
    for idx, price, label, color in peak_marker:
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[price],
            mode='markers+text',
            marker=dict(color=color, size=16, symbol='triangle-up', line=dict(color='white', width=2)),
            text=[label],
            textposition='top center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            showlegend=False
        ))
    
    # Add violation marker
    for idx, price, label, color in violation_marker:
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[price],
            mode='markers+text',
            marker=dict(color=color, size=20, symbol='x'),
            text=[label],
            textposition='bottom center',
            textfont=dict(color='#ff0000', size=14, family='Arial Black'),
            showlegend=False
        ))
    
    # Add controlling level line
    if controlling_level:
        line_color = '#ff6600' if show_violation else '#00ff00'
        line_dash = 'solid' if show_violation else 'dot'
        fig.add_trace(go.Scatter(
            x=[0, len(df)-1],
            y=[controlling_level, controlling_level],
            mode='lines',
            line=dict(color=line_color, width=3, dash=line_dash),
            showlegend=False
        ))
    
    # Add violation zone shading
    if show_violation_zone and controlling_level:
        min_price = df['low'].min()
        fig.add_trace(go.Scatter(
            x=[0, len(df)-1, len(df)-1, 0],
            y=[controlling_level, controlling_level, min_price, min_price],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
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

def create_termination_overview():
    """Create overview page for termination progression"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>USO Trend Termination Progression</title>
        <style>
            body { background-color: #1e1e1e; color: white; font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; color: #ff6666; }
            h2 { color: #ffaa00; }
            .stage { margin: 40px 0; text-align: center; position: relative; }
            .stage iframe { max-width: 100%; height: auto; border: 2px solid #333; }
            .description { margin: 20px 0; font-size: 16px; }
            .stage-nav { margin: 10px 0; }
            .stage-button { 
                background: #dc3545; color: white; border: none; padding: 10px 20px; 
                border-radius: 5px; cursor: pointer; margin: 0 5px; font-weight: bold;
                transition: background 0.3s;
            }
            .stage-button:hover { background: #c82333; }
            .nav-bar {
                text-align: center; margin: 30px 0; padding: 20px;
                background: rgba(0,0,0,0.3); border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <h1>🛑 USO Trend Termination: Step-by-Step Progression</h1>
        
        <div class="nav-bar">
            <h3>Quick Navigation:</h3>
            <button class="stage-button" onclick="location.href='term_stage1_active.html'">Stage 1: Active</button>
            <button class="stage-button" onclick="location.href='term_stage2_peak.html'">Stage 2: Peak</button>
            <button class="stage-button" onclick="location.href='term_stage3_approaching.html'">Stage 3: Approaching</button>
            <button class="stage-button" onclick="location.href='term_stage4_violation.html'">Stage 4: Violation</button>
            <button class="stage-button" onclick="location.href='term_stage5_terminated.html'">Stage 5: Terminated</button>
            <br><br>
            <button onclick="location.href='progression.html'" style="background: #28a745; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-weight: bold;">📈 View Formation</button>
        </div>
        
        <div class="stage">
            <h2>Stage 1: Active Uptrend</h2>
            <iframe src="term_stage1_active.html" width="100%" height="600" frameborder="0"></iframe>
            <div class="description">Uptrend protected by controlling swing at $76.25</div>
            <div class="stage-nav">
                <button class="stage-button" onclick="location.href='term_stage1_active.html'">📊 View Full Stage 1</button>
            </div>
        </div>
        
        <div class="stage">
            <h2>Stage 2: Peak Before Crash</h2>
            <iframe src="term_stage2_peak.html" width="100%" height="600" frameborder="0"></iframe>
            <div class="description">Peak at $83.57 - last high before the crash</div>
            <div class="stage-nav">
                <button class="stage-button" onclick="location.href='term_stage2_peak.html'">📊 View Full Stage 2</button>
            </div>
        </div>
        
        <div class="stage">
            <h2>Stage 3: Approaching Violation</h2>
            <iframe src="term_stage3_approaching.html" width="100%" height="600" frameborder="0"></iframe>
            <div class="description">Price falling toward the critical $76.25 level</div>
            <div class="stage-nav">
                <button class="stage-button" onclick="location.href='term_stage3_approaching.html'">📊 View Full Stage 3</button>
            </div>
        </div>
        
        <div class="stage">
            <h2>Stage 4: VIOLATION!</h2>
            <iframe src="term_stage4_violation.html" width="100%" height="600" frameborder="0"></iframe>
            <div class="description">🛑 $71.97 breaks below $76.25 - UPTREND TERMINATED!</div>
            <div class="stage-nav">
                <button class="stage-button" onclick="location.href='term_stage4_violation.html'">📊 View Full Stage 4</button>
            </div>
        </div>
        
        <div class="stage">
            <h2>Stage 5: Terminated</h2>
            <iframe src="term_stage5_terminated.html" width="100%" height="600" frameborder="0"></iframe>
            <div class="description">❌ Uptrend no longer active - trend terminated</div>
            <div class="stage-nav">
                <button class="stage-button" onclick="location.href='term_stage5_terminated.html'">📊 View Full Stage 5</button>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 50px; color: #888;">
            <p><strong>Key Concept:</strong> Trends terminate when controlling swing is violated</p>
            <p><strong>Termination:</strong> Peak → Crash → Violation of controlling level</p>
        </div>
    </body>
    </html>
    """
    
    with open('trend_progression/termination_overview.html', 'w') as f:
        f.write(html_content)
    
    print(f"📄 Termination overview saved as: trend_progression/termination_overview.html")

def main():
    """Create the termination progression"""
    create_trend_termination_progression()

if __name__ == "__main__":
    main()