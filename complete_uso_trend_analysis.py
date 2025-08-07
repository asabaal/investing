#!/usr/bin/env python3
"""
Complete USO Trend Analysis - Full Dataset Implementation
Finds ALL trend formations and terminations in our complete USO dataset
Uses the corrected breakout origin logic and violation detection
"""

import plotly.graph_objects as go
import plotly.io as pio
from daily_trend_examples import analyze_daily_uso_trends
from simple_trend_progression import find_first_breakout_candle
from trend_termination_progression import find_first_violation_candle
import os
import pandas as pd

# Dark theme
pio.templates.default = "plotly_dark"

def find_all_trend_formations(df, swing_points):
    """
    Find ALL trend formations in the dataset
    Returns list of formation events with proper breakout origins
    """
    
    formations = []
    
    # Convert swing points to easier format
    swing_list = [(sp['index'], sp['type'], sp['price']) for sp in swing_points]
    
    print(f"🔍 Scanning {len(swing_list)} swing points for trend formations...")
    
    # Look for SL → SH → SL patterns (uptrend formations)
    for i in range(len(swing_list) - 2):
        if (swing_list[i][1] == 'LOW' and 
            swing_list[i+1][1] == 'HIGH' and 
            swing_list[i+2][1] == 'LOW'):
            
            sl1_idx, sl1_type, sl1_price = swing_list[i]
            sh1_idx, sh1_type, sh1_price = swing_list[i+1]
            sl2_idx, sl2_type, sl2_price = swing_list[i+2]
            
            # Check for higher low (uptrend condition)
            if sl2_price > sl1_price:
                # Find breakout origin
                breakout_idx = find_first_breakout_candle(df, sl2_idx, sh1_price, 'above')
                
                if breakout_idx:
                    formation = {
                        'type': 'UPTREND',
                        'sl1': {'idx': sl1_idx, 'price': sl1_price},
                        'sh1': {'idx': sh1_idx, 'price': sh1_price}, 
                        'sl2': {'idx': sl2_idx, 'price': sl2_price},
                        'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['high']},
                        'controlling_swing': {'idx': sl1_idx, 'price': sl1_price},
                        'formation_date': df.iloc[breakout_idx]['datetime'],
                        'setup_quality': 'VALID'
                    }
                    formations.append(formation)
    
    # Look for SH → SL → SH patterns (downtrend formations)
    for i in range(len(swing_list) - 2):
        if (swing_list[i][1] == 'HIGH' and 
            swing_list[i+1][1] == 'LOW' and 
            swing_list[i+2][1] == 'HIGH'):
            
            sh1_idx, sh1_type, sh1_price = swing_list[i]
            sl1_idx, sl1_type, sl1_price = swing_list[i+1]
            sh2_idx, sh2_type, sh2_price = swing_list[i+2]
            
            # Check for lower high (downtrend condition)
            if sh2_price < sh1_price:
                # Find breakdown origin
                breakout_idx = find_first_breakout_candle(df, sh2_idx, sl1_price, 'below')
                
                if breakout_idx:
                    formation = {
                        'type': 'DOWNTREND',
                        'sh1': {'idx': sh1_idx, 'price': sh1_price},
                        'sl1': {'idx': sl1_idx, 'price': sl1_price},
                        'sh2': {'idx': sh2_idx, 'price': sh2_price}, 
                        'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['low']},
                        'controlling_swing': {'idx': sh1_idx, 'price': sh1_price},
                        'formation_date': df.iloc[breakout_idx]['datetime'],
                        'setup_quality': 'VALID'
                    }
                    formations.append(formation)
    
    return formations

def filter_overlapping_trends(formations, terminations, df_length):
    """
    Filter overlapping trends to create a mathematical function property.
    When multiple trends overlap, keep only the shortest duration trend.
    
    Args:
        formations: List of trend formation dicts
        terminations: List of trend termination dicts  
        df_length: Length of the dataframe (for active trend calculations)
        
    Returns:
        tuple: (filtered_formations, filtered_terminations)
    """
    
    print(f"🔧 Filtering {len(formations)} formations to eliminate overlaps...")
    
    # Create termination lookup for quick access
    termination_by_formation = {}
    for term in terminations:
        formation_key = id(term['formation'])
        termination_by_formation[formation_key] = term
    
    # Calculate duration and end index for each formation
    formation_ranges = []
    for formation in formations:
        formation_key = id(formation)
        start_idx = formation['breakout']['idx']
        
        # Determine end index
        if formation_key in termination_by_formation:
            end_idx = termination_by_formation[formation_key]['violation_idx']
        else:
            # Active trend goes to end of data
            end_idx = df_length - 1
        
        duration = end_idx - start_idx + 1
        
        formation_ranges.append({
            'formation': formation,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration': duration,
            'formation_key': formation_key
        })
    
    # Sort by start index for processing
    formation_ranges.sort(key=lambda x: x['start_idx'])
    
    # Filter overlapping trends - keep shortest when overlaps occur
    filtered_ranges = []
    
    for current_range in formation_ranges:
        current_start = current_range['start_idx']
        current_end = current_range['end_idx']
        current_duration = current_range['duration']
        
        # Check for overlaps with previously accepted ranges
        should_keep = True
        ranges_to_remove = []
        
        for i, existing_range in enumerate(filtered_ranges):
            existing_start = existing_range['start_idx']
            existing_end = existing_range['end_idx']
            existing_duration = existing_range['duration']
            
            # Check for overlap: ranges overlap if they share any candle indices
            overlap = not (current_end < existing_start or current_start > existing_end)
            
            if overlap:
                # They overlap - keep the shorter duration one
                if current_duration < existing_duration:
                    # Current is shorter - remove the existing one
                    ranges_to_remove.append(i)
                elif current_duration > existing_duration:
                    # Existing is shorter - don't add current
                    should_keep = False
                    break
                else:
                    # Same duration - keep the earlier one (existing)
                    should_keep = False
                    break
        
        # Remove ranges that were superseded by shorter current range
        for remove_idx in reversed(ranges_to_remove):
            filtered_ranges.pop(remove_idx)
        
        # Add current range if it should be kept
        if should_keep:
            filtered_ranges.append(current_range)
    
    # Extract filtered formations and their corresponding terminations
    filtered_formations = []
    filtered_terminations = []
    
    for range_info in filtered_ranges:
        formation = range_info['formation']
        formation_key = range_info['formation_key']
        
        filtered_formations.append(formation)
        
        # Add termination if it exists
        if formation_key in termination_by_formation:
            filtered_terminations.append(termination_by_formation[formation_key])
    
    print(f"✅ Filtered to {len(filtered_formations)} non-overlapping formations")
    print(f"   📉 Removed {len(formations) - len(filtered_formations)} overlapping trends")
    
    # Verify function property
    verify_function_property(filtered_ranges, df_length)
    
    return filtered_formations, filtered_terminations

def verify_function_property(filtered_ranges, df_length):
    """Verify that each candle index has at most one trend state"""
    
    print(f"🔍 Verifying mathematical function property...")
    
    # Create a mapping of candle_index -> trend_count
    candle_trend_count = [0] * df_length
    
    for range_info in filtered_ranges:
        start_idx = range_info['start_idx']
        end_idx = range_info['end_idx']
        
        for candle_idx in range(start_idx, min(end_idx + 1, df_length)):
            candle_trend_count[candle_idx] += 1
    
    # Check for violations
    violations = []
    for i, count in enumerate(candle_trend_count):
        if count > 1:
            violations.append((i, count))
    
    if violations:
        print(f"❌ Function property VIOLATED at {len(violations)} candle indices:")
        for idx, count in violations[:10]:  # Show first 10 violations
            print(f"   Candle {idx}: {count} overlapping trends")
        if len(violations) > 10:
            print(f"   ... and {len(violations) - 10} more violations")
    else:
        print(f"✅ Function property VERIFIED: Each candle has at most one trend state")
        
    # Statistics
    active_candles = sum(1 for count in candle_trend_count if count > 0)
    coverage_pct = (active_candles / df_length) * 100
    
    print(f"📊 Coverage: {active_candles}/{df_length} candles ({coverage_pct:.1f}%) have trend states")

def add_trend_lines(fig, df, formations, terminations):
    """Add trend lines connecting swing points to current candles"""
    
    print(f"🎨 Adding trend lines for {len(formations)} formations...")
    
    # Create a dict to quickly find terminations by formation
    termination_by_formation = {}
    for term in terminations:
        # Use the formation object's id or unique identifier
        formation_key = id(term['formation'])
        termination_by_formation[formation_key] = term
    
    for i, formation in enumerate(formations):
        formation_key = id(formation)
        termination = termination_by_formation.get(formation_key)
        
        # Determine line color and name
        if formation['type'] == 'UPTREND':
            line_color = '#00ff00'  # Green for uptrends
            trend_name = f'Uptrend {i+1}'
        else:
            line_color = '#ff0000'  # Red for downtrends  
            trend_name = f'Downtrend {i+1}'
        
        # Get swing point coordinates
        if formation['type'] == 'UPTREND':
            # For uptrends: SL1 -> SH1 -> SL2 -> Breakout -> Current/Termination
            sl1_idx = formation['sl1']['idx']
            sh1_idx = formation['sh1']['idx']
            sl2_idx = formation['sl2']['idx']
            
            # Key swing points
            swing1_date = df.iloc[sl1_idx]['datetime']
            swing1_price = formation['sl1']['price']
            swing2_date = df.iloc[sh1_idx]['datetime'] 
            swing2_price = formation['sh1']['price']
            swing3_date = df.iloc[sl2_idx]['datetime']
            swing3_price = formation['sl2']['price']
        else:
            # For downtrends: SH1 -> SL1 -> SH2 -> Breakout -> Current/Termination
            sh1_idx = formation['sh1']['idx']
            sl1_idx = formation['sl1']['idx']
            sh2_idx = formation['sh2']['idx']
            
            # Key swing points
            swing1_date = df.iloc[sh1_idx]['datetime']
            swing1_price = formation['sh1']['price']
            swing2_date = df.iloc[sl1_idx]['datetime']
            swing2_price = formation['sl1']['price']
            swing3_date = df.iloc[sh2_idx]['datetime']
            swing3_price = formation['sh2']['price']
        
        # Breakout point
        breakout_idx = formation['breakout']['idx']
        breakout_date = df.iloc[breakout_idx]['datetime']
        breakout_price = formation['breakout']['price']
        
        # End point (termination or last candle)
        if termination:
            end_idx = termination['violation_idx']
            end_date = termination['violation_date']
            end_price = termination['violation_price']
            line_dash = 'solid'  # Terminated trends are solid
        else:
            # Active trend - goes to last candle
            end_idx = len(df) - 1
            end_date = df.iloc[end_idx]['datetime']
            if formation['type'] == 'UPTREND':
                end_price = df.iloc[end_idx]['high']  # Use high for active uptrends
            else:
                end_price = df.iloc[end_idx]['low']   # Use low for active downtrends
            line_dash = 'dash'  # Active trends are dashed
        
        # Draw the trend line from controlling swing point through breakout to end
        if formation['type'] == 'UPTREND':
            # Line from SL1 (controlling swing) through breakout to end
            start_date = swing1_date
            start_price = swing1_price
        else:
            # Line from SH1 (controlling swing) through breakout to end  
            start_date = swing1_date
            start_price = swing1_price
        
        # Add the main trend line
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[start_price, end_price],
            mode='lines',
            line=dict(
                color=line_color,
                width=3,
                dash=line_dash
            ),
            name=trend_name,
            showlegend=True,
            hovertemplate=f'<b>{trend_name}</b><br>' +
                         f'Start: {start_price:.2f}<br>' +
                         f'End: {end_price:.2f}<br>' +
                         f'Status: {"Terminated" if termination else "Active"}<extra></extra>'
        ))
        
        # Add swing point markers for reference
        fig.add_trace(go.Scatter(
            x=[swing1_date, swing2_date, swing3_date],
            y=[swing1_price, swing2_price, swing3_price],
            mode='markers',
            marker=dict(
                color=line_color,
                size=6,
                symbol='circle-open'
            ),
            name=f'{trend_name} Swings',
            showlegend=False,
            hovertemplate=f'<b>{trend_name} Swing Points</b><br>' +
                         f'Price: %{{y:.2f}}<extra></extra>'
        ))
        
        # Add breakout origin marker
        fig.add_trace(go.Scatter(
            x=[breakout_date],
            y=[breakout_price],
            mode='markers',
            marker=dict(
                color=line_color,
                size=10,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            name=f'{trend_name} Origin',
            showlegend=False,
            hovertemplate=f'<b>{trend_name} Origin</b><br>' +
                         f'Breakout: {breakout_price:.2f}<br>' +
                         f'Date: {breakout_date}<extra></extra>'
        ))

def find_all_trend_terminations(df, formations):
    """
    Find ALL trend terminations for the given formations
    """
    
    terminations = []
    
    print(f"🔍 Scanning {len(formations)} formations for terminations...")
    
    for formation in formations:
        controlling_idx = formation['controlling_swing']['idx']
        controlling_price = formation['controlling_swing']['price']
        formation_idx = formation['breakout']['idx']
        
        # Look for violation after formation
        if formation['type'] == 'UPTREND':
            # Find violation below controlling swing
            violation_idx = find_first_violation_candle(df, formation_idx + 5, controlling_price, 'below')
        else:
            # Find violation above controlling swing  
            violation_idx = find_first_violation_candle(df, formation_idx + 5, controlling_price, 'above')
        
        if violation_idx:
            termination = {
                'formation': formation,
                'violation_idx': violation_idx,
                'violation_price': df.iloc[violation_idx]['low'] if formation['type'] == 'UPTREND' else df.iloc[violation_idx]['high'],
                'violation_date': df.iloc[violation_idx]['datetime'],
                'trend_duration': violation_idx - formation_idx,
                'violation_size': abs(controlling_price - (df.iloc[violation_idx]['low'] if formation['type'] == 'UPTREND' else df.iloc[violation_idx]['high']))
            }
            terminations.append(termination)
    
    return terminations

def create_complete_uso_analysis():
    """Create complete USO trend analysis with all formations and terminations"""
    
    print("🎯 COMPLETE USO TREND ANALYSIS - FULL DATASET")
    print("=" * 70)
    
    # Get FULL data - not just 100 candles
    from uso_supply_demand_visualizer import SupplyDemandVisualizer
    from swing_point_detector import SwingPointDetector
    
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    
    # Convert to format for swing detector  
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # The datetime is in the 'datetime' column from reset_index()
    # Keep it as proper datetime for the range selector to work
    if 'datetime' not in df.columns:
        print(f"📋 Available columns: {list(df.columns)}")
        raise ValueError("No datetime column found")
    
    # Get swing points from FULL dataset
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    print(f"📊 Dataset: {len(df)} daily candles")
    
    # Safe datetime printing
    start_dt = df.iloc[0]['datetime']
    end_dt = df.iloc[-1]['datetime']
    if hasattr(start_dt, 'strftime'):
        print(f"📅 Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    else:
        print(f"📅 Period: {start_dt} to {end_dt}")
    
    print(f"📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Find all formations
    formations = find_all_trend_formations(df, swing_points)
    print(f"\n✅ Found {len(formations)} trend formations:")
    
    uptrends = [f for f in formations if f['type'] == 'UPTREND']
    downtrends = [f for f in formations if f['type'] == 'DOWNTREND']
    
    print(f"   📈 Uptrends: {len(uptrends)}")
    print(f"   📉 Downtrends: {len(downtrends)}")
    
    # Find all terminations
    terminations = find_all_trend_terminations(df, formations)
    print(f"\n✅ Found {len(terminations)} trend terminations:")
    
    terminated_uptrends = [t for t in terminations if t['formation']['type'] == 'UPTREND']
    terminated_downtrends = [t for t in terminations if t['formation']['type'] == 'DOWNTREND']
    
    print(f"   🛑 Terminated uptrends: {len(terminated_uptrends)}")
    print(f"   🛑 Terminated downtrends: {len(terminated_downtrends)}")
    
    # Apply overlap filtering to create mathematical function property
    print(f"\n🔧 APPLYING TREND FILTERING FOR MATHEMATICAL FUNCTION PROPERTY:")
    print("=" * 60)
    
    filtered_formations, filtered_terminations = filter_overlapping_trends(formations, terminations, len(df))
    
    # Update counts after filtering
    filtered_uptrends = [f for f in filtered_formations if f['type'] == 'UPTREND']
    filtered_downtrends = [f for f in filtered_formations if f['type'] == 'DOWNTREND']
    filtered_terminated_up = [t for t in filtered_terminations if t['formation']['type'] == 'UPTREND']
    filtered_terminated_down = [t for t in filtered_terminations if t['formation']['type'] == 'DOWNTREND']
    
    print(f"\n✅ FILTERED RESULTS:")
    print(f"   📈 Uptrends: {len(filtered_uptrends)} (was {len(uptrends)})")
    print(f"   📉 Downtrends: {len(filtered_downtrends)} (was {len(downtrends)})")
    print(f"   🛑 Terminated uptrends: {len(filtered_terminated_up)} (was {len(terminated_uptrends)})")
    print(f"   🛑 Terminated downtrends: {len(filtered_terminated_down)} (was {len(terminated_downtrends)})")
    
    # Show detailed analysis
    print(f"\n📋 DETAILED FORMATION ANALYSIS:")
    for i, formation in enumerate(formations[:10]):  # Show first 10
        ftype = formation['type']
        formation_date = formation['formation_date']
        date = formation_date.strftime('%m/%d/%Y') if hasattr(formation_date, 'strftime') else str(formation_date)
        
        if ftype == 'UPTREND':
            sl1_price = formation['sl1']['price']
            sh1_price = formation['sh1']['price'] 
            sl2_price = formation['sl2']['price']
            breakout_price = formation['breakout']['price']
            print(f"   {i+1:2d}. {ftype} ({date}): SL1=${sl1_price:.2f} → SH1=${sh1_price:.2f} → SL2=${sl2_price:.2f} → ORIGIN=${breakout_price:.2f}")
        else:
            sh1_price = formation['sh1']['price']
            sl1_price = formation['sl1']['price']
            sh2_price = formation['sh2']['price'] 
            breakout_price = formation['breakout']['price']
            print(f"   {i+1:2d}. {ftype} ({date}): SH1=${sh1_price:.2f} → SL1=${sl1_price:.2f} → SH2=${sh2_price:.2f} → ORIGIN=${breakout_price:.2f}")
    
    if len(formations) > 10:
        print(f"   ... and {len(formations) - 10} more formations")
    
    # Show detailed termination analysis
    print(f"\n📋 DETAILED TERMINATION ANALYSIS:")
    for i, termination in enumerate(terminations[:10]):  # Show first 10
        formation = termination['formation']
        ftype = formation['type']
        formation_dt = formation['formation_date']
        violation_dt = termination['violation_date']
        formation_date = formation_dt.strftime('%m/%d') if hasattr(formation_dt, 'strftime') else str(formation_dt)
        violation_date = violation_dt.strftime('%m/%d') if hasattr(violation_dt, 'strftime') else str(violation_dt)
        controlling_price = termination['formation']['controlling_swing']['price']
        violation_price = termination['violation_price']
        duration = termination['trend_duration']
        violation_size = termination['violation_size']
        
        print(f"   {i+1:2d}. {ftype} ({formation_date}→{violation_date}): Control=${controlling_price:.2f} → Violation=${violation_price:.2f} (${violation_size:.2f} break, {duration} candles)")
    
    if len(terminations) > 10:
        print(f"   ... and {len(terminations) - 10} more terminations")
    
    # Create comprehensive visualization with filtered data
    create_complete_chart(df, filtered_formations, filtered_terminations)
    
    # Create summary statistics with filtered data  
    create_trend_statistics(filtered_formations, filtered_terminations)
    
    return filtered_formations, filtered_terminations

def create_complete_chart(df, formations, terminations):
    """Create comprehensive chart showing all formations and terminations with trend lines"""
    
    print(f"\n📊 Creating comprehensive USO trend chart with trend lines...")
    
    fig = go.Figure()
    
    # Add candlestick data
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='USO Daily',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add trend lines instead of markers
    add_trend_lines(fig, df, formations, terminations)
    
    # Update layout with range selector
    fig.update_layout(
        title=dict(
            text=f"USO Mathematical Function Trends: {len(formations)} Non-Overlapping Trend Lines",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title='Date', 
            gridcolor='rgba(100,100,100,0.2)',
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            type="date"
        ),
        yaxis=dict(
            title='Price ($)', 
            gridcolor='rgba(100,100,100,0.2)',
            fixedrange=False
        ),
        template='plotly_dark',
        height=800,
        width=1400,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    # Save chart with custom price slider
    filename = 'complete_uso_trend_analysis.html'
    
    # Get the basic HTML
    html_string = fig.to_html(include_plotlyjs='cdn')
    
    # Add custom price range slider
    enhanced_html = add_price_range_slider(html_string, df['low'].min(), df['high'].max())
    
    with open(filename, 'w') as f:
        f.write(enhanced_html)
        
    print(f"✅ Comprehensive chart with price slider saved as: {filename}")

def add_price_range_slider(html_string, min_price, max_price):
    """Add a custom price range slider to the Plotly chart"""
    
    # Add price range controls above the chart
    custom_controls = f'''
    <div style="background: rgba(0,0,0,0.3); padding: 15px; margin: 20px 0; 
                border-radius: 10px; border: 1px solid #333; max-width: 1400px;">
        <h3 style="margin: 0 0 15px 0; color: #00ff88; text-align: center;">Price Range Filter</h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 200px; gap: 20px; align-items: end;">
            <div>
                <label style="color: white; display: block; margin-bottom: 5px;">
                    Min Price: $<span id="minPriceValue">{min_price:.2f}</span>
                </label>
                <input type="range" id="minPriceSlider" 
                       min="{min_price:.2f}" max="{max_price:.2f}" 
                       value="{min_price:.2f}" step="0.1"
                       style="width: 100%;">
            </div>
            
            <div>
                <label style="color: white; display: block; margin-bottom: 5px;">
                    Max Price: $<span id="maxPriceValue">{max_price:.2f}</span>
                </label>
                <input type="range" id="maxPriceSlider" 
                       min="{min_price:.2f}" max="{max_price:.2f}" 
                       value="{max_price:.2f}" step="0.1"
                       style="width: 100%;">
            </div>
            
            <div>
                <button onclick="resetPriceRange()" 
                        style="width: 100%; padding: 10px; background: #007bff; color: white; 
                               border: none; border-radius: 5px; cursor: pointer;">
                    Reset Price Range
                </button>
            </div>
        </div>
    </div>

    <script>
        // Wait for the plot to be ready
        document.addEventListener('DOMContentLoaded', function() {{
            let minPriceSlider = document.getElementById('minPriceSlider');
            let maxPriceSlider = document.getElementById('maxPriceSlider');
            let minPriceValue = document.getElementById('minPriceValue');
            let maxPriceValue = document.getElementById('maxPriceValue');
            
            function updatePriceRange() {{
                let minPrice = parseFloat(minPriceSlider.value);
                let maxPrice = parseFloat(maxPriceSlider.value);
                
                // Ensure min <= max
                if (minPrice > maxPrice) {{
                    maxPrice = minPrice + 0.1;
                    maxPriceSlider.value = maxPrice;
                }}
                
                minPriceValue.textContent = minPrice.toFixed(2);
                maxPriceValue.textContent = maxPrice.toFixed(2);
                
                // Find the plotly div (it might have a different ID)
                let plotDiv = document.querySelector('[id^="plotly-div"], .plotly-graph-div');
                if (plotDiv) {{
                    Plotly.relayout(plotDiv, {{
                        'yaxis.range': [minPrice, maxPrice]
                    }});
                }}
            }}
            
            function resetPriceRange() {{
                minPriceSlider.value = {min_price:.2f};
                maxPriceSlider.value = {max_price:.2f};
                updatePriceRange();
            }}
            
            // Make resetPriceRange globally available
            window.resetPriceRange = resetPriceRange;
            
            minPriceSlider.addEventListener('input', updatePriceRange);
            maxPriceSlider.addEventListener('input', updatePriceRange);
        }});
    </script>
</body>'''
    
    # Insert the price and date controls after the body opening tag
    controls_panel = f'''
    <div style="background: rgba(0,0,0,0.3); padding: 20px; margin: 20px auto; 
                border-radius: 10px; border: 1px solid #333; max-width: 1400px;">
        
        <!-- Price Range Filter -->
        <div style="margin-bottom: 30px;">
            <h3 style="margin: 0 0 15px 0; color: #00ff88; text-align: center;">Price Range Filter</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 200px; gap: 20px; align-items: end;">
                <div>
                    <label style="color: white; display: block; margin-bottom: 5px;">
                        Min Price: $<span id="minPriceValue">{min_price:.2f}</span>
                    </label>
                    <input type="range" id="minPriceSlider" 
                           min="{min_price:.2f}" max="{max_price:.2f}" 
                           value="{min_price:.2f}" step="0.1"
                           style="width: 100%;">
                </div>
                
                <div>
                    <label style="color: white; display: block; margin-bottom: 5px;">
                        Max Price: $<span id="maxPriceValue">{max_price:.2f}</span>
                    </label>
                    <input type="range" id="maxPriceSlider" 
                           min="{min_price:.2f}" max="{max_price:.2f}" 
                           value="{max_price:.2f}" step="0.1"
                           style="width: 100%;">
                </div>
                
                <div>
                    <button onclick="resetPriceRange()" 
                            style="width: 100%; padding: 10px; background: #007bff; color: white; 
                                   border: none; border-radius: 5px; cursor: pointer;">
                        Reset Price Range
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Date Range Zoom Control -->
        <div>
            <h3 style="margin: 0 0 15px 0; color: #ffaa00; text-align: center;">Date Range Zoom Control</h3>
            
            <div style="display: grid; grid-template-columns: 2fr 1fr 200px; gap: 20px; align-items: end;">
                <div>
                    <label style="color: white; display: block; margin-bottom: 5px;">
                        Range Selector Zoom: <span id="zoomLevelValue">100%</span> of timeline
                    </label>
                    <input type="range" id="dateZoomSlider" 
                           min="1" max="100" value="100" step="1"
                           style="width: 100%;">
                    <div style="font-size: 12px; color: #ccc; margin-top: 5px;">
                        ← More Precise | Less Precise →
                    </div>
                </div>
                
                <div>
                    <label style="color: white; display: block; margin-bottom: 5px;">
                        Timeline Position: <span id="timelinePosition">50%</span>
                    </label>
                    <input type="range" id="timelinePositionSlider" 
                           min="0" max="100" value="50" step="1"
                           style="width: 100%;">
                </div>
                
                <div>
                    <button onclick="resetDateZoom()" 
                            style="width: 100%; padding: 10px; background: #ff8c00; color: white; 
                                   border: none; border-radius: 5px; cursor: pointer;">
                        Reset Date Zoom
                    </button>
                </div>
            </div>
        </div>
    </div>
'''

    enhanced_script = '''
    <script>
        // Wait for the plot to be ready
        document.addEventListener('DOMContentLoaded', function() {
            // Wait a bit more for Plotly to fully initialize
            setTimeout(function() {
                // Price control elements
                let minPriceSlider = document.getElementById('minPriceSlider');
                let maxPriceSlider = document.getElementById('maxPriceSlider');
                let minPriceValue = document.getElementById('minPriceValue');
                let maxPriceValue = document.getElementById('maxPriceValue');
                
                // Date zoom control elements
                let dateZoomSlider = document.getElementById('dateZoomSlider');
                let timelinePositionSlider = document.getElementById('timelinePositionSlider');
                let zoomLevelValue = document.getElementById('zoomLevelValue');
                let timelinePosition = document.getElementById('timelinePosition');
                
                // Get the plotly div
                let plotDiv = document.querySelector('div[id*="plotly"]') || document.getElementsByClassName('plotly-graph-div')[0];
                
                function updatePriceRange() {
                    let minPrice = parseFloat(minPriceSlider.value);
                    let maxPrice = parseFloat(maxPriceSlider.value);
                    
                    // Ensure min <= max
                    if (minPrice > maxPrice) {
                        maxPrice = minPrice + 0.1;
                        maxPriceSlider.value = maxPrice;
                    }
                    
                    minPriceValue.textContent = minPrice.toFixed(2);
                    maxPriceValue.textContent = maxPrice.toFixed(2);
                    
                    if (plotDiv) {
                        Plotly.relayout(plotDiv, {
                            'yaxis.range': [minPrice, maxPrice]
                        });
                    }
                }
                
                function updateDateZoom() {
                    let zoomPercent = parseInt(dateZoomSlider.value);
                    let position = parseInt(timelinePositionSlider.value);
                    
                    zoomLevelValue.textContent = zoomPercent + '%';
                    timelinePosition.textContent = position + '%';
                    
                    if (plotDiv) {
                        // Get the full data range from the plot
                        let plotlyData = plotDiv.data;
                        let xData = plotlyData[0].x; // Get x-axis data (dates)
                        
                        if (xData && xData.length > 0) {
                            let totalDataPoints = xData.length;
                            
                            // Calculate the visible window size based on zoom
                            let visiblePoints = Math.max(1, Math.floor(totalDataPoints * zoomPercent / 100));
                            
                            // Calculate the start position
                            let maxStartPos = totalDataPoints - visiblePoints;
                            let startPos = Math.floor(maxStartPos * position / 100);
                            let endPos = startPos + visiblePoints - 1;
                            
                            // Get the actual date values for start and end
                            let startDate = xData[startPos];
                            let endDate = xData[endPos];
                            
                            // Update the range slider's visible range
                            Plotly.relayout(plotDiv, {
                                'xaxis.rangeslider.range': [startDate, endDate]
                            });
                        }
                    }
                }
                
                function resetPriceRange() {
                    minPriceSlider.value = ''' + str(min_price) + ''';
                    maxPriceSlider.value = ''' + str(max_price) + ''';
                    updatePriceRange();
                }
                
                function resetDateZoom() {
                    dateZoomSlider.value = 100;
                    timelinePositionSlider.value = 50;
                    
                    zoomLevelValue.textContent = '100%';
                    timelinePosition.textContent = '50%';
                    
                    // Reset to full data range
                    if (plotDiv) {
                        let plotlyData = plotDiv.data;
                        let xData = plotlyData[0].x;
                        
                        if (xData && xData.length > 0) {
                            let startDate = xData[0];
                            let endDate = xData[xData.length - 1];
                            
                            Plotly.relayout(plotDiv, {
                                'xaxis.rangeslider.range': [startDate, endDate]
                            });
                        }
                    }
                }
                
                // Make functions globally available
                window.resetPriceRange = resetPriceRange;
                window.resetDateZoom = resetDateZoom;
                
                // Event listeners
                minPriceSlider.addEventListener('input', updatePriceRange);
                maxPriceSlider.addEventListener('input', updatePriceRange);
                dateZoomSlider.addEventListener('input', updateDateZoom);
                timelinePositionSlider.addEventListener('input', updateDateZoom);
                
            }, 1000);
        });
    </script>
'''

    # Insert controls panel after body tag and script before closing body tag
    enhanced_html = html_string.replace('<body>', '<body>' + controls_panel)
    enhanced_html = enhanced_html.replace('</body>', enhanced_script + '</body>')
    
    return enhanced_html

def create_chart_data_for_js(df, formations, terminations):
    """Prepare data structure for JavaScript filtering"""
    
    # Convert dataframe to JavaScript-friendly format
    candle_data = []
    for i, row in df.iterrows():
        candle_data.append({
            'datetime': float(i),  # Use index as datetime since datetime is float64
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    
    # Prepare formation data
    formation_data = []
    for i, formation in enumerate(formations):
        formation_idx = formation['breakout']['idx']
        termination_idx = None
        
        # Find if this formation has a termination
        for term in terminations:
            if term['formation'] == formation:
                termination_idx = term['violation_idx']
                break
        
        formation_data.append({
            'id': i,
            'type': formation['type'],
            'formation_idx': formation_idx,
            'formation_price': formation['breakout']['price'],
            'termination_idx': termination_idx,
            'termination_price': terminations[i]['violation_price'] if i < len(terminations) and terminations[i]['formation'] == formation else None,
            'controlling_price': formation['controlling_swing']['price'],
            'sl1_idx': formation.get('sl1', {}).get('idx'),
            'sh1_idx': formation.get('sh1', {}).get('idx') if formation['type'] == 'UPTREND' else formation.get('sl1', {}).get('idx'),
            'sl2_idx': formation.get('sl2', {}).get('idx') if formation['type'] == 'UPTREND' else formation.get('sh2', {}).get('idx')
        })
    
    return {
        'candles': candle_data,
        'formations': formation_data,
        'total_candles': len(df)
    }

def create_interactive_trend_chart(chart_data, total_formations, total_terminations):
    """Create HTML file with interactive range selector and trend filtering"""
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Interactive USO Trend Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js"></script>
    <style>
        body {{ 
            background-color: #1e1e1e; 
            color: white; 
            font-family: Arial, sans-serif; 
            margin: 20px; 
        }}
        .control-panel {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }}
        .range-buttons {{
            text-align: center;
            margin: 15px 0;
        }}
        .range-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }}
        .range-btn:hover {{ background: #0056b3; }}
        .range-btn.active {{ background: #28a745; }}
        
        .stats-panel {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: rgba(0,100,0,0.1);
            border: 2px solid #00ff88;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #00ff88; }}
        .stat-label {{ font-size: 14px; color: #ccc; }}
        
        #chart {{ margin-top: 20px; }}
        h1 {{ text-align: center; color: #00ff88; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>🎯 Interactive USO Trend Analysis</h1>
    
    <div class="control-panel">
        <h3>📅 Select Time Range:</h3>
        <div class="range-buttons">
            <button class="range-btn" onclick="setRange('1M')">1M</button>
            <button class="range-btn" onclick="setRange('3M')">3M</button>
            <button class="range-btn" onclick="setRange('6M')">6M</button>
            <button class="range-btn" onclick="setRange('1Y')">1Y</button>
            <button class="range-btn" onclick="setRange('2Y')">2Y</button>
            <button class="range-btn active" onclick="setRange('ALL')">ALL</button>
        </div>
        
        <div class="stats-panel">
            <div class="stat-box">
                <div class="stat-number" id="active-formations">{total_formations}</div>
                <div class="stat-label">Active Formations</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="active-uptrends">-</div>
                <div class="stat-label">Active Uptrends</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="active-downtrends">-</div>
                <div class="stat-label">Active Downtrends</div>
            </div>
        </div>
    </div>
    
    <div id="chart"></div>
    
    <script>
        // Chart data from Python
        const chartData = {chart_data};
        let currentRange = 'ALL';
        let currentRangeStart = 0;
        let currentRangeEnd = chartData.total_candles - 1;
        
        function setRange(period) {{
            // Update active button
            document.querySelectorAll('.range-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            currentRange = period;
            const totalCandles = chartData.total_candles;
            
            switch(period) {{
                case '1M':
                    currentRangeStart = Math.max(0, totalCandles - 21);
                    break;
                case '3M':
                    currentRangeStart = Math.max(0, totalCandles - 63);
                    break;
                case '6M':
                    currentRangeStart = Math.max(0, totalCandles - 126);
                    break;
                case '1Y':
                    currentRangeStart = Math.max(0, totalCandles - 252);
                    break;
                case '2Y':
                    currentRangeStart = Math.max(0, totalCandles - 504);
                    break;
                default:
                    currentRangeStart = 0;
            }}
            currentRangeEnd = totalCandles - 1;
            
            updateChart();
        }}
        
        function isFormationActiveInRange(formation) {{
            const formationIdx = formation.formation_idx;
            const terminationIdx = formation.termination_idx;
            
            // Formation must start before or within range
            if (formationIdx > currentRangeEnd) return false;
            
            // If terminated, termination must be after range start
            if (terminationIdx !== null) {{
                return terminationIdx >= currentRangeStart;
            }}
            
            // If not terminated, formation must start before range end
            return formationIdx <= currentRangeEnd;
        }}
        
        function updateChart() {{
            // Filter data for current range
            const rangeCandles = chartData.candles.slice(currentRangeStart, currentRangeEnd + 1);
            const activeFormations = chartData.formations.filter(isFormationActiveInRange);
            
            // Separate uptrends and downtrends
            const uptrends = activeFormations.filter(f => f.type === 'UPTREND');
            const downtrends = activeFormations.filter(f => f.type === 'DOWNTREND');
            
            // Update statistics
            document.getElementById('active-formations').textContent = activeFormations.length;
            document.getElementById('active-uptrends').textContent = uptrends.length;
            document.getElementById('active-downtrends').textContent = downtrends.length;
            
            // Create traces
            const traces = [];
            
            // Candlestick trace
            traces.push({{
                type: 'candlestick',
                x: rangeCandles.map((_, i) => currentRangeStart + i),
                open: rangeCandles.map(c => c.open),
                high: rangeCandles.map(c => c.high),
                low: rangeCandles.map(c => c.low),
                close: rangeCandles.map(c => c.close),
                name: 'USO Daily',
                increasing: {{ line: {{ color: '#00ff88' }} }},
                decreasing: {{ line: {{ color: '#ff4444' }} }}
            }});
            
            // Uptrend formations
            if (uptrends.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: uptrends.map(f => f.formation_idx),
                    y: uptrends.map(f => f.formation_price),
                    marker: {{
                        color: '#00ff00',
                        size: 12,
                        symbol: 'triangle-up'
                    }},
                    text: uptrends.map((_, i) => `UP${{i+1}}`),
                    textposition: 'top center',
                    textfont: {{ size: 10, color: 'white' }},
                    name: 'Uptrend Formations'
                }});
            }}
            
            // Downtrend formations  
            if (downtrends.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: downtrends.map(f => f.formation_idx),
                    y: downtrends.map(f => f.formation_price),
                    marker: {{
                        color: '#ff0000',
                        size: 12,
                        symbol: 'triangle-down'
                    }},
                    text: downtrends.map((_, i) => `DN${{i+1}}`),
                    textposition: 'bottom center',
                    textfont: {{ size: 10, color: 'white' }},
                    name: 'Downtrend Formations'
                }});
            }}
            
            // Termination markers
            const terminatedInRange = activeFormations.filter(f => 
                f.termination_idx !== null && 
                f.termination_idx >= currentRangeStart && 
                f.termination_idx <= currentRangeEnd
            );
            
            if (terminatedInRange.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: terminatedInRange.map(f => f.termination_idx),
                    y: terminatedInRange.map(f => f.termination_price),
                    marker: {{
                        color: '#ff6600',
                        size: 10,
                        symbol: 'x'
                    }},
                    text: terminatedInRange.map(() => 'END'),
                    textposition: 'middle center',
                    textfont: {{ size: 8, color: '#ff6600' }},
                    name: 'Terminations'
                }});
            }}
            
            const layout = {{
                title: {{
                    text: `USO Trends (${{currentRange}}): ${{activeFormations.length}} Active Formations`,
                    font: {{ size: 18, color: 'white' }},
                    x: 0.5
                }},
                xaxis: {{
                    title: 'Time Period',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    range: [currentRangeStart, currentRangeEnd]
                }},
                yaxis: {{
                    title: 'Price ($)',
                    gridcolor: 'rgba(100,100,100,0.2)'
                }},
                template: 'plotly_dark',
                height: 800,
                showlegend: true,
                legend: {{
                    x: 0.02, y: 0.98,
                    bgcolor: 'rgba(0,0,0,0.8)',
                    bordercolor: 'white',
                    borderwidth: 1
                }},
                plot_bgcolor: '#1e1e1e',
                paper_bgcolor: '#1e1e1e'
            }};
            
            Plotly.newPlot('chart', traces, layout);
        }}
        
        // Initialize chart
        updateChart();
    </script>
</body>
</html>'''
    
    # Save the HTML file
    filename = 'complete_uso_trend_analysis_interactive.html'
    with open(filename, 'w') as f:
        f.write(html_content)

def create_trend_statistics(formations, terminations):
    """Create detailed statistics about trend formations and terminations"""
    
    print(f"\n📊 TREND STATISTICS SUMMARY:")
    print("=" * 50)
    
    if formations:
        # Formation statistics
        uptrends = [f for f in formations if f['type'] == 'UPTREND']
        downtrends = [f for f in formations if f['type'] == 'DOWNTREND']
        
        print(f"📈 FORMATION STATS:")
        print(f"   Total formations: {len(formations)}")
        print(f"   Uptrends: {len(uptrends)} ({len(uptrends)/len(formations)*100:.1f}%)")
        print(f"   Downtrends: {len(downtrends)} ({len(downtrends)/len(formations)*100:.1f}%)")
        
        # Uptrend formation analysis
        if uptrends:
            uptrend_setups = []
            for ut in uptrends:
                sl1 = ut['sl1']['price']
                sh1 = ut['sh1']['price']
                sl2 = ut['sl2']['price']
                rally_size = sh1 - sl1
                higher_low_size = sl2 - sl1
                uptrend_setups.append({
                    'rally_size': rally_size,
                    'higher_low_size': higher_low_size,
                    'higher_low_pct': higher_low_size / rally_size if rally_size > 0 else 0
                })
            
            avg_rally = sum(s['rally_size'] for s in uptrend_setups) / len(uptrend_setups)
            avg_higher_low = sum(s['higher_low_size'] for s in uptrend_setups) / len(uptrend_setups)
            avg_hl_pct = sum(s['higher_low_pct'] for s in uptrend_setups) / len(uptrend_setups)
            
            print(f"   Uptrend avg rally size: ${avg_rally:.2f}")
            print(f"   Uptrend avg higher low: ${avg_higher_low:.2f}")
            print(f"   Higher low avg %: {avg_hl_pct*100:.1f}% of rally")
    
    if terminations:
        # Termination statistics  
        terminated_up = [t for t in terminations if t['formation']['type'] == 'UPTREND']
        terminated_down = [t for t in terminations if t['formation']['type'] == 'DOWNTREND']
        
        print(f"\n🛑 TERMINATION STATS:")
        print(f"   Total terminations: {len(terminations)}")
        print(f"   Terminated uptrends: {len(terminated_up)}")
        print(f"   Terminated downtrends: {len(terminated_down)}")
        
        if terminated_up:
            avg_duration = sum(t['trend_duration'] for t in terminated_up) / len(terminated_up)
            avg_violation = sum(t['violation_size'] for t in terminated_up) / len(terminated_up)
            print(f"   Uptrend avg duration: {avg_duration:.1f} candles")
            print(f"   Uptrend avg violation: ${avg_violation:.2f}")
        
        success_rate = (len(formations) - len(terminations)) / len(formations) * 100 if formations else 0
        print(f"\n📊 TREND SUCCESS RATE: {success_rate:.1f}% still active")

def main():
    """Run complete USO trend analysis"""
    formations, terminations = create_complete_uso_analysis()
    
    print(f"\n🎯 MATHEMATICAL FUNCTION ANALYSIS COMPLETE!")
    print(f"📁 Open 'complete_uso_trend_analysis.html' to see filtered non-overlapping trends")
    print(f"📈 {len(formations)} non-overlapping trend lines (mathematical function property)")  
    print(f"🛑 {len(terminations)} corresponding terminations")
    print(f"✅ Each candle index has at most one trend state")
    
    return formations, terminations

if __name__ == "__main__":
    main()