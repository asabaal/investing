#!/usr/bin/env python3
"""
Formation Detection Challenges
Demonstrates why our current system struggles with real market formations
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from market_data_database import MarketDataDatabase

pio.templates.default = 'plotly_dark'

def demonstrate_scale_problem():
    """
    Show why formations are hard to detect with single-candle focus
    """
    print("🎯 FORMATION DETECTION CHALLENGES")
    print("=" * 60)
    
    db = MarketDataDatabase()
    
    # Get TSLA data for the RBD formation period
    df = db.get_data('TSLA', '2023-12-20', '2024-01-15', 'daily')
    
    print(f"📊 Analyzing {len(df)} daily candles")
    print("\n🔍 CHALLENGE 1: Single Candle Analysis")
    print("-" * 40)
    
    # Show what our current system sees vs the big picture
    base_period = df['2023-12-28':'2024-01-05']
    
    print("What our system currently analyzes:")
    for i, (date, row) in enumerate(base_period.iterrows()):
        candle_type = "🟢 Bullish" if row['Unadjusted_Close'] > row['Open'] else "🔴 Bearish"
        print(f"  {date.strftime('%m/%d')}: ${row['Open']:.0f} → ${row['Unadjusted_Close']:.0f} {candle_type}")
    
    print(f"\n❌ Problem: System sees {len(base_period)} individual candles")
    print("✅ Human sees: Consolidation zone at $245-251 range")
    
    print("\n🔍 CHALLENGE 2: Formation Boundary Detection")
    print("-" * 40)
    
    # Show how formation boundaries are ambiguous
    print("Where does the 'base' actually start and end?")
    print("Peak was Dec 28, but consolidation continues...")
    print("Drop starts Jan 5, but when does base truly end?")
    print()
    print("💡 This requires pattern recognition, not candle counting")
    
    print("\n🔍 CHALLENGE 3: Context vs Formation")  
    print("-" * 40)
    
    print("Our current rules:")
    print("  • Pre-formation context must NOT be part of legs")
    print("  • Formation must have clear boundaries") 
    print("  • Post-formation context terminates legs")
    print()
    print("Real market reality:")
    print("  • Rally blends into consolidation")
    print("  • Consolidation gradually becomes decline")
    print("  • No clear 'formation ends here' moment")
    
    return df

def show_what_we_need():
    """
    Demonstrate what kind of system could detect this
    """
    print("\n🎯 WHAT WE NEED FOR REAL DETECTION")
    print("=" * 60)
    
    print("🔧 Multi-Scale Analysis:")
    print("  1. Long-term (months): Identify major swings")
    print("  2. Medium-term (weeks): Find consolidation zones") 
    print("  3. Short-term (days): Detect micro-patterns")
    print("  4. Entry-term (hours): Time precise entries")
    
    print("\n🧠 Pattern Recognition vs Rule-Based:")
    print("  Current: 'Base must be exactly N candles'")
    print("  Needed: 'Base is a consolidation phase'")
    print("  Current: 'Formation must have exact boundaries'")
    print("  Needed: 'Formation is a recognizable pattern'")
    
    print("\n📊 Adaptive Zone Definition:")
    print("  Current: Fixed zone high/low")
    print("  Needed: Dynamic zone based on price action")
    print("  Current: Binary (in zone / not in zone)")
    print("  Needed: Probabilistic (strength of zone)")
    
    print("\n⏰ Time-Aware Analysis:")
    print("  Current: Static snapshot scoring")
    print("  Needed: Dynamic scoring as time progresses")
    print("  Current: Formation either exists or doesn't")
    print("  Needed: Formation develops over time")

def simulate_improved_detection():
    """
    Simulate what an improved system might detect
    """
    print("\n🎯 SIMULATED IMPROVED DETECTION")
    print("=" * 60)
    
    db = MarketDataDatabase()
    df = db.get_data('TSLA', '2023-11-01', '2024-04-01', 'daily')
    
    # Simulate multi-scale analysis
    print("🔍 Step 1: Swing Analysis")
    major_high = df['2023-12-20':'2024-01-05']['High'].max()
    major_low = df['2024-01-15':'2024-02-15']['Low'].min()
    print(f"  Major swing high: ${major_high:.2f}")
    print(f"  Major swing low: ${major_low:.2f}")
    print(f"  Swing range: ${major_high - major_low:.2f}")
    
    print("\n🔍 Step 2: Zone Identification")
    # Find consolidation around the high
    high_period = df['2023-12-25':'2024-01-08']
    zone_high = high_period['High'].max()
    zone_low = high_period['Low'].min()
    
    # Adaptive zone sizing based on volatility
    volatility = high_period['High'].std()
    adaptive_zone_size = min(volatility * 2, (zone_high - zone_low) * 0.8)
    
    adaptive_zone_high = zone_high
    adaptive_zone_low = zone_high - adaptive_zone_size
    
    print(f"  Raw consolidation: ${zone_low:.2f} - ${zone_high:.2f}")
    print(f"  Adaptive zone: ${adaptive_zone_low:.2f} - ${adaptive_zone_high:.2f}")
    print(f"  Zone size: ${adaptive_zone_size:.2f} (based on volatility)")
    
    print("\n🔍 Step 3: Formation Classification")
    # Analyze the pattern shape
    pre_peak = df['2023-11-01':'2023-12-28']
    post_peak = df['2024-01-05':'2024-02-15']
    
    rally_strength = (major_high - pre_peak['Low'].min()) / pre_peak['Low'].min()
    drop_strength = (major_high - major_low) / major_high
    
    print(f"  Rally strength: {rally_strength*100:.1f}%")
    print(f"  Drop strength: {drop_strength*100:.1f}%")
    
    if rally_strength > 0.2 and drop_strength > 0.2:
        formation_type = "RBD (Rally-Base-Drop)"
        confidence = "High"
    else:
        formation_type = "Unclear"
        confidence = "Low"
        
    print(f"  Formation type: {formation_type}")
    print(f"  Confidence: {confidence}")
    
    print("\n🔍 Step 4: Dynamic Scoring")
    current_price = df['Close'].iloc[-1]
    distance_to_zone = abs(current_price - ((adaptive_zone_high + adaptive_zone_low) / 2))
    distance_pct = distance_to_zone / current_price
    
    # Time decay factor
    days_since_formation = 30  # Approximate
    freshness_factor = max(0.5, 1.0 - (days_since_formation / 90))
    
    base_score = 7.5  # Base formation score
    distance_penalty = min(2.0, distance_pct * 10)  # Penalty for being far away
    time_penalty = (1.0 - freshness_factor) * 2.0
    
    dynamic_score = base_score - distance_penalty - time_penalty
    
    print(f"  Base formation score: {base_score:.1f}/10.0")
    print(f"  Distance penalty: -{distance_penalty:.1f}")
    print(f"  Time decay penalty: -{time_penalty:.1f}")
    print(f"  Dynamic score: {dynamic_score:.1f}/10.0")
    
    print("\n🎯 TRADING IMPLICATIONS:")
    if distance_pct < 0.05:
        status = "🔴 ACTIVE SETUP"
        action = "Monitor for rejection at zone"
    elif distance_pct < 0.15:
        status = "🟠 APPROACHING SETUP"  
        action = "Prepare for potential entry"
    else:
        status = "🟡 DISTANT SETUP"
        action = "Monitor for price return to zone"
        
    print(f"  Status: {status}")
    print(f"  Recommended action: {action}")
    print(f"  Entry zone: ${adaptive_zone_low:.2f} - ${adaptive_zone_high:.2f}")

if __name__ == "__main__":
    # Demonstrate the challenges
    df = demonstrate_scale_problem()
    
    # Show what we need
    show_what_we_need()
    
    # Simulate improved detection
    simulate_improved_detection()
    
    print("\n" + "="*60)
    print("🎯 KEY TAKEAWAYS")
    print("="*60)
    print("1. Current system: Rule-based, single-candle focused")
    print("2. Real formations: Pattern-based, multi-timeframe")
    print("3. Need: Adaptive, probabilistic, time-aware analysis") 
    print("4. Goal: Bridge the gap between theory and practice")
    print("\n✅ This analysis shows exactly why detection is challenging!")