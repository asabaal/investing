#!/usr/bin/env python3
"""
Trade Setup Examples Generator

Creates comprehensive visual examples covering the full scoring range (0-10 points)
for the trade scoring system. Each example demonstrates different combinations
of the 6 scoring metrics with clear explanations for workshop discussions.

Scoring System Breakdown:
1. Zone Strength (0-2 pts): Leg out distance + opposing zone breakout
2. Time/Base (0-1 pts): Number of candles in base segment  
3. Freshness (0-2 pts): Zone testing/violation status
4. Trend Alignment (0-2 pts): Zone direction vs current trend
5. Price Position (0-1 pts): Position in long-term price range
6. Profit Potential (0-2 pts): Risk/reward ratio

This creates examples across the scoring spectrum to workshop definitions.
"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import trade scoring system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_scoring_system import TradeScorer, Zone, MarketContext, ZoneType, TrendDirection, FreshnessStatus

# Dark theme
pio.templates.default = "plotly_dark"

class TradeSetupVisualizer:
    """Generates comprehensive trade setup examples with visual scoring breakdowns"""
    
    def __init__(self):
        self.scorer = TradeScorer()
    
    def generate_example_data(self, scenario_type):
        """Generate OHLC data for different scoring scenarios"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        if scenario_type == "perfect_setup":
            # Score: 9.5-10.0 - Nearly perfect setup
            return self._create_perfect_setup_data(dates)
            
        elif scenario_type == "strong_setup":
            # Score: 7.5-8.5 - Strong but not perfect
            return self._create_strong_setup_data(dates)
            
        elif scenario_type == "moderate_setup":
            # Score: 5.0-6.5 - Mixed signals
            return self._create_moderate_setup_data(dates)
            
        elif scenario_type == "weak_setup":
            # Score: 2.5-4.0 - Weak opportunity
            return self._create_weak_setup_data(dates)
            
        elif scenario_type == "poor_setup":
            # Score: 0.5-2.0 - Poor setup
            return self._create_poor_setup_data(dates)
            
        elif scenario_type == "invalid_setup":
            # Score: -1.0 - Violated zone
            return self._create_invalid_setup_data(dates)
            
        elif scenario_type == "dbr_demand_uptrend":
            # Score: ~6-7 - DBR demand zone in established uptrend
            return self._create_dbr_demand_uptrend_data(dates)
            
        elif scenario_type == "counter_trend_supply":
            # Score: ~3-4 - Supply zone fighting established downtrend
            return self._create_counter_trend_supply_data(dates)
            
        elif scenario_type == "sideways_local_trend":
            # Score: ~4-5 - Zone with local sideways trend (large base)
            return self._create_sideways_local_trend_data(dates)
            
        elif scenario_type == "deep_tested_zone":
            # Score: ~2-3 - Previously tested zone with deep penetration
            return self._create_deep_tested_zone_data(dates)
        
        else:
            raise ValueError(f"Unknown scenario: {scenario_type}")
    
    def _create_perfect_setup_data(self, dates):
        """Perfect setup: Untested demand zone, strong trend alignment, excellent R:R"""
        
        # Set fixed seed for consistent data
        np.random.seed(123)
        
        ohlc_data = []
        price = 80.0  # Start low - this will be near the bottom of the range
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # DOWNWARD/SIDEWAYS movement to create proper rally in boundary
        for i in range(15):
            price += np.random.uniform(-0.5, 0.0)  # Downward/sideways to create boundary
            open_p = price
            high_p = price + abs(np.random.normal(0.4, 0.1))
            low_p = price - abs(np.random.normal(0.5, 0.2))
            close_p = price - np.random.uniform(0.0, 0.4)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === RBR FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 6 candles strong up move
        rally_start_idx = len(ohlc_data)
        rally_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(1.5, 2.5)  # Strong bullish candles
            low_p = price - np.random.uniform(0.2, 0.5)
            close_p = price + np.random.uniform(1.2, 2.0)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        rally_end = price
        zone_high = price
        
        # BASE CONSOLIDATION - 3 candles (1pt for time score)
        base_start_idx = len(ohlc_data)
        base_range = 1.5
        
        for i in range(3):
            open_p = price + np.random.uniform(-0.5, 0.5)
            high_p = zone_high + np.random.uniform(0, 0.3)
            low_p = zone_high - base_range + np.random.uniform(0, 0.3)
            close_p = zone_high - np.random.uniform(0.3, 0.8)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - base_range
        
        # RALLY OUT (Leg Out) - Strong but contained breakout (4 candles)
        rally_out_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(1.8, 2.8)  # Strong breakout
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(1.5, 2.5)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_out_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBR FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # FIRST: Consolidation/pullback to TERMINATE the rally out leg
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.3)  # Limited upside
            low_p = price - np.random.uniform(0.8, 1.5)   # Pullback/consolidation
            close_p = price - np.random.uniform(0.3, 1.0) # Net down to terminate rally
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
            
        # THEN: Resume upward movement to establish range high
        for i in range(11):
            price += np.random.uniform(0.8, 1.5)  # Upward movement
            open_p = price
            high_p = price + abs(np.random.normal(1.0, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0.5, 1.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'rally_out_start': rally_out_start_idx,
            'rally_out_end': rally_out_end_idx
        }
        
        # Define zone and market context
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=3,
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[(zone_high + 2, zone_high + 3)]  # Opposing supply zone broken
        )
        
        context = MarketContext(
            current_trend=TrendDirection.UP,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)  # Leg out distance for profit potential
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Perfect Setup: RBR demand zone in strong uptrend with excellent R:R", segment_info
    
    def _create_strong_setup_data(self, dates):
        """Strong setup: RBR demand zone with minor penetration, good trend alignment, solid R:R"""
        
        # Set fixed seed for consistent data
        np.random.seed(200)
        
        ohlc_data = []
        price = 85.0  # Start in lower third for demand zone positioning
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # DOWNWARD/SIDEWAYS movement to create proper rally in boundary
        for i in range(15):
            price += np.random.uniform(-0.6, 0.1)  # Downward/sideways drift
            open_p = price
            high_p = price + abs(np.random.normal(0.4, 0.1))
            low_p = price - abs(np.random.normal(0.5, 0.2))
            close_p = price - np.random.uniform(0.1, 0.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === RBR FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 5 candles
        rally_start_idx = len(ohlc_data)
        rally_start = price
        
        for i in range(5):
            open_p = price
            high_p = price + np.random.uniform(1.2, 2.0)
            low_p = price - np.random.uniform(0.2, 0.4)
            close_p = price + np.random.uniform(0.8, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        rally_end = price
        zone_high = price
        
        # BASE CONSOLIDATION - 3 candles (1.0pt for time - optimal)
        base_start_idx = len(ohlc_data)
        base_range = 2.0
        
        for i in range(3):
            open_p = price + np.random.uniform(-0.4, 0.4)
            high_p = zone_high + np.random.uniform(0, 0.2)
            low_p = zone_high - base_range + np.random.uniform(0, 0.2)
            close_p = zone_high - np.random.uniform(0.5, 1.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - base_range
        
        # MINOR TEST and BOUNCE - 2 candles (part of leg out, not base)
        test_start_idx = len(ohlc_data)
        for i in range(2):
            open_p = price
            # First candle: minor penetration then recovery
            if i == 0:
                high_p = price + np.random.uniform(0.2, 0.5)
                low_p = zone_low + 0.6  # Minor penetration
                close_p = zone_low + np.random.uniform(0.8, 1.2)  # Recovery
            else:
                # Second candle: bounce higher  
                high_p = price + np.random.uniform(0.8, 1.2)
                low_p = price - np.random.uniform(0.1, 0.3)
                close_p = price + np.random.uniform(0.6, 1.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        test_end_idx = len(ohlc_data) - 1
        
        # RALLY OUT (Leg Out) - Strong but contained (8 candles)
        rally_out_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(8):
            open_p = price
            high_p = price + np.random.uniform(1.0, 1.8)
            low_p = price - np.random.uniform(0.2, 0.5)
            close_p = price + np.random.uniform(0.8, 1.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_out_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBR FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (22 candles)
        # FIRST: Consolidation/pullback to TERMINATE the rally out leg
        for i in range(5):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.4)  # Limited upside
            low_p = price - np.random.uniform(1.0, 2.0)   # Pullback/consolidation
            close_p = price - np.random.uniform(0.5, 1.5) # Net down to terminate rally
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
            
        # THEN: Resume upward movement to establish range high
        for i in range(17):
            price += np.random.uniform(0.5, 1.2)  # Strong upward to establish range high
            open_p = price
            high_p = price + abs(np.random.normal(0.8, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0.4, 1.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'test_start': test_start_idx,
            'test_end': test_end_idx,
            'rally_out_start': rally_out_start_idx,
            'rally_out_end': rally_out_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=5,  # Includes test period: 3 base + 2 test = 5 total (0.5pt for time)
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.UP,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.PARTIAL_PENETRATION,
            penetration_percentage=30.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Strong Setup: RBR demand zone with minor test in uptrend", segment_info
    
    def _create_moderate_setup_data(self, dates):
        """Moderate setup: DBD supply zone, sideways trend, deep penetration, average R:R"""
        
        # Set fixed seed for consistent data
        np.random.seed(300)
        
        ohlc_data = []
        price = 110.0  # Start in middle range
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Sideways trending context leading up to formation
        for i in range(15):
            price += np.random.uniform(-0.3, 0.3)  # True sideways movement
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.5, 0.2))
            close_p = price + np.random.uniform(-0.2, 0.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === DBD FORMATION STARTS HERE ===
        
        # DROP IN (Leg In) - 4 candles strong bearish movement
        drop_start_idx = len(ohlc_data)
        drop_start = price
        
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.2, 0.5)
            low_p = price - np.random.uniform(1.5, 2.2)
            close_p = price - np.random.uniform(1.0, 1.8)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        zone_low = price
        
        # BASE CONSOLIDATION - 7 candles (0 pts for time - too many)
        base_start_idx = len(ohlc_data)
        base_range = 2.5
        
        for i in range(7):
            open_p = price + np.random.uniform(-0.3, 0.3)
            high_p = zone_low + base_range - np.random.uniform(0, 0.2)
            low_p = zone_low - np.random.uniform(0, 0.2)
            close_p = zone_low + np.random.uniform(0.8, 2.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_high = zone_low + base_range
        
        # DEEP TEST of zone (70% penetration) - 4 candles
        test_start_idx = len(ohlc_data)
        penetration_level = zone_low + (base_range * 0.7)  # 70% into zone
        
        for i in range(4):
            open_p = price
            high_p = penetration_level + np.random.uniform(0, 0.3)  # Deep into zone but not violated
            low_p = price - np.random.uniform(0.5, 1.0)
            close_p = penetration_level - np.random.uniform(0.2, 0.8)  # Pull back from penetration
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        test_end_idx = len(ohlc_data) - 1
        
        # DROP OUT (Leg Out) - 8 candles moderate bearish breakout
        drop_out_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(8):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.4)
            low_p = price - np.random.uniform(0.8, 1.4)
            close_p = price - np.random.uniform(0.5, 1.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_out_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === DBD FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Continue sideways/mixed movement to establish range context
        for i in range(15):
            price += np.random.uniform(-0.4, 0.6)  # Mixed movement to establish range
            open_p = price
            high_p = price + abs(np.random.normal(0.6, 0.2))
            low_p = price - abs(np.random.normal(0.4, 0.1))
            close_p = price + np.random.uniform(-0.2, 0.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'test_start': test_start_idx,
            'test_end': test_end_idx,
            'drop_out_start': drop_out_start_idx,
            'drop_out_end': drop_out_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=11,  # Includes test period: 7 base + 4 test = 11 total (0pt for time)
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.SIDEWAYS,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.DEEP_PENETRATION,
            penetration_percentage=70.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Moderate Setup: DBD supply zone with deep penetration in sideways market", segment_info
    
    def _create_weak_setup_data(self, dates):
        """Weak setup: RBD supply zone with large base creating local sideways trend"""
        
        # Set fixed seed for consistent data
        np.random.seed(42)
        
        ohlc_data = []
        price = 120.0  # Start high for downward pre-formation movement
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Strong DOWNWARD movement (flipped from old post-formation)
        for i in range(15):
            price -= np.random.uniform(1.0, 2.0)  # Strong downward movement
            open_p = price
            high_p = price + abs(np.random.normal(0.3, 0.1))
            low_p = price - abs(np.random.normal(1.2, 0.3))
            close_p = price - np.random.uniform(0.8, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
        
        # === RBD FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 6 candles up move
        rally_start_idx = len(ohlc_data)  # Formation starts here
        rally_start_price = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(1.0, 1.8)
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(0.7, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        rally_end_price = price
        
        # BASE CONSOLIDATION - 8 candles (0 pts for time - too many)
        # This IS the supply zone - where price consolidates after rally
        base_start_idx = len(ohlc_data)
        zone_high = price  # Top of base = zone high
        
        for i in range(8):
            # Consolidation candles - mix of up/down within range
            open_p = price + np.random.uniform(-0.4, 0.4)
            high_p = zone_high + np.random.uniform(-0.1, 0.3)  # Slightly above/at zone high
            low_p = zone_high - 2.5 + np.random.uniform(-0.2, 0.3)  # Zone range of 2.5
            close_p = zone_high - np.random.uniform(0.3, 2.0)  # Mostly toward bottom of range
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - 2.5  # Bottom of consolidation range
        
        # DROP OUT (Leg Out) - 6 candles breaking below base
        drop_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.5)
            low_p = price - np.random.uniform(0.8, 1.5)  # Strong bearish candles
            close_p = price - np.random.uniform(0.6, 1.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBD FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Flat/sideways movement - NO violation of supply zone
        for i in range(15):
            # Keep price well below zone to avoid violation
            target_price = zone_low - 5.0  # Stay safely below zone
            price += np.random.uniform(-0.8, 0.5)  # Mixed but contained movement
            price = max(price, target_price - 3.0)  # Floor to prevent going too low
            price = min(price, zone_low - 1.0)     # Ceiling to prevent zone violation
            
            open_p = price
            high_p = price + abs(np.random.normal(0.6, 0.2))
            low_p = price - abs(np.random.normal(0.4, 0.1))
            close_p = price + np.random.uniform(-0.3, 0.4)
            
            # Ensure no violation
            high_p = min(high_p, zone_low - 0.5)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=8,  # Poor time score (>6 candles)
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.SIDEWAYS,  # 8-candle base creates local sideways trend
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)  # Leg out distance for profit potential
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Weak Setup: RBD supply zone (counter-trend) with large base", segment_info
    
    def _create_poor_setup_data(self, dates):
        """Poor setup: Multiple issues, very low scoring"""
        
        # Set fixed seed for consistent data
        np.random.seed(456)
        
        ohlc_data = []
        price = 120.0  # Start high
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Very strong decline (flipped from old post-formation)
        for i in range(15):
            price += np.random.uniform(-2.0, -1.2)  # Very strong decline
            open_p = price
            high_p = price + abs(np.random.normal(0.2, 0.1))
            low_p = price - abs(np.random.normal(1.5, 0.4))
            close_p = price + np.random.uniform(-1.0, -0.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === DBR FORMATION STARTS HERE ===
        
        # DROP IN (Leg In) - 4 candles STRONG bearish movement (more convincing demand zone setup)
        drop_start_idx = len(ohlc_data)
        leg_in_start = price
        
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.3)
            low_p = price - np.random.uniform(1.8, 2.8)  # STRONG bearish candles
            close_p = price - np.random.uniform(1.5, 2.5)  # Strong down closes
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        zone_low = price
        
        # BASE CONSOLIDATION - 12 candles (poor time score)
        base_start_idx = len(ohlc_data)
        
        for i in range(12):
            open_p = price + np.random.uniform(-0.2, 0.2)
            high_p = zone_low + 2.5 + np.random.uniform(0, 0.1)
            low_p = zone_low + np.random.uniform(0, 0.1)
            close_p = zone_low + np.random.uniform(0.8, 2.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_high = zone_low + 2.5
        
        # RALLY OUT (Leg Out) - 7 candles strong rally to create proper DBR
        rally_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(7):
            open_p = price
            high_p = price + np.random.uniform(1.2, 2.0)  # Strong bullish candles
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(1.0, 1.8)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === DBR FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Flat/sideways movement - NO violation of demand zone
        for i in range(15):
            # Keep price well above zone to avoid violation
            target_price = zone_high + 5.0  # Stay safely above zone
            price += np.random.uniform(-0.5, 0.8)  # Mixed but contained movement
            price = max(price, zone_high + 1.0)    # Floor to prevent zone violation
            price = min(price, target_price + 3.0) # Ceiling to prevent going too high
            
            open_p = price
            high_p = price + abs(np.random.normal(0.6, 0.2))
            low_p = price - abs(np.random.normal(0.4, 0.1))
            close_p = price + np.random.uniform(-0.4, 0.3)
            
            # Ensure no violation
            low_p = max(low_p, zone_high + 0.5)
            
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=12,  # Poor time score (>6 candles)
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.DOWN,  # Counter-trend (demand in downtrend)
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)  # Leg out distance for profit potential
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Poor Setup: DBR demand zone with large base (12 candles)", segment_info
    
    def _create_invalid_setup_data(self, dates):
        """Invalid setup: DBD supply zone completely violated by bullish breakout"""
        
        # Set fixed seed for consistent data
        np.random.seed(400)
        
        ohlc_data = []
        price = 105.0
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Mixed upward movement leading to formation
        for i in range(15):
            price += np.random.uniform(0, 0.4)  # Slight upward bias
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.2))
            close_p = price + np.random.uniform(-0.1, 0.3)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === DBD FORMATION STARTS HERE ===
        
        # DROP IN (Leg In) - 4 candles strong bearish movement
        drop_start_idx = len(ohlc_data)
        drop_start = price
        
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.2, 0.5)
            low_p = price - np.random.uniform(1.2, 2.0)
            close_p = price - np.random.uniform(0.8, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        zone_low = price
        
        # BASE CONSOLIDATION - 5 candles 
        base_start_idx = len(ohlc_data)
        base_range = 2.0
        
        for i in range(5):
            open_p = price + np.random.uniform(-0.2, 0.2)
            high_p = zone_low + base_range - np.random.uniform(0, 0.1)
            low_p = zone_low - np.random.uniform(0, 0.1)
            close_p = zone_low + np.random.uniform(0.5, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_high = zone_low + base_range
        
        # STRONG DROP OUT (Leg Out) - 6 candles EXTREME bearish breakout (completing DBD)
        drop_out_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.3)
            low_p = price - np.random.uniform(2.5, 4.0)  # EXTREME bearish candles
            close_p = price - np.random.uniform(2.0, 3.5)  # Strong down closes
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_out_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === DBD FORMATION ENDS HERE (supply zone should now work) ===
        
        # VIOLATION BEGINS - Price reverses and breaks ABOVE zone (zone failure)
        violation_start_idx = len(ohlc_data)
        
        # First, bounce back up toward zone
        for i in range(3):
            open_p = price
            high_p = price + np.random.uniform(1.5, 2.5)  # Strong bounce
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(1.2, 2.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # Then VIOLATION - price breaks ABOVE zone completely (supply zone fails)
        for i in range(4):
            open_p = price
            high_p = zone_high + np.random.uniform(1.5, 3.0)  # Clear violation above zone
            low_p = price - np.random.uniform(0.2, 0.5)
            close_p = zone_high + np.random.uniform(1.0, 2.5)  # Close well above zone
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        violation_end_idx = len(ohlc_data) - 1
        
        # === FORMATION ENDS WITH VIOLATION ===
        
        # POST-VIOLATION CONTEXT - NOT part of formation (10 candles)
        # Continued bullish movement after violation
        for i in range(10):
            price += np.random.uniform(0.2, 0.8)  # Strong continuation after violation
            open_p = price
            high_p = price + abs(np.random.normal(0.6, 0.2))
            low_p = price - abs(np.random.normal(0.2, 0.1))
            close_p = price + np.random.uniform(0.1, 0.6)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'drop_out_start': drop_out_start_idx,
            'drop_out_end': drop_out_end_idx,
            'violation_start': violation_start_idx,
            'violation_end': violation_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=5,
            leg_out_start=leg_out_start,  # Proper leg out (extreme drop)
            leg_out_end=leg_out_end,      # Then violated by bullish reversal
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.UP,  # Bullish after violation
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.VIOLATED,
            penetration_percentage=100.0,  # Complete violation
            target_zone_distance=0.0  # No trading opportunity
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Invalid Setup: DBD supply zone completely violated - no trade opportunity", segment_info
    
    def _create_dbr_demand_uptrend_data(self, dates):
        """DBR demand zone in established uptrend - good scoring example"""
        
        # Set fixed seed for consistent data
        np.random.seed(500)
        
        ohlc_data = []
        price = 90.0  # Start lower for proper demand positioning
        
        # PRE-FORMATION CONTEXT - NOT part of formation (20 candles)
        # ESTABLISHED UPTREND context leading to formation
        for i in range(20):
            price += np.random.uniform(0.25, 0.6)  # Strong consistent uptrend
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.2, 0.1))
            close_p = price + np.random.uniform(0.2, 0.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === DBR FORMATION STARTS HERE ===
        
        # DROP IN (Leg In) - 4 candles down move (pullback in uptrend)
        drop_start_idx = len(ohlc_data)
        drop_start = price
        
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.4)
            low_p = price - np.random.uniform(1.0, 1.8)
            close_p = price - np.random.uniform(0.8, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        zone_low = price
        
        # BASE CONSOLIDATION - 4 candles (good time score)
        base_start_idx = len(ohlc_data)
        base_range = 2.0
        
        for i in range(4):
            open_p = price + np.random.uniform(-0.3, 0.3)
            high_p = zone_low + base_range - np.random.uniform(0, 0.2)
            low_p = zone_low - np.random.uniform(0, 0.2)
            close_p = zone_low + np.random.uniform(0.5, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_high = zone_low + base_range
        
        # RALLY OUT (Leg Out) - 8 candles strong breakout above base
        rally_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(8):
            open_p = price
            high_p = price + np.random.uniform(1.2, 2.0)
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(0.9, 1.7)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === DBR FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # FIRST: Consolidation/pause to TERMINATE the rally out leg
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.3)  # Limited upside
            low_p = price - np.random.uniform(0.5, 1.2)   # Consolidation/pause
            close_p = price - np.random.uniform(0.2, 0.8) # Net down to terminate rally
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
            
        # THEN: Resume uptrend to establish proper range
        for i in range(11):
            price += np.random.uniform(0.1, 0.6)  # Continue uptrend
            open_p = price
            high_p = price + abs(np.random.normal(0.6, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0.0, 0.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=4,
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.UP,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "DBR Setup: Demand zone in established uptrend with good timing", segment_info
    
    def _create_counter_trend_supply_data(self, dates):
        """RBD supply zone fighting established uptrend - poor trend alignment"""
        
        # Set fixed seed for consistent data
        np.random.seed(600)
        
        ohlc_data = []
        price = 75.0  # Start lower for uptrend
        
        # PRE-FORMATION CONTEXT - NOT part of formation (25 candles)
        # ESTABLISHED UPTREND context leading to formation
        for i in range(25):
            price += np.random.uniform(0.3, 0.7)  # Strong consistent uptrend
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0.2, 0.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === RBD FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 6 candles counter-trend bounce
        rally_start_idx = len(ohlc_data)
        rally_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(1.0, 1.8)
            low_p = price - np.random.uniform(0.1, 0.4)
            close_p = price + np.random.uniform(0.7, 1.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        zone_high = price
        
        # BASE CONSOLIDATION - 3 candles (good time score but wrong trend)
        base_start_idx = len(ohlc_data)
        base_range = 2.5
        
        for i in range(3):
            open_p = price + np.random.uniform(-0.4, 0.4)
            high_p = zone_high + np.random.uniform(0, 0.2)
            low_p = zone_high - base_range + np.random.uniform(0, 0.2)
            close_p = zone_high - np.random.uniform(0.5, 2.0)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - base_range
        
        # DROP OUT (Leg Out) - 10 candles continuing the established downtrend
        drop_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(10):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.5)
            low_p = price - np.random.uniform(0.8, 1.5)
            close_p = price - np.random.uniform(0.6, 1.3)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBD FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Uptrend resumes after supply zone fails to hold
        for i in range(15):
            price += np.random.uniform(0.2, 0.6)  # Resume uptrend (zone fails)
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0.1, 0.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=3,
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[(zone_high - 1, zone_high)]  # Opposing demand zone broken
        )
        
        context = MarketContext(
            current_trend=TrendDirection.UP,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Counter-Trend: RBD supply zone fighting established uptrend", segment_info
    
    def _create_sideways_local_trend_data(self, dates):
        """RBD supply zone creating local sideways trend due to large base"""
        
        # Set fixed seed for consistent data
        np.random.seed(700)
        
        ohlc_data = []
        price = 100.0  # Start in middle range
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Mixed context leading up to formation
        for i in range(15):
            price += np.random.uniform(-0.3, 0.4)  # Mixed movement
            open_p = price
            high_p = price + abs(np.random.normal(0.4, 0.2))
            low_p = price - abs(np.random.normal(0.4, 0.2))
            close_p = price + np.random.uniform(-0.2, 0.3)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === RBD FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 6 candles
        rally_start_idx = len(ohlc_data)
        rally_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(0.8, 1.5)
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(0.6, 1.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        zone_high = price
        
        # LARGE BASE CONSOLIDATION - 10 candles (creates local sideways trend)
        base_start_idx = len(ohlc_data)
        base_range = 3.0
        
        for i in range(10):
            open_p = price + np.random.uniform(-0.5, 0.5)
            high_p = zone_high + np.random.uniform(0, 0.3)
            low_p = zone_high - base_range + np.random.uniform(0, 0.3)
            close_p = zone_high - np.random.uniform(0.5, 2.5)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - base_range
        
        # DROP OUT (Leg Out) - 7 candles moderate bearish breakout
        drop_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(7):
            open_p = price
            high_p = price + np.random.uniform(0.2, 0.6)
            low_p = price - np.random.uniform(0.7, 1.3)
            close_p = price - np.random.uniform(0.5, 1.1)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        drop_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBD FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Continue mixed/sideways movement after formation
        for i in range(15):
            price += np.random.uniform(-0.4, 0.5)  # Continue mixed movement
            open_p = price
            high_p = price + abs(np.random.normal(0.5, 0.2))
            low_p = price - abs(np.random.normal(0.4, 0.2))
            close_p = price + np.random.uniform(-0.3, 0.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'drop_start': drop_start_idx,
            'drop_end': drop_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=10,  # Large base creates local sideways
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.SIDEWAYS,  # Local sideways due to large base
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Local Sideways: RBD supply zone with large base creates sideways trend context", segment_info
    
    def _create_deep_tested_zone_data(self, dates):
        """RBD supply zone previously tested with deep penetration - poor freshness"""
        
        # Set fixed seed for consistent data
        np.random.seed(800)
        
        ohlc_data = []
        price = 105.0
        
        # PRE-FORMATION CONTEXT - NOT part of formation (15 candles)
        # Context leading to original zone formation
        for i in range(15):
            price += np.random.uniform(0.1, 0.4)  # Upward leading to formation
            open_p = price
            high_p = price + abs(np.random.normal(0.4, 0.2))
            low_p = price - abs(np.random.normal(0.3, 0.1))
            close_p = price + np.random.uniform(0, 0.3)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        # === RBD FORMATION STARTS HERE ===
        
        # RALLY IN (Leg In) - 5 candles strong bullish movement
        rally_start_idx = len(ohlc_data)
        rally_start = price
        
        for i in range(5):
            open_p = price
            high_p = price + np.random.uniform(1.0, 1.7)
            low_p = price - np.random.uniform(0.1, 0.3)
            close_p = price + np.random.uniform(0.7, 1.4)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        rally_end_idx = len(ohlc_data) - 1
        zone_high = price
        
        # BASE CONSOLIDATION - 5 candles
        base_start_idx = len(ohlc_data)
        base_range = 2.2
        
        for i in range(5):
            open_p = price + np.random.uniform(-0.3, 0.3)
            high_p = zone_high + np.random.uniform(0, 0.2)
            low_p = zone_high - base_range + np.random.uniform(0, 0.2)
            close_p = zone_high - np.random.uniform(0.4, 1.8)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        base_end_idx = len(ohlc_data) - 1
        zone_low = zone_high - base_range
        
        # INITIAL DROP OUT - 4 candles (first leg out attempt)
        initial_drop_start_idx = len(ohlc_data)
        for i in range(4):
            open_p = price
            high_p = price + np.random.uniform(0.2, 0.6)
            low_p = price - np.random.uniform(0.8, 1.4)
            close_p = price - np.random.uniform(0.6, 1.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        initial_drop_end_idx = len(ohlc_data) - 1
        
        # DEEP TEST RETRACEMENT (70% penetration) - 6 candles
        test_start_idx = len(ohlc_data)
        penetration_price = zone_low + (zone_high - zone_low) * 0.7  # 70% into zone
        
        for i in range(6):
            open_p = price
            high_p = penetration_price + np.random.uniform(0, 0.3)  # Deep into zone but not violated
            low_p = price - np.random.uniform(0.3, 0.8)
            close_p = zone_low + np.random.uniform(0.2, 0.8)  # Pull back toward zone low
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        test_end_idx = len(ohlc_data) - 1
        
        # FINAL DROP OUT (Leg Out) - 6 candles continuation
        final_drop_start_idx = len(ohlc_data)
        leg_out_start = price
        
        for i in range(6):
            open_p = price
            high_p = price + np.random.uniform(0.1, 0.4)
            low_p = price - np.random.uniform(0.7, 1.3)
            close_p = price - np.random.uniform(0.5, 1.1)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        final_drop_end_idx = len(ohlc_data) - 1
        leg_out_end = price
        
        # === RBD FORMATION ENDS HERE ===
        
        # POST-FORMATION CONTEXT - NOT part of formation (12 candles)
        # Mixed movement after formation
        for i in range(12):
            price += np.random.uniform(-0.3, 0.3)  # Mixed movement
            open_p = price
            high_p = price + abs(np.random.normal(0.4, 0.1))
            low_p = price - abs(np.random.normal(0.4, 0.2))
            close_p = price + np.random.uniform(-0.2, 0.2)
            ohlc_data.append([open_p, high_p, low_p, close_p])
            price = close_p
        
        df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
        
        # Store segment boundaries for visualization
        segment_info = {
            'rally_start': rally_start_idx,
            'rally_end': rally_end_idx,
            'base_start': base_start_idx,
            'base_end': base_end_idx,
            'initial_drop_start': initial_drop_start_idx,
            'initial_drop_end': initial_drop_end_idx,
            'test_start': test_start_idx,
            'test_end': test_end_idx,
            'final_drop_start': final_drop_start_idx,
            'final_drop_end': final_drop_end_idx
        }
        
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=11,  # Includes test period: 5 base + 6 test = 11 total (0pt for time)
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]
        )
        
        context = MarketContext(
            current_trend=TrendDirection.SIDEWAYS,
            current_price=price,
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.DEEP_PENETRATION,  # Deeply tested
            penetration_percentage=70.0,
            target_zone_distance=abs(leg_out_end - leg_out_start)
        )
        
        # Ensure dates match data length
        dates_subset = dates[:len(df)] if len(dates) >= len(df) else pd.date_range(dates[0], periods=len(df), freq='D')
        
        return df, dates_subset, zone, context, "Deep Tested: RBD supply zone with 70% penetration - weakened setup", segment_info
    
    def create_comprehensive_visualization(self, scenario_type):
        """Create comprehensive visualization with scoring breakdown"""
        
        # Generate data
        result = self.generate_example_data(scenario_type)
        if len(result) == 6:  # All updated examples now return segment info
            df, dates, zone, context, description, segment_info = result
        else:
            df, dates, zone, context, description = result
            segment_info = None
        
        # Calculate scores
        scores = self.scorer.calculate_total_score(zone, context)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{scenario_type.replace("_", " ").title()} - Score: {scores["total"]:.1f}/10.0',
                'Scoring Breakdown',
                'Market Context Analysis', 
                'Zone Analysis Details'
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Main candlestick chart
        df['datetime'] = dates[:len(df)]
        
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)
        
        # Add segment backgrounds and zone rectangle
        zone_color = '#00ff88' if zone.zone_type == ZoneType.DEMAND else '#ff4444'
        zone_fill = 'rgba(0, 255, 136, 0.3)' if zone.zone_type == ZoneType.DEMAND else 'rgba(255, 68, 68, 0.3)'
        
        if segment_info:
            # Add segment backgrounds with flexible key handling
            
            # Determine formation start based on available keys
            formation_start = None
            if 'rally_start' in segment_info:
                formation_start = segment_info['rally_start']
            elif 'drop_start' in segment_info:
                formation_start = segment_info['drop_start']
            
            # Determine formation end based on available keys
            formation_end = None
            if 'drop_end' in segment_info:
                formation_end = segment_info['drop_end']
            elif 'rally_out_end' in segment_info:
                formation_end = segment_info['rally_out_end']
            elif 'drop_out_end' in segment_info:
                formation_end = segment_info['drop_out_end']
            elif 'rally_end' in segment_info:
                formation_end = segment_info['rally_end']
            
            # PRE-FORMATION context (light gray)
            if formation_start and formation_start > 0:
                fig.add_shape(
                    type="rect", 
                    x0=df['datetime'].iloc[0],
                    x1=df['datetime'].iloc[formation_start-1],
                    y0=df['low'].min() * 0.98,
                    y1=df['high'].max() * 1.02,
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    line=dict(color='gray', width=1, dash='dot'),
                    row=1, col=1
                )
            
            # BASE segment background (this is the actual ZONE)
            if 'base_start' in segment_info and 'base_end' in segment_info:
                fig.add_shape(
                    type="rect",
                    x0=df['datetime'].iloc[segment_info['base_start']],
                    x1=df['datetime'].iloc[segment_info['base_end']],
                    y0=df['low'].min() * 0.98,
                    y1=df['high'].max() * 1.02,
                    fillcolor='rgba(255, 170, 0, 0.3)',
                    line=dict(color='#ffaa00', width=3),
                    row=1, col=1
                )
                
                # Determine zone rectangle bounds (base + any test segments)
                zone_start = segment_info['base_start']
                zone_end = segment_info['base_end']
                
                # Extend to include test segments that are part of consolidation
                for key in segment_info.keys():
                    if ('test' in key or 'drop_attempt' in key) and 'end' in key:
                        if segment_info[key] > zone_end:
                            zone_end = segment_info[key]
                
                # Zone rectangle (covering ENTIRE consolidation period)
                fig.add_shape(
                    type="rect",
                    x0=df['datetime'].iloc[zone_start],
                    x1=df['datetime'].iloc[zone_end],
                    y0=zone.low,
                    y1=zone.high,
                    fillcolor=zone_fill,
                    line=dict(color=zone_color, width=4),
                    row=1, col=1
                )
            
            # POST-FORMATION context (light gray)
            if formation_end and formation_end < len(df) - 1:
                fig.add_shape(
                    type="rect",
                    x0=df['datetime'].iloc[formation_end+1],
                    x1=df['datetime'].iloc[-1],
                    y0=df['low'].min() * 0.98,
                    y1=df['high'].max() * 1.02,
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    line=dict(color='gray', width=1, dash='dot'),
                    row=1, col=1
                )
            
        else:
            # Default zone rectangle for other setups
            zone_start_idx = len(df) // 3  # Approximate
            zone_end_idx = len(df) * 2 // 3
            
            fig.add_shape(
                type="rect",
                x0=df['datetime'].iloc[zone_start_idx],
                x1=df['datetime'].iloc[zone_end_idx],
                y0=zone.low,
                y1=zone.high,
                fillcolor=zone_fill,
                line=dict(color=zone_color, width=3),
                row=1, col=1
            )
        
        # Add zone and segment labels
        if segment_info:
            # Zone label (always present in base segment)
            if 'base_start' in segment_info and 'base_end' in segment_info:
                base_mid = (segment_info['base_start'] + segment_info['base_end']) // 2
                fig.add_annotation(
                    x=df['datetime'].iloc[base_mid],
                    y=(zone.high + zone.low) / 2,
                    text=f"<b>{zone.zone_type.value.upper()} ZONE</b><br>Base ({zone.base_candles} candles)<br>Range: {zone.range_size:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=zone_color,
                    font=dict(color=zone_color, size=12),
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor=zone_color,
                    borderwidth=2,
                    row=1, col=1
                )
            
        else:
            # Default zone label for other setups
            zone_start_idx = len(df) // 3  # Approximate  
            fig.add_annotation(
                x=df['datetime'].iloc[zone_start_idx + 5],
                y=(zone.high + zone.low) / 2,
                text=f"{zone.zone_type.value.upper()} ZONE<br>Range: {zone.range_size:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=zone_color,
                font=dict(color=zone_color, size=12),
                bgcolor='rgba(0,0,0,0.8)',
                row=1, col=1
            )
        
        # Scoring breakdown bar chart
        if scores['status'] == 'VALID':
            score_names = ['Zone Strength', 'Time/Base', 'Freshness', 'Trend Alignment', 'Price Position', 'Profit Potential']
            score_values = [
                scores['zone_strength'], scores['time_base'], scores['freshness'],
                scores['trend_alignment'], scores['price_position'], scores['profit_potential']
            ]
            max_values = [2, 1, 2, 2, 1, 2]  # Maximum possible for each metric
            
            # Current scores
            fig.add_trace(go.Bar(
                x=score_names,
                y=score_values,
                name='Current Score',
                marker_color='#00aaff',
                text=[f'{v:.1f}' for v in score_values],
                textposition='auto'
            ), row=2, col=1)
            
            # Maximum possible (background)
            fig.add_trace(go.Bar(
                x=score_names,
                y=max_values,
                name='Max Possible',
                marker_color='rgba(100,100,100,0.3)',
                text=[f'/{m}' for m in max_values],
                textposition='auto'
            ), row=2, col=1)
        
        # Analysis details table
        if scores['status'] == 'VALID':
            analysis_data = [
                ['Zone Type', f'{zone.zone_type.value.title()}', f'{scores["zone_strength"]:.1f}/2.0'],
                ['Base Candles', f'{zone.base_candles}', f'{scores["time_base"]:.1f}/1.0'],
                ['Zone Range', f'{zone.range_size:.1f}', '-'],
                ['Leg Out Distance', f'{zone.leg_out_distance:.1f}', '-'],
                ['Current Trend', f'{context.current_trend.value.title()}', f'{scores["trend_alignment"]:.1f}/2.0'],
                ['Freshness Status', f'{context.freshness_status.value.replace("_", " ").title()}', f'{scores["freshness"]:.1f}/2.0'],
                ['Price Position', f'{(zone.high - df["low"].min()) / (df["high"].max() - df["low"].min()) * 100:.0f}% of visible range', f'{scores["price_position"]:.1f}/1.0'],
                ['Target Distance', f'{context.target_zone_distance:.1f}' if context.target_zone_distance else 'N/A', '-'],
                ['Profit Potential Ratio', f'{zone.leg_out_distance/zone.range_size:.1f}:1', f'{scores["profit_potential"]:.1f}/2.0']
            ]
        else:
            analysis_data = [
                ['Status', 'INVALID - Zone Violated', '0.0/10.0'],
                ['Zone Type', zone.zone_type.value.title(), '-'],
                ['Violation Level', f'{context.penetration_percentage:.1f}%', '-'],
                ['', '', ''],
                ['', '', ''],
                ['', '', ''],
                ['', '', ''],
                ['', '', ''],
                ['', '', '']
            ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Metric', 'Value', 'Points Scored'],
                fill_color='#404040',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    [item[0] for item in analysis_data], 
                    [item[1] for item in analysis_data],
                    [item[2] for item in analysis_data]
                ],
                fill_color=['#2a2a2a', '#1a1a1a', '#333333'],
                font=dict(color='white', size=11)
            )
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=f'{description}<br>Total Score: {scores["total"]:.1f}/10.0 - Status: {scores["status"]}',
            height=900,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2a2a2a',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(x=0.02, y=0.45)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1, rangeslider_visible=False)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Scoring Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Score Points", row=2, col=1, range=[0, 2.2])
        
        return fig, scores
    
    def generate_all_examples(self):
        """Generate complete set of trade setup examples"""
        
        scenarios = [
            "perfect_setup",
            "strong_setup", 
            "moderate_setup",
            "weak_setup",
            "poor_setup",
            "invalid_setup",
            "dbr_demand_uptrend",
            "counter_trend_supply",
            "sideways_local_trend",
            "deep_tested_zone"
        ]
        
        print("🎯 GENERATING COMPREHENSIVE TRADE SETUP EXAMPLES")
        print("=" * 60)
        print("Creating visual examples across the full 0-10 scoring range")
        print("Each example demonstrates different scoring metric combinations\n")
        
        results = {}
        
        for scenario in scenarios:
            print(f"📊 Creating {scenario.replace('_', ' ')} example...")
            
            try:
                fig, scores = self.create_comprehensive_visualization(scenario)
                
                # Save visualization
                filename = f"trade_setup_examples/{scenario}_analysis.html"
                fig.write_html(filename)
                
                results[scenario] = {
                    'filename': filename,
                    'scores': scores,
                    'total_score': scores['total']
                }
                
                status_emoji = '✅' if scores['status'] == 'VALID' else '❌'
                print(f"   {status_emoji} Saved: {filename}")
                print(f"      Score: {scores['total']:.1f}/10.0 - Status: {scores['status']}")
                
                # Print breakdown for valid setups
                if scores['status'] == 'VALID':
                    print(f"      Breakdown: ZS:{scores['zone_strength']:.1f} TB:{scores['time_base']:.1f} FR:{scores['freshness']:.1f} TA:{scores['trend_alignment']:.1f} PP:{scores['price_position']:.1f} PR:{scores['profit_potential']:.1f}")
                
            except Exception as e:
                print(f"   ❌ Error creating {scenario}: {str(e)}")
                continue
        
        # Generate summary report
        self._create_summary_report(results)
        
        print(f"\n🎯 COMPLETE! Generated {len(results)} trade setup examples:")
        for scenario, data in results.items():
            score = data['total_score']
            status = '✅ VALID' if score >= 0 else '❌ INVALID'
            print(f"   • {scenario.replace('_', ' ').title()}: {score:.1f}/10.0 {status}")
        
        print(f"\n📊 Summary report: trade_setup_examples/scoring_summary_report.html")
        print(f"   Use these examples to workshop and refine scoring definitions!")
        
        return results
    
    def _create_summary_report(self, results):
        """Create comprehensive summary report of all examples"""
        
        # Create summary figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Score Distribution Across Examples',
                'Scoring Metric Comparison',
                'Setup Quality Categories',
                'Key Insights & Patterns'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "table"}]
            ]
        )
        
        # Score distribution
        scenario_names = [s.replace('_', ' ').title() for s in results.keys()]
        total_scores = [data['total_score'] for data in results.values()]
        colors = ['red' if score < 0 else 'orange' if score < 3 else 'yellow' if score < 6 else 'lightgreen' if score < 8 else 'green' for score in total_scores]
        
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=total_scores,
            marker_color=colors,
            text=[f'{score:.1f}' for score in total_scores],
            textposition='auto',
            name='Total Score'
        ), row=1, col=1)
        
        # Bar chart of scoring metrics comparison (for valid setups only)
        valid_results = {k: v for k, v in results.items() if v['total_score'] >= 0}
        
        if valid_results:
            metrics = ['zone_strength', 'time_base', 'freshness', 'trend_alignment', 'price_position', 'profit_potential']
            metric_labels = ['Zone Strength', 'Time/Base', 'Freshness', 'Trend Alignment', 'Price Position', 'Profit Potential']
            
            # Show average scores across all valid setups for each metric
            avg_scores = []
            for metric in metrics:
                scores = [data['scores'].get(metric, 0) for data in valid_results.values()]
                avg_scores.append(sum(scores) / len(scores))
            
            fig.add_trace(go.Bar(
                x=metric_labels,
                y=avg_scores,
                name='Average Score Across Valid Setups',
                marker_color='#33aaff',
                text=[f'{score:.1f}' for score in avg_scores],
                textposition='auto'
            ), row=1, col=2)
        
        # Quality categories pie chart
        categories = {'Excellent (8-10)': 0, 'Good (6-8)': 0, 'Fair (4-6)': 0, 'Poor (2-4)': 0, 'Very Poor (0-2)': 0, 'Invalid (<0)': 0}
        
        for score in total_scores:
            if score < 0:
                categories['Invalid (<0)'] += 1
            elif score < 2:
                categories['Very Poor (0-2)'] += 1
            elif score < 4:
                categories['Poor (2-4)'] += 1
            elif score < 6:
                categories['Fair (4-6)'] += 1
            elif score < 8:
                categories['Good (6-8)'] += 1
            else:
                categories['Excellent (8-10)'] += 1
        
        fig.add_trace(go.Pie(
            labels=list(categories.keys()),
            values=list(categories.values()),
            marker_colors=['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']
        ), row=2, col=1)
        
        # Key insights table
        insights = [
            ['Highest Scoring Setup', max(results.keys(), key=lambda k: results[k]['total_score']).replace('_', ' ').title()],
            ['Average Score', f"{sum(total_scores)/len(total_scores):.1f}"],
            ['Score Range', f"{min(total_scores):.1f} to {max(total_scores):.1f}"],
            ['Valid Setups', f"{len(valid_results)}/{len(results)}"],
            ['Most Common Issue', 'Zone violations and counter-trend trades'],
            ['Best R:R Ratios', 'Perfect and strong setups (>5:1)'],
            ['Time Factor Impact', '1-3 candle bases score highest'],
            ['Trend Alignment Key', 'Counter-trend setups score poorly']
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Insight', 'Value'],
                fill_color='#404040',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[[item[0] for item in insights], [item[1] for item in insights]],
                fill_color=['#2a2a2a', '#1a1a1a'],
                font=dict(color='white', size=11)
            )
        ), row=2, col=2)
        
        fig.update_layout(
            title='Trade Setup Scoring System - Comprehensive Analysis Report',
            height=800,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2a2a2a',
            font=dict(color='white')
        )
        
        fig.update_yaxes(title_text="Score (0-10)", row=1, col=1)
        fig.update_xaxes(title_text="Setup Examples", row=1, col=1)
        
        # Save report
        fig.write_html("trade_setup_examples/scoring_summary_report.html")


def main():
    """Generate comprehensive trade setup examples"""
    
    visualizer = TradeSetupVisualizer()
    results = visualizer.generate_all_examples()
    
    print(f"\n🎯 WORKSHOP READY!")
    print(f"   Generated {len(results)} comprehensive examples")
    print(f"   Each example shows detailed scoring breakdowns")
    print(f"   Use these to refine and validate scoring definitions")
    print(f"   All files saved to: trade_setup_examples/")


if __name__ == "__main__":
    main()