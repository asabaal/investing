#!/usr/bin/env python3
"""
Two-Pass Formation Detection - Clean O(n) Algorithm
Pass 1: Find all monotonic runs using BODY PRICES rule
Pass 2: Find LEG-BASE-LEG patterns from runs
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TwoPassFormationDetector:
    """Clean two-pass formation detection"""
    
    def __init__(self, decay_factor=0.7, min_leg_threshold=1.5):
        self.decay_factor = decay_factor
        self.min_leg_threshold = min_leg_threshold
        
    def detect_formations(self, df):
        """
        Two-pass formation detection
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            list: Detected formations
        """
        print("🚀 Two-pass formation detection starting...")
        
        # Pass 1: Find all monotonic runs
        runs = self._find_all_monotonic_runs(df)
        print(f"   📊 Pass 1: Found {len(runs)} monotonic runs")
        
        # Debug: Show all runs
        for i, run in enumerate(runs):
            print(f"      Run {i}: {run['direction']} candles {run['start_idx']}-{run['end_idx']} (${run['start_price']:.2f} → ${run['end_price']:.2f})")
        
        # Pass 2: Find LEG-BASE-LEG patterns  
        formations = self._find_leg_base_leg_patterns(df, runs)
        print(f"   ✅ Pass 2: Found {len(formations)} formations")
        
        return formations
    
    def _find_all_monotonic_runs(self, df):
        """Pass 1: Find all monotonic runs using BODY PRICES rule"""
        runs = []
        current_run = None
        
        for i, candle in df.iterrows():
            if current_run is None:
                # Start first run
                if i > 0:
                    prev_close = df.iloc[i-1]['close']
                    direction = 'UP' if candle['close'] > prev_close else 'DOWN'
                else:
                    direction = 'UP' if candle['close'] > candle['open'] else 'DOWN'
                    
                current_run = {
                    'start_idx': i,
                    'direction': direction,
                    'candles': [i]
                }
            else:
                # Check if candle continues current run
                prev_candle = df.iloc[i-1]
                prev_close = prev_candle['close']
                
                # Handle gaps: if there's a significant gap, treat it as directional movement
                gap_size = abs(candle['open'] - prev_close)
                gap_pct = gap_size / prev_close * 100
                has_gap = gap_pct > 2.0  # 2% gap threshold
                
                if has_gap:
                    # Debug gap detection
                    if i >= 4 and i <= 6:
                        print(f"      🔍 Gap detected at candle {i}: prev_close=${prev_close:.2f}, open=${candle['open']:.2f}, gap={gap_pct:.1f}%")
                    
                    # Gap creates a bridge run for directional movement
                    gap_up = candle['open'] > prev_close
                    gap_direction = 'UP' if gap_up else 'DOWN'
                    
                    # End current run
                    self._finalize_run(current_run, df)
                    runs.append(current_run)
                    
                    # Create bridge run for the gap movement
                    bridge_run = {
                        'start_idx': i - 1,  # Previous candle
                        'end_idx': i,        # Current candle  
                        'direction': gap_direction,
                        'candles': [i - 1, i],
                        'is_bridge': True
                    }
                    self._finalize_run(bridge_run, df)
                    runs.append(bridge_run)
                    
                    if i >= 4 and i <= 6:
                        print(f"      🔍 Created bridge run: {gap_direction} from {i-1} to {i}")
                    
                    # Start new run with current candle direction
                    next_direction = 'UP' if candle['close'] > candle['open'] else 'DOWN'
                    current_run = {
                        'start_idx': i,
                        'direction': next_direction,
                        'candles': [i]
                    }
                    continue  # Skip the normal processing
                else:
                    # No gap: use normal BODY PRICES rule
                    if current_run['direction'] == 'UP':
                        # UP run continues if: (open >= prev_close) OR (close >= prev_close)
                        continues = (candle['open'] >= prev_close) or (candle['close'] >= prev_close)
                    else:
                        # DOWN run continues if: (open <= prev_close) OR (close <= prev_close)  
                        continues = (candle['open'] <= prev_close) or (candle['close'] <= prev_close)
                
                if continues:
                    current_run['candles'].append(i)
                else:
                    # Current run ends, save it
                    self._finalize_run(current_run, df)
                    runs.append(current_run)
                    
                    # Start new run with current candle
                    prev_close = df.iloc[i-1]['close']
                    direction = 'UP' if candle['close'] > prev_close else 'DOWN'
                    current_run = {
                        'start_idx': i,
                        'direction': direction,
                        'candles': [i]
                    }
        
        # Don't forget last run
        if current_run:
            self._finalize_run(current_run, df)
            runs.append(current_run)
        
        return runs
    
    def _finalize_run(self, run, df):
        """Add end index and price info to run"""
        run['end_idx'] = run['candles'][-1]
        run['start_price'] = df.iloc[run['start_idx']]['close']
        run['end_price'] = df.iloc[run['end_idx']]['close']
        run['movement'] = run['end_price'] - run['start_price']
        run['abs_movement'] = abs(run['movement'])
        
        # CRITICAL FIX: Correct the direction based on actual movement
        if run['movement'] > 0:
            run['direction'] = 'UP'
        else:
            run['direction'] = 'DOWN'
    
    def _find_leg_base_leg_patterns(self, df, runs):
        """Pass 2: Find LEG-BASE-LEG patterns from runs"""
        formations = []
        
        # Look for consecutive LEG-BASE-LEG patterns
        for i in range(len(runs) - 2):
            leg_in = runs[i]
            base = runs[i + 1] 
            leg_out = runs[i + 2]
            
            # Check if it's a valid formation pattern
            if self._is_valid_formation(df, leg_in, base, leg_out):
                formation_type, zone_type = self._classify_formation(leg_in, leg_out)
                
                # Build comprehensive formation data structure
                formation = self._create_formation_data(df, leg_in, base, leg_out, formation_type, zone_type)
                formations.append(formation)
                
                print(f"   ✅ Found {formation_type}: LEG_IN({leg_in['start_idx']}-{leg_in['end_idx']}) BASE({base['start_idx']}-{base['end_idx']}) LEG_OUT({leg_out['start_idx']}-{leg_out['end_idx']})")
        
        return formations
    
    def _is_valid_formation(self, df, leg_in, base, leg_out):
        """Check if LEG-BASE-LEG pattern is valid"""
        # Both legs must have significant movement compared to base
        leg_in_size = leg_in['abs_movement']
        leg_out_size = leg_out['abs_movement']
        base_size = base['abs_movement']
        
        # Legs should be bigger than base movement
        if leg_in_size <= base_size * self.min_leg_threshold:
            return False
        if leg_out_size <= base_size * self.min_leg_threshold:
            return False
            
        # Additional validation: legs must be at least 2 candles
        if len(leg_in['candles']) < 2 or len(leg_out['candles']) < 2:
            return False
            
        return True
    
    def _classify_formation(self, leg_in, leg_out):
        """Classify formation type based on actual price movements"""
        leg_in_up = leg_in['movement'] > 0
        leg_out_up = leg_out['movement'] > 0
        
        if not leg_in_up and not leg_out_up:
            return 'DBD', 'SUPPLY'  # Drop-Base-Drop
        elif not leg_in_up and leg_out_up:
            return 'DBR', 'DEMAND'  # Drop-Base-Rally  
        elif leg_in_up and leg_out_up:
            return 'RBR', 'DEMAND'  # Rally-Base-Rally
        elif leg_in_up and not leg_out_up:
            return 'RBD', 'SUPPLY'  # Rally-Base-Drop
        
        return 'UNKNOWN', 'UNKNOWN'
    
    def _create_formation_data(self, df, leg_in, base, leg_out, formation_type, zone_type):
        """Create comprehensive formation data structure matching the original format"""
        
        # Calculate base range using the same weighted approach
        base_range = self._calculate_base_range(df, base['candles'])
        
        # Calculate actual high/low range for visualization
        base_candle_data = [df.iloc[idx] for idx in base['candles']]
        actual_high = max(candle['high'] for candle in base_candle_data)
        actual_low = min(candle['low'] for candle in base_candle_data)
        
        return {
            'type': formation_type,
            'zone_type': zone_type,
            'start_idx': leg_in['start_idx'],
            'end_idx': leg_out['end_idx'],
            'leg_in': {
                'start_idx': leg_in['start_idx'],
                'end_idx': leg_in['end_idx'],
                'direction': leg_in['direction'],
                'movement': leg_in['abs_movement']
            },
            'base': {
                'start_idx': base['start_idx'],
                'end_idx': base['end_idx'],
                'base_candles': base['candles'].copy(),
                'base_range': base_range,
                'actual_high': actual_high,
                'actual_low': actual_low,
                'proximal_line': actual_high if zone_type == 'DEMAND' else actual_low,
                'distal_line': actual_low if zone_type == 'DEMAND' else actual_high
            },
            'leg_out': {
                'start_idx': leg_out['start_idx'],
                'end_idx': leg_out['end_idx'],
                'direction': leg_out['direction'],
                'movement': leg_out['abs_movement']
            },
            'valid': True,
            'validation_details': {
                'leg_in_movement': leg_in['abs_movement'],
                'leg_out_movement': leg_out['abs_movement'],
                'base_range': base_range['range'] if base_range else 0,
                'leg_in_valid': True,
                'leg_out_valid': True
            }
        }
    
    def _calculate_base_range(self, df, base_candles):
        """Calculate weighted base range (same as original)"""
        if not base_candles:
            return None
        
        weighted_high = 0
        weighted_low = 0
        total_weight = 0
        
        for i, candle_idx in enumerate(base_candles):
            candle = df.iloc[candle_idx]
            age = len(base_candles) - i - 1
            weight = self.decay_factor ** age
            
            weighted_high += candle['high'] * weight
            weighted_low += candle['low'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        return {
            'high': weighted_high / total_weight,
            'low': weighted_low / total_weight,
            'range': (weighted_high - weighted_low) / total_weight,
            'candle_count': len(base_candles),
            'total_weight': total_weight
        }

if __name__ == "__main__":
    # Test with Formation 20 data
    test_data = [
        # Context before
        {'datetime': '2019-09-09', 'open': 11.92, 'high': 12.1, 'low': 11.895, 'close': 12.08},  # 0
        {'datetime': '2019-09-10', 'open': 12.11, 'high': 12.225, 'low': 11.9, 'close': 11.98},  # 1
        {'datetime': '2019-09-11', 'open': 11.99, 'high': 12.05, 'low': 11.58, 'close': 11.68},  # 2
        {'datetime': '2019-09-12', 'open': 11.32, 'high': 11.52, 'low': 11.28, 'close': 11.47},  # 3
        {'datetime': '2019-09-13', 'open': 11.49, 'high': 11.525, 'low': 11.3833, 'close': 11.44},  # 4
        # Formation candles
        {'datetime': '2019-09-16', 'open': 12.49, 'high': 13.16, 'low': 12.45, 'close': 12.83},  # 5
        {'datetime': '2019-09-17', 'open': 12.93, 'high': 12.93, 'low': 12.15, 'close': 12.29},  # 6
        {'datetime': '2019-09-18', 'open': 12.13, 'high': 12.268, 'low': 12.01, 'close': 12.10},  # 7
        {'datetime': '2019-09-19', 'open': 12.22, 'high': 12.285, 'low': 12.11, 'close': 12.19},  # 8
        {'datetime': '2019-09-20', 'open': 12.23, 'high': 12.3156, 'low': 12.08, 'close': 12.20},  # 9
        {'datetime': '2019-09-23', 'open': 12.14, 'high': 12.24, 'low': 12.04, 'close': 12.21},  # 10
        {'datetime': '2019-09-24', 'open': 12.12, 'high': 12.1498, 'low': 11.86, 'close': 11.90},  # 11
        {'datetime': '2019-09-25', 'open': 11.66, 'high': 11.82, 'low': 11.6, 'close': 11.80},  # 12
        {'datetime': '2019-09-26', 'open': 11.68, 'high': 11.81, 'low': 11.56, 'close': 11.80},  # 13
        {'datetime': '2019-09-27', 'open': 11.72, 'high': 11.77, 'low': 11.52, 'close': 11.56},  # 14
        {'datetime': '2019-09-30', 'open': 11.63, 'high': 11.79, 'low': 11.49, 'close': 11.78},  # 15
        {'datetime': '2019-10-01', 'open': 11.77, 'high': 11.85, 'low': 11.49, 'close': 11.52},  # 16
        {'datetime': '2019-10-02', 'open': 11.16, 'high': 11.41, 'low': 10.97, 'close': 11.13},  # 17
        {'datetime': '2019-10-03', 'open': 11.03, 'high': 11.23, 'low': 10.93, 'close': 10.93},  # 18
    ]
    
    df = pd.DataFrame(test_data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.reset_index(drop=True)
    
    print("Formation 20 Test Data:")
    for i, row in df.iterrows():
        print(f"  {i}: {row['datetime'].strftime('%Y-%m-%d')} ${row['close']:.2f}")
    
    detector = TwoPassFormationDetector()
    formations = detector.detect_formations(df)
    
    print(f"\nFinal Results:")
    for i, f in enumerate(formations):
        print(f"Formation {i+1} ({f['type']}):")
        print(f"  Leg In: {f['leg_in']['start_idx']}-{f['leg_in']['end_idx']} ({f['leg_in']['direction']})")
        print(f"  Base: {min(f['base']['base_candles'])}-{max(f['base']['base_candles'])}")
        print(f"  Leg Out: {f['leg_out']['start_idx']}-{f['leg_out']['end_idx']} ({f['leg_out']['direction']})")