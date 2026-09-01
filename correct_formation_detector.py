#!/usr/bin/env python3
"""
Correct Formation Detection - Does What We KNOW Is True
No garbage, just the actual BODY PRICES rule applied correctly
"""

import pandas as pd
import numpy as np

class CorrectFormationDetector:
    """Formation detector that actually works correctly"""
    
    def __init__(self):
        self.formations = []
    
    def detect_formations(self, df):
        """Find all formations by applying BODY PRICES rule correctly"""
        print("🎯 CORRECT Formation Detection - No Garbage")
        
        # Find all monotonic runs first
        runs = self._find_all_monotonic_runs(df)
        print(f"Found {len(runs)} monotonic runs")
        
        # Find LEG-BASE-LEG patterns
        formations = self._find_leg_base_leg_patterns(df, runs)
        
        print(f"✅ Found {len(formations)} correct formations")
        return formations
    
    def _find_all_monotonic_runs(self, df):
        """Find all monotonic runs using BODY PRICES rule"""
        runs = []
        current_run = None
        
        for i, candle in df.iterrows():
            if current_run is None:
                # Start first run
                direction = 'DOWN' if candle['close'] < candle['open'] else 'UP'
                current_run = {
                    'start_idx': i,
                    'direction': direction,
                    'candles': [i]
                }
            else:
                # Check if candle continues current run
                prev_candle = df.iloc[i-1]
                prev_close = prev_candle['close']
                
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
                    current_run['end_idx'] = current_run['candles'][-1]
                    current_run['start_price'] = df.iloc[current_run['start_idx']]['close']
                    current_run['end_price'] = df.iloc[current_run['end_idx']]['close']
                    current_run['movement'] = current_run['end_price'] - current_run['start_price']
                    runs.append(current_run)
                    
                    # Start new run with current candle
                    direction = 'DOWN' if candle['close'] < candle['open'] else 'UP'
                    current_run = {
                        'start_idx': i,
                        'direction': direction,
                        'candles': [i]
                    }
        
        # Don't forget last run
        if current_run:
            current_run['end_idx'] = current_run['candles'][-1]
            current_run['start_price'] = df.iloc[current_run['start_idx']]['close']
            current_run['end_price'] = df.iloc[current_run['end_idx']]['close']
            current_run['movement'] = current_run['end_price'] - current_run['start_price']
            runs.append(current_run)
        
        return runs
    
    def _find_leg_base_leg_patterns(self, df, runs):
        """Find LEG-BASE-LEG patterns from runs"""
        formations = []
        
        for i in range(len(runs) - 2):
            leg_in = runs[i]
            base = runs[i + 1] 
            leg_out = runs[i + 2]
            
            # Check if it's a valid formation pattern
            if self._is_valid_formation(df, leg_in, base, leg_out):
                formation_type = self._classify_formation(leg_in, leg_out)
                
                formation = {
                    'type': formation_type,
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx']
                }
                formations.append(formation)
                
                print(f"Found {formation_type}: LEG_IN({leg_in['start_idx']}-{leg_in['end_idx']}) BASE({base['start_idx']}-{base['end_idx']}) LEG_OUT({leg_out['start_idx']}-{leg_out['end_idx']})")
        
        return formations
    
    def _is_valid_formation(self, df, leg_in, base, leg_out):
        """Check if LEG-BASE-LEG pattern is valid"""
        # Legs must have significant movement
        leg_in_size = abs(leg_in['movement'])
        leg_out_size = abs(leg_out['movement'])
        base_size = abs(base['movement'])
        
        # Both legs should be bigger than base
        return leg_in_size > base_size and leg_out_size > base_size
    
    def _classify_formation(self, leg_in, leg_out):
        """Classify formation type based on actual price movements"""
        leg_in_up = leg_in['movement'] > 0
        leg_out_up = leg_out['movement'] > 0
        
        if not leg_in_up and not leg_out_up:
            return 'DBD'  # Drop-Base-Drop
        elif not leg_in_up and leg_out_up:
            return 'DBR'  # Drop-Base-Rally  
        elif leg_in_up and leg_out_up:
            return 'RBR'  # Rally-Base-Rally
        elif leg_in_up and not leg_out_up:
            return 'RBD'  # Rally-Base-Drop
        
        return 'UNKNOWN'

if __name__ == "__main__":
    # Test with Formation 7 data
    test_data = [
        {'date': '2013-06-19', 'open': 34.96, 'high': 35.05, 'low': 34.63, 'close': 34.78},
        {'date': '2013-06-20', 'open': 33.98, 'high': 34.03, 'low': 33.52, 'close': 33.63},
        {'date': '2013-06-21', 'open': 33.63, 'high': 33.69, 'low': 32.98, 'close': 33.23},
        {'date': '2013-06-24', 'open': 33.05, 'high': 33.86, 'low': 33.00, 'close': 33.64},
        {'date': '2013-06-25', 'open': 33.92, 'high': 33.95, 'low': 33.55, 'close': 33.74},
        {'date': '2013-06-26', 'open': 33.82, 'high': 33.88, 'low': 33.18, 'close': 33.80},
        {'date': '2013-06-27', 'open': 34.00, 'high': 34.50, 'low': 33.92, 'close': 34.33},
        {'date': '2013-06-28', 'open': 34.43, 'high': 34.58, 'low': 34.12, 'close': 34.18},
        {'date': '2013-07-01', 'open': 34.61, 'high': 34.80, 'low': 34.46, 'close': 34.67},
        {'date': '2013-07-02', 'open': 34.90, 'high': 35.36, 'low': 34.86, 'close': 35.21},
        {'date': '2013-07-03', 'open': 35.93, 'high': 36.18, 'low': 35.60, 'close': 35.84},
        {'date': '2013-07-05', 'open': 36.05, 'high': 36.59, 'low': 35.94, 'close': 36.56},
    ]
    
    df = pd.DataFrame(test_data)
    df = df.reset_index()
    
    detector = CorrectFormationDetector()
    formations = detector.detect_formations(df)