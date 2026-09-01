#!/usr/bin/env python3
"""
Single Pass Formation Detection - O(n) Algorithm
Processes candles once in order, building formations as we go
"""

import pandas as pd
import numpy as np
from datetime import datetime

class SinglePassFormationDetector:
    """Efficient single-pass formation detection with O(n) complexity"""
    
    def __init__(self, decay_factor=0.7, min_leg_threshold=1.5):
        self.decay_factor = decay_factor
        self.min_leg_threshold = min_leg_threshold
        
        # State tracking
        self.reset_state()
    
    def reset_state(self):
        """Reset detector state for new analysis"""
        self.state = 'SEEKING_LEG_IN'  # States: SEEKING_LEG_IN, IN_LEG_IN, IN_BASE, IN_LEG_OUT
        self.current_run = None
        self.leg_in_run = None
        self.base_candles = []
        self.formations = []
        self.candle_idx = 0
        
    def detect_formations(self, df):
        """
        Single pass through candles to detect formations
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            list: Detected formations
        """
        print("🚀 Single-pass formation detection starting...")
        self.reset_state()
        
        total_candles = len(df)
        report_interval = max(100, total_candles // 20)  # Report every 5%
        
        for i, candle in df.iterrows():
            self.candle_idx = i
            
            # Progress reporting
            if i > 0 and i % report_interval == 0:
                progress = (i / total_candles) * 100
                print(f"   📊 Progress: {progress:.1f}% ({i:,}/{total_candles:,} candles) - Found {len(self.formations)} formations")
            
            self._process_candle(df, i, candle)
        
        print(f"   ✅ Completed! Found {len(self.formations)} formations in single pass")
        return self.formations
    
    def _process_candle(self, df, idx, candle):
        """Process a single candle based on current state"""
        
        # DEBUG: Log state transitions for Formation 20 analysis
        if idx >= 5 and idx <= 10:
            date_str = candle.get('datetime', f'idx_{idx}')
            print(f"   📍 Candle {idx} ({date_str}): close=${candle['close']:.2f}, state={self.state}")
            if self.current_run:
                print(f"      Current run: {self.current_run['direction']} from {self.current_run['start_idx']}")
        
        if self.state == 'SEEKING_LEG_IN':
            self._seek_leg_in(df, idx, candle)
            
        elif self.state == 'IN_BASE':
            self._continue_base(df, idx, candle)
            
        elif self.state == 'IN_LEG_OUT':
            self._continue_leg_out(df, idx, candle)
            
        # DEBUG: Log state after processing
        if idx >= 5 and idx <= 10:
            print(f"      After processing: state={self.state}")
            if self.leg_in_run:
                print(f"      Leg In: {self.leg_in_run['start_idx']}-{self.leg_in_run['end_idx']}")
            if self.base_candles:
                print(f"      Base candles: {self.base_candles}")
    
    def _seek_leg_in(self, df, idx, candle):
        """Look for the start of a leg in run"""
        # Start tracking a new potential run
        if self.current_run is None:
            self.current_run = {
                'start_idx': idx,
                'direction': self._get_run_direction(df, idx, idx),
                'start_price': candle['close'],
                'last_price': candle['close']
            }
        else:
            # Check if this candle continues the current run
            if self._candle_continues_run(df, idx, candle, self.current_run):
                self.current_run['last_price'] = candle['close']
                
                # Check if current run is now significant enough to be LEG IN
                if self._is_valid_leg_run(self.current_run, idx):
                    # Current run is valid - make it LEG IN and stay in seeking mode
                    # We'll transition to BASE when this run actually ENDS
                    pass
            else:
                # Current run ended - check if it was a valid LEG IN
                if self._is_valid_leg_run(self.current_run, idx - 1):
                    # Previous run was valid LEG IN - BASE starts with current candle
                    self.leg_in_run = {
                        'start_idx': self.current_run['start_idx'],
                        'end_idx': idx - 1,
                        'direction': self.current_run['direction'],
                        'start_price': self.current_run['start_price'],
                        'end_price': self.current_run['last_price']
                    }
                    self.state = 'IN_BASE'
                    self.base_candles = [idx]  # Current candle starts base
                else:
                    # Previous run was not valid - start new candidate run
                    self.current_run = {
                        'start_idx': idx,
                        'direction': self._get_run_direction(df, idx, idx),
                        'start_price': candle['close'],
                        'last_price': candle['close']
                    }
    
    def _continue_leg_in(self, df, idx, candle):
        """Continue tracking the leg in run"""
        if self._candle_continues_run(df, idx, candle, self.current_run):
            # Update both current run and leg_in_run
            self.current_run['last_price'] = candle['close']
            self.leg_in_run['end_idx'] = idx
            self.leg_in_run['end_price'] = candle['close']
        else:
            # Leg in run ended - validate and transition to base
            if self._is_valid_leg_run(self.leg_in_run, self.leg_in_run['end_idx']):
                # Valid leg in - start base with current candle
                self.state = 'IN_BASE'
                self.base_candles = [idx]
            else:
                # Invalid leg in - restart seeking
                self._restart_seeking()
                # Start tracking new run
                self.current_run = {
                    'start_idx': idx,
                    'direction': self._get_run_direction(df, idx, idx),
                    'start_price': candle['close'],
                    'last_price': candle['close']
                }
    
    def _continue_base(self, df, idx, candle):
        """Continue building base or detect leg out start"""
            
        # Check if this candle fits in the base
        current_base_range = self._calculate_base_range(df, self.base_candles)
        
        if current_base_range and self._candle_fits_base(candle, current_base_range):
            self.base_candles.append(idx)
        else:
            # CURRENT candle breaks base - LEG_OUT starts with THIS candle
            self.current_run = {
                'start_idx': idx,
                'direction': self._get_run_direction(df, idx, idx),
                'start_price': candle['close'],
                'last_price': candle['close']
            }
            self.state = 'IN_LEG_OUT'
    
    def _continue_leg_out(self, df, idx, candle):
        """Continue leg out or complete formation"""
        if self._candle_continues_run(df, idx, candle, self.current_run):
            self.current_run['last_price'] = candle['close']
        else:
            # Leg out run ended - validate and create formation
            leg_out_run = {
                'start_idx': self.current_run['start_idx'],
                'end_idx': idx - 1,
                'direction': self.current_run['direction'],
                'start_price': self.current_run['start_price'],
                'end_price': self.current_run['last_price']
            }
            
            # Check if transition candles should be included in base based on price
            self._adjust_base_for_transition_candles(df, leg_out_run)
            
            # Validate formation (segments are already continuous by design)
            formation = self._validate_and_create_consecutive_formation(df, leg_out_run)
            if formation:
                self.formations.append(formation)
                print(f"   ✅ Found CONTINUOUS {formation['type']} formation: LEG({self.leg_in_run['start_idx']}-{self.leg_in_run['end_idx']}) -> BASE({min(self.base_candles)}-{max(self.base_candles)}) -> LEG({leg_out_run['start_idx']}-{leg_out_run['end_idx']})")
            
            self._restart_seeking()
    
    def _adjust_base_for_transition_candles(self, df, leg_out_run):
        """Check if transition candles should be included in base based on price"""
        if not self.base_candles or not self.leg_in_run:
            return
            
        # Calculate current base range
        base_range = self._calculate_base_range(df, self.base_candles)
        if not base_range:
            return
            
        # Check if LEG IN end candle should be included in base
        leg_in_end_idx = self.leg_in_run['end_idx']
        leg_in_end_candle = df.iloc[leg_in_end_idx]
        
        if leg_in_end_idx not in self.base_candles:
            if self._candle_fits_base(leg_in_end_candle, base_range):
                self.base_candles.insert(0, leg_in_end_idx)  # Add to beginning
                self.base_candles.sort()  # Keep sorted
        
        # Check if LEG OUT start candle should be included in base  
        leg_out_start_idx = leg_out_run['start_idx']
        leg_out_start_candle = df.iloc[leg_out_start_idx]
        
        if leg_out_start_idx not in self.base_candles:
            # Recalculate base range in case we added leg_in_end
            base_range = self._calculate_base_range(df, self.base_candles)
            if base_range and self._candle_fits_base(leg_out_start_candle, base_range):
                self.base_candles.append(leg_out_start_idx)  # Add to end
                self.base_candles.sort()  # Keep sorted
    
    def _restart_seeking(self):
        """Reset state to seek new leg in"""
        self.state = 'SEEKING_LEG_IN'
        self.current_run = None
        self.leg_in_run = None
        self.base_candles = []
    
    def _get_run_direction(self, df, start_idx, current_idx):
        """Determine run direction based on actual price movement between candles"""
        if start_idx == current_idx:
            # First candle - use close vs previous close if available
            if start_idx > 0:
                prev_close = df.iloc[start_idx - 1]['close']
                current_close = df.iloc[start_idx]['close']
                return 'UP' if current_close > prev_close else 'DOWN'
            else:
                # Very first candle - use open vs close
                candle = df.iloc[start_idx]
                return 'UP' if candle['close'] > candle['open'] else 'DOWN'
        else:
            # Multi-candle run - use actual price movement
            start_close = df.iloc[start_idx]['close']
            current_close = df.iloc[current_idx]['close']
            return 'UP' if current_close > start_close else 'DOWN'
    
    def _candle_continues_run(self, df, idx, candle, run):
        """Check if candle continues the monotonic run using BODY PRICES rule"""
        if idx == 0 or idx == run['start_idx']:
            return True
            
        prev_candle = df.iloc[idx - 1]
        prev_close = prev_candle['close']
        current_open = candle['open']
        current_close = candle['close']
        
        # BODY PRICES RULE: Run continues if EITHER open OR close continues the directional movement
        if run['direction'] == 'UP':
            # UP run continues if: (open >= prev_close) OR (close >= prev_close)
            continues = (current_open >= prev_close) or (current_close >= prev_close)
        else:  # DOWN run
            # DOWN run continues if: (open <= prev_close) OR (close <= prev_close)
            continues = (current_open <= prev_close) or (current_close <= prev_close)
        
        # DEBUG: Log the decision for Formation 20 analysis
        if idx >= 5 and idx <= 8:  # Around Sep 16-19 area
            date_str = candle.get('datetime', f'idx_{idx}')
            print(f"   🔍 Candle {idx} ({date_str}): prev_close=${prev_close:.2f}, open=${current_open:.2f}, close=${current_close:.2f}")
            print(f"      Run direction: {run['direction']}, continues: {continues}")
            if run['direction'] == 'DOWN':
                print(f"      Checks: open<={prev_close:.2f}? {current_open <= prev_close}, close<={prev_close:.2f}? {current_close <= prev_close}")
        
        return continues
    
    def _is_valid_leg_run(self, run, end_idx):
        """Check if run is valid for leg (minimum length/movement)"""
        if end_idx - run['start_idx'] < 1:  # At least 2 candles
            return False
        
        # Handle both 'last_price' and 'end_price' keys
        end_price = run.get('end_price') or run.get('last_price')
        movement = abs(end_price - run['start_price'])
        return movement > 0.1  # Minimum movement threshold
    
    def _calculate_base_range(self, df, base_candles):
        """Calculate weighted base range"""
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
    
    def _candle_fits_base(self, candle, base_range):
        """Check if candle overlaps with base range"""
        return not (candle['high'] < base_range['low'] or candle['low'] > base_range['high'])
    
    def _validate_and_create_consecutive_formation(self, df, leg_out_run):
        """Validate and create formation if valid"""
        base_range = self._calculate_base_range(df, self.base_candles)
        if not base_range:
            return None
        
        leg_in_movement = abs(self.leg_in_run['end_price'] - self.leg_in_run['start_price'])
        leg_out_movement = abs(leg_out_run['end_price'] - leg_out_run['start_price'])
        
        # Validate movements exceed base range
        if (leg_in_movement <= base_range['range'] * self.min_leg_threshold or
            leg_out_movement <= base_range['range'] * self.min_leg_threshold):
            return None
        
        # CRITICAL: Validate breakout direction relative to base zone
        leg_out_start_price = leg_out_run['start_price']  # FIRST candle of leg out
        leg_out_end_price = leg_out_run['end_price']
        base_high = base_range['high']
        base_low = base_range['low']
        
        # Check if leg out STARTS by breaking the base zone in the correct direction
        # The FIRST candle must break the zone, not just the final price
        breaks_above_at_start = leg_out_start_price > base_high
        breaks_below_at_start = leg_out_start_price < base_low
        
        # Simple math - is leg movement up or down?
        leg_in_up = self.leg_in_run['end_price'] > self.leg_in_run['start_price']
        leg_out_up = leg_out_end_price > leg_out_start_price
        
        # Debug info
        if len(self.formations) < 3:
            leg_in_dir = 'UP' if leg_in_up else 'DOWN'
            leg_out_dir = 'UP' if leg_out_up else 'DOWN'
            print(f"   🔍 Formation validation: Leg In={leg_in_dir}, Leg Out={leg_out_dir}")
            print(f"      Leg In: ${self.leg_in_run['start_price']:.2f} -> ${self.leg_in_run['end_price']:.2f}")
            print(f"      Leg Out: ${leg_out_start_price:.2f} -> ${leg_out_end_price:.2f}")
            print(f"      Base Range: {base_low:.2f} - {base_high:.2f}")
        
        # Simple formation type logic
        if not leg_in_up and not leg_out_up and breaks_below_at_start:
            formation_type, zone_type = 'DBD', 'SUPPLY'  # Drop-Base-Drop
        elif not leg_in_up and leg_out_up and breaks_above_at_start:
            formation_type, zone_type = 'DBR', 'DEMAND'  # Drop-Base-Rally
        elif leg_in_up and leg_out_up and breaks_above_at_start:
            formation_type, zone_type = 'RBR', 'DEMAND'  # Rally-Base-Rally
        elif leg_in_up and not leg_out_up and breaks_below_at_start:
            formation_type, zone_type = 'RBD', 'SUPPLY'  # Rally-Base-Drop
        else:
            # Invalid formation
            if len(self.formations) < 3:
                if not leg_out_up and not breaks_below_at_start:
                    print(f"      ❌ Invalid: DOWN leg doesn't break below base")
                elif leg_out_up and not breaks_above_at_start:
                    print(f"      ❌ Invalid: UP leg doesn't break above base")
                else:
                    print(f"      ❌ Invalid: No proper breakout")
            return None
        
        if len(self.formations) < 3:
            leg_out_dir = 'UP' if leg_out_up else 'DOWN'
            print(f"      ✅ Valid {formation_type}: {leg_out_dir} leg with proper breakout")
        
        # Calculate ACTUAL high/low range for zone drawing
        base_candle_data = [df.iloc[idx] for idx in self.base_candles]
        actual_high = max(candle['high'] for candle in base_candle_data)
        actual_low = min(candle['low'] for candle in base_candle_data)
        
        return {
            'type': formation_type,
            'zone_type': zone_type,
            'start_idx': self.leg_in_run['start_idx'],
            'end_idx': leg_out_run['end_idx'],
            'leg_in': {
                'start_idx': self.leg_in_run['start_idx'],
                'end_idx': self.leg_in_run['end_idx'],
                'direction': self.leg_in_run['direction'],
                'movement': leg_in_movement
            },
            'base': {
                'start_idx': min(self.base_candles),
                'end_idx': max(self.base_candles),
                'base_candles': self.base_candles.copy(),
                'base_range': base_range,
                'actual_high': actual_high,
                'actual_low': actual_low,
                'proximal_line': actual_high if zone_type == 'DEMAND' else actual_low,
                'distal_line': actual_low if zone_type == 'DEMAND' else actual_high
            },
            'leg_out': {
                'start_idx': leg_out_run['start_idx'],
                'end_idx': leg_out_run['end_idx'],
                'direction': leg_out_run['direction'],
                'movement': leg_out_movement
            },
            'valid': True,
            'validation_details': {
                'leg_in_movement': leg_in_movement,
                'leg_out_movement': leg_out_movement,
                'base_range': base_range['range'],
                'leg_in_valid': True,
                'leg_out_valid': True
            }
        }
    
    def _classify_formation(self, leg_in_direction, leg_out_direction):
        """Legacy method - replaced by proper breakout validation in _validate_and_create_formation"""
        # This method is now unused - formation type determined by actual breakout direction
        return None, None
    
    def print_formation_summary(self, formations):
        """Print formation summary"""
        print(f"\n📊 SINGLE-PASS FORMATION DETECTION SUMMARY")
        print("=" * 50)
        
        if not formations:
            print("❌ No formations detected")
            return
        
        valid_formations = [f for f in formations if f['valid']]
        
        print(f"✅ Total Formations: {len(formations)}")
        print(f"   Valid: {len(valid_formations)}")
        
        # Count by type
        formation_counts = {}
        for formation in valid_formations:
            ftype = formation['type']
            formation_counts[ftype] = formation_counts.get(ftype, 0) + 1
        
        print(f"\n📈 FORMATION BREAKDOWN:")
        for ftype, count in formation_counts.items():
            zone_type = 'SUPPLY' if ftype in ['DBD', 'RBD'] else 'DEMAND'
            print(f"   {ftype} ({zone_type}): {count}")