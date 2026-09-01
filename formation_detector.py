#!/usr/bin/env python3
"""
Formation Detection System
Integrates the corrected formation logic with weighted base ranges and exponential decay
"""

import pandas as pd
import numpy as np
from datetime import datetime

class FormationDetector:
    """Advanced formation detection using weighted base ranges with exponential decay"""
    
    def __init__(self, decay_factor=0.7, min_leg_threshold=1.5):
        """
        Initialize formation detector
        
        Args:
            decay_factor: Exponential decay factor for base range weighting (0.7 = recent candles more important)
            min_leg_threshold: Minimum leg movement relative to base range for valid formation
        """
        self.decay_factor = decay_factor
        self.min_leg_threshold = min_leg_threshold
    
    def classify_candle_sentiment(self, candle_data):
        """
        Classify candle sentiment using 1/3-2/3 method
        
        Args:
            candle_data: Dict with OHLC values
            
        Returns:
            str: 'BASE', 'LEG_BULLISH', 'LEG_BEARISH'
        """
        high = candle_data['high']
        low = candle_data['low']
        open_price = candle_data['open']
        close = candle_data['close']
        
        total_range = high - low
        if total_range == 0:
            return 'BASE'
        
        # Calculate close position in range
        close_position = (close - low) / total_range
        
        # BASE candle: Close is between 1/3 and 2/3 of range
        if 1/3 <= close_position <= 2/3:
            return 'BASE'
        
        # LEG candles: Determine direction by close vs open
        if close > open_price:
            return 'LEG_BULLISH'
        else:
            return 'LEG_BEARISH'
    
    def detect_monotonic_runs(self, df):
        """
        Detect monotonic runs using formation-appropriate logic:
        - Directional consistency (bullish/bearish candles)
        - Edge progression (at least one body edge more extreme than previous)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            list: List of run dictionaries with start, end, direction
        """
        runs = []
        i = 0
        
        while i < len(df) - 1:
            # Find the start of a run
            start_idx = i
            start_price = df.iloc[i]['close']
            
            # Look ahead to find the direction and extent of the run
            current_direction = None
            end_idx = start_idx
            
            for j in range(i + 1, len(df)):
                prev_candle = df.iloc[j-1]
                curr_candle = df.iloc[j]
                
                # Determine current candle direction
                candle_direction = 'UP' if curr_candle['close'] > curr_candle['open'] else 'DOWN'
                
                # Check if this candle can continue the current run
                can_continue = False
                
                if current_direction is None:
                    # First candle - establish direction
                    current_direction = candle_direction
                    end_idx = j
                    can_continue = True
                elif current_direction == candle_direction:
                    # Same direction - check edge progression
                    if current_direction == 'UP':
                        # UP run continues if open higher OR close higher
                        if (curr_candle['open'] > prev_candle['open'] or 
                            curr_candle['close'] > prev_candle['close']):
                            can_continue = True
                    else:  # DOWN run
                        # DOWN run continues if open lower OR close lower
                        if (curr_candle['open'] < prev_candle['open'] or 
                            curr_candle['close'] < prev_candle['close']):
                            can_continue = True
                
                if can_continue:
                    end_idx = j
                else:
                    # Run ends here
                    break
            
            # Add run if it's valid (more than 1 candle and has direction)
            if current_direction is not None and end_idx > start_idx:
                run = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': current_direction,
                    'start_price': start_price,
                    'end_price': df.iloc[end_idx]['close']
                }
                runs.append(run)
                
                # Move to the end of this run for next iteration
                i = end_idx
            else:
                # No valid run found, move to next candle
                i += 1
        
        return runs
    
    def calculate_weighted_base_range(self, base_candles, df):
        """
        Calculate weighted base range using exponential decay
        
        Args:
            base_candles: List of candle indices in base segment
            df: DataFrame with OHLC data
            
        Returns:
            dict: Base range info with high, low, and weight details
        """
        if not base_candles:
            return None
        
        weighted_high = 0
        weighted_low = 0
        total_weight = 0
        
        # Apply exponential decay - recent candles weighted more heavily
        for i, candle_idx in enumerate(base_candles):
            candle = df.iloc[candle_idx]
            
            # Weight decreases exponentially for older candles
            age = len(base_candles) - i - 1  # 0 for most recent
            weight = self.decay_factor ** age
            
            weighted_high += candle['high'] * weight
            weighted_low += candle['low'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        return {
            'high': weighted_high / total_weight,
            'low': weighted_low / total_weight,
            'range': (weighted_high / total_weight) - (weighted_low / total_weight),
            'candle_count': len(base_candles),
            'total_weight': total_weight
        }
    
    def overlaps_with_base_range(self, candle, base_range):
        """
        Check if candle overlaps with the dynamic base range
        
        Args:
            candle: Single candle data (row from DataFrame)
            base_range: Base range dict from calculate_weighted_base_range
            
        Returns:
            bool: True if candle overlaps with base range
        """
        if base_range is None:
            return False
        
        candle_high = candle['high']
        candle_low = candle['low']
        base_high = base_range['high']
        base_low = base_range['low']
        
        # Check if ranges overlap
        return not (candle_high < base_low or candle_low > base_high)
    
    def detect_formations(self, df):
        """
        Detect formations using proper LEG -> BASE -> LEG logic:
        1. Find LEG IN (monotonic run)
        2. Find BASE (range-based consolidation after leg ends)
        3. Find LEG OUT (leg that breaks base range)
        
        Args:
            df: DataFrame with OHLC data (columns: datetime, open, high, low, close)
            
        Returns:
            list: Detected formations with detailed information
        """
        print("🎯 Starting formation detection with LEG->BASE->LEG approach...")
        
        formations = []
        runs = self.detect_monotonic_runs(df)
        
        print(f"   📈 Detected {len(runs)} monotonic runs")
        
        # Look for LEG IN -> BASE -> LEG OUT patterns (non-overlapping)
        formations_found = 0
        run_idx = 0
        
        while run_idx < len(runs):
            leg_in_run = runs[run_idx]
            
            # After leg in run ends, look for base formation
            base_start_idx = leg_in_run['end_idx'] + 1
            
            if base_start_idx >= len(df):
                run_idx += 1
                continue  # No room for base
            
            # Calculate leg in movement range for base validation
            leg_in_movement = abs(leg_in_run['end_price'] - leg_in_run['start_price'])
            
            # Build base segment dynamically
            base_formation = self._analyze_base_after_leg(df, leg_in_run, base_start_idx, leg_in_movement)
            
            if base_formation and base_formation['valid']:
                formations.append(base_formation)
                formations_found += 1
                if formations_found <= 3:
                    print(f"   ✅ Valid {base_formation['type']} formation: LEG({leg_in_run['start_idx']}-{leg_in_run['end_idx']}) -> BASE({base_formation['base']['start_idx']}-{base_formation['base']['end_idx']}) -> LEG({base_formation['leg_out']['start_idx']}-{base_formation['leg_out']['end_idx']})")
                
                # Skip past this entire formation to avoid overlaps
                formation_end_idx = base_formation['leg_out']['end_idx']
                
                # Find next run that starts after this formation ends
                next_run_idx = run_idx + 1
                while next_run_idx < len(runs) and runs[next_run_idx]['start_idx'] <= formation_end_idx:
                    next_run_idx += 1
                run_idx = next_run_idx
                
            else:
                if formations_found < 3:
                    print(f"   ❌ No valid formation after run {run_idx}")
                run_idx += 1
        
        print(f"   ✅ Detected {len(formations)} formations")
        return formations
    
    def _analyze_base_after_leg(self, df, leg_in_run, base_start_idx, leg_in_movement):
        """
        Analyze base formation after a monotonic leg run using CORRECT logic:
        1. First candle after leg must have smaller range than leg movement to START base
        2. Build base using existing weighted range + overlapping validation logic  
        3. Find leg out that breaks the weighted base range
        
        Args:
            df: DataFrame with OHLC data
            leg_in_run: The preceding monotonic run (leg in)
            base_start_idx: Start index for potential base
            leg_in_movement: Price movement of the leg in run
            
        Returns:
            dict: Formation info if valid, None otherwise
        """
        if base_start_idx >= len(df):
            return None
        
        # Check FIRST candle after leg - its range must be smaller than leg movement to START base
        first_candle = df.iloc[base_start_idx]
        first_candle_range = first_candle['high'] - first_candle['low']
        
        if first_candle_range >= leg_in_movement:
            return None  # No base - first candle breaks out immediately
        
        # Base starts with first candle - now build it using existing weighted logic
        base_candles = [base_start_idx]
        current_base_range = self.calculate_weighted_base_range(base_candles, df)
        
        if current_base_range is None:
            return None
        
        # Build base segment dynamically using weighted range + overlapping validation
        for i in range(base_start_idx + 1, min(base_start_idx + 20, len(df))):  # Limit base length
            candle = df.iloc[i]
            
            # Check if candle overlaps with current base range
            if self.overlaps_with_base_range(candle, current_base_range):
                base_candles.append(i)
                # Recalculate weighted base range with new candle
                current_base_range = self.calculate_weighted_base_range(base_candles, df)
                if current_base_range is None:
                    break
            else:
                # Break in base - this could be start of leg out
                break
        
        if len(base_candles) < 1:
            return None  # No valid base
            
        base_end_idx = base_candles[-1]
        leg_out_start_idx = base_end_idx + 1
        
        if leg_out_start_idx >= len(df):
            return None  # No room for leg out
        
        # Find next monotonic run that starts at or after leg_out_start_idx
        leg_out_run = None
        all_runs = self.detect_monotonic_runs(df)
        
        for run in all_runs:
            if run['start_idx'] >= leg_out_start_idx:
                # Check if this run actually breaks the base range
                run_movement = abs(run['end_price'] - run['start_price'])
                if run_movement > current_base_range['range'] * self.min_leg_threshold:
                    leg_out_run = run
                    break
        
        if leg_out_run is None:
            return None  # No valid leg out found
        
        # Determine formation type
        leg_in_direction = leg_in_run['direction']
        leg_out_direction = leg_out_run['direction']
        
        if leg_in_direction == 'UP' and leg_out_direction == 'UP':
            formation_type = 'RBR'  # Rally-Base-Rally
            zone_type = 'DEMAND'
        elif leg_in_direction == 'DOWN' and leg_out_direction == 'DOWN':
            formation_type = 'DBD'  # Drop-Base-Drop
            zone_type = 'SUPPLY'
        elif leg_in_direction == 'UP' and leg_out_direction == 'DOWN':
            formation_type = 'RBD'  # Rally-Base-Drop
            zone_type = 'SUPPLY'
        elif leg_in_direction == 'DOWN' and leg_out_direction == 'UP':
            formation_type = 'DBR'  # Drop-Base-Rally
            zone_type = 'DEMAND'
        else:
            return None
        
        # Validation
        leg_in_movement_calc = abs(leg_in_run['end_price'] - leg_in_run['start_price'])
        leg_out_movement = abs(leg_out_run['end_price'] - leg_out_run['start_price'])
        
        validation = {
            'leg_in_movement': leg_in_movement_calc,
            'leg_out_movement': leg_out_movement,
            'base_range': current_base_range['range'],
            'leg_in_valid': leg_in_movement_calc > current_base_range['range'] * self.min_leg_threshold,
            'leg_out_valid': leg_out_movement > current_base_range['range'] * self.min_leg_threshold
        }
        
        is_valid = validation['leg_in_valid'] and validation['leg_out_valid']
        
        return {
            'type': formation_type,
            'zone_type': zone_type,
            'start_idx': leg_in_run['start_idx'],
            'end_idx': leg_out_run['end_idx'],
            'leg_in': {
                'run': leg_in_run,
                'start_idx': leg_in_run['start_idx'],
                'end_idx': leg_in_run['end_idx'],
                'direction': leg_in_run['direction'],
                'movement': leg_in_movement_calc
            },
            'base': {
                'start_idx': base_start_idx,
                'end_idx': base_end_idx,
                'base_candles': base_candles,
                'base_range': current_base_range,
                'proximal_line': current_base_range['high'] if zone_type == 'DEMAND' else current_base_range['low'],
                'distal_line': current_base_range['low'] if zone_type == 'DEMAND' else current_base_range['high']
            },
            'leg_out': {
                'run': leg_out_run,
                'start_idx': leg_out_run['start_idx'],
                'end_idx': leg_out_run['end_idx'],
                'direction': leg_out_run['direction'],
                'movement': leg_out_movement
            },
            'valid': is_valid,
            'validation_details': validation
        }

    def _calculate_leg_movement(self, df, leg_idx):
        """Calculate the price movement of a single leg candle"""
        if leg_idx >= len(df):
            return 0
        
        candle = df.iloc[leg_idx]
        return abs(candle['close'] - candle['open'])
    
    def _classify_formation(self, leg_in_sentiment, leg_out_sentiment):
        """Classify formation type based on leg sentiments"""
        if leg_in_sentiment == 'LEG_BULLISH' and leg_out_sentiment == 'LEG_BULLISH':
            return 'RBR', 'DEMAND'  # Rally-Base-Rally
        elif leg_in_sentiment == 'LEG_BEARISH' and leg_out_sentiment == 'LEG_BEARISH':
            return 'DBD', 'SUPPLY'  # Drop-Base-Drop
        elif leg_in_sentiment == 'LEG_BULLISH' and leg_out_sentiment == 'LEG_BEARISH':
            return 'RBD', 'SUPPLY'  # Rally-Base-Drop
        elif leg_in_sentiment == 'LEG_BEARISH' and leg_out_sentiment == 'LEG_BULLISH':
            return 'DBR', 'DEMAND'  # Drop-Base-Rally
        else:
            return None, None
    
    def _analyze_run_based_formation(self, df, run_in, run_out, base_start_idx, base_end_idx):
        """
        Analyze formation based on monotonic runs
        
        Args:
            df: DataFrame with OHLC data
            run_in: Incoming monotonic run
            run_out: Outgoing monotonic run
            base_start_idx: Start of base segment
            base_end_idx: End of base segment
            
        Returns:
            dict: Formation info or None
        """
        base_candles = list(range(base_start_idx, base_end_idx + 1))
        
        if not base_candles:
            return None
            
        # Calculate base range
        base_range = self.calculate_weighted_base_range(base_candles, df)
        if base_range is None:
            return None
        
        # Calculate run movements
        run_in_movement = abs(run_in['end_price'] - run_in['start_price'])
        run_out_movement = abs(run_out['end_price'] - run_out['start_price'])
        
        # Determine formation type based on run directions
        if run_in['direction'] == 'UP' and run_out['direction'] == 'UP':
            formation_type = 'RBR'  # Rally-Base-Rally
            zone_type = 'DEMAND'
        elif run_in['direction'] == 'DOWN' and run_out['direction'] == 'DOWN':
            formation_type = 'DBD'  # Drop-Base-Drop
            zone_type = 'SUPPLY'
        elif run_in['direction'] == 'UP' and run_out['direction'] == 'DOWN':
            formation_type = 'RBD'  # Rally-Base-Drop
            zone_type = 'SUPPLY'
        elif run_in['direction'] == 'DOWN' and run_out['direction'] == 'UP':
            formation_type = 'DBR'  # Drop-Base-Rally
            zone_type = 'DEMAND'
        else:
            return None
        
        # Validate formation - run movements must exceed base range
        validation = {
            'leg_in_movement': run_in_movement,
            'leg_out_movement': run_out_movement,
            'base_range': base_range['range'],
            'leg_in_valid': run_in_movement > base_range['range'] * self.min_leg_threshold,
            'leg_out_valid': run_out_movement > base_range['range'] * self.min_leg_threshold
        }
        
        is_valid = validation['leg_in_valid'] and validation['leg_out_valid']
        
        # Additional validation: base should be properly consolidated
        # Check that runs don't overlap excessively with base range
        base_high = base_range['high']
        base_low = base_range['low']
        
        # Run validation - significant portion of runs should be outside base range
        run_in_start_price = df.iloc[run_in['start_idx']]['close'] 
        run_in_end_price = df.iloc[run_in['end_idx']]['close']
        run_out_start_price = df.iloc[run_out['start_idx']]['close']
        run_out_end_price = df.iloc[run_out['end_idx']]['close']
        
        # For valid formation, runs should clearly move away from and back to base range
        if formation_type in ['RBR', 'DBR']:  # Demand zones
            # Base should be between the run extremes
            if not (min(run_in_start_price, run_in_end_price) <= base_low <= base_high <= max(run_out_start_price, run_out_end_price)):
                is_valid = False
        else:  # Supply zones (RBD, DBD)
            # Base should be between the run extremes  
            if not (min(run_out_start_price, run_out_end_price) <= base_low <= base_high <= max(run_in_start_price, run_in_end_price)):
                is_valid = False
        
        return {
            'type': formation_type,
            'zone_type': zone_type,
            'start_idx': run_in['start_idx'],
            'end_idx': run_out['end_idx'],
            'leg_in': {
                'run': run_in,
                'start_idx': run_in['start_idx'],
                'end_idx': run_in['end_idx'],
                'direction': run_in['direction'],
                'movement': run_in_movement
            },
            'base': {
                'base_candles': base_candles,
                'base_range': base_range,
                'proximal_line': base_range['high'] if zone_type == 'DEMAND' else base_range['low'],
                'distal_line': base_range['low'] if zone_type == 'DEMAND' else base_range['high']
            },
            'leg_out': {
                'run': run_out,
                'start_idx': run_out['start_idx'],
                'end_idx': run_out['end_idx'],
                'direction': run_out['direction'],
                'movement': run_out_movement
            },
            'valid': is_valid,
            'validation_details': validation
        }
    
    def _analyze_base_segment(self, df, gap_start, gap_end, leg_in, leg_out):
        """
        Analyze gap between monotonic runs to determine if it forms a valid base
        
        Args:
            df: DataFrame with OHLC data
            gap_start: Start index of gap
            gap_end: End index of gap  
            leg_in: Previous monotonic run
            leg_out: Next monotonic run
            
        Returns:
            dict: Base formation analysis or None
        """
        base_candles = []
        
        # Build base segment dynamically using weighted range
        current_base_range = None
        
        for i in range(gap_start, gap_end + 1):
            candle = df.iloc[i]
            candle_data = {
                'high': candle['high'],
                'low': candle['low'], 
                'open': candle['open'],
                'close': candle['close']
            }
            
            # Check conditions for base inclusion
            is_base_candle = self.classify_candle_sentiment(candle_data) == 'BASE'
            
            # If this is the first potential base candle or it overlaps with current base range
            if not base_candles:
                if is_base_candle:
                    base_candles.append(i)
                    current_base_range = self.calculate_weighted_base_range(base_candles, df)
            else:
                # Check if candle overlaps with current base range
                overlaps = self.overlaps_with_base_range(candle, current_base_range)
                
                if overlaps or is_base_candle:
                    base_candles.append(i)
                    # Recalculate weighted base range with new candle
                    current_base_range = self.calculate_weighted_base_range(base_candles, df)
                else:
                    # Break in base - this could end the base segment
                    break
        
        if not base_candles or current_base_range is None:
            return None
        
        # Determine formation type
        leg_in_direction = leg_in['direction']
        leg_out_direction = leg_out['direction']
        
        if leg_in_direction == 'UP' and leg_out_direction == 'UP':
            formation_type = 'RBR'  # Rally-Base-Rally
            zone_type = 'DEMAND'
        elif leg_in_direction == 'DOWN' and leg_out_direction == 'DOWN':
            formation_type = 'DBD'  # Drop-Base-Drop
            zone_type = 'SUPPLY'
        elif leg_in_direction == 'UP' and leg_out_direction == 'DOWN':
            formation_type = 'RBD'  # Rally-Base-Drop
            zone_type = 'SUPPLY'
        elif leg_in_direction == 'DOWN' and leg_out_direction == 'UP':
            formation_type = 'DBR'  # Drop-Base-Rally
            zone_type = 'DEMAND'
        else:
            return None
        
        # Validate formation - leg movements must exceed base range
        leg_in_movement = abs(leg_in['end_price'] - leg_in['start_price'])
        leg_out_movement = abs(leg_out['end_price'] - leg_out['start_price'])
        base_range = current_base_range['range']
        
        validation = {
            'leg_in_movement': leg_in_movement,
            'leg_out_movement': leg_out_movement,
            'base_range': base_range,
            'leg_in_valid': leg_in_movement > base_range * self.min_leg_threshold,
            'leg_out_valid': leg_out_movement > base_range * self.min_leg_threshold
        }
        
        is_valid = validation['leg_in_valid'] and validation['leg_out_valid']
        
        return {
            'formation_type': formation_type,
            'zone_type': zone_type,
            'base_candles': base_candles,
            'base_range': current_base_range,
            'proximal_line': current_base_range['high'] if zone_type == 'DEMAND' else current_base_range['low'],
            'distal_line': current_base_range['low'] if zone_type == 'DEMAND' else current_base_range['high'],
            'valid': is_valid,
            'validation': validation
        }
    
    def print_formation_summary(self, formations):
        """Print detailed formation analysis"""
        
        print(f"\n📊 FORMATION DETECTION SUMMARY")
        print("=" * 50)
        
        if not formations:
            print("❌ No formations detected")
            return
        
        valid_formations = [f for f in formations if f['valid']]
        invalid_formations = [f for f in formations if not f['valid']]
        
        print(f"✅ Total Formations: {len(formations)}")
        print(f"   Valid: {len(valid_formations)}")
        print(f"   Invalid: {len(invalid_formations)}")
        
        # Count by type
        formation_counts = {}
        for formation in valid_formations:
            ftype = formation['type']
            if ftype not in formation_counts:
                formation_counts[ftype] = 0
            formation_counts[ftype] += 1
        
        print(f"\n📈 VALID FORMATION BREAKDOWN:")
        for ftype, count in formation_counts.items():
            zone_type = 'SUPPLY' if ftype in ['DBD', 'RBD'] else 'DEMAND'
            print(f"   {ftype} ({zone_type}): {count}")
        
        print(f"\n🎯 FORMATION DETAILS:")
        for i, formation in enumerate(valid_formations, 1):
            base_info = formation['base']
            validation = formation['validation_details']
            
            print(f"\n{i}. {formation['type']} Zone ({formation['zone_type']})")
            print(f"   Range: Index {formation['start_idx']}-{formation['end_idx']}")
            print(f"   Base: {len(base_info['base_candles'])} candles, Range: ${base_info['base_range']['range']:.3f}")
            print(f"   Leg In: ${validation['leg_in_movement']:.3f} ({'✅' if validation['leg_in_valid'] else '❌'})")
            print(f"   Leg Out: ${validation['leg_out_movement']:.3f} ({'✅' if validation['leg_out_valid'] else '❌'})")
            print(f"   Proximal: ${base_info['proximal_line']:.3f}")
            print(f"   Distal: ${base_info['distal_line']:.3f}")