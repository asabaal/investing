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
        Detect monotonic runs (consecutive candles moving in same direction)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            list: List of run dictionaries with start, end, direction
        """
        runs = []
        current_run = None
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            
            if curr_close > prev_close:
                direction = 'UP'
            elif curr_close < prev_close:
                direction = 'DOWN'
            else:
                direction = 'FLAT'
            
            if current_run is None:
                # Start new run
                current_run = {
                    'start_idx': i-1,
                    'end_idx': i,
                    'direction': direction,
                    'start_price': prev_close,
                    'end_price': curr_close
                }
            elif current_run['direction'] == direction and direction != 'FLAT':
                # Continue current run
                current_run['end_idx'] = i
                current_run['end_price'] = curr_close
            else:
                # End current run and start new one
                if current_run['direction'] != 'FLAT' and current_run['end_idx'] > current_run['start_idx']:
                    runs.append(current_run)
                
                current_run = {
                    'start_idx': i-1,
                    'end_idx': i,
                    'direction': direction,
                    'start_price': prev_close,
                    'end_price': curr_close
                }
        
        # Add final run if valid
        if current_run and current_run['direction'] != 'FLAT' and current_run['end_idx'] > current_run['start_idx']:
            runs.append(current_run)
        
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
    
    def detect_formations(self, df, min_base_length=1, max_base_length=10):
        """
        Detect formations using a sliding window approach
        Look for LEG -> BASE -> LEG patterns directly
        
        Args:
            df: DataFrame with OHLC data (columns: datetime, open, high, low, close)
            min_base_length: Minimum base segment length
            max_base_length: Maximum base segment length
            
        Returns:
            list: Detected formations with detailed information
        """
        print("🎯 Starting formation detection with sliding window approach...")
        
        formations = []
        
        # Classify all candles first
        candle_sentiments = []
        for i in range(len(df)):
            row = df.iloc[i]
            candle_data = {
                'high': row['high'],
                'low': row['low'],
                'open': row['open'],
                'close': row['close']
            }
            sentiment = self.classify_candle_sentiment(candle_data)
            candle_sentiments.append(sentiment)
        
        print(f"   📊 Classified {len(candle_sentiments)} candles")
        
        # Look for LEG -> BASE -> LEG patterns
        formations_found = 0
        
        for i in range(len(df) - 2):  # Need at least 3 candles
            # Try different base lengths
            for base_length in range(min_base_length, min(max_base_length + 1, len(df) - i - 1)):
                if i + base_length + 1 >= len(df):
                    continue
                
                # Pattern: LEG(i) -> BASE(i+1 to i+base_length) -> LEG(i+base_length+1)
                leg_in_idx = i
                base_start_idx = i + 1
                base_end_idx = i + base_length
                leg_out_idx = i + base_length + 1
                
                leg_in_sentiment = candle_sentiments[leg_in_idx]
                base_sentiments = candle_sentiments[base_start_idx:base_end_idx + 1]
                leg_out_sentiment = candle_sentiments[leg_out_idx]
                
                # Check if we have a valid LEG -> BASE -> LEG pattern
                if not (leg_in_sentiment.startswith('LEG') and leg_out_sentiment.startswith('LEG')):
                    continue
                
                # Check if base segment has consolidation characteristics
                base_candles = list(range(base_start_idx, base_end_idx + 1))
                base_range = self.calculate_weighted_base_range(base_candles, df)
                
                if base_range is None:
                    continue
                
                # CRITICAL: Check if base is properly consolidated and distinct from legs
                leg_in_candle = df.iloc[leg_in_idx]
                leg_out_candle = df.iloc[leg_out_idx]
                
                leg_in_range = leg_in_candle['high'] - leg_in_candle['low']
                leg_out_range = leg_out_candle['high'] - leg_out_candle['low']
                
                # Base should be significantly smaller than both legs
                if not (base_range['range'] < leg_in_range * 0.5 and base_range['range'] < leg_out_range * 0.5):
                    continue
                
                # Base should not be completely contained within either leg candle
                base_high = base_range['high']
                base_low = base_range['low']
                
                # Check if base is contained within leg in candle
                if base_high <= leg_in_candle['high'] and base_low >= leg_in_candle['low']:
                    continue
                    
                # Check if base is contained within leg out candle  
                if base_high <= leg_out_candle['high'] and base_low >= leg_out_candle['low']:
                    continue
                
                # Check if any base candles qualify or if there's consolidation
                has_base_candles = any(sentiment == 'BASE' for sentiment in base_sentiments)
                has_consolidation = base_range['range'] < min(leg_in_range, leg_out_range) * 0.7
                
                if not (has_base_candles or has_consolidation):
                    continue
                
                # Determine formation type
                formation_type, zone_type = self._classify_formation(leg_in_sentiment, leg_out_sentiment)
                
                if formation_type is None:
                    continue
                
                # Validate formation
                leg_in_movement = self._calculate_leg_movement(df, leg_in_idx)
                leg_out_movement = self._calculate_leg_movement(df, leg_out_idx)
                
                validation = {
                    'leg_in_movement': leg_in_movement,
                    'leg_out_movement': leg_out_movement,
                    'base_range': base_range['range'],
                    'leg_in_valid': leg_in_movement > base_range['range'] * self.min_leg_threshold,
                    'leg_out_valid': leg_out_movement > base_range['range'] * self.min_leg_threshold
                }
                
                is_valid = validation['leg_in_valid'] and validation['leg_out_valid']
                
                # Create formation
                formation = {
                    'type': formation_type,
                    'zone_type': zone_type,
                    'start_idx': leg_in_idx,
                    'end_idx': leg_out_idx,
                    'leg_in': {'index': leg_in_idx, 'sentiment': leg_in_sentiment},
                    'base': {
                        'base_candles': base_candles,
                        'base_range': base_range,
                        'proximal_line': base_range['high'] if zone_type == 'DEMAND' else base_range['low'],
                        'distal_line': base_range['low'] if zone_type == 'DEMAND' else base_range['high']
                    },
                    'leg_out': {'index': leg_out_idx, 'sentiment': leg_out_sentiment},
                    'valid': is_valid,
                    'validation_details': validation
                }
                
                formations.append(formation)
                formations_found += 1
                
                # Skip ahead to avoid overlapping formations
                i += base_length + 1
                break  # Found formation with this start, move to next position
        
        print(f"   ✅ Detected {len(formations)} formations")
        return formations
    
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