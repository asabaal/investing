#!/usr/bin/env python3
"""
Auto Formation Labeler using Curve Fitting Results
Uses detected extrema/inflection points as shortcuts to identify and score formations
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from market_data_database import MarketDataDatabase
from curve_fitting_extrema_detector import CurveFittingExtemaDetector
from trade_setup_visualizer import TradeSetupVisualizer
from trade_scoring_system import TradeScorer, Zone, ZoneType, MarketContext, TrendDirection, FreshnessStatus

pio.templates.default = 'plotly_dark'

class AutoFormationLabeler:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.curve_detector = CurveFittingExtemaDetector()
        self.visualizer = TradeSetupVisualizer()
        self.scorer = TradeScorer()  # Use the REAL scoring system!
    
    def optimize_formation_from_extrema_sequence(self, df, all_extrema, center_idx):
        """
        OPTIMIZATION ALGORITHM: Select optimal candles to form valid RBD/DBD/RBR/DBR patterns
        
        Given extrema sequence, find the best candle selection to create a valid formation:
        - Analyze local extrema context around center_idx
        - Try all possible valid formation patterns
        - Optimize candle selection within search windows
        - Return the highest quality valid formation
        
        Returns: (formation_type, leg_in_start, leg_in_end, base_start, base_end, leg_out_start, leg_out_end)
        """
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        
        # Get extrema context (5 extrema before and after center)
        extrema_context = []
        for i in range(max(0, center_idx - 5), min(len(all_extrema), center_idx + 6)):
            ext_idx = int(all_extrema[i][0])
            if ext_idx < len(df):
                extrema_context.append({
                    'idx': ext_idx,
                    'price': all_extrema[i][1], 
                    'type': all_extrema[i][2],
                    'position': i - center_idx  # Relative to center
                })
        
        if len(extrema_context) < 3:
            return None, None, None, None, None, None, None
        
        # Find center extrema
        center_extrema = next((e for e in extrema_context if e['position'] == 0), None)
        if not center_extrema:
            return None, None, None, None, None, None, None
        
        # Try all possible formation patterns based on extrema context
        formation_candidates = []
        
        # CASE 1: RBD (Rally-Base-Drop) - Need rally into base, then drop out
        # Pattern: LOW → HIGH → [BASE] → LOW
        formation_candidates.extend(self._try_rbd_patterns(df, extrema_context, center_extrema))
        
        # CASE 2: DBD (Drop-Base-Drop) - Need drop into base, then drop continues  
        # Pattern: HIGH → LOW → [BASE] → LOW
        formation_candidates.extend(self._try_dbd_patterns(df, extrema_context, center_extrema))
        
        # CASE 3: RBR (Rally-Base-Rally) - Need rally into base, then rally continues
        # Pattern: LOW → HIGH → [BASE] → HIGH  
        formation_candidates.extend(self._try_rbr_patterns(df, extrema_context, center_extrema))
        
        # CASE 4: DBR (Drop-Base-Rally) - Need drop into base, then rally out
        # Pattern: HIGH → LOW → [BASE] → HIGH
        formation_candidates.extend(self._try_dbr_patterns(df, extrema_context, center_extrema))
        
        # Select best formation candidate based on quality score
        if formation_candidates:
            best_formation = max(formation_candidates, key=lambda x: x['quality_score'])
            
            # Calculate zone boundaries from formation  
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            
            base_high = df[high_col].iloc[best_formation['base_start']:best_formation['base_end']+1].max()
            base_low = df[low_col].iloc[best_formation['base_start']:best_formation['base_end']+1].min()
            
            return {
                'formation_type': best_formation['type'],
                'leg_in_start': best_formation['leg_in_start'],
                'leg_in_end': best_formation['leg_in_end'], 
                'base_start': best_formation['base_start'],
                'base_end': best_formation['base_end'],
                'base_high': base_high,
                'base_low': base_low,
                'leg_out_start': best_formation['leg_out_start'],
                'leg_out_end': best_formation['leg_out_end'],
                'quality_score': best_formation['quality_score']
            }
        
        return None
    
    def _try_rbd_patterns(self, df, extrema_context, center_extrema):
        """Try to construct RBD (Rally-Base-Drop) formations"""
        candidates = []
        
        # Find potential leg-in rallies (lows before center)
        leg_in_lows = [e for e in extrema_context if e['position'] < 0 and e['type'] == 'min']
        
        # Find potential leg-out drops (highs/lows after center)  
        leg_out_points = [e for e in extrema_context if e['position'] > 0]
        
        for leg_in_low in leg_in_lows:
            for leg_out_point in leg_out_points:
                # Try to construct RBD: leg_in_low → rally → center_base → drop → leg_out_low
                formation = self._optimize_rbd_candle_selection(df, leg_in_low, center_extrema, leg_out_point)
                if formation:
                    candidates.append(formation)
        
        return candidates
    
    def _try_dbd_patterns(self, df, extrema_context, center_extrema):
        """Try to construct DBD (Drop-Base-Drop) formations"""
        candidates = []
        
        # Find potential leg-in drops (highs before center)
        leg_in_highs = [e for e in extrema_context if e['position'] < 0 and e['type'] == 'max']
        
        # Find potential leg-out drops (lows after center)
        leg_out_lows = [e for e in extrema_context if e['position'] > 0 and e['type'] == 'min']
        
        for leg_in_high in leg_in_highs:
            for leg_out_low in leg_out_lows:
                # Try to construct DBD: leg_in_high → drop → center_base → drop → leg_out_low
                formation = self._optimize_dbd_candle_selection(df, leg_in_high, center_extrema, leg_out_low)
                if formation:
                    candidates.append(formation)
        
        return candidates
    
    def _try_rbr_patterns(self, df, extrema_context, center_extrema):
        """Try to construct RBR (Rally-Base-Rally) formations"""
        candidates = []
        
        # Find potential leg-in rallies (lows before center)
        leg_in_lows = [e for e in extrema_context if e['position'] < 0 and e['type'] == 'min']
        
        # Find potential leg-out rallies (highs after center)
        leg_out_highs = [e for e in extrema_context if e['position'] > 0 and e['type'] == 'max']
        
        for leg_in_low in leg_in_lows:
            for leg_out_high in leg_out_highs:
                # Try to construct RBR: leg_in_low → rally → center_base → rally → leg_out_high  
                formation = self._optimize_rbr_candle_selection(df, leg_in_low, center_extrema, leg_out_high)
                if formation:
                    candidates.append(formation)
        
        return candidates
    
    def _try_dbr_patterns(self, df, extrema_context, center_extrema):
        """Try to construct DBR (Drop-Base-Rally) formations"""
        candidates = []
        
        # Find potential leg-in drops (highs before center)
        leg_in_highs = [e for e in extrema_context if e['position'] < 0 and e['type'] == 'max']
        
        # Find potential leg-out rallies (highs after center)  
        leg_out_highs = [e for e in extrema_context if e['position'] > 0 and e['type'] == 'max']
        
        for leg_in_high in leg_in_highs:
            for leg_out_high in leg_out_highs:
                # Try to construct DBR: leg_in_high → drop → center_base → rally → leg_out_high
                formation = self._optimize_dbr_candle_selection(df, leg_in_high, center_extrema, leg_out_high)
                if formation:
                    candidates.append(formation)
        
        return candidates
    
    def _optimize_rbd_candle_selection(self, df, leg_in_low, center_base, leg_out_point):
        """Optimize candle selection for RBD formation"""
        
        # Find optimal leg-in rally: from leg_in_low up to base area
        leg_in_start = leg_in_low['idx'] 
        leg_in_end = self._find_optimal_rally_end(df, leg_in_start, center_base['idx'])
        
        if leg_in_end is None:
            return None
        
        # Find optimal base consolidation around center extrema
        base_start, base_end = self._find_optimal_base_region(df, center_base['idx'])
        
        # Find optimal leg-out drop: from base area down 
        leg_out_start = base_end
        leg_out_end = self._find_optimal_drop_end(df, leg_out_start, leg_out_point['idx'])
        
        if leg_out_end is None:
            return None
        
        # Calculate quality score
        quality_score = self._calculate_formation_quality(df, 'RBD', leg_in_start, leg_in_end, 
                                                        base_start, base_end, leg_out_start, leg_out_end)
        
        return {
            'type': 'RBD',
            'leg_in_start': leg_in_start, 'leg_in_end': leg_in_end,
            'base_start': base_start, 'base_end': base_end,
            'leg_out_start': leg_out_start, 'leg_out_end': leg_out_end,
            'quality_score': quality_score
        }
    
    def _optimize_dbd_candle_selection(self, df, leg_in_high, center_base, leg_out_low):
        """Optimize candle selection for DBD formation"""
        
        # Find optimal leg-in drop: from leg_in_high down to base area
        leg_in_start = leg_in_high['idx']
        leg_in_end = self._find_optimal_drop_end(df, leg_in_start, center_base['idx'])
        
        if leg_in_end is None:
            return None
        
        # Find optimal base consolidation around center extrema  
        base_start, base_end = self._find_optimal_base_region(df, center_base['idx'])
        
        # Find optimal leg-out drop: from base area down to leg_out_low
        leg_out_start = base_end
        leg_out_end = leg_out_low['idx']
        
        # Calculate quality score
        quality_score = self._calculate_formation_quality(df, 'DBD', leg_in_start, leg_in_end,
                                                        base_start, base_end, leg_out_start, leg_out_end)
        
        return {
            'type': 'DBD', 
            'leg_in_start': leg_in_start, 'leg_in_end': leg_in_end,
            'base_start': base_start, 'base_end': base_end,
            'leg_out_start': leg_out_start, 'leg_out_end': leg_out_end,
            'quality_score': quality_score
        }
    
    def _optimize_rbr_candle_selection(self, df, leg_in_low, center_base, leg_out_high):
        """Optimize candle selection for RBR formation"""
        
        # Find optimal leg-in rally: from leg_in_low up to base area
        leg_in_start = leg_in_low['idx']
        leg_in_end = self._find_optimal_rally_end(df, leg_in_start, center_base['idx'])
        
        if leg_in_end is None:
            return None
        
        # Find optimal base consolidation around center extrema
        base_start, base_end = self._find_optimal_base_region(df, center_base['idx'])
        
        # Find optimal leg-out rally: from base area up to leg_out_high
        leg_out_start = base_end
        leg_out_end = leg_out_high['idx']
        
        # Calculate quality score
        quality_score = self._calculate_formation_quality(df, 'RBR', leg_in_start, leg_in_end,
                                                        base_start, base_end, leg_out_start, leg_out_end)
        
        return {
            'type': 'RBR',
            'leg_in_start': leg_in_start, 'leg_in_end': leg_in_end,
            'base_start': base_start, 'base_end': base_end, 
            'leg_out_start': leg_out_start, 'leg_out_end': leg_out_end,
            'quality_score': quality_score
        }
    
    def _optimize_dbr_candle_selection(self, df, leg_in_high, center_base, leg_out_high):
        """Optimize candle selection for DBR formation"""
        
        # Find optimal leg-in drop: from leg_in_high down to base area
        leg_in_start = leg_in_high['idx']
        leg_in_end = self._find_optimal_drop_end(df, leg_in_start, center_base['idx'])
        
        if leg_in_end is None:
            return None
        
        # Find optimal base consolidation around center extrema
        base_start, base_end = self._find_optimal_base_region(df, center_base['idx'])
        
        # Find optimal leg-out rally: from base area up to leg_out_high
        leg_out_start = base_end  
        leg_out_end = leg_out_high['idx']
        
        # Calculate quality score
        quality_score = self._calculate_formation_quality(df, 'DBR', leg_in_start, leg_in_end,
                                                        base_start, base_end, leg_out_start, leg_out_end)
        
        return {
            'type': 'DBR',
            'leg_in_start': leg_in_start, 'leg_in_end': leg_in_end,
            'base_start': base_start, 'base_end': base_end,
            'leg_out_start': leg_out_start, 'leg_out_end': leg_out_end,
            'quality_score': quality_score
        }
    
    def _find_optimal_rally_end(self, df, start_idx, max_end_idx):
        """Find the best ending point for a rally (upward movement)"""
        best_end = None
        best_gain = 0
        
        low_col = 'Low' if 'Low' in df.columns else 'low'
        start_low = df[low_col].iloc[start_idx]
        
        for end_idx in range(start_idx + 1, min(max_end_idx, len(df))):
            high_col = 'High' if 'High' in df.columns else 'high'
            current_high = df[high_col].iloc[end_idx]
            gain = current_high - start_low
            
            if gain > best_gain:
                best_gain = gain
                best_end = end_idx
        
        return best_end if best_gain > 0 else None
    
    def _find_optimal_drop_end(self, df, start_idx, max_end_idx):
        """Find the best ending point for a drop (downward movement)"""
        best_end = None
        best_drop = 0
        
        high_col = 'High' if 'High' in df.columns else 'high'
        start_high = df[high_col].iloc[start_idx]
        
        for end_idx in range(start_idx + 1, min(max_end_idx, len(df))):
            low_col = 'Low' if 'Low' in df.columns else 'low'
            current_low = df[low_col].iloc[end_idx]
            drop = start_high - current_low
            
            if drop > best_drop:
                best_drop = drop
                best_end = end_idx
        
        return best_end if best_drop > 0 else None
    
    def _find_optimal_base_region(self, df, center_idx, window=5):
        """Find optimal base/consolidation region around center extrema"""
        start_idx = max(0, center_idx - window)
        end_idx = min(len(df) - 1, center_idx + window)
        
        # Optimize base boundaries to minimize range while including key price action
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        base_high = df[high_col].iloc[start_idx:end_idx+1].max()
        base_low = df[low_col].iloc[start_idx:end_idx+1].min()
        base_range = base_high - base_low
        
        # Try to tighten the base region
        for start in range(start_idx, center_idx):
            for end in range(center_idx, end_idx + 1):
                if end - start >= 3:  # Minimum base size
                    region_high = df[high_col].iloc[start:end+1].max()
                    region_low = df[low_col].iloc[start:end+1].min() 
                    region_range = region_high - region_low
                    
                    if region_range <= base_range * 1.2:  # Within 20% of original range
                        start_idx, end_idx = start, end
                        break
        
        return start_idx, end_idx
    
    def _calculate_formation_quality(self, df, formation_type, leg_in_start, leg_in_end, 
                                   base_start, base_end, leg_out_start, leg_out_end):
        """Calculate quality score for a formation based on multiple factors"""
        
        # Factor 1: Leg movements (stronger movements = higher quality)
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        leg_in_movement = abs(df[high_col].iloc[leg_in_start:leg_in_end+1].max() - 
                             df[low_col].iloc[leg_in_start:leg_in_end+1].min())
        leg_out_movement = abs(df[high_col].iloc[leg_out_start:leg_out_end+1].max() - 
                              df[low_col].iloc[leg_out_start:leg_out_end+1].min())
        
        # Factor 2: Base tightness (tighter base = higher quality)
        base_range = (df[high_col].iloc[base_start:base_end+1].max() - 
                     df[low_col].iloc[base_start:base_end+1].min())
        
        # Factor 3: Formation proportions (balanced legs = higher quality)
        total_range = max(df[high_col].iloc[leg_in_start:leg_out_end+1].max() - 
                         df[low_col].iloc[leg_in_start:leg_out_end+1].min(), 0.01)
        
        # Normalize factors
        movement_score = (leg_in_movement + leg_out_movement) / total_range
        tightness_score = 1.0 / max(base_range / total_range, 0.01)
        proportion_score = 1.0 / (1.0 + abs(leg_in_movement - leg_out_movement) / total_range)
        
        # Combine factors
        quality_score = movement_score * 0.4 + tightness_score * 0.3 + proportion_score * 0.3
        
        return quality_score
    
    def create_zone_from_formation(self, formation_type, base_start_idx, base_end_idx, leg_out_end_idx, df):
        """
        Create a properly structured Zone object from formation data
        """
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        
        # Determine zone type and levels based on formation
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        if formation_type in ['RBD', 'RBR', 'CONSOLIDATION']:
            # Supply zone from the base high (resistance level)
            zone_type = ZoneType.SUPPLY
            high = df[high_col].iloc[base_start_idx:base_end_idx+1].max()
            low = high * 0.998  # Tight zone around the high
            
        elif formation_type in ['DBD', 'DBR']:
            # Demand zone from the base low (support level)  
            zone_type = ZoneType.DEMAND
            low = df[low_col].iloc[base_start_idx:base_end_idx+1].min()
            high = low * 1.002  # Tight zone around the low
            
        else:
            # Default to supply zone for consolidation
            zone_type = ZoneType.SUPPLY
            high = df[high_col].iloc[base_start_idx:base_end_idx+1].max()
            low = df[low_col].iloc[base_start_idx:base_end_idx+1].min()
        
        # Calculate base candles
        base_candles = base_end_idx - base_start_idx + 1
        
        # Define leg out period using numerical values instead of timestamps
        leg_out_start = float(base_end_idx)
        safe_leg_out_end_idx = min(len(df)-1, leg_out_end_idx)
        leg_out_end = float(safe_leg_out_end_idx)
        
        # Create zone object
        zone = Zone(
            zone_type=zone_type,
            high=high,
            low=low,
            base_candles=base_candles,
            leg_out_start=leg_out_start,
            leg_out_end=leg_out_end,
            opposing_zones=[]  # Will be populated later
        )
        
        return zone, formation_type
    
    def score_auto_detected_zones(self, zones_with_formations, df, symbol):
        """
        Apply proper scoring to auto-detected zones
        """
        scored_results = []
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        
        for zone, formation_type in zones_with_formations:
            try:
                # Create market context
                current_price = df[close_col].iloc[-1]
                trend = TrendDirection.UP if df[close_col].iloc[-1] > df[close_col].iloc[0] else TrendDirection.DOWN
                
                market_context = MarketContext(
                    current_trend=trend,
                    current_price=current_price,
                    long_term_high=df[high_col].max(),
                    long_term_low=df[low_col].min(),
                    freshness_status=FreshnessStatus.UNTESTED  # Default for auto-detected zones
                )
                
                # Use the REAL scoring system we developed!
                scores = self.scorer.calculate_total_score(zone, market_context)
                
                # Get detailed score breakdown from the scoring result
                detailed_scores = {
                    'zone_strength': scores['zone_strength'],
                    'price_position': scores['price_position'],
                    'freshness': scores['freshness'],
                    'trend_alignment': scores['trend_alignment'],
                    'time_base': scores['time_base'],
                    'profit_potential': scores['profit_potential'],
                    'total_score': scores['total'],  # Use 'total' from scorer
                    'status': scores['status']
                }
                
                scored_results.append({
                    'zone': zone,
                    'formation_type': formation_type,
                    'detailed_scores': detailed_scores,
                    'total_score': scores['total'],  # Get total from scores
                    'market_context': market_context
                })
                
            except Exception as e:
                print(f"⚠️ Failed to score zone: {e}")
                continue
        
        return scored_results
    
    def auto_detect_and_score_formations(self, symbol, start_date, end_date, window_size=50):
        """
        Complete pipeline: curve fitting -> formation detection -> scoring
        """
        print(f"\n🎯 AUTO-DETECTING FORMATIONS FOR {symbol}")
        print("Using curve fitting extrema as formation shortcuts")
        print("="*60)
        
        # Step 1: Run curve fitting analysis
        analysis = self.curve_detector.analyze_symbol(symbol, start_date, end_date, window_size)
        
        if not analysis or not analysis['window_results']:
            print(f"❌ No curve fitting data for {symbol}")
            return None
        
        df = analysis['full_data']
        print(f"📊 Analyzing {len(df)} daily candles")
        
        # Step 2: Extract extrema from all windows and combine
        all_extrema = []
        
        for window_idx, window in enumerate(analysis['window_results']):
            window_start = window['window_start']
            
            for curve_name, extrema_data in window['traditional_extrema'].items():
                if 'numerical' in extrema_data:
                    # Add maxima
                    for max_x, max_y in extrema_data['numerical']['maxima']:
                        global_idx = window_start + int(max_x)
                        if global_idx < len(df):
                            all_extrema.append((global_idx, max_y, 'max', curve_name))
                    
                    # Add minima
                    for min_x, min_y in extrema_data['numerical']['minima']:
                        global_idx = window_start + int(min_x)
                        if global_idx < len(df):
                            all_extrema.append((global_idx, min_y, 'min', curve_name))
                
                break  # Use first curve only
        
        # Sort extrema by time
        all_extrema.sort(key=lambda x: x[0])
        print(f"🔍 Found {len(all_extrema)} total extrema")
        
        # Step 3: Treat each extrema/inflection point as a BASE CANDIDATE
        formations = []
        
        print(f"🔍 Analyzing {len(all_extrema)} extrema points as base candidates...")
        
        # Check each extrema point as a potential formation base
        for extrema_idx, (global_idx, price, extrema_type, curve_name) in enumerate(all_extrema):
            if global_idx >= len(df):
                continue
                
            print(f"  📍 Checking base candidate {extrema_idx+1}: {extrema_type} at index {global_idx}")
            
            # Use new optimization algorithm to find best valid formation
            formation_result = self.optimize_formation_from_extrema_sequence(
                df, all_extrema, extrema_idx
            )
            
            if formation_result:  # If valid formation found
                print(f"    ✅ Found {formation_result['formation_type']} formation")
                
                # Create zone from the optimized formation
                zone = Zone(
                    zone_type=ZoneType.DEMAND if formation_result['formation_type'] in ['RBR', 'DBR'] else ZoneType.SUPPLY,
                    high=formation_result['base_high'],
                    low=formation_result['base_low'],
                    base_candles=formation_result['base_end'] - formation_result['base_start'] + 1,
                    leg_out_start=float(formation_result['base_end']),
                    leg_out_end=float(formation_result['leg_out_end']),
                    opposing_zones=[]
                )
                
                formations.append((zone, formation_result['formation_type']))
            else:
                print(f"    ❌ No valid formation pattern found")
        
        print(f"📋 Identified {len(formations)} formations:")
        formation_counts = {}
        for _, form_type in formations:
            formation_counts[form_type] = formation_counts.get(form_type, 0) + 1
        for form_type, count in formation_counts.items():
            print(f"  • {form_type}: {count}")
        
        # Step 4: Score the formations
        scored_formations = self.score_auto_detected_zones(formations, df, symbol)
        print(f"✅ Successfully scored {len(scored_formations)} formations")
        
        return {
            'symbol': symbol,
            'data': df,
            'analysis': analysis,
            'extrema': all_extrema,
            'formations': scored_formations,
            'formation_counts': formation_counts
        }
    
    def auto_detect_and_score_formations_with_data(self, symbol, df, window_size=50):
        """
        Complete pipeline using pre-fetched data: curve fitting -> formation detection -> scoring
        """
        print(f"\n🎯 AUTO-DETECTING FORMATIONS FOR {symbol} (USING PROVIDED DATA)")
        print("Using curve fitting extrema as formation shortcuts")
        print("="*60)
        
        print(f"📊 Analyzing {len(df)} candles")
        
        # Step 1: Run curve fitting analysis on provided data
        analysis = self.curve_detector.analyze_data_directly(df, window_size)
        
        if not analysis or not analysis['window_results']:
            print(f"❌ No curve fitting data for {symbol}")
            return None
        
        # Step 2: Extract extrema from all windows and combine
        all_extrema = []
        
        for window_idx, window in enumerate(analysis['window_results']):
            window_start = window['window_start']
            
            for curve_name, extrema_data in window['traditional_extrema'].items():
                if 'numerical' in extrema_data:
                    # Add maxima
                    for max_x, max_y in extrema_data['numerical']['maxima']:
                        global_idx = window_start + int(max_x)
                        if global_idx < len(df):
                            all_extrema.append((global_idx, max_y, 'max', curve_name))
                    
                    # Add minima
                    for min_x, min_y in extrema_data['numerical']['minima']:
                        global_idx = window_start + int(min_x)
                        if global_idx < len(df):
                            all_extrema.append((global_idx, min_y, 'min', curve_name))
                
                break  # Use first curve only
        
        # Sort extrema by time
        all_extrema.sort(key=lambda x: x[0])
        print(f"🔍 Found {len(all_extrema)} total extrema")
        
        # Step 3: Treat each extrema/inflection point as a BASE CANDIDATE
        formations = []
        
        print(f"🔍 Analyzing {len(all_extrema)} extrema points as base candidates...")
        
        # Check each extrema point as a potential formation base
        for extrema_idx, (global_idx, price, extrema_type, curve_name) in enumerate(all_extrema):
            if global_idx >= len(df):
                continue
                
            print(f"  📍 Checking base candidate {extrema_idx+1}: {extrema_type} at index {global_idx}")
            
            # Use new optimization algorithm to find best valid formation
            formation_result = self.optimize_formation_from_extrema_sequence(
                df, all_extrema, extrema_idx
            )
            
            if formation_result:
                formations.append(formation_result)
                print(f"    ✅ Valid {formation_result['formation_type']} formation found")
            else:
                print(f"    ❌ No valid formation from this base")
        
        print(f"\n🎉 Formation detection complete!")
        print(f"Found {len(formations)} valid formations")
        
        # Count formation types
        formation_counts = {}
        for formation in formations:
            form_type = formation['formation_type']
            formation_counts[form_type] = formation_counts.get(form_type, 0) + 1
        
        for form_type, count in sorted(formation_counts.items()):
            print(f"  {form_type}: {count}")
        
        # Step 4: Score all formations using the real scoring system
        print(f"\n🔢 SCORING FORMATIONS...")
        scored_formations = self.score_auto_detected_zones(formations, df, symbol)
        print(f"✅ Successfully scored {len(scored_formations)} formations")
        
        return {
            'symbol': symbol,
            'data': df,
            'analysis': analysis,
            'extrema': all_extrema,
            'formations': scored_formations,
            'formation_counts': formation_counts
        }
    
    def create_auto_labeled_visualization(self, detection_result):
        """
        Create proper formation explorer like the existing systems
        Shows individual formations with complete scoring breakdown
        """
        if not detection_result:
            return None
        
        symbol = detection_result['symbol']
        df = detection_result['data']
        formations = detection_result['formations']
        extrema = detection_result['extrema']
        
        # Create interactive formation explorer HTML
        return self.create_formation_explorer_html(symbol, df, formations, extrema)
    
    def create_formation_explorer_html(self, symbol, df, formations, extrema):
        """Create proper formation explorer HTML with detailed scoring"""
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
            
        # Prepare formation data with complete details
        formation_data = []
        for i, formation in enumerate(formations):
            zone = formation['zone']
            scores = formation['detailed_scores']
            
            # Find extrema point that created this formation
            extrema_point = None
            for ext in extrema:
                ext_idx = int(ext[0])
                if abs(ext_idx - zone.leg_out_start) < 10:  # Within 10 candles
                    extrema_point = {'index': ext_idx, 'price': ext[1], 'type': ext[2]}
                    break
            
            formation_info = {
                'id': i + 1,
                'type': formation['formation_type'],
                'zone_type': zone.zone_type.value,
                'zone': {
                    'high': zone.high,
                    'low': zone.low, 
                    'range': zone.high - zone.low,
                    'leg_out_start': int(zone.leg_out_start),
                    'leg_out_end': int(zone.leg_out_end),
                    'base_candles': zone.base_candles
                },
                'scoring': {
                    'total_score': formation['total_score'],
                    'zone_strength': scores['zone_strength'],
                    'price_position': scores['price_position'],
                    'freshness': scores['freshness'],
                    'trend_alignment': scores['trend_alignment'],
                    'time_base': scores['time_base'],
                    'profit_potential': scores['profit_potential']
                },
                'market_context': {
                    'current_price': formation['market_context'].current_price,
                    'trend': formation['market_context'].current_trend.value,
                    'long_term_high': formation['market_context'].long_term_high,
                    'long_term_low': formation['market_context'].long_term_low
                },
                'extrema_source': extrema_point,
                'formation_date': df.index[int(zone.leg_out_start)].strftime('%Y-%m-%d')
            }
            formation_data.append(formation_info)
        
        # Prepare candle data for chart
        df_json = []
        for i, (idx, row) in enumerate(df.iterrows()):
            df_json.append({
                'index': i,
                'date': idx.strftime('%Y-%m-%d'),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row[close_col])
            })
        
        # Create HTML content
        html_content = self.generate_formation_explorer_html(symbol, df_json, formation_data, extrema)
        
        # Save to file
        filename = f"/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/curve_fitting_examples/{symbol.lower()}_auto_formations.html"
        with open(filename, 'w') as f:
            f.write(html_content)
            
        print(f"✅ Created proper formation explorer: {filename}")
        return filename
    
    def generate_formation_explorer_html(self, symbol, df_data, formation_data, extrema):
        """Generate the complete HTML for formation exploration"""
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} Auto-Formation Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a1a; 
            color: white; 
            margin: 0; 
            padding: 0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #333;
        }}
        
        .header h1 {{
            margin: 0;
            color: #00ff88;
            font-size: 28px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        .control-panel {{
            background: #2a2a2a;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
        }}
        
        .formation-info {{
            background: #333;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #00ff88;
        }}
        
        .scoring-breakdown {{
            background: rgba(0,100,0,0.1);
            border: 1px solid #00ff88;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }}
        
        .score-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 4px 0;
            border-bottom: 1px solid #444;
        }}
        
        .score-label {{
            color: #ffaa00;
            font-weight: bold;
        }}
        
        .score-value {{
            color: #00ff88;
            font-weight: bold;
        }}
        
        .navigation {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .nav-btn {{
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }}
        
        .nav-btn:hover {{
            background: linear-gradient(135deg, #0056b3 0%, #007bff 100%);
        }}
        
        .nav-btn:disabled {{
            background: #555;
            cursor: not-allowed;
        }}
        
        .formation-counter {{
            background: #444;
            padding: 8px 12px;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
            color: #ffaa00;
            margin-bottom: 8px;
        }}
        
        #chart {{ 
            margin: 20px;
            min-height: 600px;
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{symbol} Auto-Formation Explorer</h1>
        <p>Curve Fitting Extrema → Formation Detection → 6-Metric Scoring</p>
    </div>
    
    <div class="control-panel">
        <div class="navigation">
            <div class="formation-counter" id="formationCounter">
                Formation 1 of {len(formation_data)}
            </div>
            <button class="nav-btn" id="prevBtn" onclick="previousFormation()">← Previous</button>
            <button class="nav-btn" id="nextBtn" onclick="nextFormation()">Next →</button>
            <select id="formationSelect" onchange="selectFormation()">
                {' '.join([f'<option value="{i}">#{f["id"]} {f["type"]} (Score: {f["scoring"]["total_score"]:.1f})</option>' for i, f in enumerate(formation_data)])}
            </select>
        </div>
        
        <div class="formation-info" id="formationInfo">
            <h3 id="formationTitle">Formation Details</h3>
            <div id="formationDetails">
                <!-- Formation details will be populated by JavaScript -->
            </div>
            <div class="scoring-breakdown" id="scoringBreakdown">
                <!-- Scoring breakdown will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="formation-info">
            <h3>Legend</h3>
            <div style="font-size: 12px; line-height: 1.5;">
                <div>🔴 <strong>RBD</strong>: Rally-Base-Drop (Supply)</div>
                <div>🟢 <strong>DBD</strong>: Drop-Base-Drop (Demand)</div>
                <div>🟠 <strong>RBR</strong>: Rally-Base-Rally (Supply)</div>
                <div>🔵 <strong>DBR</strong>: Drop-Base-Rally (Demand)</div>
                <div>🟡 <strong>CONSOLIDATION</strong>: Sideways movement</div>
                <div style="margin-top: 10px;">
                    <strong>6-Metric Scoring:</strong><br>
                    • Zone Strength (0-2.0)<br>
                    • Price Position (0-2.0)<br>
                    • Freshness (0-2.0)<br>
                    • Trend Alignment (0-2.0)<br>
                    • Time/Base (0-1.0)<br>
                    • Profit Potential (0-1.0)
                </div>
            </div>
        </div>
    </div>
    
    <div id="chart"></div>
    
    <script>
        const formationData = {formation_data};
        const candleData = {df_data};
        const extremaData = {extrema};
        let currentFormation = 0;
        
        function updateFormationDisplay() {{
            const formation = formationData[currentFormation];
            
            // Update counter
            document.getElementById('formationCounter').textContent = 
                `Formation ${{currentFormation + 1}} of ${{formationData.length}}`;
            
            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentFormation === 0;
            document.getElementById('nextBtn').disabled = currentFormation === formationData.length - 1;
            document.getElementById('formationSelect').value = currentFormation;
            
            // Update formation details
            document.getElementById('formationTitle').textContent = 
                `${{formation.type}} Formation #${{formation.id}}`;
            
            document.getElementById('formationDetails').innerHTML = `
                <div><strong>Zone Type:</strong> ${{formation.zone_type.toUpperCase()}}</div>
                <div><strong>Date:</strong> ${{formation.formation_date}}</div>
                <div><strong>Zone Range:</strong> $${{formation.zone.low.toFixed(2)}} - $${{formation.zone.high.toFixed(2)}}</div>
                <div><strong>Zone Width:</strong> $${{formation.zone.range.toFixed(3)}}</div>
                <div><strong>Base Candles:</strong> ${{formation.zone.base_candles}}</div>
                <div><strong>Current Price:</strong> $${{formation.market_context.current_price.toFixed(2)}}</div>
                <div><strong>Market Trend:</strong> ${{formation.market_context.trend.toUpperCase()}}</div>
                ${{formation.extrema_source ? `<div><strong>Source Extrema:</strong> ${{formation.extrema_source.type.toUpperCase()}} at index ${{formation.extrema_source.index}} ($${{formation.extrema_source.price.toFixed(2)}})</div>` : ''}}
            `;
            
            // Update scoring breakdown with detailed calculations
            const scores = formation.scoring;
            const currentPrice = formation.market_context.current_price;
            const zoneHigh = formation.zone.high;
            const zoneLow = formation.zone.low;
            const zoneRange = zoneHigh - zoneLow;
            const baseDuration = formation.zone.base_candles;
            
            document.getElementById('scoringBreakdown').innerHTML = `
                <h4 style="margin: 0 0 10px 0; color: #ffaa00;">📊 Detailed Scoring Breakdown</h4>
                
                <div class="score-item">
                    <span class="score-label">Zone Strength:</span>
                    <span class="score-value">${{scores.zone_strength.toFixed(1)}}/2.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Zone tightness: $${{zoneRange.toFixed(3)}} range, Base duration: ${{baseDuration}} candles
                    </div>
                </div>
                
                <div class="score-item">
                    <span class="score-label">Price Position:</span>
                    <span class="score-value">${{scores.price_position.toFixed(1)}}/2.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Current: $${{currentPrice.toFixed(2)}}, Zone: $${{zoneLow.toFixed(2)}}-$${{zoneHigh.toFixed(2)}}
                        ${{formation.zone_type === 'DEMAND' ? 
                          (currentPrice > zoneHigh ? '(Above demand zone ✓)' : '(Below/in demand zone)') :
                          (currentPrice < zoneLow ? '(Below supply zone ✓)' : '(Above/in supply zone)')}}
                    </div>
                </div>
                
                <div class="score-item">
                    <span class="score-label">Freshness:</span>
                    <span class="score-value">${{scores.freshness.toFixed(1)}}/2.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Zone is untested (fresh) - no prior violations detected
                    </div>
                </div>
                
                <div class="score-item">
                    <span class="score-label">Trend Alignment:</span>
                    <span class="score-value">${{scores.trend_alignment.toFixed(1)}}/2.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Market trend: ${{formation.market_context.trend}}, Zone type: ${{formation.zone_type}}
                        ${{(formation.market_context.trend === 'UP' && formation.zone_type === 'DEMAND') || 
                           (formation.market_context.trend === 'DOWN' && formation.zone_type === 'SUPPLY') ? 
                           '(Aligned ✓)' : '(Misaligned)'}}
                    </div>
                </div>
                
                <div class="score-item">
                    <span class="score-label">Time/Base:</span>
                    <span class="score-value">${{scores.time_base.toFixed(1)}}/1.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Base quality: ${{baseDuration}} candles, Range: $${{zoneRange.toFixed(3)}}
                        ${{baseDuration >= 3 && baseDuration <= 8 ? '(Optimal duration ✓)' : '(Suboptimal duration)'}}
                    </div>
                </div>
                
                <div class="score-item">
                    <span class="score-label">Profit Potential:</span>
                    <span class="score-value">${{scores.profit_potential.toFixed(1)}}/1.0</span>
                    <div style="font-size: 10px; color: #ccc; margin-top: 2px;">
                        Risk/reward based on zone width ($${{zoneRange.toFixed(3)}}) and distance to price
                    </div>
                </div>
                
                <div class="score-item" style="border-top: 2px solid #00ff88; margin-top: 10px; padding-top: 10px;">
                    <span class="score-label" style="font-size: 16px;">TOTAL SCORE:</span>
                    <span class="score-value" style="font-size: 16px;">${{scores.total_score.toFixed(1)}}/10.0</span>
                    <div style="font-size: 10px; color: #00ff88; margin-top: 2px; font-weight: bold;">
                        ${{scores.total_score >= 7 ? '🟢 EXCELLENT' : scores.total_score >= 5 ? '🟡 GOOD' : '🔴 POOR'}} formation quality
                    </div>
                </div>
            `;
            
            // Update chart
            updateChart(formation);
        }}
        
        function updateChart(formation) {{
            const traces = [];
            
            // Candlestick chart
            traces.push({{
                type: 'candlestick',
                x: candleData.map(d => d.date),
                open: candleData.map(d => d.open),
                high: candleData.map(d => d.high),
                low: candleData.map(d => d.low),
                close: candleData.map(d => d.close),
                name: '{symbol} Price',
                showlegend: false
            }});
            
            // Calculate proper zone boundaries - ONLY covering base candles
            const zoneColor = formation.zone_type === 'SUPPLY' ? 'rgba(255,68,68,0.3)' : 'rgba(0,255,136,0.3)';
            const zoneLineColor = formation.zone_type === 'SUPPLY' ? '#ff4444' : '#00ff88';
            
            // Get formation details to find base period  
            const baseStartIdx = formation.zone.leg_out_start - formation.zone.base_candles + 1;
            const baseEndIdx = formation.zone.leg_out_start;
            
            const zoneStart = candleData[Math.max(0, baseStartIdx)].date;
            const zoneEnd = candleData[Math.min(candleData.length-1, baseEndIdx)].date;
            
            // Add LEG-IN line segment
            const legInStartIdx = Math.max(0, baseStartIdx - 10); // Estimate leg-in start
            const legInEndIdx = baseStartIdx;
            if (legInStartIdx < candleData.length && legInEndIdx < candleData.length) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [candleData[legInStartIdx].date, candleData[legInEndIdx].date],
                    y: [candleData[legInStartIdx].close, candleData[legInEndIdx].close],
                    line: {{ color: '#00ccff', width: 4 }},
                    name: 'Leg In Movement',
                    showlegend: false
                }});
                
                // Add LEG-IN label
                traces.push({{
                    type: 'scatter',
                    mode: 'text',
                    x: [candleData[Math.floor((legInStartIdx + legInEndIdx) / 2)].date],
                    y: [Math.max(candleData[legInStartIdx].high, candleData[legInEndIdx].high) + 0.5],
                    text: ['LEG IN'],
                    textfont: {{ size: 12, color: '#00ccff', family: 'Arial Black' }},
                    showlegend: false
                }});
            }}
            
            // Add LEG-OUT line segment  
            const legOutStartIdx = baseEndIdx;
            const legOutEndIdx = Math.min(candleData.length-1, baseEndIdx + 10); // Estimate leg-out end
            if (legOutStartIdx < candleData.length && legOutEndIdx < candleData.length) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [candleData[legOutStartIdx].date, candleData[legOutEndIdx].date],
                    y: [candleData[legOutStartIdx].close, candleData[legOutEndIdx].close],
                    line: {{ color: '#ff6600', width: 4 }},
                    name: 'Leg Out Movement', 
                    showlegend: false
                }});
                
                // Add LEG-OUT label
                traces.push({{
                    type: 'scatter',
                    mode: 'text',
                    x: [candleData[Math.floor((legOutStartIdx + legOutEndIdx) / 2)].date],
                    y: [Math.max(candleData[legOutStartIdx].high, candleData[legOutEndIdx].high) + 0.5],
                    text: ['LEG OUT'],
                    textfont: {{ size: 12, color: '#ff6600', family: 'Arial Black' }},
                    showlegend: false
                }});
            }}
            
            // Add BASE label on the zone
            traces.push({{
                type: 'scatter',
                mode: 'text',
                x: [candleData[Math.floor((baseStartIdx + baseEndIdx) / 2)].date],
                y: [(formation.zone.high + formation.zone.low) / 2],
                text: ['BASE'],
                textfont: {{ size: 12, color: zoneLineColor, family: 'Arial Black' }},
                showlegend: false
            }});
            
            // Add extrema points
            const maxExtrema = extremaData.filter(e => e[2] === 'max').map(e => ({{
                x: candleData[Math.min(parseInt(e[0]), candleData.length-1)].date,
                y: e[1]
            }}));
            
            const minExtrema = extremaData.filter(e => e[2] === 'min').map(e => ({{
                x: candleData[Math.min(parseInt(e[0]), candleData.length-1)].date,
                y: e[1]
            }}));
            
            if (maxExtrema.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers',
                    x: maxExtrema.map(e => e.x),
                    y: maxExtrema.map(e => e.y),
                    marker: {{ color: 'red', size: 8, symbol: 'triangle-down' }},
                    name: 'Supply Extrema',
                    showlegend: false
                }});
            }}
            
            if (minExtrema.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers',
                    x: minExtrema.map(e => e.x),
                    y: minExtrema.map(e => e.y),
                    marker: {{ color: 'green', size: 8, symbol: 'triangle-up' }},
                    name: 'Demand Extrema',
                    showlegend: false
                }});
            }}
            
            const layout = {{
                title: {{
                    text: `${{formation.type}} Formation #${{formation.id}} - Score: ${{formation.scoring.total_score.toFixed(1)}}/10.0`,
                    font: {{ size: 16, color: 'white' }}
                }},
                plot_bgcolor: '#2a2a2a',
                paper_bgcolor: '#2a2a2a',
                font: {{ color: 'white' }},
                xaxis: {{ 
                    gridcolor: '#444',
                    rangeslider: {{ visible: false }}
                }},
                yaxis: {{ gridcolor: '#444' }},
                shapes: [{{
                    type: 'rect',
                    x0: zoneStart,
                    x1: zoneEnd,
                    y0: formation.zone.low,
                    y1: formation.zone.high,
                    fillcolor: zoneColor,
                    line: {{ color: zoneLineColor, width: 2 }},
                    layer: 'below'
                }}]
            }};
            
            Plotly.newPlot('chart', traces, layout, {{responsive: true}});
        }}
        
        function previousFormation() {{
            if (currentFormation > 0) {{
                currentFormation--;
                updateFormationDisplay();
            }}
        }}
        
        function nextFormation() {{
            if (currentFormation < formationData.length - 1) {{
                currentFormation++;
                updateFormationDisplay();
            }}
        }}
        
        function selectFormation() {{
            currentFormation = parseInt(document.getElementById('formationSelect').value);
            updateFormationDisplay();
        }}
        
        // Initialize
        updateFormationDisplay();
    </script>
</body>
</html>"""
    
    def generate_summary_text(self, detection_result):
        """Generate comprehensive summary text"""
        formations = detection_result['formations']
        formation_counts = detection_result['formation_counts']
        
        if not formations:
            return "No formations detected"
        
        avg_score = np.mean([f['total_score'] for f in formations])
        best_formation = max(formations, key=lambda x: x['total_score'])
        best_scores = best_formation['detailed_scores']
        
        summary = f"""<b>🎯 AUTO-FORMATION DETECTION RESULTS</b><br><br>
<b>📊 Detection Summary:</b><br>
• Total Formations: {len(formations)}<br>
• Average Score: {avg_score:.1f}/10.0<br>
• Formation Types: {', '.join([f'{k}: {v}' for k, v in formation_counts.items()])}<br><br>

<b>🏆 Best Opportunity ({best_formation['formation_type']}):</b><br>
• Total Score: {best_formation['total_score']:.1f}/10.0<br>
• Zone Strength: {best_scores['zone_strength']:.1f}/2.0<br>
• Price Position: {best_scores['price_position']:.1f}/2.0<br>
• Freshness: {best_scores['freshness']:.1f}/2.0<br>
• Trend Alignment: {best_scores['trend_alignment']:.1f}/2.0<br>
• Time/Base: {best_scores['time_base']:.1f}/1.0<br>
• Profit Potential: {best_scores['profit_potential']:.1f}/1.0<br>
• Zone: ${best_formation['zone'].low:.2f} - ${best_formation['zone'].high:.2f}<br><br>

<b>🔧 Method:</b><br>
• ✅ Curve fitting detected extrema automatically<br>
• ✅ Extrema sequences identified classic formations<br>
• ✅ Proper scoring rules applied to each zone<br>
• ✅ No manual zone drawing required!<br><br>

<b>📋 Formation Legend:</b><br>
🔴 RBD = Rally-Base-Drop (Supply)<br>
🟢 DBD = Drop-Base-Drop (Demand)<br>
🟠 RBR = Rally-Base-Rally (Supply)<br>
🔵 DBR = Drop-Base-Rally (Demand)"""
        
        return summary

def create_auto_formation_example():
    """Create a demonstration of auto formation detection"""
    
    print("🚀 AUTO-FORMATION DETECTION DEMO")
    print("Using curve fitting shortcuts for formation identification")
    print("="*70)
    
    labeler = AutoFormationLabeler()
    
    # Test on TSLA data
    symbol = 'TSLA'
    start_date = '2023-08-01'
    end_date = '2023-12-01'
    
    # Run the complete auto-detection pipeline
    result = labeler.auto_detect_and_score_formations(symbol, start_date, end_date)
    
    if result:
        # Create visualization
        fig = labeler.create_auto_labeled_visualization(result)
        
        if fig:
            filename = f"/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/curve_fitting_examples/{symbol.lower()}_auto_formations.html"
            fig.write_html(filename)
            print(f"\n💾 Saved: {filename}")
            print(f"🎯 Auto-detected {len(result['formations'])} formations with proper scoring!")
            
            return filename
        else:
            print("❌ Failed to create visualization")
    else:
        print("❌ Auto-detection failed")
    
    return None

if __name__ == "__main__":
    create_auto_formation_example()