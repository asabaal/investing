#!/usr/bin/env python3
"""
Metric Deep Dive Analyzer

Creates focused examples for each of the 6 scoring metrics to help workshop
and refine the definitions. Each metric gets isolated examples showing
the full range of that specific scoring component.

This complements the full setup examples by diving deep into individual
metric behaviors and edge cases.
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

class MetricDeepDiveAnalyzer:
    """Creates focused examples for individual scoring metrics"""
    
    def __init__(self):
        self.scorer = TradeScorer()
    
    def create_zone_strength_examples(self):
        """Zone Strength (0-2 pts): Leg out distance + opposing zone breakout"""
        
        print("📊 Creating Zone Strength examples...")
        
        examples = []
        
        # Example 1: Perfect zone strength (2.0 pts)
        # - 2:1 leg out ratio + opposing zone breakout
        zone1 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,  # 2.0 range
            base_candles=3,
            leg_out_start=101.0,
            leg_out_end=106.0,  # 5.0 movement = 2.5:1 ratio
            opposing_zones=[(104.0, 105.0)]  # Opposing zone broken through
        )
        examples.append(("Perfect Zone Strength", zone1, 2.0, "2.5:1 leg out ratio + opposing zone breakout"))
        
        # Example 2: Good zone strength (1.0 pts)
        # - 2:1 leg out ratio but no opposing zone breakout
        zone2 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,  # 2.0 range
            base_candles=3,
            leg_out_start=101.0,
            leg_out_end=105.5,  # 4.5 movement = 2.25:1 ratio
            opposing_zones=[]  # No opposing zones
        )
        examples.append(("Good Zone Strength", zone2, 1.0, "2.25:1 leg out ratio, no opposing zones"))
        
        # Example 3: Partial zone strength (1.0 pts)
        # - Only opposing zone breakout, no 2:1 ratio
        zone3 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,  # 2.0 range
            base_candles=3,
            leg_out_start=101.0,
            leg_out_end=103.5,  # 2.5 movement = 1.25:1 ratio
            opposing_zones=[(103.0, 104.0)]  # Opposing zone broken through
        )
        examples.append(("Partial Zone Strength", zone3, 1.0, "1.25:1 leg out ratio but broke opposing zone"))
        
        # Example 4: Weak zone strength (0.0 pts)
        # - <2:1 leg out ratio and no opposing zone breakout
        zone4 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,  # 2.0 range
            base_candles=3,
            leg_out_start=101.0,
            leg_out_end=103.0,  # 2.0 movement = 1:1 ratio
            opposing_zones=[]
        )
        examples.append(("Weak Zone Strength", zone4, 0.0, "1:1 leg out ratio, no opposing zones"))
        
        return self._create_metric_visualization("Zone Strength", examples, "Zone strength measures if the leg out moved far enough and broke opposing levels")
    
    def create_time_base_examples(self):
        """Time/Base (0-1 pts): Number of candles in base segment"""
        
        print("📊 Creating Time/Base examples...")
        
        examples = []
        
        # Example 1: Perfect time (1.0 pts) - 1-3 candles
        zone1 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,
            base_candles=2,
            leg_out_start=101.0,
            leg_out_end=106.0,
            opposing_zones=[]
        )
        examples.append(("Perfect Time Score", zone1, 1.0, "2 candles in base (1-3 range)"))
        
        # Example 2: Good time (0.5 pts) - 4-6 candles  
        zone2 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,
            base_candles=5,
            leg_out_start=101.0,
            leg_out_end=106.0,
            opposing_zones=[]
        )
        examples.append(("Good Time Score", zone2, 0.5, "5 candles in base (4-6 range)"))
        
        # Example 3: Poor time (0.0 pts) - >6 candles
        zone3 = Zone(
            zone_type=ZoneType.DEMAND,
            high=102.0,
            low=100.0,
            base_candles=10,
            leg_out_start=101.0,
            leg_out_end=106.0,
            opposing_zones=[]
        )
        examples.append(("Poor Time Score", zone3, 0.0, "10 candles in base (>6 range)"))
        
        return self._create_metric_visualization("Time/Base", examples, "Shorter consolidations are stronger - they show decisive moves")
    
    def create_freshness_examples(self):
        """Freshness (0-2 pts): Zone testing/violation status"""
        
        print("📊 Creating Freshness examples...")
        
        examples = []
        base_zone = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 106.0, [])
        
        # Example 1: Untested zone (2.0 pts)
        context1 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, 
                                FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Untested Zone", (base_zone, context1), 2.0, "Zone never tested after formation"))
        
        # Example 2: Partial penetration (1.0 pts) - <50%
        context2 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0,
                                FreshnessStatus.PARTIAL_PENETRATION, 30.0, 8.0)
        examples.append(("Partial Test", (base_zone, context2), 1.0, "Zone tested but <50% penetration"))
        
        # Example 3: Deep penetration (0.0 pts) - >50% but not violated
        context3 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0,
                                FreshnessStatus.DEEP_PENETRATION, 70.0, 8.0)
        examples.append(("Deep Test", (base_zone, context3), 0.0, "Zone deeply tested >50% but not violated"))
        
        # Example 4: Violated zone (-1.0 pts) - Invalid
        context4 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0,
                                FreshnessStatus.VIOLATED, 100.0, 8.0)
        examples.append(("Violated Zone", (base_zone, context4), -1.0, "Zone completely violated - INVALID"))
        
        return self._create_freshness_visualization(examples)
    
    def create_trend_alignment_examples(self):
        """Trend Alignment (0-2 pts): Zone direction vs current trend"""
        
        print("📊 Creating Trend Alignment examples...")
        
        examples = []
        demand_zone = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 106.0, [])
        supply_zone = Zone(ZoneType.SUPPLY, 102.0, 100.0, 3, 101.0, 96.0, [])
        
        # Perfect alignment examples (2.0 pts)
        context1 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Perfect Alignment", (demand_zone, context1), 2.0, "Demand zone in uptrend - perfectly aligned"))
        
        context2 = MarketContext(TrendDirection.DOWN, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)  
        examples.append(("Perfect Alignment", (supply_zone, context2), 2.0, "Supply zone in downtrend - perfectly aligned"))
        
        # Sideways trend (1.0 pts)
        context3 = MarketContext(TrendDirection.SIDEWAYS, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Sideways Trend", (demand_zone, context3), 1.0, "Any zone type in sideways trend"))
        
        # Counter-trend (0.0 pts)
        context4 = MarketContext(TrendDirection.DOWN, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Counter-Trend", (demand_zone, context4), 0.0, "Demand zone in downtrend - fighting trend"))
        
        return self._create_metric_visualization("Trend Alignment", examples, "Trading with the trend increases probability of success")
    
    def create_price_position_examples(self):
        """Price Position (0-1 pts): Position in long-term price range"""
        
        print("📊 Creating Price Position examples...")
        
        examples = []
        
        # Long-term range: 90-120 (30 point range, thirds = 10 points each)
        # Bottom third: 90-100, Middle: 100-110, Top: 110-120
        
        # Example 1: Demand zone in bottom third (1.0 pts) - Favorable
        demand_zone1 = Zone(ZoneType.DEMAND, 97.0, 95.0, 3, 96.0, 102.0, [])
        context1 = MarketContext(TrendDirection.UP, 96.0, 120.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Demand Bottom Third", (demand_zone1, context1), 1.0, "Demand zone at bottom of range - favorable for up moves"))
        
        # Example 2: Supply zone in top third (1.0 pts) - Favorable  
        supply_zone1 = Zone(ZoneType.SUPPLY, 117.0, 115.0, 3, 116.0, 110.0, [])
        context2 = MarketContext(TrendDirection.DOWN, 116.0, 120.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Supply Top Third", (supply_zone1, context2), 1.0, "Supply zone at top of range - favorable for down moves"))
        
        # Example 3: Zone in middle third (0.5 pts)
        demand_zone2 = Zone(ZoneType.DEMAND, 107.0, 105.0, 3, 106.0, 112.0, [])
        context3 = MarketContext(TrendDirection.UP, 106.0, 120.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Middle Third", (demand_zone2, context3), 0.5, "Zone in middle third - neutral position"))
        
        # Example 4: Demand zone in top third (0.0 pts) - Unfavorable
        demand_zone3 = Zone(ZoneType.DEMAND, 117.0, 115.0, 3, 116.0, 122.0, [])
        context4 = MarketContext(TrendDirection.UP, 116.0, 120.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Unfavorable Position", (demand_zone3, context4), 0.0, "Demand zone at top - unfavorable for up moves"))
        
        return self._create_metric_visualization("Price Position", examples, "Zones at price extremes in the favorable direction score higher")
    
    def create_profit_potential_examples(self):
        """Profit Potential (0-2 pts): Leg out distance to zone range ratio"""
        
        print("📊 Creating Profit Potential examples...")
        
        examples = []
        
        # Example 1: Excellent ratio (2.0 pts) - >=5:1
        zone1 = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 111.0, [])  # 2.0 range, 10.0 leg out = 5:1
        context1 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
        examples.append(("Excellent Ratio", (zone1, context1), 2.0, "5:1 leg out to zone ratio (>=5:1)"))
        
        # Example 2: Good ratio (1.0 pts) - >=3:1 but <5:1
        zone2 = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 108.0, [])  # 2.0 range, 7.0 leg out = 3.5:1
        context2 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 7.0)
        examples.append(("Good Ratio", (zone2, context2), 1.0, "3.5:1 leg out to zone ratio (3:1 to 5:1)"))
        
        # Example 3: Poor ratio (0.0 pts) - <3:1
        zone3 = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 105.0, [])  # 2.0 range, 4.0 leg out = 2:1
        context3 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 4.0)
        examples.append(("Poor Ratio", (zone3, context3), 0.0, "2:1 leg out to zone ratio (<3:1)"))
        
        # Example 4: Very poor ratio (0.0 pts) - <2:1
        zone4 = Zone(ZoneType.DEMAND, 102.0, 100.0, 3, 101.0, 102.5, [])  # 2.0 range, 1.5 leg out = 0.75:1
        context4 = MarketContext(TrendDirection.UP, 101.5, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 2.5)
        examples.append(("Very Poor Ratio", (zone4, context4), 0.0, "0.75:1 leg out to zone ratio - avoid"))
        
        return self._create_metric_visualization("Profit Potential", examples, "Higher leg out to zone ratios provide better profit potential")
    
    def _create_metric_visualization(self, metric_name, examples, description):
        """Create visualization for a specific metric"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{metric_name} - Score Examples',
                f'{metric_name} - Score Distribution',
                'Metric Analysis', 
                'Key Decision Points'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )
        
        # Example scores - handle different example formats
        example_names = [ex[0] for ex in examples]
        scores = [ex[2] for ex in examples]
        descriptions = [ex[3] for ex in examples]
        
        # Calculate scores for zone examples
        calculated_scores = []
        for i, (name, data, expected_score, desc) in enumerate(examples):
            if isinstance(data, Zone):
                # For zone-only examples, create a default context
                context = MarketContext(TrendDirection.UP, 101.0, 110.0, 90.0, FreshnessStatus.UNTESTED, 0.0, 8.0)
                if metric_name == "Zone Strength":
                    score = self.scorer.score_zone_strength(data)
                elif metric_name == "Time/Base":
                    score = self.scorer.score_time_base(data)
                else:
                    score = expected_score
                calculated_scores.append(score)
            elif isinstance(data, tuple):
                # For (zone, context) examples
                zone, context = data
                if metric_name == "Trend Alignment":
                    score = self.scorer.score_trend_alignment(zone, context)
                elif metric_name == "Price Position":
                    score = self.scorer.score_price_position(zone, context)
                elif metric_name == "Profit Potential":
                    score = self.scorer.score_profit_potential(zone, context)
                else:
                    score = expected_score
                calculated_scores.append(score)
            else:
                calculated_scores.append(expected_score)
        
        # Use calculated scores for visualization
        display_scores = calculated_scores if calculated_scores else scores
        
        # Score examples bar chart
        colors = ['green' if s == max(display_scores) else 'orange' if s > 0 else 'red' for s in display_scores]
        fig.add_trace(go.Bar(
            x=example_names,
            y=display_scores,
            marker_color=colors,
            text=[f'{s:.1f}' for s in display_scores],
            textposition='auto',
            name='Score'
        ), row=1, col=1)
        
        # Score distribution
        unique_scores = sorted(list(set(display_scores)))
        score_counts = [display_scores.count(s) for s in unique_scores]
        
        fig.add_trace(go.Bar(
            x=[f'{s:.1f} pts' for s in unique_scores],
            y=score_counts,
            marker_color='#33aaff',
            text=score_counts,
            textposition='auto',
            name='Frequency'
        ), row=1, col=2)
        
        # Get max score for this metric
        metric_max_scores = {
            "Zone Strength": 2, "Time/Base": 1, "Freshness": 2,
            "Trend Alignment": 2, "Price Position": 1, "Profit Potential": 2
        }
        
        # Analysis table
        analysis_data = [
            ['Metric Name', metric_name],
            ['Score Range', f'0-{metric_max_scores.get(metric_name, 2)} points'],
            ['Best Example', example_names[display_scores.index(max(display_scores))]],
            ['Worst Example', example_names[display_scores.index(min(display_scores))]],
            ['Average Score', f'{sum(display_scores)/len(display_scores):.1f}'],
            ['Description', description[:50] + '...' if len(description) > 50 else description]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Analysis', 'Value'], fill_color='#404040', font=dict(color='white')),
            cells=dict(values=[[item[0] for item in analysis_data], [item[1] for item in analysis_data]],
                      fill_color=['#2a2a2a', '#1a1a1a'], font=dict(color='white'))
        ), row=2, col=1)
        
        # Decision points table
        decision_points = []
        if metric_name == "Zone Strength":
            decision_points = [
                ['2:1 Leg Out Ratio', '1 point if achieved'],
                ['Opposing Zone Break', '1 point if achieved'],
                ['Both Independent', 'Can get 0, 1, or 2 points total'],
                ['Movement Calculation', 'abs(leg_out_end - leg_out_start)']
            ]
        elif metric_name == "Time/Base":
            decision_points = [
                ['1-3 Candles', '1.0 points - strongest'],
                ['4-6 Candles', '0.5 points - acceptable'],
                ['>6 Candles', '0.0 points - too long'],
                ['Logic', 'Shorter = more decisive']
            ]
        elif metric_name == "Trend Alignment":
            decision_points = [
                ['Perfect Alignment', '2.0 points'],
                ['Sideways Trend', '1.0 points'],
                ['Counter-Trend', '0.0 points'],
                ['Rule', 'Trade with the trend']
            ]
        elif metric_name == "Price Position":
            decision_points = [
                ['Favorable Third', '1.0 points'],
                ['Middle Third', '0.5 points'],
                ['Unfavorable Third', '0.0 points'],
                ['Logic', 'Extremes favor mean reversion']
            ]
        elif metric_name == "Profit Potential":
            decision_points = [
                ['>=5:1 Leg Out Ratio', '2.0 points - excellent'],
                ['>=3:1 Leg Out Ratio', '1.0 points - good'],
                ['<3:1 Leg Out Ratio', '0.0 points - avoid'],
                ['Calculation', 'leg_out_distance / zone_range']
            ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Decision Point', 'Scoring'], fill_color='#404040', font=dict(color='white')),
            cells=dict(values=[[item[0] for item in decision_points], [item[1] for item in decision_points]],
                      fill_color=['#2a2a2a', '#1a1a1a'], font=dict(color='white'))
        ), row=2, col=2)
        
        fig.update_layout(
            title=f'{metric_name} - Deep Dive Analysis<br>{description}',
            height=700,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2a2a2a',
            font=dict(color='white'),
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        return fig
    
    def _create_freshness_visualization(self, examples):
        """Special visualization for freshness metric"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Freshness Status Examples',
                'Zone Testing Scenarios',
                'Penetration Levels',
                'Decision Framework'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # Example scores
        example_names = [ex[0] for ex in examples]
        scores = [ex[2] for ex in examples]
        
        colors = ['green', 'orange', 'red', 'darkred']
        fig.add_trace(go.Bar(
            x=example_names,
            y=scores,
            marker_color=colors,
            text=[f'{s:.1f}' for s in scores],
            textposition='auto',
            name='Freshness Score'
        ), row=1, col=1)
        
        # Penetration levels scatter
        penetration_levels = [0, 30, 70, 100]
        fig.add_trace(go.Scatter(
            x=example_names,
            y=penetration_levels,
            mode='markers+lines',
            marker=dict(size=15, color=colors),
            line=dict(width=3, color='white'),
            name='Penetration %'
        ), row=1, col=2)
        
        # Score vs penetration
        valid_scores = [s for s in scores if s >= 0]
        valid_penetrations = [p for p, s in zip(penetration_levels, scores) if s >= 0]
        
        fig.add_trace(go.Bar(
            x=['0%', '30%', '70%'],
            y=[2.0, 1.0, 0.0],
            marker_color=['green', 'orange', 'red'],
            text=['Untested<br>2.0 pts', 'Partial<br>1.0 pts', 'Deep<br>0.0 pts'],
            textposition='auto',
            name='Valid Zones'
        ), row=2, col=1)
        
        # Decision framework
        framework = [
            ['Untested', '0% penetration', '2.0 points'],
            ['Partial Test', '<50% penetration', '1.0 points'], 
            ['Deep Test', '>50% but not violated', '0.0 points'],
            ['Violated', '100% penetration', 'INVALID (-1.0)'],
            ['', '', ''],
            ['Key Rule', 'Less testing = stronger zone', '']
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Status', 'Penetration', 'Score'], fill_color='#404040', font=dict(color='white')),
            cells=dict(values=[[item[0] for item in framework], [item[1] for item in framework], [item[2] for item in framework]],
                      fill_color=['#2a2a2a', '#1a1a1a', '#3a1a1a'], font=dict(color='white'))
        ), row=2, col=2)
        
        fig.update_layout(
            title='Freshness - Zone Testing Analysis<br>Measures how much the zone has been tested after formation',
            height=700,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2a2a2a',
            font=dict(color='white'),
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Penetration %", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        
        return fig
    
    def generate_all_metric_examples(self):
        """Generate deep dive examples for all 6 metrics"""
        
        print("🎯 GENERATING METRIC DEEP DIVE EXAMPLES")
        print("=" * 60)
        print("Creating focused examples for each of the 6 scoring metrics")
        print("These isolate individual metric behaviors for workshop discussions\n")
        
        metrics = [
            ("zone_strength", self.create_zone_strength_examples),
            ("time_base", self.create_time_base_examples), 
            ("freshness", self.create_freshness_examples),
            ("trend_alignment", self.create_trend_alignment_examples),
            ("price_position", self.create_price_position_examples),
            ("profit_potential", self.create_profit_potential_examples)
        ]
        
        results = {}
        
        for metric_name, create_func in metrics:
            try:
                fig = create_func()
                filename = f"trade_setup_examples/metric_{metric_name}_deep_dive.html"
                fig.write_html(filename)
                
                results[metric_name] = filename
                print(f"   ✅ Saved: {filename}")
                
            except Exception as e:
                print(f"   ❌ Error creating {metric_name}: {str(e)}")
                continue
        
        print(f"\n🎯 METRIC DEEP DIVES COMPLETE!")
        print(f"   Generated {len(results)} focused metric examples")
        print(f"   Each example isolates one scoring component")
        print(f"   Perfect for workshopping individual definitions")
        
        return results


def main():
    """Generate comprehensive metric deep dive examples"""
    
    analyzer = MetricDeepDiveAnalyzer()
    results = analyzer.generate_all_metric_examples()
    
    print(f"\n📊 WORKSHOP MATERIALS READY!")
    print(f"   Main setup examples: trade_setup_examples/*_setup_analysis.html")
    print(f"   Metric deep dives: trade_setup_examples/metric_*_deep_dive.html")
    print(f"   Summary report: trade_setup_examples/scoring_summary_report.html")
    print(f"\n   Use these to refine and validate your scoring system definitions!")


if __name__ == "__main__":
    main()