#!/usr/bin/env python3
"""
Data Variance & Trading Implications Guide
Understanding data differences across providers and timeframes for trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataVarianceGuide:
    """Guide for understanding data variance impact on trading"""
    
    def __init__(self):
        self.variance_patterns = self._load_variance_patterns()
        self.trading_implications = self._load_trading_implications()
    
    def _load_variance_patterns(self):
        """Expected variance patterns across data sources and timeframes"""
        return {
            "price_data": {
                "daily_close": {
                    "typical_variance": "0.01-0.05%",
                    "max_expected": "0.1%",
                    "common_causes": [
                        "Different closing times (4PM vs 4:15PM)",
                        "Settlement vs real-time prices",
                        "Dividend adjustments timing"
                    ],
                    "reliability_ranking": ["Bloomberg", "Refinitiv", "Alpha Vantage", "Yahoo", "Free sources"]
                },
                "intraday_ohlc": {
                    "typical_variance": "0.05-0.2%",
                    "max_expected": "0.5%",
                    "common_causes": [
                        "Tick aggregation differences",
                        "Exchange vs consolidated feeds",
                        "Bid/ask vs trade prices",
                        "Time zone handling"
                    ]
                },
                "opening_prices": {
                    "typical_variance": "0.1-0.5%",
                    "max_expected": "2%",
                    "common_causes": [
                        "Pre-market activity inclusion",
                        "Opening auction vs first trade",
                        "Gap handling"
                    ]
                }
            },
            "volume_data": {
                "daily_volume": {
                    "typical_variance": "5-15%",
                    "max_expected": "50%",
                    "common_causes": [
                        "Consolidated vs primary exchange",
                        "Dark pool inclusion/exclusion",
                        "Block trade reporting differences",
                        "ETF creation/redemption units"
                    ]
                },
                "intraday_volume": {
                    "typical_variance": "10-50%",
                    "max_expected": "200%",
                    "common_causes": [
                        "Real-time vs delayed reporting",
                        "Market maker activity inclusion",
                        "Cross-trading inclusion",
                        "Tick aggregation methods"
                    ]
                }
            }
        }
    
    def _load_trading_implications(self):
        """Trading implications of data variance"""
        return {
            "risk_management": {
                "position_sizing": {
                    "principle": "Account for data uncertainty in position calculations",
                    "methods": [
                        "Use wider stop losses when data quality is questionable",
                        "Reduce position size for strategies sensitive to precise entries",
                        "Implement data quality filters in automated systems"
                    ]
                },
                "entry_timing": {
                    "principle": "Adjust entry precision based on data reliability",
                    "daily_trades": "High precision possible - data variance <0.1%",
                    "intraday_scalping": "Lower precision - expect 0.2-0.5% variance",
                    "swing_trading": "Data variance negligible for multi-day holds"
                },
                "stop_loss_adjustment": {
                    "conservative": "Add 0.1-0.2% buffer for daily strategies",
                    "aggressive": "Add 0.2-0.5% buffer for intraday strategies",
                    "reasoning": "Prevent false stops from data discrepancies"
                }
            },
            "strategy_adaptation": {
                "high_frequency": {
                    "data_requirements": "Sub-second, exchange-direct feeds",
                    "variance_tolerance": "<0.01%",
                    "recommended_sources": "Professional market data (Bloomberg, Refinitiv)"
                },
                "day_trading": {
                    "data_requirements": "Real-time or 15-min delayed acceptable",
                    "variance_tolerance": "0.1-0.3%",
                    "recommended_sources": "Broker feeds, Alpha Vantage real-time"
                },
                "swing_trading": {
                    "data_requirements": "End-of-day sufficient",
                    "variance_tolerance": "0.5%+",
                    "recommended_sources": "Alpha Vantage, Yahoo Finance acceptable"
                },
                "long_term": {
                    "data_requirements": "Weekly/monthly aggregates",
                    "variance_tolerance": "1%+",
                    "recommended_sources": "Any reputable source"
                }
            }
        }
    
    def analyze_strategy_data_fit(self, strategy_type: str, holding_period: str, 
                                 precision_required: str) -> dict:
        """Analyze if your data quality fits your trading strategy"""
        
        print(f"🎯 Strategy-Data Fit Analysis")
        print(f"Strategy: {strategy_type}")
        print(f"Holding Period: {holding_period}")
        print(f"Precision Required: {precision_required}")
        print("=" * 50)
        
        recommendations = {
            "data_sources": [],
            "quality_requirements": {},
            "risk_adjustments": [],
            "cost_benefit": {}
        }
        
        # Strategy categorization
        if "scalping" in strategy_type.lower() or "hft" in strategy_type.lower():
            category = "high_frequency"
        elif "day" in strategy_type.lower() or holding_period in ["intraday", "hours"]:
            category = "day_trading"
        elif "swing" in strategy_type.lower() or holding_period in ["days", "weeks"]:
            category = "swing_trading"
        else:
            category = "long_term"
        
        strategy_req = self.trading_implications["strategy_adaptation"][category]
        
        print(f"📊 Strategy Category: {category.replace('_', ' ').title()}")
        print(f"Data Requirements: {strategy_req['data_requirements']}")
        print(f"Variance Tolerance: {strategy_req['variance_tolerance']}")
        print(f"Recommended Sources: {strategy_req['recommended_sources']}")
        
        # Alpha Vantage suitability assessment
        av_suitability = {
            "high_frequency": {"fit": "Poor", "score": 2, "reason": "15-min delay, insufficient precision"},
            "day_trading": {"fit": "Good", "score": 8, "reason": "Adequate precision, acceptable delay"},
            "swing_trading": {"fit": "Excellent", "score": 9, "reason": "Perfect for multi-day holds"},
            "long_term": {"fit": "Excellent", "score": 10, "reason": "More than sufficient precision"}
        }
        
        av_fit = av_suitability[category]
        
        print(f"\n🔍 Alpha Vantage Suitability:")
        print(f"Fit Score: {av_fit['score']}/10 ({av_fit['fit']})")
        print(f"Reasoning: {av_fit['reason']}")
        
        # Risk adjustment recommendations
        risk_mgmt = self.trading_implications["risk_management"]
        
        print(f"\n⚠️  Risk Management Adjustments:")
        
        if category in ["high_frequency", "day_trading"]:
            print("• Position Sizing: Reduce by 10-20% to account for data variance")
            print("• Stop Losses: Add 0.2-0.5% buffer above normal technical levels")
            print("• Entry Timing: Use limit orders instead of market orders")
            print("• Validation: Cross-check signals with broker's real-time data")
        else:
            print("• Position Sizing: Normal sizing acceptable")
            print("• Stop Losses: Add 0.1% buffer for daily strategies")
            print("• Entry Timing: Market orders acceptable")
            print("• Validation: End-of-day validation sufficient")
        
        return {
            "category": category,
            "alpha_vantage_fit": av_fit,
            "recommendations": recommendations
        }
    
    def create_data_quality_checklist(self):
        """Create a practical checklist for data quality in trading"""
        
        checklist = {
            "daily_checks": [
                "□ Verify latest close matches your broker's data",
                "□ Check for any obvious gaps in price series", 
                "□ Confirm volume isn't showing anomalous spikes (>5x normal)",
                "□ Validate data freshness (within expected delay window)"
            ],
            "weekly_checks": [
                "□ Run data quality evaluator on recent data",
                "□ Compare key levels (support/resistance) across sources",
                "□ Verify dividend/split adjustments are correct",
                "□ Check for any systematic timing differences"
            ],
            "before_major_trades": [
                "□ Cross-validate entry price with real-time broker data",
                "□ Confirm volume pattern is consistent with market conditions",
                "□ Check if any corporate actions might affect data",
                "□ Verify stop-loss calculations include data variance buffer"
            ],
            "system_setup": [
                "□ Implement data quality filters in automated strategies",
                "□ Set up alerts for extreme price/volume anomalies",
                "□ Create backup data source for critical decisions",
                "□ Document expected variance ranges for your timeframes"
            ]
        }
        
        return checklist
    
    def print_variance_expectations(self):
        """Print expected variance ranges for different data types"""
        
        print("📊 Expected Data Variance Across Providers")
        print("=" * 60)
        
        for data_type, categories in self.variance_patterns.items():
            print(f"\n🔸 {data_type.replace('_', ' ').title()}:")
            
            for category, details in categories.items():
                print(f"\n  📈 {category.replace('_', ' ').title()}:")
                print(f"     Typical: {details['typical_variance']}")
                print(f"     Maximum: {details['max_expected']}")
                print(f"     Causes:")
                for cause in details['common_causes']:
                    print(f"       • {cause}")
    
    def print_trading_guide(self):
        """Print comprehensive trading guide"""
        
        print("🎯 TRADING WITH DATA VARIANCE - PRACTICAL GUIDE")
        print("=" * 60)
        
        print("\n💡 KEY PRINCIPLES:")
        print("1. Match data quality to strategy requirements")
        print("2. Account for variance in risk management")  
        print("3. Use appropriate buffers for entries/exits")
        print("4. Validate critical decisions with multiple sources")
        
        print("\n📏 POSITION SIZING ADJUSTMENTS:")
        print("Strategy Type          | Position Size Adjustment")
        print("----------------------|-------------------------")
        print("Scalping (<1 min)     | Reduce 30-50% (poor data fit)")
        print("Day Trading (hours)   | Reduce 10-20%")
        print("Swing Trading (days)  | Normal sizing")
        print("Long-term (weeks+)    | Normal sizing")
        
        print("\n🛡️  STOP-LOSS BUFFERS:")
        print("Timeframe             | Additional Buffer")
        print("----------------------|------------------")
        print("1-minute charts       | +0.3-0.5%")
        print("5-15 minute charts    | +0.2-0.3%") 
        print("Hourly charts         | +0.1-0.2%")
        print("Daily charts          | +0.05-0.1%")
        
        print("\n✅ DATA QUALITY CHECKLIST:")
        checklist = self.create_data_quality_checklist()
        
        for category, items in checklist.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                print(f"  {item}")

def main():
    guide = DataVarianceGuide()
    
    # Print comprehensive guide
    guide.print_variance_expectations()
    print("\n" + "="*80)
    guide.print_trading_guide()
    
    # Example strategy analysis
    print("\n" + "="*80)
    print("EXAMPLE: Swing Trading Analysis")
    print("="*80)
    guide.analyze_strategy_data_fit(
        strategy_type="swing trading",
        holding_period="3-7 days", 
        precision_required="moderate"
    )

if __name__ == "__main__":
    main()