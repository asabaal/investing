#!/usr/bin/env python3
"""
Complete Symphony Development Example

This script demonstrates how to address all three key challenges:
1. Forecasting for future testing before production deployment
2. SPY benchmark comparison and optimization for low-risk strategies  
3. Composer DSL reconciliation and validation

Usage: python complete_symphony_example.py
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def main():
    print("🎼 Complete Symphony Development Example")
    print("=" * 70)
    print("Addressing all three key challenges:")
    print("1. 🔮 Forecasting for future testing")
    print("2. 📊 SPY benchmark comparison & optimization") 
    print("3. 🔄 Composer DSL reconciliation")
    print("=" * 70)
    
    # Challenge 1: Forecasting Example
    print("\n" + "🔮 CHALLENGE 1: FORECASTING FOR FUTURE TESTING" + "=" * 30)
    forecasting_example()
    
    # Challenge 2: SPY Benchmark & Optimization
    print("\n" + "📊 CHALLENGE 2: SPY BENCHMARK COMPARISON & OPTIMIZATION" + "=" * 20)
    spy_benchmark_example()
    
    # Challenge 3: Composer Reconciliation
    print("\n" + "🔄 CHALLENGE 3: COMPOSER DSL RECONCILIATION" + "=" * 30)
    composer_reconciliation_example()
    
    print("\n" + "🎉 COMPLETE EXAMPLE FINISHED" + "=" * 40)
    print("Next steps for production deployment:")
    print("1. ✅ Run forecasting on your strategies before deployment")
    print("2. ✅ Only deploy strategies that beat SPY consistently")  
    print("3. ✅ Validate your implementations against Composer results")
    print("4. 🚀 Start with low-risk SPY-plus strategies")
    print("5. 📈 Scale to higher-risk strategies once proven")

def forecasting_example():
    """Demonstrate forecasting for future testing"""
    
    print("🔮 Forecasting Analysis - Test Before You Deploy!")
    print("-" * 50)
    
    # Example command for forecasting
    print("Command to run forecasting analysis:")
    print("python symphony_runner.py --config my_strategy.json --forecast --forecast-days 252")
    print()
    
    # Example forecast interpretation
    print("📈 How to interpret forecast results:")
    print("✅ DEPLOY if:")
    print("   • Expected annual return > 12%")
    print("   • Expected Sharpe ratio > 1.0") 
    print("   • Probability of beating benchmark > 65%")
    print("   • Worst case scenario > -30%")
    print()
    print("⚠️  CAUTION if:")
    print("   • Expected annual return 8-12%")
    print("   • Expected Sharpe ratio 0.5-1.0")
    print("   • Review stress test scenarios carefully")
    print()
    print("❌ AVOID if:")
    print("   • Expected annual return < 8%")
    print("   • Expected Sharpe ratio < 0.5")
    print("   • High probability of large losses")
    
    # Example forecasting workflow
    print("\n📋 Recommended Forecasting Workflow:")
    print("1. Run 1-year forecast (252 days) for annual expectations")
    print("2. Run 3-month forecast (63 days) for near-term outlook")
    print("3. Analyze stress test scenarios (market crash, high vol, etc.)")
    print("4. Compare forecast vs historical backtest performance")
    print("5. Only deploy if forecast meets your risk criteria")
    
    # Create sample forecast command file
    create_forecast_example_script()

def spy_benchmark_example():
    """Demonstrate SPY benchmark comparison and optimization"""
    
    print("📊 SPY Benchmark Comparison - Beat the Market Safely!")
    print("-" * 55)
    
    print("🛡️ Low-Risk Strategy Approach:")
    print("• Start with SPY-plus strategies (modest improvements over SPY)")
    print("• Target 9-12% annual returns vs SPY's ~8-10%")
    print("• Keep max drawdown under 25% (vs SPY's ~20%)")
    print("• Maintain correlation with SPY > 0.7 for stability")
    print()
    
    # SPY-plus strategy creation
    print("Command to create SPY-plus strategies:")
    print("python symphony_runner.py --create-spy-plus --start-date 2022-01-01")
    print()
    
    print("🔧 Optimization for beating SPY:")
    print("python symphony_runner.py --config my_strategy.json --optimize --benchmark SPY")
    print()
    
    # Example SPY-beating criteria
    print("✅ Deploy if strategy beats SPY with:")
    print("   • Annual return: Strategy > SPY + 2%")
    print("   • Sharpe ratio: Strategy > SPY + 0.3")
    print("   • Max drawdown: Strategy < SPY + 5%")
    print("   • Win rate: > 55% of months")
    print()
    
    print("📊 Example SPY-Plus Strategies to Create:")
    strategies = [
        "SPY Momentum Plus: SPY + momentum overlay (70% SPY, 30% QQQ when SPY > 200 MA)",
        "SPY Volatility Shield: SPY + bonds during high volatility (70% SPY, 30% TLT when VIX > 25)",
        "SPY Sector Rotation: SPY + tactical sector allocation (80% SPY, 20% best performing sector)"
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"   {i}. {strategy}")
    
    # Create SPY-plus example
    create_spy_plus_example()

def composer_reconciliation_example():
    """Demonstrate Composer DSL reconciliation"""
    
    print("🔄 Composer DSL Reconciliation - Validate Your Implementation!")
    print("-" * 60)
    
    print("Your Composer symphony DSL:")
    composer_dsl = '''
    (defsymphony
     "Copy of 200d MA 3x Leverage"
     {:asset-class "EQUITIES", :rebalance-threshold 0.05}
     (weight-equal
      [(if
        (> (current-price "SPY") (moving-average-price "SPY" {:window 200}))
        [(weight-equal
          [(if
            (> (rsi "TQQQ" {:window 10}) 79)
            [(asset "UVXY")]
            [(asset "TQQQ")])])]
        [(weight-equal
          [(if
            (< (rsi "TQQQ" {:window 10}) 31)
            [(asset "TECL")]
            [(asset "QQQ")])])])]))
    '''
    
    print("📝 Composer DSL (simplified version shown)")
    print("🔄 Converting to our format...")
    print()
    
    # Show conversion process
    print("Command to convert and validate:")
    print("python symphony_runner.py --convert-composer composer_symphony.txt --composer-csv backtest_results.csv")
    print()
    
    print("🔍 Reconciliation Process:")
    print("1. Parse Composer DSL → Our JSON format")
    print("2. Run our backtest on same date range")
    print("3. Compare allocation decisions day by day")
    print("4. Calculate allocation difference percentage")
    print("5. Validate performance matches within tolerance")
    print()
    
    print("✅ EXCELLENT reconciliation: < 2.5% allocation difference")
    print("👍 GOOD reconciliation: 2.5-5% allocation difference")  
    print("⚠️  FAIR reconciliation: 5-10% allocation difference")
    print("❌ POOR reconciliation: > 10% allocation difference")
    print()
    
    print("📊 Your Composer backtest shows:")
    print("• Recent allocation: 100% TQQQ (bullish tech)")
    print("• Strategy switches between: TQQQ, UVXY, TECL, QQQ, PSQ")
    print("• Complex nested conditions based on RSI and moving averages")
    
    # Create conversion example
    create_composer_conversion_example(composer_dsl)

def create_forecast_example_script():
    """Create example forecasting script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Forecasting Example Script

Run this to forecast your symphony's future performance before deployment.
"""

import subprocess
import sys

def run_forecast_analysis(symphony_file):
    """Run comprehensive forecasting analysis"""
    
    print(f"🔮 Running forecast analysis for {symphony_file}")
    
    # 1-year forecast
    cmd_1year = [
        sys.executable, "symphony_runner.py",
        "--config", symphony_file,
        "--forecast",
        "--forecast-days", "252",
        "--output-dir", "./forecast_1year"
    ]
    
    print("📅 Running 1-year forecast...")
    subprocess.run(cmd_1year)
    
    # 3-month forecast  
    cmd_3month = [
        sys.executable, "symphony_runner.py",
        "--config", symphony_file,
        "--forecast", 
        "--forecast-days", "63",
        "--output-dir", "./forecast_3month"
    ]
    
    print("📅 Running 3-month forecast...")
    subprocess.run(cmd_3month)
    
    print("✅ Forecast analysis complete!")
    print("📊 Check forecast_1year/ and forecast_3month/ directories")

if __name__ == "__main__":
    # Replace with your symphony file
    symphony_file = "sample_symphony_v2.json"
    run_forecast_analysis(symphony_file)
    '''
    
    with open("forecast_example.py", "w") as f:
        f.write(script_content)
    
    print("📄 Created: forecast_example.py")

def create_spy_plus_example():
    """Create example SPY-plus strategy"""
    
    spy_plus_strategy = {
        "name": "SPY Momentum Plus",
        "description": "SPY with momentum-based tactical allocation for beating SPY safely",
        "universe": ["SPY", "QQQ", "TLT"],
        "rebalance_frequency": "monthly",
        
        "logic": {
            "conditions": [
                {
                    "id": "spy_momentum_check",
                    "type": "if_statement",
                    "condition": {
                        "metric": "cumulative_return",
                        "asset_1": "SPY",
                        "operator": "greater_than",
                        "asset_2": {"type": "fixed_value", "value": 0.03},  # 3% threshold
                        "lookback_days": 60
                    },
                    "if_true": "momentum_allocation",
                    "if_false": "defensive_allocation"
                }
            ],
            
            "allocations": {
                "momentum_allocation": {
                    "type": "fixed_allocation",
                    "weights": {
                        "SPY": 0.7,   # Still mostly SPY
                        "QQQ": 0.3    # Add tech momentum
                    }
                },
                
                "defensive_allocation": {
                    "type": "fixed_allocation", 
                    "weights": {
                        "SPY": 0.8,   # Mostly SPY
                        "TLT": 0.2    # Some bonds for protection
                    }
                }
            }
        }
    }
    
    with open("spy_plus_example.json", "w") as f:
        json.dump(spy_plus_strategy, f, indent=2)
    
    print("📄 Created: spy_plus_example.json")
    print("💡 This strategy aims for 10-12% annual returns vs SPY's 8-10%")

def create_composer_conversion_example(composer_dsl):
    """Create Composer conversion example"""
    
    # Save the DSL to a file
    with open("composer_example.txt", "w") as f:
        f.write(composer_dsl)
    
    # Create our equivalent format
    our_equivalent = {
        "name": "200d MA 3x Leverage Strategy",
        "description": "Converted from Composer DSL - Complex leverage and volatility strategy",
        "universe": ["SPY", "QQQ", "TQQQ", "TECL", "PSQ", "UVXY", "TLT"],
        "rebalance_frequency": "daily",
        
        "logic": {
            "conditions": [
                {
                    "id": "spy_200ma_check",
                    "type": "if_statement",
                    "condition": {
                        "metric": "current_price",
                        "asset_1": "SPY",
                        "operator": "greater_than",
                        "asset_2": "SPY",  # This would need moving average implementation
                        "lookback_days": 200
                    },
                    "if_true": "bullish_allocation",
                    "if_false": "bearish_allocation"
                }
            ],
            
            "allocations": {
                "bullish_allocation": {
                    "type": "fixed_allocation",
                    "weights": {
                        "TQQQ": 1.0  # Simplified - would need RSI conditions
                    }
                },
                
                "bearish_allocation": {
                    "type": "fixed_allocation",
                    "weights": {
                        "TLT": 1.0  # Simplified defensive allocation
                    }
                }
            }
        }
    }
    
    with open("converted_composer_example.json", "w") as f:
        json.dump(our_equivalent, f, indent=2)
    
    print("📄 Created: composer_example.txt (original DSL)")
    print("📄 Created: converted_composer_example.json (our format)")
    print("💡 Note: Full conversion requires handling nested RSI conditions")
    
    # Create reconciliation script
    reconciliation_script = '''#!/usr/bin/env python3
"""
Composer Reconciliation Script

Compare our implementation with Composer's backtest results.
"""

import pandas as pd
import sys

def compare_allocations(composer_csv, our_results_csv):
    """Compare allocation decisions between Composer and our system"""
    
    print("🔍 Comparing allocation decisions...")
    
    # Load Composer results
    composer_data = pd.read_csv(composer_csv)
    composer_data['Date'] = pd.to_datetime(composer_data['Date'])
    
    print(f"📊 Composer data: {len(composer_data)} records")
    print(f"📅 Date range: {composer_data['Date'].min()} to {composer_data['Date'].max()}")
    
    # Show recent allocations
    print("\\n📈 Recent Composer allocations:")
    recent = composer_data.head(10)
    for _, row in recent.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        allocations = []
        for col in ['PSQ', 'QQQ', 'TECL', 'TQQQ', 'UVXY']:
            if row[col] != '-' and float(row[col].replace('%', '')) > 0:
                allocations.append(f"{col}: {row[col]}")
        print(f"  {date}: {', '.join(allocations) if allocations else 'No positions'}")
    
    print("\\n💡 To fully reconcile:")
    print("1. Implement exact RSI and moving average calculations")
    print("2. Handle nested if-then conditions properly") 
    print("3. Match rebalancing dates exactly")
    print("4. Validate against this CSV data")

if __name__ == "__main__":
    composer_csv = "Copy of 200d MA 3x Leverage.csv"
    compare_allocations(composer_csv, None)
    '''
    
    with open("reconciliation_example.py", "w") as f:
        f.write(reconciliation_script)
    
    print("📄 Created: reconciliation_example.py")

if __name__ == "__main__":
    main()
