#!/usr/bin/env python3
"""
Paper Trading Workflow Example

This script demonstrates how to use the Trade Likelihood Estimator
for making informed paper trading decisions. Perfect for getting started!

Run this script to see the complete workflow from data loading to trade decisions.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to path to import our package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_likelihood import (
    TradeAnalyzer, 
    TradeSetup,
    quick_analysis,
    create_bidirectional_setup,
    get_conservative_config,
    get_aggressive_config
)


def generate_sample_price_data(initial_price=100.0, 
                              n_days=60, 
                              annual_vol=0.25, 
                              annual_drift=0.08):
    """
    Generate sample price data for demonstration.
    In real use, you'd load this from your broker API or data feed.
    """
    print("📊 Generating sample price data...")
    
    # Create realistic intraday timestamps (every 5 minutes during market hours)
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = []
    
    current = start_date
    while current < datetime.now():
        # Market hours: 9:30 AM to 4:00 PM EST (6.5 hours = 78 five-minute periods)
        if current.weekday() < 5:  # Weekdays only
            for minute in range(570, 960, 5):  # 9:30 AM to 4:00 PM in minutes from midnight
                ts = current.replace(hour=minute//60, minute=minute%60, second=0, microsecond=0)
                timestamps.append(ts)
        current += timedelta(days=1)
    
    # Generate price path using GBM
    n_points = len(timestamps)
    dt = 5 / (365.25 * 24 * 60)  # 5 minutes in years
    
    # Random walk parameters
    mu = annual_drift
    sigma = annual_vol
    
    # Generate price path
    random_shocks = np.random.normal(0, 1, n_points-1)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
    
    prices = [initial_price]
    for log_ret in log_returns:
        prices.append(prices[-1] * np.exp(log_ret))
    
    price_series = pd.Series(prices, index=timestamps[:len(prices)])
    
    print(f"✅ Generated {len(price_series)} price points over {n_days} days")
    print(f"   Price range: ${price_series.min():.2f} - ${price_series.max():.2f}")
    print(f"   Final price: ${price_series.iloc[-1]:.2f}")
    
    return price_series


def basic_workflow_example():
    """Demonstrate the basic workflow for paper trading decisions"""
    
    print("\n" + "="*60)
    print("🚀 BASIC PAPER TRADING WORKFLOW")
    print("="*60)
    
    # Step 1: Generate or load your price data
    prices = generate_sample_price_data()
    current_price = prices.iloc[-1]
    
    # Step 2: Initialize the analyzer
    print("\n📈 Initializing Trade Analyzer...")
    analyzer = TradeAnalyzer()
    
    # Step 3: Load price data
    print("📊 Loading price data...")
    summary = analyzer.load_price_data(prices)
    print(f"   Loaded {summary['data_points']} price points")
    print(f"   Data quality score: {summary['data_quality']['quality_score']:.2f}")
    print(f"   Parameters estimated: {summary['parameters_estimated']}")
    
    # Step 4: Check market status
    print("\n🔍 Market Status:")
    status = analyzer.get_market_status()
    print(f"   Current Price: ${float(status['current_price']):.2f}")
    print(f"   Market Regime: {status['regime']}")
    print(f"   Volatility: {status['volatility']} (confidence: {status['volatility_confidence']})")
    print(f"   Drift: {status['drift']} (confidence: {status['drift_confidence']})")
    
    # Step 5: Create simple trade setups
    print(f"\n⚙️  Creating trade setups for current price ${current_price:.2f}...")
    setups = analyzer.create_simple_setups(
        current_price=current_price,
        entry_pct=0.015,   # 1.5% entry distance
        stop_pct=0.008,    # 0.8% stop distance
        target_pct=0.025,  # 2.5% target distance  
        max_time_days=2.0  # 2-day maximum window
    )
    
    # Step 6: Analyze each setup type
    print("\n📊 ANALYZING TRADE SETUPS")
    print("-" * 40)
    
    results = {}
    for setup_name, setup in setups.items():
        print(f"\n{setup_name.upper()} SETUP:")
        analysis = analyzer.analyze_trade_setup(setup)
        results[setup_name] = analysis
        
        # Print key metrics
        if analysis.return_rate_total is not None:
            print(f"   Return Rate: {analysis.return_rate_total:.4f} per day ({analysis.return_rate_total*365:.1f}% annual)")
        
        if analysis.get_combined_entry_probability() is not None:
            print(f"   Entry Probability: {analysis.get_combined_entry_probability():.1%}")
        
        if analysis.expected_value_total is not None:
            print(f"   Expected Value: {analysis.expected_value_total:.4f}")
        
        print(f"   Best Direction: {analysis.get_best_direction()}")
        print(f"   Attractive Setup: {'✅ YES' if analysis.is_attractive_setup() else '❌ NO'}")
        
        # Show any warnings
        if analysis.warnings:
            print(f"   ⚠️  Warnings: {len(analysis.warnings)}")
            for warning in analysis.warnings[:2]:  # Show first 2
                print(f"      • {warning}")
    
    # Step 7: Compare setups
    print(f"\n📋 SETUP COMPARISON")
    print("-" * 40)
    
    comparison_df = analyzer.compare_setups(list(setups.values()), list(setups.keys()))
    
    # Print formatted comparison
    print("\nKey Metrics Summary:")
    for _, row in comparison_df.iterrows():
        name = row['Setup']
        return_rate = row['Return_Rate_Total']
        ev = row['EV_Total']
        attractive = row['Is_Attractive']
        
        print(f"   {name:12} | Return Rate: {return_rate:8.4f} | EV: {ev:8.4f} | Attractive: {'✅' if attractive else '❌'}")
    
    # Step 8: Recommendation
    print(f"\n🎯 PAPER TRADING RECOMMENDATION")
    print("-" * 40)
    
    best_setup = comparison_df.iloc[0]  # Already sorted by return rate
    best_name = best_setup['Setup']
    best_analysis = results[best_name.lower()]
    
    print(f"RECOMMENDED SETUP: {best_name}")
    print(f"Expected Return Rate: {best_setup['Return_Rate_Total']:.4f} per day")
    print(f"Expected Value: {best_setup['EV_Total']:.4f}")
    print(f"Best Direction: {best_setup['Best_Direction']}")
    
    if best_analysis.is_attractive_setup():
        print("✅ This is an ATTRACTIVE setup for paper trading!")
        print("\nNext Steps:")
        print("1. Set up the trade in your paper trading account")
        print("2. Monitor entry conditions") 
        print("3. Execute when price hits entry levels")
        print("4. Follow strict stop-loss and take-profit rules")
    else:
        print("❌ Current market conditions are not favorable")
        print("Consider waiting for better setups or different time frames")
    
    return best_analysis


def quick_analysis_example():
    """Demonstrate the quick analysis function for fast decisions"""
    
    print("\n" + "="*60)
    print("⚡ QUICK ANALYSIS EXAMPLE")
    print("="*60)
    
    # Generate some sample data
    prices = generate_sample_price_data(n_days=30)
    current_price = prices.iloc[-1]
    
    print(f"\n🔥 Quick analysis for ${current_price:.2f}...")
    
    # Use the quick_analysis function
    result = quick_analysis(
        prices=prices.values,  # Convert to list/array
        current_price=current_price,
        entry_pct=0.02,   # 2% entry
        stop_pct=0.01,    # 1% stop
        target_pct=0.03   # 3% target
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    print(f"\n📊 QUICK ANALYSIS RESULTS")
    print(f"   Current Price: ${result['current_price']:.2f}")
    print(f"   Market Regime: {result['market_regime']}")
    print(f"   Volatility: {result['volatility']}")
    print(f"   Drift: {result['drift']}")
    print(f"   Best Direction: {result['best_direction']}")
    print(f"   Return Rate: {result['return_rate']:.4f} per day")
    print(f"   Entry Probability: {result['combined_entry_prob']:.1%}")
    print(f"   Expected Value: {result['expected_value']:.4f}")
    print(f"   Attractive: {'✅ YES' if result['is_attractive'] else '❌ NO'}")
    print(f"   Recommendation: {result['recommendation']}")
    
    if result['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in result['warnings']:
            print(f"   • {warning}")


def advanced_configuration_example():
    """Show how to use different configurations for different trading styles"""
    
    print("\n" + "="*60)
    print("🔧 ADVANCED CONFIGURATION EXAMPLE")
    print("="*60)
    
    prices = generate_sample_price_data(n_days=45)
    current_price = prices.iloc[-1]
    
    print(f"Comparing different estimation configurations for ${current_price:.2f}...")
    
    configs = {
        'Conservative': get_conservative_config(),
        'Aggressive': get_aggressive_config(),
        'Default': None  # Use default config
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{config_name} Configuration:")
        
        # Create analyzer with specific config
        analyzer = TradeAnalyzer(estimation_config=config)
        analyzer.load_price_data(prices)
        
        # Create standard setup
        setup = create_bidirectional_setup(current_price)
        analysis = analyzer.analyze_trade_setup(setup)
        
        results[config_name] = analysis
        
        # Print key differences
        print(f"   Volatility: {analysis.market_params.volatility:.2%} "
              f"(confidence: {analysis.market_params.volatility_confidence:.1%})")
        print(f"   Drift: {analysis.market_params.drift:.2%} "
              f"(confidence: {analysis.market_params.drift_confidence:.1%})")
        print(f"   Return Rate: {analysis.return_rate_total:.4f}")
        print(f"   Regime: {analysis.market_params.regime}")
    
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("-" * 50)
    print("Config        | Return Rate | Vol Confidence | Drift Confidence")
    print("-" * 50)
    
    for name, analysis in results.items():
        rr = analysis.return_rate_total or 0
        vol_conf = analysis.market_params.volatility_confidence
        drift_conf = analysis.market_params.drift_confidence
        print(f"{name:12} | {rr:10.4f} | {vol_conf:13.1%} | {drift_conf:15.1%}")


def real_time_monitoring_example():
    """Simulate real-time price monitoring and trade decisions"""
    
    print("\n" + "="*60) 
    print("📡 REAL-TIME MONITORING SIMULATION")
    print("="*60)
    
    # Initialize with historical data
    historical_prices = generate_sample_price_data(n_days=30)
    analyzer = TradeAnalyzer()
    analyzer.load_price_data(historical_prices)
    
    print("📊 Loaded historical data for parameter estimation")
    
    # Simulate real-time price updates
    print("\n⏰ Simulating real-time price updates...")
    
    current_price = historical_prices.iloc[-1]
    attractive_count = 0
    
    # Simulate 10 price updates
    for i in range(10):
        # Simulate new price (small random walk)
        price_change = np.random.normal(0, current_price * 0.005)  # 0.5% std move
        current_price += price_change
        
        # Add new price to analyzer
        new_timestamp = pd.Timestamp.now() + pd.Timedelta(minutes=i*5)
        analyzer.add_new_price(current_price, new_timestamp, update_parameters=True)
        
        # Quick setup analysis
        setup = create_bidirectional_setup(current_price, max_time_days=1.0)
        analysis = analyzer.analyze_trade_setup(setup)
        
        # Check if setup is attractive
        is_attractive = analysis.is_attractive_setup()
        if is_attractive:
            attractive_count += 1
        
        print(f"   Update {i+1:2d}: ${current_price:7.2f} | "
              f"Return Rate: {analysis.return_rate_total:7.4f} | "
              f"Attractive: {'✅' if is_attractive else '❌'} | "
              f"Best: {analysis.get_best_direction() or 'None':5s}")
    
    print(f"\n📈 Summary: {attractive_count}/10 updates showed attractive setups")
    
    if attractive_count >= 3:
        print("✅ Good market conditions detected! Consider active paper trading.")
    else:
        print("⚠️  Market conditions are mixed. Be selective with trades.")


def main():
    """Run all examples"""
    
    print("🎯 TRADE LIKELIHOOD ESTIMATOR - PAPER TRADING EXAMPLES")
    print("=" * 65)
    print("This script demonstrates how to use the system for paper trading decisions.")
    print("In real trading, replace the sample data with your actual market data.")
    
    try:
        # Run examples
        best_analysis = basic_workflow_example()
        quick_analysis_example() 
        advanced_configuration_example()
        real_time_monitoring_example()
        
        print("\n" + "="*65)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*65)
        print("\n📚 NEXT STEPS FOR PAPER TRADING:")
        print("1. Replace sample data with real market data from your broker")
        print("2. Adjust setup parameters (entry %, stop %, target %) to your strategy")
        print("3. Set up automatic data feeds for real-time analysis")
        print("4. Create alerts when attractive setups are detected")
        print("5. Track your paper trading results vs. model predictions")
        
        print(f"\n💡 PRO TIP: The model predicted return rate of {best_analysis.return_rate_total:.4f} per day")
        print("   means you should expect roughly this performance in paper trading.")
        print("   Use this to set realistic expectations and validate the model!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducible examples
    np.random.seed(42)
    
    main()