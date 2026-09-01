#!/usr/bin/env python3
"""
Trade Optimization Analyzer
Analyzes what would make an unprofitable trade profitable through various adjustments
"""

import pandas as pd
import numpy as np
import sqlite3
from market_data_database import get_default_database_path
from probability_profit_calculator import calculate_trade_probabilities, calculate_minimum_win_rate, calculate_expected_return
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_price_point_optimization(symbol, target_gain_pct, stop_loss_pct, days_back=365, price_increment=0.25):
    """
    Test different entry price points while keeping percentage gains/losses constant
    """
    print(f"🎯 PRICE POINT OPTIMIZATION ANALYSIS: {symbol}")
    print(f"Target Gain: {target_gain_pct:.2f}%")
    print(f"Stop Loss: {stop_loss_pct:.2f}%")
    print("=" * 80)
    
    # Get historical price range
    conn = sqlite3.connect(get_default_database_path())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    query = """
    SELECT symbol, datetime, open, high, low, close, volume
    FROM intraday_data 
    WHERE symbol = ? AND DATE(datetime) >= DATE(?)
    ORDER BY datetime
    """
    
    price_data = pd.read_sql_query(query, conn, params=[symbol, start_date.strftime('%Y-%m-%d')])
    
    if price_data.empty:
        print(f"❌ No data found for {symbol}")
        return None
    
    # Aggregate to daily
    price_data['date'] = pd.to_datetime(price_data['datetime']).dt.date
    daily_data = price_data.groupby(['symbol', 'date']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    # Find price range
    min_price = daily_data['low'].min()
    max_price = daily_data['high'].max()
    
    print(f"Historical Price Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Test different entry points
    optimization_results = []
    
    # Start from a reasonable range around current levels
    test_start = max(min_price + 1, min_price * 1.1)  # Start 10% above min
    test_end = min(max_price - 1, max_price * 0.9)    # End 10% below max
    
    test_prices = np.arange(test_start, test_end, price_increment)
    
    print(f"\n🔍 Testing {len(test_prices)} price points from ${test_start:.2f} to ${test_end:.2f}")
    print("Entry Price | Target Price | Stop Price | Scenarios | Win Rate | Profitable?")
    print("-" * 80)
    
    for entry_price in test_prices:
        target_price = entry_price * (1 + target_gain_pct/100)
        stop_price = entry_price * (1 - stop_loss_pct/100)
        
        # Calculate probabilities for this entry point
        prob_results = calculate_trade_probabilities(symbol, entry_price, target_price, stop_price, days_back)
        
        if prob_results and prob_results['total_trades'] >= 5:  # Need at least 5 scenarios
            win_rate = prob_results['win_rate']
            required_win_rate = stop_loss_pct / (target_gain_pct + stop_loss_pct)
            is_profitable = win_rate > required_win_rate
            
            optimization_results.append({
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'scenarios': prob_results['total_trades'],
                'win_rate': win_rate,
                'required_win_rate': required_win_rate,
                'is_profitable': is_profitable,
                'margin_of_safety': win_rate - required_win_rate,
                'avg_days_to_profit': prob_results.get('avg_days_to_profit', 0),
                'avg_days_to_loss': prob_results.get('avg_days_to_loss', 0)
            })
            
            status = "✅ YES" if is_profitable else "❌ NO"
            print(f"${entry_price:7.2f} | ${target_price:9.2f} | ${stop_price:8.2f} | {prob_results['total_trades']:9d} | {win_rate*100:7.1f}% | {status}")
    
    conn.close()
    
    if not optimization_results:
        print("❌ No viable entry points found with sufficient data")
        return None
    
    # Analyze results
    results_df = pd.DataFrame(optimization_results)
    profitable_trades = results_df[results_df['is_profitable'] == True]
    
    print(f"\n📊 OPTIMIZATION SUMMARY:")
    print(f"Total price points tested: {len(results_df)}")
    print(f"Profitable entry points found: {len(profitable_trades)}")
    
    if len(profitable_trades) > 0:
        best_trade = profitable_trades.loc[profitable_trades['margin_of_safety'].idxmax()]
        print(f"\n🏆 BEST ENTRY POINT:")
        print(f"Entry Price: ${best_trade['entry_price']:.2f}")
        print(f"Target Price: ${best_trade['target_price']:.2f}")
        print(f"Stop Price: ${best_trade['stop_price']:.2f}")
        print(f"Win Rate: {best_trade['win_rate']*100:.1f}%")
        print(f"Margin of Safety: +{best_trade['margin_of_safety']*100:.1f} percentage points")
        print(f"Historical Scenarios: {best_trade['scenarios']}")
        
        # Show price range for profitable trades
        min_profitable_entry = profitable_trades['entry_price'].min()
        max_profitable_entry = profitable_trades['entry_price'].max()
        print(f"\n💡 PROFITABLE ENTRY RANGE: ${min_profitable_entry:.2f} - ${max_profitable_entry:.2f}")
    else:
        print("\n❌ NO PROFITABLE ENTRY POINTS FOUND")
        print("Consider adjusting target/stop percentages or choosing a different security")
    
    return results_df

def analyze_risk_reward_optimization(symbol, entry_price, current_target_pct, current_stop_pct, days_back=365):
    """
    Test different risk/reward ratios to find optimal setup
    """
    print(f"\n🎯 RISK/REWARD OPTIMIZATION: {symbol} @ ${entry_price:.2f}")
    print("=" * 80)
    
    optimization_results = []
    
    # Test different target percentages (keeping stop constant)
    print(f"\n📈 TESTING DIFFERENT TARGET LEVELS (Stop fixed at {current_stop_pct:.2f}%):")
    print("Target % | Stop % | Risk:Reward | Min Win Rate | Actual Win Rate | Profitable?")
    print("-" * 75)
    
    target_tests = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for target_pct in target_tests:
        target_price = entry_price * (1 + target_pct/100)
        stop_price = entry_price * (1 - current_stop_pct/100)
        
        prob_results = calculate_trade_probabilities(symbol, entry_price, target_price, stop_price, days_back)
        
        if prob_results and prob_results['total_trades'] >= 3:
            min_win_rate = calculate_minimum_win_rate(target_pct, current_stop_pct)
            actual_win_rate = prob_results['win_rate']
            is_profitable = actual_win_rate > min_win_rate
            risk_reward_ratio = target_pct / current_stop_pct
            
            optimization_results.append({
                'type': 'target_optimization',
                'target_pct': target_pct,
                'stop_pct': current_stop_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'min_win_rate': min_win_rate,
                'actual_win_rate': actual_win_rate,
                'is_profitable': is_profitable,
                'scenarios': prob_results['total_trades']
            })
            
            status = "✅ YES" if is_profitable else "❌ NO"
            print(f"{target_pct:7.1f}% | {current_stop_pct:5.1f}% | {risk_reward_ratio:10.1f}:1 | {min_win_rate*100:11.1f}% | {actual_win_rate*100:14.1f}% | {status}")
    
    # Test different stop percentages (keeping target constant)
    print(f"\n📉 TESTING DIFFERENT STOP LEVELS (Target fixed at {current_target_pct:.2f}%):")
    print("Target % | Stop % | Risk:Reward | Min Win Rate | Actual Win Rate | Profitable?")
    print("-" * 75)
    
    stop_tests = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    for stop_pct in stop_tests:
        target_price = entry_price * (1 + current_target_pct/100)
        stop_price = entry_price * (1 - stop_pct/100)
        
        prob_results = calculate_trade_probabilities(symbol, entry_price, target_price, stop_price, days_back)
        
        if prob_results and prob_results['total_trades'] >= 3:
            min_win_rate = calculate_minimum_win_rate(current_target_pct, stop_pct)
            actual_win_rate = prob_results['win_rate']
            is_profitable = actual_win_rate > min_win_rate
            risk_reward_ratio = current_target_pct / stop_pct
            
            optimization_results.append({
                'type': 'stop_optimization',
                'target_pct': current_target_pct,
                'stop_pct': stop_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'min_win_rate': min_win_rate,
                'actual_win_rate': actual_win_rate,
                'is_profitable': is_profitable,
                'scenarios': prob_results['total_trades']
            })
            
            status = "✅ YES" if is_profitable else "❌ NO"
            print(f"{current_target_pct:7.1f}% | {stop_pct:5.1f}% | {risk_reward_ratio:10.1f}:1 | {min_win_rate*100:11.1f}% | {actual_win_rate*100:14.1f}% | {status}")
    
    return optimization_results

def analyze_entry_tolerance_optimization(symbol, base_entry_price, target_price, stop_price, days_back=365):
    """
    Test different entry tolerances to maximize sample size and win rate
    """
    print(f"\n🎯 ENTRY TOLERANCE OPTIMIZATION: {symbol}")
    print(f"Base Entry: ${base_entry_price:.2f} | Target: ${target_price:.2f} | Stop: ${stop_price:.2f}")
    print("=" * 80)
    
    tolerance_tests = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15]
    
    print("Tolerance | Sample Size | Win Rate | Required | Profitable?")
    print("-" * 55)
    
    optimization_results = []
    target_gain_pct = ((target_price - base_entry_price) / base_entry_price) * 100
    stop_loss_pct = ((base_entry_price - stop_price) / base_entry_price) * 100
    min_win_rate = calculate_minimum_win_rate(target_gain_pct, stop_loss_pct)
    
    for tolerance in tolerance_tests:
        # Get data
        conn = sqlite3.connect(get_default_database_path())
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT symbol, datetime, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol = ? AND DATE(datetime) >= DATE(?)
        ORDER BY datetime
        """
        
        price_data = pd.read_sql_query(query, conn, params=[symbol, start_date.strftime('%Y-%m-%d')])
        
        if price_data.empty:
            continue
            
        # Aggregate to daily
        price_data['date'] = pd.to_datetime(price_data['datetime']).dt.date
        daily_data = price_data.groupby(['symbol', 'date']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.sort_values('date').reset_index(drop=True)
        
        # Find entry scenarios with this tolerance
        potential_entries = []
        for i, row in daily_data.iterrows():
            close_price = row['close']
            if abs((close_price - base_entry_price) / base_entry_price) <= tolerance:
                potential_entries.append(i)
        
        sample_size = len(potential_entries)
        
        if sample_size >= 5:  # Need minimum sample
            # Calculate win rate with this tolerance
            prob_results = calculate_trade_probabilities(symbol, base_entry_price, target_price, stop_price, days_back)
            
            # Manually calculate with this specific tolerance
            wins = 0
            total_trades = 0
            
            for entry_idx in potential_entries:
                if entry_idx >= len(daily_data) - 1:
                    continue
                    
                entry_price = daily_data.iloc[entry_idx]['close']
                actual_target = entry_price * (target_price / base_entry_price)
                actual_stop = entry_price * (stop_price / base_entry_price)
                
                # Look forward for resolution
                for j in range(entry_idx + 1, min(entry_idx + 31, len(daily_data))):
                    future_high = daily_data.iloc[j]['high']
                    future_low = daily_data.iloc[j]['low']
                    
                    target_hit = future_high >= actual_target
                    stop_hit = future_low <= actual_stop
                    
                    if target_hit or stop_hit:
                        total_trades += 1
                        if target_hit and (not stop_hit or j == entry_idx + 1):  # Target hit first or same day
                            wins += 1
                        break
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            is_profitable = win_rate > min_win_rate
            
            optimization_results.append({
                'tolerance': tolerance,
                'sample_size': sample_size,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'is_profitable': is_profitable
            })
            
            status = "✅ YES" if is_profitable else "❌ NO"
            print(f"{tolerance*100:8.1f}% | {sample_size:10d} | {win_rate*100:7.1f}% | {min_win_rate*100:7.1f}% | {status}")
        
        conn.close()
    
    return optimization_results

def generate_optimization_recommendations(symbol, entry_price, target_price, stop_price):
    """
    Generate comprehensive recommendations for making the trade profitable
    """
    print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS: {symbol}")
    print("=" * 80)
    
    # Current setup analysis
    target_gain_pct = ((target_price - entry_price) / entry_price) * 100
    stop_loss_pct = ((entry_price - stop_price) / entry_price) * 100
    current_min_win_rate = calculate_minimum_win_rate(target_gain_pct, stop_loss_pct)
    
    current_prob = calculate_trade_probabilities(symbol, entry_price, target_price, stop_price)
    current_win_rate = current_prob['win_rate'] if current_prob else 0
    
    print(f"📊 CURRENT SETUP:")
    print(f"Entry: ${entry_price:.2f} | Target: ${target_price:.2f} (+{target_gain_pct:.2f}%) | Stop: ${stop_price:.2f} (-{stop_loss_pct:.2f}%)")
    print(f"Current Win Rate: {current_win_rate*100:.1f}% | Required: {current_min_win_rate*100:.1f}%")
    print(f"Shortfall: {(current_min_win_rate - current_win_rate)*100:.1f} percentage points")
    
    print(f"\n💡 RECOMMENDATIONS TO MAKE PROFITABLE:")
    
    # Strategy 1: Reduce target to improve win rate
    print(f"\n1️⃣ REDUCE TARGET (Easier wins):")
    for new_target_pct in [4, 3, 2.5, 2]:
        new_min_win_rate = calculate_minimum_win_rate(new_target_pct, stop_loss_pct)
        new_target_price = entry_price * (1 + new_target_pct/100)
        
        improvement_needed = (current_min_win_rate - new_min_win_rate) * 100
        if new_min_win_rate < current_win_rate:
            print(f"   Target: +{new_target_pct:.1f}% (${new_target_price:.2f}) → Need {new_min_win_rate*100:.1f}% win rate ✅ PROFITABLE!")
        else:
            print(f"   Target: +{new_target_pct:.1f}% (${new_target_price:.2f}) → Need {new_min_win_rate*100:.1f}% win rate (reduces requirement by {improvement_needed:.1f}%)")
    
    # Strategy 2: Tighten stop to improve risk/reward
    print(f"\n2️⃣ TIGHTEN STOP LOSS (Better risk/reward):")
    for new_stop_pct in [1.0, 0.75, 0.5]:
        new_min_win_rate = calculate_minimum_win_rate(target_gain_pct, new_stop_pct)
        new_stop_price = entry_price * (1 - new_stop_pct/100)
        
        if new_min_win_rate < current_win_rate:
            print(f"   Stop: -{new_stop_pct:.1f}% (${new_stop_price:.2f}) → Need {new_min_win_rate*100:.1f}% win rate ✅ PROFITABLE!")
        else:
            improvement_needed = (current_min_win_rate - new_min_win_rate) * 100
            print(f"   Stop: -{new_stop_pct:.1f}% (${new_stop_price:.2f}) → Need {new_min_win_rate*100:.1f}% win rate (reduces requirement by {improvement_needed:.1f}%)")
    
    # Strategy 3: Find better entry points
    print(f"\n3️⃣ FIND BETTER ENTRY POINTS:")
    print(f"   Run price point optimization to find entry levels with higher historical win rates")
    print(f"   Current entry ${entry_price:.2f} may not be optimal based on historical patterns")
    
    # Strategy 4: Increase sample size
    print(f"\n4️⃣ INCREASE SAMPLE SIZE:")
    if current_prob and current_prob['total_trades'] < 20:
        print(f"   Current sample: {current_prob['total_trades']} scenarios (low)")
        print(f"   Use wider entry tolerance or longer historical period")
        print(f"   More data points = more reliable win rate estimate")
    
    # Strategy 5: Consider different timeframes
    print(f"\n5️⃣ ALTERNATIVE APPROACHES:")
    print(f"   • Wait for better market conditions (higher volatility periods)")
    print(f"   • Use multiple position sizing (scale in/out)")
    print(f"   • Consider swing trading with wider stops and targets")
    print(f"   • Look for this setup on different securities with better win rates")

def main():
    """
    Complete optimization analysis for BE trade
    """
    # BE trade parameters
    symbol = "BE"
    entry_price = 38.27
    target_price = 40.50
    stop_price = 37.75
    
    target_gain_pct = ((target_price - entry_price) / entry_price) * 100
    stop_loss_pct = ((entry_price - stop_price) / entry_price) * 100
    
    print("🔍 COMPREHENSIVE TRADE OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # 1. Price point optimization
    price_results = analyze_price_point_optimization(symbol, target_gain_pct, stop_loss_pct)
    
    # 2. Risk/reward optimization  
    rr_results = analyze_risk_reward_optimization(symbol, entry_price, target_gain_pct, stop_loss_pct)
    
    # 3. Entry tolerance optimization
    tolerance_results = analyze_entry_tolerance_optimization(symbol, entry_price, target_price, stop_price)
    
    # 4. Generate recommendations
    generate_optimization_recommendations(symbol, entry_price, target_price, stop_price)
    
    print(f"\n✅ OPTIMIZATION ANALYSIS COMPLETE!")
    print(f"Use these insights to improve your BE trading strategy.")

if __name__ == "__main__":
    main()