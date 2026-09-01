#!/usr/bin/env python3
"""
Probability & Profit Calculator
1. Calculate probability of realizing profit based on historical data
2. Calculate minimum win rate needed for long-term profitability
"""

import pandas as pd
import numpy as np
import sqlite3
from market_data_database import get_default_database_path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_trade_probabilities(symbol, entry_price, target_price, stop_price, days_back=365):
    """
    Calculate actual probabilities based on your specific trade setup
    """
    print(f"🎯 PROBABILITY ANALYSIS FOR {symbol} TRADE")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Target: ${target_price:.2f} (+{((target_price-entry_price)/entry_price)*100:.2f}%)")
    print(f"Stop: ${stop_price:.2f} ({((entry_price-stop_price)/entry_price)*100:.2f}% risk)")
    print("=" * 60)
    
    # Load data
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
    
    print(f"📊 Analyzing {len(daily_data)} trading days...")
    
    # Find all days where price was near your entry (within 2%)
    entry_tolerance = 0.02  # 2% tolerance
    potential_entries = []
    
    for i, row in daily_data.iterrows():
        close_price = row['close']
        # Check if close price is within tolerance of entry price
        if abs((close_price - entry_price) / entry_price) <= entry_tolerance:
            potential_entries.append({
                'date': row['date'],
                'entry_index': i,
                'actual_entry': close_price
            })
    
    print(f"📈 Found {len(potential_entries)} potential entry points near ${entry_price:.2f}")
    
    if len(potential_entries) < 10:
        print("⚠️ Too few entry scenarios - expanding search range...")
        entry_tolerance = 0.05  # Expand to 5%
        potential_entries = []
        
        for i, row in daily_data.iterrows():
            close_price = row['close']
            if abs((close_price - entry_price) / entry_price) <= entry_tolerance:
                potential_entries.append({
                    'date': row['date'],
                    'entry_index': i,
                    'actual_entry': close_price
                })
        print(f"📈 Expanded search: Found {len(potential_entries)} potential entry points")
    
    # Analyze outcomes from each potential entry
    trade_outcomes = []
    
    for entry in potential_entries:
        entry_idx = entry['entry_index']
        actual_entry_price = entry['actual_entry']
        
        # Calculate target and stop based on actual entry price
        actual_target = actual_entry_price * (target_price / entry_price)
        actual_stop = actual_entry_price * (stop_price / entry_price)
        
        # Look forward for target or stop hit
        target_hit = False
        stop_hit = False
        days_to_target = None
        days_to_stop = None
        outcome = None
        
        # Look ahead up to 30 trading days (about 6 weeks)
        for j in range(entry_idx + 1, min(entry_idx + 31, len(daily_data))):
            future_high = daily_data.iloc[j]['high']
            future_low = daily_data.iloc[j]['low']
            days_elapsed = j - entry_idx
            
            # Check target hit
            if not target_hit and future_high >= actual_target:
                target_hit = True
                days_to_target = days_elapsed
            
            # Check stop hit
            if not stop_hit and future_low <= actual_stop:
                stop_hit = True
                days_to_stop = days_elapsed
            
            # Determine outcome (which happened first)
            if target_hit and stop_hit:
                if days_to_target <= days_to_stop:
                    outcome = 'target_first'
                else:
                    outcome = 'stop_first'
                break
            elif target_hit:
                outcome = 'target_only'
                break
            elif stop_hit:
                outcome = 'stop_only'
                break
        
        # If neither hit within 30 days, consider it a timeout
        if outcome is None:
            outcome = 'timeout'
        
        trade_outcomes.append({
            'entry_date': entry['date'],
            'entry_price': actual_entry_price,
            'target_price': actual_target,
            'stop_price': actual_stop,
            'outcome': outcome,
            'days_to_resolution': days_to_target if outcome.startswith('target') else (days_to_stop if outcome.startswith('stop') else 30),
            'profitable': outcome.startswith('target')
        })
    
    # Calculate probabilities
    if not trade_outcomes:
        print("❌ No trade scenarios found")
        return None
        
    outcomes_df = pd.DataFrame(trade_outcomes)
    
    total_trades = len(outcomes_df)
    profitable_trades = len(outcomes_df[outcomes_df['profitable'] == True])
    losing_trades = len(outcomes_df[outcomes_df['outcome'].str.startswith('stop')])
    timeout_trades = len(outcomes_df[outcomes_df['outcome'] == 'timeout'])
    
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
    timeout_rate = (timeout_trades / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\n🎲 PROBABILITY RESULTS:")
    print(f"Total scenarios analyzed: {total_trades}")
    print(f"Profitable outcomes: {profitable_trades} ({win_rate:.1f}%)")
    print(f"Loss outcomes: {losing_trades} ({loss_rate:.1f}%)")
    print(f"Timeout (neither hit): {timeout_trades} ({timeout_rate:.1f}%)")
    
    # Average timing for winners and losers
    winners = outcomes_df[outcomes_df['profitable'] == True]
    losers = outcomes_df[outcomes_df['outcome'].str.startswith('stop')]
    
    if len(winners) > 0:
        avg_days_to_profit = winners['days_to_resolution'].mean()
        print(f"\n⏱️ Average days to profit: {avg_days_to_profit:.1f}")
    
    if len(losers) > 0:
        avg_days_to_loss = losers['days_to_resolution'].mean()
        print(f"⏱️ Average days to loss: {avg_days_to_loss:.1f}")
    
    conn.close()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate / 100,  # As decimal
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'timeout_trades': timeout_trades,
        'outcomes_df': outcomes_df,
        'avg_days_to_profit': avg_days_to_profit if len(winners) > 0 else None,
        'avg_days_to_loss': avg_days_to_loss if len(losers) > 0 else None
    }

def calculate_minimum_win_rate(target_gain_pct, max_loss_pct):
    """
    Calculate minimum win rate needed for profitability
    Formula: Min Win Rate = |Loss| / (Gain + |Loss|)
    """
    print(f"\n🧮 MINIMUM WIN RATE CALCULATION")
    print("=" * 40)
    
    # Your trade setup
    gain = target_gain_pct  # +3.88%
    loss = abs(max_loss_pct)  # 0.73%
    
    min_win_rate = loss / (gain + loss)
    
    print(f"Potential Gain: +{gain:.2f}%")
    print(f"Potential Loss: -{loss:.2f}%")
    print(f"Risk/Reward Ratio: 1:{gain/loss:.2f}")
    
    print(f"\n🎯 MINIMUM WIN RATE FOR BREAKEVEN: {min_win_rate*100:.1f}%")
    
    # Show different profit scenarios
    print(f"\n💰 WIN RATE SCENARIOS:")
    for target_profit in [5, 10, 15, 20, 25]:
        required_win_rate = (loss + target_profit) / (gain + loss)
        print(f"For +{target_profit}% annual profit: Need {required_win_rate*100:.1f}% win rate")
    
    return min_win_rate

def calculate_expected_return(win_rate, gain_pct, loss_pct, trades_per_year):
    """
    Calculate expected annual return
    """
    expected_return_per_trade = (win_rate * gain_pct) + ((1 - win_rate) * (-loss_pct))
    annual_return = expected_return_per_trade * trades_per_year
    
    return expected_return_per_trade, annual_return

def analyze_trade_probability(symbol, entry_price, target_price, stop_price, days_back=365):
    """Complete probability analysis for any trade"""
    
    print(f"\n🎯 ANALYZING TRADE: {symbol}")
    print("=" * 60)
    
    # Calculate gain/loss percentages
    gain_pct = ((target_price - entry_price) / entry_price) * 100
    loss_pct = ((entry_price - stop_price) / entry_price) * 100
    
    # 1. Calculate actual probabilities from historical data
    prob_results = calculate_trade_probabilities(symbol, entry_price, target_price, stop_price, days_back)
    
    # 2. Calculate minimum win rate needed
    min_win_rate = calculate_minimum_win_rate(gain_pct, loss_pct)
    
    # 3. Compare actual vs required win rate
    if prob_results and prob_results['win_rate'] > 0:
        actual_win_rate = prob_results['win_rate']
        
        print(f"\n📊 PROFITABILITY ANALYSIS:")
        print("=" * 40)
        print(f"Historical Win Rate: {actual_win_rate*100:.1f}%")
        print(f"Required Win Rate: {min_win_rate*100:.1f}%")
        
        if actual_win_rate > min_win_rate:
            margin = actual_win_rate - min_win_rate
            print(f"✅ PROFITABLE! Margin of safety: +{margin*100:.1f} percentage points")
            
            # Calculate expected returns with different trade frequencies
            print(f"\n🎯 EXPECTED RETURNS:")
            for freq in [20, 30, 40, 50]:
                expected_per_trade, annual_return = calculate_expected_return(
                    actual_win_rate, gain_pct, loss_pct, freq
                )
                print(f"{freq} trades/year: {annual_return:.1f}% annual return")
                
        else:
            shortfall = min_win_rate - actual_win_rate
            print(f"❌ NOT PROFITABLE. Need +{shortfall*100:.1f} percentage points higher win rate")
    
    # 4. Risk management insights
    print(f"\n⚠️ RISK MANAGEMENT INSIGHTS:")
    print(f"Risk per trade: {loss_pct:.2f}%")
    print(f"Reward per trade: {gain_pct:.2f}%")
    print(f"Risk:Reward = 1:{gain_pct/loss_pct:.1f}")
    
    if prob_results and prob_results.get('avg_days_to_profit') and prob_results.get('avg_days_to_loss'):
        print(f"Avg time to profit: {prob_results['avg_days_to_profit']:.1f} days")
        print(f"Avg time to loss: {prob_results['avg_days_to_loss']:.1f} days")
    
    return prob_results

def main():
    """Demo - analyze multiple trades"""
    
    # CHWY trade parameters
    print("🔍 CHWY ANALYSIS:")
    analyze_trade_probability("CHWY", 35.86, 37.25, 35.60)
    
    print("\n" + "="*80)
    
    # BE trade parameters  
    print("🔍 BE ANALYSIS:")
    analyze_trade_probability("BE", 38.27, 40.50, 37.75)

if __name__ == "__main__":
    main()