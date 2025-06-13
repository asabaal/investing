#!/usr/bin/env python3
"""
Working Strategy Tester - Simple and Actually Works

Single strategy:
python working_strategy_tester.py black_swan_protection 2024-01-10 2024-03-31

Multi-strategy:
python working_strategy_tester.py "black_swan_protection,crisis_alpha_hunter" 2024-01-10 2024-03-31 --weights 0.7,0.3
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from enhanced_symphony_engine import EnhancedSymphonyEngine
from market_data_database import MarketDataDatabase
from simple_data_access import get_simple_data

def test_strategy(strategy_name: str, start_date: str, end_date: str, initial_capital: float = 10000):
    """Test a strategy with proper portfolio simulation"""
    
    # Load strategy
    with open('black_swan_symphonies.json', 'r') as f:
        config = json.load(f)
    
    if strategy_name not in config['symphonies']:
        available = list(config['symphonies'].keys())
        print(f"❌ Strategy '{strategy_name}' not found. Available: {available}")
        return
    
    strategy_config = config['symphonies'][strategy_name]
    
    print(f"🚀 TESTING STRATEGY: {strategy_config['name']}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print(f"🎯 Universe: {strategy_config['universe']}")
    print("=" * 60)
    
    # Initialize engine
    db = MarketDataDatabase()
    engine = EnhancedSymphonyEngine(strategy_config, db)
    
    # Get date range (business days)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.bdate_range(start=start_dt, end=end_dt)
    
    # Track portfolio
    portfolio_history = []
    current_holdings = {}  # symbol -> shares
    cash = initial_capital
    
    print(f"🔄 Simulating {len(date_range)} trading days...")
    
    for i, date in enumerate(date_range):
        try:
            # Get strategy allocations
            allocations = engine.calculate_allocations(date)
            
            if not allocations:
                print(f"⚠️ No allocations for {date.strftime('%Y-%m-%d')}")
                continue
            
            # Get current prices
            prices = {}
            for symbol in strategy_config['universe']:
                try:
                    data = get_simple_data(symbol, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
                    if not data.empty:
                        prices[symbol] = data['Close'].iloc[-1]
                except:
                    pass
            
            if not prices:
                print(f"⚠️ No prices available for {date.strftime('%Y-%m-%d')}")
                continue
            
            # Calculate current portfolio value
            portfolio_value = cash
            for symbol, shares in current_holdings.items():
                if symbol in prices:
                    portfolio_value += shares * prices[symbol]
            
            # Rebalance portfolio
            new_holdings = {}
            total_trades = 0
            
            # Sell all current holdings
            for symbol, shares in current_holdings.items():
                if symbol in prices and shares > 0:
                    cash += shares * prices[symbol]
                    total_trades += abs(shares)
            
            # Buy new allocations
            for symbol, weight in allocations.items():
                if weight > 0 and symbol in prices:
                    target_value = portfolio_value * weight
                    shares = target_value / prices[symbol]
                    new_holdings[symbol] = shares
                    cash -= target_value
                    total_trades += shares
            
            current_holdings = new_holdings
            
            # Calculate final portfolio value
            final_portfolio_value = cash
            for symbol, shares in current_holdings.items():
                if symbol in prices:
                    final_portfolio_value += shares * prices[symbol]
            
            # Record results
            daily_return = (final_portfolio_value / initial_capital - 1) * 100
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': final_portfolio_value,
                'cash': cash,
                'total_return': daily_return,
                'allocations': dict(allocations),
                'prices': dict(prices),
                'holdings': dict(current_holdings),
                'trades': total_trades
            })
            
            # Progress update
            if (i + 1) % 20 == 0 or i == len(date_range) - 1:
                progress = (i + 1) / len(date_range) * 100
                print(f"📊 Progress: {progress:.1f}% | "
                      f"Date: {date.strftime('%Y-%m-%d')} | "
                      f"Value: ${final_portfolio_value:,.2f} | "
                      f"Return: {daily_return:.2f}%")
            
        except Exception as e:
            print(f"❌ Error on {date.strftime('%Y-%m-%d')}: {e}")
            continue
    
    if not portfolio_history:
        print("❌ No valid trading data found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(portfolio_history)
    df.set_index('date', inplace=True)
    
    # Calculate performance metrics
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    # Daily returns
    df['daily_return'] = df['portfolio_value'].pct_change()
    daily_returns = df['daily_return'].dropna()
    
    if len(daily_returns) > 1:
        annualized_return = (final_value / initial_capital) ** (252 / len(df)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        running_max = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (daily_returns > 0).mean() * 100
        
        print(f"\n📊 PERFORMANCE METRICS")
        print("=" * 40)
        print(f"Initial Capital:     ${initial_capital:,.2f}")
        print(f"Final Value:         ${final_value:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Annualized Return:   {annualized_return*100:.2f}%")
        print(f"Volatility:          {volatility*100:.2f}%")
        print(f"Sharpe Ratio:        {sharpe:.2f}")
        print(f"Max Drawdown:        {max_drawdown*100:.2f}%")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Total Days:          {len(df)}")
        
        # Show final allocation
        print(f"\n🎯 FINAL ALLOCATION:")
        final_allocations = df['allocations'].iloc[-1]
        for symbol, weight in sorted(final_allocations.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                final_value_allocation = final_value * weight
                print(f"  {symbol}: {weight*100:.1f}% (${final_value_allocation:,.2f})")
    
    # Save results
    filename = f"strategy_test_{strategy_name}_{start_date}_{end_date}.csv"
    
    # Flatten nested data for CSV
    export_df = df[['portfolio_value', 'cash', 'total_return', 'trades']].copy()
    
    # Add allocation columns
    all_symbols = set()
    for allocations in df['allocations']:
        all_symbols.update(allocations.keys())
    
    for symbol in all_symbols:
        export_df[f'{symbol}_allocation'] = df['allocations'].apply(lambda x: x.get(symbol, 0))
        export_df[f'{symbol}_price'] = df['prices'].apply(lambda x: x.get(symbol, 0))
        export_df[f'{symbol}_shares'] = df['holdings'].apply(lambda x: x.get(symbol, 0))
    
    export_df.to_csv(filename)
    print(f"\n💾 Results saved to: {filename}")
    
    # Create chart
    create_performance_chart(df, strategy_name, start_date, end_date, initial_capital)
    
    print(f"\n✅ STRATEGY TEST COMPLETED!")
    return df

def create_performance_chart(df, strategy_name, start_date, end_date, initial_capital):
    """Create performance chart"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value
    ax1.plot(df.index, df['portfolio_value'], linewidth=2, color='blue')
    ax1.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title(f'{strategy_name} - Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Daily returns
    daily_returns = df['portfolio_value'].pct_change() * 100
    ax2.plot(df.index, daily_returns, alpha=0.7, color='green')
    ax2.set_title('Daily Returns (%)')
    ax2.set_ylabel('Daily Return (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Cumulative returns
    cumulative_returns = (df['portfolio_value'] / initial_capital - 1) * 100
    ax3.plot(df.index, cumulative_returns, linewidth=2, color='purple')
    ax3.set_title('Cumulative Returns (%)')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Allocation over time (stacked area)
    allocations_df = pd.DataFrame(list(df['allocations']))
    allocations_df.index = df.index
    
    if not allocations_df.empty:
        allocations_df.plot(kind='area', stacked=True, ax=ax4, alpha=0.7)
        ax4.set_title('Asset Allocation Over Time')
        ax4.set_ylabel('Allocation (%)')
        ax4.set_xlabel('Date')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    chart_filename = f"strategy_chart_{strategy_name}_{start_date}_{end_date}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Chart saved to: {chart_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python working_strategy_tester.py <strategy_name> <start_date> <end_date> [capital]")
        print("Example: python working_strategy_tester.py black_swan_protection 2024-01-10 2024-03-31 10000")
        sys.exit(1)
    
    strategy_name = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    capital = float(sys.argv[4]) if len(sys.argv) > 4 else 10000
    
    test_strategy(strategy_name, start_date, end_date, capital)