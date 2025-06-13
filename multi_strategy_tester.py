#!/usr/bin/env python3
"""
Multi-Strategy Tester - Based on Working Strategy Tester

python multi_strategy_tester.py --strategies black_swan_protection crisis_alpha_hunter --weights 0.7 0.3 --start 2024-01-01 --end 2024-12-31 --capital 10000
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse
from enhanced_symphony_engine import EnhancedSymphonyEngine
from market_data_database import MarketDataDatabase
from simple_data_access import get_simple_data

def test_multi_strategy(strategies: list, weights: list, start_date: str, end_date: str, initial_capital: float = 10000):
    """Test multiple strategies with proper portfolio simulation"""
    
    # Load strategy configs
    with open('black_swan_symphonies.json', 'r') as f:
        config = json.load(f)
    
    # Validate strategies
    strategy_configs = {}
    for strategy_name in strategies:
        if strategy_name not in config['symphonies']:
            available = list(config['symphonies'].keys())
            print(f"❌ Strategy '{strategy_name}' not found. Available: {available}")
            return
        strategy_configs[strategy_name] = config['symphonies'][strategy_name]
    
    # Validate weights
    if len(weights) != len(strategies):
        print(f"❌ Number of weights ({len(weights)}) must match strategies ({len(strategies)})")
        return
    
    if abs(sum(weights) - 1.0) > 0.001:
        print(f"❌ Weights must sum to 1.0, got {sum(weights):.3f}")
        return
    
    print(f"🚀 TESTING MULTI-STRATEGY PORTFOLIO")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print("=" * 60)
    
    print(f"🎯 STRATEGIES:")
    for strategy, weight in zip(strategies, weights):
        allocation = initial_capital * weight
        print(f"  {strategy}: {weight*100:.1f}% (${allocation:,.2f})")
    print("=" * 60)
    
    # Initialize engines
    db = MarketDataDatabase()
    engines = {}
    all_symbols = set()
    
    for strategy_name, strategy_config in strategy_configs.items():
        engines[strategy_name] = EnhancedSymphonyEngine(strategy_config, db)
        all_symbols.update(strategy_config['universe'])
    
    print(f"📊 Combined universe: {sorted(list(all_symbols))}")
    
    # Get date range (business days)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.bdate_range(start=start_dt, end=end_dt)
    
    # Track portfolio
    portfolio_history = []
    strategy_holdings = {name: {} for name in strategies}  # strategy -> {symbol: shares}
    strategy_cash = {name: initial_capital * weight for name, weight in zip(strategies, weights)}
    
    print(f"\n🔄 Simulating {len(date_range)} trading days...")
    
    for i, date in enumerate(date_range):
        try:
            # Get prices for all symbols
            prices = {}
            for symbol in all_symbols:
                try:
                    data = get_simple_data(symbol, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
                    if not data.empty:
                        prices[symbol] = data['Close'].iloc[-1]
                except:
                    pass
            
            if not prices:
                continue
            
            # Track daily results
            daily_result = {
                'date': date,
                'total_portfolio_value': 0,
                'strategy_values': {},
                'strategy_allocations': {},
                'strategy_returns': {},
                'prices': dict(prices)
            }
            
            # Process each strategy
            for j, (strategy_name, weight) in enumerate(zip(strategies, weights)):
                strategy_capital = initial_capital * weight
                
                try:
                    # Get strategy allocations
                    engine = engines[strategy_name]
                    allocations = engine.calculate_allocations(date)
                    
                    if not allocations:
                        # No allocations - keep current value
                        current_value = strategy_cash[strategy_name]
                        for symbol, shares in strategy_holdings[strategy_name].items():
                            if symbol in prices:
                                current_value += shares * prices[symbol]
                        daily_result['strategy_values'][strategy_name] = current_value
                        daily_result['strategy_allocations'][strategy_name] = {}
                        continue
                    
                    # Calculate current strategy value
                    current_value = strategy_cash[strategy_name]
                    for symbol, shares in strategy_holdings[strategy_name].items():
                        if symbol in prices:
                            current_value += shares * prices[symbol]
                    
                    # Rebalance strategy: sell all current holdings
                    for symbol, shares in strategy_holdings[strategy_name].items():
                        if symbol in prices and shares > 0:
                            strategy_cash[strategy_name] += shares * prices[symbol]
                    
                    strategy_holdings[strategy_name] = {}
                    
                    # Buy new allocations
                    for symbol, alloc_weight in allocations.items():
                        if alloc_weight > 0 and symbol in prices:
                            target_value = current_value * alloc_weight
                            shares = target_value / prices[symbol]
                            strategy_holdings[strategy_name][symbol] = shares
                            strategy_cash[strategy_name] -= target_value
                    
                    # Calculate final strategy value
                    final_strategy_value = strategy_cash[strategy_name]
                    for symbol, shares in strategy_holdings[strategy_name].items():
                        if symbol in prices:
                            final_strategy_value += shares * prices[symbol]
                    
                    daily_result['strategy_values'][strategy_name] = final_strategy_value
                    daily_result['strategy_allocations'][strategy_name] = dict(allocations)
                    daily_result['strategy_returns'][strategy_name] = (final_strategy_value / strategy_capital - 1) * 100
                    
                except Exception as e:
                    # Keep previous value on error
                    current_value = strategy_cash[strategy_name]
                    for symbol, shares in strategy_holdings[strategy_name].items():
                        if symbol in prices:
                            current_value += shares * prices[symbol]
                    daily_result['strategy_values'][strategy_name] = current_value
                    daily_result['strategy_allocations'][strategy_name] = {}
                    daily_result['strategy_returns'][strategy_name] = (current_value / strategy_capital - 1) * 100
            
            # Calculate total portfolio value
            daily_result['total_portfolio_value'] = sum(daily_result['strategy_values'].values())
            daily_result['total_return'] = (daily_result['total_portfolio_value'] / initial_capital - 1) * 100
            
            portfolio_history.append(daily_result)
            
            # Progress update
            if (i + 1) % 50 == 0 or i == len(date_range) - 1:
                progress = (i + 1) / len(date_range) * 100
                total_value = daily_result['total_portfolio_value']
                total_return = daily_result['total_return']
                print(f"📊 Progress: {progress:.1f}% | "
                      f"Date: {date.strftime('%Y-%m-%d')} | "
                      f"Value: ${total_value:,.2f} | "
                      f"Return: {total_return:.2f}%")
            
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
    final_value = df['total_portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    # Daily returns
    df['daily_return'] = df['total_portfolio_value'].pct_change()
    daily_returns = df['daily_return'].dropna()
    
    if len(daily_returns) > 1:
        annualized_return = (final_value / initial_capital) ** (252 / len(df)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        running_max = df['total_portfolio_value'].expanding().max()
        drawdown = (df['total_portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (daily_returns > 0).mean() * 100
        
        print(f"\n📊 MULTI-STRATEGY PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Initial Capital:     ${initial_capital:,.2f}")
        print(f"Final Value:         ${final_value:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Annualized Return:   {annualized_return*100:.2f}%")
        print(f"Volatility:          {volatility*100:.2f}%")
        print(f"Sharpe Ratio:        {sharpe:.2f}")
        print(f"Max Drawdown:        {max_drawdown*100:.2f}%")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Total Days:          {len(df)}")
        
        # Individual strategy performance
        print(f"\n🎯 INDIVIDUAL STRATEGY PERFORMANCE:")
        for strategy_name, weight in zip(strategies, weights):
            if strategy_name in df['strategy_values'].iloc[-1]:
                strategy_capital = initial_capital * weight
                strategy_final = df['strategy_values'].iloc[-1][strategy_name]
                strategy_return = (strategy_final / strategy_capital - 1) * 100
                print(f"  {strategy_name}:")
                print(f"    Weight: {weight*100:.1f}% (${strategy_capital:,.2f})")
                print(f"    Return: {strategy_return:+.2f}%")
                print(f"    Final Value: ${strategy_final:,.2f}")
    
    # Save results
    strategy_names = "_".join(strategies)
    filename = f"multi_strategy_{strategy_names}_{start_date}_{end_date}.csv"
    
    # Flatten nested data for CSV
    export_df = df[['total_portfolio_value', 'total_return']].copy()
    
    # Add strategy columns
    for strategy_name in strategies:
        export_df[f'{strategy_name}_value'] = df['strategy_values'].apply(lambda x: x.get(strategy_name, 0))
        export_df[f'{strategy_name}_return'] = df['strategy_returns'].apply(lambda x: x.get(strategy_name, 0))
    
    export_df.to_csv(filename)
    print(f"\n💾 Results saved to: {filename}")
    
    # Create chart
    create_multi_strategy_chart(df, strategies, start_date, end_date, initial_capital)
    
    print(f"\n✅ MULTI-STRATEGY TEST COMPLETED!")
    return df

def create_multi_strategy_chart(df, strategies, start_date, end_date, initial_capital):
    """Create multi-strategy performance chart"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Portfolio value
    ax1.plot(df.index, df['total_portfolio_value'], linewidth=3, color='blue')
    ax1.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title(f'Multi-Strategy Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Individual strategy values
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    for i, strategy in enumerate(strategies):
        values = [row.get(strategy, 0) for row in df['strategy_values']]
        ax2.plot(df.index, values, linewidth=2, color=colors[i], label=strategy, alpha=0.8)
    
    ax2.set_title('Individual Strategy Performance')
    ax2.set_ylabel('Strategy Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative returns
    cumulative_returns = (df['total_portfolio_value'] / initial_capital - 1) * 100
    ax3.plot(df.index, cumulative_returns, linewidth=3, color='green')
    ax3.set_title('Cumulative Returns (%)')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Drawdown
    running_max = df['total_portfolio_value'].expanding().max()
    drawdown = (df['total_portfolio_value'] - running_max) / running_max * 100
    ax4.fill_between(df.index, drawdown, 0, alpha=0.6, color='red')
    ax4.set_title('Portfolio Drawdown (%)')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    strategy_names = "_".join(strategies)
    chart_filename = f"multi_strategy_chart_{strategy_names}_{start_date}_{end_date}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Chart saved to: {chart_filename}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Strategy Tester')
    parser.add_argument('--strategies', nargs='+', required=True, help='List of strategy names')
    parser.add_argument('--weights', nargs='+', type=float, help='Strategy weights (must sum to 1.0)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Default to equal weights if not provided
    if not args.weights:
        args.weights = [1.0 / len(args.strategies)] * len(args.strategies)
    
    test_multi_strategy(args.strategies, args.weights, args.start, args.end, args.capital)

if __name__ == "__main__":
    main()