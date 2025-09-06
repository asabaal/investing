#!/usr/bin/env python3
"""
Trade Likelihood Backtest Runner

Main script to run backtests on your paper trades using your existing database
and generate comprehensive dark mode reports.

Usage:
    python run_backtest.py --symbol DPZ --entry 464.80 --stop 468.46 --target 441.47 --date 2025-09-01
    
    or for interactive mode:
    python run_backtest.py --interactive
"""

import argparse
import sys
from pathlib import Path
import os

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_system import PaperTradeBacktester, PaperTrade
from report_generator import BacktestReportGenerator
from datetime import datetime


def run_single_trade_backtest(symbol: str, 
                             entry_price: float,
                             stop_loss: float,
                             target_price: float, 
                             trade_date: str,
                             generate_report: bool = True) -> str:
    """
    Run a complete backtest for a single trade and generate report
    
    Args:
        symbol: Stock symbol
        entry_price: Entry price
        stop_loss: Stop loss price  
        target_price: Target price
        trade_date: Trade date in YYYY-MM-DD format
        generate_report: Whether to generate HTML report
        
    Returns:
        Path to generated report or empty string if no report
    """
    
    print(f"🚀 Starting backtest for {symbol} trade")
    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Stop: ${stop_loss:.2f}")  
    print(f"   Target: ${target_price:.2f}")
    print(f"   Date: {trade_date}")
    
    # Initialize backtester
    backtester = PaperTradeBacktester()
    
    # Run the backtest
    try:
        result = backtester.run_single_trade_backtest(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            trade_date=trade_date,
            direction='auto'  # Auto-detect long vs short
        )
        
        print(f"\n✅ Backtest completed successfully!")
        
        # Print summary to console
        print_backtest_summary(result)
        
        # Generate report if requested
        report_path = ""
        if generate_report:
            print(f"\n📄 Generating HTML report...")
            
            report_generator = BacktestReportGenerator()
            report_path = report_generator.generate_single_trade_report(result)
            
            print(f"✅ Report saved to: {report_path}")
            
            # Try to open in browser
            try:
                import webbrowser
                full_path = os.path.abspath(report_path)
                webbrowser.open(f'file://{full_path}')
                print(f"🌐 Opening report in browser...")
            except:
                print(f"💡 Open this file in your browser: {os.path.abspath(report_path)}")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def print_backtest_summary(result):
    """Print a nice console summary of the backtest results"""
    
    trade = result.trade
    print(f"\n📊 BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Trade: {trade.symbol} {trade.direction.upper()}")
    print(f"Entry: ${trade.entry_price:.2f}")
    print(f"Stop: ${trade.stop_loss:.2f}")
    print(f"Target: ${trade.target_price:.2f}")
    print(f"R:R Ratio: {trade.get_risk_reward_ratio():.2f}:1")
    
    # Actual outcome
    if result.actual_outcome and result.actual_outcome != 'unknown':
        print(f"\n🎯 ACTUAL OUTCOME: {result.actual_outcome.upper()}")
        if result.actual_exit_price:
            print(f"Exit Price: ${result.actual_exit_price:.2f}")
        if result.days_to_exit is not None:
            print(f"Days to Exit: {result.days_to_exit}")
    else:
        print(f"\n⚠️  Actual outcome unknown (insufficient future data)")
    
    # Model predictions
    if result.model_predictions:
        print(f"\n🤖 MODEL PREDICTIONS:")
        print("-" * 30)
        
        for window, analysis in result.model_predictions.items():
            attractive = "✅ YES" if analysis.is_attractive_setup() else "❌ NO"
            return_rate = analysis.return_rate_total or 0
            entry_prob = analysis.get_combined_entry_probability() or 0
            
            print(f"{window:>8}: Return Rate {return_rate:7.4f}/day | "
                  f"Entry Prob {entry_prob:5.1%} | "
                  f"Attractive {attractive}")
        
        # Calculate consistency
        attractive_count = sum(1 for a in result.model_predictions.values() if a.is_attractive_setup())
        consistency = attractive_count / len(result.model_predictions) * 100
        print(f"\nModel Consistency: {consistency:.0f}% of windows predict attractive setup")
    
    else:
        print(f"\n❌ No model predictions (insufficient historical data)")


def run_interactive_mode():
    """Interactive mode for entering trade details"""
    
    print("🎯 Interactive Trade Backtest Mode")
    print("=" * 40)
    
    try:
        symbol = input("Enter symbol (e.g., DPZ): ").upper().strip()
        entry_price = float(input("Enter entry price: "))
        stop_loss = float(input("Enter stop loss price: "))
        target_price = float(input("Enter target price: "))
        
        # Auto-detect direction for validation
        if target_price > entry_price:
            direction = "LONG"
        else:
            direction = "SHORT"
        
        print(f"\nDetected direction: {direction}")
        print(f"Risk-Reward Ratio: {abs(target_price - entry_price) / abs(entry_price - stop_loss):.2f}:1")
        
        # Get trade date
        date_input = input("Enter trade date (YYYY-MM-DD) or press Enter for 2025-09-01: ").strip()
        if not date_input:
            date_input = "2025-09-01"
        
        # Validate date format
        datetime.strptime(date_input, '%Y-%m-%d')
        
        # Confirm details
        print(f"\n📋 Trade Details:")
        print(f"Symbol: {symbol}")
        print(f"Entry: ${entry_price:.2f}")
        print(f"Stop: ${stop_loss:.2f}")
        print(f"Target: ${target_price:.2f}")
        print(f"Date: {date_input}")
        print(f"Direction: {direction}")
        
        confirm = input("\nProceed with backtest? (y/n): ").lower().strip()
        
        if confirm in ['y', 'yes']:
            return run_single_trade_backtest(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                trade_date=date_input,
                generate_report=True
            )
        else:
            print("Cancelled.")
            return ""
            
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return ""
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Run Trade Likelihood Backtest on Paper Trades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest the DPZ trade
  python run_backtest.py --symbol DPZ --entry 464.80 --stop 468.46 --target 441.47 --date 2025-09-01
  
  # Interactive mode
  python run_backtest.py --interactive
  
  # Quick DPZ test (uses default parameters)
  python run_backtest.py --dpz
        """
    )
    
    parser.add_argument('--symbol', type=str, help='Stock symbol (e.g., DPZ)')
    parser.add_argument('--entry', type=float, help='Entry price')
    parser.add_argument('--stop', type=float, help='Stop loss price')
    parser.add_argument('--target', type=float, help='Target price')
    parser.add_argument('--date', type=str, help='Trade date (YYYY-MM-DD)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--dpz', action='store_true', help='Quick test with DPZ trade parameters')
    parser.add_argument('--no-report', action='store_true', help='Skip HTML report generation')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.interactive:
        run_interactive_mode()
        return
    
    if args.dpz:
        # Quick DPZ test with your parameters
        print("🍕 Running DPZ trade backtest with your parameters...")
        run_single_trade_backtest(
            symbol='DPZ',
            entry_price=464.80,
            stop_loss=468.46,
            target_price=441.47,
            trade_date='2025-09-01',
            generate_report=not args.no_report
        )
        return
    
    # Validate required arguments for manual mode
    required_args = ['symbol', 'entry', 'stop', 'target', 'date']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args:
        print(f"❌ Missing required arguments: {', '.join(missing_args)}")
        print("Use --interactive for guided input or --dpz for quick test")
        parser.print_help()
        return
    
    # Run backtest with provided arguments
    run_single_trade_backtest(
        symbol=args.symbol,
        entry_price=args.entry,
        stop_loss=args.stop,
        target_price=args.target,
        trade_date=args.date,
        generate_report=not args.no_report
    )


if __name__ == "__main__":
    main()