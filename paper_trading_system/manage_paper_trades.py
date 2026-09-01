#!/usr/bin/env python3
"""
Paper Trading Management CLI
Simple command-line interface for managing paper trades
"""

import sys
from paper_trading_journal import PaperTradingJournal

def main():
    journal = PaperTradingJournal()
    
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "summary":
        journal.print_summary()
    
    elif command == "report":
        output_file = sys.argv[2] if len(sys.argv) > 2 else "paper_trading_report.html"
        journal.generate_report(output_file)
    
    elif command == "fill":
        if len(sys.argv) < 4:
            print("Usage: python manage_paper_trades.py fill <trade_id> <fill_price>")
            return
        trade_id = sys.argv[2]
        fill_price = float(sys.argv[3])
        journal.fill_entry_order(trade_id, fill_price)
    
    elif command == "close":
        if len(sys.argv) < 5:
            print("Usage: python manage_paper_trades.py close <trade_id> <exit_price> <reason>")
            return
        trade_id = sys.argv[2]
        exit_price = float(sys.argv[3])
        exit_reason = sys.argv[4]
        journal.close_trade(trade_id, exit_price, exit_reason)
    
    elif command == "active":
        active_trades = journal.get_active_trades()
        print(f"\n🎯 ACTIVE PAPER TRADES ({len(active_trades)})")
        print("=" * 60)
        
        for trade in active_trades:
            status = trade["current_status"]["overall_status"]
            entry_price = trade["entry_order"].get("fill_price", trade["entry_order"]["limit_price"])
            
            print(f"🔸 {trade['symbol']} - {trade['strategy_name']}")
            print(f"   ID: {trade['trade_id']}")
            print(f"   Status: {status.replace('_', ' ').title()}")
            print(f"   Entry: ${entry_price:.2f}")
            print(f"   Stop: ${trade['stop_loss_order']['trigger_price']:.2f}")
            print(f"   Target: ${trade['take_profit_order']['limit_price']:.2f}")
            print(f"   Risk:Reward: {trade['trade_analysis']['risk_reward_ratio']:.2f}:1")
            print()
    
    elif command == "trade":
        if len(sys.argv) < 3:
            print("Usage: python manage_paper_trades.py trade <trade_id>")
            return
        trade_id = sys.argv[2]
        trade = journal.get_trade_by_id(trade_id)
        
        if trade:
            print(f"\n📊 TRADE DETAILS: {trade['symbol']}")
            print("=" * 50)
            print(f"Strategy: {trade['strategy_name']}")
            print(f"Trade ID: {trade['trade_id']}")
            print(f"Setup Date: {trade['setup_date']}")
            print(f"Group: {trade['group_number']} ({trade['group_name']})")
            print(f"Status: {trade['current_status']['overall_status'].replace('_', ' ').title()}")
            print()
            print("📈 ENTRY ORDER:")
            entry = trade['entry_order']
            print(f"   Type: {entry['order_type']} {entry['execution_type']}")
            print(f"   Quantity: {entry['quantity']} shares")
            print(f"   Price: ${entry['limit_price']:.2f}")
            print(f"   Value: ${entry['position_value']:.2f}")
            print(f"   Status: {entry['status'].title()}")
            
            print()
            print("🔻 STOP LOSS:")
            stop = trade['stop_loss_order']
            print(f"   Trigger: ${stop['trigger_price']:.2f}")
            print(f"   Status: {stop['status'].title()}")
            
            print()
            print("🎯 TAKE PROFIT:")
            target = trade['take_profit_order']
            print(f"   Target: ${target['limit_price']:.2f}")
            print(f"   Status: {target['status'].title()}")
            
            print()
            print("💰 RISK/REWARD ANALYSIS:")
            analysis = trade['trade_analysis']
            print(f"   Risk per share: ${analysis['risk_per_share']:.2f}")
            print(f"   Reward per share: ${analysis['reward_per_share']:.2f}")
            print(f"   Total risk: ${analysis['total_risk']:.2f}")
            print(f"   Total reward potential: ${analysis['total_reward']:.2f}")
            print(f"   Risk:Reward ratio: {analysis['risk_reward_ratio']:.2f}:1")
            print(f"   Risk %: {analysis['risk_percentage']:.2f}%")
            
            if "exit_details" in trade:
                print()
                print("✅ TRADE CLOSED:")
                exit_info = trade['exit_details']
                print(f"   Exit Price: ${exit_info['exit_price']:.2f}")
                print(f"   Exit Date: {exit_info['exit_date']}")
                print(f"   Exit Reason: {exit_info['exit_reason']}")
                print(f"   P&L: ${exit_info['total_pnl']:.2f} ({exit_info['pnl_percentage']:.2f}%)")
                print(f"   Return on Allocated Capital: {exit_info['return_on_allocated_capital']:.2f}%")
            
        else:
            print(f"❌ Trade {trade_id} not found")
    
    else:
        print_help()

def print_help():
    print("""
🎯 PAPER TRADING MANAGEMENT CLI

Commands:
  summary                           - Show trading summary
  report [filename]                 - Generate HTML report  
  active                           - List active trades
  trade <trade_id>                 - Show detailed trade info
  fill <trade_id> <fill_price>     - Mark entry order as filled
  close <trade_id> <exit_price> <reason> - Close a trade

Examples:
  python manage_paper_trades.py summary
  python manage_paper_trades.py active  
  python manage_paper_trades.py trade 2210069095
  python manage_paper_trades.py fill 2210069095 35.90
  python manage_paper_trades.py close 2210069095 37.30 "take_profit"
  python manage_paper_trades.py report my_paper_trades.html
""")

if __name__ == "__main__":
    main()