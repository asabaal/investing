#!/usr/bin/env python3
"""
Universal Paper Trade Logger
Reads trade data from pending_trade.json and adds it to the journal
"""

import json
import datetime
from paper_trading_journal import PaperTradingJournal

def load_pending_trade():
    """Load trade data from pending_trade.json"""
    try:
        with open('pending_trade.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ pending_trade.json not found")
        return None
    except json.JSONDecodeError:
        print("❌ Invalid JSON in pending_trade.json")
        return None

def create_trade_structure(trade_data):
    """Convert pending trade data to full journal structure"""
    symbol = trade_data['symbol'].upper()
    is_short = trade_data['order_type'].lower() == 'sell'
    
    # Calculate risk/reward analysis
    entry_price = trade_data['limit_price']
    stop_loss = trade_data['stop_loss']
    take_profit = trade_data['take_profit']
    quantity = trade_data['quantity']
    
    if is_short:
        risk_per_share = stop_loss - entry_price
        reward_per_share = entry_price - take_profit
    else:
        risk_per_share = entry_price - stop_loss
        reward_per_share = take_profit - entry_price
    
    total_risk = abs(risk_per_share * quantity)
    total_reward = abs(reward_per_share * quantity)
    risk_reward_ratio = total_reward / total_risk if total_risk > 0 else 0
    risk_percentage = (total_risk / trade_data['position_value']) * 100
    
    # Generate trade IDs (assuming they're sequential)
    base_id = trade_data['trade_id']
    stop_loss_id = str(int(base_id) + 1)
    take_profit_id = str(int(base_id) + 2)
    
    # Create full trade structure
    trade = {
        "strategy_name": f"{symbol} {'Short' if is_short else 'Long'} Position",
        "symbol": f"NYSE:{symbol}",
        "trade_id": base_id,
        "setup_date": trade_data['entry_date'],
        "group_number": len(PaperTradingJournal().data["paper_trades"]) + 1,
        "group_name": f"Paper_Trade_Group_{len(PaperTradingJournal().data['paper_trades']) + 1:02d}",
        "entry_order": {
            "symbol": symbol,
            "order_type": trade_data['order_type'],
            "execution_type": trade_data['execution_type'],
            "quantity": quantity,
            "limit_price": entry_price,
            "target_price": take_profit,
            "stop_loss": stop_loss,
            "status": "working",
            "entry_date": trade_data['entry_date'],
            "trade_id": base_id,
            "expiration_date": trade_data['expiration_date'],
            "risk_reward_ratio": "1:1",
            "position_value": trade_data['position_value'],
            "currency": "USD"
        },
        "stop_loss_order": {
            "symbol": symbol,
            "order_type": "Buy" if is_short else "Sell",
            "execution_type": "Stop Loss",
            "quantity": quantity,
            "trigger_price": stop_loss,
            "status": "inactive",
            "entry_date": trade_data['entry_date'],
            "trade_id": stop_loss_id,
            "expiration_date": trade_data.get('stop_expiration_date', trade_data['expiration_date'])
        },
        "take_profit_order": {
            "symbol": symbol,
            "order_type": "Buy" if is_short else "Sell",
            "execution_type": "Take Profit",
            "quantity": quantity,
            "limit_price": take_profit,
            "status": "inactive",
            "entry_date": trade_data['entry_date'],
            "trade_id": take_profit_id,
            "expiration_date": trade_data.get('tp_expiration_date', trade_data['expiration_date'])
        },
        "trade_analysis": {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_per_share": abs(risk_per_share),
            "reward_per_share": abs(reward_per_share),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "total_risk": total_risk,
            "total_reward": total_reward,
            "breakeven_price": entry_price,
            "max_loss": trade_data['position_value'] - total_reward if is_short else trade_data['position_value'] - total_risk,
            "max_gain": trade_data['position_value'] + total_reward if not is_short else trade_data['position_value'] - total_reward,
            "risk_percentage": round(risk_percentage, 2)
        },
        "trade_rationale": {
            "sector": "TBD",
            "industry": "TBD", 
            "market_cap": "TBD",
            "trading_score": "TBD",
            "watchlist_rank": "TBD",
            "correlation_group": len(PaperTradingJournal().data["paper_trades"]) + 1,
            "max_group_correlation": "TBD",
            "selection_reason": trade_data.get('notes', f"{'Short' if is_short else 'Long'} position setup")
        },
        "current_status": {
            "overall_status": "pending_entry",
            "entry_filled": False,
            "stop_loss_active": False,
            "take_profit_active": False,
            "last_update": trade_data['entry_date']
        }
    }
    
    return trade

def add_paper_trade():
    """Main function to add paper trade from pending_trade.json"""
    # Load pending trade data
    trade_data = load_pending_trade()
    if not trade_data:
        return
    
    # Validate required fields
    required_fields = ['symbol', 'order_type', 'execution_type', 'quantity', 'limit_price', 'stop_loss', 'take_profit', 'trade_id', 'entry_date', 'position_value']
    for field in required_fields:
        if not trade_data.get(field):
            print(f"❌ Missing required field: {field}")
            return
    
    # Create trade structure
    trade = create_trade_structure(trade_data)
    
    # Add to journal
    journal = PaperTradingJournal()
    journal.add_paper_trade(trade)
    
    # Generate updated report
    journal.generate_report()
    
    # Print confirmation
    is_short = trade_data['order_type'].lower() == 'sell'
    print(f"✅ {trade['symbol']} {'Short' if is_short else 'Long'} Position successfully added to journal!")
    print(f"   Entry: {trade_data['order_type']} ${trade_data['limit_price']:.2f}")
    print(f"   Stop Loss: ${trade_data['stop_loss']:.2f}")
    print(f"   Take Profit: ${trade_data['take_profit']:.2f}")
    print(f"   Position Value: ${trade_data['position_value']:.2f}")
    print(f"   Risk:Reward: {trade['trade_analysis']['risk_reward_ratio']:.1f}:1")
    
    # Clear the pending trade file
    clear_trade = {
        "symbol": "",
        "order_type": "",
        "execution_type": "",
        "quantity": 0,
        "limit_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "trade_id": "",
        "entry_date": "",
        "expiration_date": "",
        "position_value": 0.0,
        "notes": ""
    }
    
    with open('pending_trade.json', 'w') as f:
        json.dump(clear_trade, f, indent=2)
    
    print("✅ pending_trade.json cleared for next trade")
    
    return journal

if __name__ == "__main__":
    add_paper_trade()