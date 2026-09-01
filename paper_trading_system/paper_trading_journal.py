#!/usr/bin/env python3
"""
Paper Trading Journal System
Tracks paper trades, calculates metrics, and generates reports for practice trading
"""

import json
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

@dataclass
class PaperTrade:
    strategy_name: str
    symbol: str
    trade_id: str
    setup_date: str
    group_number: int
    entry_order: Dict
    stop_loss_order: Dict
    take_profit_order: Dict
    trade_analysis: Dict
    trade_rationale: Dict
    current_status: Dict

class PaperTradingJournal:
    def __init__(self, journal_file: str = "paper_trading_journal.json"):
        self.journal_file = journal_file
        self.data = self._load_journal()
    
    def _load_journal(self) -> Dict:
        """Load journal data from JSON file"""
        if os.path.exists(self.journal_file):
            with open(self.journal_file, 'r') as f:
                return json.load(f)
        else:
            return {"paper_trades": [], "paper_trading_summary": {}}
    
    def _save_journal(self):
        """Save journal data to JSON file"""
        with open(self.journal_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_paper_trade(self, trade_data: Dict):
        """Add a new paper trade to the journal"""
        self.data["paper_trades"].append(trade_data)
        self._update_summary()
        self._save_journal()
        print(f"✅ Added paper trade: {trade_data['symbol']} - {trade_data['strategy_name']}")
    
    def update_trade_status(self, trade_id: str, status_update: Dict):
        """Update the status of a specific trade"""
        for trade in self.data["paper_trades"]:
            if trade["trade_id"] == trade_id:
                trade["current_status"].update(status_update)
                trade["current_status"]["last_update"] = datetime.datetime.now().isoformat()
                self._save_journal()
                print(f"✅ Updated trade {trade_id} status")
                return
        print(f"⚠️ Trade {trade_id} not found")
    
    def fill_entry_order(self, trade_id: str, fill_price: float, fill_date: str = None):
        """Mark entry order as filled"""
        if fill_date is None:
            fill_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for trade in self.data["paper_trades"]:
            if trade["trade_id"] == trade_id:
                trade["entry_order"]["status"] = "filled"
                trade["entry_order"]["fill_price"] = fill_price
                trade["entry_order"]["fill_date"] = fill_date
                
                # Activate stop loss and take profit
                trade["stop_loss_order"]["status"] = "active"
                trade["take_profit_order"]["status"] = "active"
                
                # Update overall status
                trade["current_status"]["overall_status"] = "position_open"
                trade["current_status"]["entry_filled"] = True
                trade["current_status"]["stop_loss_active"] = True
                trade["current_status"]["take_profit_active"] = True
                
                self._save_journal()
                print(f"✅ Entry filled for {trade['symbol']} at ${fill_price}")
                return
        print(f"⚠️ Trade {trade_id} not found")
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str, exit_date: str = None):
        """Close a trade (either stop loss or take profit hit)"""
        if exit_date is None:
            exit_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for trade in self.data["paper_trades"]:
            if trade["trade_id"] == trade_id:
                # Calculate P&L - handle different data structures
                entry_price = trade["entry_order"].get("fill_price") or trade["entry_order"].get("entry_price") or trade["entry_order"].get("limit_price")
                quantity = trade["entry_order"]["quantity"]
                
                # For short positions, reverse the P&L calculation
                is_short = trade["entry_order"].get("order_type") == "Sell" or trade["entry_order"].get("side") == "sell"
                
                if is_short:
                    # For short positions: profit when exit price < entry price
                    pnl_per_share = entry_price - exit_price
                else:
                    # For long positions: profit when exit price > entry price  
                    pnl_per_share = exit_price - entry_price
                    
                total_pnl = pnl_per_share * quantity
                pnl_percentage = (pnl_per_share / entry_price) * 100
                
                # Calculate return on allocated capital for this trade
                position_value = float(trade["entry_order"]["position_value"])
                roac_trade = (total_pnl / position_value) * 100
                
                # Update trade record
                trade["exit_details"] = {
                    "exit_price": exit_price,
                    "exit_date": exit_date,
                    "exit_reason": exit_reason,
                    "pnl_per_share": pnl_per_share,
                    "total_pnl": total_pnl,
                    "pnl_percentage": pnl_percentage,
                    "return_on_allocated_capital": roac_trade,
                    "trade_duration": None  # Could calculate if needed
                }
                
                # Update status
                trade["current_status"]["overall_status"] = "closed"
                trade["current_status"]["stop_loss_active"] = False
                trade["current_status"]["take_profit_active"] = False
                
                self._update_summary()
                self._save_journal()
                print(f"✅ Trade closed: {trade['symbol']} at ${exit_price}")
                print(f"   P&L: ${total_pnl:.2f} ({pnl_percentage:.2f}%) - {exit_reason}")
                return
                
        print(f"⚠️ Trade {trade_id} not found")
    
    def _update_summary(self):
        """Update the trading summary statistics"""
        trades = self.data["paper_trades"]
        
        total_strategies = len(trades)
        active_positions = sum(1 for t in trades if t["current_status"]["overall_status"] == "position_open")
        pending_entries = sum(1 for t in trades if t["current_status"]["overall_status"] == "pending_entry")
        closed_trades = sum(1 for t in trades if t["current_status"]["overall_status"] == "closed")
        
        total_capital = sum(float(t["entry_order"]["position_value"]) for t in trades)
        total_risk = sum(float(t["trade_analysis"]["total_risk"]) for t in trades)
        
        avg_risk_reward = 0
        if trades:
            avg_risk_reward = sum(float(t["trade_analysis"]["risk_reward_ratio"]) for t in trades) / len(trades)
        
        # Calculate realized P&L for closed trades
        realized_pnl = 0
        winning_trades = 0
        losing_trades = 0
        total_closed_capital = 0
        
        for trade in trades:
            if "exit_details" in trade:
                pnl = trade["exit_details"]["total_pnl"]
                realized_pnl += pnl
                total_closed_capital += float(trade["entry_order"]["position_value"])
                
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        win_rate = 0
        if closed_trades > 0:
            win_rate = (winning_trades / closed_trades) * 100
        
        # Calculate Realized ROAC and Projected ROAC
        realized_roac = 0
        if total_capital > 0:
            realized_roac = (realized_pnl / total_capital) * 100
        
        # Calculate Projected ROAC based on probability analysis
        projected_roac = 0
        if total_capital > 0:
            # Known win rates from probability analysis
            # Note: BE uses generalized entry strategy (price invariance) showing 35-45% win rates
            trade_projections = {
                'CHWY': {'win_rate': 0.28, 'reward_pct': 3.88, 'risk_pct': 0.73, 'strategy': 'standard'},
                'BE': {'win_rate': 0.40, 'reward_pct': 5.83, 'risk_pct': 1.36, 'strategy': 'generalized_entry'}, 
                'PFE': {'win_rate': 0.49, 'reward_pct': 3.08, 'risk_pct': 0.46, 'strategy': 'standard'}
            }
            
            total_expected_return = 0
            for trade in trades:
                symbol = trade['entry_order']['symbol']
                capital = trade['entry_order']['position_value']
                
                if symbol in trade_projections:
                    proj = trade_projections[symbol]
                    expected_return_pct = (proj['win_rate'] * proj['reward_pct']) - ((1 - proj['win_rate']) * proj['risk_pct'])
                    expected_return_dollars = capital * (expected_return_pct / 100)
                    total_expected_return += expected_return_dollars
            
            projected_roac = (total_expected_return / total_capital) * 100
        
        self.data["paper_trading_summary"] = {
            "total_strategies": total_strategies,
            "active_positions": active_positions,
            "pending_entries": pending_entries,
            "closed_trades": closed_trades,
            "total_capital_allocated": round(total_capital, 2),
            "total_closed_capital": round(total_closed_capital, 2),
            "total_risk_exposure": round(total_risk, 2),
            "average_risk_reward": round(avg_risk_reward, 2),
            "realized_pnl": round(realized_pnl, 2),
            "realized_roac": round(realized_roac, 2),
            "projected_roac": round(projected_roac, 2),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def load_watchlist_symbols(self, watchlist_file: str = "../complete_watchlist_trading_groups.csv") -> set:
        """Load watchlist symbols from CSV file"""
        import csv
        import os
        
        watchlist_symbols = set()
        full_path = os.path.join(os.path.dirname(__file__), watchlist_file)
        
        try:
            with open(full_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    watchlist_symbols.add(row['ticker'])
        except FileNotFoundError:
            print(f"⚠️ Watchlist file not found: {full_path}")
        
        return watchlist_symbols
    
    def _get_clean_symbol(self, symbol: str) -> str:
        """Remove exchange prefix from symbol (e.g., NYSE:CHWY -> CHWY)"""
        if ':' in symbol:
            return symbol.split(':')[1]
        return symbol
    
    def generate_report(self, output_file: str = "paper_trading_report.html"):
        """Generate an HTML report of paper trading performance with watchlist filtering options"""
        self._update_summary()
        
        trades = self.data["paper_trades"]
        summary = self.data["paper_trading_summary"]
        
        # Create visualizations
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Trade Status Distribution',
                'Risk vs Reward Analysis', 
                'P&L by Trade (Closed Only)',
                'Return on Allocated Capital',
                'Capital Allocation by Symbol',
                'Cumulative ROAC Over Time'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Trade Status Distribution
        status_counts = {}
        for trade in trades:
            status = trade["current_status"]["overall_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                name="Trade Status",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Risk vs Reward Scatter
        risk_values = []
        reward_values = []
        symbols = []
        colors = []
        
        for trade in trades:
            risk_values.append(trade["trade_analysis"]["total_risk"])
            reward_values.append(trade["trade_analysis"]["total_reward"])
            symbols.append(trade["symbol"])
            
            # Color by status
            if trade["current_status"]["overall_status"] == "closed":
                if "exit_details" in trade and trade["exit_details"]["total_pnl"] > 0:
                    colors.append("green")
                elif "exit_details" in trade:
                    colors.append("red") 
                else:
                    colors.append("gray")
            elif trade["current_status"]["overall_status"] == "position_open":
                colors.append("blue")
            else:
                colors.append("orange")
        
        fig.add_trace(
            go.Scatter(
                x=risk_values,
                y=reward_values,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                marker=dict(color=colors, size=10),
                name="Trades"
            ),
            row=1, col=2
        )
        
        # 3. P&L by Trade (Closed Only)
        closed_trades_pnl = []
        closed_symbols = []
        pnl_colors = []
        
        for trade in trades:
            if "exit_details" in trade:
                pnl = trade["exit_details"]["total_pnl"]
                closed_trades_pnl.append(pnl)
                closed_symbols.append(trade["symbol"])
                pnl_colors.append("green" if pnl > 0 else "red")
        
        if closed_trades_pnl:
            fig.add_trace(
                go.Bar(
                    x=closed_symbols,
                    y=closed_trades_pnl,
                    name="P&L",
                    marker_color=pnl_colors
                ),
                row=2, col=1
            )
        
        # 4. Return on Allocated Capital (Closed Trades)
        closed_roac = []
        closed_symbols_roac = []
        roac_colors = []
        
        for trade in trades:
            if "exit_details" in trade:
                roac = trade["exit_details"]["return_on_allocated_capital"]
                closed_roac.append(roac)
                closed_symbols_roac.append(trade["symbol"])
                roac_colors.append("green" if roac > 0 else "red")
        
        if closed_roac:
            fig.add_trace(
                go.Bar(
                    x=closed_symbols_roac,
                    y=closed_roac,
                    name="ROAC %",
                    marker_color=roac_colors
                ),
                row=2, col=2
            )
        
        # 5. Capital Allocation
        symbol_allocation = {}
        for trade in trades:
            symbol = trade["symbol"]
            allocation = trade["entry_order"]["position_value"]
            symbol_allocation[symbol] = symbol_allocation.get(symbol, 0) + allocation
        
        fig.add_trace(
            go.Pie(
                labels=list(symbol_allocation.keys()),
                values=list(symbol_allocation.values()),
                name="Capital",
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Cumulative ROAC Over Time
        closed_trades_with_dates = []
        for trade in trades:
            if "exit_details" in trade:
                closed_trades_with_dates.append({
                    'symbol': trade['symbol'],
                    'exit_date': trade['exit_details']['exit_date'],
                    'roac': trade['exit_details']['return_on_allocated_capital'],
                    'pnl': trade['exit_details']['total_pnl']
                })
        
        if closed_trades_with_dates:
            # Sort by exit date
            closed_trades_with_dates.sort(key=lambda x: x['exit_date'])
            
            # Calculate cumulative ROAC
            cumulative_pnl = 0
            cumulative_capital = 0
            cumulative_roac = []
            dates = []
            
            for i, trade in enumerate(closed_trades_with_dates):
                cumulative_pnl += trade['pnl']
                # Approximate cumulative capital (would need better tracking in practice)
                cumulative_capital += 100  # Simplified assumption
                
                if cumulative_capital > 0:
                    cum_roac = (cumulative_pnl / cumulative_capital) * 100
                else:
                    cum_roac = 0
                    
                cumulative_roac.append(cum_roac)
                dates.append(trade['exit_date'])
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=cumulative_roac,
                    mode='lines+markers',
                    name="Cumulative ROAC",
                    line=dict(color='#4CAF50'),
                    marker=dict(size=8)
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Paper Trading Journal Report",
            title_font_size=24,
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white')
        )
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paper Trading Journal Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #1e1e1e; color: white; }}
        .summary {{ background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .neutral {{ color: #FFC107; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #2d2d2d; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #555; }}
        th {{ background-color: #333; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .filter-btn {{ 
            background-color: #333; 
            color: white; 
            border: 1px solid #555; 
            padding: 8px 16px; 
            margin-right: 10px; 
            cursor: pointer; 
            border-radius: 4px;
        }}
        .filter-btn.active {{ 
            background-color: #4CAF50; 
            border-color: #4CAF50; 
        }}
        .filter-btn:hover {{ 
            background-color: #555; 
        }}
        .trade-row {{ display: table-row; }}
        .trade-row.hidden {{ display: none; }}
    </style>
</head>
<body>
    <h1>📊 Paper Trading Journal Report</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="metric">
            <div>Total Strategies</div>
            <div class="metric-value">{summary.get('total_strategies', 0)}</div>
        </div>
        <div class="metric">
            <div>Active Positions</div>
            <div class="metric-value neutral">{summary.get('active_positions', 0)}</div>
        </div>
        <div class="metric">
            <div>Pending Entries</div>
            <div class="metric-value neutral">{summary.get('pending_entries', 0)}</div>
        </div>
        <div class="metric">
            <div>Closed Trades</div>
            <div class="metric-value">{summary.get('closed_trades', 0)}</div>
        </div>
        <div class="metric">
            <div>Capital Allocated</div>
            <div class="metric-value">${summary.get('total_capital_allocated', 0):.2f}</div>
        </div>
        <div class="metric">
            <div>Total Risk</div>
            <div class="metric-value negative">${summary.get('total_risk_exposure', 0):.2f}</div>
        </div>
        <div class="metric">
            <div>Realized P&L</div>
            <div class="metric-value {'positive' if summary.get('realized_pnl', 0) >= 0 else 'negative'}">${summary.get('realized_pnl', 0):.2f}</div>
        </div>
        <div class="metric">
            <div>Win Rate</div>
            <div class="metric-value">{summary.get('win_rate', 0):.1f}%</div>
        </div>
        <div class="metric">
            <div>Avg Risk:Reward</div>
            <div class="metric-value">{summary.get('average_risk_reward', 0):.2f}:1</div>
        </div>
        <div class="metric">
            <div>Realized ROAC</div>
            <div class="metric-value {'positive' if summary.get('realized_roac', 0) >= 0 else 'negative'}">{summary.get('realized_roac', 0):.2f}%</div>
        </div>
        <div class="metric">
            <div>Projected ROAC</div>
            <div class="metric-value {'positive' if summary.get('projected_roac', 0) >= 0 else 'negative'}">{summary.get('projected_roac', 0):.2f}%</div>
        </div>
    </div>
    
    <div id="charts">
        {fig.to_html(include_plotlyjs='cdn', div_id='charts')}
    </div>
    
    <div class="summary">
        <h2>📊 Projected ROAC Methodology</h2>
        <p><strong>Calculation Methods by Security:</strong></p>
        <ul>
            <li><strong>CHWY:</strong> Standard probability analysis (28% win rate at current entry level)</li>
            <li><strong>BE:</strong> Generalized entry strategy using price invariance principle (40% win rate based on optimal entry patterns)</li>
            <li><strong>PFE:</strong> Standard probability analysis (49% win rate at current entry level)</li>
        </ul>
        <p><em>BE projection uses enhanced methodology accounting for candle pattern invariance across price levels, 
        reflecting expected performance with disciplined entry selection rather than fixed price entry.</em></p>
    </div>
    
    <h2>📋 Trade Details</h2>
    
    <div style="margin-bottom: 20px;">
        <button onclick="filterTrades('all')" id="btn-all" class="filter-btn active">All Trades</button>
        <button onclick="filterTrades('watchlist')" id="btn-watchlist" class="filter-btn">Watchlist Only</button>
        <button onclick="filterTrades('non-watchlist')" id="btn-non-watchlist" class="filter-btn">Non-Watchlist Only</button>
    </div>
    
    <table id="trades-table">
        <tr>
            <th>Symbol</th>
            <th>Watchlist</th>
            <th>Strategy</th>
            <th>Entry Price</th>
            <th>Stop Loss</th>
            <th>Take Profit</th>
            <th>Risk:Reward</th>
            <th>Status</th>
            <th>P&L</th>
        </tr>
        """
        
        # Load watchlist for status indicators
        watchlist_symbols = self.load_watchlist_symbols()
        
        for trade in trades:
            entry_price = trade["entry_order"].get("fill_price") or trade["entry_order"].get("entry_price") or trade["entry_order"].get("limit_price")
            status = trade["current_status"]["overall_status"]
            
            # Determine watchlist status
            clean_symbol = self._get_clean_symbol(trade["symbol"])
            is_in_watchlist = clean_symbol in watchlist_symbols
            watchlist_status = "✅" if is_in_watchlist else "❌"
            watchlist_class = "watchlist-yes" if is_in_watchlist else "watchlist-no"
            
            pnl_display = "N/A"
            pnl_class = ""
            if "exit_details" in trade:
                pnl = trade["exit_details"]["total_pnl"]
                pnl_display = f"${pnl:.2f}"
                pnl_class = "positive" if pnl >= 0 else "negative"
            
            # Handle different data structures for stop loss and take profit
            stop_loss_price = trade['stop_loss_order'].get('trigger_price', 'N/A')
            take_profit_price = trade['take_profit_order'].get('limit_price', 'N/A')
            
            html_content += f"""
        <tr class="trade-row {watchlist_class}">
            <td>{trade['symbol']}</td>
            <td>{watchlist_status}</td>
            <td>{trade['strategy_name']}</td>
            <td>${entry_price:.2f}</td>
            <td>{f'${stop_loss_price:.2f}' if stop_loss_price != 'N/A' else 'N/A'}</td>
            <td>{f'${take_profit_price:.2f}' if take_profit_price != 'N/A' else 'N/A'}</td>
            <td>{trade['trade_analysis']['risk_reward_ratio']:.2f}:1</td>
            <td>{status.replace('_', ' ').title()}</td>
            <td class="{pnl_class}">{pnl_display}</td>
        </tr>
            """
        
        html_content += f"""
    </table>
    
    <p><em>Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    
    <script>
    function filterTrades(filter) {{
        const rows = document.querySelectorAll('.trade-row');
        const buttons = document.querySelectorAll('.filter-btn');
        
        // Update button states
        buttons.forEach(btn => btn.classList.remove('active'));
        document.getElementById(`btn-${{filter}}`).classList.add('active');
        
        // Show/hide rows based on filter
        rows.forEach(row => {{
            row.classList.remove('hidden');
            
            if (filter === 'watchlist' && row.classList.contains('watchlist-no')) {{
                row.classList.add('hidden');
            }} else if (filter === 'non-watchlist' && row.classList.contains('watchlist-yes')) {{
                row.classList.add('hidden');
            }}
        }});
    }}
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Paper trading report saved: {output_file}")
        return output_file
    
    def get_active_trades(self) -> List[Dict]:
        """Get all active trades"""
        return [trade for trade in self.data["paper_trades"] 
                if trade["current_status"]["overall_status"] in ["pending_entry", "position_open"]]
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Dict]:
        """Get a specific trade by ID"""
        for trade in self.data["paper_trades"]:
            if trade["trade_id"] == trade_id:
                return trade
        return None
    
    def print_summary(self):
        """Print a summary of the paper trading journal"""
        summary = self.data["paper_trading_summary"]
        
        print("🎯 PAPER TRADING JOURNAL SUMMARY")
        print("=" * 50)
        print(f"Total Strategies: {summary.get('total_strategies', 0)}")
        print(f"Active Positions: {summary.get('active_positions', 0)}")
        print(f"Pending Entries: {summary.get('pending_entries', 0)}")
        print(f"Closed Trades: {summary.get('closed_trades', 0)}")
        print(f"Capital Allocated: ${summary.get('total_capital_allocated', 0):.2f}")
        print(f"Closed Trade Capital: ${summary.get('total_closed_capital', 0):.2f}")
        print(f"Total Risk Exposure: ${summary.get('total_risk_exposure', 0):.2f}")
        print(f"Realized P&L: ${summary.get('realized_pnl', 0):.2f}")
        print(f"Realized ROAC: {summary.get('realized_roac', 0):.2f}%")
        print(f"Projected ROAC: {summary.get('projected_roac', 0):.2f}%")
        print(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
        print(f"Average Risk:Reward: {summary.get('average_risk_reward', 0):.2f}:1")
        print(f"Last Updated: {summary.get('last_updated', 'Never')}")

def main():
    """Demo usage of paper trading journal"""
    journal = PaperTradingJournal()
    
    # Print current summary
    journal.print_summary()
    
    # Generate report
    journal.generate_report()

if __name__ == "__main__":
    main()