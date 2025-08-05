#!/usr/bin/env python3
"""
Trading Journal System for Covered Strategies
Tracks trades, calculates metrics, and generates HTML reports
"""

import json
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os

@dataclass
class Trade:
    symbol: str
    entry_price: float
    entry_date: str
    quantity: int
    trade_type: str  # 'market_buy', 'limit_buy', 'limit_sell'
    
    # For limit orders
    limit_price: Optional[float] = None
    expiration_date: Optional[str] = None
    
    # Status tracking
    status: str = 'pending'  # 'pending', 'filled', 'cancelled'
    fill_date: Optional[str] = None
    fill_price: Optional[float] = None

@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

@dataclass
class Strategy:
    name: str
    symbol: str
    initial_position: Trade
    buy_limit: Optional[Trade] = None
    sell_limit: Optional[Trade] = None
    
    def calculate_risk_reward(self) -> Dict:
        if not self.buy_limit or not self.sell_limit:
            return {}
            
        entry_price = self.initial_position.entry_price
        buy_price = self.buy_limit.limit_price
        sell_price = self.sell_limit.limit_price
        
        # Risk: How much the asset could drop before placing another buy order
        # This is the cushion you're comfortable with
        risk_cushion = entry_price - buy_price
        
        # Reward scenarios:
        # Scenario 1: Only sell fills (sell at $80, bought at $76.36)
        reward_single_fill = sell_price - entry_price
        
        # Scenario 2: Both fill (sell at $80, average cost is ($76.36 + $75.50)/2)
        avg_cost_if_both = (entry_price + buy_price) / 2
        reward_both_fill = sell_price - avg_cost_if_both
        
        # Two distinct Risk/Reward ratios
        rr_ratio_single = reward_single_fill / risk_cushion if risk_cushion > 0 else float('inf')
        rr_ratio_both = reward_both_fill / risk_cushion if risk_cushion > 0 else float('inf')
        
        # ROI calculations based on capital deployed (asset ownership requirement)
        # Single fill: ROI on initial position only
        roi_single = (reward_single_fill / entry_price) * 100
        
        # Both fill: ROI on total capital deployed (initial + additional buy)
        total_capital_both = entry_price + buy_price
        roi_both = (reward_both_fill / total_capital_both) * 100
        
        return {
            'risk_cushion': risk_cushion,
            'reward_single_fill': reward_single_fill,
            'reward_both_fill': reward_both_fill,
            'rr_ratio_single': rr_ratio_single,
            'rr_ratio_both': rr_ratio_both,
            'roi_single': roi_single,
            'roi_both': roi_both,
            'entry_price': entry_price,
            'buy_limit_price': buy_price,
            'sell_limit_price': sell_price
        }
    
    def get_potential_outcomes(self) -> List[Dict]:
        """Calculate potential outcomes: 0, 1, or 2 shares"""
        entry_price = self.initial_position.entry_price
        buy_price = self.buy_limit.limit_price if self.buy_limit else None
        sell_price = self.sell_limit.limit_price if self.sell_limit else None
        
        outcomes = []
        
        # Scenario 1: Both limits fill (0 shares)
        if buy_price and sell_price:
            pnl = (sell_price - entry_price) + (buy_price - entry_price)
            outcomes.append({
                'shares': 0,
                'scenario': 'Both limits fill',
                'pnl': pnl,
                'description': f'Sell at ${sell_price:.2f}, buy at ${buy_price:.2f}'
            })
        
        # Scenario 2: Only sell limit fills (0 shares)
        if sell_price:
            pnl = sell_price - entry_price
            outcomes.append({
                'shares': 0,
                'scenario': 'Only sell limit fills',
                'pnl': pnl,
                'description': f'Sell at ${sell_price:.2f}'
            })
        
        # Scenario 3: Only buy limit fills (2 shares)
        if buy_price:
            avg_cost = (entry_price + buy_price) / 2
            outcomes.append({
                'shares': 2,
                'scenario': 'Only buy limit fills',
                'avg_cost': avg_cost,
                'description': f'Buy additional at ${buy_price:.2f}, avg cost ${avg_cost:.2f}'
            })
        
        # Scenario 4: No limits fill (1 share)
        outcomes.append({
            'shares': 1,
            'scenario': 'No limits fill',
            'cost_basis': entry_price,
            'description': f'Hold original position at ${entry_price:.2f}'
        })
        
        return outcomes

class TradingJournal:
    def __init__(self, journal_file: str = 'trading_journal.json'):
        self.journal_file = journal_file
        self.strategies: List[Strategy] = []
        self.load_journal()
    
    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)
        self.save_journal()
    
    def save_journal(self):
        data = {
            'strategies': [asdict(strategy) for strategy in self.strategies],
            'last_updated': datetime.datetime.now().isoformat()
        }
        with open(self.journal_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_journal(self):
        if os.path.exists(self.journal_file):
            with open(self.journal_file, 'r') as f:
                data = json.load(f)
                # Reconstruct strategies from saved data
                for strategy_data in data.get('strategies', []):
                    initial_trade = Trade(**strategy_data['initial_position'])
                    buy_limit = Trade(**strategy_data['buy_limit']) if strategy_data['buy_limit'] else None
                    sell_limit = Trade(**strategy_data['sell_limit']) if strategy_data['sell_limit'] else None
                    
                    strategy = Strategy(
                        name=strategy_data['name'],
                        symbol=strategy_data['symbol'],
                        initial_position=initial_trade,
                        buy_limit=buy_limit,
                        sell_limit=sell_limit
                    )
                    self.strategies.append(strategy)
    
    def generate_html_report(self, output_file: str = 'trading_journal_report.html'):
        html_content = self._create_html_report()
        with open(output_file, 'w') as f:
            f.write(html_content)
        return output_file
    
    def _create_html_report(self) -> str:
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Journal Report</title>
    <style>
        body {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            color: #ffffff;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #4CAF50;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }}
        
        .strategy-card {{
            background: linear-gradient(145deg, #2d5aa0 0%, #1e3c72 100%);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border: 2px solid #4CAF50;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }}
        
        .strategy-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 15px;
        }}
        
        .strategy-name {{
            font-size: 1.8em;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .symbol {{
            font-size: 1.5em;
            background: #FF6B35;
            padding: 8px 15px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #B0BEC5;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .outcomes-section {{
            margin-top: 25px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            border: 1px solid #FF6B35;
        }}
        
        .outcomes-title {{
            font-size: 1.3em;
            color: #FF6B35;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        
        .outcome-item {{
            background: rgba(255, 255, 255, 0.05);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        
        .outcome-scenario {{
            font-weight: bold;
            color: #4CAF50;
            font-size: 1.1em;
        }}
        
        .outcome-details {{
            color: #E0E0E0;
            margin-top: 5px;
        }}
        
        .positive {{
            color: #4CAF50 !important;
        }}
        
        .negative {{
            color: #F44336 !important;
        }}
        
        .timestamp {{
            text-align: center;
            margin-top: 30px;
            color: #B0BEC5;
            font-style: italic;
        }}
        
        .trade-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .trade-card {{
            background: rgba(255, 255, 255, 0.08);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 107, 53, 0.3);
        }}
        
        .trade-type {{
            font-weight: bold;
            color: #FF6B35;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Trading Journal Report</h1>
        {self._generate_strategies_html()}
        <div class="timestamp">
            Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_strategies_html(self) -> str:
        if not self.strategies:
            return '<p style="text-align: center; color: #B0BEC5;">No strategies recorded yet.</p>'
        
        strategies_html = ""
        for strategy in self.strategies:
            metrics = strategy.calculate_risk_reward()
            outcomes = strategy.get_potential_outcomes()
            
            strategies_html += f"""
        <div class="strategy-card">
            <div class="strategy-header">
                <div class="strategy-name">{strategy.name}</div>
                <div class="symbol">{strategy.symbol}</div>
            </div>
            
            <div class="trade-details">
                <div class="trade-card">
                    <div class="trade-type">Initial Position</div>
                    <div>Price: ${strategy.initial_position.entry_price:.2f}</div>
                    <div>Quantity: {strategy.initial_position.quantity}</div>
                    <div>Date: {strategy.initial_position.entry_date}</div>
                </div>
                
                {f'''<div class="trade-card">
                    <div class="trade-type">Buy Limit Order</div>
                    <div>Price: ${strategy.buy_limit.limit_price:.2f}</div>
                    <div>Quantity: {strategy.buy_limit.quantity}</div>
                    <div>Expires: {strategy.buy_limit.expiration_date}</div>
                </div>''' if strategy.buy_limit else ''}
                
                {f'''<div class="trade-card">
                    <div class="trade-type">Sell Limit Order</div>
                    <div>Price: ${strategy.sell_limit.limit_price:.2f}</div>
                    <div>Quantity: {strategy.sell_limit.quantity}</div>
                    <div>Expires: {strategy.sell_limit.expiration_date}</div>
                </div>''' if strategy.sell_limit else ''}
            </div>
            
            {f'''<div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Risk Cushion</div>
                    <div class="metric-value negative">${metrics['risk_cushion']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Reward (Single)</div>
                    <div class="metric-value positive">${metrics['reward_single_fill']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">R/R (Single)</div>
                    <div class="metric-value">{metrics['rr_ratio_single']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">ROI (Single)</div>
                    <div class="metric-value positive">{metrics['roi_single']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Reward (Both)</div>
                    <div class="metric-value positive">${metrics['reward_both_fill']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">R/R (Both)</div>
                    <div class="metric-value">{metrics['rr_ratio_both']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">ROI (Both)</div>
                    <div class="metric-value positive">{metrics['roi_both']:.1f}%</div>
                </div>
            </div>''' if metrics else ''}
            
            <div class="outcomes-section">
                <div class="outcomes-title">🎯 Potential Outcomes</div>
                {self._generate_outcomes_html(outcomes)}
            </div>
        </div>
            """
        
        return strategies_html
    
    def _generate_outcomes_html(self, outcomes: List[Dict]) -> str:
        outcomes_html = ""
        for outcome in outcomes:
            pnl_class = "positive" if outcome.get('pnl', 0) > 0 else "negative" if outcome.get('pnl', 0) < 0 else ""
            pnl_display = f"<strong>P&L: <span class='{pnl_class}'>${outcome['pnl']:.2f}</span></strong>" if 'pnl' in outcome else ""
            cost_display = f"<strong>Avg Cost: ${outcome['avg_cost']:.2f}</strong>" if 'avg_cost' in outcome else ""
            cost_basis_display = f"<strong>Cost Basis: ${outcome['cost_basis']:.2f}</strong>" if 'cost_basis' in outcome else ""
            
            outcomes_html += f"""
                <div class="outcome-item">
                    <div class="outcome-scenario">
                        {outcome['shares']} Shares - {outcome['scenario']}
                    </div>
                    <div class="outcome-details">
                        {outcome['description']}<br>
                        {pnl_display}{cost_display}{cost_basis_display}
                    </div>
                </div>
            """
        
        return outcomes_html

if __name__ == "__main__":
    # Example usage with USO trade - only add if not already exists
    journal = TradingJournal()
    
    # Check if USO strategy already exists
    uso_exists = any(s.name == "USO Covered Strategy" and s.symbol == "USO" for s in journal.strategies)
    
    if not uso_exists:
        # Create USO strategy
        uso_initial = Trade(
            symbol="USO",
            entry_price=76.36,
            entry_date="2025-08-04",
            quantity=1,
            trade_type="market_buy",
            status="filled"
        )
        
        uso_buy_limit = Trade(
            symbol="USO",
            entry_price=75.50,  # For limit orders, entry_price is the limit price
            entry_date="2025-08-04",
            quantity=1,
            trade_type="limit_buy",
            limit_price=75.50,
            expiration_date="2025-11-02"  # 90 days out
        )
        
        uso_sell_limit = Trade(
            symbol="USO",
            entry_price=80.00,  # For limit orders, entry_price is the limit price
            entry_date="2025-08-04",
            quantity=1,
            trade_type="limit_sell",
            limit_price=80.00,
            expiration_date="2025-11-02"  # 90 days out
        )
        
        uso_strategy = Strategy(
            name="USO Covered Strategy",
            symbol="USO",
            initial_position=uso_initial,
            buy_limit=uso_buy_limit,
            sell_limit=uso_sell_limit
        )
        
        journal.add_strategy(uso_strategy)
        print("USO strategy added to journal")
    else:
        print("USO strategy already exists in journal")
    
    report_file = journal.generate_html_report()
    print(f"Trading journal report generated: {report_file}")