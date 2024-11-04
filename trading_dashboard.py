import sqlite3

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from trading_system import TradingAnalytics
from typing import Optional, Dict, List

class TradingDashboard:
    """
    Dashboard for monitoring trading performance and risk metrics.
    Builds on top of TradingAnalytics to provide actionable insights.
    """
    
    def __init__(self, analytics):
        self.analytics = analytics
        
    def daily_summary(self, date: Optional[str] = None) -> Dict:
        """Generate daily performance summary"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        yesterday = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get metrics for today and yesterday for comparison
        today_metrics = self.analytics.calculate_metrics(date, date)
        yesterday_metrics = self.analytics.calculate_metrics(yesterday, yesterday)
        
        return {
            'date': date,
            'daily_pnl': self._get_daily_pnl(date),
            'trades_today': today_metrics.total_trades if today_metrics else 0,
            'win_rate_today': today_metrics.win_rate if today_metrics else 0,
            'win_rate_change': self._calculate_metric_change(
                today_metrics.win_rate if today_metrics else 0,
                yesterday_metrics.win_rate if yesterday_metrics else 0
            ),
            'largest_winner': self._get_largest_trade(date, win=True),
            'largest_loser': self._get_largest_trade(date, win=False),
            'open_positions': self._get_open_positions(),
            'risk_alerts': self._get_risk_alerts()
        }
    
    def risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        now = datetime.now()
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        
        return {
            'position_risk': self._analyze_position_risk(),
            'portfolio_metrics': self._get_portfolio_metrics(start_date, end_date),
            'stop_loss_analysis': self._analyze_stop_losses(),
            'exposure_analysis': self._analyze_exposures(),
            'recommendations': self._generate_risk_recommendations()
        }
    
    def _get_daily_pnl(self, date: str) -> float:
        """Calculate daily P&L"""
        conn = sqlite3.connect(self.analytics.db_path)
        query = """
            SELECT SUM(profit_loss) as daily_pnl
            FROM trades
            WHERE date(timestamp) = ?
        """
        result = pd.read_sql_query(query, conn, params=[date])
        conn.close()
        return float(result['daily_pnl'].iloc[0] or 0)
    
    def _get_largest_trade(self, date: str, win: bool = True) -> Dict:
        """Get details of largest winning or losing trade"""
        conn = sqlite3.connect(self.analytics.db_path)
        query = """
            SELECT *
            FROM trades
            WHERE date(timestamp) = ?
            AND profit_loss {} 0
            ORDER BY ABS(profit_loss) DESC
            LIMIT 1
        """.format('>' if win else '<')
        
        trade = pd.read_sql_query(query, conn, params=[date])
        conn.close()
        
        if len(trade) == 0:
            return {'symbol': None, 'profit_loss': 0}
            
        return {
            'symbol': trade['symbol'].iloc[0],
            'profit_loss': float(trade['profit_loss'].iloc[0])
        }
    
    def _get_open_positions(self) -> List[Dict]:
        """Get current open positions with risk metrics"""
        conn = sqlite3.connect(self.analytics.db_path)
        query = """
            SELECT 
                symbol,
                SUM(CASE WHEN side = 'buy' THEN quantity ELSE -quantity END) as net_position,
                AVG(entry_price) as avg_entry,
                MIN(stop_loss) as stop_loss
            FROM open_positions
            GROUP BY symbol
            HAVING net_position != 0
        """
        positions = pd.read_sql_query(query, conn)
        conn.close()
        
        return positions.to_dict('records')
    
    def _get_risk_alerts(self) -> List[str]:
        """Generate risk alerts based on current positions and market conditions"""
        alerts = []
        
        # Check position sizes
        positions = self._get_open_positions()
        for position in positions:
            if abs(position['net_position'] * position['avg_entry']) > 100000:  # Example threshold
                alerts.append(f"Large position alert: {position['symbol']}")
            
            if position['stop_loss'] is None:
                alerts.append(f"Missing stop-loss: {position['symbol']}")
        
        # Check portfolio concentration
        if len(positions) > 0:
            largest_position = max(positions, key=lambda x: abs(x['net_position'] * x['avg_entry']))
            portfolio_value = sum(abs(p['net_position'] * p['avg_entry']) for p in positions)
            concentration = abs(largest_position['net_position'] * largest_position['avg_entry']) / portfolio_value
            
            if concentration > 0.25:  # Example threshold
                alerts.append(f"High concentration in {largest_position['symbol']}: {concentration:.1%}")
        
        return alerts
    
    def _analyze_position_risk(self) -> Dict:
        """Analyze risk metrics for all positions"""
        positions = self._get_open_positions()
        
        total_exposure = 0
        risk_metrics = {}
        
        for position in positions:
            symbol = position['symbol']
            quantity = position['net_position']
            entry = position['avg_entry']
            stop = position['stop_loss']
            
            position_value = abs(quantity * entry)
            total_exposure += position_value
            
            if stop:
                risk_amount = abs(entry - stop) * abs(quantity)
                risk_percent = risk_amount / position_value
            else:
                risk_amount = None
                risk_percent = None
            
            risk_metrics[symbol] = {
                'position_value': position_value,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent
            }
        
        return {
            'total_exposure': total_exposure,
            'position_metrics': risk_metrics
        }
    
    def _get_portfolio_metrics(self, start_date: str, end_date: str) -> Dict:
        """Calculate portfolio-wide risk metrics"""
        metrics = self.analytics.calculate_metrics(start_date, end_date)
        
        return {
            'sharpe_ratio': metrics.sharpe_ratio if metrics else 0,
            'max_drawdown': metrics.max_drawdown if metrics else 0,
            'win_rate': metrics.win_rate if metrics else 0,
            'profit_factor': metrics.profit_factor if metrics else 0
        }
    
    def _analyze_stop_losses(self) -> Dict:
        """Analyze stop-loss performance"""
        conn = sqlite3.connect(self.analytics.db_path)
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN exit_price <= stop_loss AND profit_loss < 0 THEN 1 ELSE 0 END) as stops_hit,
                AVG(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) END) as avg_loss
            FROM trades
            WHERE stop_loss IS NOT NULL
        """
        results = pd.read_sql_query(query, conn)
        conn.close()
        
        row = results.iloc[0]
        return {
            'total_trades': int(row['total_trades']),
            'stops_hit': int(row['stops_hit']),
            'avg_loss': float(row['avg_loss'] or 0),
            'stop_effectiveness': row['stops_hit'] / row['total_trades'] if row['total_trades'] > 0 else 0
        }
    
    def _analyze_exposures(self) -> Dict:
        """Analyze various types of exposure"""
        positions = self._get_open_positions()
        
        # Calculate exposures by symbol
        symbol_exposure = {}
        total_long = 0
        total_short = 0
        
        for position in positions:
            value = position['net_position'] * position['avg_entry']
            symbol_exposure[position['symbol']] = value
            
            if value > 0:
                total_long += value
            else:
                total_short += abs(value)
        
        # Get sector exposures
        sector_exposure = self._calculate_sector_exposure(positions)
        
        return {
            'gross_exposure': total_long + total_short,
            'net_exposure': total_long - total_short,
            'long_exposure': total_long,
            'short_exposure': total_short,
            'symbol_exposure': symbol_exposure,
            'sector_exposure': sector_exposure
        }
    
    def _calculate_sector_exposure(self, positions: List[Dict]) -> Dict:
        """Calculate exposure by sector"""
        conn = sqlite3.connect(self.analytics.db_path)
        sectors = {}
        
        for position in positions:
            query = "SELECT sector FROM asset_metadata WHERE symbol = ?"
            result = pd.read_sql_query(query, conn, params=[position['symbol']])
            
            if len(result) > 0:
                sector = result['sector'].iloc[0]
                exposure = position['net_position'] * position['avg_entry']
                
                if sector in sectors:
                    sectors[sector] += exposure
                else:
                    sectors[sector] = exposure
        
        conn.close()
        return sectors
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Analyze position risk
        risk_analysis = self._analyze_position_risk()
        if risk_analysis['total_exposure'] > 0:
            largest_position = max(
                risk_analysis['position_metrics'].items(),
                key=lambda x: x[1]['position_value']
            )
            
            if largest_position[1]['position_value'] / risk_analysis['total_exposure'] > 0.25:
                recommendations.append(
                    f"Consider reducing position size in {largest_position[0]} "
                    f"to improve portfolio diversification"
                )
        
        # Analyze stop-loss performance
        stop_analysis = self._analyze_stop_losses()
        if stop_analysis['stop_effectiveness'] < 0.5:
            recommendations.append(
                "Stop-loss effectiveness below 50%. Consider reviewing stop-loss placement strategy"
            )
        
        # Analyze exposures
        exposures = self._analyze_exposures()
        if abs(exposures['net_exposure']) / exposures['gross_exposure'] > 0.7:
            recommendations.append(
                "Portfolio heavily directional. Consider adding positions to balance exposure"
            )
        
        return recommendations

    @staticmethod
    def _calculate_metric_change(current: float, previous: float) -> float:
        """Calculate percentage change in a metric"""
        if previous == 0:
            return 0
        return (current - previous) / previous
    
def generate_sample_trades(days=30):
    """Generate realistic sample trading data"""
    trades = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Consumer',
        'META': 'Technology'
    }
    
    base_prices = {
        'AAPL': 175.0,
        'MSFT': 330.0,
        'GOOGL': 140.0,
        'AMZN': 125.0,
        'META': 290.0
    }
    
    now = datetime.now()
    
    for day in range(days):
        date = now - timedelta(days=day)
        
        # Generate 3-8 trades per day
        for _ in range(np.random.randint(3, 9)):
            symbol = np.random.choice(symbols)
            base_price = base_prices[symbol]
            
            # Generate realistic price with some randomness
            entry_price = base_price * (1 + np.random.normal(0, 0.02))
            
            # Determine if it's a winning trade (60% probability)
            is_winner = np.random.random() < 0.6
            
            # Generate exit price based on win/loss
            if is_winner:
                exit_price = entry_price * (1 + abs(np.random.normal(0, 0.01)))
            else:
                exit_price = entry_price * (1 - abs(np.random.normal(0, 0.01)))
            
            # Generate quantity
            quantity = np.random.randint(50, 200)
            
            # Calculate P&L
            profit_loss = (exit_price - entry_price) * quantity
            
            # Generate stop loss and take profit
            stop_loss = entry_price * 0.98  # 2% stop loss
            take_profit = entry_price * 1.02  # 2% take profit
            
            trade = {
                'timestamp': date.replace(
                    hour=np.random.randint(9, 16),
                    minute=np.random.randint(0, 60)
                ).isoformat(),
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'side': 'buy',
                'strategy_name': 'momentum_reversal',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'profit_loss': profit_loss,
                'execution_time_ms': np.random.randint(100, 500)
            }
            trades.append(trade)
    
    return trades, sectors

def populate_demo_database():
    """Populate database with sample data"""
    # Generate sample trades
    trades, sectors = generate_sample_trades()
    
    # Initialize analytics and database
    analytics = TradingAnalytics()
    analytics.initialize_database()
    
    # Create and populate asset_metadata table
    conn = sqlite3.connect(analytics.db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS asset_metadata (
            symbol TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            market_cap REAL
        )
    ''')
    
    # Add sector data
    for symbol, sector in sectors.items():
        c.execute('''
            INSERT OR REPLACE INTO asset_metadata (symbol, sector)
            VALUES (?, ?)
        ''', (symbol, sector))
    
    # Add sample open positions
    c.execute('''
        CREATE TABLE IF NOT EXISTS open_positions (
            position_id INTEGER PRIMARY KEY,
            symbol TEXT,
            quantity REAL,
            entry_price REAL,
            side TEXT,
            timestamp TEXT,
            stop_loss REAL,
            take_profit REAL
        )
    ''')
    
    # Add some open positions
    open_positions = [
        ('AAPL', 100, 175.50, 'buy', datetime.now().isoformat(), 172.0, 180.0),
        ('MSFT', 50, 330.25, 'buy', datetime.now().isoformat(), 325.0, 337.0),
        ('GOOGL', -75, 141.30, 'sell', datetime.now().isoformat(), 144.0, 138.0)
    ]
    
    for pos in open_positions:
        c.execute('''
            INSERT OR REPLACE INTO open_positions 
            (symbol, quantity, entry_price, side, timestamp, stop_loss, take_profit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', pos)
    
    conn.commit()
    
    # Log sample trades
    for trade in trades:
        analytics.log_trade(trade)
    
    return analytics

def run_dashboard_demo():
    """Run comprehensive dashboard demo"""
    print("=== Trading Dashboard Demo ===")
    print("\nInitializing system with sample data...")
    
    # Initialize system with sample data
    analytics = populate_demo_database()
    dashboard = TradingDashboard(analytics)
    
    # Get daily summary
    print("\n=== Today's Summary ===")
    summary = dashboard.daily_summary()
    
    print(f"Daily P&L: ${summary['daily_pnl']:,.2f}")
    print(f"Trades Today: {summary['trades_today']}")
    print(f"Win Rate Today: {summary['win_rate_today']:.1%}")
    print(f"Win Rate Change: {summary['win_rate_change']:+.1%}")
    
    print("\nLargest Winner:", end=" ")
    if summary['largest_winner']['symbol']:
        print(f"{summary['largest_winner']['symbol']}: ${summary['largest_winner']['profit_loss']:,.2f}")
    else:
        print("None")
    
    print("Largest Loser:", end=" ")
    if summary['largest_loser']['symbol']:
        print(f"{summary['largest_loser']['symbol']}: ${summary['largest_loser']['profit_loss']:,.2f}")
    else:
        print("None")
    
    print("\nOpen Positions:")
    for position in summary['open_positions']:
        print(f"- {position['symbol']}: {position['net_position']} shares @ ${position['avg_entry']:.2f}")
    
    print("\nRisk Alerts:")
    for alert in summary['risk_alerts']:
        print(f"- {alert}")
    
    # Get risk report
    print("\n=== Risk Analysis ===")
    risk_report = dashboard.risk_report()
    
    print("\nPortfolio Metrics:")
    metrics = risk_report['portfolio_metrics']
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print("\nPosition Risk:")
    for symbol, metrics in risk_report['position_risk']['position_metrics'].items():
        if metrics['risk_percent'] is not None:
            print(f"- {symbol}: ${metrics['position_value']:,.2f} (Risk: {metrics['risk_percent']:.1%})")
        else:
            print(f"- {symbol}: ${metrics['position_value']:,.2f} (Risk: No stop-loss)")
    
    print("\nExposure Analysis:")
    exposures = risk_report['exposure_analysis']
    print(f"Gross Exposure: ${exposures['gross_exposure']:,.2f}")
    print(f"Net Exposure: ${exposures['net_exposure']:,.2f}")
    print(f"Long Exposure: ${exposures['long_exposure']:,.2f}")
    print(f"Short Exposure: ${exposures['short_exposure']:,.2f}")
    
    print("\nSector Exposure:")
    for sector, exposure in exposures['sector_exposure'].items():
        print(f"- {sector}: ${exposure:,.2f}")
    
    print("\nRecommendations:")
    for rec in risk_report['recommendations']:
        print(f"- {rec}")    