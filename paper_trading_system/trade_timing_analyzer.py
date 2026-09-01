#!/usr/bin/env python3
"""
Trade Timing Analyzer
Analyzes historical price movements to determine expected wait periods for target moves
"""

import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from market_data_database import get_default_database_path
import warnings
warnings.filterwarnings('ignore')

class TradeTimingAnalyzer:
    def __init__(self, db_path):
        """Initialize with database connection"""
        self.db_path = db_path
        self.conn = None
        self.daily_data = None
        
    def connect_database(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def load_historical_data(self, symbol, days_back=365):
        """Load historical daily data for analysis"""
        if not self.conn:
            self.connect_database()
        
        print(f"📊 Loading {days_back} days of historical data for {symbol}...")
        
        # Get data from the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT symbol, datetime, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol = ?
        AND DATE(datetime) >= DATE(?)
        ORDER BY datetime
        """
        
        try:
            params = [symbol, start_date.strftime('%Y-%m-%d')]
            price_data = pd.read_sql_query(query, self.conn, params=params)
            
            if price_data.empty:
                print(f"❌ No data found for {symbol}")
                return None
            
            # Aggregate to daily OHLCV
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
            
            print(f"✅ Loaded {len(daily_data)} trading days from {daily_data['date'].min()} to {daily_data['date'].max()}")
            
            self.daily_data = daily_data
            return daily_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_target_move_timing(self, entry_price, target_price, stop_price=None):
        """Analyze how long it typically takes to hit target moves"""
        if self.daily_data is None:
            print("❌ No historical data loaded")
            return None
        
        print(f"\n🎯 ANALYZING TARGET MOVE TIMING")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Target Price: ${target_price:.2f}")
        if stop_price:
            print(f"Stop Price: ${stop_price:.2f}")
        print("=" * 60)
        
        df = self.daily_data.copy()
        
        # Calculate price movement metrics
        target_move_pct = ((target_price - entry_price) / entry_price) * 100
        stop_move_pct = ((entry_price - stop_price) / entry_price) * 100 if stop_price else 0
        
        print(f"Target Move: {target_move_pct:+.2f}%")
        if stop_price:
            print(f"Stop Move: {stop_move_pct:+.2f}%")
        
        # Analyze all possible entry points
        results = []
        
        for i in range(len(df) - 1):  # Don't include last day as entry
            entry_date = df.iloc[i]['date']
            entry_close = df.iloc[i]['close']
            
            # Skip if current price is too far from our entry price
            if abs((entry_close - entry_price) / entry_price) > 0.10:  # 10% tolerance
                continue
            
            # Look forward for target/stop hits
            target_hit = False
            stop_hit = False
            target_days = None
            stop_days = None
            
            for j in range(i + 1, min(i + 91, len(df))):  # Look ahead up to 90 trading days
                future_date = df.iloc[j]['date']
                future_high = df.iloc[j]['high']
                future_low = df.iloc[j]['low']
                days_elapsed = j - i
                
                # Check if target hit
                if not target_hit and future_high >= target_price:
                    target_hit = True
                    target_days = days_elapsed
                
                # Check if stop hit  
                if stop_price and not stop_hit and future_low <= stop_price:
                    stop_hit = True
                    stop_days = days_elapsed
                
                # If both hit, record which came first
                if target_hit or stop_hit:
                    break
            
            # Record the outcome
            if target_hit or stop_hit:
                results.append({
                    'entry_date': entry_date,
                    'entry_price': entry_close,
                    'target_hit': target_hit,
                    'stop_hit': stop_hit,
                    'target_days': target_days,
                    'stop_days': stop_days,
                    'outcome': 'target' if target_hit and (not stop_hit or (stop_days and target_days and target_days <= stop_days)) else 'stop',
                    'days_to_resolution': target_days if target_hit and (not stop_hit or (stop_days and target_days and target_days <= stop_days)) else stop_days
                })
        
        if not results:
            print("❌ No comparable entry scenarios found in historical data")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        total_scenarios = len(results_df)
        target_wins = len(results_df[results_df['outcome'] == 'target'])
        stop_losses = len(results_df[results_df['outcome'] == 'stop'])
        
        win_rate = (target_wins / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        # Time analysis for winning trades
        winning_trades = results_df[results_df['outcome'] == 'target']
        if not winning_trades.empty:
            avg_days_to_target = winning_trades['days_to_resolution'].mean()
            median_days_to_target = winning_trades['days_to_resolution'].median()
            min_days = winning_trades['days_to_resolution'].min()
            max_days = winning_trades['days_to_resolution'].max()
            
            # Percentiles
            p25 = winning_trades['days_to_resolution'].quantile(0.25)
            p75 = winning_trades['days_to_resolution'].quantile(0.75)
        else:
            avg_days_to_target = median_days_to_target = min_days = max_days = p25 = p75 = None
        
        # Time analysis for losing trades
        losing_trades = results_df[results_df['outcome'] == 'stop']
        if not losing_trades.empty:
            avg_days_to_stop = losing_trades['days_to_resolution'].mean()
            median_days_to_stop = losing_trades['days_to_resolution'].median()
        else:
            avg_days_to_stop = median_days_to_stop = None
        
        print(f"\n📈 HISTORICAL PERFORMANCE ANALYSIS")
        print(f"Total scenarios analyzed: {total_scenarios}")
        print(f"Target hits: {target_wins} ({win_rate:.1f}%)")
        print(f"Stop hits: {stop_losses} ({100-win_rate:.1f}%)")
        
        if winning_trades.empty:
            print("\n⚠️ No successful target hits found in historical data")
            print("Consider adjusting target price or analyzing longer time period")
        else:
            print(f"\n⏰ TARGET TIMING ANALYSIS ({target_wins} successful trades):")
            print(f"Average days to target: {avg_days_to_target:.1f}")
            print(f"Median days to target: {median_days_to_target:.1f}")
            print(f"Range: {min_days} to {max_days} days")
            print(f"25th percentile: {p25:.1f} days")
            print(f"75th percentile: {p75:.1f} days")
            
            # Convert to calendar days (approximate)
            calendar_avg = avg_days_to_target * 1.43  # ~7 trading days = 10 calendar days
            calendar_median = median_days_to_target * 1.43
            
            print(f"\n📅 CALENDAR TIME ESTIMATES:")
            print(f"Average wait: ~{calendar_avg:.0f} calendar days ({calendar_avg/30:.1f} months)")
            print(f"Median wait: ~{calendar_median:.0f} calendar days ({calendar_median/30:.1f} months)")
            
        if not losing_trades.empty:
            print(f"\n💔 STOP LOSS TIMING ({stop_losses} losing trades):")
            print(f"Average days to stop: {avg_days_to_stop:.1f}")
            print(f"Median days to stop: {median_days_to_stop:.1f}")
        
        # Calculate annualized returns
        if avg_days_to_target:
            expected_return_pct = target_move_pct * (win_rate / 100)
            trade_frequency_per_year = 252 / avg_days_to_target  # 252 trading days per year
            annualized_return = expected_return_pct * trade_frequency_per_year
            
            print(f"\n💰 INCOME PLAN PROJECTIONS:")
            print(f"Expected return per trade: {expected_return_pct:.2f}%")
            print(f"Potential trades per year: {trade_frequency_per_year:.1f}")
            print(f"Estimated annualized return: {annualized_return:.1f}%")
            
            # Monthly income projection
            monthly_frequency = trade_frequency_per_year / 12
            monthly_return = expected_return_pct * monthly_frequency
            
            print(f"\n📊 MONTHLY INCOME PROJECTION:")
            print(f"Expected trades per month: {monthly_frequency:.2f}")
            print(f"Expected monthly return: {monthly_return:.2f}%")
            
            if monthly_return > 0:
                capital_for_1pct_monthly = 100 / monthly_return
                print(f"Capital needed for 1% monthly income: ${capital_for_1pct_monthly:,.0f}")
        
        return {
            'total_scenarios': total_scenarios,
            'win_rate': win_rate,
            'target_wins': target_wins,
            'stop_losses': stop_losses,
            'avg_days_to_target': avg_days_to_target,
            'median_days_to_target': median_days_to_target,
            'avg_days_to_stop': avg_days_to_stop,
            'median_days_to_stop': median_days_to_stop,
            'results_df': results_df,
            'expected_return_per_trade': expected_return_pct if 'expected_return_pct' in locals() else None,
            'trades_per_year': trade_frequency_per_year if 'trade_frequency_per_year' in locals() else None,
            'annualized_return': annualized_return if 'annualized_return' in locals() else None
        }
    
    def create_timing_visualization(self, symbol, entry_price, target_price, stop_price, analysis_results):
        """Create comprehensive timing analysis visualization"""
        if not analysis_results or analysis_results['results_df'].empty:
            print("❌ No data to visualize")
            return None
        
        results_df = analysis_results['results_df']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'{symbol} Price Chart with Trade Levels',
                'Days to Target Distribution',
                'Win Rate by Time Period', 
                'Monthly Success Probability',
                'Outcome Timeline',
                'Return Frequency Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}], 
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12
        )
        
        # 1. Price chart with trade levels
        df = self.daily_data
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='white', width=1)
            ),
            row=1, col=1
        )
        
        # Add horizontal lines for trade levels
        date_range = [df['date'].min(), df['date'].max()]
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=[entry_price, entry_price],
                mode='lines',
                name='Entry Price',
                line=dict(color='yellow', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=date_range, 
                y=[target_price, target_price],
                mode='lines',
                name='Target Price',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        if stop_price:
            fig.add_trace(
                go.Scatter(
                    x=date_range,
                    y=[stop_price, stop_price], 
                    mode='lines',
                    name='Stop Price',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Days to target distribution
        winning_trades = results_df[results_df['outcome'] == 'target']
        if not winning_trades.empty:
            fig.add_trace(
                go.Histogram(
                    x=winning_trades['days_to_resolution'],
                    nbinsx=20,
                    name='Days to Target',
                    marker_color='green',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. Win rate by time period
        time_buckets = ['1-5 days', '6-10 days', '11-20 days', '21-30 days', '31-60 days', '60+ days']
        win_rates = []
        
        for bucket in time_buckets:
            if bucket == '1-5 days':
                subset = results_df[results_df['days_to_resolution'].between(1, 5)]
            elif bucket == '6-10 days':
                subset = results_df[results_df['days_to_resolution'].between(6, 10)]
            elif bucket == '11-20 days': 
                subset = results_df[results_df['days_to_resolution'].between(11, 20)]
            elif bucket == '21-30 days':
                subset = results_df[results_df['days_to_resolution'].between(21, 30)]
            elif bucket == '31-60 days':
                subset = results_df[results_df['days_to_resolution'].between(31, 60)]
            else:
                subset = results_df[results_df['days_to_resolution'] > 60]
            
            if len(subset) > 0:
                win_rate = (len(subset[subset['outcome'] == 'target']) / len(subset)) * 100
            else:
                win_rate = 0
            win_rates.append(win_rate)
        
        fig.add_trace(
            go.Bar(
                x=time_buckets,
                y=win_rates,
                name='Win Rate %',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # 4. Monthly success probability
        monthly_data = results_df.copy()
        monthly_data['month'] = pd.to_datetime(monthly_data['entry_date']).dt.strftime('%b')
        month_stats = monthly_data.groupby('month').agg({
            'outcome': lambda x: (x == 'target').sum() / len(x) * 100
        }).reset_index()
        month_stats.columns = ['month', 'win_rate']
        
        fig.add_trace(
            go.Bar(
                x=month_stats['month'],
                y=month_stats['win_rate'],
                name='Monthly Win Rate',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # 5. Outcome timeline
        fig.add_trace(
            go.Scatter(
                x=results_df['entry_date'],
                y=results_df['days_to_resolution'],
                mode='markers',
                marker=dict(
                    color=['green' if x == 'target' else 'red' for x in results_df['outcome']],
                    size=8,
                    symbol=['circle' if x == 'target' else 'x' for x in results_df['outcome']]
                ),
                name='Trade Outcomes',
                text=[f"{row['outcome'].title()}: {row['days_to_resolution']} days" for _, row in results_df.iterrows()],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Return frequency
        if not winning_trades.empty:
            return_buckets = ['<1 week', '1-2 weeks', '2-4 weeks', '1-2 months', '2+ months']
            return_counts = []
            
            for bucket in return_buckets:
                if bucket == '<1 week':
                    count = len(winning_trades[winning_trades['days_to_resolution'] <= 5])
                elif bucket == '1-2 weeks':
                    count = len(winning_trades[winning_trades['days_to_resolution'].between(6, 10)])
                elif bucket == '2-4 weeks':
                    count = len(winning_trades[winning_trades['days_to_resolution'].between(11, 20)])
                elif bucket == '1-2 months':
                    count = len(winning_trades[winning_trades['days_to_resolution'].between(21, 40)])
                else:
                    count = len(winning_trades[winning_trades['days_to_resolution'] > 40])
                return_counts.append(count)
            
            fig.add_trace(
                go.Bar(
                    x=return_buckets,
                    y=return_counts,
                    name='Successful Trades',
                    marker_color='lightgreen'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Trade Timing Analysis: {symbol}",
            title_font_size=20,
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white'),
            showlegend=True
        )
        
        # Update all subplot backgrounds
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(gridcolor='#404040', row=i, col=j)
                fig.update_yaxes(gridcolor='#404040', row=i, col=j)
        
        return fig
    
    def save_timing_report(self, symbol, entry_price, target_price, stop_price, analysis_results, output_file=None):
        """Save comprehensive timing analysis report"""
        if not output_file:
            output_file = f"{symbol}_timing_analysis.html"
        
        fig = self.create_timing_visualization(symbol, entry_price, target_price, stop_price, analysis_results)
        
        if fig:
            fig.write_html(output_file)
            print(f"✅ Timing analysis report saved: {output_file}")
        
        return output_file

def main():
    """Demo timing analysis for CHWY"""
    analyzer = TradeTimingAnalyzer(get_default_database_path())
    
    # CHWY trade parameters
    symbol = "CHWY"
    entry_price = 35.86
    target_price = 37.25
    stop_price = 35.60
    
    # Load data and analyze
    data = analyzer.load_historical_data(symbol, days_back=365)
    if data is not None:
        results = analyzer.analyze_target_move_timing(entry_price, target_price, stop_price)
        if results:
            analyzer.save_timing_report(symbol, entry_price, target_price, stop_price, results)

if __name__ == "__main__":
    main()