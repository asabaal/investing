#!/usr/bin/env python3
"""
TOP 50 DETAILED ANALYSIS

Comprehensive analysis focused ONLY on the TOP 50 trading candidates
with transparent scoring breakdown and detailed metrics.

Shows exactly how each security was scored and why it made the TOP 50.
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Top50DetailedAnalyzer:
    def __init__(self, top50_csv_path, watchlist_path, db_path):
        """Initialize analyzer for TOP 50 securities only"""
        self.top50_csv_path = top50_csv_path
        self.watchlist_path = watchlist_path
        self.db_path = db_path
        self.top50_df = None
        self.watchlist_df = None
        self.conn = None
        self.detailed_metrics = None
        
    def load_data(self):
        """Load TOP 50 data and connect to database"""
        print("Loading TOP 50 securities data...")
        
        # Load TOP 50 results
        self.top50_df = pd.read_csv(self.top50_csv_path)
        print(f"✅ Loaded TOP 50 securities: {self.top50_df['symbol'].tolist()[:10]}...")
        
        # Load original watchlist for reference
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        
        # Connect to database
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def calculate_detailed_metrics(self):
        """Calculate detailed metrics for TOP 50 securities ONLY"""
        print("📊 Calculating detailed metrics for TOP 50 securities...")
        
        top50_symbols = self.top50_df['symbol'].tolist()
        
        # Get comprehensive data for TOP 50 only
        placeholders = ','.join(['?' for _ in top50_symbols])
        
        # Data quality metrics
        data_quality_query = f"""
        SELECT 
            symbol,
            COUNT(*) as total_records,
            COUNT(DISTINCT DATE(datetime)) as trading_days,
            AVG(volume) as avg_daily_volume,
            COUNT(CASE WHEN volume > 0 THEN 1 END) as volume_records,
            MIN(datetime) as data_start,
            MAX(datetime) as data_end,
            AVG(close) as avg_price,
            COUNT(CASE WHEN close > 0 THEN 1 END) as valid_price_records
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        """
        
        data_quality_df = pd.read_sql_query(data_quality_query, self.conn, params=top50_symbols)
        
        # Volatility and trading metrics (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        volatility_query = f"""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        params = top50_symbols + [start_date.strftime('%Y-%m-%d')]
        recent_data = pd.read_sql_query(volatility_query, self.conn, params=params)
        
        # Calculate detailed metrics for each TOP 50 security
        detailed_metrics = []
        
        for symbol in top50_symbols:
            print(f"   Analyzing {symbol}...")
            
            # Get data quality info
            dq_info = data_quality_df[data_quality_df['symbol'] == symbol]
            
            # Get recent trading data
            symbol_recent = recent_data[recent_data['symbol'] == symbol].copy()
            
            # Get watchlist info
            watchlist_info = self.watchlist_df[self.watchlist_df['Ticker'] == symbol].iloc[0]
            
            # Initialize metrics
            metrics = {
                'symbol': symbol,
                'rank': int(self.top50_df[self.top50_df['symbol'] == symbol]['rank'].iloc[0]),
                'total_score': float(self.top50_df[self.top50_df['symbol'] == symbol]['total_score'].iloc[0]),
                'sector': watchlist_info['Sector'],
                'industry': watchlist_info['Industry'],
                'country': watchlist_info['Country'],
                'market_cap': watchlist_info['Market Cap'],
                'pe_ratio': watchlist_info['P/E'],
                'price': float(watchlist_info['Price']),
                'daily_volume': watchlist_info['Volume']
            }
            
            # Data Quality Metrics
            if not dq_info.empty:
                dq_row = dq_info.iloc[0]
                metrics.update({
                    'total_data_records': int(dq_row['total_records']),
                    'trading_days_coverage': int(dq_row['trading_days']),
                    'avg_daily_volume_15min': float(dq_row['avg_daily_volume']) if not pd.isna(dq_row['avg_daily_volume']) else 0,
                    'data_completeness_pct': float(dq_row['volume_records'] / dq_row['total_records'] * 100) if dq_row['total_records'] > 0 else 0,
                    'data_start_date': dq_row['data_start'],
                    'data_end_date': dq_row['data_end']
                })
            
            # Trading Metrics (if recent data available)
            if len(symbol_recent) >= 10:
                symbol_recent['returns'] = symbol_recent['close'].pct_change()
                symbol_recent['true_range'] = np.maximum(
                    symbol_recent['high'] - symbol_recent['low'],
                    np.maximum(
                        abs(symbol_recent['high'] - symbol_recent['close'].shift(1)),
                        abs(symbol_recent['low'] - symbol_recent['close'].shift(1))
                    )
                )
                
                # Volatility metrics
                daily_vol = symbol_recent['returns'].std() * np.sqrt(252 * 26)  # Annualized
                atr_pct = (symbol_recent['true_range'] / symbol_recent['close']).mean() * 100
                
                # Price movement metrics
                price_range_pct = (symbol_recent['close'].max() - symbol_recent['close'].min()) / symbol_recent['close'].mean() * 100
                avg_spread_pct = ((symbol_recent['high'] - symbol_recent['low']) / symbol_recent['close']).mean() * 100
                
                # Calculate daily volumes by summing 15min intervals by date
                symbol_recent['date'] = pd.to_datetime(symbol_recent['datetime']).dt.date
                daily_volumes = symbol_recent.groupby('date')['volume'].sum()
                avg_daily_volume = daily_volumes.mean()
                
                # Trading consistency (use daily volumes for better consistency measure)
                volume_consistency = 1 - (daily_volumes.std() / daily_volumes.mean()) if daily_volumes.mean() > 0 else 0
                
                # Trend metrics
                symbol_recent['ma5'] = symbol_recent['close'].rolling(5).mean()
                symbol_recent['ma20'] = symbol_recent['close'].rolling(20).mean()
                
                trend_strength = (symbol_recent['ma5'] > symbol_recent['ma20']).sum() / len(symbol_recent.dropna()) * 100
                
                # Max drawdown
                cumulative = (1 + symbol_recent['returns'].fillna(0)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                metrics.update({
                    'return_volatility_annualized': float(daily_vol),
                    'atr_percent': float(atr_pct),
                    'price_range_30d_pct': float(price_range_pct),
                    'avg_daily_spread_pct': float(avg_spread_pct),
                    'volume_consistency': float(volume_consistency),
                    'trend_strength_pct': float(trend_strength),
                    'max_drawdown_30d_pct': float(max_drawdown),
                    'avg_daily_volume': float(avg_daily_volume),
                    'avg_intraday_volume': float(symbol_recent['volume'].mean()),
                    'recent_data_points': len(symbol_recent),
                    'trading_days_recent': len(daily_volumes)
                })
            else:
                # No recent data available
                metrics.update({
                    'return_volatility_annualized': 0,
                    'atr_percent': 0,
                    'price_range_30d_pct': 0,
                    'avg_daily_spread_pct': 0,
                    'volume_consistency': 0,
                    'trend_strength_pct': 0,
                    'max_drawdown_30d_pct': 0,
                    'avg_daily_volume': 0,
                    'avg_intraday_volume': 0,
                    'recent_data_points': 0,
                    'trading_days_recent': 0
                })
            
            # Score breakdown (reconstruct from individual components)
            score_breakdown = self._calculate_score_breakdown(metrics)
            metrics.update(score_breakdown)
            
            detailed_metrics.append(metrics)
        
        self.detailed_metrics = pd.DataFrame(detailed_metrics)
        print(f"✅ Detailed metrics calculated for {len(self.detailed_metrics)} TOP 50 securities")
        
        return self.detailed_metrics
    
    def _calculate_score_breakdown(self, metrics):
        """Calculate transparent score breakdown"""
        # Data Quality Score (25% weight)
        data_completeness = min(100, metrics.get('data_completeness_pct', 0))
        record_count_score = min(100, metrics.get('total_data_records', 0) / 50000 * 100)  # Scale based on record count
        data_quality_score = (data_completeness * 0.7 + record_count_score * 0.3)
        
        # Liquidity Score (20% weight) 
        market_cap_m = self._parse_market_cap(metrics.get('market_cap', '0'))
        volume_numeric = self._parse_volume(metrics.get('daily_volume', '0'))
        
        market_cap_score = min(100, np.log10(max(1, market_cap_m)) / np.log10(400000) * 100)  # Scale to $400B max
        volume_score = min(100, np.log10(max(1, volume_numeric)) / np.log10(100000000) * 100)  # Scale to 100M volume max
        liquidity_score = (market_cap_score * 0.6 + volume_score * 0.4)
        
        # Volatility Score (20% weight)
        vol = metrics.get('return_volatility_annualized', 0)
        if 0.3 <= vol <= 1.5:
            volatility_score = 100
        elif vol < 0.3:
            volatility_score = (vol / 0.3) * 70
        else:
            volatility_score = max(20, 100 - (vol - 1.5) * 30)
        
        # Trend Score (15% weight)
        trend_strength = metrics.get('trend_strength_pct', 50)
        volume_consistency = metrics.get('volume_consistency', 0.5)
        trend_score = (trend_strength + volume_consistency * 50)
        
        # Sector Diversity Score (10% weight) - based on sector rarity
        sector_counts = {'Technology': 42, 'Financial': 62, 'Healthcare': 39, 'Energy': 54, 
                        'Consumer Cyclical': 53, 'Real Estate': 51, 'Basic Materials': 33,
                        'Industrials': 33, 'Communication Services': 29, 'Consumer Defensive': 17, 'Utilities': 13}
        sector = metrics.get('sector', 'Unknown')
        max_count = max(sector_counts.values()) if sector_counts else 1
        sector_count = sector_counts.get(sector, max_count)
        sector_diversity_score = (max_count - sector_count) / max_count * 50 + 50
        
        # Risk Score (10% weight)
        pe = float(metrics.get('pe_ratio', 0)) if metrics.get('pe_ratio') not in ['-', '', None] else 50
        price = metrics.get('price', 20)
        
        pe_score = 100 if 5 <= pe <= 30 else (80 if 1 <= pe <= 50 else 40)
        price_score = 100 if 10 <= price <= 100 else (80 if 5 <= price <= 200 else 50)
        market_cap_risk_score = 100 if market_cap_m >= 1000 else (80 if market_cap_m >= 300 else 60)
        risk_score = (pe_score * 0.3 + price_score * 0.4 + market_cap_risk_score * 0.3)
        
        return {
            'data_quality_score': float(data_quality_score),
            'data_quality_weight': 25.0,
            'data_quality_weighted': float(data_quality_score * 0.25),
            
            'liquidity_score': float(liquidity_score),
            'liquidity_weight': 20.0,
            'liquidity_weighted': float(liquidity_score * 0.20),
            'market_cap_m': float(market_cap_m),
            'volume_numeric': float(volume_numeric),
            
            'volatility_score': float(volatility_score),
            'volatility_weight': 20.0,
            'volatility_weighted': float(volatility_score * 0.20),
            
            'trend_score': float(trend_score),
            'trend_weight': 15.0,
            'trend_weighted': float(trend_score * 0.15),
            
            'sector_diversity_score': float(sector_diversity_score),
            'sector_diversity_weight': 10.0,
            'sector_diversity_weighted': float(sector_diversity_score * 0.10),
            
            'risk_score': float(risk_score),
            'risk_weight': 10.0,
            'risk_weighted': float(risk_score * 0.10),
            
            'calculated_total_score': float(
                data_quality_score * 0.25 + liquidity_score * 0.20 + volatility_score * 0.20 +
                trend_score * 0.15 + sector_diversity_score * 0.10 + risk_score * 0.10
            )
        }
    
    def analyze_top50_sectors(self):
        """Detailed sector analysis for TOP 50"""
        print("\n📊 TOP 50 SECTOR ANALYSIS:")
        
        if self.detailed_metrics is None:
            return
        
        sector_analysis = self.detailed_metrics.groupby('sector').agg({
            'symbol': 'count',
            'total_score': ['mean', 'min', 'max'],
            'return_volatility_annualized': 'mean',
            'market_cap_m': 'mean',
            'data_quality_score': 'mean',
            'liquidity_score': 'mean',
            'rank': 'mean'
        }).round(2)
        
        sector_analysis.columns = ['Count', 'Avg_Score', 'Min_Score', 'Max_Score', 
                                 'Avg_Volatility', 'Avg_MarketCap_M', 'Avg_DataQuality', 
                                 'Avg_Liquidity', 'Avg_Rank']
        
        print(sector_analysis)
        
        return sector_analysis
    
    def create_top50_detailed_dashboard(self, output_file='top50_detailed_analysis_dark.html'):
        """Create detailed analysis dashboard for TOP 50 securities"""
        print("\n📊 Creating detailed TOP 50 analysis dashboard...")
        
        if self.detailed_metrics is None:
            return
        
        df = self.detailed_metrics
        
        # Create comprehensive subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Score Breakdown by Component', 'Market Cap vs Volatility (TOP 50)',
                'Data Quality Distribution', 'Liquidity vs Risk Profile', 
                'Sector Performance Analysis', 'Trading Metrics Heatmap',
                'Volume Consistency Analysis', 'Trend Strength Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        
        # 1. Score breakdown (stacked bar for top 10)
        top10 = df.head(10)
        score_components = ['data_quality_weighted', 'liquidity_weighted', 'volatility_weighted', 
                           'trend_weighted', 'sector_diversity_weighted', 'risk_weighted']
        component_names = ['Data Quality (25%)', 'Liquidity (20%)', 'Volatility (20%)', 
                          'Trend (15%)', 'Sector Div. (10%)', 'Risk (10%)']
        
        for i, (component, name) in enumerate(zip(score_components, component_names)):
            fig.add_trace(
                go.Bar(
                    x=top10['symbol'],
                    y=top10[component],
                    name=name,
                    marker_color=colors[i],
                    text=[f"{val:.0f}" for val in top10[component]],
                    textposition='inside'
                ),
                row=1, col=1
            )
        
        # 2. Market Cap vs Volatility (ALL TOP 50 with labels)
        fig.add_trace(
            go.Scatter(
                x=df['market_cap_m'],
                y=df['return_volatility_annualized'],
                mode='markers+text',
                text=df['symbol'],
                textposition='top center',
                marker=dict(
                    size=12,
                    color=df['total_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Total Score",
                        x=1.02,
                        y=0.8,
                        len=0.15,
                        thickness=12
                    )
                ),
                name="Market Cap vs Volatility (TOP 50)"
            ),
            row=1, col=2
        )
        
        # 3. Data Quality Distribution
        fig.add_trace(
            go.Histogram(
                x=df['data_quality_score'],
                nbinsx=15,
                name="Data Quality",
                marker_color='#4ecdc4',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Liquidity vs Risk
        fig.add_trace(
            go.Scatter(
                x=df['liquidity_score'],
                y=df['risk_score'],
                mode='markers',
                text=df['symbol'],
                marker=dict(
                    size=8,
                    color=df['rank'],
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                name="Liquidity vs Risk"
            ),
            row=2, col=2
        )
        
        # 5. Sector Performance
        sector_scores = df.groupby('sector')['total_score'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(
                x=sector_scores.values,
                y=sector_scores.index,
                orientation='h',
                name="Avg Sector Score",
                marker_color='#ff6b6b',
                text=[f"{score:.0f}" for score in sector_scores.values],
                textposition='outside'
            ),
            row=3, col=1
        )
        
        # 6. Trading Metrics Heatmap (TOP 10)
        metrics_matrix = top10[['return_volatility_annualized', 'atr_percent', 'volume_consistency', 
                               'trend_strength_pct', 'max_drawdown_30d_pct']].T
        
        fig.add_trace(
            go.Heatmap(
                z=metrics_matrix.values,
                x=top10['symbol'],
                y=['Return Vol %', 'ATR %', 'Vol Consistency', 'Trend Strength %', 'Max DD %'],
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(
                    title="Metric Value",
                    x=1.02,
                    y=0.35,
                    len=0.15,
                    thickness=12
                )
            ),
            row=3, col=2
        )
        
        # 7. Volume Consistency
        fig.add_trace(
            go.Scatter(
                x=df['avg_daily_volume'],
                y=df['volume_consistency'],
                mode='markers+text',
                text=df['symbol'],
                textposition='top center',
                marker=dict(size=8, color='#96ceb4'),
                name="Volume Analysis"
            ),
            row=4, col=1
        )
        
        # 8. Trend Strength Distribution
        fig.add_trace(
            go.Histogram(
                x=df['trend_strength_pct'],
                nbinsx=15,
                name="Trend Strength",
                marker_color='#feca57',
                opacity=0.7
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=2000,
            title_text="TOP 50 SECURITIES - DETAILED ANALYSIS & TRANSPARENT SCORING",
            title_font_size=24,
            title_font_color='white',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white', size=11),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.03,
                xanchor="center",
                x=0.5,
                font=dict(color='white', size=10)
            )
        )
        
        # Update subplot backgrounds
        for i in range(1, 5):
            for j in range(1, 3):
                fig.update_xaxes(
                    gridcolor='#404040',
                    tickfont=dict(color='white'),
                    row=i, col=j
                )
                fig.update_yaxes(
                    gridcolor='#404040',
                    tickfont=dict(color='white'),
                    row=i, col=j
                )
        
        # Add axis labels
        fig.update_xaxes(title_text="Security", row=1, col=1)
        fig.update_yaxes(title_text="Weighted Score Points", row=1, col=1)
        fig.update_xaxes(title_text="Market Cap (Millions)", row=1, col=2)
        fig.update_yaxes(title_text="Return Volatility (%)", row=1, col=2)
        fig.update_xaxes(title_text="Data Quality Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Liquidity Score", row=2, col=2)
        fig.update_yaxes(title_text="Risk Score", row=2, col=2)
        fig.update_xaxes(title_text="Average Total Score", row=3, col=1)
        fig.update_xaxes(title_text="Security", row=3, col=2)
        fig.update_xaxes(title_text="Average Daily Volume", row=4, col=1)
        fig.update_yaxes(title_text="Volume Consistency", row=4, col=1)
        fig.update_xaxes(title_text="Trend Strength (%)", row=4, col=2)
        fig.update_yaxes(title_text="Frequency", row=4, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Detailed TOP 50 Dashboard saved: {output_file}")
        
        return fig
    
    def export_detailed_breakdown(self, output_file='top50_detailed_breakdown.csv'):
        """Export comprehensive breakdown of all metrics"""
        print(f"\n📄 Exporting detailed breakdown to {output_file}...")
        
        if self.detailed_metrics is None:
            return
        
        # Select key columns for export
        export_columns = [
            'rank', 'symbol', 'sector', 'industry', 'country',
            'total_score', 'calculated_total_score',
            'data_quality_score', 'data_quality_weighted',
            'liquidity_score', 'liquidity_weighted', 'market_cap_m', 'volume_numeric',
            'volatility_score', 'volatility_weighted', 'return_volatility_annualized',
            'trend_score', 'trend_weighted', 'trend_strength_pct',
            'sector_diversity_score', 'sector_diversity_weighted',
            'risk_score', 'risk_weighted',
            'price', 'pe_ratio', 'market_cap',
            'total_data_records', 'trading_days_coverage', 'data_completeness_pct',
            'avg_daily_volume', 'avg_intraday_volume', 'trading_days_recent',
            'atr_percent', 'max_drawdown_30d_pct', 'volume_consistency'
        ]
        
        export_df = self.detailed_metrics[export_columns].round(2)
        export_df.to_csv(output_file, index=False)
        print(f"✅ Exported detailed breakdown with {len(export_columns)} metrics")
    
    def print_detailed_summary(self):
        """Print comprehensive summary with scoring transparency"""
        if self.detailed_metrics is None:
            return
        
        df = self.detailed_metrics
        
        print("\n" + "="*100)
        print("TOP 50 SECURITIES - DETAILED ANALYSIS & TRANSPARENT SCORING")
        print("="*100)
        
        print(f"\n🏆 TOP 10 WITH DETAILED BREAKDOWN:")
        for i, row in df.head(10).iterrows():
            print(f"\n{row['rank']:2d}. {row['symbol']:6s} - {row['sector']:20s} - Total: {row['total_score']:6.1f}")
            print(f"    Data Quality: {row['data_quality_score']:5.1f} → {row['data_quality_weighted']:5.1f} (25%)")
            print(f"    Liquidity:    {row['liquidity_score']:5.1f} → {row['liquidity_weighted']:5.1f} (20%)")
            print(f"    Volatility:   {row['volatility_score']:5.1f} → {row['volatility_weighted']:5.1f} (20%)")
            print(f"    Trend:        {row['trend_score']:5.1f} → {row['trend_weighted']:5.1f} (15%)")
            print(f"    Sector Div:   {row['sector_diversity_score']:5.1f} → {row['sector_diversity_weighted']:5.1f} (10%)")
            print(f"    Risk:         {row['risk_score']:5.1f} → {row['risk_weighted']:5.1f} (10%)")
            print(f"    Market Cap: ${row['market_cap_m']:,.0f}M | Vol: {row['return_volatility_annualized']:.2f}% | Records: {row['total_data_records']:,}")
        
        print(f"\n📊 SCORING METHODOLOGY TRANSPARENCY:")
        print(f"   Data Quality (25%): Record completeness, volume data, trading days")
        print(f"   Liquidity (20%):    Market cap + trading volume (log-scaled)")
        print(f"   Volatility (20%):   Optimal range 0.3-1.5% daily (sweet spot = 100 points)")
        print(f"   Trend (15%):        Price consistency + volume patterns")
        print(f"   Sector Div (10%):   Portfolio balance (underrepresented sectors favored)")
        print(f"   Risk (10%):         P/E ratios, price levels, market cap stability")
        
        print(f"\n📈 TOP 50 AGGREGATE STATISTICS:")
        print(f"   Total Records Analyzed: {df['total_data_records'].sum():,}")
        print(f"   Avg Market Cap: ${df['market_cap_m'].mean():,.0f}M")
        print(f"   Avg Volatility: {df['return_volatility_annualized'].mean():.2f}%")
        print(f"   Avg Data Quality: {df['data_quality_score'].mean():.1f}/100")
        print(f"   Score Range: {df['total_score'].min():.1f} - {df['total_score'].max():.1f}")
        
        self.analyze_top50_sectors()
    
    def run_detailed_analysis(self):
        """Run complete detailed analysis for TOP 50"""
        print("="*100)
        print("TOP 50 SECURITIES - COMPREHENSIVE DETAILED ANALYSIS")
        print("🔍 Transparent scoring breakdown and in-depth metrics")
        print("="*100)
        
        # Load data
        if not self.load_data():
            return
        
        # Calculate detailed metrics
        self.calculate_detailed_metrics()
        
        # Create detailed dashboard
        self.create_top50_detailed_dashboard()
        
        # Export detailed breakdown
        self.export_detailed_breakdown()
        
        # Print comprehensive summary
        self.print_detailed_summary()
        
        if self.conn:
            self.conn.close()
        
        print("\n🎉 Detailed TOP 50 analysis complete!")
        print("📁 Files generated:")
        print("   - top50_detailed_analysis_dark.html (Interactive dashboard)")
        print("   - top50_detailed_breakdown.csv (Complete metrics)")
    
    def _parse_market_cap(self, mc_str):
        """Parse market cap string to numeric (millions)"""
        if pd.isna(mc_str) or mc_str == '-':
            return 0
        try:
            if 'B' in str(mc_str):
                return float(str(mc_str).replace('B', '')) * 1000
            elif 'M' in str(mc_str):
                return float(str(mc_str).replace('M', ''))
            else:
                return float(mc_str)
        except:
            return 0
    
    def _parse_volume(self, vol_str):
        """Parse volume string to numeric"""
        if pd.isna(vol_str) or vol_str == '-':
            return 0
        try:
            return float(str(vol_str).replace(',', ''))
        except:
            return 0

def main():
    """Main execution"""
    from market_data_database import get_default_database_path
    
    top50_csv_path = "/home/asabaal/asabaal_ventures/repos/investing/top50_trading_candidates.csv"
    watchlist_path = "/home/asabaal/asabaal_ventures/repos/investing/watchlist/WATCHLIST - Sheet1.csv"
    db_path = get_default_database_path()
    
    analyzer = Top50DetailedAnalyzer(top50_csv_path, watchlist_path, db_path)
    analyzer.run_detailed_analysis()

if __name__ == "__main__":
    main()