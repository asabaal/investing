#!/usr/bin/env python3
"""
WATCHLIST-ONLY ANALYSIS

Analyzes ONLY the 426 securities from your watchlist CSV:
- Sector/industry distribution 
- Cross-sector correlations using ONLY watchlist securities
- Volatility metrics for ONLY watchlist securities
- Dark mode visualizations
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import itertools
warnings.filterwarnings('ignore')

class WatchlistOnlyAnalyzer:
    def __init__(self, watchlist_path, db_path):
        """Initialize analyzer for WATCHLIST SECURITIES ONLY"""
        self.watchlist_path = watchlist_path
        self.db_path = db_path
        self.watchlist_df = None
        self.watchlist_tickers = None
        self.conn = None
        
    def load_watchlist(self):
        """Load ONLY watchlist securities"""
        print("Loading WATCHLIST-ONLY data...")
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        self.watchlist_tickers = self.watchlist_df['Ticker'].tolist()
        print(f"✅ Loaded {len(self.watchlist_tickers)} WATCHLIST securities")
        print(f"📋 Sample tickers: {self.watchlist_tickers[:10]}")
        return self.watchlist_df
    
    def connect_database(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def check_watchlist_data_coverage(self):
        """Check data coverage for WATCHLIST securities only"""
        print("\n" + "="*60)
        print("WATCHLIST DATA COVERAGE")
        print("="*60)
        
        # Create placeholder string for watchlist tickers
        placeholders = ','.join(['?' for _ in self.watchlist_tickers])
        
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as record_count,
            MIN(datetime) as first_date,
            MAX(datetime) as last_date,
            COUNT(DISTINCT DATE(datetime)) as trading_days
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        GROUP BY symbol 
        ORDER BY record_count DESC
        """
        
        try:
            coverage_df = pd.read_sql_query(query, self.conn, params=self.watchlist_tickers)
            
            watchlist_with_data = len(coverage_df)
            watchlist_without_data = len(self.watchlist_tickers) - watchlist_with_data
            
            print(f"📊 WATCHLIST Data Coverage:")
            print(f"   Watchlist securities: {len(self.watchlist_tickers)}")
            print(f"   With 15min data: {watchlist_with_data}")
            print(f"   Without data: {watchlist_without_data}")
            print(f"   Total 15min records: {coverage_df['record_count'].sum():,}")
            print(f"   Avg records per symbol: {coverage_df['record_count'].mean():.0f}")
            
            if watchlist_without_data > 0:
                # Find which watchlist securities have no data
                symbols_with_data = set(coverage_df['symbol'].tolist())
                symbols_without_data = [s for s in self.watchlist_tickers if s not in symbols_with_data]
                print(f"   Missing data for: {symbols_without_data[:10]}{'...' if len(symbols_without_data) > 10 else ''}")
            
            print(f"\n🔝 Top 10 WATCHLIST symbols by data volume:")
            for i, row in coverage_df.head(10).iterrows():
                print(f"   {row['symbol']}: {row['record_count']:,} records ({row['trading_days']} days)")
            
            return coverage_df
            
        except Exception as e:
            print(f"❌ Error checking watchlist coverage: {e}")
            return None
    
    def analyze_watchlist_sectors(self):
        """Analyze sector distribution for WATCHLIST securities only"""
        print("\n" + "="*60)
        print("WATCHLIST SECTOR ANALYSIS")
        print("="*60)
        
        # Sector distribution
        sector_counts = self.watchlist_df['Sector'].value_counts()
        print(f"📊 WATCHLIST Sectors ({len(sector_counts)} total):")
        for sector, count in sector_counts.items():
            percentage = (count / len(self.watchlist_df)) * 100
            print(f"   {sector}: {count} securities ({percentage:.1f}%)")
        
        # Top industries within WATCHLIST
        industry_counts = self.watchlist_df['Industry'].value_counts()
        print(f"\n🏭 Top 15 WATCHLIST Industries:")
        for industry, count in industry_counts.head(15).items():
            percentage = (count / len(self.watchlist_df)) * 100
            print(f"   {industry}: {count} ({percentage:.1f}%)")
        
        return sector_counts, industry_counts
    
    def calculate_watchlist_correlations(self, days_back=30):
        """Calculate correlations ONLY between WATCHLIST securities"""
        print(f"\n📈 Calculating WATCHLIST-ONLY correlations (last {days_back} days)...")
        
        # Get recent data for WATCHLIST securities only
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        placeholders = ','.join(['?' for _ in self.watchlist_tickers])
        query = f"""
        SELECT symbol, datetime as timestamp, close 
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        try:
            params = self.watchlist_tickers + [start_date.strftime('%Y-%m-%d')]
            price_data = pd.read_sql_query(query, self.conn, params=params)
            
            if len(price_data) == 0:
                print("❌ No price data found for WATCHLIST securities")
                return None
            
            print(f"📊 Retrieved {len(price_data):,} price records for WATCHLIST securities")
            print(f"📈 Symbols with data: {price_data['symbol'].nunique()}")
            
            # Remove duplicates and pivot to get symbol columns
            price_data_clean = price_data.drop_duplicates(subset=['timestamp', 'symbol'])
            price_pivot = price_data_clean.pivot(index='timestamp', columns='symbol', values='close')
            
            # Calculate returns
            returns = price_pivot.pct_change().dropna()
            
            # Calculate correlation matrix for WATCHLIST securities
            correlation_matrix = returns.corr()
            
            # Map symbols to sectors
            symbol_sectors = {}
            for symbol in self.watchlist_tickers:
                sector_match = self.watchlist_df[self.watchlist_df['Ticker'] == symbol]
                if not sector_match.empty:
                    symbol_sectors[symbol] = sector_match.iloc[0]['Sector']
            
            # Calculate sector-level correlations within WATCHLIST
            sectors = list(self.watchlist_df['Sector'].unique())
            sector_pairs = list(itertools.combinations(sectors, 2))
            
            sector_correlations = {}
            for sector1, sector2 in sector_pairs:
                sector1_symbols = [s for s in correlation_matrix.index if symbol_sectors.get(s) == sector1]
                sector2_symbols = [s for s in correlation_matrix.index if symbol_sectors.get(s) == sector2]
                
                if sector1_symbols and sector2_symbols:
                    cross_corrs = []
                    for s1 in sector1_symbols:
                        for s2 in sector2_symbols:
                            if s1 in correlation_matrix.index and s2 in correlation_matrix.columns:
                                cross_corrs.append(correlation_matrix.loc[s1, s2])
                    
                    if cross_corrs:
                        avg_correlation = np.mean(cross_corrs)
                        sector_correlations[f"{sector1} vs {sector2}"] = avg_correlation
            
            print("\n🔗 WATCHLIST Cross-Sector Correlations:")
            sorted_corrs = sorted(sector_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for pair, corr in sorted_corrs[:15]:
                print(f"   {pair}: {corr:.3f}")
            
            return {
                'correlation_matrix': correlation_matrix,
                'sector_correlations': sector_correlations,
                'returns': returns,
                'symbol_sectors': symbol_sectors
            }
            
        except Exception as e:
            print(f"❌ Error calculating WATCHLIST correlations: {e}")
            return None
    
    def calculate_watchlist_volatility(self, lookback_days=30):
        """Calculate volatility for WATCHLIST securities only"""
        print(f"\n📊 Calculating WATCHLIST volatility metrics (last {lookback_days} days)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        placeholders = ','.join(['?' for _ in self.watchlist_tickers])
        query = f"""
        SELECT symbol, datetime as timestamp, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        try:
            params = self.watchlist_tickers + [start_date.strftime('%Y-%m-%d')]
            ohlc_data = pd.read_sql_query(query, self.conn, params=params)
            
            if len(ohlc_data) == 0:
                print("❌ No OHLC data found for WATCHLIST")
                return None
            
            symbols_with_data = ohlc_data['symbol'].unique()
            print(f"📈 Calculating volatility for {len(symbols_with_data)} WATCHLIST symbols")
            
            volatility_results = []
            
            for symbol in symbols_with_data:
                symbol_data = ohlc_data[ohlc_data['symbol'] == symbol].copy()
                
                if len(symbol_data) < 10:
                    continue
                
                # Calculate returns and volatility
                symbol_data['returns'] = symbol_data['close'].pct_change()
                symbol_data['true_range'] = np.maximum(
                    symbol_data['high'] - symbol_data['low'],
                    np.maximum(
                        abs(symbol_data['high'] - symbol_data['close'].shift(1)),
                        abs(symbol_data['low'] - symbol_data['close'].shift(1))
                    )
                )
                
                # Volatility metrics
                return_vol = symbol_data['returns'].std() * np.sqrt(252 * 26)  # Annualized
                atr_pct = (symbol_data['true_range'] / symbol_data['close']).mean() * 100
                price_range = (symbol_data['close'].max() - symbol_data['close'].min()) / symbol_data['close'].mean() * 100
                
                # Max drawdown
                cumulative = (1 + symbol_data['returns'].fillna(0)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd = drawdown.min() * 100
                
                # Get sector
                sector = self.watchlist_df[self.watchlist_df['Ticker'] == symbol]['Sector'].iloc[0]
                
                volatility_results.append({
                    'symbol': symbol,
                    'sector': sector,
                    'return_volatility': return_vol,
                    'atr_percent': atr_pct,
                    'price_range': price_range,
                    'max_drawdown': max_dd,
                    'records': len(symbol_data)
                })
            
            volatility_df = pd.DataFrame(volatility_results)
            
            # Sector volatility summary
            print("\n📊 WATCHLIST Volatility by Sector:")
            sector_vol = volatility_df.groupby('sector')[['return_volatility', 'atr_percent', 'max_drawdown']].mean()
            for sector, row in sector_vol.iterrows():
                print(f"   {sector}: Vol={row['return_volatility']:.1f}%, ATR={row['atr_percent']:.2f}%, MaxDD={row['max_drawdown']:.1f}%")
            
            print(f"\n🔥 Most Volatile WATCHLIST Securities:")
            top_volatile = volatility_df.nlargest(10, 'return_volatility')
            for _, row in top_volatile.iterrows():
                print(f"   {row['symbol']} ({row['sector']}): {row['return_volatility']:.1f}% vol, {row['atr_percent']:.2f}% ATR")
            
            return volatility_df
            
        except Exception as e:
            print(f"❌ Error calculating WATCHLIST volatility: {e}")
            return None
    
    def create_dark_mode_dashboard(self, volatility_df=None, correlations=None, output_file='watchlist_only_dark_analysis.html'):
        """Create DARK MODE dashboard for WATCHLIST securities only"""
        print(f"\n📊 Creating DARK MODE WATCHLIST dashboard...")
        
        # Dark theme template
        dark_template = {
            'layout': {
                'paper_bgcolor': '#1e1e1e',
                'plot_bgcolor': '#2d2d2d',
                'font': {'color': '#ffffff'},
                'colorway': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
            }
        }
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('WATCHLIST Sector Distribution', 'Sector Performance (Daily %)', 
                          'Market Cap Distribution', 'Volatility Analysis',
                          'Country Distribution', 'Top Industries',
                          'Sector Correlation Matrix', 'Top Security Pair Correlations'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "bar"}]],
            vertical_spacing=0.12,  # More space between rows
            horizontal_spacing=0.1  # More space between columns
        )
        
        # Parse financial data
        self.watchlist_df['Market_Cap_M'] = self.watchlist_df['Market Cap'].apply(self._parse_market_cap)
        self.watchlist_df['Change_Numeric'] = pd.to_numeric(self.watchlist_df['Change'].str.replace('%', ''), errors='coerce')
        
        # 1. Sector pie chart - show values on chart
        sector_counts = self.watchlist_df['Sector'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sector_counts.index, 
                values=sector_counts.values, 
                name="Sectors",
                hole=0.3,
                textfont=dict(color='white', size=11),
                textinfo='label+percent',
                textposition='auto',
                showlegend=False,
                marker=dict(colors=dark_template['layout']['colorway'][:len(sector_counts)])
            ),
            row=1, col=1
        )
        
        # 2. Sector performance bar
        sector_performance = self.watchlist_df.groupby('Sector')['Change_Numeric'].mean().sort_values(ascending=False)
        colors = ['#ff6b6b' if x >= 0 else '#ff4757' for x in sector_performance.values]
        fig.add_trace(
            go.Bar(
                x=sector_performance.index, 
                y=sector_performance.values,
                name="Sector Performance",
                marker_color=colors,
                text=[f"{x:.1f}%" for x in sector_performance.values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Market cap box plot by sector
        top_sectors = sector_counts.head(6).index
        for i, sector in enumerate(top_sectors):
            sector_data = self.watchlist_df[self.watchlist_df['Sector'] == sector]['Market_Cap_M'].dropna()
            if len(sector_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=sector_data, 
                        name=sector,
                        marker_color=dark_template['layout']['colorway'][i % len(dark_template['layout']['colorway'])]
                    ),
                    row=2, col=1
                )
        
        # 4. Volatility scatter (if available)
        if volatility_df is not None and len(volatility_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=volatility_df['atr_percent'],
                    y=volatility_df['return_volatility'],
                    mode='markers',
                    text=volatility_df['symbol'],
                    marker=dict(
                        size=8,
                        color=volatility_df['max_drawdown'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(
                            title="Max Drawdown %",
                            x=1.02,  # Position right next to the volatility chart
                            y=0.68,   # Align with the volatility chart vertical position
                            len=0.18,  # Match roughly the height of the scatter plot
                            thickness=12
                        )
                    ),
                    name="Volatility"
                ),
                row=2, col=2
            )
        
        # 5. Country pie chart - show values on chart
        country_counts = self.watchlist_df['Country'].value_counts().head(8)
        fig.add_trace(
            go.Pie(
                labels=country_counts.index, 
                values=country_counts.values,
                name="Countries",
                hole=0.3,
                textfont=dict(color='white', size=11),
                textinfo='label+percent',
                textposition='auto',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Top industries bar
        top_industries = self.watchlist_df['Industry'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=top_industries.values,
                y=top_industries.index,
                orientation='h',
                name="Top Industries",
                marker_color='#4ecdc4',
                text=top_industries.values,
                textposition='outside'
            ),
            row=3, col=2
        )
        
        # 7. Sector correlation heatmap
        if correlations is not None and 'correlation_matrix' in correlations:
            # Create sector-level correlation matrix
            symbol_sectors = correlations['symbol_sectors']
            corr_matrix = correlations['correlation_matrix']
            
            # Calculate average correlations between sectors
            sectors = list(set(symbol_sectors.values()))
            sector_corr_matrix = np.zeros((len(sectors), len(sectors)))
            
            for i, sector1 in enumerate(sectors):
                for j, sector2 in enumerate(sectors):
                    if i == j:
                        sector_corr_matrix[i, j] = 1.0
                    else:
                        sector1_symbols = [s for s, sec in symbol_sectors.items() if sec == sector1 and s in corr_matrix.index]
                        sector2_symbols = [s for s, sec in symbol_sectors.items() if sec == sector2 and s in corr_matrix.index]
                        
                        if sector1_symbols and sector2_symbols:
                            cross_corrs = []
                            for s1 in sector1_symbols:
                                for s2 in sector2_symbols:
                                    if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                                        cross_corrs.append(corr_matrix.loc[s1, s2])
                            
                            if cross_corrs:
                                sector_corr_matrix[i, j] = np.mean(cross_corrs)
            
            fig.add_trace(
                go.Heatmap(
                    z=sector_corr_matrix,
                    x=sectors,
                    y=sectors,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=True,
                    colorbar=dict(
                        title="Correlation",
                        x=0.47,   # Position right next to the heatmap
                        y=0.18,   # Align with the heatmap vertical position  
                        len=0.15, # Match roughly the height of the heatmap
                        thickness=12
                    )
                ),
                row=4, col=1
            )
        
        # 8. Top individual security correlations with industry info
        if correlations is not None and 'correlation_matrix' in correlations:
            corr_matrix = correlations['correlation_matrix']
            
            # Create symbol to industry mapping
            symbol_industry = {}
            for symbol in self.watchlist_tickers:
                industry_match = self.watchlist_df[self.watchlist_df['Ticker'] == symbol]
                if not industry_match.empty:
                    symbol_industry[symbol] = industry_match.iloc[0]['Industry']
            
            # Find highest correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.index)):
                for j in range(i+1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.index[i]
                    symbol2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        # Get industries for both symbols
                        industry1 = symbol_industry.get(symbol1, 'Unknown')
                        industry2 = symbol_industry.get(symbol2, 'Unknown')
                        corr_pairs.append((symbol1, symbol2, corr_value, industry1, industry2))
            
            # Sort by absolute correlation and take top 12 for better readability
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_corrs = corr_pairs[:12]
            
            if top_corrs:
                # Create detailed labels with industry info
                pair_labels = []
                hover_text = []
                for pair in top_corrs:
                    symbol1, symbol2, corr_val, ind1, ind2 = pair
                    # Truncate long industry names
                    ind1_short = ind1[:20] + "..." if len(ind1) > 20 else ind1
                    ind2_short = ind2[:20] + "..." if len(ind2) > 20 else ind2
                    
                    pair_labels.append(f"{symbol1}-{symbol2}")
                    hover_text.append(f"{symbol1} ({ind1_short})<br>{symbol2} ({ind2_short})<br>Correlation: {corr_val:.3f}")
                
                corr_values = [pair[2] for pair in top_corrs]
                colors = ['#ff6b6b' if x >= 0 else '#4169e1' for x in corr_values]
                
                fig.add_trace(
                    go.Bar(
                        x=corr_values,
                        y=pair_labels,
                        orientation='h',
                        name="Top Correlations",
                        marker_color=colors,
                        text=[f"{x:.3f}" for x in corr_values],
                        textposition='outside',
                        hovertext=hover_text,
                        hoverinfo='text'
                    ),
                    row=4, col=2
                )
        
        # Update layout for dark mode - clean without overlapping legends
        fig.update_layout(
            height=1800,  # Taller for 4 rows and better spacing
            title_text="WATCHLIST SECURITIES ANALYSIS - DARK MODE",
            title_font_size=24,
            title_font_color='white',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white', size=12),
            showlegend=False,  # No legends to avoid overlap
            margin=dict(l=80, r=80, t=100, b=100)
        )
        
        # Update all subplot backgrounds
        for i in range(1, 5):  # Now 4 rows
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
        fig.update_xaxes(title_text="Sector", row=1, col=2)
        fig.update_yaxes(title_text="Avg Daily Change (%)", row=1, col=2)
        fig.update_yaxes(title_text="Market Cap (Millions)", row=2, col=1)
        fig.update_xaxes(title_text="ATR (%)", row=2, col=2)
        fig.update_yaxes(title_text="Return Volatility (%)", row=2, col=2)
        fig.update_xaxes(title_text="Number of Securities", row=3, col=2)
        fig.update_xaxes(title_text="Sector", row=4, col=1)
        fig.update_yaxes(title_text="Sector", row=4, col=1)
        fig.update_xaxes(title_text="Security Pairs", row=4, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ DARK MODE Dashboard saved: {output_file}")
        
        return fig
    
    def run_watchlist_only_analysis(self):
        """Run complete analysis for WATCHLIST securities only"""
        print("="*80)
        print("WATCHLIST-ONLY ANALYSIS")
        print("📋 Analyzing ONLY the 426 securities from your watchlist")
        print("="*80)
        
        # Load watchlist
        self.load_watchlist()
        
        if not self.connect_database():
            return
        
        # Check coverage for WATCHLIST securities only
        coverage = self.check_watchlist_data_coverage()
        
        # Analyze WATCHLIST sectors
        sector_counts, industry_counts = self.analyze_watchlist_sectors()
        
        # Calculate WATCHLIST correlations
        correlations = self.calculate_watchlist_correlations()
        
        # Calculate WATCHLIST volatility
        volatility = self.calculate_watchlist_volatility()
        
        # Create DARK MODE dashboard
        self.create_dark_mode_dashboard(volatility, correlations)
        
        # Final summary
        print("\n" + "="*60)
        print("WATCHLIST ANALYSIS SUMMARY")
        print("="*60)
        print(f"✅ Watchlist Securities: {len(self.watchlist_tickers)}")
        print(f"✅ With 15min Data: {len(coverage) if coverage is not None else 0}")
        print(f"✅ Total 15min Records: {coverage['record_count'].sum():,}" if coverage is not None else "")
        print(f"✅ Sectors Analyzed: {len(sector_counts)}")
        print(f"✅ Correlation Pairs: {len(correlations['sector_correlations']) if correlations else 0}")
        print(f"✅ Volatility Calculated: {len(volatility) if volatility is not None else 0} securities")
        print(f"✅ Dark Mode Dashboard: watchlist_only_dark_analysis.html")
        
        if self.conn:
            self.conn.close()
        
        print("\n🎉 WATCHLIST-ONLY analysis complete!")
    
    def _parse_market_cap(self, mc_str):
        """Parse market cap string"""
        if pd.isna(mc_str) or mc_str == '-':
            return np.nan
        try:
            if 'B' in mc_str:
                return float(mc_str.replace('B', '')) * 1000
            elif 'M' in mc_str:
                return float(mc_str.replace('M', ''))
            else:
                return float(mc_str)
        except:
            return np.nan

def main():
    """Main execution for WATCHLIST-ONLY analysis"""
    from market_data_database import get_default_database_path
    
    watchlist_path = "/home/asabaal/asabaal_ventures/repos/investing/watchlist/WATCHLIST - Sheet1.csv"
    db_path = get_default_database_path()
    
    analyzer = WatchlistOnlyAnalyzer(watchlist_path, db_path)
    analyzer.run_watchlist_only_analysis()

if __name__ == "__main__":
    main()