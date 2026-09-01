#!/usr/bin/env python3
"""
TRADING VIABILITY SCORER

Comprehensive scoring system to identify the TOP 50 most viable 
trading candidates from your 426 watchlist securities.

Scoring Criteria:
1. Data Quality (25%) - Consistent data availability
2. Liquidity (20%) - Volume and market cap
3. Volatility (20%) - Optimal volatility for trading
4. Trend Consistency (15%) - Stable price movements
5. Sector Diversity (10%) - Balanced exposure
6. Risk Profile (10%) - Manageable drawdowns

Output: Dark mode dashboard with TOP 50 ranked securities
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

class TradingViabilityScorer:
    def __init__(self, watchlist_path, db_path):
        """Initialize trading viability scorer"""
        self.watchlist_path = watchlist_path
        self.db_path = db_path
        self.watchlist_df = None
        self.conn = None
        self.scores_df = None
        
    def load_data(self):
        """Load watchlist and connect to database"""
        print("Loading watchlist and connecting to database...")
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        
        # Parse financial metrics
        self.watchlist_df['Market_Cap_M'] = self.watchlist_df['Market Cap'].apply(self._parse_market_cap)
        self.watchlist_df['PE_Numeric'] = pd.to_numeric(self.watchlist_df['P/E'], errors='coerce')
        self.watchlist_df['Price_Numeric'] = pd.to_numeric(self.watchlist_df['Price'], errors='coerce')
        self.watchlist_df['Change_Numeric'] = pd.to_numeric(self.watchlist_df['Change'].str.replace('%', ''), errors='coerce')
        self.watchlist_df['Volume_Numeric'] = pd.to_numeric(
            self.watchlist_df['Volume'].str.replace(',', ''), errors='coerce'
        )
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def calculate_data_quality_scores(self):
        """Score based on data availability and consistency"""
        print("📊 Calculating data quality scores...")
        
        watchlist_tickers = self.watchlist_df['Ticker'].tolist()
        placeholders = ','.join(['?' for _ in watchlist_tickers])
        
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as total_records,
            COUNT(DISTINCT DATE(datetime)) as trading_days,
            AVG(volume) as avg_volume,
            COUNT(CASE WHEN volume > 0 THEN 1 END) as volume_records,
            MIN(datetime) as first_date,
            MAX(datetime) as last_date
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        """
        
        data_quality_df = pd.read_sql_query(query, self.conn, params=watchlist_tickers)
        
        # Calculate data quality metrics
        data_quality_df['data_completeness'] = data_quality_df['total_records'] / data_quality_df['total_records'].max()
        data_quality_df['volume_completeness'] = data_quality_df['volume_records'] / data_quality_df['total_records']
        data_quality_df['trading_day_coverage'] = data_quality_df['trading_days'] / data_quality_df['trading_days'].max()
        
        # Combined data quality score (0-100)
        data_quality_df['data_quality_score'] = (
            data_quality_df['data_completeness'] * 40 +
            data_quality_df['volume_completeness'] * 30 +
            data_quality_df['trading_day_coverage'] * 30
        ) * 100
        
        print(f"   Calculated data quality for {len(data_quality_df)} securities")
        return data_quality_df[['symbol', 'data_quality_score', 'total_records', 'avg_volume']]
    
    def calculate_liquidity_scores(self):
        """Score based on volume and market cap"""
        print("💰 Calculating liquidity scores...")
        
        # Market cap scores (log scale for better distribution)
        market_cap_scores = np.log10(self.watchlist_df['Market_Cap_M'].fillna(1)) / np.log10(self.watchlist_df['Market_Cap_M'].max()) * 100
        
        # Volume scores
        volume_scores = np.log10(self.watchlist_df['Volume_Numeric'].fillna(1)) / np.log10(self.watchlist_df['Volume_Numeric'].max()) * 100
        
        # Combined liquidity score
        liquidity_scores = (market_cap_scores * 0.6 + volume_scores * 0.4)
        
        liquidity_df = pd.DataFrame({
            'symbol': self.watchlist_df['Ticker'],
            'liquidity_score': liquidity_scores,
            'market_cap_m': self.watchlist_df['Market_Cap_M'],
            'volume': self.watchlist_df['Volume_Numeric']
        })
        
        print(f"   Calculated liquidity for {len(liquidity_df)} securities")
        return liquidity_df
    
    def calculate_volatility_scores(self, lookback_days=30):
        """Score based on optimal volatility for trading"""
        print("📈 Calculating volatility scores...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        watchlist_tickers = self.watchlist_df['Ticker'].tolist()
        placeholders = ','.join(['?' for _ in watchlist_tickers])
        
        query = f"""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        params = watchlist_tickers + [start_date.strftime('%Y-%m-%d')]
        ohlc_data = pd.read_sql_query(query, self.conn, params=params)
        
        volatility_scores = []
        
        for symbol in watchlist_tickers:
            symbol_data = ohlc_data[ohlc_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                volatility_scores.append({
                    'symbol': symbol,
                    'volatility_score': 0,
                    'return_volatility': 0,
                    'atr_percent': 0,
                    'price_stability': 0
                })
                continue
            
            # Calculate metrics
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['true_range'] = np.maximum(
                symbol_data['high'] - symbol_data['low'],
                np.maximum(
                    abs(symbol_data['high'] - symbol_data['close'].shift(1)),
                    abs(symbol_data['low'] - symbol_data['close'].shift(1))
                )
            )
            
            return_vol = symbol_data['returns'].std() * np.sqrt(252 * 26)  # Annualized
            atr_pct = (symbol_data['true_range'] / symbol_data['close']).mean() * 100
            
            # Price stability (inverse of excessive volatility)
            price_changes = symbol_data['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.05).sum() / len(price_changes)  # >5% moves
            price_stability = max(0, 1 - extreme_moves * 2) * 100
            
            # Optimal volatility scoring (sweet spot: 0.3-1.5% daily vol)
            if 0.3 <= return_vol <= 1.5:
                vol_score = 100
            elif return_vol < 0.3:
                vol_score = (return_vol / 0.3) * 70  # Too low volatility
            else:
                vol_score = max(20, 100 - (return_vol - 1.5) * 30)  # Too high volatility
            
            # Combined volatility score
            volatility_score = (vol_score * 0.6 + price_stability * 0.4)
            
            volatility_scores.append({
                'symbol': symbol,
                'volatility_score': volatility_score,
                'return_volatility': return_vol,
                'atr_percent': atr_pct,
                'price_stability': price_stability
            })
        
        volatility_df = pd.DataFrame(volatility_scores)
        print(f"   Calculated volatility for {len(volatility_df)} securities")
        return volatility_df
    
    def calculate_trend_consistency_scores(self, lookback_days=60):
        """Score based on trend consistency and predictability"""
        print("📊 Calculating trend consistency scores...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        watchlist_tickers = self.watchlist_df['Ticker'].tolist()
        placeholders = ','.join(['?' for _ in watchlist_tickers])
        
        query = f"""
        SELECT symbol, datetime, close
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        params = watchlist_tickers + [start_date.strftime('%Y-%m-%d')]
        price_data = pd.read_sql_query(query, self.conn, params=params)
        
        trend_scores = []
        
        for symbol in watchlist_tickers:
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:
                trend_scores.append({
                    'symbol': symbol,
                    'trend_score': 0,
                    'trend_strength': 0,
                    'trend_consistency': 0
                })
                continue
            
            # Calculate moving averages for trend analysis
            symbol_data['ma_short'] = symbol_data['close'].rolling(window=10).mean()
            symbol_data['ma_long'] = symbol_data['close'].rolling(window=30).mean()
            
            # Trend strength (how often short MA > long MA or vice versa)
            trend_direction = (symbol_data['ma_short'] > symbol_data['ma_long']).dropna()
            trend_changes = (trend_direction != trend_direction.shift(1)).sum()
            trend_consistency = max(0, 1 - (trend_changes / len(trend_direction))) * 100
            
            # Price momentum consistency
            returns = symbol_data['close'].pct_change().dropna()
            momentum_consistency = 1 - abs(returns.autocorr() or 0)  # Less autocorrelation = more predictable
            
            # Combined trend score
            trend_score = (trend_consistency * 0.7 + momentum_consistency * 30)
            
            trend_scores.append({
                'symbol': symbol,
                'trend_score': max(0, trend_score),
                'trend_strength': trend_consistency,
                'trend_consistency': momentum_consistency * 100
            })
        
        trend_df = pd.DataFrame(trend_scores)
        print(f"   Calculated trend consistency for {len(trend_df)} securities")
        return trend_df
    
    def calculate_sector_diversity_scores(self):
        """Score based on sector representation for portfolio balance"""
        print("🏢 Calculating sector diversity scores...")
        
        # Count securities per sector
        sector_counts = self.watchlist_df['Sector'].value_counts()
        
        # Score higher for underrepresented sectors (for diversity)
        max_count = sector_counts.max()
        sector_scores = {}
        
        for sector, count in sector_counts.items():
            # Inverse scoring - smaller sectors get higher scores for diversity
            diversity_score = (max_count - count) / max_count * 50 + 50
            sector_scores[sector] = diversity_score
        
        # Apply scores to securities
        self.watchlist_df['sector_diversity_score'] = self.watchlist_df['Sector'].map(sector_scores)
        
        diversity_df = pd.DataFrame({
            'symbol': self.watchlist_df['Ticker'],
            'sector_diversity_score': self.watchlist_df['sector_diversity_score'],
            'sector': self.watchlist_df['Sector']
        })
        
        print(f"   Calculated sector diversity for {len(diversity_df)} securities")
        return diversity_df
    
    def calculate_risk_profile_scores(self):
        """Score based on manageable risk profile"""
        print("⚠️ Calculating risk profile scores...")
        
        # P/E ratio scoring (avoid extreme valuations)
        pe_scores = np.where(
            self.watchlist_df['PE_Numeric'].between(5, 30, inclusive='both'),
            100,  # Good P/E range
            np.where(
                self.watchlist_df['PE_Numeric'].between(1, 50, inclusive='both'),
                80,   # Acceptable P/E range
                40    # Extreme or missing P/E
            )
        )
        
        # Price level scoring (avoid penny stocks and very expensive stocks)
        price_scores = np.where(
            self.watchlist_df['Price_Numeric'].between(10, 100, inclusive='both'),
            100,  # Good price range
            np.where(
                self.watchlist_df['Price_Numeric'].between(5, 200, inclusive='both'),
                80,   # Acceptable price range
                50    # Too cheap or expensive
            )
        )
        
        # Market cap scoring (prefer mid to large cap)
        market_cap_scores = np.where(
            self.watchlist_df['Market_Cap_M'] >= 1000,  # >= $1B
            100,
            np.where(
                self.watchlist_df['Market_Cap_M'] >= 300,  # >= $300M
                80,
                60    # Small cap
            )
        )
        
        # Combined risk score
        risk_scores = (pe_scores * 0.3 + price_scores * 0.4 + market_cap_scores * 0.3)
        
        risk_df = pd.DataFrame({
            'symbol': self.watchlist_df['Ticker'],
            'risk_score': risk_scores,
            'pe_ratio': self.watchlist_df['PE_Numeric'],
            'price': self.watchlist_df['Price_Numeric'],
            'market_cap_m': self.watchlist_df['Market_Cap_M']
        })
        
        print(f"   Calculated risk profile for {len(risk_df)} securities")
        return risk_df
    
    def combine_scores_and_rank(self, data_quality_df, liquidity_df, volatility_df, 
                               trend_df, diversity_df, risk_df):
        """Combine all scores and rank securities"""
        print("🏆 Combining scores and ranking securities...")
        
        # Merge all scores
        scores_df = self.watchlist_df[['Ticker', 'Sector', 'Industry', 'Country']].copy()
        scores_df = scores_df.rename(columns={'Ticker': 'symbol'})
        
        # Merge score components
        scores_df = scores_df.merge(data_quality_df[['symbol', 'data_quality_score']], on='symbol', how='left')
        scores_df = scores_df.merge(liquidity_df[['symbol', 'liquidity_score']], on='symbol', how='left')
        scores_df = scores_df.merge(volatility_df[['symbol', 'volatility_score', 'return_volatility']], on='symbol', how='left')
        scores_df = scores_df.merge(trend_df[['symbol', 'trend_score']], on='symbol', how='left')
        scores_df = scores_df.merge(diversity_df[['symbol', 'sector_diversity_score']], on='symbol', how='left')
        scores_df = scores_df.merge(risk_df[['symbol', 'risk_score']], on='symbol', how='left')
        
        # Fill missing scores with 0
        score_columns = ['data_quality_score', 'liquidity_score', 'volatility_score', 
                        'trend_score', 'sector_diversity_score', 'risk_score']
        scores_df[score_columns] = scores_df[score_columns].fillna(0)
        
        # Calculate weighted total score
        weights = {
            'data_quality_score': 0.25,    # 25%
            'liquidity_score': 0.20,       # 20%
            'volatility_score': 0.20,      # 20%
            'trend_score': 0.15,           # 15%
            'sector_diversity_score': 0.10, # 10%
            'risk_score': 0.10             # 10%
        }
        
        scores_df['total_score'] = sum(
            scores_df[col] * weight for col, weight in weights.items()
        )
        
        # Rank securities
        scores_df = scores_df.sort_values('total_score', ascending=False).reset_index(drop=True)
        scores_df['rank'] = scores_df.index + 1
        
        # Add financial metrics for context
        financial_data = self.watchlist_df[['Ticker', 'Market Cap', 'P/E', 'Price', 'Volume']].copy()
        financial_data = financial_data.rename(columns={'Ticker': 'symbol'})
        scores_df = scores_df.merge(financial_data, on='symbol', how='left')
        
        self.scores_df = scores_df
        
        print(f"✅ Ranked {len(scores_df)} securities")
        print(f"🥇 Top 5: {scores_df.head(5)['symbol'].tolist()}")
        
        return scores_df
    
    def create_top50_dashboard(self, output_file='top50_trading_candidates_dark.html'):
        """Create comprehensive dashboard for TOP 50 trading candidates"""
        print("📊 Creating TOP 50 trading candidates dashboard...")
        
        if self.scores_df is None:
            print("❌ No scores calculated yet")
            return
        
        top50 = self.scores_df.head(50)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'TOP 50 Total Scores', 'Score Component Breakdown',
                'Sector Distribution (Top 50)', 'Market Cap vs Volatility',
                'Risk vs Liquidity Profile', 'Data Quality Analysis'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "box"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Dark theme colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        
        # 1. Total scores bar chart
        fig.add_trace(
            go.Bar(
                x=top50['total_score'],
                y=[f"{row['symbol']} ({row['rank']})" for _, row in top50.iterrows()],
                orientation='h',
                name="Total Score",
                marker_color='#ff6b6b',
                text=[f"{score:.1f}" for score in top50['total_score']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Score components (stacked bar for top 10)
        top10 = top50.head(10)
        components = ['data_quality_score', 'liquidity_score', 'volatility_score', 
                     'trend_score', 'sector_diversity_score', 'risk_score']
        component_names = ['Data Quality', 'Liquidity', 'Volatility', 
                          'Trend', 'Sector Div.', 'Risk']
        
        for i, (component, name) in enumerate(zip(components, component_names)):
            fig.add_trace(
                go.Bar(
                    x=top10['symbol'],
                    y=top10[component] * [0.25, 0.20, 0.20, 0.15, 0.10, 0.10][i],  # Apply weights
                    name=name,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
        
        # 3. Sector distribution pie
        sector_counts = top50['Sector'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                name="Sectors",
                hole=0.3,
                textfont=dict(color='white', size=11),
                textinfo='label+percent',
                showlegend=False,
                marker=dict(colors=colors[:len(sector_counts)])
            ),
            row=2, col=1
        )
        
        # 4. Market Cap vs Volatility scatter
        market_caps = [self._parse_market_cap(mc) for mc in top50['Market Cap']]
        fig.add_trace(
            go.Scatter(
                x=market_caps,
                y=top50['return_volatility'],
                mode='markers+text',
                text=top50['symbol'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=top50['total_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Total Score",
                        x=1.02,
                        y=0.45,
                        len=0.2,
                        thickness=12
                    )
                ),
                name="Market Cap vs Vol"
            ),
            row=2, col=2
        )
        
        # 5. Risk vs Liquidity
        fig.add_trace(
            go.Scatter(
                x=top50['risk_score'],
                y=top50['liquidity_score'],
                mode='markers+text',
                text=top50['symbol'],
                textposition='top center',
                marker=dict(size=8, color='#4ecdc4'),
                name="Risk vs Liquidity"
            ),
            row=3, col=1
        )
        
        # 6. Data quality box plots by sector
        top_sectors = top50['Sector'].value_counts().head(5).index
        for i, sector in enumerate(top_sectors):
            sector_data = top50[top50['Sector'] == sector]
            fig.add_trace(
                go.Box(
                    y=sector_data['data_quality_score'],
                    name=sector,
                    marker_color=colors[i % len(colors)]
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="TOP 50 TRADING CANDIDATES - COMPREHENSIVE ANALYSIS",
            title_font_size=24,
            title_font_color='white',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white', size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(color='white')
            )
        )
        
        # Update subplot backgrounds
        for i in range(1, 4):
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
        fig.update_xaxes(title_text="Total Score", row=1, col=1)
        fig.update_yaxes(title_text="Security (Rank)", row=1, col=1)
        fig.update_xaxes(title_text="Security", row=1, col=2)
        fig.update_yaxes(title_text="Weighted Score", row=1, col=2)
        fig.update_xaxes(title_text="Market Cap (M)", row=2, col=2)
        fig.update_yaxes(title_text="Return Volatility (%)", row=2, col=2)
        fig.update_xaxes(title_text="Risk Score", row=3, col=1)
        fig.update_yaxes(title_text="Liquidity Score", row=3, col=1)
        fig.update_yaxes(title_text="Data Quality Score", row=3, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ TOP 50 Dashboard saved: {output_file}")
        
        return fig
    
    def print_top50_summary(self):
        """Print detailed summary of TOP 50 candidates"""
        if self.scores_df is None:
            return
        
        top50 = self.scores_df.head(50)
        
        print("\n" + "="*80)
        print("TOP 50 TRADING CANDIDATES SUMMARY")
        print("="*80)
        
        print(f"\n🏆 TOP 10 CANDIDATES:")
        for i, row in top50.head(10).iterrows():
            print(f"   {row['rank']:2d}. {row['symbol']:6s} ({row['Sector']:20s}) Score: {row['total_score']:5.1f}")
        
        print(f"\n📊 SECTOR BREAKDOWN (Top 50):")
        sector_dist = top50['Sector'].value_counts()
        for sector, count in sector_dist.items():
            pct = (count/50)*100
            print(f"   {sector:25s}: {count:2d} securities ({pct:4.1f}%)")
        
        print(f"\n💰 MARKET CAP DISTRIBUTION:")
        market_caps = [self._parse_market_cap(mc) for mc in top50['Market Cap']]
        market_caps_clean = [mc for mc in market_caps if not pd.isna(mc)]
        if market_caps_clean:
            print(f"   Average: ${np.mean(market_caps_clean):,.0f}M")
            print(f"   Median:  ${np.median(market_caps_clean):,.0f}M")
            print(f"   Range:   ${min(market_caps_clean):,.0f}M - ${max(market_caps_clean):,.0f}M")
        
        print(f"\n📈 SCORE STATISTICS:")
        print(f"   Highest Score: {top50['total_score'].max():.1f} ({top50.iloc[0]['symbol']})")
        print(f"   50th Score:    {top50['total_score'].min():.1f} ({top50.iloc[49]['symbol']})")
        print(f"   Average:       {top50['total_score'].mean():.1f}")
        
        # Export CSV for detailed analysis
        csv_file = 'top50_trading_candidates.csv'
        top50.to_csv(csv_file, index=False)
        print(f"\n📄 Detailed results exported: {csv_file}")
    
    def run_complete_analysis(self):
        """Run complete trading viability analysis"""
        print("="*80)
        print("TRADING VIABILITY SCORING SYSTEM")
        print("🎯 Identifying TOP 50 candidates from 426 watchlist securities")
        print("="*80)
        
        # Load data
        if not self.load_data():
            return
        
        # Calculate all score components
        data_quality_df = self.calculate_data_quality_scores()
        liquidity_df = self.calculate_liquidity_scores()
        volatility_df = self.calculate_volatility_scores()
        trend_df = self.calculate_trend_consistency_scores()
        diversity_df = self.calculate_sector_diversity_scores()
        risk_df = self.calculate_risk_profile_scores()
        
        # Combine and rank
        self.combine_scores_and_rank(
            data_quality_df, liquidity_df, volatility_df,
            trend_df, diversity_df, risk_df
        )
        
        # Create dashboard
        self.create_top50_dashboard()
        
        # Print summary
        self.print_top50_summary()
        
        if self.conn:
            self.conn.close()
        
        print("\n🎉 Trading viability analysis complete!")
    
    def _parse_market_cap(self, mc_str):
        """Parse market cap string to numeric (millions)"""
        if pd.isna(mc_str) or mc_str == '-':
            return np.nan
        try:
            if 'B' in str(mc_str):
                return float(str(mc_str).replace('B', '')) * 1000
            elif 'M' in str(mc_str):
                return float(str(mc_str).replace('M', ''))
            else:
                return float(mc_str)
        except:
            return np.nan

def main():
    """Main execution"""
    from market_data_database import get_default_database_path
    
    watchlist_path = "/home/asabaal/asabaal_ventures/repos/investing/watchlist/WATCHLIST - Sheet1.csv"
    db_path = get_default_database_path()
    
    scorer = TradingViabilityScorer(watchlist_path, db_path)
    scorer.run_complete_analysis()

if __name__ == "__main__":
    main()