#!/usr/bin/env python3
"""
COMPLETE WATCHLIST PORTFOLIO BUILDER

Divides ALL 426 watchlist securities into groups of 10 for daily trading.
Each security assigned to exactly ONE group with maximum sector/industry diversity.

Strategy: 10 trades/day from different groups, rotating through all securities
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import sqlite3
from market_data_database import get_default_database_path
import warnings
warnings.filterwarnings('ignore')

class CompleteWatchlistPortfolioBuilder:
    def __init__(self, watchlist_path, db_path):
        """Initialize with complete watchlist"""
        self.watchlist_path = watchlist_path
        self.db_path = db_path
        self.watchlist_df = None
        self.conn = None
        self.portfolio_groups = []
        
    def load_complete_watchlist(self):
        """Load ALL 426 watchlist securities with basic scoring"""
        print("Loading COMPLETE watchlist for portfolio building...")
        
        # Load watchlist
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        
        # Parse financial metrics
        self.watchlist_df['Market_Cap_M'] = self.watchlist_df['Market Cap'].apply(self._parse_market_cap)
        self.watchlist_df['PE_Numeric'] = pd.to_numeric(self.watchlist_df['P/E'], errors='coerce').fillna(20)
        self.watchlist_df['Price_Numeric'] = pd.to_numeric(self.watchlist_df['Price'], errors='coerce').fillna(50)
        self.watchlist_df['Volume_Numeric'] = pd.to_numeric(
            self.watchlist_df['Volume'].str.replace(',', ''), errors='coerce'
        ).fillna(1000000)
        
        # Connect to database for data quality scoring
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database for data quality assessment")
        except Exception as e:
            print(f"⚠️ Database connection failed: {e}")
            self.conn = None
        
        print(f"✅ Loaded {len(self.watchlist_df)} total watchlist securities")
        print(f"📊 Sectors: {self.watchlist_df['Sector'].nunique()}")
        print(f"🏭 Industries: {self.watchlist_df['Industry'].nunique()}")
        print(f"🌍 Countries: {self.watchlist_df['Country'].nunique()}")
        
        return True
    
    def calculate_basic_scores(self):
        """Calculate basic viability scores for ALL securities"""
        print("📊 Calculating basic viability scores for all securities...")
        
        # Get data quality scores from database
        if self.conn:
            self._add_data_quality_scores()
        else:
            self.watchlist_df['data_quality_score'] = 50  # Default if no DB
        
        # Calculate basic component scores
        self._calculate_liquidity_scores()
        self._calculate_risk_scores()
        self._calculate_sector_diversity_scores()
        
        # Calculate total weighted score
        weights = {
            'data_quality_score': 0.30,     # 30% - most important
            'liquidity_score': 0.25,        # 25% 
            'risk_score': 0.25,             # 25%
            'sector_diversity_score': 0.20   # 20%
        }
        
        self.watchlist_df['total_score'] = sum(
            self.watchlist_df[col] * weight for col, weight in weights.items()
        )
        
        # Rank all securities
        self.watchlist_df = self.watchlist_df.sort_values('total_score', ascending=False).reset_index(drop=True)
        self.watchlist_df['rank'] = self.watchlist_df.index + 1
        
        print(f"✅ Calculated scores for all {len(self.watchlist_df)} securities")
        print(f"🏆 Top 5: {self.watchlist_df.head(5)['Ticker'].tolist()}")
        print(f"📈 Score range: {self.watchlist_df['total_score'].min():.0f} - {self.watchlist_df['total_score'].max():.0f}")
        
        return self.watchlist_df
    
    def _add_data_quality_scores(self):
        """Add data quality scores from database"""
        print("   📈 Assessing data quality from 15min database...")
        
        watchlist_tickers = self.watchlist_df['Ticker'].tolist()
        placeholders = ','.join(['?' for _ in watchlist_tickers])
        
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as total_records,
            COUNT(DISTINCT DATE(datetime)) as trading_days,
            AVG(volume) as avg_volume
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        """
        
        try:
            data_quality_df = pd.read_sql_query(query, self.conn, params=watchlist_tickers)
            
            # Calculate data quality score (0-100)
            if not data_quality_df.empty:
                max_records = data_quality_df['total_records'].max()
                max_days = data_quality_df['trading_days'].max()
                
                data_quality_df['data_quality_score'] = (
                    (data_quality_df['total_records'] / max_records) * 50 +
                    (data_quality_df['trading_days'] / max_days) * 50
                ) * 100 / 100
                
                # Merge back to watchlist
                self.watchlist_df = self.watchlist_df.merge(
                    data_quality_df[['symbol', 'data_quality_score']], 
                    left_on='Ticker', right_on='symbol', how='left'
                )
                self.watchlist_df['data_quality_score'] = self.watchlist_df['data_quality_score'].fillna(25)  # Low score for missing data
                
                print(f"   ✅ Data quality assessed for {len(data_quality_df)} securities")
            else:
                self.watchlist_df['data_quality_score'] = 25
                
        except Exception as e:
            print(f"   ⚠️ Data quality assessment failed: {e}")
            self.watchlist_df['data_quality_score'] = 50
    
    def _calculate_liquidity_scores(self):
        """Calculate liquidity scores"""
        # Market cap scores
        market_cap_scores = np.log10(self.watchlist_df['Market_Cap_M'].fillna(100)) / np.log10(self.watchlist_df['Market_Cap_M'].max()) * 100
        
        # Volume scores  
        volume_scores = np.log10(self.watchlist_df['Volume_Numeric'].fillna(1000000)) / np.log10(self.watchlist_df['Volume_Numeric'].max()) * 100
        
        # Combined liquidity score
        self.watchlist_df['liquidity_score'] = (market_cap_scores * 0.6 + volume_scores * 0.4)
    
    def _calculate_risk_scores(self):
        """Calculate risk profile scores"""
        # P/E ratio scoring
        pe_scores = np.where(
            self.watchlist_df['PE_Numeric'].between(5, 30, inclusive='both'),
            100, np.where(self.watchlist_df['PE_Numeric'].between(1, 50, inclusive='both'), 80, 60)
        )
        
        # Price level scoring  
        price_scores = np.where(
            self.watchlist_df['Price_Numeric'].between(10, 100, inclusive='both'),
            100, np.where(self.watchlist_df['Price_Numeric'].between(5, 200, inclusive='both'), 80, 70)
        )
        
        # Market cap scoring
        market_cap_scores = np.where(
            self.watchlist_df['Market_Cap_M'] >= 1000, 100,
            np.where(self.watchlist_df['Market_Cap_M'] >= 300, 80, 70)
        )
        
        self.watchlist_df['risk_score'] = (pe_scores * 0.3 + price_scores * 0.4 + market_cap_scores * 0.3)
    
    def _calculate_sector_diversity_scores(self):
        """Calculate sector diversity scores"""
        sector_counts = self.watchlist_df['Sector'].value_counts()
        max_count = sector_counts.max()
        sector_scores = {}
        
        for sector, count in sector_counts.items():
            # Favor less represented sectors
            diversity_score = (max_count - count) / max_count * 50 + 50
            sector_scores[sector] = diversity_score
        
        self.watchlist_df['sector_diversity_score'] = self.watchlist_df['Sector'].map(sector_scores)
    
    def create_complete_portfolio_groups(self, group_size=10):
        """Create portfolio groups from ALL watchlist securities"""
        print(f"\n🎯 CREATING PORTFOLIO GROUPS OF {group_size} SECURITIES EACH")
        print(f"📋 Total securities to assign: {len(self.watchlist_df)}")
        print("="*70)
        
        total_groups = len(self.watchlist_df) // group_size
        remaining_securities = len(self.watchlist_df) % group_size
        
        print(f"📊 Will create {total_groups} complete groups of {group_size}")
        if remaining_securities > 0:
            print(f"➕ Plus 1 partial group with {remaining_securities} securities")
        
        # Get unique sectors and industries for diversity tracking
        sectors = self.watchlist_df['Sector'].unique()
        industries = self.watchlist_df['Industry'].unique()
        
        groups = []
        assigned_securities = set()
        
        # Create complete groups of 10
        for group_num in range(1, total_groups + 1):
            print(f"\n🔍 Building Group {group_num}...")
            group_securities = []
            group_sectors = set()
            group_industries = set()
            
            # Available securities for this group
            available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
            
            # Strategy: Sector diversity first, then highest scores within each sector
            sectors_needed = min(len(sectors), group_size)
            
            # First pass: One security per sector (maximize diversity)
            sectors_used = 0
            for sector in sectors:
                if len(group_securities) >= group_size or sectors_used >= sectors_needed:
                    break
                
                # Find best available security in this sector
                sector_available = available[
                    (available['Sector'] == sector) & 
                    (~available['Ticker'].isin([s['ticker'] for s in group_securities]))
                ]
                
                if not sector_available.empty:
                    best_security = sector_available.iloc[0]  # Highest score (already sorted)
                    group_securities.append({
                        'ticker': best_security['Ticker'],
                        'company': best_security['Company'],
                        'sector': best_security['Sector'],
                        'industry': best_security['Industry'],
                        'country': best_security['Country'],
                        'total_score': best_security['total_score'],
                        'rank': best_security['rank'],
                        'market_cap_m': best_security['Market_Cap_M'],
                        'volume': best_security['Volume_Numeric'],
                        'selection_reason': f'Best {sector} sector'
                    })
                    group_sectors.add(sector)
                    group_industries.add(best_security['Industry'])
                    assigned_securities.add(best_security['Ticker'])
                    sectors_used += 1
            
            # Second pass: Fill remaining slots with highest scoring available
            remaining_slots = group_size - len(group_securities)
            for _ in range(remaining_slots):
                available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
                
                if available.empty:
                    break
                
                # Prefer new industries for diversity
                new_industry_available = available[~available['Industry'].isin(group_industries)]
                
                if not new_industry_available.empty:
                    best_security = new_industry_available.iloc[0]
                    reason = f'New industry: {best_security["Industry"][:25]}...'
                else:
                    best_security = available.iloc[0]
                    reason = 'Highest remaining score'
                
                group_securities.append({
                    'ticker': best_security['Ticker'],
                    'company': best_security['Company'],
                    'sector': best_security['Sector'],
                    'industry': best_security['Industry'],
                    'country': best_security['Country'],
                    'total_score': best_security['total_score'],
                    'rank': best_security['rank'],
                    'market_cap_m': best_security['Market_Cap_M'],
                    'volume': best_security['Volume_Numeric'],
                    'selection_reason': reason
                })
                assigned_securities.add(best_security['Ticker'])
                group_industries.add(best_security['Industry'])
            
            # Calculate group statistics
            group_df = pd.DataFrame(group_securities)
            group_stats = {
                'group_number': group_num,
                'securities': group_securities,
                'total_securities': len(group_securities),
                'unique_sectors': len(group_df['sector'].unique()),
                'unique_industries': len(group_df['industry'].unique()),
                'unique_countries': len(group_df['country'].unique()),
                'avg_score': group_df['total_score'].mean(),
                'score_range': f"{group_df['total_score'].min():.0f}-{group_df['total_score'].max():.0f}",
                'avg_market_cap_b': group_df['market_cap_m'].mean() / 1000,
                'sectors': list(group_df['sector'].unique()),
                'market_cap_range_b': f"${group_df['market_cap_m'].min()/1000:.1f}B-${group_df['market_cap_m'].max()/1000:.1f}B"
            }
            
            groups.append(group_stats)
            
            print(f"   ✅ Group {group_num}: {len(group_securities)} securities")
            print(f"      Sectors: {group_stats['unique_sectors']} | Industries: {group_stats['unique_industries']} | Countries: {group_stats['unique_countries']}")
            print(f"      Avg Score: {group_stats['avg_score']:.0f} | Range: {group_stats['score_range']}")
            print(f"      Top securities: {', '.join([s['ticker'] for s in group_securities[:5]])}")
        
        # Handle remaining securities (partial group)
        if remaining_securities > 0:
            remaining_available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
            if not remaining_available.empty:
                group_num = total_groups + 1
                print(f"\n🔍 Building Final Group {group_num} ({remaining_securities} securities)...")
                
                group_securities = []
                for _, security in remaining_available.iterrows():
                    group_securities.append({
                        'ticker': security['Ticker'],
                        'company': security['Company'],
                        'sector': security['Sector'],
                        'industry': security['Industry'],
                        'country': security['Country'],
                        'total_score': security['total_score'],
                        'rank': security['rank'],
                        'market_cap_m': security['Market_Cap_M'],
                        'volume': security['Volume_Numeric'],
                        'selection_reason': 'Remaining securities'
                    })
                    assigned_securities.add(security['Ticker'])
                
                group_df = pd.DataFrame(group_securities)
                group_stats = {
                    'group_number': group_num,
                    'securities': group_securities,
                    'total_securities': len(group_securities),
                    'unique_sectors': len(group_df['sector'].unique()),
                    'unique_industries': len(group_df['industry'].unique()),
                    'unique_countries': len(group_df['country'].unique()),
                    'avg_score': group_df['total_score'].mean(),
                    'score_range': f"{group_df['total_score'].min():.0f}-{group_df['total_score'].max():.0f}",
                    'avg_market_cap_b': group_df['market_cap_m'].mean() / 1000,
                    'sectors': list(group_df['sector'].unique()),
                    'market_cap_range_b': f"${group_df['market_cap_m'].min()/1000:.1f}B-${group_df['market_cap_m'].max()/1000:.1f}B"
                }
                
                groups.append(group_stats)
                
                print(f"   ✅ Final Group {group_num}: {len(group_securities)} securities")
                print(f"      Sectors: {group_stats['unique_sectors']} | Industries: {group_stats['unique_industries']}")
        
        self.portfolio_groups = groups
        
        print(f"\n🎉 PORTFOLIO ASSIGNMENT COMPLETE!")
        print(f"📊 Created {len(groups)} groups covering ALL {len(assigned_securities)} securities")
        print(f"✅ Every security assigned to exactly ONE group")
        
        return groups
    
    def export_complete_portfolio_groups(self, output_file='complete_watchlist_trading_groups.csv'):
        """Export all portfolio groups"""
        print(f"\n📄 Exporting complete portfolio groups...")
        
        export_data = []
        for group in self.portfolio_groups:
            for sec in group['securities']:
                export_data.append({
                    'group_number': group['group_number'],
                    'group_name': f"Trading_Group_{group['group_number']:02d}",
                    'ticker': sec['ticker'],
                    'company': sec['company'],
                    'sector': sec['sector'],
                    'industry': sec['industry'],
                    'country': sec['country'],
                    'total_score': sec['total_score'],
                    'rank': sec['rank'],
                    'market_cap_m': sec['market_cap_m'],
                    'volume': sec['volume'],
                    'selection_reason': sec['selection_reason'],
                    'group_avg_score': group['avg_score'],
                    'group_sectors': group['unique_sectors'],
                    'group_industries': group['unique_industries']
                })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(export_data)} securities across {len(self.portfolio_groups)} groups")
        print(f"📁 File: {output_file}")
        
        return export_df
    
    def create_complete_portfolio_dashboard(self, output_file='complete_watchlist_portfolio_dashboard.html'):
        """Create comprehensive dashboard for all portfolio groups"""
        print(f"\n📊 Creating comprehensive portfolio dashboard...")
        
        if not self.portfolio_groups:
            print("❌ No portfolio groups created yet")
            return
        
        # Prepare data for visualization
        all_securities = []
        group_summaries = []
        
        for group in self.portfolio_groups:
            # Group summary
            group_summaries.append({
                'group_number': group['group_number'],
                'avg_score': group['avg_score'],
                'unique_sectors': group['unique_sectors'],
                'unique_industries': group['unique_industries'],
                'unique_countries': group['unique_countries'],
                'total_securities': group['total_securities'],
                'avg_market_cap_b': group['avg_market_cap_b']
            })
            
            # Individual securities
            for sec in group['securities']:
                sec_data = sec.copy()
                sec_data['group_number'] = group['group_number']
                sec_data['group_avg_score'] = group['avg_score']
                all_securities.append(sec_data)
        
        portfolio_df = pd.DataFrame(all_securities)
        group_df = pd.DataFrame(group_summaries)
        
        # Create comprehensive subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Group Performance Overview', 'Sector Distribution Across All Groups',
                'Score Distribution by Group Tier', 'Market Cap vs Score (Top 100 Securities)',
                'Group Diversity Metrics', 'Geographic Distribution',
                'Industry Concentration Analysis', 'Group Size vs Quality'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
        
        # 1. Group Performance Overview (Top 20 groups)
        top20_groups = group_df.nlargest(20, 'avg_score')
        fig.add_trace(
            go.Bar(
                x=[f"G{g}" for g in top20_groups['group_number']],
                y=top20_groups['avg_score'],
                name="Group Score",
                marker_color='#ff6b6b',
                text=[f"{score:.0f}" for score in top20_groups['avg_score']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Sector Distribution Across All Groups
        sector_counts = portfolio_df['sector'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=sector_counts.values,
                y=sector_counts.index,
                orientation='h',
                name="Securities per Sector",
                marker_color='#4ecdc4',
                text=sector_counts.values,
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Score Distribution by Group Tier
        group_df['tier'] = pd.cut(group_df['avg_score'], bins=5, labels=['Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'])
        tier_counts = group_df['tier'].value_counts()
        fig.add_trace(
            go.Histogram(
                x=group_df['avg_score'],
                nbinsx=15,
                name="Score Distribution",
                marker_color='#45b7d1',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Market Cap vs Score (Top 100)
        top100 = portfolio_df.nlargest(100, 'total_score')
        fig.add_trace(
            go.Scatter(
                x=top100['market_cap_m'] / 1000,  # Convert to billions
                y=top100['total_score'],
                mode='markers+text',
                text=top100['ticker'],
                textposition='top center',
                marker=dict(
                    size=8,
                    color=top100['group_number'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Group Number",
                        x=1.02,
                        y=0.68,
                        len=0.18,
                        thickness=12
                    )
                ),
                name="Top 100 Securities"
            ),
            row=2, col=2
        )
        
        # 5. Group Diversity Metrics (Stacked bar for top 15 groups)
        top15_groups = group_df.head(15)
        
        fig.add_trace(
            go.Bar(
                x=[f"G{g}" for g in top15_groups['group_number']],
                y=top15_groups['unique_sectors'],
                name="Unique Sectors",
                marker_color='#96ceb4'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=[f"G{g}" for g in top15_groups['group_number']],
                y=top15_groups['unique_industries'],
                name="Unique Industries",
                marker_color='#feca57'
            ),
            row=3, col=1
        )
        
        # 6. Geographic Distribution
        country_counts = portfolio_df['country'].value_counts().head(8)
        fig.add_trace(
            go.Pie(
                labels=country_counts.index,
                values=country_counts.values,
                name="Countries",
                hole=0.3,
                textfont=dict(color='white', size=11),
                textinfo='label+percent',
                showlegend=False,
                marker=dict(colors=colors[:len(country_counts)])
            ),
            row=3, col=2
        )
        
        # 7. Industry Concentration Heatmap (Top industries by group)
        top_industries = portfolio_df['industry'].value_counts().head(10).index
        top_groups = group_df.head(10)['group_number'].tolist()
        
        # Create industry-group matrix
        industry_matrix = []
        for industry in top_industries:
            industry_row = []
            for group_num in top_groups:
                count = len(portfolio_df[(portfolio_df['industry'] == industry) & 
                                       (portfolio_df['group_number'] == group_num)])
                industry_row.append(count)
            industry_matrix.append(industry_row)
        
        fig.add_trace(
            go.Heatmap(
                z=industry_matrix,
                x=[f"Group {g}" for g in top_groups],
                y=[ind[:25] + "..." if len(ind) > 25 else ind for ind in top_industries],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(
                    title="Securities Count",
                    x=0.47,
                    y=0.18,
                    len=0.15,
                    thickness=12
                )
            ),
            row=4, col=1
        )
        
        # 8. Group Size vs Quality
        fig.add_trace(
            go.Scatter(
                x=group_df['total_securities'],
                y=group_df['avg_score'],
                mode='markers+text',
                text=[f"G{g}" for g in group_df['group_number']],
                textposition='top center',
                marker=dict(
                    size=12,
                    color=group_df['unique_sectors'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                name="Group Analysis"
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=2000,
            title_text="COMPLETE WATCHLIST PORTFOLIO ANALYSIS - ALL 426 SECURITIES",
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
        fig.update_xaxes(title_text="Top 20 Groups", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=1)
        fig.update_xaxes(title_text="Securities Count", row=1, col=2)
        fig.update_yaxes(title_text="Sector", row=1, col=2)
        fig.update_xaxes(title_text="Group Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Market Cap (Billions)", row=2, col=2)
        fig.update_yaxes(title_text="Total Score", row=2, col=2)
        fig.update_xaxes(title_text="Top 15 Groups", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_xaxes(title_text="Securities in Group", row=4, col=2)
        fig.update_yaxes(title_text="Average Group Score", row=4, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Complete Portfolio Dashboard saved: {output_file}")
        
        return fig
    
    def print_portfolio_summary(self):
        """Print detailed summary of all portfolio groups"""
        print("\n" + "="*80)
        print("COMPLETE WATCHLIST PORTFOLIO SUMMARY")
        print("="*80)
        
        total_securities = sum(len(group['securities']) for group in self.portfolio_groups)
        avg_group_score = np.mean([group['avg_score'] for group in self.portfolio_groups])
        
        print(f"📊 OVERVIEW:")
        print(f"   Total Groups: {len(self.portfolio_groups)}")
        print(f"   Total Securities: {total_securities}")
        print(f"   Securities per Group: {total_securities // len(self.portfolio_groups)}-{total_securities // len(self.portfolio_groups) + 1}")
        print(f"   Average Group Score: {avg_group_score:.0f}")
        
        print(f"\n🎯 TOP 10 GROUPS BY AVERAGE SCORE:")
        sorted_groups = sorted(self.portfolio_groups, key=lambda x: x['avg_score'], reverse=True)
        for i, group in enumerate(sorted_groups[:10], 1):
            top_tickers = ', '.join([s['ticker'] for s in group['securities'][:3]])
            print(f"   {i:2d}. Group {group['group_number']:2d}: Score {group['avg_score']:5.0f} | {group['unique_sectors']} sectors | Top: {top_tickers}...")
        
        print(f"\n📈 GROUP STATISTICS:")
        sector_counts = [group['unique_sectors'] for group in self.portfolio_groups]
        industry_counts = [group['unique_industries'] for group in self.portfolio_groups]
        country_counts = [group['unique_countries'] for group in self.portfolio_groups]
        
        print(f"   Avg Sectors per Group: {np.mean(sector_counts):.1f}")
        print(f"   Avg Industries per Group: {np.mean(industry_counts):.1f}")  
        print(f"   Avg Countries per Group: {np.mean(country_counts):.1f}")
        print(f"   Max Sector Diversity: {max(sector_counts)} sectors in one group")
        print(f"   Max Industry Diversity: {max(industry_counts)} industries in one group")
        
        print(f"\n🌍 GLOBAL DIVERSIFICATION:")
        # Count all countries across portfolio
        all_countries = set()
        for group in self.portfolio_groups:
            for sec in group['securities']:
                all_countries.add(sec['country'])
        print(f"   Total Countries Represented: {len(all_countries)}")
        print(f"   Countries: {', '.join(sorted(all_countries))}")
        
        print(f"\n🏆 DETAILED GROUP BREAKDOWN (Top 15):")
        print("Group | Avg Score | Sectors | Industries | Countries | Top Securities")
        print("------|-----------|---------|------------|-----------|----------------")
        for i, group in enumerate(sorted_groups[:15], 1):
            top_3 = ', '.join([s['ticker'] for s in group['securities'][:3]])
            print(f"  {group['group_number']:2d}  |    {group['avg_score']:5.0f}  |    {group['unique_sectors']:2d}   |     {group['unique_industries']:2d}     |     {group['unique_countries']:2d}     | {top_3}")
    
    def print_detailed_group_analysis(self, num_groups=5):
        """Print detailed analysis for top groups"""
        print(f"\n" + "="*90)
        print(f"DETAILED ANALYSIS - TOP {num_groups} PORTFOLIO GROUPS")
        print("="*90)
        
        sorted_groups = sorted(self.portfolio_groups, key=lambda x: x['avg_score'], reverse=True)
        
        for i, group in enumerate(sorted_groups[:num_groups], 1):
            print(f"\n🎯 GROUP {group['group_number']} - RANK #{i}")
            print("-" * 70)
            print(f"📊 Overview: {group['total_securities']} securities | Score: {group['avg_score']:.0f} | Range: {group['score_range']}")
            print(f"🏢 Diversity: {group['unique_sectors']} sectors | {group['unique_industries']} industries | {group['unique_countries']} countries")
            print(f"💰 Market Cap: {group['market_cap_range_b']}")
            print(f"🌍 Sectors: {', '.join(group['sectors'])}")
            print()
            
            print("Rank | Ticker | Company                          | Sector              | Score | Reason")
            print("-----|--------|----------------------------------|---------------------|-------|--------")
            
            for sec in group['securities']:
                company_short = sec['company'][:32] + "..." if len(sec['company']) > 32 else sec['company']
                sector_short = sec['sector'][:19]
                reason_short = sec['selection_reason'][:20] + "..." if len(sec['selection_reason']) > 20 else sec['selection_reason']
                
                print(f" {sec['rank']:3d} | {sec['ticker']:6s} | {company_short:<32s} | {sector_short:<19s} | {sec['total_score']:5.0f} | {reason_short}")
    
    def run_complete_portfolio_builder(self):
        """Run complete portfolio building for entire watchlist"""
        print("="*80)
        print("COMPLETE WATCHLIST PORTFOLIO BUILDER")
        print("🎯 Dividing ALL 426 securities into trading groups")
        print("="*80)
        
        # Load data
        self.load_complete_watchlist()
        
        # Calculate scores
        self.calculate_basic_scores()
        
        # Create groups
        self.create_complete_portfolio_groups()
        
        # Create comprehensive dashboard
        self.create_complete_portfolio_dashboard()
        
        # Print detailed summary
        self.print_portfolio_summary()
        
        # Print detailed analysis for top groups
        self.print_detailed_group_analysis()
        
        # Export data
        self.export_complete_portfolio_groups()
        
        if self.conn:
            self.conn.close()
        
        print(f"\n🎉 Complete portfolio building finished!")
        print(f"📁 Files generated:")
        print(f"   - complete_watchlist_portfolio_dashboard.html (Interactive dashboard)")
        print(f"   - complete_watchlist_trading_groups.csv (Trading reference)")
        print(f"🚀 Ready for systematic daily trading across ALL securities!")
    
    def _parse_market_cap(self, mc_str):
        """Parse market cap string to numeric (millions)"""
        if pd.isna(mc_str) or mc_str == '-':
            return 100  # Default 100M
        try:
            if 'B' in str(mc_str):
                return float(str(mc_str).replace('B', '')) * 1000
            elif 'M' in str(mc_str):
                return float(str(mc_str).replace('M', ''))
            else:
                return float(mc_str)
        except:
            return 100

def main():
    """Main execution"""
    watchlist_path = "/home/asabaal/asabaal_ventures/repos/investing/watchlist/WATCHLIST - Sheet1.csv"
    db_path = get_default_database_path()
    
    builder = CompleteWatchlistPortfolioBuilder(watchlist_path, db_path)
    builder.run_complete_portfolio_builder()

if __name__ == "__main__":
    main()