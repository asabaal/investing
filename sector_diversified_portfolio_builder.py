#!/usr/bin/env python3
"""
SECTOR-DIVERSIFIED PORTFOLIO BUILDER

Creates optimal groups of 10 securities from different sectors/industries
for daily trading strategies. Each group is designed for maximum diversity
and trading potential.

Strategy: 10 trades/day across different sectors to maximize opportunities
and minimize sector-specific risk.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class SectorDiversifiedPortfolioBuilder:
    def __init__(self, top50_csv_path, watchlist_path):
        """Initialize portfolio builder with TOP 50 data"""
        self.top50_csv_path = top50_csv_path
        self.watchlist_path = watchlist_path
        self.top50_df = None
        self.watchlist_df = None
        self.portfolio_groups = []
        
    def load_data(self):
        """Load TOP 50 and watchlist data"""
        print("Loading TOP 50 securities and watchlist data...")
        
        # Load TOP 50 results (already has all needed columns including Industry)
        self.top50_df = pd.read_csv(self.top50_csv_path)
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        
        # Parse market cap to numeric (millions)
        self.top50_df['market_cap_m'] = self.top50_df['Market Cap'].apply(self._parse_market_cap)
        
        print(f"✅ Loaded {len(self.top50_df)} TOP 50 securities")
        print(f"📊 Sectors available: {self.top50_df['Sector'].nunique()}")
        print(f"🏭 Industries available: {self.top50_df['Industry'].nunique()}")
        
        return True
    
    def analyze_sector_industry_coverage(self):
        """Analyze sector and industry distribution"""
        print("\n" + "="*60)
        print("SECTOR & INDUSTRY ANALYSIS FOR PORTFOLIO BUILDING")
        print("="*60)
        
        # Sector distribution
        sector_counts = self.top50_df['Sector'].value_counts()
        print(f"\n📊 SECTOR DISTRIBUTION:")
        for sector, count in sector_counts.items():
            avg_score = self.top50_df[self.top50_df['Sector'] == sector]['total_score'].mean()
            print(f"   {sector}: {count} securities (avg score: {avg_score:.0f})")
        
        # Industry distribution
        industry_counts = self.top50_df['Industry'].value_counts()
        print(f"\n🏭 TOP 15 INDUSTRIES:")
        for industry, count in industry_counts.head(15).items():
            avg_score = self.top50_df[self.top50_df['Industry'] == industry]['total_score'].mean()
            print(f"   {industry}: {count} securities (avg score: {avg_score:.0f})")
        
        return sector_counts, industry_counts
    
    def create_sector_diversified_groups(self, group_size=10, num_groups=5):
        """Create diversified groups of securities for trading"""
        print(f"\n🎯 CREATING {num_groups} DIVERSIFIED GROUPS OF {group_size} SECURITIES EACH")
        print("="*70)
        
        # Strategy: Maximize sector diversity within each group
        sectors = self.top50_df['Sector'].unique()
        
        groups = []
        used_symbols = set()
        
        for group_num in range(1, num_groups + 1):
            print(f"\n🔍 Building Group {group_num}...")
            
            group_securities = []
            group_sectors = set()
            group_industries = set()
            
            # Sort available securities by score (highest first)
            available = self.top50_df[~self.top50_df['symbol'].isin(used_symbols)].copy()
            available = available.sort_values('total_score', ascending=False)
            
            # First pass: One security per sector (maximize sector diversity)
            for sector in sectors:
                if len(group_securities) >= group_size:
                    break
                    
                sector_securities = available[
                    (available['Sector'] == sector) & 
                    (~available['symbol'].isin([s['symbol'] for s in group_securities]))
                ]
                
                if not sector_securities.empty:
                    best_security = sector_securities.iloc[0]
                    group_securities.append({
                        'symbol': best_security['symbol'],
                        'sector': best_security['Sector'],
                        'industry': best_security['Industry'],
                        'total_score': best_security['total_score'],
                        'rank': best_security['rank'],
                        'volatility': best_security['return_volatility'],
                        'market_cap_m': best_security['market_cap_m'],
                        'selection_reason': f'Best {sector} sector pick'
                    })
                    group_sectors.add(sector)
                    group_industries.add(best_security['Industry'])
            
            # Second pass: Fill remaining slots with highest scoring available securities
            # Prioritize different industries within sectors
            remaining_slots = group_size - len(group_securities)
            used_in_group = {s['symbol'] for s in group_securities}
            
            for _ in range(remaining_slots):
                candidates = available[~available['symbol'].isin(used_in_group)]
                
                if candidates.empty:
                    break
                
                # Prefer securities from new industries
                new_industry_candidates = candidates[~candidates['Industry'].isin(group_industries)]
                
                if not new_industry_candidates.empty:
                    best_security = new_industry_candidates.iloc[0]
                    reason = f'Best new industry pick ({best_security["Industry"][:30]}...)'
                else:
                    best_security = candidates.iloc[0]
                    reason = 'Highest remaining score'
                
                group_securities.append({
                    'symbol': best_security['symbol'],
                    'sector': best_security['Sector'],
                    'industry': best_security['Industry'],
                    'total_score': best_security['total_score'],
                    'rank': best_security['rank'],
                    'volatility': best_security['return_volatility'],
                    'market_cap_m': best_security['market_cap_m'],
                    'selection_reason': reason
                })
                
                used_in_group.add(best_security['symbol'])
                group_industries.add(best_security['Industry'])
            
            # Track used symbols globally
            for sec in group_securities:
                used_symbols.add(sec['symbol'])
            
            # Calculate group statistics
            group_df = pd.DataFrame(group_securities)
            group_stats = {
                'group_number': group_num,
                'securities': group_securities,
                'total_securities': len(group_securities),
                'unique_sectors': len(group_df['sector'].unique()),
                'unique_industries': len(group_df['industry'].unique()),
                'avg_score': group_df['total_score'].mean(),
                'avg_volatility': group_df['volatility'].mean(),
                'score_range': f"{group_df['total_score'].min():.0f} - {group_df['total_score'].max():.0f}",
                'sectors': list(group_df['sector'].unique()),
                'market_cap_range_b': f"${group_df['market_cap_m'].min()/1000:.1f}B - ${group_df['market_cap_m'].max()/1000:.1f}B"
            }
            
            groups.append(group_stats)
            
            # Print group summary
            print(f"   ✅ Group {group_num}: {len(group_securities)} securities")
            print(f"      Sectors: {group_stats['unique_sectors']} | Industries: {group_stats['unique_industries']}")
            print(f"      Avg Score: {group_stats['avg_score']:.0f} | Avg Vol: {group_stats['avg_volatility']:.2f}%")
            print(f"      Securities: {', '.join([s['symbol'] for s in group_securities])}")
        
        self.portfolio_groups = groups
        print(f"\n🎉 Created {len(groups)} diversified trading groups!")
        return groups
    
    def print_detailed_groups(self):
        """Print detailed breakdown of each group"""
        print("\n" + "="*80)
        print("DETAILED GROUP BREAKDOWN - DAILY TRADING PORTFOLIOS")
        print("="*80)
        
        for group in self.portfolio_groups:
            print(f"\n🎯 GROUP {group['group_number']} - DIVERSIFIED TRADING PORTFOLIO")
            print("-" * 60)
            print(f"📊 Overview: {group['total_securities']} securities | {group['unique_sectors']} sectors | {group['unique_industries']} industries")
            print(f"📈 Avg Score: {group['avg_score']:.0f} | Avg Volatility: {group['avg_volatility']:.2f}% | Score Range: {group['score_range']}")
            print(f"💰 Market Cap Range: {group['market_cap_range_b']}")
            print(f"🏢 Sectors: {', '.join(group['sectors'])}")
            print()
            
            print("Rank | Symbol | Sector              | Industry                     | Score | Vol%  | Reason")
            print("-----|--------|---------------------|------------------------------|-------|-------|--------")
            
            for sec in group['securities']:
                sector_short = sec['sector'][:19]
                industry_short = sec['industry'][:28] if sec['industry'] else 'Unknown'
                reason_short = sec['selection_reason'][:25] + "..." if len(sec['selection_reason']) > 25 else sec['selection_reason']
                
                print(f" {sec['rank']:3d} | {sec['symbol']:6s} | {sector_short:<19s} | {industry_short:<28s} | {sec['total_score']:5.0f} | {sec['volatility']:4.2f} | {reason_short}")
    
    def create_portfolio_dashboard(self, output_file='sector_diversified_portfolios_dark.html'):
        """Create comprehensive dashboard for all portfolio groups"""
        print(f"\n📊 Creating portfolio groups dashboard...")
        
        if not self.portfolio_groups:
            print("❌ No portfolio groups created yet")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Group Performance Comparison', 'Sector Distribution Across Groups',
                'Volatility vs Score by Group', 'Market Cap Distribution by Group',
                'Group Diversity Metrics', 'Top Securities by Group'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Colors for groups
        group_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        
        # Prepare data
        all_securities = []
        for group in self.portfolio_groups:
            for sec in group['securities']:
                sec_data = sec.copy()
                sec_data['group'] = f"Group {group['group_number']}"
                sec_data['group_number'] = group['group_number']
                all_securities.append(sec_data)
        
        portfolio_df = pd.DataFrame(all_securities)
        
        # 1. Group Performance Comparison
        group_stats = []
        for group in self.portfolio_groups:
            group_stats.append({
                'Group': f"Group {group['group_number']}",
                'Avg_Score': group['avg_score'],
                'Avg_Volatility': group['avg_volatility'],
                'Unique_Sectors': group['unique_sectors'],
                'Unique_Industries': group['unique_industries']
            })
        
        group_stats_df = pd.DataFrame(group_stats)
        
        fig.add_trace(
            go.Bar(
                x=group_stats_df['Group'],
                y=group_stats_df['Avg_Score'],
                name="Average Score",
                marker_color='#ff6b6b',
                text=[f"{score:.0f}" for score in group_stats_df['Avg_Score']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Sector Distribution
        sector_group_counts = portfolio_df.groupby(['group', 'sector']).size().reset_index(name='count')
        sectors = sector_group_counts['sector'].unique()
        
        for i, sector in enumerate(sectors):
            sector_data = sector_group_counts[sector_group_counts['sector'] == sector]
            fig.add_trace(
                go.Bar(
                    x=sector_data['group'],
                    y=sector_data['count'],
                    name=sector,
                    marker_color=group_colors[i % len(group_colors)]
                ),
                row=1, col=2
            )
        
        # 3. Volatility vs Score by Group
        for i, group_num in enumerate(portfolio_df['group_number'].unique()):
            group_data = portfolio_df[portfolio_df['group_number'] == group_num]
            fig.add_trace(
                go.Scatter(
                    x=group_data['volatility'],
                    y=group_data['total_score'],
                    mode='markers+text',
                    text=group_data['symbol'],
                    textposition='top center',
                    name=f"Group {group_num}",
                    marker=dict(size=10, color=group_colors[i % len(group_colors)])
                ),
                row=2, col=1
            )
        
        # 4. Market Cap Distribution by Group
        for group_num in portfolio_df['group_number'].unique():
            group_data = portfolio_df[portfolio_df['group_number'] == group_num]
            fig.add_trace(
                go.Box(
                    y=group_data['market_cap_m'] / 1000,  # Convert to billions
                    name=f"Group {group_num}",
                    marker_color=group_colors[(group_num-1) % len(group_colors)]
                ),
                row=2, col=2
            )
        
        # 5. Diversity Metrics
        diversity_metrics = ['Avg_Score', 'Unique_Sectors', 'Unique_Industries', 'Avg_Volatility']
        diversity_data = group_stats_df[diversity_metrics].T
        
        fig.add_trace(
            go.Bar(
                x=diversity_metrics,
                y=group_stats_df['Avg_Score'],
                name="Group 1",
                marker_color='#ff6b6b'
            ),
            row=3, col=1
        )
        
        # 6. Top Securities Heatmap
        # Create matrix of top securities by group
        top_securities_matrix = []
        group_names = []
        security_names = []
        
        for group in self.portfolio_groups:
            group_names.append(f"Group {group['group_number']}")
            group_scores = []
            if not security_names:  # First iteration
                security_names = [sec['symbol'] for sec in group['securities']]
                
            for sec in group['securities']:
                group_scores.append(sec['total_score'])
            top_securities_matrix.append(group_scores)
        
        fig.add_trace(
            go.Heatmap(
                z=top_securities_matrix,
                x=security_names,
                y=group_names,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Score",
                    x=1.02,
                    y=0.15,
                    len=0.2,
                    thickness=12
                )
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1800,
            title_text="SECTOR-DIVERSIFIED TRADING PORTFOLIOS - DAILY OPPORTUNITIES",
            title_font_size=24,
            title_font_color='white',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#2d2d2d',
            font=dict(color='white', size=12),
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
        fig.update_xaxes(title_text="Portfolio Group", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=1)
        fig.update_xaxes(title_text="Portfolio Group", row=1, col=2)
        fig.update_yaxes(title_text="Securities Count", row=1, col=2)
        fig.update_xaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Total Score", row=2, col=1)
        fig.update_yaxes(title_text="Market Cap (Billions)", row=2, col=2)
        fig.update_xaxes(title_text="Diversity Metrics", row=3, col=1)
        fig.update_xaxes(title_text="Securities", row=3, col=2)
        fig.update_yaxes(title_text="Portfolio Groups", row=3, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Portfolio Dashboard saved: {output_file}")
        
        return fig
    
    def export_trading_groups(self, output_file='sector_diversified_trading_groups.csv'):
        """Export all groups to CSV for trading reference"""
        print(f"\n📄 Exporting trading groups to {output_file}...")
        
        if not self.portfolio_groups:
            print("❌ No portfolio groups to export")
            return
        
        export_data = []
        for group in self.portfolio_groups:
            for sec in group['securities']:
                export_data.append({
                    'group_number': group['group_number'],
                    'group_name': f"Trading_Group_{group['group_number']}",
                    'symbol': sec['symbol'],
                    'sector': sec['sector'],
                    'industry': sec['industry'],
                    'total_score': sec['total_score'],
                    'rank': sec['rank'],
                    'volatility_pct': sec['volatility'],
                    'market_cap_m': sec['market_cap_m'],
                    'selection_reason': sec['selection_reason'],
                    'group_avg_score': group['avg_score'],
                    'group_unique_sectors': group['unique_sectors'],
                    'group_unique_industries': group['unique_industries']
                })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(export_data)} securities across {len(self.portfolio_groups)} trading groups")
        
        return export_df
    
    def run_portfolio_builder(self, group_size=10, num_groups=5):
        """Run complete portfolio building analysis"""
        print("="*80)
        print("SECTOR-DIVERSIFIED PORTFOLIO BUILDER")
        print("🎯 Creating optimal trading groups for daily opportunities")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Analyze coverage
        self.analyze_sector_industry_coverage()
        
        # Create groups
        self.create_sector_diversified_groups(group_size, num_groups)
        
        # Print detailed breakdown
        self.print_detailed_groups()
        
        # Create dashboard
        self.create_portfolio_dashboard()
        
        # Export data
        self.export_trading_groups()
        
        print(f"\n🎉 Portfolio building complete!")
        print(f"📁 Files generated:")
        print(f"   - sector_diversified_portfolios_dark.html (Interactive dashboard)")
        print(f"   - sector_diversified_trading_groups.csv (Trading reference)")
        print(f"\n🎯 Ready for daily trading with {len(self.portfolio_groups)} diversified groups!")
    
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

def main():
    """Main execution"""
    top50_csv_path = "/home/asabaal/asabaal_ventures/repos/investing/top50_trading_candidates.csv"
    watchlist_path = "/home/asabaal/asabaal_ventures/repos/investing/watchlist/WATCHLIST - Sheet1.csv"
    
    builder = SectorDiversifiedPortfolioBuilder(top50_csv_path, watchlist_path)
    builder.run_portfolio_builder(group_size=10, num_groups=5)

if __name__ == "__main__":
    main()