#!/usr/bin/env python3
"""
GROUP CORRELATION ANALYZER

Analyzes correlations within existing portfolio groups to identify
problem areas and optimize group composition.
"""

import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from market_data_database import get_default_database_path
import warnings
warnings.filterwarnings('ignore')

class GroupCorrelationAnalyzer:
    def __init__(self, groups_csv_path, db_path):
        """Initialize analyzer with existing groups"""
        self.groups_csv_path = groups_csv_path
        self.db_path = db_path
        self.groups_df = None
        self.conn = None
        self.correlation_results = {}
        
    def load_groups_data(self):
        """Load existing portfolio groups"""
        print("Loading existing portfolio groups...")
        
        self.groups_df = pd.read_csv(self.groups_csv_path)
        
        # Connect to database
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
            
        print(f"✅ Loaded {len(self.groups_df)} securities across {self.groups_df['group_number'].nunique()} groups")
        return True
    
    def analyze_group_correlations(self, lookback_days=30):
        """Analyze correlations within each group"""
        print(f"📊 Analyzing intra-group correlations (last {lookback_days} days)...")
        
        # Get recent price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        all_tickers = self.groups_df['ticker'].unique().tolist()
        placeholders = ','.join(['?' for _ in all_tickers])
        
        query = f"""
        SELECT symbol, datetime as timestamp, close 
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        try:
            params = all_tickers + [start_date.strftime('%Y-%m-%d')]
            price_data = pd.read_sql_query(query, self.conn, params=params)
            
            if price_data.empty:
                print("❌ No price data found")
                return None
                
            print(f"📈 Retrieved {len(price_data):,} price records for {price_data['symbol'].nunique()} securities")
            
            # Clean and pivot data
            price_data_clean = price_data.drop_duplicates(subset=['timestamp', 'symbol'])
            price_pivot = price_data_clean.pivot(index='timestamp', columns='symbol', values='close')
            
            # Calculate returns
            returns = price_pivot.pct_change().dropna()
            
            # Analyze each group
            group_analysis = []
            unique_groups = sorted(self.groups_df['group_number'].unique())
            
            for group_num in unique_groups:
                print(f"   Analyzing Group {group_num}...")
                
                group_securities = self.groups_df[self.groups_df['group_number'] == group_num]
                group_tickers = group_securities['ticker'].tolist()
                
                # Filter returns for this group's securities that have data
                available_tickers = [t for t in group_tickers if t in returns.columns]
                
                if len(available_tickers) < 2:
                    print(f"   ⚠️ Group {group_num}: Insufficient data ({len(available_tickers)} securities)")
                    continue
                
                group_returns = returns[available_tickers]
                
                # Calculate correlation matrix for this group
                group_corr = group_returns.corr()
                
                # Calculate statistics
                # Get upper triangle correlations (excluding diagonal)
                mask = np.triu(np.ones_like(group_corr, dtype=bool), k=1)
                upper_corr_values = group_corr.values[mask]
                
                # Remove NaN values
                valid_corrs = upper_corr_values[~np.isnan(upper_corr_values)]
                
                if len(valid_corrs) == 0:
                    print(f"   ⚠️ Group {group_num}: No valid correlations")
                    continue
                
                # Group sector/industry analysis
                sectors = group_securities['sector'].unique()
                industries = group_securities['industry'].unique()
                
                # Find high correlation pairs
                high_corr_pairs = []
                for i in range(len(available_tickers)):
                    for j in range(i+1, len(available_tickers)):
                        corr_val = group_corr.iloc[i, j]
                        if not np.isnan(corr_val) and abs(corr_val) > 0.5:  # High correlation threshold
                            ticker1, ticker2 = available_tickers[i], available_tickers[j]
                            
                            # Get sector/industry info
                            info1 = group_securities[group_securities['ticker'] == ticker1].iloc[0]
                            info2 = group_securities[group_securities['ticker'] == ticker2].iloc[0]
                            
                            high_corr_pairs.append({
                                'ticker1': ticker1,
                                'ticker2': ticker2,
                                'correlation': corr_val,
                                'sector1': info1['sector'],
                                'sector2': info2['sector'],
                                'industry1': info1['industry'],
                                'industry2': info2['industry'],
                                'same_sector': info1['sector'] == info2['sector'],
                                'same_industry': info1['industry'] == info2['industry']
                            })
                
                group_stats = {
                    'group_number': group_num,
                    'total_securities': len(group_securities),
                    'securities_with_data': len(available_tickers),
                    'unique_sectors': len(sectors),
                    'unique_industries': len(industries),
                    'avg_correlation': float(np.mean(valid_corrs)),
                    'max_correlation': float(np.max(valid_corrs)),
                    'min_correlation': float(np.min(valid_corrs)),
                    'std_correlation': float(np.std(valid_corrs)),
                    'high_corr_pairs_count': len(high_corr_pairs),
                    'high_corr_pairs': high_corr_pairs,
                    'correlation_matrix': group_corr,
                    'sectors': list(sectors),
                    'industries': list(industries)
                }
                
                group_analysis.append(group_stats)
                
                print(f"   ✅ Group {group_num}: {len(available_tickers)} securities | Avg Corr: {group_stats['avg_correlation']:.3f} | Max: {group_stats['max_correlation']:.3f}")
            
            self.correlation_results = group_analysis
            print(f"✅ Analyzed {len(group_analysis)} groups")
            
            return group_analysis
            
        except Exception as e:
            print(f"❌ Error analyzing correlations: {e}")
            return None
    
    def identify_problem_groups(self, high_corr_threshold=0.5):
        """Identify groups with problematic correlations"""
        print(f"\n🔍 IDENTIFYING PROBLEM GROUPS (correlation > {high_corr_threshold})...")
        print("="*70)
        
        if not self.correlation_results:
            print("❌ No correlation results available")
            return
        
        problem_groups = []
        
        for group_stats in self.correlation_results:
            group_num = group_stats['group_number']
            high_corr_pairs = group_stats['high_corr_pairs']
            avg_corr = group_stats['avg_correlation']
            max_corr = group_stats['max_correlation']
            
            # Flag problematic groups
            is_problem = False
            reasons = []
            
            if max_corr > high_corr_threshold:
                is_problem = True
                reasons.append(f"Max correlation: {max_corr:.3f}")
            
            if avg_corr > 0.3:
                is_problem = True
                reasons.append(f"High avg correlation: {avg_corr:.3f}")
            
            if len(high_corr_pairs) > 2:
                is_problem = True
                reasons.append(f"{len(high_corr_pairs)} high correlation pairs")
            
            if is_problem:
                problem_groups.append({
                    'group_number': group_num,
                    'reasons': reasons,
                    'stats': group_stats
                })
        
        print(f"🚨 FOUND {len(problem_groups)} PROBLEM GROUPS:")
        print()
        
        for prob_group in problem_groups:
            group_num = prob_group['group_number']
            stats = prob_group['stats']
            
            print(f"🔴 GROUP {group_num}:")
            print(f"   Securities: {stats['securities_with_data']} | Sectors: {stats['unique_sectors']} | Industries: {stats['unique_industries']}")
            print(f"   Avg Corr: {stats['avg_correlation']:.3f} | Max Corr: {stats['max_correlation']:.3f}")
            print(f"   Issues: {', '.join(prob_group['reasons'])}")
            
            # Show high correlation pairs
            if stats['high_corr_pairs']:
                print(f"   High Correlation Pairs:")
                for pair in stats['high_corr_pairs']:
                    same_info = ""
                    if pair['same_sector']:
                        same_info += " (Same Sector)"
                    if pair['same_industry']:
                        same_info += " (Same Industry)"
                    print(f"     {pair['ticker1']} - {pair['ticker2']}: {pair['correlation']:.3f}{same_info}")
            print()
        
        return problem_groups
    
    def print_correlation_summary(self):
        """Print overall correlation summary"""
        print("\n" + "="*70)
        print("PORTFOLIO GROUPS CORRELATION SUMMARY")
        print("="*70)
        
        if not self.correlation_results:
            print("❌ No correlation results available")
            return
        
        # Overall statistics
        all_avg_corrs = [g['avg_correlation'] for g in self.correlation_results]
        all_max_corrs = [g['max_correlation'] for g in self.correlation_results]
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Groups Analyzed: {len(self.correlation_results)}")
        print(f"   Average Intra-Group Correlation: {np.mean(all_avg_corrs):.3f}")
        print(f"   Median Intra-Group Correlation: {np.median(all_avg_corrs):.3f}")
        print(f"   Highest Group Correlation: {np.max(all_max_corrs):.3f}")
        print(f"   Lowest Group Correlation: {np.min(all_avg_corrs):.3f}")
        
        # Top 10 best groups (lowest correlation)
        best_groups = sorted(self.correlation_results, key=lambda x: x['avg_correlation'])[:10]
        print(f"\n🏆 TOP 10 BEST GROUPS (Lowest Correlation):")
        for i, group in enumerate(best_groups, 1):
            print(f"   {i:2d}. Group {group['group_number']:2d}: Avg {group['avg_correlation']:.3f} | Max {group['max_correlation']:.3f} | {group['unique_sectors']} sectors")
        
        # Top 10 worst groups (highest correlation)
        worst_groups = sorted(self.correlation_results, key=lambda x: x['avg_correlation'], reverse=True)[:10]
        print(f"\n🚨 TOP 10 WORST GROUPS (Highest Correlation):")
        for i, group in enumerate(worst_groups, 1):
            print(f"   {i:2d}. Group {group['group_number']:2d}: Avg {group['avg_correlation']:.3f} | Max {group['max_correlation']:.3f} | {group['unique_sectors']} sectors")
    
    def create_correlation_dashboard(self, output_file='group_correlation_analysis.html'):
        """Create correlation analysis dashboard"""
        print(f"\n📊 Creating correlation analysis dashboard...")
        
        if not self.correlation_results:
            print("❌ No correlation results available")
            return
        
        # Prepare data
        group_stats_list = []
        for group in self.correlation_results:
            group_stats_list.append({
                'group_number': group['group_number'],
                'avg_correlation': group['avg_correlation'],
                'max_correlation': group['max_correlation'],
                'std_correlation': group['std_correlation'],
                'unique_sectors': group['unique_sectors'],
                'unique_industries': group['unique_industries'],
                'securities_count': group['securities_with_data'],
                'high_corr_pairs': group['high_corr_pairs_count']
            })
        
        df = pd.DataFrame(group_stats_list)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Average Correlation by Group', 'Max Correlation by Group',
                'Correlation vs Sector Diversity', 'High Correlation Pairs Count',
                'Correlation Distribution', 'Group Quality Assessment'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Average Correlation by Group
        fig.add_trace(
            go.Scatter(
                x=df['group_number'],
                y=df['avg_correlation'],
                mode='markers+lines',
                name='Avg Correlation',
                marker=dict(
                    size=8,
                    color=df['avg_correlation'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(
                        title="Avg Correlation",
                        x=1.02,
                        y=0.8,
                        len=0.25,
                        thickness=12
                    )
                )
            ),
            row=1, col=1
        )
        
        # 2. Max Correlation by Group
        fig.add_trace(
            go.Bar(
                x=df['group_number'],
                y=df['max_correlation'],
                name='Max Correlation',
                marker_color='orange',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Correlation vs Sector Diversity
        fig.add_trace(
            go.Scatter(
                x=df['unique_sectors'],
                y=df['avg_correlation'],
                mode='markers',
                text=[f"G{g}" for g in df['group_number']],
                textposition='top center',
                marker=dict(size=10, color='blue'),
                name='Sectors vs Correlation'
            ),
            row=2, col=1
        )
        
        # 4. High Correlation Pairs Count
        fig.add_trace(
            go.Bar(
                x=df['group_number'],
                y=df['high_corr_pairs'],
                name='High Corr Pairs',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # 5. Correlation Distribution
        fig.add_trace(
            go.Histogram(
                x=df['avg_correlation'],
                nbinsx=20,
                name='Correlation Distribution',
                marker_color='green',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # 6. Group Quality Assessment (correlation vs securities count)
        fig.add_trace(
            go.Scatter(
                x=df['securities_count'],
                y=df['avg_correlation'],
                mode='markers',
                text=[f"G{g}" for g in df['group_number']],
                textposition='top center',
                marker=dict(
                    size=df['high_corr_pairs'] * 3 + 5,
                    color='purple',
                    opacity=0.6
                ),
                name='Size vs Quality'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1500,
            title_text="GROUP CORRELATION ANALYSIS - IDENTIFY PROBLEM AREAS",
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
        fig.update_xaxes(title_text="Group Number", row=1, col=1)
        fig.update_yaxes(title_text="Average Correlation", row=1, col=1)
        fig.update_xaxes(title_text="Group Number", row=1, col=2)
        fig.update_yaxes(title_text="Max Correlation", row=1, col=2)
        fig.update_xaxes(title_text="Unique Sectors", row=2, col=1)
        fig.update_yaxes(title_text="Average Correlation", row=2, col=1)
        fig.update_xaxes(title_text="Group Number", row=2, col=2)
        fig.update_yaxes(title_text="High Correlation Pairs", row=2, col=2)
        fig.update_xaxes(title_text="Average Correlation", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)
        fig.update_xaxes(title_text="Securities Count", row=3, col=2)
        fig.update_yaxes(title_text="Average Correlation", row=3, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Correlation Dashboard saved: {output_file}")
        
        return fig
    
    def run_correlation_analysis(self):
        """Run complete correlation analysis"""
        print("="*70)
        print("GROUP CORRELATION ANALYSIS")
        print("🔍 Analyzing intra-group correlations to identify problems")
        print("="*70)
        
        # Load data
        if not self.load_groups_data():
            return
        
        # Analyze correlations
        self.analyze_group_correlations()
        
        # Identify problems
        self.identify_problem_groups()
        
        # Print summary
        self.print_correlation_summary()
        
        # Create dashboard
        self.create_correlation_dashboard()
        
        if self.conn:
            self.conn.close()
        
        print(f"\n🎉 Correlation analysis complete!")
        print(f"📁 File: group_correlation_analysis.html")

def main():
    """Main execution"""
    groups_csv_path = "/home/asabaal/asabaal_ventures/repos/investing/complete_watchlist_trading_groups.csv"
    db_path = get_default_database_path()
    
    analyzer = GroupCorrelationAnalyzer(groups_csv_path, db_path)
    analyzer.run_correlation_analysis()

if __name__ == "__main__":
    main()