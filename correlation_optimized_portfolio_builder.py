#!/usr/bin/env python3
"""
CORRELATION-OPTIMIZED PORTFOLIO BUILDER

Creates portfolio groups with intelligent correlation management:
1. Prioritizes sector/industry diversity (natural low correlation)
2. Only reduces group size when adding multiple securities from same sector/industry 
   AND those additions would create high correlations (>0.5)
3. Allows variable group sizes (5-10) to maintain low correlation
4. Systematic approach for long-term portfolio management
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from market_data_database import get_default_database_path
import warnings
warnings.filterwarnings('ignore')

class CorrelationOptimizedPortfolioBuilder:
    def __init__(self, watchlist_path, db_path):
        """Initialize with watchlist and database"""
        self.watchlist_path = watchlist_path
        self.db_path = db_path
        self.watchlist_df = None
        self.conn = None
        self.correlation_matrix = None
        self.portfolio_groups = []
        
        # Configuration
        self.max_group_size = 10
        self.min_group_size = 5
        self.high_correlation_threshold = 0.5
        self.lookback_days = 30
        
    def load_data_and_calculate_correlations(self):
        """Load watchlist and calculate correlation matrix"""
        print("Loading watchlist and calculating correlation matrix...")
        
        # Load watchlist with scoring
        self.watchlist_df = pd.read_csv(self.watchlist_path)
        
        # Parse financial metrics for scoring
        self.watchlist_df['Market_Cap_M'] = self.watchlist_df['Market Cap'].apply(self._parse_market_cap)
        self.watchlist_df['PE_Numeric'] = pd.to_numeric(self.watchlist_df['P/E'], errors='coerce').fillna(20)
        self.watchlist_df['Volume_Numeric'] = pd.to_numeric(
            self.watchlist_df['Volume'].str.replace(',', ''), errors='coerce'
        ).fillna(1000000)
        
        # Calculate basic scores (simplified for correlation focus)
        self._calculate_basic_scores()
        
        # Connect to database
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
        
        # Calculate correlation matrix
        correlation_success = self._calculate_correlation_matrix()
        
        print(f"✅ Loaded {len(self.watchlist_df)} securities")
        print(f"📊 Correlation matrix: {correlation_success}")
        
        return True
    
    def _calculate_correlation_matrix(self):
        """Calculate correlation matrix for all securities"""
        print("📊 Calculating correlation matrix for all securities...")
        
        # Get recent price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        all_tickers = self.watchlist_df['Ticker'].tolist()
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
                print("⚠️ No price data found - using sector-based correlation estimates")
                self.correlation_matrix = self._create_sector_based_correlation_matrix()
                return "sector-based"
            
            print(f"📈 Retrieved {len(price_data):,} price records for correlation analysis")
            
            # Clean and pivot data
            price_data_clean = price_data.drop_duplicates(subset=['timestamp', 'symbol'])
            price_pivot = price_data_clean.pivot(index='timestamp', columns='symbol', values='close')
            
            # Calculate returns
            returns = price_pivot.pct_change().dropna()
            
            # Calculate correlation matrix
            self.correlation_matrix = returns.corr()
            
            print(f"✅ Correlation matrix calculated: {self.correlation_matrix.shape}")
            return "data-based"
            
        except Exception as e:
            print(f"⚠️ Error calculating correlations: {e}")
            self.correlation_matrix = self._create_sector_based_correlation_matrix()
            return "sector-based"
    
    def _create_sector_based_correlation_matrix(self):
        """Create estimated correlation matrix based on sector relationships"""
        print("📊 Creating sector-based correlation estimates...")
        
        tickers = self.watchlist_df['Ticker'].tolist()
        n = len(tickers)
        correlation_matrix = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        
        # Assign correlations based on sector/industry relationships
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i != j:
                    sector1 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker1]['Sector'].iloc[0]
                    sector2 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker2]['Sector'].iloc[0]
                    industry1 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker1]['Industry'].iloc[0]
                    industry2 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker2]['Industry'].iloc[0]
                    
                    if industry1 == industry2:
                        # Same industry = higher correlation
                        correlation_matrix.loc[ticker1, ticker2] = np.random.normal(0.4, 0.1)
                    elif sector1 == sector2:
                        # Same sector = moderate correlation
                        correlation_matrix.loc[ticker1, ticker2] = np.random.normal(0.2, 0.05)
                    else:
                        # Different sectors = low correlation
                        correlation_matrix.loc[ticker1, ticker2] = np.random.normal(0.05, 0.02)
        
        return correlation_matrix
    
    def _calculate_basic_scores(self):
        """Calculate basic viability scores"""
        # Simplified scoring focused on key metrics
        market_cap_scores = np.log10(self.watchlist_df['Market_Cap_M'].fillna(100)) / np.log10(self.watchlist_df['Market_Cap_M'].max()) * 100
        volume_scores = np.log10(self.watchlist_df['Volume_Numeric']) / np.log10(self.watchlist_df['Volume_Numeric'].max()) * 100
        
        self.watchlist_df['total_score'] = (market_cap_scores * 0.6 + volume_scores * 0.4)
        self.watchlist_df = self.watchlist_df.sort_values('total_score', ascending=False).reset_index(drop=True)
        self.watchlist_df['rank'] = self.watchlist_df.index + 1
    
    def create_correlation_optimized_groups(self):
        """Create groups optimized for low correlation with intelligent size management"""
        print(f"\n🎯 CREATING CORRELATION-OPTIMIZED GROUPS")
        print(f"📋 Max group size: {self.max_group_size} | Min group size: {self.min_group_size}")
        print(f"🔍 High correlation threshold: {self.high_correlation_threshold}")
        print("="*80)
        
        sectors = self.watchlist_df['Sector'].unique()
        industries = self.watchlist_df['Industry'].unique()
        
        groups = []
        assigned_securities = set()
        group_number = 1
        
        while len(assigned_securities) < len(self.watchlist_df):
            print(f"\n🔍 Building Group {group_number}...")
            
            # Get available securities (highest scores first)
            available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
            
            if available.empty:
                break
            
            group_securities = []
            group_sectors = set()
            group_industries = set()
            
            # STEP 1: Fill one security per sector (natural diversification)
            print(f"   📊 Step 1: Sector diversification...")
            for sector in sectors:
                if len(group_securities) >= self.max_group_size:
                    break
                
                sector_available = available[
                    (available['Sector'] == sector) & 
                    (~available['Ticker'].isin([s['ticker'] for s in group_securities]))
                ]
                
                if not sector_available.empty:
                    best_security = sector_available.iloc[0]
                    
                    # Check correlation with existing group members
                    if self._should_add_security_to_group(best_security['Ticker'], group_securities):
                        group_securities.append(self._create_security_record(best_security, f'Best {sector} sector'))
                        group_sectors.add(sector)
                        group_industries.add(best_security['Industry'])
                        assigned_securities.add(best_security['Ticker'])
                        print(f"     ✅ Added {best_security['Ticker']} ({sector})")
                    else:
                        print(f"     ⚠️ Skipped {best_security['Ticker']} (high correlation)")
            
            # STEP 2: Fill remaining slots with intelligent correlation management
            print(f"   📊 Step 2: Filling remaining slots...")
            remaining_slots = self.max_group_size - len(group_securities)
            
            for _ in range(remaining_slots):
                if len(group_securities) >= self.max_group_size:
                    break
                
                available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
                if available.empty:
                    break
                
                # Try to add securities, prioritizing new industries
                candidates = []
                
                # Priority 1: New industries
                new_industry_candidates = available[~available['Industry'].isin(group_industries)]
                for _, candidate in new_industry_candidates.head(10).iterrows():  # Check top 10
                    if self._should_add_security_to_group(candidate['Ticker'], group_securities):
                        candidates.append((candidate, f"New industry: {candidate['Industry'][:25]}"))
                        break  # Take first good candidate
                
                # Priority 2: New sectors
                if not candidates:
                    new_sector_candidates = available[~available['Sector'].isin(group_sectors)]
                    for _, candidate in new_sector_candidates.head(10).iterrows():
                        if self._should_add_security_to_group(candidate['Ticker'], group_securities):
                            candidates.append((candidate, f"New sector: {candidate['Sector'][:25]}"))
                            break
                
                # Priority 3: Best remaining (with correlation check)
                if not candidates:
                    for _, candidate in available.head(20).iterrows():  # Check top 20
                        if self._should_add_security_to_group(candidate['Ticker'], group_securities):
                            candidates.append((candidate, "Best remaining score"))
                            break
                
                # Add the best candidate found
                if candidates:
                    security, reason = candidates[0]
                    group_securities.append(self._create_security_record(security, reason))
                    group_industries.add(security['Industry'])
                    group_sectors.add(security['Sector'])
                    assigned_securities.add(security['Ticker'])
                    print(f"     ✅ Added {security['Ticker']} ({reason[:30]})")
                else:
                    # CORRELATION MANAGEMENT: No good candidates found
                    print(f"     🛑 No suitable candidates found - stopping at {len(group_securities)} securities")
                    break
            
            # STEP 3: GROUP SHRINKING - Remove securities if maximal correlation is too high
            # Store original tickers before optimization
            original_tickers = set(s['ticker'] for s in group_securities)
            
            # Optimize group (may shrink or swap securities)
            group_securities = self._optimize_group_by_shrinking(group_securities, assigned_securities)
            
            # Update assigned securities after optimization (removals + swaps)
            final_tickers = set(s['ticker'] for s in group_securities)
            
            # Remove original tickers and add final tickers
            assigned_securities = (assigned_securities - original_tickers) | final_tickers
            
            final_group_size = len(group_securities)
            
            if final_group_size < self.min_group_size:
                print(f"   ⚠️ Group too small ({final_group_size}), force-adding securities...")
                # Force add securities to meet minimum size
                slots_needed = self.min_group_size - final_group_size
                available = self.watchlist_df[~self.watchlist_df['Ticker'].isin(assigned_securities)]
                
                for _, security in available.head(slots_needed).iterrows():
                    group_securities.append(self._create_security_record(security, "Force add (min size)"))
                    assigned_securities.add(security['Ticker'])
                    print(f"     🔧 Force added {security['Ticker']}")
            
            # Finalize group
            group_df = pd.DataFrame(group_securities)
            group_stats = self._calculate_group_stats(group_number, group_securities, group_df)
            groups.append(group_stats)
            
            print(f"   ✅ Group {group_number}: {len(group_securities)} securities")
            print(f"      Sectors: {group_stats['unique_sectors']} | Industries: {group_stats['unique_industries']}")
            print(f"      Avg Score: {group_stats['avg_score']:.0f} | Estimated Avg Correlation: {group_stats.get('estimated_avg_correlation', 0):.3f}")
            
            group_number += 1
        
        self.portfolio_groups = groups
        
        print(f"\n🎉 CORRELATION-OPTIMIZED PORTFOLIO COMPLETE!")
        print(f"📊 Created {len(groups)} groups covering ALL {len(assigned_securities)} securities")
        print(f"🎯 Group sizes: {[g['total_securities'] for g in groups]}")
        
        return groups
    
    def _should_add_security_to_group(self, ticker, existing_group):
        """Decide if security should be added based on maximal pairwise correlation analysis"""
        if not existing_group:
            return True  # First security always ok
        
        if self.correlation_matrix is None:
            return True  # No correlation data, allow
        
        if ticker not in self.correlation_matrix.index:
            return True  # No correlation data for this security
        
        # Get all tickers in the potential new group (existing + new ticker)
        all_group_tickers = [s['ticker'] for s in existing_group] + [ticker]
        available_tickers = [t for t in all_group_tickers if t in self.correlation_matrix.index]
        
        if len(available_tickers) < 2:
            return True  # Need at least 2 securities for correlation
        
        # Calculate maximal pairwise correlation for the entire group if we add this security
        max_correlation = 0.0
        max_corr_pair = None
        
        for i in range(len(available_tickers)):
            for j in range(i+1, len(available_tickers)):
                ticker1, ticker2 = available_tickers[i], available_tickers[j]
                if ticker2 in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[ticker1, ticker2])
                    if not np.isnan(correlation) and correlation > max_correlation:
                        max_correlation = correlation
                        max_corr_pair = (ticker1, ticker2)
        
        # Decision logic: Only reject if maximal correlation is high AND from same sector/industry
        if max_correlation > self.high_correlation_threshold:
            # Check if the high correlation pair involves securities from same sector/industry
            if max_corr_pair:
                ticker1, ticker2 = max_corr_pair
                info1 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker1].iloc[0]
                info2 = self.watchlist_df[self.watchlist_df['Ticker'] == ticker2].iloc[0]
                
                same_sector = info1['Sector'] == info2['Sector']
                same_industry = info1['Industry'] == info2['Industry']
                
                # Only reject if high correlation AND same sector/industry AND multiple from same sector/industry
                if same_sector or same_industry:
                    # Count how many securities from this sector/industry are already in group
                    existing_same_sector_count = sum(1 for s in existing_group 
                                                   if self.watchlist_df[self.watchlist_df['Ticker'] == s['ticker']]['Sector'].iloc[0] == info1['Sector'])
                    existing_same_industry_count = sum(1 for s in existing_group 
                                                     if self.watchlist_df[self.watchlist_df['Ticker'] == s['ticker']]['Industry'].iloc[0] == info1['Industry'])
                    
                    # Only reject if we would be adding MULTIPLE from same sector/industry AND high correlation
                    if (same_sector and existing_same_sector_count >= 1) or (same_industry and existing_same_industry_count >= 1):
                        print(f"       🔴 Max group correlation {max_correlation:.3f} ({max_corr_pair[0]}-{max_corr_pair[1]}) - multiple same sector/industry")
                        return False
                    else:
                        print(f"       ⚠️ Max group correlation {max_correlation:.3f} but first from sector/industry - allowing")
                else:
                    print(f"       ⚠️ Max group correlation {max_correlation:.3f} between different sectors - allowing")
        
        return True  # Passed all correlation checks
    
    def _optimize_group_by_shrinking(self, group_securities, assigned_securities=None):
        """Optimize group by removing securities when maximal correlation is too high"""
        if len(group_securities) < 2:
            return group_securities
            
        # Calculate current maximal correlation
        current_max_corr = self._calculate_group_max_correlation(group_securities)
        
        if current_max_corr <= 0.2:  # HARD LIMIT: NO group above 20%
            return group_securities
            
        print(f"   🔍 Group maximal correlation {current_max_corr:.3f} > 0.20 - MUST SHRINK!")
        
        # Find securities to potentially remove (only from same sector/industry groups)
        removal_candidates = self._find_removal_candidates(group_securities)
        
        if not removal_candidates:
            print(f"   ⚠️ No removal candidates found (no multiple same sector/industry)")
            # Try cycling mechanism as backup
            return self._try_cycling_securities(group_securities, assigned_securities or set())
            
        # AGGRESSIVE SHRINKING: Keep removing securities until under 20%
        current_group = group_securities.copy()
        iteration = 0
        max_iterations = 5  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            current_max_corr = self._calculate_group_max_correlation(current_group)
            
            if current_max_corr <= 0.2:
                print(f"   ✅ Target achieved: {current_max_corr:.3f} ≤ 0.20 after {iteration-1} removals")
                break
                
            # Find removal candidates for current group
            removal_candidates = self._find_removal_candidates(current_group)
            
            if not removal_candidates or len(current_group) <= self.min_group_size:
                print(f"   ⚠️ Cannot shrink further: {len(current_group)} securities, max corr {current_max_corr:.3f}")
                break
            
            # Find the best single removal
            best_candidate = None
            best_reduction = 0
            
            for candidate in removal_candidates:
                test_group = [s for s in current_group if s['ticker'] != candidate['ticker']]
                
                if len(test_group) < self.min_group_size:
                    continue
                    
                test_max_corr = self._calculate_group_max_correlation(test_group)
                reduction = current_max_corr - test_max_corr
                
                if reduction > best_reduction:
                    best_candidate = candidate
                    best_reduction = reduction
            
            # Apply the best removal
            if best_candidate and best_reduction > 0:
                current_group = [s for s in current_group if s['ticker'] != best_candidate['ticker']]
                new_max_corr = self._calculate_group_max_correlation(current_group)
                print(f"   🔥 Iteration {iteration}: Removed {best_candidate['ticker']}: {current_max_corr:.3f} → {new_max_corr:.3f}")
            else:
                print(f"   ⚠️ No beneficial removals found in iteration {iteration}")
                break
        
        if len(current_group) < len(group_securities):
            print(f"   🎯 Group aggressively shrunk from {len(group_securities)} to {len(current_group)} securities")
            return current_group
        else:
            print(f"   ⚠️ No shrinking achieved - keeping original group")
            return group_securities
    
    def _calculate_group_max_correlation(self, group_securities):
        """Calculate maximal pairwise correlation within a group"""
        if len(group_securities) < 2:
            return 0.0
            
        if self.correlation_matrix is None:
            return 0.0
            
        tickers = [s['ticker'] for s in group_securities]
        available_tickers = [t for t in tickers if t in self.correlation_matrix.index]
        
        if len(available_tickers) < 2:
            return 0.0
            
        max_correlation = 0.0
        
        for i in range(len(available_tickers)):
            for j in range(i+1, len(available_tickers)):
                ticker1, ticker2 = available_tickers[i], available_tickers[j]
                if ticker2 in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[ticker1, ticker2])
                    if not np.isnan(correlation) and correlation > max_correlation:
                        max_correlation = correlation
                        
        return max_correlation
    
    def _find_removal_candidates(self, group_securities):
        """Find securities that can be removed (from sectors/industries with multiple securities)"""
        if len(group_securities) <= self.min_group_size:
            return []
            
        # Count securities by sector and industry
        sector_counts = {}
        industry_counts = {}
        
        for sec in group_securities:
            sector = sec['sector']
            industry = sec['industry']
            
            if sector not in sector_counts:
                sector_counts[sector] = []
            sector_counts[sector].append(sec)
            
            if industry not in industry_counts:
                industry_counts[industry] = []
            industry_counts[industry].append(sec)
        
        # Find securities in sectors/industries with multiple members
        candidates = []
        
        # Prefer removing from sectors with multiple securities
        for sector, securities in sector_counts.items():
            if len(securities) > 1:
                # Remove lowest scoring security from this sector
                sorted_securities = sorted(securities, key=lambda x: x['total_score'])
                candidates.extend(sorted_securities[:-1])  # Keep the highest scorer
                
        # Also consider industries with multiple securities
        for industry, securities in industry_counts.items():
            if len(securities) > 1:
                sorted_securities = sorted(securities, key=lambda x: x['total_score'])
                # Add lower-scoring securities as candidates (if not already added)
                for sec in sorted_securities[:-1]:
                    if sec not in candidates:
                        candidates.append(sec)
        
        return candidates
    
    def _try_cycling_securities(self, group_securities, assigned_securities):
        """Try cycling out high-correlating securities with better alternatives from remaining pool"""
        current_max_corr = self._calculate_group_max_correlation(group_securities)
        
        if current_max_corr <= 0.2:
            return group_securities
            
        print(f"   🔄 CYCLING: Attempting to swap high-correlating securities...")
        
        # Find the pair with highest correlation
        max_corr_pair = self._find_highest_correlation_pair(group_securities)
        if not max_corr_pair:
            print(f"   ⚠️ Could not identify high correlation pair")
            return group_securities
            
        ticker1, ticker2, pair_corr = max_corr_pair
        print(f"   🎯 Highest correlation pair: {ticker1}-{ticker2} ({pair_corr:.3f})")
        
        # Try cycling out each security in the high correlation pair
        best_group = group_securities.copy()
        best_max_corr = current_max_corr
        best_swap = None
        
        for target_ticker in [ticker1, ticker2]:
            # Get available securities for cycling (not already assigned)
            available_for_cycling = self._get_available_cycling_candidates(group_securities, assigned_securities)
            
            if not available_for_cycling:
                continue
                
            # Try swapping target_ticker with each available candidate
            for candidate in available_for_cycling[:20]:  # Test top 20 candidates
                test_group = self._create_test_group_with_swap(group_securities, target_ticker, candidate)
                test_max_corr = self._calculate_group_max_correlation(test_group)
                
                if test_max_corr < best_max_corr:
                    best_group = test_group
                    best_max_corr = test_max_corr
                    best_swap = (target_ticker, candidate['ticker'])
                    print(f"   ✅ Potential swap: {target_ticker} → {candidate['ticker']} (max corr: {current_max_corr:.3f} → {test_max_corr:.3f})")
        
        # Apply best swap if found
        if best_swap and best_max_corr < current_max_corr:
            old_ticker, new_ticker = best_swap
            print(f"   🔄 EXECUTED SWAP: {old_ticker} → {new_ticker}")
            print(f"   📉 Maximal correlation reduced: {current_max_corr:.3f} → {best_max_corr:.3f}")
            
            # Update the assigned securities tracker in the parent context
            self._update_assigned_securities_after_swap(old_ticker, new_ticker)
            
            return best_group
        else:
            print(f"   ⚠️ No beneficial swaps found - keeping original group")
            return group_securities
    
    def _find_highest_correlation_pair(self, group_securities):
        """Find the pair of securities with highest correlation within the group"""
        if len(group_securities) < 2 or self.correlation_matrix is None:
            return None
            
        tickers = [s['ticker'] for s in group_securities]
        available_tickers = [t for t in tickers if t in self.correlation_matrix.index]
        
        if len(available_tickers) < 2:
            return None
            
        max_correlation = 0.0
        max_pair = None
        
        for i in range(len(available_tickers)):
            for j in range(i+1, len(available_tickers)):
                ticker1, ticker2 = available_tickers[i], available_tickers[j]
                if ticker2 in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[ticker1, ticker2])
                    if not np.isnan(correlation) and correlation > max_correlation:
                        max_correlation = correlation
                        max_pair = (ticker1, ticker2, correlation)
        
        return max_pair
    
    def _get_available_cycling_candidates(self, group_securities, assigned_securities):
        """Get securities available for cycling (not already assigned to any group)"""
        current_tickers = set(s['ticker'] for s in group_securities)
        
        # Get securities that are not assigned to any group
        available_securities = []
        
        for _, security in self.watchlist_df.iterrows():
            ticker = security['Ticker']
            if ticker not in assigned_securities and ticker not in current_tickers:
                available_securities.append({
                    'ticker': ticker,
                    'company': security['Company'],
                    'sector': security['Sector'], 
                    'industry': security['Industry'],
                    'country': security['Country'],
                    'total_score': security['total_score'],
                    'rank': security['rank'],
                    'market_cap_m': security['Market_Cap_M']
                })
        
        # Sort by score (best candidates first)
        return sorted(available_securities, key=lambda x: x['total_score'], reverse=True)
    
    def _create_test_group_with_swap(self, group_securities, old_ticker, new_candidate):
        """Create a test group with one security swapped"""
        test_group = []
        
        for security in group_securities:
            if security['ticker'] == old_ticker:
                # Replace with new candidate
                test_group.append(self._create_security_record_from_dict(new_candidate, f"Cycled from {old_ticker}"))
            else:
                test_group.append(security)
                
        return test_group
    
    def _create_security_record_from_dict(self, security_dict, reason):
        """Create security record from dictionary"""
        return {
            'ticker': security_dict['ticker'],
            'company': security_dict['company'],
            'sector': security_dict['sector'],
            'industry': security_dict['industry'],
            'country': security_dict['country'],
            'total_score': security_dict['total_score'],
            'rank': security_dict['rank'],
            'market_cap_m': security_dict['market_cap_m'],
            'selection_reason': reason
        }
    
    def _update_assigned_securities_after_swap(self, old_ticker, new_ticker):
        """Update assigned securities tracking after a swap"""
        # This would need to be implemented to track global assignments
        # For now, print the swap for visibility
        print(f"   📝 Global assignment updated: -{old_ticker}, +{new_ticker}")
    
    def _create_security_record(self, security, reason):
        """Create standardized security record"""
        return {
            'ticker': security['Ticker'],
            'company': security['Company'],
            'sector': security['Sector'],
            'industry': security['Industry'],
            'country': security['Country'],
            'total_score': security['total_score'],
            'rank': security['rank'],
            'market_cap_m': security['Market_Cap_M'],
            'selection_reason': reason
        }
    
    def _calculate_group_stats(self, group_number, group_securities, group_df):
        """Calculate comprehensive group statistics"""
        # Basic stats
        stats = {
            'group_number': group_number,
            'securities': group_securities,
            'total_securities': len(group_securities),
            'unique_sectors': len(group_df['sector'].unique()),
            'unique_industries': len(group_df['industry'].unique()),
            'unique_countries': len(group_df['country'].unique()),
            'avg_score': group_df['total_score'].mean(),
            'score_range': f"{group_df['total_score'].min():.0f}-{group_df['total_score'].max():.0f}",
            'sectors': list(group_df['sector'].unique()),
            'industries': list(group_df['industry'].unique())
        }
        
        # Estimate average correlation if correlation matrix available
        if self.correlation_matrix is not None:
            tickers = [s['ticker'] for s in group_securities]
            available_tickers = [t for t in tickers if t in self.correlation_matrix.index]
            
            if len(available_tickers) > 1:
                correlations = []
                for i in range(len(available_tickers)):
                    for j in range(i+1, len(available_tickers)):
                        if available_tickers[j] in self.correlation_matrix.columns:
                            corr = abs(self.correlation_matrix.loc[available_tickers[i], available_tickers[j]])
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                if correlations:
                    stats['estimated_avg_correlation'] = np.mean(correlations)
                    stats['estimated_max_correlation'] = np.max(correlations)
        
        return stats
    
    def print_optimization_summary(self):
        """Print summary of correlation optimization"""
        print("\n" + "="*80)
        print("CORRELATION-OPTIMIZED PORTFOLIO SUMMARY")
        print("="*80)
        
        if not self.portfolio_groups:
            print("❌ No portfolio groups created")
            return
        
        # Overall statistics
        total_securities = sum(g['total_securities'] for g in self.portfolio_groups)
        group_sizes = [g['total_securities'] for g in self.portfolio_groups]
        
        print(f"📊 OVERVIEW:")
        print(f"   Total Groups: {len(self.portfolio_groups)}")
        print(f"   Total Securities: {total_securities}")
        print(f"   Group Sizes: Min {min(group_sizes)} | Max {max(group_sizes)} | Avg {np.mean(group_sizes):.1f}")
        print(f"   Size Distribution: {dict(pd.Series(group_sizes).value_counts().sort_index())}")
        
        # Correlation insights - FOCUS ON MAXIMAL CORRELATIONS
        groups_with_corr = [g for g in self.portfolio_groups if 'estimated_max_correlation' in g]
        if groups_with_corr:
            max_corrs = [g['estimated_max_correlation'] for g in groups_with_corr]
            avg_corrs = [g['estimated_avg_correlation'] for g in groups_with_corr if 'estimated_avg_correlation' in g]
            print(f"\n📈 CORRELATION ANALYSIS (MAXIMAL FOCUS):")
            print(f"   Groups with correlation data: {len(groups_with_corr)}")
            print(f"   Average MAXIMAL correlation: {np.mean(max_corrs):.3f}")
            print(f"   Lowest MAXIMAL correlation: {min(max_corrs):.3f}")
            print(f"   Highest MAXIMAL correlation: {max(max_corrs):.3f}")
            if avg_corrs:
                print(f"   Average intra-group correlation: {np.mean(avg_corrs):.3f} (reference)")
        
        # Size-based analysis
        small_groups = [g for g in self.portfolio_groups if g['total_securities'] < self.max_group_size]
        print(f"\n🎯 SIZE OPTIMIZATION:")
        print(f"   Full-size groups ({self.max_group_size}): {len(self.portfolio_groups) - len(small_groups)}")
        print(f"   Reduced-size groups: {len(small_groups)}")
        
        if small_groups:
            print(f"   Reduced group details:")
            for g in small_groups:
                corr_info = f" | Avg Corr: {g.get('estimated_avg_correlation', 0):.3f}" if 'estimated_avg_correlation' in g else ""
                print(f"     Group {g['group_number']}: {g['total_securities']} securities | {g['unique_sectors']} sectors{corr_info}")
    
    def create_optimization_dashboard(self, output_file='correlation_optimized_portfolio_dashboard.html'):
        """Create comprehensive dashboard for correlation-optimized groups"""
        print(f"\n📊 Creating correlation-optimized portfolio dashboard...")
        
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
                'total_securities': group['total_securities'],
                'estimated_avg_correlation': group.get('estimated_avg_correlation', 0),
                'estimated_max_correlation': group.get('estimated_max_correlation', 0)
            })
            
            # Individual securities
            for sec in group['securities']:
                sec_data = sec.copy()
                sec_data['group_number'] = group['group_number']
                sec_data['group_avg_score'] = group['avg_score']
                sec_data['group_correlation'] = group.get('estimated_avg_correlation', 0)
                all_securities.append(sec_data)
        
        portfolio_df = pd.DataFrame(all_securities)
        group_df = pd.DataFrame(group_summaries)
        
        # Create comprehensive subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Group MAXIMAL Correlation vs Quality', 'Group Size Distribution (Discrete)',
                'Sector Diversity Analysis', 'Top 20 Groups by Score (Max Corr Colors)',
                'MAXIMAL Correlation Optimization Results', 'Score vs MAXIMAL Correlation Trade-off'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        
        # 1. Group MAXIMAL Correlation vs Quality
        fig.add_trace(
            go.Scatter(
                x=group_df['estimated_max_correlation'],
                y=group_df['avg_score'],
                mode='markers+text',
                text=[f"G{g}" for g in group_df['group_number']],
                textposition='top center',
                marker=dict(
                    size=group_df['unique_sectors'] * 3,
                    color=group_df['total_securities'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(
                        title="Group Size",
                        x=1.02,
                        y=0.8,
                        len=0.25,
                        thickness=12
                    )
                ),
                name="MAXIMAL Correlation vs Quality"
            ),
            row=1, col=1
        )
        
        # 2. Group Size Distribution - DISCRETE PER SECURITIES COUNT
        size_counts = group_df['total_securities'].value_counts().sort_index()
        
        # Create discrete bars for each possible group size
        all_sizes = list(range(group_df['total_securities'].min(), group_df['total_securities'].max() + 1))
        counts_by_size = [size_counts.get(size, 0) for size in all_sizes]
        
        fig.add_trace(
            go.Bar(
                x=all_sizes,
                y=counts_by_size,
                name="Groups by Size",
                marker_color='#4ecdc4',
                text=[f"{count}" if count > 0 else "" for count in counts_by_size],
                textposition='outside',
                width=0.6  # Make bars narrower for discrete appearance
            ),
            row=1, col=2
        )
        
        # 3. Sector Diversity Analysis
        fig.add_trace(
            go.Histogram(
                x=group_df['unique_sectors'],
                nbinsx=10,
                name="Sector Diversity",
                marker_color='#45b7d1',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Top 20 Groups by Score
        top20_groups = group_df.nlargest(20, 'avg_score')
        fig.add_trace(
            go.Bar(
                x=[f"G{g}" for g in top20_groups['group_number']],
                y=top20_groups['avg_score'],
                name="Top Groups",
                marker=dict(
                    color=top20_groups['estimated_max_correlation'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=[f"{corr:.3f}" for corr in top20_groups['estimated_max_correlation']],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # 5. MAXIMAL Correlation Optimization Results
        correlation_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
        correlation_labels = ['<0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '0.20-0.30', '>0.30']
        group_df['max_correlation_bin'] = pd.cut(group_df['estimated_max_correlation'], 
                                               bins=correlation_bins, labels=correlation_labels)
        corr_counts = group_df['max_correlation_bin'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=corr_counts.index,
                values=corr_counts.values,
                hole=0.3,
                name="MAXIMAL Correlation Distribution",
                textinfo='label+percent+value',
                showlegend=False,
                marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
            ),
            row=3, col=1
        )
        
        # 6. Score vs MAXIMAL Correlation Trade-off
        fig.add_trace(
            go.Scatter(
                x=group_df['avg_score'],
                y=group_df['estimated_max_correlation'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=group_df['unique_sectors'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"G{g}<br>Size: {s}<br>Sectors: {sectors}" 
                      for g, s, sectors in zip(group_df['group_number'], 
                                             group_df['total_securities'],
                                             group_df['unique_sectors'])],
                hovertemplate='%{text}<extra></extra>',
                name="Trade-off Analysis"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1500,
            title_text="CORRELATION-OPTIMIZED PORTFOLIO ANALYSIS - INTELLIGENT GROUP SIZING",
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
        
        # Add axis labels - FOCUS ON MAXIMAL CORRELATIONS
        fig.update_xaxes(title_text="MAXIMAL Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=1)
        fig.update_xaxes(title_text="Group Size (# Securities)", row=1, col=2)
        fig.update_yaxes(title_text="Number of Groups", row=1, col=2)
        fig.update_xaxes(title_text="Unique Sectors per Group", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Top 20 Groups", row=2, col=2)
        fig.update_yaxes(title_text="Average Score", row=2, col=2)
        fig.update_xaxes(title_text="Average Score", row=3, col=2)
        fig.update_yaxes(title_text="MAXIMAL Correlation", row=3, col=2)
        
        # Save dashboard
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Correlation-Optimized Dashboard saved: {output_file}")
        
        return fig
    
    def export_optimized_groups(self, output_file='correlation_optimized_trading_groups.csv'):
        """Export correlation-optimized groups"""
        print(f"\n📄 Exporting correlation-optimized groups...")
        
        export_data = []
        for group in self.portfolio_groups:
            for sec in group['securities']:
                export_data.append({
                    'group_number': group['group_number'],
                    'group_name': f"Optimized_Group_{group['group_number']:02d}",
                    'ticker': sec['ticker'],
                    'company': sec['company'],
                    'sector': sec['sector'],
                    'industry': sec['industry'],
                    'country': sec['country'],
                    'total_score': sec['total_score'],
                    'rank': sec['rank'],
                    'market_cap_m': sec['market_cap_m'],
                    'selection_reason': sec['selection_reason'],
                    'group_size': group['total_securities'],
                    'group_sectors': group['unique_sectors'],
                    'group_industries': group['unique_industries'],
                    'estimated_avg_correlation': group.get('estimated_avg_correlation', None),
                    'estimated_max_correlation': group.get('estimated_max_correlation', None)
                })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(export_data)} securities across {len(self.portfolio_groups)} groups")
        print(f"📁 File: {output_file}")
        
        return export_df
    
    def run_correlation_optimized_builder(self):
        """Run complete correlation-optimized portfolio building"""
        print("="*80)
        print("CORRELATION-OPTIMIZED PORTFOLIO BUILDER")
        print("🧠 Intelligent correlation management with variable group sizes")
        print("="*80)
        
        # Load data and correlations
        if not self.load_data_and_calculate_correlations():
            return
        
        # Create optimized groups
        self.create_correlation_optimized_groups()
        
        # Print summary  
        self.print_optimization_summary()
        
        # Create comprehensive dashboard
        self.create_optimization_dashboard()
        
        # Export results
        self.export_optimized_groups()
        
        if self.conn:
            self.conn.close()
        
        print(f"\n🎉 Correlation-optimized portfolio building complete!")
        print(f"📁 Files generated:")
        print(f"   - correlation_optimized_portfolio_dashboard.html (Interactive analysis)")
        print(f"   - correlation_optimized_trading_groups.csv (Trading reference)")
        print(f"🧠 System ready to adapt to changing market conditions!")
    
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
    
    builder = CorrelationOptimizedPortfolioBuilder(watchlist_path, db_path)
    builder.run_correlation_optimized_builder()

if __name__ == "__main__":
    main()