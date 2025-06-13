#!/usr/bin/env python3
"""
Multi-Symphony Trading Engine with Black Swan Protection
Manages multiple trading strategies simultaneously with risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from symphony_engine import SymphonyEngine
from market_data_database import MarketDataDatabase
from symphony_backtester import SymphonyBacktester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrisisIndicators:
    """Track market crisis indicators"""
    vix_level: float
    vix_percentile: float
    credit_spread: float
    put_call_ratio: float
    term_structure: str  # 'normal' or 'inverted'
    crisis_score: float
    regime: str  # 'bull', 'bear', 'crisis', 'uncertainty'

@dataclass
class PortfolioAllocation:
    """Track allocations across multiple symphonies"""
    symphony_name: str
    allocation_pct: float
    dollar_amount: float
    positions: Dict[str, float]
    expected_return: float
    risk_contribution: float

class MultiSymphonyEngine:
    """Engine to run multiple symphony strategies with coordinated risk management"""
    
    def __init__(self, config_path: str = "black_swan_symphonies.json"):
        self.config = self._load_config(config_path)
        self.db = MarketDataDatabase()
        self.engines = {}
        self.backtester = SymphonyBacktester(self.db)
        self.total_capital = self.config['portfolio_allocation']['total_capital']
        self._initialize_engines()
        
    def _load_config(self, config_path: str) -> dict:
        """Load multi-symphony configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_engines(self):
        """Initialize individual symphony engines"""
        for symphony_name, symphony_config in self.config['symphonies'].items():
            self.engines[symphony_name] = SymphonyEngine(symphony_config, self.db)
            logger.info(f"Initialized symphony: {symphony_name}")
    
    def calculate_crisis_indicators(self, date: datetime) -> CrisisIndicators:
        """Calculate comprehensive crisis indicators"""
        try:
            # Get VIX data (using VXX as proxy)
            vix_data = self.db.get_data_for_date_range(
                'VXX', 
                date - timedelta(days=60), 
                date
            )
            
            if vix_data.empty:
                logger.warning("No VIX data available")
                current_vix = 15.0  # Default low VIX
            else:
                current_vix = vix_data['close'].iloc[-1]
                vix_percentile = (vix_data['close'] < current_vix).mean()
            
            # Calculate credit spread (HYG vs TLT)
            hyg_data = self.db.get_data_for_date_range('HYG', date - timedelta(days=30), date)
            tlt_data = self.db.get_data_for_date_range('TLT', date - timedelta(days=30), date)
            
            if not hyg_data.empty and not tlt_data.empty:
                hyg_return = hyg_data['close'].pct_change().rolling(5).mean().iloc[-1]
                tlt_return = tlt_data['close'].pct_change().rolling(5).mean().iloc[-1]
                credit_spread = (tlt_return - hyg_return) * 10000  # basis points
            else:
                credit_spread = 200  # Default normal spread
            
            # Calculate put/call ratio (simplified using volatility skew)
            spy_data = self.db.get_data_for_date_range('SPY', date - timedelta(days=30), date)
            if not spy_data.empty:
                returns = spy_data['close'].pct_change().dropna()
                downside_vol = returns[returns < 0].std()
                upside_vol = returns[returns > 0].std()
                put_call_ratio = downside_vol / upside_vol if upside_vol > 0 else 1.0
            else:
                put_call_ratio = 1.0
            
            # Determine VIX term structure
            if current_vix > 25:
                term_structure = 'inverted'
            else:
                term_structure = 'normal'
            
            # Calculate crisis score
            crisis_score = (
                (current_vix / 20) * 2 +
                (credit_spread / 300) +
                put_call_ratio +
                (0.5 if term_structure == 'inverted' else 0)
            )
            
            # Determine regime
            if crisis_score > 4:
                regime = 'crisis'
            elif crisis_score > 2.5:
                regime = 'bear'
            elif current_vix < 20 and credit_spread < 200:
                regime = 'bull'
            else:
                regime = 'uncertainty'
            
            return CrisisIndicators(
                vix_level=current_vix,
                vix_percentile=vix_percentile if 'vix_percentile' in locals() else 0.5,
                credit_spread=credit_spread,
                put_call_ratio=put_call_ratio,
                term_structure=term_structure,
                crisis_score=crisis_score,
                regime=regime
            )
            
        except Exception as e:
            logger.error(f"Error calculating crisis indicators: {e}")
            # Return default safe values
            return CrisisIndicators(
                vix_level=20.0,
                vix_percentile=0.5,
                credit_spread=300,
                put_call_ratio=1.0,
                term_structure='normal',
                crisis_score=2.0,
                regime='uncertainty'
            )
    
    def get_portfolio_allocations(self, date: datetime) -> List[PortfolioAllocation]:
        """Get allocations for all symphonies based on current market conditions"""
        crisis_indicators = self.calculate_crisis_indicators(date)
        allocations = []
        
        # Get allocations by strategy type
        protection_config = self.config['portfolio_allocation']['allocations']['protection_strategies']
        profit_config = self.config['portfolio_allocation']['allocations']['profit_strategies']
        
        # Protection strategies (more allocation during crisis)
        protection_capital = protection_config['capital']
        if crisis_indicators.regime == 'crisis':
            protection_capital *= 1.5  # Increase protection allocation
        elif crisis_indicators.regime == 'bull':
            protection_capital *= 0.7  # Decrease protection allocation
        
        # Ensure we don't exceed total capital
        protection_capital = min(protection_capital, self.total_capital * 0.5)
        profit_capital = self.total_capital - protection_capital
        
        # Allocate to protection strategies
        for symphony_name, weight in protection_config['symphonies'].items():
            dollar_amount = protection_capital * weight
            positions = self._get_symphony_positions(symphony_name, date)
            
            allocation = PortfolioAllocation(
                symphony_name=symphony_name,
                allocation_pct=dollar_amount / self.total_capital,
                dollar_amount=dollar_amount,
                positions=positions,
                expected_return=self._estimate_expected_return(symphony_name, crisis_indicators),
                risk_contribution=self._estimate_risk_contribution(symphony_name, positions)
            )
            allocations.append(allocation)
        
        # Allocate to profit strategies
        for symphony_name, weight in profit_config['symphonies'].items():
            dollar_amount = profit_capital * weight
            positions = self._get_symphony_positions(symphony_name, date)
            
            allocation = PortfolioAllocation(
                symphony_name=symphony_name,
                allocation_pct=dollar_amount / self.total_capital,
                dollar_amount=dollar_amount,
                positions=positions,
                expected_return=self._estimate_expected_return(symphony_name, crisis_indicators),
                risk_contribution=self._estimate_risk_contribution(symphony_name, positions)
            )
            allocations.append(allocation)
        
        return allocations
    
    def _get_symphony_positions(self, symphony_name: str, date: datetime) -> Dict[str, float]:
        """Get current positions for a symphony"""
        try:
            engine = self.engines[symphony_name]
            allocations = engine.calculate_allocations(date)
            return allocations
        except Exception as e:
            logger.error(f"Error getting positions for {symphony_name}: {e}")
            return {}
    
    def _estimate_expected_return(self, symphony_name: str, indicators: CrisisIndicators) -> float:
        """Estimate expected return based on strategy type and market regime"""
        # Protection strategies
        if 'black_swan' in symphony_name or 'crisis' in symphony_name:
            if indicators.regime == 'crisis':
                return 0.25  # 25% expected during crisis
            else:
                return -0.03  # -3% drag during normal times
        
        # Momentum strategies
        elif 'momentum' in symphony_name:
            if indicators.regime == 'bull':
                return 0.18  # 18% in bull markets
            elif indicators.regime == 'crisis':
                return -0.15  # -15% in crisis
            else:
                return 0.08  # 8% in uncertainty
        
        # All-weather strategies
        elif 'all_weather' in symphony_name:
            return 0.10  # Consistent 10% target
        
        # Sector rotation
        else:
            if indicators.regime == 'bull':
                return 0.15
            else:
                return 0.05
    
    def _estimate_risk_contribution(self, symphony_name: str, positions: Dict[str, float]) -> float:
        """Estimate risk contribution of a strategy"""
        if not positions:
            return 0.0
        
        # High risk assets
        high_risk = ['UVXY', 'VXX', 'ARKK']
        medium_risk = ['SPY', 'QQQ', 'XLK', 'XLY']
        low_risk = ['TLT', 'GLD', 'XLP', 'XLU']
        
        risk_score = 0.0
        for asset, weight in positions.items():
            if asset in high_risk:
                risk_score += weight * 3.0
            elif asset in medium_risk:
                risk_score += weight * 1.5
            elif asset in low_risk:
                risk_score += weight * 0.5
            else:
                risk_score += weight * 1.0
        
        return risk_score
    
    def execute_portfolio(self, date: datetime) -> Dict:
        """Execute the complete portfolio for a given date"""
        # Get crisis indicators
        indicators = self.calculate_crisis_indicators(date)
        logger.info(f"Date: {date}, Regime: {indicators.regime}, Crisis Score: {indicators.crisis_score:.2f}")
        
        # Get allocations
        allocations = self.get_portfolio_allocations(date)
        
        # Aggregate positions across all symphonies
        combined_positions = defaultdict(float)
        total_allocated = 0.0
        
        for allocation in allocations:
            logger.info(f"{allocation.symphony_name}: ${allocation.dollar_amount:.2f} "
                       f"({allocation.allocation_pct*100:.1f}%)")
            
            for asset, weight in allocation.positions.items():
                dollar_value = allocation.dollar_amount * weight
                combined_positions[asset] += dollar_value
                total_allocated += dollar_value
        
        # Convert to weights
        final_weights = {}
        for asset, dollar_value in combined_positions.items():
            final_weights[asset] = dollar_value / self.total_capital
        
        # Add cash if not fully allocated
        if total_allocated < self.total_capital:
            final_weights['CASH'] = (self.total_capital - total_allocated) / self.total_capital
        
        return {
            'date': date,
            'indicators': indicators,
            'allocations': allocations,
            'final_weights': final_weights,
            'regime': indicators.regime
        }
    
    def backtest_multi_symphony(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Backtest the multi-symphony portfolio"""
        logger.info(f"Backtesting multi-symphony portfolio from {start_date} to {end_date}")
        
        # Get date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Track portfolio value and allocations
        portfolio_values = []
        allocation_history = []
        regime_history = []
        
        current_value = self.total_capital
        current_positions = {}
        
        for date in dates:
            try:
                # Get portfolio for this date
                portfolio = self.execute_portfolio(date)
                
                # Calculate returns if we have positions
                if current_positions:
                    daily_return = self._calculate_daily_return(
                        current_positions, 
                        date - timedelta(days=1), 
                        date
                    )
                    current_value *= (1 + daily_return)
                
                # Update positions
                current_positions = portfolio['final_weights'].copy()
                
                # Record data
                portfolio_values.append({
                    'date': date,
                    'value': current_value,
                    'return': (current_value / self.total_capital) - 1,
                    'regime': portfolio['regime'],
                    'crisis_score': portfolio['indicators'].crisis_score,
                    'vix_level': portfolio['indicators'].vix_level
                })
                
                allocation_history.append({
                    'date': date,
                    'allocations': current_positions.copy()
                })
                
                regime_history.append(portfolio['regime'])
                
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_values)
        results_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        total_return = (current_value / self.total_capital) - 1
        daily_returns = results_df['value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        max_drawdown = (results_df['value'] / results_df['value'].cummax() - 1).min()
        
        # Count regime occurrences
        regime_counts = pd.Series(regime_history).value_counts()
        
        logger.info(f"\nBacktest Results:")
        logger.info(f"Total Return: {total_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"Final Value: ${current_value:.2f}")
        logger.info(f"\nRegime Distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count/len(regime_history)*100:.1f}%")
        
        # Save detailed results
        self._save_backtest_results(results_df, allocation_history)
        
        return results_df
    
    def _calculate_daily_return(self, positions: Dict[str, float], 
                               start_date: datetime, end_date: datetime) -> float:
        """Calculate portfolio return for a single day"""
        total_return = 0.0
        
        for asset, weight in positions.items():
            if asset == 'CASH':
                continue  # No return on cash
            
            try:
                data = self.db.get_data_for_date_range(asset, start_date, end_date)
                if len(data) >= 2:
                    asset_return = (data['close'].iloc[-1] / data['close'].iloc[-2]) - 1
                    total_return += weight * asset_return
            except Exception as e:
                logger.debug(f"Could not calculate return for {asset}: {e}")
                continue
        
        return total_return
    
    def _save_backtest_results(self, results_df: pd.DataFrame, 
                              allocation_history: List[Dict]):
        """Save backtest results to files"""
        # Save performance data
        results_df.to_csv('multi_symphony_backtest_results.csv')
        
        # Save allocation history
        with open('multi_symphony_allocation_history.json', 'w') as f:
            json.dump(allocation_history, f, indent=2, default=str)
        
        logger.info("Backtest results saved to files")
    
    def forward_test(self, paper_trade_days: int = 30):
        """Run forward testing (paper trading) for specified number of days"""
        logger.info(f"Starting {paper_trade_days}-day forward test")
        
        start_date = datetime.now() - timedelta(days=paper_trade_days)
        end_date = datetime.now()
        
        # Run backtest on recent data
        results = self.backtest_multi_symphony(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Get current portfolio
        current_portfolio = self.execute_portfolio(datetime.now())
        
        logger.info("\nCurrent Portfolio Allocation:")
        for asset, weight in sorted(current_portfolio['final_weights'].items(), 
                                   key=lambda x: x[1], reverse=True):
            if weight > 0.01:  # Only show positions > 1%
                dollar_value = weight * self.total_capital
                logger.info(f"  {asset}: {weight*100:.1f}% (${dollar_value:.2f})")
        
        logger.info(f"\nCurrent Regime: {current_portfolio['regime']}")
        logger.info(f"Crisis Score: {current_portfolio['indicators'].crisis_score:.2f}")
        
        return results, current_portfolio

def main():
    """Run multi-symphony engine with example parameters"""
    engine = MultiSymphonyEngine()
    
    # Run historical backtest
    logger.info("Running historical backtest...")
    backtest_results = engine.backtest_multi_symphony('2022-01-01', '2024-12-31')
    
    # Run forward test
    logger.info("\nRunning forward test...")
    forward_results, current_portfolio = engine.forward_test(30)
    
    # Display summary
    logger.info("\n=== DEPLOYMENT READY ===")
    logger.info(f"Total Capital: ${engine.total_capital}")
    logger.info(f"Number of Strategies: {len(engine.engines)}")
    logger.info(f"Current Market Regime: {current_portfolio['regime']}")
    logger.info("\nRecommended Actions:")
    logger.info("1. Review current allocations above")
    logger.info("2. Set up daily monitoring of crisis indicators")
    logger.info("3. Implement position sizing based on Kelly Criterion")
    logger.info("4. Set stop-loss at 15% portfolio drawdown")
    logger.info("5. Take profits on volatility positions when VIX spikes >40")

if __name__ == "__main__":
    main()