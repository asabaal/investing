#!/usr/bin/env python3
"""
Paper Trading Backtest System

Leverages your existing market data database and infrastructure to backtest
paper trading decisions using the Trade Likelihood Estimator.

Key Features:
- Uses your existing MarketDataDatabase
- Respects historical data cutoff (no lookahead bias)
- Tests multiple historical data windows
- Generates comprehensive dark-mode reports
- Integrates with your paper trading journal
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
from dataclasses import dataclass, asdict

# Import your existing systems
from market_data_database import MarketDataDatabase
from trade_likelihood import TradeAnalyzer, TradeSetup, TradeAnalysis

# Import trade likelihood components
from trade_likelihood.data_structures import create_bidirectional_setup


@dataclass
class PaperTrade:
    """Paper trade configuration"""
    symbol: str
    entry_price: float
    stop_loss: float
    target_price: float
    trade_date: datetime
    direction: str  # 'long' or 'short'
    trade_id: Optional[str] = None
    notes: Optional[str] = None
    
    def get_risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio"""
        if self.direction == 'long':
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.target_price - self.entry_price)
        else:  # short
            risk = abs(self.stop_loss - self.entry_price) 
            reward = abs(self.entry_price - self.target_price)
        
        return reward / risk if risk > 0 else 0


@dataclass
class BacktestResult:
    """Results from backtesting a single trade"""
    trade: PaperTrade
    model_predictions: Dict[str, TradeAnalysis]  # Different historical windows
    actual_outcome: Optional[str] = None  # 'win', 'loss', 'pending', 'no_entry'
    actual_exit_price: Optional[float] = None
    actual_exit_date: Optional[datetime] = None
    days_to_entry: Optional[int] = None  # Days from trade placement to entry trigger
    days_to_exit: Optional[int] = None   # Days from trade placement to exit
    entry_triggered: Optional[bool] = None  # Whether entry was actually triggered
    model_accuracy: Dict[str, float] = None  # Accuracy metrics per window
    
    # NEW: Optimization results
    optimization_results: Optional[Any] = None  # OptimizationResult from trade_setup_optimizer
    optimized_trade_predictions: Optional[Dict[str, TradeAnalysis]] = None  # Analysis of optimized setup
    
    def __post_init__(self):
        if self.model_accuracy is None:
            self.model_accuracy = {}


class PaperTradeBacktester:
    """
    Comprehensive backtesting system for paper trades using Trade Likelihood Estimator
    """
    
    def __init__(self, 
                 market_db_path: str = None,
                 historical_windows: List[int] = None):
        """
        Initialize the backtester
        
        Args:
            market_db_path: Path to your market data database (uses your existing one)
            historical_windows: Days of historical data to test (default: [30, 60, 90, 120])
        """
        # Use your existing market database
        db_path = market_db_path or "/home/asabaal/asabaal_ventures/repos/investing/trading_data.db"
        if not os.path.exists(db_path):
            # Fallback to your MarketDataDatabase default
            self.market_db = MarketDataDatabase()
        else:
            # Connect to your existing database
            self.db_path = db_path
            
        self.historical_windows = historical_windows or [30, 60, 90, 120]
        
        # Initialize trade analyzer (will be recreated for each test)
        self.base_analyzer = None
        
        print(f"📊 Initialized backtester with {len(self.historical_windows)} historical windows")
    
    def load_market_data_from_your_db(self, symbol: str, end_date: datetime, days_back: int = 120) -> pd.Series:
        """
        Load market data from your existing database, with fallback to MarketDataDatabase
        
        Args:
            symbol: Stock symbol
            end_date: End date for data (trade date)
            days_back: How many days back to load
            
        Returns:
            Price series indexed by datetime
        """
        # Calculate start date
        start_date = end_date - timedelta(days=days_back + 10)  # Extra buffer
        
        try:
            # First try your existing trading_data.db
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, close 
                FROM market_data 
                WHERE symbol = ? 
                AND timestamp <= ? 
                AND timestamp >= ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=[symbol, end_date.isoformat(), start_date.isoformat()]
            )
            
            conn.close()
            
            if not df.empty:
                # Convert to pandas Series with datetime index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                prices = df.set_index('timestamp')['close']
                
                # Filter to only data BEFORE the trade date (no lookahead)
                prices = prices[prices.index < end_date]
                
                print(f"📈 Loaded {len(prices)} price points for {symbol} from trading_data.db ending {end_date.date()}")
                return prices
            
        except Exception as e:
            print(f"⚠️  Error accessing trading_data.db: {e}")
        
        # Fallback to MarketDataDatabase (your proper market data system)
        print(f"🔄 Trying MarketDataDatabase for {symbol}...")
        
        try:
            from market_data_database import MarketDataDatabase
            
            market_db = MarketDataDatabase()
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date - timedelta(days=1)).strftime('%Y-%m-%d')  # Day before trade
            
            print(f"📊 Fetching {symbol} data from {start_str} to {end_str}")
            df = market_db.get_data(symbol, start_str, end_str, 'daily')
            
            if not df.empty:
                # Find the close price column - prioritize 'Close' (adjusted)
                close_col = None
                for col in ['Close', 'close', 'adj_close', 'Unadjusted_Close']:
                    if col in df.columns:
                        close_col = col
                        break
                
                if close_col:
                    prices = df[close_col]
                    print(f"✅ Loaded {len(prices)} price points for {symbol} from MarketDataDatabase (using {close_col})")
                    return prices
                else:
                    print(f"❌ No close price column found. Available: {list(df.columns)}")
            else:
                print(f"❌ No {symbol} data available from MarketDataDatabase")
                
        except Exception as e:
            print(f"❌ MarketDataDatabase fallback failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"⚠️  No data found for {symbol} - cannot proceed with backtest")
        return pd.Series(dtype=float)
    
    def analyze_historical_trade(self, paper_trade: PaperTrade, include_optimization: bool = True) -> BacktestResult:
        """
        Analyze a paper trade using different historical data windows
        
        Args:
            paper_trade: The trade to analyze
            
        Returns:
            Comprehensive backtest result
        """
        print(f"\n🔍 Analyzing {paper_trade.symbol} trade from {paper_trade.trade_date.date()}")
        print(f"   Direction: {paper_trade.direction}")
        print(f"   Entry: ${paper_trade.entry_price:.2f}")
        print(f"   Stop: ${paper_trade.stop_loss:.2f}")  
        print(f"   Target: ${paper_trade.target_price:.2f}")
        print(f"   R:R = {paper_trade.get_risk_reward_ratio():.2f}:1")
        
        model_predictions = {}
        
        # Test different historical windows
        for window_days in self.historical_windows:
            print(f"\n📊 Testing with {window_days} days of historical data...")
            
            # Load historical data up to (but not including) trade date
            prices = self.load_market_data_from_your_db(
                paper_trade.symbol, 
                paper_trade.trade_date, 
                window_days
            )
            
            if len(prices) < 20:  # Need minimum data
                print(f"⚠️  Insufficient data ({len(prices)} points) for {window_days}-day window")
                continue
            
            try:
                # Create fresh analyzer for this test
                analyzer = TradeAnalyzer()
                
                # Load the historical data (only up to trade date!)
                summary = analyzer.load_price_data(prices.values, prices.index)
                
                if not summary['parameters_estimated']:
                    print(f"⚠️  Could not estimate parameters for {window_days}-day window")
                    continue
                
                # Create trade setup based on the paper trade
                setup = self._create_trade_setup(paper_trade, prices.iloc[-1])
                
                # Analyze the setup as it would have appeared on trade date
                analysis = analyzer.analyze_trade_setup(setup, f"{paper_trade.symbol}_{window_days}d")
                
                model_predictions[f"{window_days}d"] = analysis
                
                # Print key predictions
                print(f"   Return Rate: {analysis.return_rate_total:.4f} per day")
                print(f"   Entry Prob: {analysis.get_combined_entry_probability():.1%}")
                print(f"   Attractive: {'✅' if analysis.is_attractive_setup() else '❌'}")
                print(f"   Best Direction: {analysis.get_best_direction()}")
                
            except Exception as e:
                print(f"❌ Error analyzing {window_days}-day window: {e}")
                continue
        
        # Determine actual outcome (if possible)
        actual_outcome = self._determine_actual_outcome(paper_trade)
        
        # Run optimization if requested
        optimization_results = None
        optimized_predictions = {}
        
        if include_optimization and model_predictions:
            print(f"\n🎯 RUNNING TRADE SETUP OPTIMIZATION...")
            try:
                optimization_results, optimized_predictions = self._optimize_trade_setup(
                    paper_trade, model_predictions
                )
            except Exception as e:
                print(f"⚠️  Optimization failed: {e}")
        
        return BacktestResult(
            trade=paper_trade,
            model_predictions=model_predictions,
            optimization_results=optimization_results,
            optimized_trade_predictions=optimized_predictions,
            **actual_outcome
        )
    
    def _optimize_trade_setup(self, paper_trade: PaperTrade, model_predictions: Dict[str, TradeAnalysis]) -> Tuple[Any, Dict[str, TradeAnalysis]]:
        """
        Optimize the trade setup using the best performing historical window
        
        Args:
            paper_trade: Original paper trade
            model_predictions: Analysis results from different historical windows
            
        Returns:
            Tuple of (OptimizationResult, optimized_predictions_dict)
        """
        from .trade_setup_optimizer import TradeSetupOptimizer, OptimizationConstraints
        
        # Find the best historical window (highest return rate)
        best_window = None
        best_return_rate = float('-inf')
        best_analysis = None
        
        for window, analysis in model_predictions.items():
            if analysis.return_rate_total and analysis.return_rate_total > best_return_rate:
                best_return_rate = analysis.return_rate_total
                best_window = window
                best_analysis = analysis
        
        if not best_analysis:
            print("⚠️  No valid analysis found for optimization")
            return None, {}
        
        print(f"📊 Using {best_window} historical window for optimization (best return rate: {best_return_rate:.4f})")
        
        # Get the analyzer used for the best window (need to recreate it)
        # Use the same data loading approach as the original analysis
        prices = self.load_market_data_from_your_db(
            paper_trade.symbol, 
            paper_trade.trade_date, 
            int(best_window.replace('d', ''))  # Extract days from window string
        )
        
        if len(prices) < 20:
            print("⚠️  Insufficient data for optimization")
            return None, {}
        
        # Create fresh analyzer for optimization
        from .trade_analyzer import TradeAnalyzer
        analyzer = TradeAnalyzer()
        analyzer.load_price_data(prices.values, prices.index)
        
        # Create original setup
        current_price = prices.iloc[-1]
        original_setup = self._create_trade_setup(paper_trade, current_price)
        
        # Calculate original R:R ratio for max constraint
        original_rr = paper_trade.get_risk_reward_ratio()
        
        # Set up optimization constraints
        constraints = OptimizationConstraints(
            min_risk_reward=3.0,  # Your minimum requirement
            max_risk_reward=max(original_rr, 3.0),  # Don't make it worse than original
            preserve_direction=True
        )
        
        # Initialize optimizer
        optimizer = TradeSetupOptimizer(analyzer)
        
        # Run optimization for multiple objectives
        objectives = ["return_rate_total", "prob_win", "expected_value_total"]
        optimization_results = {}
        optimized_predictions = {}
        
        for objective in objectives:
            print(f"🔍 Optimizing for: {objective}")
            try:
                opt_result = optimizer.optimize_setup(
                    original_setup=original_setup,
                    constraints=constraints,
                    objective=objective,
                    method="differential_evolution",
                    max_iterations=500  # Reduced for faster backtesting
                )
                optimization_results[objective] = opt_result
                
                # Analyze optimized setup with all historical windows
                optimized_predictions[objective] = {}
                
                # Re-analyze the optimized setup with the same historical windows as original
                for window, _ in model_predictions.items():
                    try:
                        window_days = int(window.replace('d', ''))
                        window_prices = self.load_market_data_from_your_db(
                            paper_trade.symbol, paper_trade.trade_date, window_days
                        )
                        
                        if len(window_prices) >= 20:
                            window_analyzer = TradeAnalyzer()
                            window_analyzer.load_price_data(window_prices.values, window_prices.index)
                            
                            opt_analysis = window_analyzer.analyze_trade_setup(
                                opt_result.optimized_setup, 
                                f"optimized_{objective}_{window}"
                            )
                            optimized_predictions[objective][window] = opt_analysis
                    except Exception as e:
                        print(f"⚠️  Failed to analyze optimized setup for {window}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Optimization failed for {objective}: {e}")
                continue
        
        # Return the best optimization result (by return rate)
        if optimization_results:
            best_opt_objective = max(optimization_results.keys(), 
                                   key=lambda obj: optimization_results[obj].optimized_analysis.return_rate_total or -999)
            best_opt_result = optimization_results[best_opt_objective]
            best_opt_predictions = optimized_predictions[best_opt_objective]
            
            print(f"✅ Best optimization: {best_opt_objective}")
            return best_opt_result, best_opt_predictions
        
        return None, {}
    
    def _create_trade_setup(self, paper_trade: PaperTrade, current_price: float) -> TradeSetup:
        """
        Create a TradeSetup object from a paper trade
        
        Args:
            paper_trade: The paper trade
            current_price: Current market price at trade date
            
        Returns:
            TradeSetup object for analysis
        """
        if paper_trade.direction == 'long':
            return TradeSetup(
                current_price=current_price,
                entry_long=paper_trade.entry_price,
                stop_long=paper_trade.stop_loss,
                target_long=paper_trade.target_price,
                max_time_window=5.0  # 5 days default
            )
        else:  # short
            return TradeSetup(
                current_price=current_price,
                entry_short=paper_trade.entry_price,
                stop_short=paper_trade.stop_loss,
                target_short=paper_trade.target_price,
                max_time_window=5.0  # 5 days default
            )
    
    def _determine_actual_outcome(self, paper_trade: PaperTrade) -> Dict[str, Any]:
        """
        Determine actual trade outcome by looking at subsequent price data
        
        Args:
            paper_trade: The trade to check
            
        Returns:
            Dictionary with actual outcome data
        """
        try:
            # Load data AFTER the trade date to see what actually happened
            future_data = self.load_market_data_after_trade(
                paper_trade.symbol,
                paper_trade.trade_date,
                days_forward=30  # Look 30 days forward
            )
            
            if future_data.empty:
                return {
                    'actual_outcome': 'unknown',
                    'actual_exit_price': None,
                    'actual_exit_date': None,
                    'days_to_exit': None
                }
            
            # Check if trade would have triggered and outcome
            return self._simulate_trade_outcome(paper_trade, future_data)
            
        except Exception as e:
            print(f"⚠️  Could not determine actual outcome: {e}")
            return {
                'actual_outcome': 'unknown',
                'actual_exit_price': None,
                'actual_exit_date': None,
                'days_to_exit': None
            }
    
    def load_market_data_after_trade(self, symbol: str, trade_date: datetime, days_forward: int = 30) -> pd.DataFrame:
        """Load market data AFTER the trade date to determine outcome - return full OHLC data"""
        try:
            # Use MarketDataDatabase for consistency
            from market_data_database import MarketDataDatabase
            
            market_db = MarketDataDatabase()
            start_str = trade_date.strftime('%Y-%m-%d')
            end_str = (trade_date + timedelta(days=days_forward)).strftime('%Y-%m-%d')
            
            df = market_db.get_data(symbol, start_str, end_str, 'daily')
            
            if not df.empty:
                # Filter to only data AFTER the trade date
                df = df[df.index > trade_date.strftime('%Y-%m-%d')]
                print(f"📊 Loaded {len(df)} days of post-trade data for outcome analysis")
                return df
            else:
                print(f"❌ No post-trade data found for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error loading future data: {e}")
            return pd.DataFrame()
    
    def _simulate_trade_outcome(self, trade: PaperTrade, future_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate what would have happened with the trade using full OHLC data"""
        
        if future_data.empty:
            return {'actual_outcome': 'unknown'}
        
        entry_price = trade.entry_price
        stop_loss = trade.stop_loss
        target = trade.target_price
        direction = trade.direction
        
        # Determine if trade was entered and outcome using OHLC data
        trade_entered = False
        entry_date = None
        
        # For each day, check if entry was triggered first, then stop/target
        for date_str, row in future_data.iterrows():
            # Get OHLC values, handling different column naming conventions
            high = row.get('High', row.get('high', row.get('H', None)))
            low = row.get('Low', row.get('low', row.get('L', None)))
            open_price = row.get('Open', row.get('open', row.get('O', None)))
            close_price = row.get('Close', row.get('close', row.get('C', None)))
            
            if any(x is None for x in [high, low, open_price, close_price]):
                print(f"⚠️  Missing OHLC data for {date_str}, skipping day")
                continue
                
            # Convert string date to datetime for comparison
            if isinstance(date_str, str):
                date_obj = pd.to_datetime(date_str)
            else:
                date_obj = date_str
                
            days_elapsed = (date_obj.date() - trade.trade_date.date()).days
            
            # Check if entry was triggered on this day
            if not trade_entered:
                if direction == 'short':
                    # For short breakout: entry triggers if price rises to/above entry
                    if high >= entry_price:
                        trade_entered = True
                        entry_date = date_obj
                        print(f"📍 Short entry triggered on {date_obj.date()} at ${entry_price:.2f} (High: ${high:.2f})")
                else:  # long
                    # For long breakout: entry triggers if price rises to/above entry  
                    if high >= entry_price:
                        trade_entered = True
                        entry_date = date_obj
                        print(f"📍 Long entry triggered on {date_obj.date()} at ${entry_price:.2f} (High: ${high:.2f})")
            
            # If trade was entered, check for exit conditions
            if trade_entered:
                if direction == 'short':
                    # Short trade: loss if price hits stop (above entry), win if hits target (below entry)
                    if high >= stop_loss:
                        return {
                            'actual_outcome': 'loss',
                            'actual_exit_price': stop_loss,
                            'actual_exit_date': date_obj,
                            'days_to_entry': (entry_date.date() - trade.trade_date.date()).days,
                            'days_to_exit': days_elapsed,
                            'entry_triggered': True
                        }
                    elif low <= target:
                        return {
                            'actual_outcome': 'win',
                            'actual_exit_price': target,
                            'actual_exit_date': date_obj,
                            'days_to_entry': (entry_date.date() - trade.trade_date.date()).days,
                            'days_to_exit': days_elapsed,
                            'entry_triggered': True
                        }
                else:  # long
                    # Long trade: loss if price hits stop (below entry), win if hits target (above entry)
                    if low <= stop_loss:
                        return {
                            'actual_outcome': 'loss',
                            'actual_exit_price': stop_loss,
                            'actual_exit_date': date_obj,
                            'days_to_entry': (entry_date.date() - trade.trade_date.date()).days,
                            'days_to_exit': days_elapsed,
                            'entry_triggered': True
                        }
                    elif high >= target:
                        return {
                            'actual_outcome': 'win',
                            'actual_exit_price': target,
                            'actual_exit_date': date_obj,
                            'days_to_entry': (entry_date.date() - trade.trade_date.date()).days,
                            'days_to_exit': days_elapsed,
                            'entry_triggered': True
                        }
        
        # If we get here, determine what happened
        if not trade_entered:
            return {
                'actual_outcome': 'no_entry',
                'actual_exit_price': None,
                'actual_exit_date': None,
                'days_to_entry': None,
                'days_to_exit': None,
                'entry_triggered': False
            }
        else:
            # Entry triggered but neither target nor stop hit within timeframe
            last_date = pd.to_datetime(future_data.index[-1]) 
            last_close = future_data.iloc[-1].get('Close', future_data.iloc[-1].get('close', None))
            return {
                'actual_outcome': 'pending', 
                'actual_exit_price': last_close,
                'actual_exit_date': last_date,
                'days_to_entry': (entry_date.date() - trade.trade_date.date()).days,
                'days_to_exit': (last_date.date() - trade.trade_date.date()).days,
                'entry_triggered': True
            }
    
    def run_single_trade_backtest(self, 
                                 symbol: str,
                                 entry_price: float,
                                 stop_loss: float, 
                                 target_price: float,
                                 trade_date: str,  # 'YYYY-MM-DD'
                                 direction: str = 'auto') -> BacktestResult:
        """
        Run backtest for a single trade (convenience method)
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price  
            trade_date: Trade date string 'YYYY-MM-DD'
            direction: 'long', 'short', or 'auto' to detect
            
        Returns:
            Backtest result
        """
        
        # Auto-detect direction if not specified
        if direction == 'auto':
            if target_price > entry_price:
                direction = 'long'
            else:
                direction = 'short'
        
        # Create paper trade object
        paper_trade = PaperTrade(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            trade_date=datetime.strptime(trade_date, '%Y-%m-%d'),
            direction=direction,
            trade_id=f"{symbol}_{trade_date}"
        )
        
        return self.analyze_historical_trade(paper_trade)


def quick_backtest_dpz():
    """Quick test with your DPZ trade"""
    
    print("🎯 BACKTESTING DPZ TRADE")
    print("=" * 50)
    
    backtester = PaperTradeBacktester()
    
    # Your DPZ trade details
    result = backtester.run_single_trade_backtest(
        symbol='DPZ',
        entry_price=464.80,
        stop_loss=468.46, 
        target_price=441.47,
        trade_date='2025-09-01',
        direction='short'  # Confirmed short since target < entry
    )
    
    print(f"\n📊 BACKTEST RESULTS FOR DPZ")
    print("=" * 40)
    
    trade = result.trade
    print(f"Trade: {trade.symbol} {trade.direction}")
    print(f"Entry: ${trade.entry_price:.2f}")
    print(f"Stop: ${trade.stop_loss:.2f}")
    print(f"Target: ${trade.target_price:.2f}")
    print(f"R:R Ratio: {trade.get_risk_reward_ratio():.2f}:1")
    
    if result.actual_outcome != 'unknown':
        print(f"\n🎯 ACTUAL OUTCOME: {result.actual_outcome.upper()}")
        print(f"Entry Triggered: {'✅' if result.entry_triggered else '❌'}")
        if result.days_to_entry is not None:
            print(f"Days to Entry: {result.days_to_entry}")
        if result.actual_exit_price:
            print(f"Exit Price: ${result.actual_exit_price:.2f}")
        if result.days_to_exit is not None:
            print(f"Days to Exit: {result.days_to_exit}")
    
    print(f"\n📊 MODEL PREDICTIONS:")
    for window, analysis in result.model_predictions.items():
        print(f"\n{window} Historical Data:")
        print(f"  Return Rate: {analysis.return_rate_total:.4f}/day")
        print(f"  Entry Prob: {analysis.get_combined_entry_probability():.1%}")
        # Add missing probability information
        if analysis.prob_win_long:
            print(f"  Long Win Prob: {analysis.prob_win_long:.1%}")
        if analysis.prob_win_short:
            print(f"  Short Win Prob: {analysis.prob_win_short:.1%}")
        # Add expected time information  
        if analysis.expected_entry_time_long:
            print(f"  Expected Entry Time (Long): {analysis.expected_entry_time_long:.2f} days")
        if analysis.expected_entry_time_short:
            print(f"  Expected Entry Time (Short): {analysis.expected_entry_time_short:.2f} days")
        if analysis.expected_trade_duration_long:
            print(f"  Expected Trade Duration (Long): {analysis.expected_trade_duration_long:.2f} days")
        if analysis.expected_trade_duration_short:
            print(f"  Expected Trade Duration (Short): {analysis.expected_trade_duration_short:.2f} days")
        print(f"  Attractive: {'✅' if analysis.is_attractive_setup() else '❌'}")
        print(f"  Regime: {analysis.market_params.regime}")
    
    return result


if __name__ == "__main__":
    # Run quick test
    result = quick_backtest_dpz()