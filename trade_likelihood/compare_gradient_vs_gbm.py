#!/usr/bin/env python3
"""
Compare Gradient-Based vs GBM Trade Analysis on DPZ

Tests both approaches on the DPZ trade setup to evaluate accuracy differences.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gradient_candle_clustering import GradientCandleAnalyzer
from gradient_trade_analyzer import GradientTradeAnalyzer
from data_structures import TradeSetup
from market_data_database import MarketDataDatabase

# For GBM comparison, we'll use backtest_system which has the analysis
from backtest_system import PaperTradeBacktester, PaperTrade

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gradient_vs_gbm():
    """Compare gradient and GBM approaches on DPZ trade"""
    
    print("🔬 GRADIENT vs GBM COMPARISON TEST")
    print("=" * 50)
    
    # Initialize components
    market_db = MarketDataDatabase()
    
    # Setup DPZ data
    symbol = 'DPZ'
    df = market_db.get_data(symbol, '2024-01-01', '2025-08-31', 'daily')
    
    if df.empty:
        print(f"❌ No data found for {symbol}")
        return False
        
    print(f"📊 Loaded {len(df)} candles for {symbol}")
    
    # Get current OHLC (last available)
    close_col = 'Close' if 'Close' in df.columns else 'close'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    
    last_candle = df.iloc[-1]
    current_price = last_candle[close_col]
    current_ohlc = {
        'open': last_candle[open_col],
        'high': last_candle[high_col],
        'low': last_candle[low_col],
        'close': last_candle[close_col]
    }
    
    print(f"💰 Current price: ${current_price:.2f}")
    print(f"📊 Current OHLC: O={current_ohlc['open']:.2f}, H={current_ohlc['high']:.2f}, "
          f"L={current_ohlc['low']:.2f}, C={current_ohlc['close']:.2f}")
    
    # Create DPZ trade setup (from previous analysis)
    trade_setup = TradeSetup(
        current_price=current_price,
        entry_short=464.80,
        stop_short=468.46,
        target_short=441.47,
        max_time_window=5.0  # 5 days
    )
    
    print(f"\n🎯 DPZ SHORT TRADE SETUP:")
    print(f"   Entry: ${trade_setup.entry_short:.2f}")
    print(f"   Stop: ${trade_setup.stop_short:.2f}")
    print(f"   Target: ${trade_setup.target_short:.2f}")
    print(f"   R:R Ratio: {trade_setup.get_short_risk_reward():.2f}:1")
    print(f"   Time Window: {trade_setup.max_time_window} days")
    
    # 1. GBM Analysis
    print(f"\n🔄 RUNNING GBM ANALYSIS...")
    print("-" * 30)
    
    try:
        # Create a paper trade for analysis
        paper_trade = PaperTrade(
            symbol=symbol,
            entry_price=trade_setup.entry_short,
            stop_loss=trade_setup.stop_short,
            target_price=trade_setup.target_short,
            trade_date=datetime.now().date(),
            direction='short'
        )
        
        gbm_backtester = PaperTradeBacktester(market_db)
        result = gbm_backtester.backtest_paper_trade(paper_trade)
        
        if result and result.model_predictions:
            # Get the best window analysis
            best_analysis = None
            best_return_rate = -999
            
            for window, analysis in result.model_predictions.items():
                if analysis.return_rate_short and analysis.return_rate_short > best_return_rate:
                    best_return_rate = analysis.return_rate_short
                    best_analysis = analysis
            
            if best_analysis:
                gbm_analysis = best_analysis
                print(f"✅ GBM Analysis Results:")
                print(f"   Entry Probability: {gbm_analysis.prob_entry_short:.1%}" if gbm_analysis.prob_entry_short else "   Entry Probability: N/A")
                print(f"   Win Probability: {gbm_analysis.prob_win_short:.1%}" if gbm_analysis.prob_win_short else "   Win Probability: N/A")
                print(f"   Expected Value: ${gbm_analysis.expected_value_short:.3f}" if gbm_analysis.expected_value_short else "   Expected Value: N/A")
                print(f"   Return Rate: {gbm_analysis.return_rate_short:.6f}/day" if gbm_analysis.return_rate_short else "   Return Rate: N/A")
                print(f"   Market Regime: {gbm_analysis.market_params.regime}")
            else:
                print("❌ No valid GBM analysis found")
                gbm_analysis = None
        else:
            print("❌ GBM backtesting failed")
            gbm_analysis = None
        
    except Exception as e:
        print(f"❌ GBM Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        gbm_analysis = None
    
    # 2. Gradient Analysis
    print(f"\n🌊 RUNNING GRADIENT ANALYSIS...")
    print("-" * 30)
    
    try:
        # Setup gradient analyzer
        gradient_analyzer = GradientCandleAnalyzer(market_db)
        gradient_trade_analyzer = GradientTradeAnalyzer(gradient_analyzer)
        
        # Setup gradient analysis for DPZ
        success = gradient_trade_analyzer.setup_symbol_analysis(symbol)
        
        if not success:
            print("❌ Failed to setup gradient analysis")
            return False
            
        # Run gradient analysis
        gradient_analysis = gradient_trade_analyzer.analyze_trade_with_gradients(
            trade_setup, current_ohlc, time_horizon_days=5
        )
        
        print(f"✅ Gradient Analysis Results:")
        print(f"   Entry Probability: {gradient_analysis.prob_entry_short:.1%}" if gradient_analysis.prob_entry_short else "   Entry Probability: N/A")
        print(f"   Win Probability: {gradient_analysis.prob_win_short:.1%}" if gradient_analysis.prob_win_short else "   Win Probability: N/A")
        print(f"   Expected Value: ${gradient_analysis.expected_value_short:.3f}" if gradient_analysis.expected_value_short else "   Expected Value: N/A")
        print(f"   Return Rate: {gradient_analysis.return_rate_short:.6f}/day" if gradient_analysis.return_rate_short else "   Return Rate: N/A")
        print(f"   Method: {gradient_analysis.market_params.regime}")
        
    except Exception as e:
        print(f"❌ Gradient Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        gradient_analysis = None
    
    # 3. Detailed Comparison
    if gbm_analysis and gradient_analysis:
        print(f"\n📊 DETAILED COMPARISON")
        print("=" * 40)
        
        # Compare entry probabilities
        if gbm_analysis.prob_entry_short and gradient_analysis.prob_entry_short:
            gbm_entry = gbm_analysis.prob_entry_short
            grad_entry = gradient_analysis.prob_entry_short
            diff_entry = grad_entry - gbm_entry
            
            print(f"📈 Entry Probability Comparison:")
            print(f"   GBM:      {gbm_entry:.1%}")
            print(f"   Gradient: {grad_entry:.1%}")
            print(f"   Difference: {diff_entry:+.1%} ({'Higher' if diff_entry > 0 else 'Lower'} with Gradient)")
            print()
        
        # Compare win probabilities
        if gbm_analysis.prob_win_short and gradient_analysis.prob_win_short:
            gbm_win = gbm_analysis.prob_win_short
            grad_win = gradient_analysis.prob_win_short
            diff_win = grad_win - gbm_win
            
            print(f"🎯 Win Probability Comparison:")
            print(f"   GBM:      {gbm_win:.1%}")
            print(f"   Gradient: {grad_win:.1%}")
            print(f"   Difference: {diff_win:+.1%} ({'Higher' if diff_win > 0 else 'Lower'} with Gradient)")
            print()
        
        # Compare expected values
        if gbm_analysis.expected_value_short and gradient_analysis.expected_value_short:
            gbm_ev = gbm_analysis.expected_value_short
            grad_ev = gradient_analysis.expected_value_short
            diff_ev = grad_ev - gbm_ev
            
            print(f"💰 Expected Value Comparison:")
            print(f"   GBM:      ${gbm_ev:.3f}")
            print(f"   Gradient: ${grad_ev:.3f}")
            print(f"   Difference: ${diff_ev:+.3f} ({'Better' if diff_ev > 0 else 'Worse'} with Gradient)")
            print()
        
        # Compare return rates
        if gbm_analysis.return_rate_short and gradient_analysis.return_rate_short:
            gbm_rate = gbm_analysis.return_rate_short
            grad_rate = gradient_analysis.return_rate_short
            diff_rate = grad_rate - gbm_rate
            
            print(f"📊 Return Rate Comparison:")
            print(f"   GBM:      {gbm_rate:.6f}/day")
            print(f"   Gradient: {grad_rate:.6f}/day")
            print(f"   Difference: {diff_rate:+.6f}/day ({'Better' if diff_rate > 0 else 'Worse'} with Gradient)")
            print()
        
        # Methodology comparison
        print(f"🔬 METHODOLOGY COMPARISON:")
        print(f"   GBM Approach:")
        print(f"   • Continuous price random walk")
        print(f"   • Assumes log-normal price distribution")
        print(f"   • Uses historical drift and volatility")
        print(f"   • Coordinate-dependent")
        print()
        print(f"   Gradient Approach:")
        print(f"   • Candle geometry evolution patterns")
        print(f"   • Data-driven cluster transitions")
        print(f"   • Captures market microstructure")
        print(f"   • Coordinate-invariant gradients")
        print()
        
        # Determine which is more conservative/aggressive
        if (gbm_analysis.expected_value_short and gradient_analysis.expected_value_short and
            gbm_analysis.prob_entry_short and gradient_analysis.prob_entry_short):
            
            gbm_conservative = gbm_analysis.expected_value_short < gradient_analysis.expected_value_short
            
            print(f"📋 ANALYSIS CHARACTERISTICS:")
            print(f"   More Conservative: {'GBM' if gbm_conservative else 'Gradient'}")
            print(f"   More Aggressive: {'Gradient' if gbm_conservative else 'GBM'}")
            print()
            
            # Recommendation
            if gradient_analysis.expected_value_short > 0:
                if gbm_analysis.expected_value_short > 0:
                    print(f"✅ RECOMMENDATION: Both methods suggest POSITIVE expected value")
                    print(f"   Gradient method shows {'higher' if gradient_analysis.expected_value_short > gbm_analysis.expected_value_short else 'lower'} confidence")
                else:
                    print(f"⚠️  RECOMMENDATION: Mixed signals - Gradient positive, GBM negative")
            else:
                if gbm_analysis.expected_value_short < 0:
                    print(f"❌ RECOMMENDATION: Both methods suggest NEGATIVE expected value")
                    print(f"   Trade likely unprofitable")
                else:
                    print(f"⚠️  RECOMMENDATION: Mixed signals - GBM positive, Gradient negative")
    
    print(f"\n✅ Comparison analysis complete!")
    return True

if __name__ == "__main__":
    test_gradient_vs_gbm()