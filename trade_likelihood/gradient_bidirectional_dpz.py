#!/usr/bin/env python3

"""
Gradient-Enhanced Bidirectional DPZ Analysis

Integrates the invariant gradient approach with bidirectional trade analysis
for more accurate trade probability modeling that captures market microstructure.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add parent directory for imports
sys.path.insert(0, '/home/asabaal/asabaal_ventures/repos/investing')

# Import all required components
from trade_likelihood.bidirectional_analyzer import BidirectionalTradeAnalyzer, BidirectionalSetup
from trade_likelihood.gradient_trade_analyzer import GradientTradeAnalyzer
from trade_likelihood.gradient_candle_clustering import GradientCandleAnalyzer
from trade_likelihood.data_structures import TradeSetup, TradeAnalysis
from market_data_database import MarketDataDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_enhanced_dpz_analysis():
    """Create enhanced DPZ analysis using gradient approach"""
    print("🚀 GRADIENT-ENHANCED BIDIRECTIONAL DPZ ANALYSIS")
    print("=" * 60)
    
    # Initialize components
    market_db = MarketDataDatabase()
    gradient_analyzer = GradientCandleAnalyzer(market_db)
    gradient_trade_analyzer = GradientTradeAnalyzer(gradient_analyzer)
    
    # Setup gradient analysis for DPZ
    print("📊 Setting up gradient analysis for DPZ...")
    success = gradient_trade_analyzer.setup_symbol_analysis('DPZ', '2024-01-01', '2025-08-31')
    
    if not success:
        print("❌ Failed to setup gradient analysis for DPZ")
        return None
    
    # Load DPZ data to get current price and OHLC
    print("💰 Loading DPZ market data...")
    df = market_db.get_data('DPZ', '2025-07-01', '2025-08-31', 'daily')
    
    if df.empty:
        print("❌ No DPZ data found")
        return None
    
    close_col = 'Close' if 'Close' in df.columns else 'close'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    
    current_price = df[close_col].iloc[-1]
    current_ohlc = {
        'open': df[open_col].iloc[-1],
        'high': df[high_col].iloc[-1],
        'low': df[low_col].iloc[-1],
        'close': current_price
    }
    
    print(f"📈 Current DPZ price: ${current_price:.2f}")
    print(f"📊 Current OHLC: O={current_ohlc['open']:.2f}, H={current_ohlc['high']:.2f}, L={current_ohlc['low']:.2f}, C={current_ohlc['close']:.2f}")
    
    # Create bidirectional setup (same as original analysis)
    bidirectional_setup = BidirectionalSetup(
        symbol='DPZ',
        trade_date=datetime(2025, 9, 1),
        current_price=current_price,
        long_entry=441.0,    # Long entry
        long_stop=435.0,     # Long stop  
        long_target=464.0,   # Long target
        short_entry=464.80,  # Short entry
        short_stop=468.46,   # Short stop
        short_target=441.47, # Short target
        max_time_window=5.0
    )
    
    print("\n✅ Created bidirectional setup with gradient enhancement:")
    print(f"   Long:  Entry ${bidirectional_setup.long_entry:.2f}, Stop ${bidirectional_setup.long_stop:.2f}, Target ${bidirectional_setup.long_target:.2f} (R:R {bidirectional_setup.get_long_rr():.2f}:1)")
    print(f"   Short: Entry ${bidirectional_setup.short_entry:.2f}, Stop ${bidirectional_setup.short_stop:.2f}, Target ${bidirectional_setup.short_target:.2f} (R:R {bidirectional_setup.get_short_rr():.2f}:1)")
    
    # Analyze using gradient approach
    print("\n🧠 Analyzing trades using gradient-based evolution...")
    
    # Create trade setups for gradient analysis
    long_setup = bidirectional_setup.to_long_trade_setup(current_price)
    short_setup = bidirectional_setup.to_short_trade_setup(current_price)
    
    # Run gradient-based analysis across multiple historical windows
    historical_windows = [60, 90, 120, 180]
    print(f"📊 Analyzing across {len(historical_windows)} gradient windows: {historical_windows}")
    
    print("📊 Analyzing long setup with gradient method...")
    gradient_long_analyses = gradient_trade_analyzer.analyze_trade_multiple_windows(
        long_setup, current_ohlc, 'DPZ', datetime(2025, 9, 1), historical_windows, time_horizon_days=5
    )
    
    print("📊 Analyzing short setup with gradient method...")
    gradient_short_analyses = gradient_trade_analyzer.analyze_trade_multiple_windows(
        short_setup, current_ohlc, 'DPZ', datetime(2025, 9, 1), historical_windows, time_horizon_days=5
    )
    
    # Compare with traditional bidirectional analyzer
    print("\n🔄 Running traditional bidirectional analysis for comparison...")
    traditional_analyzer = BidirectionalTradeAnalyzer()
    traditional_analysis = traditional_analyzer.analyze_bidirectional_setup(bidirectional_setup)
    
    # Present comparative results
    print("\n🎯 GRADIENT HISTORICAL WINDOW SENSITIVITY")
    print("=" * 60)
    
    # Show gradient results across all windows
    if gradient_long_analyses and gradient_short_analyses:
        print("📊 GRADIENT RESULTS BY HISTORICAL WINDOW:")
        print(f"{'Window':<8} {'Long Entry':<12} {'Short Entry':<12} {'Long Win':<10} {'Short Win':<10} {'Total RR':<12} {'Total EV':<10}")
        print("-" * 80)
        
        gradient_window_results = {}
        for window in sorted(gradient_long_analyses.keys()):
            if window in gradient_short_analyses:
                long_analysis = gradient_long_analyses[window]
                short_analysis = gradient_short_analyses[window]
                
                long_entry = long_analysis.prob_entry_long or 0
                short_entry = short_analysis.prob_entry_short or 0
                long_win = long_analysis.prob_win_long or 0
                short_win = short_analysis.prob_win_short or 0
                
                long_rr = long_analysis.return_rate_long or 0
                short_rr = short_analysis.return_rate_short or 0
                total_rr = long_rr + short_rr
                
                long_ev = long_analysis.expected_value_long or 0
                short_ev = short_analysis.expected_value_short or 0
                total_ev = long_ev + short_ev
                
                gradient_window_results[window] = {
                    'long_entry': long_entry,
                    'short_entry': short_entry,
                    'long_win': long_win,
                    'short_win': short_win,
                    'total_return_rate': total_rr,
                    'total_expected_value': total_ev,
                    'long_analysis': long_analysis,
                    'short_analysis': short_analysis
                }
                
                print(f"{window:<8} {long_entry:<11.1%} {short_entry:<11.1%} {long_win:<9.1%} {short_win:<9.1%} {total_rr:<11.4f} ${total_ev:<9.3f}")
    
    print("\n🎯 TRADITIONAL VS GRADIENT COMPARISON")
    print("=" * 60)
    
    # Get best traditional analysis window
    best_traditional_long = None
    best_traditional_short = None
    best_trad_window = None
    
    if traditional_analysis.long_analysis and traditional_analysis.short_analysis:
        best_combined_return = float('-inf')
        for window in traditional_analysis.long_analysis.keys():
            if window in traditional_analysis.short_analysis:
                long_return = traditional_analysis.long_analysis[window].return_rate_total or 0
                short_return = traditional_analysis.short_analysis[window].return_rate_total or 0
                combined_return = long_return + short_return
                if combined_return > best_combined_return:
                    best_combined_return = combined_return
                    best_trad_window = window
                    best_traditional_long = traditional_analysis.long_analysis[window]
                    best_traditional_short = traditional_analysis.short_analysis[window]
    
    # Get best gradient window
    best_gradient_long = None
    best_gradient_short = None
    best_grad_window = None
    
    if gradient_window_results:
        best_gradient_return = float('-inf')
        for window, results in gradient_window_results.items():
            if results['total_return_rate'] > best_gradient_return:
                best_gradient_return = results['total_return_rate']
                best_grad_window = window
                best_gradient_long = results['long_analysis']
                best_gradient_short = results['short_analysis']
    
    if best_traditional_long and best_traditional_short and best_gradient_long and best_gradient_short:
        print(f"Best Traditional Window: {best_trad_window}")
        print(f"Best Gradient Window: {best_grad_window}")
        
        # Entry probabilities comparison
        print("\n📈 ENTRY PROBABILITIES:")
        grad_long_entry = best_gradient_long.prob_entry_long or 0
        grad_short_entry = best_gradient_short.prob_entry_short or 0
        trad_long_entry = best_traditional_long.get_combined_entry_probability() or 0
        trad_short_entry = best_traditional_short.get_combined_entry_probability() or 0
        
        print(f"  Long Entry:")
        print(f"    Gradient:    {grad_long_entry:.1%}")
        print(f"    Traditional: {trad_long_entry:.1%}")
        print(f"    Difference:  {(grad_long_entry - trad_long_entry):.1%}")
        
        print(f"  Short Entry:")
        print(f"    Gradient:    {grad_short_entry:.1%}")
        print(f"    Traditional: {trad_short_entry:.1%}")
        print(f"    Difference:  {(grad_short_entry - trad_short_entry):.1%}")
        
        # Win probabilities comparison
        print("\n🏆 WIN PROBABILITIES (given entry):")
        grad_long_win = best_gradient_long.prob_win_long or 0
        grad_short_win = best_gradient_short.prob_win_short or 0
        trad_long_win = best_traditional_long.prob_win_long or 0
        trad_short_win = best_traditional_short.prob_win_short or 0
        
        print(f"  Long Win:")
        print(f"    Gradient:    {grad_long_win:.1%}")
        print(f"    Traditional: {trad_long_win:.1%}")
        print(f"    Difference:  {(grad_long_win - trad_long_win):.1%}")
        
        print(f"  Short Win:")
        print(f"    Gradient:    {grad_short_win:.1%}")
        print(f"    Traditional: {trad_short_win:.1%}")
        print(f"    Difference:  {(grad_short_win - trad_short_win):.1%}")
        
        # Expected values comparison
        print("\n💰 EXPECTED VALUES:")
        grad_long_ev = best_gradient_long.expected_value_long or 0
        grad_short_ev = best_gradient_short.expected_value_short or 0
        grad_total_ev = grad_long_ev + grad_short_ev
        
        trad_long_ev = best_traditional_long.expected_value_total or 0
        trad_short_ev = best_traditional_short.expected_value_total or 0
        trad_total_ev = trad_long_ev + trad_short_ev
        
        print(f"  Long Expected Value:")
        print(f"    Gradient:    ${grad_long_ev:.3f}")
        print(f"    Traditional: ${trad_long_ev:.3f}")
        print(f"    Difference:  ${(grad_long_ev - trad_long_ev):.3f}")
        
        print(f"  Short Expected Value:")
        print(f"    Gradient:    ${grad_short_ev:.3f}")
        print(f"    Traditional: ${trad_short_ev:.3f}")
        print(f"    Difference:  ${(grad_short_ev - trad_short_ev):.3f}")
        
        print(f"  Total Expected Value:")
        print(f"    Gradient:    ${grad_total_ev:.3f}")
        print(f"    Traditional: ${trad_total_ev:.3f}")
        print(f"    Difference:  ${(grad_total_ev - trad_total_ev):.3f}")
        
        # Return rates comparison
        print("\n📊 RETURN RATES (per day):")
        grad_long_rr = best_gradient_long.return_rate_long or 0
        grad_short_rr = best_gradient_short.return_rate_short or 0
        grad_total_rr = grad_long_rr + grad_short_rr
        
        trad_long_rr = best_traditional_long.return_rate_total or 0
        trad_short_rr = best_traditional_short.return_rate_total or 0
        trad_total_rr = trad_long_rr + trad_short_rr
        
        print(f"  Long Return Rate:")
        print(f"    Gradient:    {grad_long_rr:.4f}/day")
        print(f"    Traditional: {trad_long_rr:.4f}/day")
        print(f"    Difference:  {(grad_long_rr - trad_long_rr):.4f}/day")
        
        print(f"  Short Return Rate:")
        print(f"    Gradient:    {grad_short_rr:.4f}/day")
        print(f"    Traditional: {trad_short_rr:.4f}/day")
        print(f"    Difference:  {(grad_short_rr - trad_short_rr):.4f}/day")
        
        print(f"  Total Return Rate:")
        print(f"    Gradient:    {grad_total_rr:.4f}/day")
        print(f"    Traditional: {trad_total_rr:.4f}/day")
        print(f"    Difference:  {(grad_total_rr - trad_total_rr):.4f}/day")
        
        # Strategy recommendations
        print("\n💡 STRATEGY RECOMMENDATIONS:")
        
        # Combined entry probabilities
        grad_any_entry = grad_long_entry + grad_short_entry - (grad_long_entry * grad_short_entry)
        trad_any_entry = trad_long_entry + trad_short_entry - (trad_long_entry * trad_short_entry)
        
        print(f"  Any Entry Probability:")
        print(f"    Gradient:    {grad_any_entry:.1%}")
        print(f"    Traditional: {trad_any_entry:.1%}")
        
        # Strategy attractiveness
        grad_attractive = (grad_total_rr > 0.1 and grad_any_entry > 0.3)
        trad_attractive = (trad_total_rr > 0.1 and trad_any_entry > 0.3)
        
        print(f"  Strategy Attractive:")
        print(f"    Gradient:    {'✅ Yes' if grad_attractive else '❌ No'}")
        print(f"    Traditional: {'✅ Yes' if trad_attractive else '❌ No'}")
        
        # Method advantages
        print("\n🔬 METHOD ANALYSIS:")
        print("  Gradient Method Advantages:")
        print("    • Captures candle geometry evolution patterns")
        print("    • Coordinate-invariant (handles different price scales)")
        print("    • Uses market microstructure data")
        print("    • Learns from actual candle formation dynamics")
        
        print("  Traditional Method Advantages:")
        print("    • Based on proven GBM financial models")
        print("    • Faster computation")
        print("    • Well-understood mathematical properties")
        print("    • Good for continuous price modeling")
        
        # Final recommendation
        if grad_total_rr > trad_total_rr * 1.1:  # 10% better
            recommendation = "GRADIENT METHOD"
            reason = "significantly higher expected returns"
        elif trad_total_rr > grad_total_rr * 1.1:
            recommendation = "TRADITIONAL METHOD"
            reason = "significantly higher expected returns"
        elif abs(grad_total_rr - trad_total_rr) < 0.01:
            recommendation = "SIMILAR RESULTS"
            reason = "both methods show comparable performance"
        else:
            recommendation = "GRADIENT METHOD"
            reason = "better microstructure modeling"
        
        print(f"\n🎯 FINAL RECOMMENDATION: {recommendation}")
        print(f"   Reason: {reason}")
        
        return {
            'gradient_analysis': {
                'all_windows': {
                    'long': gradient_long_analyses,
                    'short': gradient_short_analyses
                },
                'best_window': {
                    'window': best_grad_window,
                    'long': best_gradient_long,
                    'short': best_gradient_short
                }
            },
            'traditional_analysis': traditional_analysis,
            'comparison_summary': {
                'gradient_total_return_rate': grad_total_rr,
                'traditional_total_return_rate': trad_total_rr,
                'gradient_any_entry_prob': grad_any_entry,
                'traditional_any_entry_prob': trad_any_entry,
                'recommendation': recommendation,
                'reason': reason,
                'gradient_window_results': gradient_window_results
            },
            'setup': bidirectional_setup
        }
    
    else:
        print("❌ Traditional analysis failed to produce results")
        return {
            'gradient_analysis': {
                'all_windows': {
                    'long': gradient_long_analyses,
                    'short': gradient_short_analyses
                }
            },
            'traditional_analysis': None,
            'setup': bidirectional_setup
        }

def generate_gradient_enhanced_report(analysis_results):
    """Generate enhanced report with gradient analysis"""
    if not analysis_results:
        print("❌ No analysis results to generate report")
        return
        
    print("\n📄 GENERATING GRADIENT-ENHANCED REPORT...")
    
    try:
        from trade_likelihood.report_generator import BacktestReportGenerator
        
        report_gen = BacktestReportGenerator()
        
        # Prepare data for report
        setup = analysis_results['setup']
        
        # Use all windows for comprehensive report
        gradient_long_dict = analysis_results['gradient_analysis']['all_windows']['long']
        gradient_short_dict = analysis_results['gradient_analysis']['all_windows']['short']
        
        # Add window results summary for enhanced reporting
        window_results = analysis_results['comparison_summary'].get('gradient_window_results', {})
        
        # Generate enhanced report
        filename = report_gen.generate_bidirectional_report(
            long_analysis=gradient_long_dict,
            short_analysis=gradient_short_dict,
            bidirectional_setup=setup,
            output_path="gradient_enhanced_DPZ_20250901.html"
        )
        
        print(f"✅ Generated gradient-enhanced report: {filename}")
        
        # Also save comparison data
        comparison_summary = analysis_results.get('comparison_summary', {})
        if comparison_summary:
            print(f"📊 Comparison Summary:")
            print(f"   Gradient Return Rate: {comparison_summary['gradient_total_return_rate']:.4f}/day")
            if 'traditional_total_return_rate' in comparison_summary:
                print(f"   Traditional Return Rate: {comparison_summary['traditional_total_return_rate']:.4f}/day")
            print(f"   Recommendation: {comparison_summary['recommendation']}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the enhanced analysis
    print("Starting gradient-enhanced DPZ bidirectional analysis...")
    
    results = create_enhanced_dpz_analysis()
    
    if results:
        # Generate the report
        report_file = generate_gradient_enhanced_report(results)
        
        if report_file:
            print(f"\n🎉 Analysis complete! Report saved as: {report_file}")
        else:
            print("\n✅ Analysis complete! (Report generation failed)")
    else:
        print("\n❌ Analysis failed!")