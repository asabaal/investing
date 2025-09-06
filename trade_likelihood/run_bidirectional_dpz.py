#!/usr/bin/env python3

import sys
import pandas as pd
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, '/home/asabaal/asabaal_ventures/repos/investing')

from trade_likelihood.bidirectional_analyzer import BidirectionalTradeAnalyzer, BidirectionalSetup
from trade_likelihood.data_structures import TradeSetup

# Create DPZ bidirectional analysis
print("🎯 INITIALIZING BIDIRECTIONAL DPZ ANALYSIS")
print("=" * 50)

analyzer = BidirectionalTradeAnalyzer()

# Load DPZ data to get current price
from market_data_database import MarketDataDatabase

market_db = MarketDataDatabase()
df = market_db.get_data('DPZ', '2025-07-01', '2025-08-31', 'daily')

if not df.empty:
    close_col = 'Close' if 'Close' in df.columns else 'close'
    prices = df[close_col]
    
    print(f"📊 Loaded {len(prices)} price points for DPZ")
    
    # Get current price
    current_price = prices.iloc[-1]
    print(f"Current price on 2025-08-31: ${current_price:.2f}")
    
    # Create bidirectional setup using the proper class
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
    
    print("✅ Created bidirectional setup")
    print(f"   Long R:R: {bidirectional_setup.get_long_rr():.2f}:1")
    print(f"   Short R:R: {bidirectional_setup.get_short_rr():.2f}:1")
    
    # Run bidirectional analysis
    analysis = analyzer.analyze_bidirectional_setup(bidirectional_setup)
    
    print(f"\n🎯 BIDIRECTIONAL DPZ ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Current Price: ${analysis.setup.current_price:.2f}")
    
    # Get the best analysis window
    best_long_analysis = None
    best_short_analysis = None
    best_window = None
    
    if analysis.long_analysis and analysis.short_analysis:
        # Find best window based on combined return rate
        best_combined_return = float('-inf')
        for window in analysis.long_analysis.keys():
            if window in analysis.short_analysis:
                long_return = analysis.long_analysis[window].return_rate_total or 0
                short_return = analysis.short_analysis[window].return_rate_total or 0
                combined_return = long_return + short_return
                if combined_return > best_combined_return:
                    best_combined_return = combined_return
                    best_window = window
                    best_long_analysis = analysis.long_analysis[window]
                    best_short_analysis = analysis.short_analysis[window]
        
        if best_long_analysis:
            print(f"Best Analysis Window: {best_window}")
            print(f"Market Regime: {best_long_analysis.market_params.regime}")
            print("")
            
            print("Long Setup:")
            print(f"  Entry: ${analysis.setup.long_entry:.2f}")  
            print(f"  Stop: ${analysis.setup.long_stop:.2f}")
            print(f"  Target: ${analysis.setup.long_target:.2f}")
            print(f"  R:R: {analysis.setup.get_long_rr():.2f}:1")
            print("")
            
            print("Short Setup:")
            print(f"  Entry: ${analysis.setup.short_entry:.2f}")
            print(f"  Stop: ${analysis.setup.short_stop:.2f}") 
            print(f"  Target: ${analysis.setup.short_target:.2f}")
            print(f"  R:R: {analysis.setup.get_short_rr():.2f}:1")
            print("")
            
            print("Entry Probabilities:")
            long_entry_prob = best_long_analysis.get_combined_entry_probability() or 0
            short_entry_prob = best_short_analysis.get_combined_entry_probability() or 0
            print(f"  Long Entry: {long_entry_prob:.1%}")
            print(f"  Short Entry: {short_entry_prob:.1%}")
            
            # Combined probabilities
            any_entry_prob = long_entry_prob + short_entry_prob - (long_entry_prob * short_entry_prob)
            both_entries_prob = long_entry_prob * short_entry_prob
            print(f"  Any Entry: {any_entry_prob:.1%}")
            print(f"  Both Entries: {both_entries_prob:.1%}")
            print("")
            
            print("Win Probabilities:")
            long_win_prob = best_long_analysis.prob_win_long or 0
            short_win_prob = best_short_analysis.prob_win_short or 0
            print(f"  Long Win: {long_win_prob:.1%}")
            print(f"  Short Win: {short_win_prob:.1%}")
            print("")
            
            print("Expected Values:")
            long_ev = best_long_analysis.expected_value_total or 0
            short_ev = best_short_analysis.expected_value_total or 0
            total_ev = long_ev + short_ev
            print(f"  Long EV: ${long_ev:.2f}")
            print(f"  Short EV: ${short_ev:.2f}") 
            print(f"  Total EV: ${total_ev:.2f}")
            print("")
            
            print("Return Rates:")
            long_return = best_long_analysis.return_rate_total or 0
            short_return = best_short_analysis.return_rate_total or 0
            total_return = long_return + short_return
            print(f"  Long Return Rate: {long_return:.4f}/day")
            print(f"  Short Return Rate: {short_return:.4f}/day")
            print(f"  Total Return Rate: {total_return:.4f}/day")
            print("")
            
            print("Strategy Metrics:")
            combined_rr = (analysis.setup.get_long_rr() + analysis.setup.get_short_rr()) / 2
            print(f"  Average R:R Ratio: {combined_rr:.2f}:1")
            
            # Strategy attractiveness
            is_attractive = (total_return > 0.1 and any_entry_prob > 0.3)
            print(f"  Strategy Attractive: {'✅' if is_attractive else '❌'}")
    else:
        print("❌ No analysis data available")
    
    # Generate bidirectional report
    print(f"\n📄 GENERATING BIDIRECTIONAL REPORT...")
    try:
        from trade_likelihood.report_generator import BacktestReportGenerator
        
        report_gen = BacktestReportGenerator()
        
        # Generate professional bidirectional report
        filename = report_gen.generate_bidirectional_report(
            long_analysis=analysis.long_analysis,
            short_analysis=analysis.short_analysis,
            bidirectional_setup=analysis.setup,
            output_path="bidirectional_DPZ_20250901.html"
        )
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("❌ No DPZ data found")