#!/usr/bin/env python3
"""
Market Phase Space Analysis Runner
Easy-to-use script for running different types of market analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from market_phase_space_processor import MarketPhaseSpaceProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Popular symbol collections
SYMBOL_COLLECTIONS = {
    'mega_cap': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA'],
    'sp500_etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO'],
    'sector_etfs': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU'],
    'volatility': ['VIX', 'UVXY', 'VXX', 'SVXY'],
    'bonds': ['TLT', 'SHY', 'LQD', 'HYG', 'TIP'],
    'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
    'crypto_related': ['COIN', 'MSTR', 'RIOT', 'MARA'],
    'meme_stocks': ['GME', 'AMC', 'BBBY', 'NOK'],
    'all_popular': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
                   'SPY', 'QQQ', 'TLT', 'GLD', 'VIX']
}

def check_env_vars():
    """Check if required environment variables are set."""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("❌ ALPHA_VANTAGE_API_KEY environment variable not set!")
        logger.info("   Please set it with: export ALPHA_VANTAGE_API_KEY=your_key_here")
        return False
    
    logger.info(f"✅ AlphaVantage API key found: {api_key[:8]}...")
    return True

def run_single_security_analysis(processor, symbol, days_back):
    """Run analysis for a single security."""
    logger.info(f"🎯 Running single security analysis for {symbol}")
    
    # Update data first
    update_result = processor.update_database([symbol], days_back=days_back)
    if not update_result.get(symbol, False):
        logger.error(f"❌ Failed to update data for {symbol}")
        return None
    
    # Run analysis
    analysis = processor.analyze_security_phase_space(symbol)
    if analysis:
        logger.info(f"✅ Analysis complete for {symbol}")
        logger.info(f"   📊 {analysis.n_candles} candles processed")
        logger.info(f"   🎯 {analysis.phase_space_coverage:.1f}% phase space coverage")
        logger.info(f"   📁 {len(analysis.files_generated)} files generated")
        
        # Show file locations
        for file_path in analysis.files_generated:
            logger.info(f"      📄 {Path(file_path).name}")
    
    return analysis

def run_batch_analysis(processor, symbols, days_back):
    """Run batch analysis for multiple securities."""
    logger.info(f"🚀 Running batch analysis for {len(symbols)} securities")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    
    results = processor.batch_process_securities(
        symbols=symbols,
        update_data=True
    )
    
    logger.info(f"🎉 Batch analysis complete!")
    logger.info(f"   ✅ Successful: {len(results)}/{len(symbols)} securities")
    
    # Show summary statistics
    if results:
        avg_candles = sum(r.n_candles for r in results.values()) / len(results)
        avg_coverage = sum(r.phase_space_coverage for r in results.values()) / len(results)
        total_files = sum(len(r.files_generated) for r in results.values())
        
        logger.info(f"   📊 Average candles: {avg_candles:.0f}")
        logger.info(f"   🎯 Average coverage: {avg_coverage:.1f}%")
        logger.info(f"   📁 Total files generated: {total_files}")
    
    return results

def run_market_regime_study(processor):
    """Run a comprehensive market regime study."""
    logger.info(f"🔬 Running comprehensive market regime study...")
    
    # Use a diverse set of securities
    study_symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX', 'AAPL', 'GOOGL', 'TSLA']
    
    logger.info(f"   Studying: {', '.join(study_symbols)}")
    
    results = processor.batch_process_securities(
        symbols=study_symbols,
        update_data=True
    )
    
    if results:
        # Analyze regime patterns
        logger.info(f"\n🔍 MARKET REGIME ANALYSIS:")
        
        regime_frequency = {}
        clustering_by_symbol = {}
        
        for symbol, analysis in results.items():
            clustering_by_symbol[symbol] = analysis.clustering_strength
            
            for regime in analysis.market_regimes:
                regime_frequency[regime] = regime_frequency.get(regime, 0) + 1
        
        # Most common regimes
        logger.info(f"   📈 Most common market regimes:")
        for regime, count in sorted(regime_frequency.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            logger.info(f"      {regime}: {count}/{len(results)} securities ({percentage:.1f}%)")
        
        # Clustering patterns
        logger.info(f"   🎯 Clustering strength by symbol:")
        for symbol, strength in sorted(clustering_by_symbol.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"      {symbol}: {strength:.3f}")
    
    return results

def list_available_symbols(processor):
    """List all symbols available in the database."""
    symbols = processor.get_available_symbols()
    
    if symbols:
        logger.info(f"📊 {len(symbols)} symbols available in database:")
        
        # Group by first letter for easier reading
        grouped = {}
        for symbol in symbols:
            first_letter = symbol[0]
            if first_letter not in grouped:
                grouped[first_letter] = []
            grouped[first_letter].append(symbol)
        
        for letter in sorted(grouped.keys()):
            symbols_str = ', '.join(grouped[letter])
            logger.info(f"   {letter}: {symbols_str}")
    else:
        logger.info("📊 No symbols found in database")
        logger.info("   Run with --update-popular to add some popular symbols")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Market Phase Space Analysis Runner')
    
    # Analysis type
    parser.add_argument('--mode', choices=['single', 'batch', 'regime-study', 'list-symbols'], 
                       default='single',
                       help='Analysis mode to run')
    
    # Symbol selection
    parser.add_argument('--symbol', type=str, 
                       help='Single symbol to analyze (for single mode)')
    
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='List of symbols to analyze (for batch mode)')
    
    parser.add_argument('--collection', choices=list(SYMBOL_COLLECTIONS.keys()),
                       help='Use a predefined collection of symbols')
    
    # Data options
    parser.add_argument('--days-back', type=int, default=365,
                       help='Number of days of historical data to fetch (default: 365)')
    
    parser.add_argument('--update-popular', action='store_true',
                       help='Update database with popular symbols first')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./phase_space_analysis',
                       help='Directory to store visualization files')
    
    parser.add_argument('--clean-old', type=int, metavar='DAYS',
                       help='Clean up visualization files older than N days')
    
    args = parser.parse_args()
    
    # Check environment
    if not check_env_vars():
        sys.exit(1)
    
    # Initialize processor
    logger.info(f"🚀 Initializing Market Phase Space Processor...")
    processor = MarketPhaseSpaceProcessor(output_dir=args.output_dir)
    
    # Clean old files if requested
    if args.clean_old:
        processor.cleanup_old_files(args.clean_old)
    
    # Update with popular symbols if requested
    if args.update_popular:
        logger.info(f"📊 Updating database with popular symbols...")
        processor.update_database(SYMBOL_COLLECTIONS['all_popular'], days_back=args.days_back)
    
    # Execute based on mode
    if args.mode == 'list-symbols':
        list_available_symbols(processor)
    
    elif args.mode == 'single':
        if not args.symbol:
            logger.error("❌ --symbol required for single mode")
            sys.exit(1)
        
        run_single_security_analysis(processor, args.symbol.upper(), args.days_back)
    
    elif args.mode == 'batch':
        symbols = []
        
        if args.collection:
            symbols = SYMBOL_COLLECTIONS[args.collection]
            logger.info(f"Using {args.collection} collection: {', '.join(symbols)}")
        elif args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            logger.error("❌ --symbols or --collection required for batch mode")
            sys.exit(1)
        
        run_batch_analysis(processor, symbols, args.days_back)
    
    elif args.mode == 'regime-study':
        run_market_regime_study(processor)
    
    logger.info(f"🎉 Analysis complete! Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()