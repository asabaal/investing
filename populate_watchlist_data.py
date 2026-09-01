#!/usr/bin/env python3
"""
WATCHLIST DATA POPULATION SCRIPT

Populates the database with 15min and daily data for ALL securities
from your watchlist CSV file using Alpha Vantage API.

Based on your existing collection pattern.
"""

import pandas as pd
import csv
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from direct_api_puller import get_daily_data_api, get_intraday_data_api
from market_data_database import MarketDataDatabase

# Configuration
DAILY_START_DATE = '2022-01-01'  # 3 years of daily data
INTRADAY_START_DATE = '2024-08-01'  # 1 year of 15min data
END_DATE = datetime.now().strftime('%Y-%m-%d')
INTRADAY_INTERVAL = '15min'
RATE_LIMIT_SECONDS = 12.0  # Alpha Vantage allows 5 calls/min
CHECKPOINT_EVERY = 10  # Save progress every N symbols
LOG_FILE = 'watchlist_population.log'

class ProgressLogger:
    """Track and log collection progress"""
    def __init__(self):
        self.start_time = datetime.now()
        self.log_file = LOG_FILE
        self.completed_symbols = []
        self.failed_symbols = []
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def symbol_completed(self, symbol, daily_records, intraday_records, time_taken):
        self.completed_symbols.append({
            'symbol': symbol,
            'daily_records': daily_records,
            'intraday_records': intraday_records,
            'time': time_taken
        })
        
    def symbol_failed(self, symbol, error):
        self.failed_symbols.append({
            'symbol': symbol,
            'error': str(error)
        })

def load_watchlist_tickers():
    """Load ticker symbols from watchlist CSV"""
    watchlist_path = Path(__file__).parent / "watchlist" / "WATCHLIST - Sheet1.csv"
    
    if not watchlist_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {watchlist_path}")
    
    tickers = []
    with open(watchlist_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = row['Ticker'].strip().upper()
            if ticker and ticker != 'TICKER':
                tickers.append(ticker)
    
    return tickers

def check_existing_data(symbol, db):
    """Check what data already exists for a symbol"""
    try:
        # Check daily data
        daily_data = db._get_daily_data_direct(symbol, skip_update=True)
        daily_count = len(daily_data) if not daily_data.empty else 0
        
        # Check 15min data  
        intraday_data = db._get_intraday_data_direct(symbol, interval=INTRADAY_INTERVAL, skip_update=True)
        intraday_count = len(intraday_data) if not intraday_data.empty else 0
        
        return daily_count, intraday_count
        
    except Exception:
        return 0, 0

def collect_daily_data(symbol, db, logger):
    """Collect daily data for symbol"""
    logger.log(f"   📈 Collecting daily data for {symbol}...")
    
    try:
        # Get daily data from API
        daily_data = get_daily_data_api(symbol, DAILY_START_DATE, END_DATE)
        
        if not daily_data.empty:
            # Store in database
            daily_for_db = daily_data.copy()
            daily_for_db['datetime'] = daily_for_db.index
            db._store_daily_data(symbol, daily_for_db)
            
            logger.log(f"   ✅ Daily: {len(daily_data):,} records from {daily_data.index.min().date()} to {daily_data.index.max().date()}")
            return len(daily_data)
        else:
            logger.log(f"   ⚠️ Daily: No data available")
            return 0
            
    except Exception as e:
        logger.log(f"   ❌ Daily failed: {e}")
        return 0

def collect_intraday_data(symbol, db, logger):
    """Collect 15min intraday data for symbol"""
    logger.log(f"   📊 Collecting 15min data for {symbol}...")
    
    try:
        # Generate monthly chunks for better API handling
        start_dt = pd.to_datetime(INTRADAY_START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        
        total_collected = 0
        current = start_dt.replace(day=1)
        
        while current <= end_dt:
            # Calculate month range
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            
            month_start = current.strftime('%Y-%m-%d')
            month_end = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                # Get month's data
                month_data = get_intraday_data_api(symbol, INTRADAY_INTERVAL, month_start, month_end)
                
                if not month_data.empty:
                    # Store in database
                    month_for_db = month_data.copy()
                    month_for_db['datetime'] = month_for_db.index
                    db._store_intraday_data(symbol, month_for_db, INTRADAY_INTERVAL)
                    
                    total_collected += len(month_data)
                
                # Rate limiting between month requests
                time.sleep(RATE_LIMIT_SECONDS)
                
            except Exception as e:
                logger.log(f"   ⚠️ Month {month_start} failed: {e}")
            
            current = next_month
        
        if total_collected > 0:
            logger.log(f"   ✅ 15min: {total_collected:,} records collected")
        else:
            logger.log(f"   ⚠️ 15min: No data available")
            
        return total_collected
        
    except Exception as e:
        logger.log(f"   ❌ 15min failed: {e}")
        return 0

def collect_symbol_data(symbol, db, logger):
    """Collect both daily and 15min data for a single symbol"""
    symbol_start_time = datetime.now()
    
    logger.log(f"🎯 Processing {symbol}...")
    
    # Check existing data
    existing_daily, existing_intraday = check_existing_data(symbol, db)
    logger.log(f"📊 {symbol} existing data: {existing_daily:,} daily, {existing_intraday:,} 15min")
    
    # Collect daily data
    daily_collected = collect_daily_data(symbol, db, logger)
    time.sleep(RATE_LIMIT_SECONDS)  # Rate limit between data types
    
    # Collect 15min data
    intraday_collected = collect_intraday_data(symbol, db, logger)
    
    # Final check
    try:
        final_daily, final_intraday = check_existing_data(symbol, db)
        daily_growth = final_daily - existing_daily
        intraday_growth = final_intraday - existing_intraday
        
        symbol_time = datetime.now() - symbol_start_time
        
        logger.log(f"✅ {symbol} COMPLETE: Daily +{daily_growth:,} ({final_daily:,} total), "
                  f"15min +{intraday_growth:,} ({final_intraday:,} total) - {symbol_time}")
        
        logger.symbol_completed(symbol, daily_collected, intraday_collected, symbol_time)
        
        return {
            'symbol': symbol,
            'success': True,
            'daily_collected': daily_collected,
            'intraday_collected': intraday_collected,
            'daily_growth': daily_growth,
            'intraday_growth': intraday_growth,
            'final_daily': final_daily,
            'final_intraday': final_intraday,
            'time_taken': symbol_time
        }
        
    except Exception as e:
        logger.log(f"❌ {symbol} final check failed: {e}")
        logger.symbol_failed(symbol, e)
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }

def create_checkpoint(completed_symbols, results, logger):
    """Create a checkpoint file with progress"""
    successful = [r for r in results if r.get('success', False)]
    
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'completed_symbols': completed_symbols,
        'total_processed': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful),
        'total_daily_collected': sum(r.get('daily_collected', 0) for r in successful),
        'total_intraday_collected': sum(r.get('intraday_collected', 0) for r in successful),
        'total_daily_growth': sum(r.get('daily_growth', 0) for r in successful),
        'total_intraday_growth': sum(r.get('intraday_growth', 0) for r in successful)
    }
    
    checkpoint_file = f"watchlist_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(f"WATCHLIST POPULATION CHECKPOINT\n")
        f.write(f"================================\n")
        f.write(f"Timestamp: {checkpoint_data['timestamp']}\n")
        f.write(f"Processed: {checkpoint_data['total_processed']} symbols\n")
        f.write(f"Successful: {checkpoint_data['successful']}\n")
        f.write(f"Failed: {checkpoint_data['failed']}\n")
        f.write(f"Daily records collected: {checkpoint_data['total_daily_collected']:,}\n")
        f.write(f"15min records collected: {checkpoint_data['total_intraday_collected']:,}\n")
        f.write(f"Daily database growth: {checkpoint_data['total_daily_growth']:,}\n")
        f.write(f"15min database growth: {checkpoint_data['total_intraday_growth']:,}\n")
        f.write(f"\nCompleted symbols: {', '.join(completed_symbols)}\n")
    
    logger.log(f"💾 Checkpoint saved: {checkpoint_file}")

def main():
    """Main watchlist population function"""
    logger = ProgressLogger()
    
    logger.log("🚀 WATCHLIST DATA POPULATION STARTING!")
    logger.log("=" * 60)
    logger.log(f"📈 Daily data: {DAILY_START_DATE} to {END_DATE}")
    logger.log(f"📊 15min data: {INTRADAY_START_DATE} to {END_DATE}")
    logger.log(f"⏱️  Rate limit: {RATE_LIMIT_SECONDS} seconds between requests")
    
    # Load watchlist
    logger.log("📋 Loading watchlist...")
    try:
        tickers = load_watchlist_tickers()
        logger.log(f"✅ Found {len(tickers)} symbols in watchlist")
        logger.log(f"🎯 Symbols: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    except Exception as e:
        logger.log(f"❌ Failed to load watchlist: {e}")
        return
    
    # Initialize database
    logger.log("🔧 Connecting to database...")
    db = MarketDataDatabase()
    
    # Estimate time
    estimated_minutes = len(tickers) * 2 * RATE_LIMIT_SECONDS / 60  # 2 data types per symbol
    logger.log(f"⏱️  Estimated time: ~{estimated_minutes:.0f} minutes")
    
    # Confirm start
    print(f"\n📋 About to populate data for {len(tickers)} watchlist symbols")
    print(f"⏱️  Estimated time: {estimated_minutes:.0f} minutes")
    print(f"🔄 This will collect both daily and 15min data")
    
    response = input(f"\nProceed with watchlist population? (y/N): ")
    if response.lower() != 'y':
        logger.log("❌ Population cancelled by user")
        return
    
    # Start collection
    results = []
    completed_symbols = []
    collection_start = datetime.now()
    
    for i, symbol in enumerate(tickers, 1):
        logger.log(f"\n🎯 [{i}/{len(tickers)}] Processing {symbol}...")
        
        try:
            result = collect_symbol_data(symbol, db, logger)
            results.append(result)
            
            if result['success']:
                completed_symbols.append(symbol)
            
            # Progress update
            elapsed = datetime.now() - collection_start
            if i > 0:
                avg_time_per_symbol = elapsed.total_seconds() / i
                remaining_symbols = len(tickers) - i
                eta = timedelta(seconds=avg_time_per_symbol * remaining_symbols)
                
                logger.log(f"📈 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%) - ETA: {eta}")
            
            # Create checkpoint
            if i % CHECKPOINT_EVERY == 0:
                create_checkpoint(completed_symbols, results, logger)
                
        except Exception as e:
            logger.log(f"💥 CRITICAL ERROR processing {symbol}: {e}")
            results.append({'symbol': symbol, 'success': False, 'error': str(e)})
            continue
    
    # Final summary
    total_time = datetime.now() - collection_start
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    total_daily = sum(r.get('daily_collected', 0) for r in successful)
    total_intraday = sum(r.get('intraday_collected', 0) for r in successful)
    total_daily_growth = sum(r.get('daily_growth', 0) for r in successful)
    total_intraday_growth = sum(r.get('intraday_growth', 0) for r in successful)
    
    logger.log(f"\n🎉 WATCHLIST POPULATION COMPLETE!")
    logger.log(f"⏱️  Total time: {total_time}")
    logger.log(f"✅ Successful: {len(successful)}/{len(tickers)} symbols")
    logger.log(f"❌ Failed: {len(failed)} symbols")
    logger.log(f"📈 Daily records collected: {total_daily:,}")
    logger.log(f"📊 15min records collected: {total_intraday:,}")
    logger.log(f"📈 Daily database growth: {total_daily_growth:,}")
    logger.log(f"📊 15min database growth: {total_intraday_growth:,}")
    logger.log("=" * 60)
    
    if failed:
        failed_symbols = [r['symbol'] for r in failed]
        logger.log(f"❌ Failed symbols: {', '.join(failed_symbols)}")
    
    logger.log(f"🚀 WATCHLIST POPULATION COMPLETE! Check {LOG_FILE} for details.")
    
    # Final checkpoint
    create_checkpoint(completed_symbols, results, logger)

if __name__ == "__main__":
    main()