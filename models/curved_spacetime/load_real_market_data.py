#!/usr/bin/env python3
"""
Load real market trajectory data from all available securities.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from market_data_database import MarketDataDatabase, get_default_database_path
from curved_candle_geometry import create_candle_metrics_from_ohlc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_symbols():
    """Get all symbols available in the database."""
    db_path = get_default_database_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT DISTINCT symbol FROM daily_data ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    return symbols

def load_real_market_trajectories():
    """Load market trajectory data from all available securities."""
    
    logger.info("🔍 Loading real market trajectories from all securities...")
    
    symbols = get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols: {symbols}")
    
    db = MarketDataDatabase()
    all_trajectory_points = []
    all_candle_data = []
    symbol_counts = {}
    successful_symbols = []
    
    # Get ALL available data from the database
    end_date = None  # No end date limit
    start_date = None  # No start date limit - get everything
    
    for symbol in symbols:
        try:
            logger.info(f"Loading {symbol}...")
            data = db.get_data(symbol, start_date=start_date, end_date=end_date)
            
            if len(data) < 10:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Convert to standard format
            market_data = pd.DataFrame({
                'open': data['Open'],
                'high': data['High'],
                'low': data['Low'],
                'close': data['Close'],
                'volume': data['Volume']
            })
            
            # Create candle metrics
            candle_metrics = create_candle_metrics_from_ohlc(market_data)
            
            # Extract valid trajectory points with proper data validation
            valid_points = 0
            invalid_points = 0
            for candle in candle_metrics:
                # Data validation checks
                if (not np.isfinite(candle.sentiment) or 
                    not np.isfinite(candle.upper_wick_ratio) or
                    abs(candle.sentiment) > 10 or  # Reasonable sentiment bounds
                    candle.upper_wick_ratio < 0 or candle.upper_wick_ratio > 1 or
                    candle.range_value <= 0):
                    invalid_points += 1
                    continue
                
                # Apply phase space constraint: |sentiment| + UWR ≤ 1
                if abs(candle.sentiment) + candle.upper_wick_ratio <= 1.0:
                    all_trajectory_points.append([candle.sentiment, candle.upper_wick_ratio])
                    all_candle_data.append({
                        'symbol': symbol,
                        'sentiment': candle.sentiment,
                        'uwr': candle.upper_wick_ratio,
                        'range_value': candle.range_value,
                        'volume': candle.volume
                    })
                    valid_points += 1
            
            if valid_points > 0:
                symbol_counts[symbol] = valid_points
                successful_symbols.append(symbol)
                logger.info(f"  {symbol}: {valid_points} valid points, {invalid_points} invalid points")
            elif invalid_points > 0:
                logger.warning(f"  {symbol}: 0 valid points, {invalid_points} invalid points")
            
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            continue
    
    # Convert to numpy array
    trajectory_points = np.array(all_trajectory_points) if all_trajectory_points else np.array([]).reshape(0, 2)
    candle_df = pd.DataFrame(all_candle_data)
    
    total_points = len(trajectory_points)
    logger.info(f"\\n📊 REAL MARKET TRAJECTORY DATA:")
    logger.info(f"   • Total securities: {len(symbols)}")
    logger.info(f"   • Successful securities: {len(successful_symbols)}")
    logger.info(f"   • Total trajectory points: {total_points}")
    
    if total_points > 0:
        logger.info(f"   • Sentiment range: [{trajectory_points[:, 0].min():.3f}, {trajectory_points[:, 0].max():.3f}]")
        logger.info(f"   • UWR range: [{trajectory_points[:, 1].min():.3f}, {trajectory_points[:, 1].max():.3f}]")
        
        # Show distribution by security
        logger.info(f"\\n📈 DISTRIBUTION BY SECURITY:")
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_points * 100
            logger.info(f"   • {symbol}: {count} points ({percentage:.1f}%)")
    
    return {
        'trajectory_points': trajectory_points,
        'candle_data': candle_df,
        'symbol_counts': symbol_counts,
        'successful_symbols': successful_symbols
    }

if __name__ == "__main__":
    market_data = load_real_market_trajectories()
    
    # Save the data
    output_dir = Path("phase_space_analysis")
    output_dir.mkdir(exist_ok=True)
    
    if len(market_data['trajectory_points']) > 0:
        np.savez(str(output_dir / "real_market_trajectories.npz"), 
                 trajectory_points=market_data['trajectory_points'],
                 symbols=market_data['successful_symbols'])
        
        market_data['candle_data'].to_csv(output_dir / "real_candle_data.csv", index=False)
        
        logger.info(f"✅ Saved real market trajectory data")
    else:
        logger.error("❌ No trajectory data to save")