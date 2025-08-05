#!/usr/bin/env python3
"""
Get all available market data from the database across all securities.
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
        # Get all symbols from daily_data table
        cursor = conn.execute("SELECT DISTINCT symbol FROM daily_data ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    
    return symbols

def load_all_market_trajectories():
    """Load market trajectory data from all available securities."""
    
    logger.info("🔍 Loading market trajectories from all available securities...")
    
    # Get available symbols
    symbols = get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols in database: {symbols}")
    
    db = MarketDataDatabase()
    all_trajectories = []
    all_candle_data = []
    symbol_counts = {}
    
    # Get recent data for analysis (last 200 days)
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    
    for symbol in symbols:
        try:
            logger.info(f"Loading data for {symbol}...")
            
            # Get market data
            data = db.get_data(symbol, start_date=start_date, end_date=end_date)
            
            if len(data) < 10:
                logger.warning(f"Insufficient data for {symbol} ({len(data)} candles)")
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
            
            # Extract trajectory points
            trajectory_points = []
            for candle in candle_metrics:
                # Apply phase space constraint: |sentiment| + UWR ≤ 1
                if abs(candle.sentiment) + candle.upper_wick_ratio <= 1.0:
                    trajectory_points.append([candle.sentiment, candle.upper_wick_ratio])
                    
                    # Store individual candle data for analysis
                    all_candle_data.append({
                        'symbol': symbol,
                        'sentiment': candle.sentiment,
                        'uwr': candle.upper_wick_ratio,
                        'range_value': candle.range_value,
                        'low_value': candle.low_value,
                        'volume': candle.volume
                    })
            
            if len(trajectory_points) > 0:
                trajectory_array = np.array(trajectory_points)
                all_trajectories.append({
                    'symbol': symbol,
                    'trajectory_points': trajectory_array,
                    'n_points': len(trajectory_points)
                })
                symbol_counts[symbol] = len(trajectory_points)
                
                logger.info(f"  {symbol}: {len(trajectory_points)} valid trajectory points")
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            continue
    
    # Combine all trajectory points
    combined_trajectory_points = []
    for traj in all_trajectories:
        combined_trajectory_points.extend(traj['trajectory_points'].tolist())
    
    combined_trajectory_points = np.array(combined_trajectory_points)
    
    # Create summary
    total_points = len(combined_trajectory_points)
    logger.info(f"\\n📊 COMBINED MARKET TRAJECTORY DATA:")
    logger.info(f"   • Securities: {len(symbols)}")
    logger.info(f"   • Total trajectory points: {total_points}")
    logger.info(f"   • Sentiment range: [{combined_trajectory_points[:, 0].min():.3f}, {combined_trajectory_points[:, 0].max():.3f}]")
    logger.info(f"   • UWR range: [{combined_trajectory_points[:, 1].min():.3f}, {combined_trajectory_points[:, 1].max():.3f}]")
    
    # Show distribution by security
    logger.info(f"\\n📈 POINTS BY SECURITY:")
    total_check = 0
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_points * 100
        logger.info(f"   • {symbol}: {count} points ({percentage:.1f}%)")
        total_check += count
    
    logger.info(f"   Total check: {total_check} points")
    
    return {
        'symbols': symbols,
        'individual_trajectories': all_trajectories,
        'combined_trajectory_points': combined_trajectory_points,
        'candle_data': pd.DataFrame(all_candle_data),
        'symbol_counts': symbol_counts
    }

if __name__ == "__main__":
    market_data = load_all_market_trajectories()
    
    # Save the combined data for use in other scripts
    output_path = Path("phase_space_analysis/combined_market_trajectories.npz")
    output_path.parent.mkdir(exist_ok=True)
    
    np.savez(str(output_path), 
             trajectory_points=market_data['combined_trajectory_points'],
             symbols=market_data['symbols'])
    
    # Save candle data
    candle_df = market_data['candle_data']
    candle_df.to_csv("phase_space_analysis/combined_candle_data.csv", index=False)
    
    logger.info(f"✅ Saved combined trajectory data to {output_path}")
    logger.info(f"✅ Saved candle data to phase_space_analysis/combined_candle_data.csv")