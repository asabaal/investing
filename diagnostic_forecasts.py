#!/usr/bin/env python3
"""
Diagnostic script to analyze forecast data structure in Symphony Trading System.
This script helps identify issues in the forecast data pipeline.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system modules
try:
    from prophet_forecasting import StockForecast, ProphetEnsemble
    from alpha_vantage_api import AlphaVantageClient
    from symphony_analyzer import SymphonyAnalyzer
    logger.info("Successfully imported system modules")
except ImportError as e:
    logger.error(f"Failed to import system modules: {e}")
    sys.exit(1)

def dump_json(data, filename):
    """
    Save data to a JSON file with pretty formatting.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=json_serialize)
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save data to {filename}: {e}")
        return False

def json_serialize(obj):
    """
    Custom JSON serializer to handle non-serializable objects.
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

def diagnose_forecast_structure(symbol, days=30):
    """
    Test the forecast pipeline for a single symbol and analyze the data structure.
    """
    logger.info(f"Running forecast diagnosis for {symbol} with {days} days forecast")
    
    # Check for API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment")
        return None
    
    # Create client and forecaster
    client = AlphaVantageClient(api_key=api_key)
    forecaster = StockForecast(client)
    ensemble_forecaster = ProphetEnsemble(client)
    
    results = {}
    
    # Test single model forecasting
    try:
        logger.info("Testing single model forecasting")
        df, forecast, model = forecaster.forecast(symbol, days=days)
        summary = forecaster.get_forecast_summary(forecast)
        
        # Save intermediate results
        results['single_model'] = {
            'summary': summary,
            'forecast_head': forecast.head().to_dict(),
            'forecast_tail': forecast.tail().to_dict(),
            'forecast_shape': forecast.shape,
            'df_shape': df.shape
        }
        
        # Save complete forecast for detailed analysis
        dump_json(summary, f"diagnosis/{symbol}_single_summary.json")
    except Exception as e:
        logger.error(f"Single model forecasting failed: {e}")
        results['single_model'] = {'error': str(e)}
    
    # Test ensemble forecasting
    try:
        logger.info("Testing ensemble forecasting")
        ensemble_result = ensemble_forecaster.forecast_ensemble(symbol, days=days)
        ensemble_summary = ensemble_forecaster.get_ensemble_forecast_summary(ensemble_result)
        
        # Save intermediate results
        results['ensemble'] = {
            'summary': ensemble_summary,
            'forecast_head': ensemble_result['forecast'].head().to_dict(),
            'forecast_tail': ensemble_result['forecast'].tail().to_dict(),
            'forecast_shape': ensemble_result['forecast'].shape,
            'model_count': len(ensemble_result.get('models', []))
        }
        
        # Save complete ensemble summary for detailed analysis
        dump_json(ensemble_summary, f"diagnosis/{symbol}_ensemble_summary.json")
    except Exception as e:
        logger.error(f"Ensemble forecasting failed: {e}")
        results['ensemble'] = {'error': str(e)}
    
    return results

def diagnose_symphony_analyzer(symphony_file):
    """
    Test the SymphonyAnalyzer forecasting functionality.
    """
    logger.info(f"Running diagnosis for SymphonyAnalyzer with {symphony_file}")
    
    if not os.path.exists(symphony_file):
        logger.error(f"Symphony file not found: {symphony_file}")
        return None
    
    try:
        # Create analyzer
        analyzer = SymphonyAnalyzer(symphony_file)
        
        # Get symbols
        symbols = analyzer.get_symbols()
        logger.info(f"Found {len(symbols)} symbols in symphony: {', '.join(symbols)}")
        
        # Run forecast
        logger.info("Running forecast_symphony method")
        forecast_results = analyzer.forecast_symphony(days=30)
        
        # Save results
        dump_json(forecast_results, f"diagnosis/symphony_forecast_results.json")
        
        # Check for specific data
        has_7d_forecast = 'average_7d_forecast' in forecast_results
        has_30d_forecast = 'average_30d_forecast' in forecast_results
        has_sentiment = 'forecast_sentiment' in forecast_results
        
        # Check individual symbol forecasts
        symbol_forecasts = {}
        for symbol in symbols:
            if symbol in forecast_results:
                symbol_data = forecast_results[symbol]
                has_7d = '7_day' in symbol_data and 'percent_change' in symbol_data.get('7_day', {})
                has_30d = '30_day' in symbol_data and 'percent_change' in symbol_data.get('30_day', {})
                
                symbol_forecasts[symbol] = {
                    'has_7d': has_7d,
                    'has_30d': has_30d,
                    '7d_value': symbol_data.get('7_day', {}).get('percent_change') if has_7d else None,
                    '30d_value': symbol_data.get('30_day', {}).get('percent_change') if has_30d else None,
                    'has_error': 'error' in symbol_data
                }
        
        # Create summary
        summary = {
            'symbols_count': len(symbols),
            'forecast_results_keys': list(forecast_results.keys()),
            'has_7d_forecast': has_7d_forecast,
            'has_30d_forecast': has_30d_forecast,
            'has_sentiment': has_sentiment,
            '7d_forecast': forecast_results.get('average_7d_forecast'),
            '30d_forecast': forecast_results.get('average_30d_forecast'),
            'sentiment': forecast_results.get('forecast_sentiment'),
            'symbol_forecasts': symbol_forecasts
        }
        
        return summary
    except Exception as e:
        logger.error(f"SymphonyAnalyzer diagnosis failed: {e}")
        return {'error': str(e)}

def main():
    """
    Main function to run all diagnostic tests.
    """
    # Create diagnosis directory if it doesn't exist
    os.makedirs("diagnosis", exist_ok=True)
    
    # Test with a single symbol
    test_symbol = "SPY"
    logger.info(f"Running direct forecast testing with {test_symbol}")
    symbol_results = diagnose_forecast_structure(test_symbol)
    dump_json(symbol_results, "diagnosis/symbol_forecast_results.json")
    
    # Test with a sample symphony file
    sample_symphony = "sample_symphony.json"
    if os.path.exists(sample_symphony):
        logger.info(f"Running SymphonyAnalyzer testing with {sample_symphony}")
        analyzer_results = diagnose_symphony_analyzer(sample_symphony)
        dump_json(analyzer_results, "diagnosis/analyzer_results.json")
    else:
        # Try to find another symphony file
        symphony_files = [f for f in os.listdir('.') if f.endswith('.json') and 'symphony' in f.lower()]
        if symphony_files:
            logger.info(f"Found alternative symphony file: {symphony_files[0]}")
            analyzer_results = diagnose_symphony_analyzer(symphony_files[0])
            dump_json(analyzer_results, "diagnosis/analyzer_results.json")
        else:
            logger.error("No symphony files found for testing")
    
    logger.info("Diagnosis complete. Results saved to 'diagnosis' directory.")

if __name__ == "__main__":
    main()
