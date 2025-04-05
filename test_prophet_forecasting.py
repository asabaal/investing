#!/usr/bin/env python3
"""
Test script for Prophet forecasting functionality.

This script provides a way to test the Prophet forecasting modules
and generate example forecasts for stock symbols.
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the current directory to Python's path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import local modules
from prophet_forecasting import StockForecast, ProphetEnsemble
from alpha_vantage_api import AlphaVantageClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tests/logs/prophet_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_single_symbol_forecast():
    """Test forecasting for a single symbol"""
    logger.info("Testing single symbol forecasting")
    
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Please set ALPHA_VANTAGE_API_KEY environment variable")
        return
    
    # Initialize client and forecaster
    client = AlphaVantageClient(api_key=api_key)
    forecaster = StockForecast(client=client)
    
    # Forecast for SPY
    symbol = "SPY"
    periods = 30
    logger.info(f"Forecasting {symbol} for {periods} days")
    
    try:
        forecast = forecaster.forecast(symbol, periods=periods)
        
        # Display results
        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"Forecast columns: {forecast.columns}")
        logger.info(f"Forecast preview:\n{forecast.head()}")
        
        # Save visualization
        plt.figure(figsize=(12, 6))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='blue')
        plt.title(f"{symbol} Price Forecast for Next {periods} Days")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{symbol}_forecast.png")
        logger.info(f"Saved forecast visualization to {symbol}_forecast.png")
        
        return True
    except Exception as e:
        logger.error(f"Error during single symbol forecast: {str(e)}")
        return False

def test_ensemble_forecast():
    """Test ensemble forecasting for a symbol"""
    logger.info("Testing ensemble forecasting")
    
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Please set ALPHA_VANTAGE_API_KEY environment variable")
        return
    
    # Initialize client and forecaster
    client = AlphaVantageClient(api_key=api_key)
    ensemble = ProphetEnsemble(client=client)
    
    # Forecast for SPY
    symbol = "SPY"
    periods = 30
    num_models = 3
    logger.info(f"Ensemble forecasting {symbol} with {num_models} models for {periods} days")
    
    try:
        result = ensemble.forecast_ensemble(
            symbol, 
            days=periods,
            num_models=num_models
        )
        
        forecast = result['forecast']
        models = result['models']
        individual_forecasts = result['individual_forecasts']
        
        # Display results
        logger.info(f"Created {len(models)} models for the ensemble")
        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"Forecast columns: {forecast.columns}")
        logger.info(f"Forecast preview:\n{forecast.head()}")
        
        # Get summary
        summary = ensemble.get_ensemble_forecast_summary(result)
        logger.info(f"Forecast summary: {summary}")
        
        # Save visualization
        plt.figure(figsize=(12, 6))
        
        # Plot historical data points
        historical_dates = pd.to_datetime(result['df'].index)
        historical_prices = result['df']['adjusted_close']
        plt.scatter(historical_dates, historical_prices, s=10, c='black', alpha=0.5, label='Historical')
        
        # Plot individual model forecasts
        for i, individual_forecast in enumerate(individual_forecasts):
            plt.plot(individual_forecast['ds'], individual_forecast['yhat'], 
                    alpha=0.3, linewidth=1, color='green', 
                    label=f'Model {i+1}' if i == 0 else None)
        
        # Plot ensemble forecast
        plt.plot(forecast['ds'], forecast['yhat'], 'b-', linewidth=2, label='Ensemble Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        
        # Add labels and legend
        plt.title(f"{symbol} Ensemble Forecast ({num_models} models)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f"{symbol}_ensemble_forecast.png")
        logger.info(f"Saved ensemble forecast visualization to {symbol}_ensemble_forecast.png")
        
        return True
    except Exception as e:
        logger.error(f"Error during ensemble forecast: {str(e)}")
        return False

def test_multi_symbol_forecast():
    """Test forecasting for multiple symbols in a portfolio"""
    logger.info("Testing multi-symbol portfolio forecasting")
    
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Please set ALPHA_VANTAGE_API_KEY environment variable")
        return
    
    # Initialize client and forecaster
    client = AlphaVantageClient(api_key=api_key)
    forecaster = StockForecast(client=client)
    
    # Create a sample portfolio
    symbols = ["SPY", "QQQ", "IWM"]
    periods = 30
    logger.info(f"Forecasting portfolio of: {', '.join(symbols)}")
    
    results = {}
    
    try:
        # Forecast each symbol
        for symbol in symbols:
            try:
                logger.info(f"Forecasting {symbol}")
                forecast = forecaster.forecast(symbol, periods=periods)
                results[symbol] = forecast
                logger.info(f"{symbol} forecast shape: {forecast.shape}")
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {str(e)}")
        
        # Create combined visualization
        plt.figure(figsize=(14, 8))
        
        for symbol, forecast in results.items():
            plt.plot(forecast['ds'], forecast['yhat'], label=symbol)
        
        plt.title(f"Multi-Symbol Portfolio Forecast ({periods} days)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig("portfolio_forecast.png")
        logger.info("Saved portfolio forecast visualization to portfolio_forecast.png")
        
        # Create normalized visualization (all starting at 100)
        plt.figure(figsize=(14, 8))
        
        # Get current date to separate historical and forecast
        current_date = datetime.now()
        
        for symbol, forecast in results.items():
            # Find the value closest to today
            today_idx = (forecast['ds'] - current_date).abs().argmin()
            base_value = forecast['yhat'][today_idx]
            
            # Normalize to 100
            normalized = forecast['yhat'] / base_value * 100
            
            # Plot
            plt.plot(forecast['ds'], normalized, label=symbol)
        
        plt.title(f"Normalized Portfolio Forecast (Starting at 100)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True)
        
        # Add a vertical line at current date
        plt.axvline(x=current_date, color='black', linestyle='--', alpha=0.5)
        plt.text(current_date, plt.ylim()[0], 'Today', ha='center', va='bottom')
        
        # Save figure
        plt.savefig("portfolio_forecast_normalized.png")
        logger.info("Saved normalized portfolio forecast visualization")
        
        return True
    except Exception as e:
        logger.error(f"Error during multi-symbol forecast: {str(e)}")
        return False

def main():
    """Main function to run prophet forecasting tests"""
    logger.info("Starting Prophet forecasting tests")
    
    # Create logs directory if it doesn't exist
    os.makedirs("tests/logs", exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "single":
            test_single_symbol_forecast()
        elif test_type == "ensemble":
            test_ensemble_forecast()
        elif test_type == "multi":
            test_multi_symbol_forecast()
        else:
            logger.error(f"Unknown test type: {test_type}")
            print(f"Usage: {sys.argv[0]} [single|ensemble|multi|all]")
    else:
        # Run all tests
        logger.info("Running all tests")
        test_single_symbol_forecast()
        test_ensemble_forecast()
        test_multi_symbol_forecast()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()
