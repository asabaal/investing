"""
Prophet Forecasting Module

This module provides time series forecasting capabilities using Facebook's
Prophet library. It implements interfaces to forecast market data and
custom models suitable for financial time series.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from plotly import graph_objs as go
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from alpha_vantage_api import AlphaVantageClient

# Configure logging
logger = logging.getLogger(__name__)

class StockForecast:
    """
    Stock price forecasting using Prophet.
    
    This class provides methods to forecast stock prices using
    the Prophet time series forecasting library.
    """
    
    def __init__(
        self,
        client: AlphaVantageClient,
        cache_dir: str = './cache',
        use_cache: bool = True
    ):
        """
        Initialize StockForecast.
        
        Args:
            client: Alpha Vantage client for market data
            cache_dir: Directory to store cached models and forecasts
            use_cache: Whether to use cached data and models
        """
        self.client = client
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, forecast_days: int) -> str:
        """
        Get cache file path for a given symbol and forecast days.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            
        Returns:
            File path for cache
        """
        # Use today's date in the filename for daily refresh
        today = datetime.now().strftime('%Y_%m_%d')
        return os.path.join(self.cache_dir, f"{symbol}_{forecast_days}d_{today}.pickle")
    
    def _load_from_cache(self, symbol: str, forecast_days: int) -> Optional[Dict]:
        """
        Try to load forecasted data from cache.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            
        Returns:
            Cached forecast data or None if not found
        """
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(symbol, forecast_days)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {str(e)}")
                
        return None
    
    def _save_to_cache(self, symbol: str, forecast_days: int, data: Dict):
        """
        Save forecasted data to cache.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            data: Forecast data to cache
        """
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(symbol, forecast_days)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {str(e)}")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'adjusted_close') -> pd.DataFrame:
        """
        Prepare time series data for Prophet.
        
        Args:
            df: DataFrame with market data
            target_column: Column to forecast
            
        Returns:
            DataFrame formatted for Prophet
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check for NaN values in volume and clean them if present
        if 'volume' in df_copy.columns and df_copy['volume'].isna().any():
            logger.info(f"Cleaning NaN values in volume column")
            
            # Try forward/backward fill first
            df_copy['volume'] = df_copy['volume'].fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, use median
            if df_copy['volume'].isna().any():
                median_volume = df_copy['volume'].median()
                if np.isnan(median_volume):  # If median is also NaN, use a default value
                    median_volume = 1000000  # Arbitrary default value
                df_copy['volume'] = df_copy['volume'].fillna(median_volume)
                logger.info(f"Used median volume {median_volume} to fill remaining NaNs")
        
        # Prophet requires columns named 'ds' (date) and 'y' (target variable)
        prophet_df = pd.DataFrame({
            'ds': df_copy.index,
            'y': df_copy[target_column]
        })
        
        # Add day-of-week feature
        prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
        
        # Add month feature
        prophet_df['month'] = prophet_df['ds'].dt.month
        
        # Add volume as a regressor if available
        if 'volume' in df_copy.columns:
            # Ensure volume is clean and positive before log transform
            volume = df_copy['volume'].copy()
            volume = volume.clip(lower=1)  # Ensure all values are at least 1 for log transform
            
            # Normalize volume to reduce impact of extreme values
            prophet_df['volume'] = np.log1p(volume)
            
            # Double-check for NaN values
            if prophet_df['volume'].isna().any():
                logger.warning(f"Still found NaN values in volume after cleaning, replacing with mean")
                prophet_df['volume'] = prophet_df['volume'].fillna(prophet_df['volume'].mean())
        
        return prophet_df
    
    def fit_prophet_model(
        self,
        df: pd.DataFrame,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        include_volume: bool = True,
        yearly_seasonality: Union[bool, int] = True,
        weekly_seasonality: Union[bool, int] = True,
        daily_seasonality: Union[bool, int] = False
    ) -> Prophet:
        """
        Fit a Prophet model to time series data.
        
        Args:
            df: DataFrame prepared for Prophet (with 'ds' and 'y' columns)
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Flexibility of the seasonality
            include_volume: Whether to include volume as a regressor
            yearly_seasonality: Yearly seasonality setting
            weekly_seasonality: Weekly seasonality setting
            daily_seasonality: Daily seasonality setting
            
        Returns:
            Fitted Prophet model
        """
        # Create model with seasonality settings
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add volume regressor if available and requested
        if include_volume and 'volume' in df.columns:
            # Double-check for any NaN values
            if df['volume'].isna().any():
                logger.warning("Found NaN in volume column, replacing with mean")
                df['volume'] = df['volume'].fillna(df['volume'].mean())
            
            model.add_regressor('volume')
        
        # Add day of week and month as additional regressors
        model.add_regressor('day_of_week')
        model.add_regressor('month')
        
        # Fit the model
        model.fit(df)
        
        return model
    
    def forecast(
        self,
        symbol: str,
        days: int = 30,
        target_column: str = 'adjusted_close',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        include_volume: bool = True,
        include_prediction_intervals: bool = True,
        prediction_interval_width: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Prophet]:
        """
        Forecast stock prices for a specified number of days.
        
        Args:
            symbol: Stock symbol
            days: Number of days to forecast
            target_column: Column to forecast
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Flexibility of the seasonality
            include_volume: Whether to include volume as a regressor
            include_prediction_intervals: Whether to include prediction intervals
            prediction_interval_width: Width of prediction intervals (0 to 1)
            
        Returns:
            Tuple of (historical data, forecast data, fitted model)
        """
        # Check cache first
        cached_data = self._load_from_cache(symbol, days)
        if cached_data:
            logger.info(f"Using cached forecast for {symbol}")
            return cached_data['df'], cached_data['forecast'], cached_data['model']
        
        # Get daily data
        df = self.client.get_daily(symbol, outputsize='full')
        
        # Keep last 5 years max to avoid very old data influencing the model
        cutoff_date = datetime.now() - timedelta(days=365 * 5)
        df = df[df.index >= cutoff_date]
        
        # Prepare data for Prophet
        prophet_df = self.prepare_data(df, target_column)
        
        # Fit Prophet model
        model = self.fit_prophet_model(
            prophet_df,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            include_volume=include_volume
        )
        
        # Prepare future dataframe for prediction
        future = model.make_future_dataframe(periods=days)
        
        # Add day of week and month features to future dataframe
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['month'] = future['ds'].dt.month
        
        # Add volume to future dataframe if included
        if include_volume and 'volume' in prophet_df.columns:
            # For future dates, use the average of recent volume
            recent_volume = prophet_df['volume'].iloc[-30:].mean()
            
            # Set known volumes for historical dates
            future.loc[future['ds'].isin(prophet_df['ds']), 'volume'] = prophet_df.set_index('ds')['volume']
            
            # Set average volume for future dates
            future.loc[~future['ds'].isin(prophet_df['ds']), 'volume'] = recent_volume
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Set prediction intervals
        if include_prediction_intervals:
            model.uncertainty_samples = 1000
            forecast = model.predict(future, interval_width=prediction_interval_width)
        
        # Cache the results
        cached_data = {
            'df': df,
            'forecast': forecast,
            'model': model
        }
        self._save_to_cache(symbol, days, cached_data)
        
        return df, forecast, model
    
    def evaluate_model(
        self,
        model: Prophet,
        df: pd.DataFrame,
        initial_period: str = '30 days',
        period: str = '7 days',
        horizon: str = '30 days'
    ) -> pd.DataFrame:
        """
        Evaluate a Prophet model using cross-validation.
        
        Args:
            model: Fitted Prophet model
            df: Data used to fit the model
            initial_period: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            DataFrame with performance metrics
        """
        # Perform cross-validation
        cv_results = cross_validation(
            model,
            initial=initial_period,
            period=period,
            horizon=horizon
        )
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        
        return metrics
    
    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame,
        model: Prophet,
        target_column: str = 'adjusted_close',
        symbol: str = '',
        title: Optional[str] = None,
        show_changepoints: bool = True,
        plot_components: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot Prophet forecast results.
        
        Args:
            df: Historical market data
            forecast: Prophet forecast DataFrame
            model: Fitted Prophet model
            target_column: Target column being forecasted
            symbol: Stock symbol for the title
            title: Custom title (or None to use default)
            show_changepoints: Whether to show changepoints
            plot_components: Whether to plot model components
            figsize: Figure size as (width, height)
        """
        # Plot the forecast
        fig = plt.figure(figsize=figsize)
        
        if title is None:
            title = f"{symbol} - {target_column} Forecast"
        
        ax = fig.add_subplot(111)
        
        # Plot historical data
        ax.plot(df.index, df[target_column], 'k.', label='Historical')
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
        
        # Plot prediction intervals
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            ax.fill_between(
                forecast['ds'], 
                forecast['yhat_lower'], 
                forecast['yhat_upper'], 
                color='blue', 
                alpha=0.2,
                label='Prediction Interval'
            )
        
        # Show changepoints
        if show_changepoints:
            changepoints = model.changepoints
            for changepoint in changepoints:
                ax.axvline(changepoint, color='r', linestyle='--', alpha=0.4)
            
            # Add a line for the legend
            ax.plot([], [], 'r--', alpha=0.4, label='Changepoints')
        
        # Add labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel(target_column)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot model components if requested
        if plot_components:
            model.plot_components(forecast)
            plt.show()
    
    def plot_forecast_interactive(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame,
        model: Prophet,
        target_column: str = 'adjusted_close',
        symbol: str = '',
        title: Optional[str] = None
    ):
        """
        Create an interactive Plotly plot of the forecast.
        
        Args:
            df: Historical market data
            forecast: Prophet forecast DataFrame
            model: Fitted Prophet model
            target_column: Target column being forecasted
            symbol: Stock symbol for the title
            title: Custom title (or None to use default)
        """
        if title is None:
            title = f"{symbol} - {target_column} Forecast"
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[target_column],
            mode='markers',
            name='Historical',
            marker=dict(color='black', size=4)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue')
        ))
        
        # Add prediction intervals
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                showlegend=True
            ))
        
        # Add changepoints
        for changepoint in model.changepoints:
            fig.add_shape(
                type="line",
                x0=changepoint,
                y0=0,
                x1=changepoint,
                y1=1,
                yref="paper",
                line=dict(color="red", width=1, dash="dash"),
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=target_column,
            legend_title="Data",
            hovermode="x unified"
        )
        
        # Show plot
        fig.show()
    
    def get_forecast_summary(
        self,
        forecast: pd.DataFrame,
        days: List[int] = [1, 7, 30, 90]
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a summary of the forecast for specific horizons.
        
        Args:
            forecast: Prophet forecast DataFrame
            days: List of forecast horizons to summarize
            
        Returns:
            Dictionary with forecast summaries
        """
        summary = {}
        
        # Get the last historical date (up to today)
        today = pd.Timestamp(datetime.now().date())
        last_historical_date = forecast[forecast['ds'] <= today]['ds'].max()
        
        # If no historical dates found (rare case), use the earliest date
        if pd.isnull(last_historical_date):
            last_historical_date = forecast['ds'].min()
            logger.warning("No historical dates found in forecast, using earliest date")
        
        for day in days:
            # Find the closest forecast date to the requested horizon
            target_date = last_historical_date + pd.Timedelta(days=day)
            forecast_index = (forecast['ds'] - target_date).abs().idxmin()
            forecast_row = forecast.iloc[forecast_index]
            
            # Calculate forecast percent change
            last_historical_index = forecast[forecast['ds'] == last_historical_date].index[0]
            last_historical_value = forecast.iloc[last_historical_index]['yhat']
            
            forecast_value = forecast_row['yhat']
            percent_change = (forecast_value - last_historical_value) / last_historical_value * 100
            
            # Get prediction intervals if available
            lower_bound = forecast_row.get('yhat_lower', None)
            upper_bound = forecast_row.get('yhat_upper', None)
            
            # Calculate uncertainty range if intervals are available
            if lower_bound is not None and upper_bound is not None:
                uncertainty_range = (upper_bound - lower_bound) / forecast_value * 100
            else:
                uncertainty_range = None
            
            # Add to summary
            summary[f'{day}_day'] = {
                'date': forecast_row['ds'].strftime('%Y-%m-%d'),
                'forecast': float(forecast_value),  # Convert to native Python float
                'percent_change': float(percent_change),  # Convert to native Python float
                'lower_bound': float(lower_bound) if lower_bound is not None else None,
                'upper_bound': float(upper_bound) if upper_bound is not None else None,
                'uncertainty_range': float(uncertainty_range) if uncertainty_range is not None else None
            }
        
        # Add a confidence score based on uncertainty ranges (higher score = more confident)
        if '30_day' in summary and summary['30_day'].get('uncertainty_range') is not None:
            uncertainty = summary['30_day']['uncertainty_range']
            # Confidence score: 1.0 (low uncertainty) to 0.0 (high uncertainty)
            confidence = max(0.0, min(1.0, 1.0 - uncertainty / 100.0))
            summary['confidence'] = confidence
        
        return summary
    
    def bulk_forecast(
        self,
        symbols: List[str],
        days: int = 30,
        target_column: str = 'adjusted_close'
    ) -> Dict[str, Dict]:
        """
        Generate forecasts for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to forecast
            target_column: Column to forecast
            
        Returns:
            Dictionary mapping symbols to forecast summaries
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Forecasting {symbol}")
                
                df, forecast, model = self.forecast(symbol, days, target_column)
                summary = self.get_forecast_summary(forecast)
                
                results[symbol] = {
                    'forecast': forecast,
                    'summary': summary,
                    'model': model,
                    'df': df
                }
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        return results

class ProphetEnsemble:
    """
    Ensemble of Prophet models for improved forecasting.
    
    This class creates an ensemble of Prophet models with different
    hyperparameters to generate more robust forecasts.
    """
    
    def __init__(
        self,
        client: AlphaVantageClient,
        cache_dir: str = './cache',
        use_cache: bool = True
    ):
        """
        Initialize ProphetEnsemble.
        
        Args:
            client: Alpha Vantage client for market data
            cache_dir: Directory to store cached models and forecasts
            use_cache: Whether to use cached data and models
        """
        self.client = client
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.forecaster = StockForecast(client, cache_dir, use_cache)
        
    def _get_cache_path(self, symbol: str, forecast_days: int) -> str:
        """
        Get cache file path for a given symbol and forecast days.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            
        Returns:
            File path for cache
        """
        # Use today's date in the filename for daily refresh
        today = datetime.now().strftime('%Y_%m_%d')
        return os.path.join(self.cache_dir, f"{symbol}_ensemble_{forecast_days}d_{today}.pickle")
    
    def _load_from_cache(self, symbol: str, forecast_days: int) -> Optional[Dict]:
        """
        Try to load ensemble forecast data from cache.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            
        Returns:
            Cached forecast data or None if not found
        """
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(symbol, forecast_days)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load ensemble cache for {symbol}: {str(e)}")
                
        return None
    
    def _save_to_cache(self, symbol: str, forecast_days: int, data: Dict):
        """
        Save ensemble forecast data to cache.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast
            data: Forecast data to cache
        """
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(symbol, forecast_days)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save ensemble cache for {symbol}: {str(e)}")
    
    def forecast_ensemble(
        self,
        symbol: str,
        days: int = 30,
        target_column: str = 'adjusted_close',
        num_models: int = 5,
        changepoint_prior_scales: Optional[List[float]] = None,
        seasonality_prior_scales: Optional[List[float]] = None
    ) -> Dict:
        """
        Generate an ensemble forecast using multiple Prophet models.
        
        Args:
            symbol: Stock symbol
            days: Number of days to forecast
            target_column: Column to forecast
            num_models: Number of models in the ensemble
            changepoint_prior_scales: List of changepoint prior scales to use
            seasonality_prior_scales: List of seasonality prior scales to use
            
        Returns:
            Dictionary with ensemble forecast results
        """
        # Check cache first
        cached_data = self._load_from_cache(symbol, days)
        if cached_data:
            logger.info(f"Using cached ensemble forecast for {symbol}")
            return cached_data
        
        # If not provided, use default hyperparameter ranges
        if changepoint_prior_scales is None:
            changepoint_prior_scales = [0.01, 0.05, 0.1, 0.15, 0.2]
            
        if seasonality_prior_scales is None:
            seasonality_prior_scales = [0.1, 1.0, 5.0, 10.0, 15.0]
        
        # Ensure we have enough hyperparameter combinations
        if len(changepoint_prior_scales) < num_models or len(seasonality_prior_scales) < num_models:
            logger.warning("Not enough hyperparameter combinations for requested ensemble size. Using available combinations.")
            num_models = min(len(changepoint_prior_scales), len(seasonality_prior_scales))
        
        # Get daily data
        df = self.client.get_daily(symbol, outputsize='full')
        
        # Keep last 5 years max to avoid very old data influencing the model
        cutoff_date = datetime.now() - timedelta(days=365 * 5)
        df = df[df.index >= cutoff_date]
        
        # Prepare data for Prophet
        prophet_df = self.forecaster.prepare_data(df, target_column)
        
        # Create and fit ensemble models
        models = []
        forecasts = []
        
        for i in range(num_models):
            try:
                changepoint_prior_scale = changepoint_prior_scales[i % len(changepoint_prior_scales)]
                seasonality_prior_scale = seasonality_prior_scales[i % len(seasonality_prior_scales)]
                
                # Fit Prophet model with different hyperparameters
                model = self.forecaster.fit_prophet_model(
                    prophet_df,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale
                )
                
                # Prepare future dataframe for prediction
                future = model.make_future_dataframe(periods=days)
                
                # Add regressor columns
                future['day_of_week'] = future['ds'].dt.dayofweek
                future['month'] = future['ds'].dt.month
                
                # Add volume to future dataframe if included
                if 'volume' in prophet_df.columns:
                    # For future dates, use the average of recent volume
                    recent_volume = prophet_df['volume'].iloc[-30:].mean()
                    
                    # Set known volumes for historical dates
                    future.loc[future['ds'].isin(prophet_df['ds']), 'volume'] = prophet_df.set_index('ds')['volume']
                    
                    # Set average volume for future dates
                    future.loc[~future['ds'].isin(prophet_df['ds']), 'volume'] = recent_volume
                
                # Generate forecast
                forecast = model.predict(future)
                
                models.append(model)
                forecasts.append(forecast)
            except Exception as e:
                logger.warning(f"Model {i} failed: {str(e)}")
                continue
        
        # If no models were successfully fit, raise exception
        if not models or not forecasts:
            raise ValueError(f"Failed to fit any models for {symbol}")
        
        # Combine forecasts
        ensemble_forecast = pd.DataFrame({'ds': forecasts[0]['ds']})
        
        # Calculate mean and standard deviation across models
        ensemble_forecast['yhat'] = np.mean([forecast['yhat'] for forecast in forecasts], axis=0)
        ensemble_forecast['yhat_std'] = np.std([forecast['yhat'] for forecast in forecasts], axis=0)
        
        # Calculate ensemble prediction intervals
        z_score = 1.96  # 95% confidence interval
        ensemble_forecast['yhat_lower'] = ensemble_forecast['yhat'] - z_score * ensemble_forecast['yhat_std']
        ensemble_forecast['yhat_upper'] = ensemble_forecast['yhat'] + z_score * ensemble_forecast['yhat_std']
        
        # Prepare result
        result = {
            'df': df,
            'forecast': ensemble_forecast,
            'models': models,
            'individual_forecasts': forecasts,
            'hyperparameters': {
                'changepoint_prior_scales': changepoint_prior_scales[:len(models)],
                'seasonality_prior_scales': seasonality_prior_scales[:len(models)]
            }
        }
        
        # Cache the result
        self._save_to_cache(symbol, days, result)
        
        return result
    
    def plot_ensemble_forecast(
        self,
        result: Dict,
        target_column: str = 'adjusted_close',
        symbol: str = '',
        title: Optional[str] = None,
        show_individual_forecasts: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot the ensemble forecast with historical data and prediction intervals.
        
        Args:
            result: Result from forecast_ensemble method
            target_column: Target column being forecasted
            symbol: Stock symbol for the title
            title: Custom title (or None to use default)
            show_individual_forecasts: Whether to show individual model forecasts
            figsize: Figure size as (width, height)
        """
        # Extract data from result
        df = result['df']
        ensemble_forecast = result['forecast']
        individual_forecasts = result.get('individual_forecasts', [])
        
        # Set up the plot
        if title is None:
            title = f"{symbol} - {target_column} Ensemble Forecast"
        
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(df.index, df[target_column], 'k.', label='Historical')
        
        # Plot individual forecasts if requested
        if show_individual_forecasts and individual_forecasts:
            for i, forecast in enumerate(individual_forecasts):
                plt.plot(forecast['ds'], forecast['yhat'], 'g-', alpha=0.2, linewidth=1)
        
        # Plot ensemble forecast
        plt.plot(ensemble_forecast['ds'], ensemble_forecast['yhat'], 'b-', linewidth=2, label='Ensemble Forecast')
        
        # Plot prediction intervals
        plt.fill_between(
            ensemble_forecast['ds'],
            ensemble_forecast['yhat_lower'],
            ensemble_forecast['yhat_upper'],
            color='blue',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.title(title)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_ensemble_forecast_summary(
        self,
        result: Dict,
        days: List[int] = [1, 7, 30, 90]
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a summary of the ensemble forecast for specific horizons.
        
        Args:
            result: Result from forecast_ensemble method
            days: List of forecast horizons to summarize
            
        Returns:
            Dictionary with forecast summaries
        """
        forecast = result['forecast']
        summary = {}
        
        # Get the last historical date (up to today)
        today = pd.Timestamp(datetime.now().date())
        last_historical_date = forecast[forecast['ds'] <= today]['ds'].max()
        
        # If no historical dates found (rare case), use the earliest date
        if pd.isnull(last_historical_date):
            last_historical_date = forecast['ds'].min()
            logger.warning("No historical dates found in forecast, using earliest date")
        
        for day in days:
            # Find the closest forecast date to the requested horizon
            target_date = last_historical_date + pd.Timedelta(days=day)
            forecast_index = (forecast['ds'] - target_date).abs().idxmin()
            forecast_row = forecast.iloc[forecast_index]
            
            # Calculate forecast percent change
            last_historical_index = forecast[forecast['ds'] == last_historical_date].index[0]
            last_historical_value = forecast.iloc[last_historical_index]['yhat']
            
            forecast_value = forecast_row['yhat']
            percent_change = (forecast_value - last_historical_value) / last_historical_value * 100
            
            # Get prediction intervals
            lower_bound = forecast_row['yhat_lower']
            upper_bound = forecast_row['yhat_upper']
            
            # Calculate uncertainty range
            uncertainty_range = (upper_bound - lower_bound) / forecast_value * 100
            
            # Calculate agreement between models
            std_dev = forecast_row['yhat_std']
            coefficient_of_variation = (std_dev / forecast_value) * 100
            
            # Add to summary
            summary[f'{day}_day'] = {
                'date': forecast_row['ds'].strftime('%Y-%m-%d'),
                'forecast': float(forecast_value),  # Convert to native Python float
                'percent_change': float(percent_change),  # Convert to native Python float
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'uncertainty_range': float(uncertainty_range),
                'model_agreement': float(100 - coefficient_of_variation)  # Higher is better
            }
        
        # Add a confidence score based on model agreement and uncertainty
        if '30_day' in summary:
            model_agreement = summary['30_day']['model_agreement']
            # Normalize to a 0.0-1.0 scale (higher is better)
            confidence = max(0.0, min(1.0, model_agreement / 100.0))
            summary['confidence'] = confidence
        
        return summary

class FinancialMarketConditions:
    """
    Analyzes overall market conditions using various indicators.
    
    This class provides methods to assess market conditions by analyzing
    multiple indicators such as market indices, volatility, and breadth.
    """
    
    def __init__(
        self,
        client: AlphaVantageClient,
        cache_dir: str = './cache',
        market_indices: List[str] = ['SPY', 'QQQ', 'IWM'],
        volatility_indices: List[str] = ['VIX', 'VVIX'],
        breadth_etfs: List[str] = ['SPY', 'RSP']
    ):
        """
        Initialize FinancialMarketConditions.
        
        Args:
            client: Alpha Vantage client for market data
            cache_dir: Directory to store cached data
            market_indices: List of market index symbols to track
            volatility_indices: List of volatility index symbols to track
            breadth_etfs: List of ETFs to use for market breadth analysis
        """
        self.client = client
        self.cache_dir = cache_dir
        self.market_indices = market_indices
        self.volatility_indices = volatility_indices
        self.breadth_etfs = breadth_etfs
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize Prophet forecasters
        self.forecaster = StockForecast(client, cache_dir)
        self.ensemble_forecaster = ProphetEnsemble(client, cache_dir)
    
    def get_market_condition_summary(self) -> Dict:
        """
        Generate a comprehensive summary of current market conditions.
        
        Returns:
            Dictionary with market condition summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'market_indices': self._analyze_market_indices(),
            'volatility': self._analyze_volatility(),
            'breadth': self._analyze_market_breadth(),
            'trend': self._analyze_market_trend(),
            'overall_assessment': {}
        }
        
        # Generate overall market assessment
        overall = summary['overall_assessment']
        
        # Market direction
        trend_signals = [trend['signal'] for trend in summary['trend'].values()]
        if trend_signals.count('bullish') > trend_signals.count('bearish'):
            overall['direction'] = 'bullish'
        elif trend_signals.count('bearish') > trend_signals.count('bullish'):
            overall['direction'] = 'bearish'
        else:
            overall['direction'] = 'neutral'
        
        # Market volatility
        vix_level = summary['volatility'].get('VIX', {}).get('current_level')
        if vix_level:
            if vix_level < 15:
                overall['volatility'] = 'low'
            elif vix_level > 25:
                overall['volatility'] = 'high'
            else:
                overall['volatility'] = 'moderate'
        else:
            overall['volatility'] = 'unknown'
        
        # Market breadth
        breadth_assessment = summary['breadth'].get('assessment')
        if breadth_assessment:
            overall['breadth'] = breadth_assessment
        
        return summary
    
    def _analyze_market_indices(self) -> Dict:
        """
        Analyze the current state and trends of major market indices.
        
        Returns:
            Dictionary with index analysis results
        """
        index_analysis = {}
        
        for symbol in self.market_indices:
            try:
                # Get daily data
                df = self.client.get_daily(symbol)
                
                # Calculate recent performance
                current_price = df['adjusted_close'].iloc[-1]
                price_1d_ago = df['adjusted_close'].iloc[-2]
                price_5d_ago = df['adjusted_close'].iloc[-6] if len(df) > 6 else df['adjusted_close'].iloc[0]
                price_20d_ago = df['adjusted_close'].iloc[-21] if len(df) > 21 else df['adjusted_close'].iloc[0]
                price_50d_ago = df['adjusted_close'].iloc[-51] if len(df) > 51 else df['adjusted_close'].iloc[0]
                price_200d_ago = df['adjusted_close'].iloc[-201] if len(df) > 201 else df['adjusted_close'].iloc[0]
                
                # Calculate moving averages
                ma_20 = df['adjusted_close'].rolling(window=20).mean().iloc[-1]
                ma_50 = df['adjusted_close'].rolling(window=50).mean().iloc[-1]
                ma_200 = df['adjusted_close'].rolling(window=200).mean().iloc[-1]
                
                # Calculate RSI
                delta = df['adjusted_close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Determine positioning relative to moving averages
                ma_signals = []
                
                if current_price > ma_20:
                    ma_signals.append('above 20-day MA')
                else:
                    ma_signals.append('below 20-day MA')
                    
                if current_price > ma_50:
                    ma_signals.append('above 50-day MA')
                else:
                    ma_signals.append('below 50-day MA')
                    
                if current_price > ma_200:
                    ma_signals.append('above 200-day MA')
                else:
                    ma_signals.append('below 200-day MA')
                
                # Determine golden/death cross status
                if ma_50 > ma_200 and ma_50 > ma_200 * 1.01:  # 1% buffer for confirmation
                    cross_status = 'golden cross'
                elif ma_50 < ma_200 and ma_50 < ma_200 * 0.99:  # 1% buffer for confirmation
                    cross_status = 'death cross'
                else:
                    cross_status = 'no cross'
                
                # Store results
                index_analysis[symbol] = {
                    'current_price': current_price,
                    'daily_change_pct': (current_price - price_1d_ago) / price_1d_ago * 100,
                    'weekly_change_pct': (current_price - price_5d_ago) / price_5d_ago * 100,
                    'monthly_change_pct': (current_price - price_20d_ago) / price_20d_ago * 100,
                    'rsi': current_rsi,
                    'ma_status': ma_signals,
                    'cross_status': cross_status,
                    'price_vs_ma_20_pct': (current_price / ma_20 - 1) * 100,
                    'price_vs_ma_50_pct': (current_price / ma_50 - 1) * 100,
                    'price_vs_ma_200_pct': (current_price / ma_200 - 1) * 100,
                    'ma_50_vs_ma_200_pct': (ma_50 / ma_200 - 1) * 100
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                index_analysis[symbol] = {'error': str(e)}
        
        return index_analysis
    
    def _analyze_volatility(self) -> Dict:
        """
        Analyze volatility indices and implied market volatility.
        
        Returns:
            Dictionary with volatility analysis results
        """
        volatility_analysis = {}
        
        for symbol in self.volatility_indices:
            try:
                # Get daily data
                df = self.client.get_daily(symbol)
                
                # Calculate recent levels and changes
                current_level = df['adjusted_close'].iloc[-1]
                level_1d_ago = df['adjusted_close'].iloc[-2]
                level_5d_ago = df['adjusted_close'].iloc[-6] if len(df) > 6 else df['adjusted_close'].iloc[0]
                level_20d_ago = df['adjusted_close'].iloc[-21] if len(df) > 21 else df['adjusted_close'].iloc[0]
                
                # Calculate historical percentiles
                percentile_rank = percentileofscore(df['adjusted_close'], current_level)
                
                # Determine volatility regime
                if percentile_rank < 25:
                    regime = 'low_volatility'
                elif percentile_rank > 75:
                    regime = 'high_volatility'
                else:
                    regime = 'normal_volatility'
                
                # Store results
                volatility_analysis[symbol] = {
                    'current_level': current_level,
                    'daily_change_pct': (current_level - level_1d_ago) / level_1d_ago * 100,
                    'weekly_change_pct': (current_level - level_5d_ago) / level_5d_ago * 100,
                    'monthly_change_pct': (current_level - level_20d_ago) / level_20d_ago * 100,
                    'percentile_rank': percentile_rank,
                    'volatility_regime': regime
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                volatility_analysis[symbol] = {'error': str(e)}
        
        return volatility_analysis
    
    def _analyze_market_breadth(self) -> Dict:
        """
        Analyze market breadth using various indicators.
        
        Returns:
            Dictionary with market breadth analysis results
        """
        # SPY (market-cap weighted) vs RSP (equal weighted) can show breadth
        try:
            if len(self.breadth_etfs) < 2:
                return {'error': 'Not enough ETFs for breadth analysis'}
            
            spy_data = self.client.get_daily(self.breadth_etfs[0])
            rsp_data = self.client.get_daily(self.breadth_etfs[1])
            
            # Normalize to the same starting point
            start_date = max(spy_data.index[0], rsp_data.index[0])
            spy_data = spy_data[spy_data.index >= start_date]
            rsp_data = rsp_data[rsp_data.index >= start_date]
            
            spy_normalized = spy_data['adjusted_close'] / spy_data['adjusted_close'].iloc[0]
            rsp_normalized = rsp_data['adjusted_close'] / rsp_data['adjusted_close'].iloc[0]
            
            # Calculate relative performance
            ratio = rsp_normalized / spy_normalized
            
            # Analyze recent trend in the ratio
            current_ratio = ratio.iloc[-1]
            ratio_1m_ago = ratio.iloc[-21] if len(ratio) > 21 else ratio.iloc[0]
            ratio_3m_ago = ratio.iloc[-63] if len(ratio) > 63 else ratio.iloc[0]
            
            monthly_change = (current_ratio - ratio_1m_ago) / ratio_1m_ago * 100
            quarterly_change = (current_ratio - ratio_3m_ago) / ratio_3m_ago * 100
            
            # Interpret breadth trend
            if monthly_change > 0 and quarterly_change > 0:
                breadth_trend = 'improving'
                assessment = 'healthy'
            elif monthly_change < 0 and quarterly_change < 0:
                breadth_trend = 'deteriorating'
                assessment = 'narrow'
            elif monthly_change > 0 and quarterly_change < 0:
                breadth_trend = 'recovering'
                assessment = 'improving'
            else:
                breadth_trend = 'weakening'
                assessment = 'weakening'
            
            return {
                'equal_weight_vs_cap_weight': {
                    'current_ratio': current_ratio,
                    'monthly_change_pct': monthly_change,
                    'quarterly_change_pct': quarterly_change
                },
                'breadth_trend': breadth_trend,
                'assessment': assessment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market breadth: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_market_trend(self) -> Dict:
        """
        Analyze market trends over multiple timeframes.
        
        Returns:
            Dictionary with market trend analysis results
        """
        trend_analysis = {}
        
        # Analyze each major index
        for symbol in self.market_indices:
            try:
                # Get daily data
                df = self.client.get_daily(symbol)
                
                # Calculate moving averages
                df['ma_20'] = df['adjusted_close'].rolling(window=20).mean()
                df['ma_50'] = df['adjusted_close'].rolling(window=50).mean()
                df['ma_200'] = df['adjusted_close'].rolling(window=200).mean()
                
                # Current price and MAs
                current_price = df['adjusted_close'].iloc[-1]
                ma_20 = df['ma_20'].iloc[-1]
                ma_50 = df['ma_50'].iloc[-1]
                ma_200 = df['ma_200'].iloc[-1]
                
                # MA slopes (rate of change)
                ma_20_slope = (df['ma_20'].iloc[-1] - df['ma_20'].iloc[-5]) / df['ma_20'].iloc[-5] * 100
                ma_50_slope = (df['ma_50'].iloc[-1] - df['ma_50'].iloc[-5]) / df['ma_50'].iloc[-5] * 100
                ma_200_slope = (df['ma_200'].iloc[-1] - df['ma_200'].iloc[-5]) / df['ma_200'].iloc[-5] * 100
                
                # Determine overall trend signal
                trend_points = 0
                
                # Price relative to MAs
                if current_price > ma_20:
                    trend_points += 1
                if current_price > ma_50:
                    trend_points += 1
                if current_price > ma_200:
                    trend_points += 2  # Give more weight to 200-day MA
                
                # MA slopes
                if ma_20_slope > 0:
                    trend_points += 1
                if ma_50_slope > 0:
                    trend_points += 1
                if ma_200_slope > 0:
                    trend_points += 2  # Give more weight to 200-day MA
                
                # MA order (golden/death cross)
                if ma_20 > ma_50 and ma_50 > ma_200:
                    trend_points += 3  # Strong bullish alignment
                elif ma_20 < ma_50 and ma_50 < ma_200:
                    trend_points -= 3  # Strong bearish alignment
                
                # Determine signal
                if trend_points >= 6:
                    signal = 'bullish'
                elif trend_points <= -2:
                    signal = 'bearish'
                else:
                    signal = 'neutral'
                
                # Store results
                trend_analysis[symbol] = {
                    'price_vs_ma_20': 'above' if current_price > ma_20 else 'below',
                    'price_vs_ma_50': 'above' if current_price > ma_50 else 'below',
                    'price_vs_ma_200': 'above' if current_price > ma_200 else 'below',
                    'ma_20_slope': ma_20_slope,
                    'ma_50_slope': ma_50_slope,
                    'ma_200_slope': ma_200_slope,
                    'ma_alignment': 'bullish' if ma_20 > ma_50 and ma_50 > ma_200 else 
                                    'bearish' if ma_20 < ma_50 and ma_50 < ma_200 else 
                                    'mixed',
                    'trend_points': trend_points,
                    'signal': signal
                }
                
            except Exception as e:
                logger.error(f"Error analyzing trend for {symbol}: {str(e)}")
                trend_analysis[symbol] = {'error': str(e)}
        
        return trend_analysis
    
    def forecast_market_conditions(self, days: int = 30) -> Dict:
        """
        Forecast future market conditions.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Dictionary with market condition forecasts
        """
        forecasts = {}
        
        # Forecast market indices
        for symbol in self.market_indices:
            try:
                # Use ensemble forecasting for better accuracy
                result = self.ensemble_forecaster.forecast_ensemble(symbol, days)
                summary = self.ensemble_forecaster.get_ensemble_forecast_summary(result)
                
                forecasts[symbol] = summary
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {str(e)}")
                forecasts[symbol] = {'error': str(e)}
        
        # Forecast volatility
        for symbol in self.volatility_indices:
            try:
                result = self.ensemble_forecaster.forecast_ensemble(symbol, days)
                summary = self.ensemble_forecaster.get_ensemble_forecast_summary(result)
                
                forecasts[symbol] = summary
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {str(e)}")
                forecasts[symbol] = {'error': str(e)}
        
        # Generate overall forecast
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        for symbol in self.market_indices:
            if symbol in forecasts and '30_day' in forecasts[symbol]:
                forecast_data = forecasts[symbol]['30_day']
                percent_change = forecast_data.get('percent_change', 0)
                
                if percent_change > 1:
                    bullish_signals += 1
                elif percent_change < -1:
                    bearish_signals += 1
                else:
                    neutral_signals += 1
        
        if bullish_signals > bearish_signals and bullish_signals > neutral_signals:
            forecast_direction = 'bullish'
        elif bearish_signals > bullish_signals and bearish_signals > neutral_signals:
            forecast_direction = 'bearish'
        else:
            forecast_direction = 'neutral'
        
        # Check volatility forecast
        vix_forecast = None
        vix_symbol = next((s for s in self.volatility_indices if 'VIX' in s), None)
        
        if vix_symbol and vix_symbol in forecasts and '30_day' in forecasts[vix_symbol]:
            vix_forecast = forecasts[vix_symbol]['30_day']['forecast']
            
            if vix_forecast < 15:
                volatility_forecast = 'low'
            elif vix_forecast > 25:
                volatility_forecast = 'high'
            else:
                volatility_forecast = 'moderate'
        else:
            volatility_forecast = 'unknown'
        
        # Add overall assessment
        forecasts['overall_assessment'] = {
            'forecast_direction': forecast_direction,
            'volatility_forecast': volatility_forecast,
            'confidence': 'medium'  # This could be refined based on model agreement
        }
        
        return forecasts
    
def percentileofscore(a, score):
    """
    Calculate the percentile rank of a score relative to an array.
    
    This is a simplified version of scipy's percentileofscore function.
    
    Args:
        a: Array of values
        score: Value to find the percentile rank for
        
    Returns:
        Percentile rank (0-100)
    """
    a = np.asarray(a)
    n = len(a)
    
    if n == 0:
        return np.nan
        
    return np.sum(a <= score) / n * 100

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Alpha Vantage client
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
    else:
        client = AlphaVantageClient(api_key=api_key)
        
        # Create forecaster and forecast
        forecaster = StockForecast(client)
        df, forecast, model = forecaster.forecast('SPY', days=90)
        
        # Plot forecast
        forecaster.plot_forecast(df, forecast, model, symbol='SPY')
