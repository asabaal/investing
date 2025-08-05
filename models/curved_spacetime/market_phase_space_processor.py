#!/usr/bin/env python3
"""
Market Phase Space Processor - Production System for Real Market Data
Processes real market data from AlphaVantage and generates phase space analysis at scale.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import pickle
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from alpha_vantage_api import AlphaVantageClient
from market_data_database import MarketDataDatabase
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from phase_space_trajectory_3d import create_phase_space_trajectory_3d, create_phase_space_density_gradient, create_phase_space_time_gradient, create_interactive_combined_analysis
from animated_phase_space_evolution import create_animated_evolution
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PhaseSpaceAnalysis:
    """Results of phase space analysis for a security."""
    symbol: str
    analysis_date: datetime
    data_period: str  # e.g., "2020-01-01_to_2024-12-31"
    n_candles: int
    sentiment_range: Tuple[float, float]  # (min, max)
    uwr_range: Tuple[float, float]  # (min, max)
    phase_space_coverage: float  # Percentage of triangular region covered
    clustering_strength: float  # Measure of how clustered the data is
    market_regimes: List[str]  # Identified market regimes
    trajectory_path_length: float  # Length of trajectory in phase space
    mean_curvature: float  # Average spacetime curvature
    files_generated: List[str]  # List of visualization files generated
    clustering_analysis: Dict[str, Any]  # Comprehensive clustering results

@dataclass
class ProcessingProgress:
    """Track processing progress for each security."""
    symbol: str
    last_updated: datetime
    data_start_date: str
    data_end_date: str
    n_candles_processed: int
    status: str  # 'pending', 'processing', 'completed', 'error'
    error_message: Optional[str] = None
    files_generated: List[str] = None

class MarketPhaseSpaceProcessor:
    """Production system for processing market data into phase space analysis."""
    
    def __init__(self, 
                 database_path: Optional[str] = None,
                 output_dir: str = "./phase_space_analysis",
                 cache_dir: str = "./phase_space_cache"):
        """
        Initialize the market phase space processor.
        
        Args:
            database_path: Path to market data database
            output_dir: Directory to store generated visualizations (git ignored)
            cache_dir: Directory to store analysis cache and progress tracking
        """
        self.database_path = database_path or self._get_default_database_path()
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.db = MarketDataDatabase(self.database_path)
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
        self.api = AlphaVantageClient(api_key)
        
        # Progress tracking
        self.progress_file = self.cache_dir / "processing_progress.json"
        self.progress = self._load_progress()
        
        logger.info(f"Initialized MarketPhaseSpaceProcessor")
        logger.info(f"  Database: {self.database_path}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Cache directory: {self.cache_dir}")
    
    def _get_default_database_path(self) -> str:
        """Get default database path."""
        data_dir = Path.home() / '.market_data'
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / 'market_data.db')
    
    def _load_progress(self) -> Dict[str, ProcessingProgress]:
        """Load processing progress from cache."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return {
                    symbol: ProcessingProgress(**item) 
                    for symbol, item in data.items()
                }
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return {}
    
    def _save_progress(self):
        """Save processing progress to cache."""
        try:
            data = {
                symbol: asdict(progress) 
                for symbol, progress in self.progress.items()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save progress file: {e}")
    
    def update_database(self, symbols: List[str], days_back: int = 365) -> Dict[str, bool]:
        """
        Update database with latest market data for specified symbols.
        
        Args:
            symbols: List of stock symbols to update
            days_back: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping symbol to success status
        """
        results = {}
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"🔄 Updating database for {len(symbols)} symbols...")
        logger.info(f"   Date range: {start_date} to {end_date}")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"   [{i}/{len(symbols)}] Updating {symbol}...")
                
                # Check if we have recent data
                latest_date = self.db.get_latest_date(symbol)
                if latest_date and (end_date - latest_date).days < 2:
                    logger.info(f"      ✅ {symbol} is up to date (latest: {latest_date})")
                    results[symbol] = True
                    continue
                
                # Fetch new data
                data = self.api.get_daily(symbol, outputsize='full')
                if data is not None and len(data) > 0:
                    # Transform data to match database format
                    data_formatted = data.copy()
                    data_formatted = data_formatted.reset_index()  # Move date from index to column
                    data_formatted.rename(columns={'index': 'date', 'adjusted_close': 'adj_close'}, inplace=True)
                    
                    # Store in database
                    self.db.store_daily_data(symbol, data_formatted)
                    logger.info(f"      ✅ Updated {symbol} with {len(data)} records")
                    results[symbol] = True
                else:
                    logger.warning(f"      ❌ No data retrieved for {symbol}")
                    results[symbol] = False
                
                # Rate limiting
                time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
                
            except Exception as e:
                logger.error(f"      ❌ Error updating {symbol}: {e}")
                results[symbol] = False
        
        logger.info(f"✅ Database update complete: {sum(results.values())}/{len(symbols)} successful")
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols available in the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM daily_data ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def analyze_security_phase_space(self, symbol: str, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   min_candles: int = 100) -> Optional[PhaseSpaceAnalysis]:
        """
        Perform comprehensive phase space analysis for a single security.
        
        Args:
            symbol: Stock symbol to analyze
            start_date: Start date (YYYY-MM-DD) or None for all available data
            end_date: End date (YYYY-MM-DD) or None for latest
            min_candles: Minimum number of candles required for analysis
            
        Returns:
            PhaseSpaceAnalysis object or None if analysis failed
        """
        try:
            logger.info(f"🔬 Analyzing phase space for {symbol}...")
            
            # Update progress
            self.progress[symbol] = ProcessingProgress(
                symbol=symbol,
                last_updated=datetime.now(),
                data_start_date=start_date or "earliest",
                data_end_date=end_date or "latest",
                n_candles_processed=0,
                status='processing'
            )
            self._save_progress()
            
            # Get data from database
            data = self.db.get_daily_data(symbol, start_date, end_date)
            if data is None or len(data) < min_candles:
                error_msg = f"Insufficient data: {len(data) if data is not None else 0} candles (need {min_candles})"
                logger.warning(f"   ❌ {error_msg}")
                self.progress[symbol].status = 'error'
                self.progress[symbol].error_message = error_msg
                self._save_progress()
                return None
            
            logger.info(f"   📊 Processing {len(data)} candles from {data.index[0]} to {data.index[-1]}")
            
            # Transform column names to match expected format (lowercase)
            data_processed = data.copy()
            data_processed.columns = data_processed.columns.str.lower()
            # Map database column names to expected names
            column_mapping = {
                'unadjusted_close': 'close',  # Use unadjusted close as the main close price
                'close': 'adjusted_close'     # Keep adjusted close separately if needed
            }
            data_processed.rename(columns=column_mapping, inplace=True)
            
            # Create candle metrics
            candle_metrics = create_candle_metrics_from_ohlc(data_processed)
            geometry = CurvedCandleGeometry(candle_metrics)
            
            # Extract phase space coordinates
            sentiments = np.array([c.sentiment for c in candle_metrics])
            uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
            proper_times = geometry.compute_proper_time_series()
            
            # Calculate analysis metrics
            sentiment_range = (float(sentiments.min()), float(sentiments.max()))
            uwr_range = (float(uwrs.min()), float(uwrs.max()))
            
            # Phase space coverage (percentage of triangular region with data)
            from scipy.spatial import ConvexHull
            points = np.column_stack([sentiments, uwrs])
            hull = ConvexHull(points)
            triangle_area = 1.0  # Area of constraint triangle: |s| + u <= 1
            coverage = min(hull.volume / triangle_area, 1.0) * 100
            
            # Clustering strength (inverse of spread)
            clustering_strength = 1.0 / (np.std(sentiments) + np.std(uwrs) + 1e-6)
            
            # Trajectory path length
            path_diffs = np.sqrt(np.diff(sentiments)**2 + np.diff(uwrs)**2)
            trajectory_length = float(np.sum(path_diffs))
            
            # Mean curvature
            curvatures = [geometry.compute_intrinsic_curvature(i) for i in range(len(candle_metrics))]
            mean_curvature = float(np.mean(curvatures))
            
            # Perform comprehensive clustering analysis
            logger.info(f"   🔍 Performing clustering analysis...")
            clustering_analysis = self._perform_clustering_analysis(sentiments, uwrs)
            
            # Analyze cluster characteristics if optimal clustering found
            if clustering_analysis['optimal']['labels'] is not None:
                cluster_characteristics = self._analyze_cluster_characteristics(
                    sentiments, uwrs, clustering_analysis['optimal']['labels']
                )
                clustering_analysis['cluster_characteristics'] = cluster_characteristics
                
                logger.info(f"      📊 Found {len(cluster_characteristics)} natural clusters using {clustering_analysis['optimal']['method']}")
                for cluster_id, stats in cluster_characteristics.items():
                    logger.info(f"         • {cluster_id}: {stats['interpretation']} ({stats['percentage']:.1f}%)")
            
            # Identify market regimes (simplified)
            regimes = self._identify_market_regimes(sentiments, uwrs)
            
            # Generate visualizations (pass the processed data and clustering analysis)
            files_generated = self._generate_visualizations(symbol, data_processed, candle_metrics, 
                                                          sentiments, uwrs, clustering_analysis)
            
            # Create analysis result
            analysis = PhaseSpaceAnalysis(
                symbol=symbol,
                analysis_date=datetime.now(),
                data_period=f"{data.index[0].strftime('%Y-%m-%d')}_to_{data.index[-1].strftime('%Y-%m-%d')}",
                n_candles=len(candle_metrics),
                sentiment_range=sentiment_range,
                uwr_range=uwr_range,
                phase_space_coverage=coverage,
                clustering_strength=clustering_strength,
                market_regimes=regimes,
                trajectory_path_length=trajectory_length,
                mean_curvature=mean_curvature,
                files_generated=files_generated,
                clustering_analysis=clustering_analysis
            )
            
            # Save analysis to cache
            analysis_file = self.cache_dir / f"{symbol}_analysis.pkl"
            with open(analysis_file, 'wb') as f:
                pickle.dump(analysis, f)
            
            # Update progress
            self.progress[symbol].status = 'completed'
            self.progress[symbol].n_candles_processed = len(candle_metrics)
            self.progress[symbol].files_generated = files_generated
            self._save_progress()
            
            logger.info(f"   ✅ Analysis complete for {symbol}")
            logger.info(f"      📈 Sentiment range: {sentiment_range[0]:.3f} to {sentiment_range[1]:.3f}")
            logger.info(f"      📊 UWR range: {uwr_range[0]:.3f} to {uwr_range[1]:.3f}")
            logger.info(f"      🎯 Phase space coverage: {coverage:.1f}%")
            logger.info(f"      🌟 Files generated: {len(files_generated)}")
            
            return analysis
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            self.progress[symbol].status = 'error'
            self.progress[symbol].error_message = error_msg
            self._save_progress()
            return None
    
    def _perform_clustering_analysis(self, sentiments: np.ndarray, uwrs: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive clustering analysis to identify natural phase space structures."""
        
        # Prepare data for clustering
        data = np.column_stack([sentiments, uwrs])
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        clustering_results = {}
        
        # 1. K-Means clustering (try different k values)
        kmeans_results = {}
        for k in range(2, 8):  # Test 2-7 clusters
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data_scaled)
                
                # Calculate silhouette score and inertia
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(data_scaled, labels)
                
                kmeans_results[k] = {
                    'labels': labels,
                    'centers': scaler.inverse_transform(kmeans.cluster_centers_),
                    'silhouette_score': silhouette,
                    'inertia': kmeans.inertia_
                }
            except:
                continue
        
        clustering_results['kmeans'] = kmeans_results
        
        # 2. DBSCAN clustering (density-based)
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=max(5, len(data) // 100))
            dbscan_labels = dbscan.fit_predict(data_scaled)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)
            
            clustering_results['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(data) if len(data) > 0 else 0
            }
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
        
        # 3. Gaussian Mixture Model
        gmm_results = {}
        for n_components in range(2, 6):  # Test 2-5 components
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(data_scaled)
                labels = gmm.predict(data_scaled)
                
                gmm_results[n_components] = {
                    'labels': labels,
                    'means': scaler.inverse_transform(gmm.means_),
                    'covariances': gmm.covariances_,
                    'weights': gmm.weights_,
                    'aic': gmm.aic(data_scaled),
                    'bic': gmm.bic(data_scaled)
                }
            except:
                continue
        
        clustering_results['gmm'] = gmm_results
        
        # 4. Agglomerative clustering (hierarchical)
        try:
            from sklearn.cluster import AgglomerativeClustering
            agg_results = {}
            for n_clusters in range(2, 6):
                try:
                    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    agg_labels = agg.fit_predict(data_scaled)
                    
                    silhouette = silhouette_score(data_scaled, agg_labels)
                    
                    agg_results[n_clusters] = {
                        'labels': agg_labels,
                        'silhouette_score': silhouette,
                        'linkage': 'ward'
                    }
                except:
                    continue
            
            clustering_results['agglomerative'] = agg_results
        except Exception as e:
            logger.warning(f"Agglomerative clustering failed: {e}")
        
        # 5. Spectral clustering (for non-convex clusters)
        try:
            from sklearn.cluster import SpectralClustering
            spectral_results = {}
            for n_clusters in range(2, 6):
                try:
                    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                                gamma=1.0, affinity='rbf')
                    spectral_labels = spectral.fit_predict(data_scaled)
                    
                    silhouette = silhouette_score(data_scaled, spectral_labels)
                    
                    spectral_results[n_clusters] = {
                        'labels': spectral_labels,
                        'silhouette_score': silhouette,
                        'affinity': 'rbf'
                    }
                except:
                    continue
            
            clustering_results['spectral'] = spectral_results
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            
        # 6. Mean Shift clustering (finds natural number of clusters)
        try:
            from sklearn.cluster import MeanShift, estimate_bandwidth
            bandwidth = estimate_bandwidth(data_scaled, quantile=0.2, n_samples=min(500, len(data)))
            if bandwidth > 0:
                ms = MeanShift(bandwidth=bandwidth)
                ms_labels = ms.fit_predict(data_scaled)
                n_clusters = len(set(ms_labels))
                
                if n_clusters > 1:
                    silhouette = silhouette_score(data_scaled, ms_labels)
                    clustering_results['meanshift'] = {
                        'labels': ms_labels,
                        'centers': scaler.inverse_transform(ms.cluster_centers_),
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette,
                        'bandwidth': bandwidth
                    }
        except Exception as e:
            logger.warning(f"Mean Shift clustering failed: {e}")
        
        # 7. Find optimal clustering
        optimal_clustering = self._find_optimal_clustering(clustering_results)
        clustering_results['optimal'] = optimal_clustering
        
        return clustering_results
    
    def _find_optimal_clustering(self, clustering_results: Dict) -> Dict[str, Any]:
        """Find the optimal clustering based on various metrics."""
        
        best_method = None
        best_score = -1
        best_params = None
        best_labels = None
        
        # Evaluate K-means results
        if 'kmeans' in clustering_results:
            for k, result in clustering_results['kmeans'].items():
                score = result['silhouette_score']
                if score > best_score:
                    best_score = score
                    best_method = 'kmeans'
                    best_params = {'k': k}
                    best_labels = result['labels']
        
        # Evaluate Agglomerative clustering results
        if 'agglomerative' in clustering_results:
            for n_clusters, result in clustering_results['agglomerative'].items():
                score = result['silhouette_score']
                if score > best_score:
                    best_score = score
                    best_method = 'agglomerative'
                    best_params = {'n_clusters': n_clusters}
                    best_labels = result['labels']
        
        # Evaluate Spectral clustering results
        if 'spectral' in clustering_results:
            for n_clusters, result in clustering_results['spectral'].items():
                score = result['silhouette_score']
                if score > best_score:
                    best_score = score
                    best_method = 'spectral'
                    best_params = {'n_clusters': n_clusters}
                    best_labels = result['labels']
        
        # Evaluate Mean Shift results
        if 'meanshift' in clustering_results:
            result = clustering_results['meanshift']
            score = result['silhouette_score']
            if score > best_score:
                best_score = score
                best_method = 'meanshift'
                best_params = {'n_clusters': result['n_clusters']}
                best_labels = result['labels']
        
        # Evaluate GMM results (using BIC as criterion, but still compare with silhouette)
        if 'gmm' in clustering_results:
            for n_comp, result in clustering_results['gmm'].items():
                # Convert BIC to a comparable score (lower BIC is better, so negate and normalize)
                bic_score = -result['bic'] / 1000  # Rough normalization
                if bic_score > best_score * 0.9:  # Allow GMM if BIC suggests it's competitive
                    best_score = bic_score
                    best_method = 'gmm'
                    best_params = {'n_components': n_comp}
                    best_labels = result['labels']
        
        return {
            'method': best_method,
            'params': best_params,
            'labels': best_labels,
            'score': best_score
        }
    
    def _analyze_cluster_characteristics(self, sentiments: np.ndarray, uwrs: np.ndarray, 
                                       labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of identified clusters."""
        
        unique_labels = np.unique(labels)
        cluster_analysis = {}
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            mask = labels == label
            cluster_sentiments = sentiments[mask]
            cluster_uwrs = uwrs[mask]
            
            # Calculate cluster statistics
            cluster_stats = {
                'size': int(np.sum(mask)),
                'percentage': float(np.mean(mask) * 100),
                'sentiment_mean': float(np.mean(cluster_sentiments)),
                'sentiment_std': float(np.std(cluster_sentiments)),
                'uwr_mean': float(np.mean(cluster_uwrs)),
                'uwr_std': float(np.std(cluster_uwrs)),
                'sentiment_range': [float(np.min(cluster_sentiments)), float(np.max(cluster_sentiments))],
                'uwr_range': [float(np.min(cluster_uwrs)), float(np.max(cluster_uwrs))]
            }
            
            # Interpret cluster meaning
            interpretation = self._interpret_cluster(cluster_stats)
            cluster_stats['interpretation'] = interpretation
            
            cluster_analysis[f'cluster_{label}'] = cluster_stats
        
        return cluster_analysis
    
    def _interpret_cluster(self, stats: Dict[str, Any]) -> str:
        """Interpret what a cluster represents in market terms."""
        
        sentiment_mean = stats['sentiment_mean']
        uwr_mean = stats['uwr_mean']
        
        if sentiment_mean > 0.3 and uwr_mean < 0.3:
            return 'strong_bullish'
        elif sentiment_mean > 0.1 and uwr_mean < 0.4:
            return 'moderate_bullish'
        elif sentiment_mean < -0.3 and uwr_mean > 0.4:
            return 'strong_bearish'
        elif sentiment_mean < -0.1 and uwr_mean > 0.3:
            return 'moderate_bearish'
        elif abs(sentiment_mean) < 0.1 and uwr_mean > 0.5:
            return 'high_volatility'
        elif abs(sentiment_mean) < 0.2 and uwr_mean < 0.3:
            return 'consolidation'
        elif uwr_mean > 0.6:
            return 'extreme_volatility'
        else:
            return 'mixed_regime'
    
    def _identify_market_regimes(self, sentiments: np.ndarray, uwrs: np.ndarray) -> List[str]:
        """Identify market regimes based on clustering in phase space (legacy method)."""
        regimes = []
        
        # Bullish regime: positive sentiment, low UWR
        bullish_mask = (sentiments > 0.2) & (uwrs < 0.4)
        if np.sum(bullish_mask) > len(sentiments) * 0.1:
            regimes.append('bullish_trending')
        
        # Bearish regime: negative sentiment, high UWR
        bearish_mask = (sentiments < -0.2) & (uwrs > 0.4)
        if np.sum(bearish_mask) > len(sentiments) * 0.1:
            regimes.append('bearish_correction')
        
        # High volatility: spread across phase space
        volatility_mask = (np.abs(sentiments) < 0.1) & (uwrs > 0.5)
        if np.sum(volatility_mask) > len(sentiments) * 0.1:
            regimes.append('high_volatility')
        
        # Consolidation: centered around origin
        consolidation_mask = (np.abs(sentiments) < 0.2) & (uwrs < 0.3)
        if np.sum(consolidation_mask) > len(sentiments) * 0.2:
            regimes.append('consolidation')
        
        return regimes if regimes else ['undefined']
    
    def _create_clustering_visualization(self, data: pd.DataFrame, sentiments: np.ndarray, 
                                       uwrs: np.ndarray, clustering_analysis: Dict[str, Any]) -> go.Figure:
        """Create a visualization showing the identified clusters in phase space with silhouette score inset."""
        
        # Create figure with subplots - main plot and inset
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.75, 0.25],
            subplot_titles=('Phase Space Clusters', 'Silhouette Scores'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        optimal = clustering_analysis.get('optimal', {})
        labels = optimal.get('labels')
        
        if labels is None:
            # Create a simple scatter plot if no clustering found
            fig.add_trace(go.Scatter(
                x=sentiments, y=uwrs,
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.6),
                name='Market Data',
                hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
            ), row=1, col=1)
        else:
            # Create scatter plot with cluster colors
            unique_labels = np.unique(labels)
            colors = pc.qualitative.Set1[:len(unique_labels)]
            
            cluster_characteristics = clustering_analysis.get('cluster_characteristics', {})
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    color = 'black'
                    name = 'Noise'
                else:
                    color = colors[i % len(colors)]
                    cluster_info = cluster_characteristics.get(f'cluster_{label}', {})
                    interpretation = cluster_info.get('interpretation', f'Cluster {label}')
                    percentage = cluster_info.get('percentage', 0)
                    name = f'{interpretation.replace("_", " ").title()} ({percentage:.1f}%)'
                
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=sentiments[mask], 
                    y=uwrs[mask],
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.7),
                    name=name,
                    hovertemplate=f'<b>{name}</b><br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)
        
        # Add constraint boundary
        sentiment_boundary = np.linspace(-1, 1, 100)
        uwr_upper = 1 - np.abs(sentiment_boundary)
        
        fig.add_trace(go.Scatter(
            x=sentiment_boundary, y=uwr_upper,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Phase Space Boundary',
            hovertemplate='Constraint: |sentiment| + UWR ≤ 1<extra></extra>',
            showlegend=True
        ), row=1, col=1)
        
        # Add silhouette score plot
        kmeans_results = clustering_analysis.get('kmeans', {})
        if kmeans_results:
            k_values = []
            silhouette_scores = []
            for k, result in sorted(kmeans_results.items()):
                k_values.append(k)
                silhouette_scores.append(result['silhouette_score'])
            
            # Plot silhouette scores
            fig.add_trace(go.Scatter(
                x=k_values,
                y=silhouette_scores,
                mode='lines+markers',
                line=dict(color='cyan', width=2),
                marker=dict(size=8, color='cyan'),
                name='Silhouette Score',
                showlegend=False,
                hovertemplate='k=%{x}<br>Score: %{y:.3f}<extra></extra>'
            ), row=1, col=2)
            
            # Highlight the optimal k
            optimal_k = optimal.get('params', {}).get('k')
            if optimal_k and optimal_k in k_values:
                idx = k_values.index(optimal_k)
                fig.add_trace(go.Scatter(
                    x=[optimal_k],
                    y=[silhouette_scores[idx]],
                    mode='markers',
                    marker=dict(size=12, color='yellow', symbol='star'),
                    name=f'Optimal (k={optimal_k})',
                    showlegend=False,
                    hovertemplate=f'Optimal k={optimal_k}<br>Score: {silhouette_scores[idx]:.3f}<extra></extra>'
                ), row=1, col=2)
        
        method_name = optimal.get('method', 'None')
        params = optimal.get('params', {})
        title = f"Phase Space Clustering Analysis - {method_name.upper()}"
        if params:
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            title += f" ({param_str})"
        
        # Update layout
        fig.update_xaxes(title_text="Sentiment", row=1, col=1)
        fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            hovermode='closest',
            width=1200,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        return fig
    
    def _create_comprehensive_clustering_visualization(self, data: pd.DataFrame, sentiments: np.ndarray, 
                                                     uwrs: np.ndarray, clustering_analysis: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive clustering comparison visualization with density background and dark mode."""
        
        # Calculate log density background for all subplots
        from scipy.stats import gaussian_kde
        points = np.column_stack([sentiments, uwrs])
        kde = gaussian_kde(points.T, bw_method='scott')
        
        # Create density grid
        grid_res = 60
        sentiment_range = np.linspace(-0.99, 0.99, grid_res)
        uwr_range = np.linspace(0.01, 0.99, grid_res)
        S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
        
        density_grid = np.zeros_like(S_grid)
        for i in range(grid_res):
            for j in range(grid_res):
                s, u = S_grid[i, j], U_grid[i, j]
                if abs(s) + u <= 1.0:
                    density_grid[i, j] = kde([s, u])[0]
                else:
                    density_grid[i, j] = np.nan
        
        # Apply log scale
        epsilon = 1e-10
        density_grid_log = np.log1p(density_grid + epsilon)
        density_grid_log[np.isnan(density_grid)] = np.nan
        
        # Define algorithms to display and their order
        algorithm_order = ['kmeans', 'dbscan', 'agglomerative', 'spectral', 'meanshift', 'gmm']
        algorithm_names = {
            'kmeans': 'K-Means',
            'dbscan': 'DBSCAN',
            'agglomerative': 'Agglomerative',
            'spectral': 'Spectral',
            'meanshift': 'Mean Shift',
            'gmm': 'Gaussian Mixture'
        }
        
        # Filter available algorithms
        available_algos = [algo for algo in algorithm_order if algo in clustering_analysis]
        n_algos = len(available_algos)
        
        if n_algos == 0:
            # Fallback to simple visualization
            return self._create_clustering_visualization(data, sentiments, uwrs, clustering_analysis)
        
        # Create subplots (2 rows, 3 columns max, or adjust based on available algorithms)
        if n_algos <= 3:
            rows, cols = 1, n_algos
        elif n_algos <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        subplot_titles = []
        for algo in available_algos[:rows*cols]:
            name = algorithm_names.get(algo, algo.title())
            if algo == 'dbscan':
                result = clustering_analysis[algo]
                title = f"{name}<br>({result['n_clusters']} clusters, {result['noise_ratio']:.1%} noise)"
            elif algo == 'meanshift':
                result = clustering_analysis[algo]
                title = f"{name}<br>({result['n_clusters']} clusters)"
            else:
                # Find best result for this algorithm
                if algo in ['kmeans', 'agglomerative', 'spectral']:
                    best_k = max(clustering_analysis[algo].keys(), 
                               key=lambda k: clustering_analysis[algo][k]['silhouette_score'])
                    title = f"{name}<br>(k={best_k}, sil={clustering_analysis[algo][best_k]['silhouette_score']:.3f})"
                elif algo == 'gmm':
                    best_n = min(clustering_analysis[algo].keys(),
                               key=lambda n: clustering_analysis[algo][n]['bic'])
                    title = f"{name}<br>(n={best_n}, BIC={clustering_analysis[algo][best_n]['bic']:.0f})"
                else:
                    title = name
            subplot_titles.append(title)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        # Color schemes for different algorithms
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F8C471']
        
        optimal_method = clustering_analysis.get('optimal', {}).get('method')
        
        for idx, algo in enumerate(available_algos[:rows*cols]):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Add density background to each subplot
            fig.add_trace(
                go.Heatmap(
                    x=sentiment_range,
                    y=uwr_range,
                    z=density_grid_log,
                    colorscale=[
                        [0, 'rgba(13, 17, 23, 0.9)'],    # Dark background
                        [0.2, 'rgba(26, 27, 94, 0.7)'],  # Deep blue
                        [0.4, 'rgba(46, 79, 153, 0.5)'], # Medium blue  
                        [0.6, 'rgba(76, 124, 219, 0.3)'], # Light blue
                        [0.8, 'rgba(123, 167, 255, 0.2)'], # Bright blue
                        [1, 'rgba(179, 217, 255, 0.1)']    # Very bright blue
                    ],
                    showscale=False,
                    hoverinfo='skip',
                    opacity=0.6
                ),
                row=row, col=col
            )
            
            # Get labels for this algorithm
            labels = None
            if algo == 'dbscan':
                labels = clustering_analysis[algo]['labels']
            elif algo == 'meanshift':
                labels = clustering_analysis[algo]['labels']
            elif algo in ['kmeans', 'agglomerative', 'spectral']:
                # Get best result
                best_k = max(clustering_analysis[algo].keys(),
                           key=lambda k: clustering_analysis[algo][k]['silhouette_score'])
                labels = clustering_analysis[algo][best_k]['labels']
            elif algo == 'gmm':
                best_n = min(clustering_analysis[algo].keys(),
                           key=lambda n: clustering_analysis[algo][n]['bic'])
                labels = clustering_analysis[algo][best_n]['labels']
            
            if labels is not None:
                unique_labels = np.unique(labels)
                
                for i, label in enumerate(unique_labels):
                    if label == -1:  # Noise points (DBSCAN)
                        color = 'gray'
                        name = 'Noise'
                        opacity = 0.4
                    else:
                        color = colors[i % len(colors)]
                        name = f'Cluster {label}'
                        opacity = 0.8
                    
                    mask = labels == label
                    
                    # Add highlight if this is the optimal method
                    marker_line = dict(color='yellow', width=3) if algo == optimal_method else dict(color='white', width=1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sentiments[mask],
                            y=uwrs[mask],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=color,
                                opacity=opacity,
                                line=marker_line
                            ),
                            name=f'{algorithm_names[algo]} {name}',
                            showlegend=bool(idx == 0 and label != -1),  # Only show legend for first algo, non-noise
                            hovertemplate=f'<b>{algorithm_names[algo]} {name}</b><br>' +
                                          'Sentiment: %{x:.3f}<br>' +
                                          'UWR: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=col
                    )
            
            # Add phase space boundary
            boundary_s = np.linspace(-0.99, 0.99, 50)
            boundary_u = 1.0 - np.abs(boundary_s)
            
            fig.add_trace(
                go.Scatter(
                    x=boundary_s,
                    y=boundary_u,
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
        
        # Update layout for dark mode
        fig.update_layout(
            title=dict(
                text='🔬 Comprehensive Clustering Algorithm Comparison<br>' +
                     '<sub>Multiple algorithms with log density background - Yellow borders indicate optimal method</sub>',
                x=0.5,
                font=dict(size=16, color='white')
            ),
            template='plotly_dark',
            height=400 * rows,
            width=500 * cols,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(13, 17, 23, 0.8)',
                bordercolor='white',
                borderwidth=1,
                font=dict(color='white')
            ),
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117'
        )
        
        # Update axes for all subplots
        for i in range(1, rows * cols + 1):
            row = (i - 1) // cols + 1
            col = (i - 1) % cols + 1
            fig.update_xaxes(
                title='← Bearish | Sentiment | Bullish →',
                range=[-1, 1],
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                row=row, col=col
            )
            fig.update_yaxes(
                title='Upper Wick Ratio ↑',
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                row=row, col=col
            )
        
        return fig
    
    def _generate_visualizations(self, symbol: str, data: pd.DataFrame, candle_metrics: List,
                               sentiments: np.ndarray, uwrs: np.ndarray, 
                               clustering_analysis: Dict[str, Any]) -> List[str]:
        """Generate all phase space visualizations for a security."""
        files_generated = []
        
        try:
            # Create symbol-specific output directory
            symbol_dir = self.output_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # 1. 3D Phase Space Trajectory
            logger.info(f"      🌌 Generating 3D trajectory...")
            fig_3d = create_phase_space_trajectory_3d(data)
            file_3d = symbol_dir / f"{symbol}_3d_trajectory.html"
            fig_3d.write_html(str(file_3d))
            files_generated.append(str(file_3d))
            
            # 2. Density Gradient
            logger.info(f"      🎯 Generating density gradient...")
            fig_density = create_phase_space_density_gradient(data)
            file_density = symbol_dir / f"{symbol}_density_gradient.html"
            fig_density.write_html(str(file_density))
            files_generated.append(str(file_density))
            
            # 3. Proper Time Gradient
            logger.info(f"      ⏰ Generating proper time gradient...")
            fig_proper_time = create_phase_space_time_gradient(data, use_proper_time=True)
            file_proper_time = symbol_dir / f"{symbol}_proper_time_gradient.html"
            fig_proper_time.write_html(str(file_proper_time))
            files_generated.append(str(file_proper_time))
            
            # 4. Coordinate Time Gradient
            logger.info(f"      🕰️ Generating coordinate time gradient...")
            fig_coord_time = create_phase_space_time_gradient(data, use_proper_time=False)
            file_coord_time = symbol_dir / f"{symbol}_coord_time_gradient.html"
            fig_coord_time.write_html(str(file_coord_time))
            files_generated.append(str(file_coord_time))
            
            # 5. Combined Interactive Analysis
            logger.info(f"      🎯 Generating combined interactive analysis...")
            fig_combined = create_interactive_combined_analysis(data)
            file_combined = symbol_dir / f"{symbol}_combined_analysis.html"
            fig_combined.write_html(str(file_combined))
            files_generated.append(str(file_combined))
            
            # 6. Comprehensive Clustering Analysis Visualization
            logger.info(f"      🔍 Generating clustering analysis...")
            fig_clustering = self._create_comprehensive_clustering_visualization(data, sentiments, uwrs, clustering_analysis)
            file_clustering = symbol_dir / f"{symbol}_clustering_analysis.html"
            fig_clustering.write_html(str(file_clustering))
            files_generated.append(str(file_clustering))
            
            # 7. Animated Evolution (for smaller datasets)
            if len(data) <= 300:  # Only for manageable sizes
                logger.info(f"      🎬 Generating animated evolution...")
                fig_animation = create_animated_evolution(data, start_frame=10, end_frame=len(data)-1)
                file_animation = symbol_dir / f"{symbol}_animated_evolution.html"
                fig_animation.write_html(str(file_animation))
                files_generated.append(str(file_animation))
            
            logger.info(f"      ✅ Generated {len(files_generated)} visualization files")
            
        except Exception as e:
            logger.error(f"      ❌ Error generating visualizations: {e}")
        
        return files_generated
    
    def batch_process_securities(self, symbols: List[str], 
                                update_data: bool = True,
                                parallel: bool = False) -> Dict[str, PhaseSpaceAnalysis]:
        """
        Process multiple securities in batch.
        
        Args:
            symbols: List of symbols to process
            update_data: Whether to update database first
            parallel: Whether to process in parallel (future enhancement)
            
        Returns:
            Dictionary mapping symbol to analysis results
        """
        logger.info(f"🚀 Starting batch processing of {len(symbols)} securities...")
        
        results = {}
        
        # Update database if requested
        if update_data:
            update_results = self.update_database(symbols)
            failed_updates = [s for s, success in update_results.items() if not success]
            if failed_updates:
                logger.warning(f"⚠️ Failed to update data for: {failed_updates}")
        
        # Process each security
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📈 [{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                analysis = self.analyze_security_phase_space(symbol)
                if analysis:
                    results[symbol] = analysis
                    logger.info(f"   ✅ {symbol} processed successfully")
                else:
                    logger.warning(f"   ❌ {symbol} processing failed")
            
            except Exception as e:
                logger.error(f"   ❌ Error processing {symbol}: {e}")
        
        logger.info(f"🎉 Batch processing complete: {len(results)}/{len(symbols)} successful")
        
        # Generate summary report
        self._generate_batch_summary(results)
        
        return results
    
    def _generate_batch_summary(self, results: Dict[str, PhaseSpaceAnalysis]):
        """Generate a summary report of batch processing results."""
        try:
            summary_file = self.output_dir / "batch_processing_summary.json"
            
            summary = {
                'processing_date': datetime.now().isoformat(),
                'total_securities': len(results),
                'summary_stats': {
                    'avg_candles': np.mean([r.n_candles for r in results.values()]),
                    'avg_coverage': np.mean([r.phase_space_coverage for r in results.values()]),
                    'avg_clustering': np.mean([r.clustering_strength for r in results.values()]),
                    'common_regimes': self._get_common_regimes(results)
                },
                'securities': {
                    symbol: asdict(analysis) 
                    for symbol, analysis in results.items()
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📊 Summary report saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
    
    def _get_common_regimes(self, results: Dict[str, PhaseSpaceAnalysis]) -> Dict[str, int]:
        """Get frequency count of market regimes across all securities."""
        regime_counts = {}
        for analysis in results.values():
            for regime in analysis.market_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return regime_counts
    
    def get_processing_status(self) -> Dict[str, str]:
        """Get current processing status for all securities."""
        return {symbol: progress.status for symbol, progress in self.progress.items()}
    
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old visualization files."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cleaned_count = 0
        for file_path in self.output_dir.rglob("*.html"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                cleaned_count += 1
        
        logger.info(f"🧹 Cleaned up {cleaned_count} old files")


def main():
    """Example usage of the Market Phase Space Processor."""
    
    # Initialize processor
    processor = MarketPhaseSpaceProcessor()
    
    # Get list of available symbols
    available_symbols = processor.get_available_symbols()
    logger.info(f"📊 Found {len(available_symbols)} symbols in database")
    
    if not available_symbols:
        logger.info("🔄 No symbols found. Let's add some popular ones...")
        # Common symbols to start with
        popular_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'TLT', 'GLD']
        processor.update_database(popular_symbols, days_back=1000)  # 3+ years of data
        available_symbols = processor.get_available_symbols()
    
    # Process a subset for demonstration
    symbols_to_process = available_symbols[:5]  # First 5 symbols
    logger.info(f"🎯 Processing {len(symbols_to_process)} symbols: {symbols_to_process}")
    
    # Batch process
    results = processor.batch_process_securities(
        symbols=symbols_to_process,
        update_data=True
    )
    
    # Print summary
    logger.info(f"\n🎉 PROCESSING COMPLETE!")
    logger.info(f"✅ Successfully processed: {len(results)} securities")
    
    for symbol, analysis in results.items():
        logger.info(f"   📈 {symbol}:")
        logger.info(f"      Candles: {analysis.n_candles}")
        logger.info(f"      Coverage: {analysis.phase_space_coverage:.1f}%")
        logger.info(f"      Regimes: {', '.join(analysis.market_regimes)}")
        logger.info(f"      Files: {len(analysis.files_generated)}")
    
    logger.info(f"\n📁 Visualizations saved to: {processor.output_dir}")
    logger.info(f"📊 Summary report: {processor.output_dir}/batch_processing_summary.json")


if __name__ == "__main__":
    main()