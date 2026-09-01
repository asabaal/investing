"""
Gradient-Based Candle Clustering for Trade Likelihood Estimation

Uses gradient (rate of change) features to create coordinate-invariant
candle evolution patterns for improved trade probability modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from market_data_database import MarketDataDatabase

logger = logging.getLogger(__name__)

@dataclass
class GradientState:
    """Represents coordinate-invariant gradient state of candle evolution"""
    body_ratio_gradient: float      # Δ body_ratio
    upper_wick_gradient: float      # Δ upper_wick_ratio  
    log_range_gradient: float       # Δ log(range)
    log_price_gradient: float       # Δ log(price) = returns
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for clustering"""
        return np.array([
            self.body_ratio_gradient,
            self.upper_wick_gradient,
            self.log_range_gradient,
            self.log_price_gradient
        ])

@dataclass
class GradientCluster:
    """Properties of a gradient cluster"""
    cluster_id: int
    n_points: int
    percentage: float
    centroid: np.ndarray  # Mean gradients
    covariance: np.ndarray
    interpretation: str
    
class GradientCandleAnalyzer:
    """Analyzes candle evolution using gradient-based clustering"""
    
    def __init__(self, market_db: Optional[MarketDataDatabase] = None):
        self.market_db = market_db or MarketDataDatabase()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_method = None
        self.gradient_clusters: List[GradientCluster] = []
        self.transition_matrix: Optional[np.ndarray] = None
        
    def compute_candle_gradients(self, df: pd.DataFrame) -> List[GradientState]:
        """
        Convert OHLC data to gradient states
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            List of gradient states (one less than input candles)
        """
        # Detect column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        
        gradients = []
        
        for i in range(1, len(df)):
            # Current and previous candle
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Compute geometric ratios for both candles
            def compute_ratios(row):
                h, l, o, c = row[high_col], row[low_col], row[open_col], row[close_col]
                range_val = h - l
                if range_val == 0:
                    return 0, 0, 0  # Avoid division by zero
                
                body_ratio = (c - o) / range_val
                upper_wick_ratio = (h - max(o, c)) / range_val
                lower_wick_ratio = (min(o, c) - l) / range_val
                
                return body_ratio, upper_wick_ratio, lower_wick_ratio
            
            prev_body, prev_upper, prev_lower = compute_ratios(prev)
            curr_body, curr_upper, curr_lower = compute_ratios(curr)
            
            # Compute gradients
            body_gradient = curr_body - prev_body
            upper_gradient = curr_upper - prev_upper
            lower_gradient = curr_lower - prev_lower
            
            # Coordinate system gradients
            prev_range = prev[high_col] - prev[low_col]
            curr_range = curr[high_col] - curr[low_col]
            
            if prev_range > 0 and curr_range > 0:
                log_range_gradient = np.log(curr_range / prev_range)
            else:
                log_range_gradient = 0
                
            if prev[close_col] > 0 and curr[close_col] > 0:
                log_price_gradient = np.log(curr[close_col] / prev[close_col])
            else:
                log_price_gradient = 0
            
            gradient_state = GradientState(
                body_ratio_gradient=body_gradient,
                upper_wick_gradient=upper_gradient,
                log_range_gradient=log_range_gradient,
                log_price_gradient=log_price_gradient
            )
            
            gradients.append(gradient_state)
            
        return gradients
    
    def cluster_gradients(self, gradient_states: List[GradientState], 
                         n_clusters_range: Tuple[int, int] = (3, 8)) -> Dict[str, Any]:
        """
        Cluster gradient states using multiple algorithms
        
        Args:
            gradient_states: List of gradient states
            n_clusters_range: Range of cluster counts to test
            
        Returns:
            Dict with clustering results
        """
        if len(gradient_states) < 10:
            logger.warning("Not enough gradient states for clustering")
            return {}
            
        # Convert to matrix
        X = np.array([gs.to_array() for gs in gradient_states])
        
        # Handle any NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Clustering {len(gradient_states)} gradient states")
        logger.info(f"Feature matrix shape: {X_scaled.shape}")
        logger.info(f"Feature ranges: {X_scaled.min(axis=0)} to {X_scaled.max(axis=0)}")
        
        results = {}
        best_score = -1
        best_method = None
        best_labels = None
        best_params = {}
        
        # Test different clustering methods
        min_clusters, max_clusters = n_clusters_range
        
        # 1. KMeans
        kmeans_results = {}
        for n in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                kmeans_results[n] = {
                    'labels': labels,
                    'score': score,
                    'centroids': kmeans.cluster_centers_
                }
                
                if score > best_score:
                    best_score = score
                    best_method = 'kmeans'
                    best_labels = labels
                    best_params = {'n_clusters': n}
                    
            except Exception as e:
                logger.warning(f"KMeans with {n} clusters failed: {e}")
                
        results['kmeans'] = kmeans_results
        
        # 2. Gaussian Mixture Models
        gmm_results = {}
        for n in range(min_clusters, max_clusters + 1):
            try:
                gmm = GaussianMixture(n_components=n, random_state=42)
                labels = gmm.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                gmm_results[n] = {
                    'labels': labels,
                    'score': score,
                    'bic': gmm.bic(X_scaled),
                    'aic': gmm.aic(X_scaled)
                }
                
                if score > best_score:
                    best_score = score
                    best_method = 'gmm'
                    best_labels = labels
                    best_params = {'n_components': n}
                    
            except Exception as e:
                logger.warning(f"GMM with {n} components failed: {e}")
                
        results['gmm'] = gmm_results
        
        # 3. Agglomerative Clustering
        agg_results = {}
        for n in range(min_clusters, max_clusters + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=n, linkage='ward')
                labels = agg.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                agg_results[n] = {
                    'labels': labels,
                    'score': score
                }
                
                if score > best_score:
                    best_score = score
                    best_method = 'agglomerative'
                    best_labels = labels
                    best_params = {'n_clusters': n}
                    
            except Exception as e:
                logger.warning(f"Agglomerative with {n} clusters failed: {e}")
                
        results['agglomerative'] = agg_results
        
        # Store best results
        results['best'] = {
            'method': best_method,
            'params': best_params,
            'labels': best_labels,
            'score': best_score,
            'n_clusters': len(np.unique(best_labels)) if best_labels is not None else 0
        }
        
        logger.info(f"Best clustering: {best_method} with {best_params} (score: {best_score:.3f})")
        
        return results
    
    def interpret_gradient_cluster(self, centroid: np.ndarray) -> str:
        """
        Interpret what a gradient cluster represents in market terms
        
        Args:
            centroid: [body_gradient, upper_wick_gradient, 
                      log_range_gradient, log_price_gradient]
        """
        body_grad, upper_grad, range_grad, price_grad = centroid
        # Lower wick gradient can be derived: -(body_grad + upper_grad)
        lower_grad = -(body_grad + upper_grad)
        
        # Strong directional moves
        if abs(body_grad) > 0.3:
            direction = "Bullish" if body_grad > 0 else "Bearish"
            if abs(range_grad) > 0.1:
                volatility = "High Vol" if range_grad > 0 else "Low Vol"
                return f"Strong {direction} + {volatility}"
            return f"Strong {direction} Trend"
        
        # Volatility changes
        if abs(range_grad) > 0.2:
            vol_change = "Expanding" if range_grad > 0 else "Contracting"
            if abs(upper_grad) > 0.2:
                rejection = "High Rejection" if upper_grad > 0 else "Low Rejection"
                return f"{vol_change} Volatility + {rejection}"
            return f"{vol_change} Volatility"
        
        # Wick pattern changes
        if abs(upper_grad) > 0.2 or abs(lower_grad) > 0.2:
            if upper_grad > 0.2:
                return "Increasing Rejection"
            elif upper_grad < -0.2:
                return "Decreasing Rejection"
            elif lower_grad > 0.2:
                return "Increasing Support Test"
            else:
                return "Decreasing Support Test"
        
        # Price momentum
        if abs(price_grad) > 0.02:  # 2% moves
            momentum = "Strong Up" if price_grad > 0 else "Strong Down"
            return f"{momentum} Momentum"
        
        # Default
        return "Consolidation/Neutral"
    
    def build_gradient_clusters(self, gradient_states: List[GradientState]) -> bool:
        """Build gradient clusters from data"""
        clustering_results = self.cluster_gradients(gradient_states)
        
        if not clustering_results or 'best' not in clustering_results:
            logger.error("Clustering failed")
            return False
            
        best = clustering_results['best']
        labels = best['labels']
        n_clusters = best['n_clusters']
        
        if labels is None or n_clusters == 0:
            logger.error("No valid clustering found")
            return False
            
        # Convert gradients to matrix for analysis
        X = np.array([gs.to_array() for gs in gradient_states])
        X_scaled = self.scaler.transform(X)
        
        # Create cluster objects
        self.gradient_clusters = []
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_points = X_scaled[mask]
            
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                covariance = np.cov(cluster_points.T)
                interpretation = self.interpret_gradient_cluster(centroid)
                
                cluster = GradientCluster(
                    cluster_id=cluster_id,
                    n_points=len(cluster_points),
                    percentage=len(cluster_points) / len(X_scaled) * 100,
                    centroid=centroid,
                    covariance=covariance,
                    interpretation=interpretation
                )
                
                self.gradient_clusters.append(cluster)
        
        logger.info(f"Created {len(self.gradient_clusters)} gradient clusters:")
        for cluster in self.gradient_clusters:
            logger.info(f"  Cluster {cluster.cluster_id}: {cluster.interpretation} "
                       f"({cluster.n_points} points, {cluster.percentage:.1f}%)")
        
        return True
    
    def calculate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Calculate transition probabilities between gradient clusters"""
        n_clusters = len(np.unique(labels))
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        # Count transitions
        for i in range(len(labels) - 1):
            from_cluster = labels[i]
            to_cluster = labels[i + 1]
            transition_matrix[from_cluster, to_cluster] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_clusters):
            if row_sums[i] > 0:
                transition_matrix[i] = transition_matrix[i] / row_sums[i]
        
        self.transition_matrix = transition_matrix
        logger.info(f"Built transition matrix: {transition_matrix.shape}")
        
        return transition_matrix
    
    def analyze_symbol(self, symbol: str, start_date: str = '2024-01-01', 
                      end_date: str = '2025-08-31') -> bool:
        """
        Complete gradient analysis for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            True if analysis successful
        """
        logger.info(f"Analyzing gradient patterns for {symbol}")
        
        # Get market data
        df = self.market_db.get_data(symbol, start_date, end_date, 'daily')
        
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return False
            
        logger.info(f"Loaded {len(df)} candles for {symbol}")
        
        # Compute gradients
        gradient_states = self.compute_candle_gradients(df)
        
        if len(gradient_states) < 10:
            logger.error(f"Not enough gradient states: {len(gradient_states)}")
            return False
            
        logger.info(f"Computed {len(gradient_states)} gradient states")
        
        # Build clusters
        if not self.build_gradient_clusters(gradient_states):
            return False
            
        # Get cluster labels for transition matrix
        clustering_results = self.cluster_gradients(gradient_states)
        if clustering_results and 'best' in clustering_results:
            labels = clustering_results['best']['labels']
            self.calculate_transition_matrix(labels)
        
        logger.info(f"Gradient analysis complete for {symbol}")
        return True
    
    def predict_next_gradient_probabilities(self, current_gradient: GradientState) -> Dict[int, float]:
        """
        Predict probabilities of next gradient cluster given current gradient
        
        Args:
            current_gradient: Current gradient state
            
        Returns:
            Dict mapping cluster_id to probability
        """
        if not self.gradient_clusters or self.transition_matrix is None:
            return {}
            
        # Find closest cluster to current gradient
        current_vector = current_gradient.to_array()
        current_vector_scaled = self.scaler.transform([current_vector])[0]
        
        # Calculate distances to all cluster centroids
        distances = []
        for cluster in self.gradient_clusters:
            dist = np.linalg.norm(current_vector_scaled - cluster.centroid)
            distances.append(dist)
        
        # Assign to closest cluster
        current_cluster = np.argmin(distances)
        
        # Get transition probabilities
        if current_cluster < len(self.transition_matrix):
            probabilities = {}
            for i, prob in enumerate(self.transition_matrix[current_cluster]):
                probabilities[i] = prob
            return probabilities
        
        return {}