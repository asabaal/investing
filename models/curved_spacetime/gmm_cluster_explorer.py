#!/usr/bin/env python3
"""
GMM Cluster Explorer - Interactive visualization for analyzing GMM clusters across securities
Allows selection of securities, displays cluster samples, and compares cluster properties.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from market_data_database import MarketDataDatabase
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterStatistics:
    """Statistics for a single cluster."""
    cluster_id: int
    n_points: int
    percentage: float
    sentiment_mean: float
    sentiment_std: float
    sentiment_range: Tuple[float, float]
    uwr_mean: float
    uwr_std: float
    uwr_range: Tuple[float, float]
    center: Tuple[float, float]  # (sentiment, uwr)
    covariance_matrix: np.ndarray
    interpretation: str
    density_peak: Tuple[float, float]  # Location of highest density
    
@dataclass
class SecurityClusterAnalysis:
    """Complete cluster analysis for a security."""
    symbol: str
    analysis_date: datetime
    n_clusters: int
    cluster_stats: List[ClusterStatistics]
    gmm_params: Dict[str, Any]  # weights, means, covariances
    sample_indices: Dict[int, np.ndarray]  # cluster_id -> array of sample indices
    transition_matrix: np.ndarray  # Probability of transitioning between clusters
    
class GMMClusterExplorer:
    """Interactive explorer for GMM clusters across securities."""
    
    def __init__(self, cache_dir: str = "./phase_space_cache", 
                 database_path: Optional[str] = None,
                 output_dir: str = "./phase_space_analysis"):
        """Initialize the GMM Cluster Explorer."""
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize database connection
        self.database_path = database_path or self._get_default_database_path()
        self.db = MarketDataDatabase(self.database_path)
        
        # Load all available analyses
        self.available_analyses = self._load_available_analyses()
        logger.info(f"Loaded {len(self.available_analyses)} security analyses")
        
    def _get_default_database_path(self) -> str:
        """Get default database path."""
        data_dir = Path.home() / '.market_data'
        return str(data_dir / 'market_data.db')
        
    def _load_available_analyses(self) -> Dict[str, Any]:
        """Load all available phase space analyses from cache."""
        analyses = {}
        
        for pkl_file in self.cache_dir.glob("*_analysis.pkl"):
            symbol = pkl_file.stem.replace("_analysis", "")
            try:
                with open(pkl_file, 'rb') as f:
                    analysis = pickle.load(f)
                    analyses[symbol] = analysis
                    logger.info(f"Loaded analysis for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load analysis for {symbol}: {e}")
                
        return analyses
    
    def analyze_security_clusters(self, symbol: str) -> Optional[SecurityClusterAnalysis]:
        """Analyze GMM clusters for a specific security."""
        
        if symbol not in self.available_analyses:
            logger.error(f"No analysis found for {symbol}")
            return None
            
        analysis = self.available_analyses[symbol]
        
        # Extract clustering results - handle both dict and object formats
        if hasattr(analysis, 'clustering_analysis'):
            clustering_analysis = analysis.clustering_analysis
        elif isinstance(analysis, dict) and 'clustering_analysis' in analysis:
            clustering_analysis = analysis['clustering_analysis']
        else:
            logger.error(f"No clustering_analysis found in data structure for {symbol}")
            return None
            
        # Get GMM results specifically
        gmm_results = clustering_analysis.get('gmm', {})
        
        if not gmm_results:
            logger.error(f"No GMM clustering found for {symbol}")
            return None
            
        # Find the best GMM by BIC (lowest BIC is best)
        best_n_components = min(gmm_results.keys(), 
                               key=lambda n: gmm_results[n]['bic'])
        
        logger.info(f"Using GMM with {best_n_components} components (lowest BIC)")
        
        # Get the GMM results
        gmm_result = gmm_results[best_n_components]
        optimal_labels = gmm_result['labels']
        means = gmm_result['means']
        covariances = gmm_result['covariances']
        weights = gmm_result['weights']
        
        optimal_method = 'gmm'
        optimal_params = {'n_components': best_n_components}
        
        # Get market data for detailed analysis
        try:
            data = self.db.get_daily_data(symbol)
            if data is None:
                logger.error(f"No market data found for {symbol}")
                return None
                
            # Process data
            data_processed = data.copy()
            data_processed.columns = data_processed.columns.str.lower()
            column_mapping = {
                'unadjusted_close': 'close',
                'close': 'adjusted_close'
            }
            data_processed.rename(columns=column_mapping, inplace=True)
            
            # Create candle metrics
            candle_metrics = create_candle_metrics_from_ohlc(data_processed)
            
            # Extract phase space coordinates
            sentiments = np.array([c.sentiment for c in candle_metrics])
            uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
            
        except Exception as e:
            logger.error(f"Failed to load market data for {symbol}: {e}")
            return None
        
        # Use optimal labels
        labels = optimal_labels
        
        # Ensure labels and data have same length
        logger.info(f"Data lengths - labels: {len(labels)}, sentiments: {len(sentiments)}, uwrs: {len(uwrs)}")
        min_length = min(len(labels), len(sentiments), len(uwrs))
        labels = labels[:min_length]
        sentiments = sentiments[:min_length]
        uwrs = uwrs[:min_length]
        logger.info(f"Aligned to length: {min_length}")
        
        # Determine number of clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise label if present
        
        # Calculate means from data if not provided
        if means is None or (isinstance(means, list) and len(means) == 0):
            means = []
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if np.sum(mask) > 0:
                    cluster_mean = [
                        float(np.mean(sentiments[mask])),
                        float(np.mean(uwrs[mask]))
                    ]
                    means.append(cluster_mean)
                else:
                    means.append([0.0, 0.0])
        
        cluster_stats = []
        sample_indices = {}
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
            mask = labels == cluster_id
            cluster_sentiments = sentiments[mask]
            cluster_uwrs = uwrs[mask]
            
            if len(cluster_sentiments) == 0:
                continue
                
            # Calculate statistics
            stats = ClusterStatistics(
                cluster_id=cluster_id,
                n_points=int(np.sum(mask)),
                percentage=float(np.mean(mask) * 100),
                sentiment_mean=float(np.mean(cluster_sentiments)),
                sentiment_std=float(np.std(cluster_sentiments)),
                sentiment_range=(float(np.min(cluster_sentiments)), 
                               float(np.max(cluster_sentiments))),
                uwr_mean=float(np.mean(cluster_uwrs)),
                uwr_std=float(np.std(cluster_uwrs)),
                uwr_range=(float(np.min(cluster_uwrs)), 
                          float(np.max(cluster_uwrs))),
                center=(float(means[cluster_id][0]) if cluster_id < len(means) else float(np.mean(cluster_sentiments)), 
                       float(means[cluster_id][1]) if cluster_id < len(means) else float(np.mean(cluster_uwrs))),
                covariance_matrix=covariances[cluster_id] if covariances is not None and cluster_id < len(covariances) else np.eye(2),
                interpretation=self._interpret_cluster(
                    float(np.mean(cluster_sentiments)), 
                    float(np.mean(cluster_uwrs))
                ),
                density_peak=(float(means[cluster_id][0]) if cluster_id < len(means) else float(np.mean(cluster_sentiments)), 
                             float(means[cluster_id][1]) if cluster_id < len(means) else float(np.mean(cluster_uwrs)))
            )
            
            cluster_stats.append(stats)
            
            # Sample indices for visualization (max 50 samples per cluster)
            indices = np.where(mask)[0]
            if len(indices) > 50:
                sample_indices[cluster_id] = np.random.choice(indices, 50, replace=False)
            else:
                sample_indices[cluster_id] = indices
                
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(labels)
        
        # Get analysis date
        if hasattr(analysis, 'analysis_date'):
            analysis_date = analysis.analysis_date
        elif isinstance(analysis, dict) and 'analysis_date' in analysis:
            analysis_date = analysis['analysis_date']
        else:
            analysis_date = datetime.now()
            
        # Store method-specific params
        method_params = {
            'method': optimal_method,
            'params': optimal_params,
            'weights': weights if weights is not None else [1.0/n_clusters] * n_clusters,
            'means': means,
            'covariances': covariances
        }
        
        return SecurityClusterAnalysis(
            symbol=symbol,
            analysis_date=analysis_date,
            n_clusters=n_clusters,
            cluster_stats=cluster_stats,
            gmm_params=method_params,
            sample_indices=sample_indices,
            transition_matrix=transition_matrix
        )
    
    def _interpret_cluster(self, sentiment_mean: float, uwr_mean: float) -> str:
        """Interpret what a cluster represents in market terms."""
        if sentiment_mean > 0.3 and uwr_mean < 0.3:
            return 'Strong Bullish Trend'
        elif sentiment_mean > 0.1 and uwr_mean < 0.4:
            return 'Moderate Bullish'
        elif sentiment_mean < -0.3 and uwr_mean > 0.4:
            return 'Strong Bearish Pressure'
        elif sentiment_mean < -0.1 and uwr_mean > 0.3:
            return 'Moderate Bearish'
        elif abs(sentiment_mean) < 0.1 and uwr_mean > 0.5:
            return 'High Volatility'
        elif abs(sentiment_mean) < 0.2 and uwr_mean < 0.3:
            return 'Consolidation'
        elif uwr_mean > 0.6:
            return 'Extreme Volatility'
        else:
            return 'Mixed Regime'
    
    def _calculate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Calculate probability of transitioning between clusters."""
        n_clusters = len(np.unique(labels))
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        for i in range(len(labels) - 1):
            from_cluster = labels[i]
            to_cluster = labels[i + 1]
            transition_matrix[from_cluster, to_cluster] += 1
            
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def create_cluster_timeline_visualization(self, symbol: str) -> Optional[go.Figure]:
        """Create timeline visualization with candles colored by cluster identity."""
        
        analysis = self.analyze_security_clusters(symbol)
        if not analysis:
            return None
            
        # Get market data
        data = self.db.get_daily_data(symbol)
        if data is None:
            return None
            
        # Process data - align with cluster analysis length
        data_processed = data.copy()
        data_processed.columns = data_processed.columns.str.lower()
        column_mapping = {
            'unadjusted_close': 'close',
            'close': 'adjusted_close'
        }
        data_processed.rename(columns=column_mapping, inplace=True)
        
        # Get cluster labels from analysis (reconstruct from GMM)
        analysis_raw = self.available_analyses[symbol]
        clustering_analysis = analysis_raw.clustering_analysis if hasattr(analysis_raw, 'clustering_analysis') else analysis_raw['clustering_analysis']
        gmm_results = clustering_analysis.get('gmm', {})
        best_n_components = min(gmm_results.keys(), key=lambda n: gmm_results[n]['bic'])
        cluster_labels = gmm_results[best_n_components]['labels']
        
        # Align data and labels
        min_length = min(len(cluster_labels), len(data_processed))
        data_processed = data_processed.iloc[:min_length]
        cluster_labels = cluster_labels[:min_length]
        
        # Define cluster colors
        cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F8C471']
        
        # Create figure
        fig = go.Figure()
        
        # Add candles for each cluster separately to get different colors
        for cluster_id in range(analysis.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = data_processed[mask]
            
            if len(cluster_data) == 0:
                continue
                
            # Get cluster info
            cluster_stats = next((s for s in analysis.cluster_stats if s.cluster_id == cluster_id), None)
            cluster_name = f"Cluster {cluster_id}: {cluster_stats.interpretation}" if cluster_stats else f"Cluster {cluster_id}"
            cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Determine outline colors (green for up, red for down days)
            up_mask = cluster_data['close'] >= cluster_data['open']
            down_mask = cluster_data['close'] < cluster_data['open']
            
            # Add up candles
            if up_mask.any():
                up_data = cluster_data[up_mask]
                fig.add_trace(go.Candlestick(
                    x=up_data.index,
                    open=up_data['open'],
                    high=up_data['high'],
                    low=up_data['low'],
                    close=up_data['close'],
                    name=f'{cluster_name} (Up)',
                    increasing_fillcolor=cluster_color,
                    increasing_line_color='green',
                    increasing_line_width=2,
                    decreasing_fillcolor=cluster_color,  # Won't be used but set for consistency
                    decreasing_line_color='green',
                    showlegend=True
                ))
            
            # Add down candles
            if down_mask.any():
                down_data = cluster_data[down_mask]
                fig.add_trace(go.Candlestick(
                    x=down_data.index,
                    open=down_data['open'],
                    high=down_data['high'],
                    low=down_data['low'],
                    close=down_data['close'],
                    name=f'{cluster_name} (Down)',
                    increasing_fillcolor=cluster_color,  # Won't be used but set for consistency
                    increasing_line_color='red',
                    decreasing_fillcolor=cluster_color,
                    decreasing_line_color='red',
                    decreasing_line_width=2,
                    showlegend=True
                ))
        
        # Get clustering method info
        method_info = analysis.gmm_params.get('method', 'Unknown').upper()
        method_params = analysis.gmm_params.get('params', {})
        param_str = ', '.join([f"{k}={v}" for k, v in method_params.items()])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"📈 {symbol} - Timeline Colored by Cluster ({method_info})<br>" +
                     f"<sub>Candles filled by cluster identity, outlined by price direction ({param_str})</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,
            width=1400,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            ),
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_cluster_properties_comparison(self, symbols: Optional[List[str]] = None) -> Optional[go.Figure]:
        """Create visualization comparing cluster properties across multiple securities."""
        
        if symbols is None:
            symbols = list(self.available_analyses.keys())[:10]  # Top 10 if not specified
            
        # Collect cluster data for all securities
        all_cluster_data = []
        
        for symbol in symbols:
            analysis = self.analyze_security_clusters(symbol)
            if not analysis:
                continue
                
            for stats in analysis.cluster_stats:
                all_cluster_data.append({
                    'symbol': symbol,
                    'cluster_id': stats.cluster_id,
                    'interpretation': stats.interpretation,
                    'percentage': stats.percentage,
                    'sentiment_mean': stats.sentiment_mean,
                    'sentiment_std': stats.sentiment_std,
                    'uwr_mean': stats.uwr_mean,
                    'uwr_std': stats.uwr_std,
                    'n_points': stats.n_points
                })
        
        if not all_cluster_data:
            return None
            
        df = pd.DataFrame(all_cluster_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cluster Size Distribution by Security',
                'Mean Sentiment vs Mean UWR by Interpretation',
                'Cluster Volatility (Std Dev) Comparison',
                'Market Regime Distribution'
            ),
            specs=[
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Box plot of cluster sizes by security
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]
            fig.add_trace(
                go.Box(
                    y=symbol_data['percentage'],
                    name=symbol,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=1, col=1
            )
        
        # 2. Scatter plot of mean sentiment vs mean UWR
        interpretation_colors = {
            'Strong Bullish Trend': 'green',
            'Moderate Bullish': 'lightgreen',
            'Strong Bearish Pressure': 'red',
            'Moderate Bearish': 'lightcoral',
            'High Volatility': 'orange',
            'Extreme Volatility': 'darkred',
            'Consolidation': 'blue',
            'Mixed Regime': 'gray'
        }
        
        for interpretation in df['interpretation'].unique():
            mask = df['interpretation'] == interpretation
            fig.add_trace(
                go.Scatter(
                    x=df[mask]['sentiment_mean'],
                    y=df[mask]['uwr_mean'],
                    mode='markers',
                    marker=dict(
                        size=df[mask]['percentage'],
                        color=interpretation_colors.get(interpretation, 'gray'),
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    name=interpretation,
                    text=df[mask]['symbol'],
                    hovertemplate='%{text}<br>Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Size: %{marker.size:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Volatility comparison
        fig.add_trace(
            go.Scatter(
                x=df['sentiment_std'],
                y=df['uwr_std'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['n_points'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Points", x=1.15)
                ),
                text=[f"{row['symbol']}: {row['interpretation']}" for _, row in df.iterrows()],
                hovertemplate='%{text}<br>Sentiment Std: %{x:.3f}<br>UWR Std: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Market regime distribution
        regime_counts = df.groupby('interpretation')['symbol'].count().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=regime_counts.values,
                y=regime_counts.index,
                orientation='h',
                marker=dict(
                    color=[interpretation_colors.get(interp, 'gray') for interp in regime_counts.index]
                ),
                hovertemplate='%{y}: %{x} clusters<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Security", row=1, col=1)
        fig.update_yaxes(title_text="Cluster Size (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Mean Sentiment", range=[-1, 1], row=1, col=2)
        fig.update_yaxes(title_text="Mean UWR", range=[0, 1], row=1, col=2)
        
        fig.update_xaxes(title_text="Sentiment Std Dev", row=2, col=1)
        fig.update_yaxes(title_text="UWR Std Dev", row=2, col=1)
        
        fig.update_xaxes(title_text="Number of Clusters", row=2, col=2)
        fig.update_yaxes(title_text="Market Regime", row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text="📊 Cross-Security GMM Cluster Analysis<br>" +
                     f"<sub>Comparing cluster properties across {len(symbols)} securities</sub>",
                x=0.5,
                xanchor='center'
            ),
            height=800,
            width=1400,
            template="plotly_dark",
            showlegend=True,
            legend=dict(x=1.02, y=0.5)
        )
        
        return fig
    
    
    def create_interactive_dashboard(self) -> None:
        """Create an interactive Dash dashboard for exploring GMM clusters."""
        
        try:
            import dash
            from dash import dcc, html, Input, Output, State
            import dash_bootstrap_components as dbc
        except ImportError:
            logger.error("Dash not installed. Run: pip install dash dash-bootstrap-components")
            return
            
        # Initialize Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        # Get available symbols
        symbols = sorted(list(self.available_analyses.keys()))
        
        # Layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔬 GMM Cluster Explorer", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select Security:"),
                    dcc.Dropdown(
                        id='security-dropdown',
                        options=[{'label': s, 'value': s} for s in symbols],
                        value=symbols[0] if symbols else None,
                        style={'color': 'black'}
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Samples per Cluster:"),
                    dcc.Slider(
                        id='samples-slider',
                        min=5,
                        max=50,
                        step=5,
                        value=20,
                        marks={i: str(i) for i in range(5, 51, 10)}
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Comparison Securities:"),
                    dcc.Dropdown(
                        id='comparison-dropdown',
                        options=[{'label': s, 'value': s} for s in symbols],
                        value=symbols[:5] if len(symbols) >= 5 else symbols,
                        multi=True,
                        style={'color': 'black'}
                    )
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-1",
                        children=[dcc.Graph(id='cluster-samples-plot')],
                        type="default"
                    )
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-2",
                        children=[dcc.Graph(id='transition-heatmap')],
                        type="default"
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Loading(
                        id="loading-3",
                        children=[dcc.Graph(id='cluster-comparison-plot')],
                        type="default"
                    )
                ], width=6)
            ])
        ], fluid=True)
        
        # Callbacks
        @app.callback(
            Output('cluster-samples-plot', 'figure'),
            [Input('security-dropdown', 'value'),
             Input('samples-slider', 'value')]
        )
        def update_samples_plot(symbol, n_samples):
            if not symbol:
                return go.Figure()
            fig = self.create_cluster_sample_visualization(symbol, n_samples)
            return fig if fig else go.Figure()
        
        @app.callback(
            Output('transition-heatmap', 'figure'),
            [Input('security-dropdown', 'value')]
        )
        def update_transition_heatmap(symbol):
            if not symbol:
                return go.Figure()
            fig = self.create_cluster_transition_heatmap(symbol)
            return fig if fig else go.Figure()
        
        @app.callback(
            Output('cluster-comparison-plot', 'figure'),
            [Input('comparison-dropdown', 'value')]
        )
        def update_comparison_plot(symbols):
            if not symbols:
                return go.Figure()
            fig = self.create_cluster_properties_comparison(symbols)
            return fig if fig else go.Figure()
        
        # Run server
        logger.info("Starting Dash server on http://127.0.0.1:8050")
        app.run_server(debug=True, port=8050)
    
    def export_cluster_analysis(self, output_file: str = "gmm_cluster_analysis_report.json") -> None:
        """Export comprehensive cluster analysis for all securities."""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'n_securities': len(self.available_analyses),
            'securities': {}
        }
        
        for symbol in self.available_analyses:
            analysis = self.analyze_security_clusters(symbol)
            if not analysis:
                continue
                
            report['securities'][symbol] = {
                'analysis_date': analysis.analysis_date.isoformat(),
                'n_clusters': analysis.n_clusters,
                'clusters': []
            }
            
            for stats in analysis.cluster_stats:
                cluster_info = {
                    'cluster_id': int(stats.cluster_id),
                    'interpretation': stats.interpretation,
                    'percentage': float(stats.percentage),
                    'n_points': int(stats.n_points),
                    'sentiment': {
                        'mean': stats.sentiment_mean,
                        'std': stats.sentiment_std,
                        'range': stats.sentiment_range
                    },
                    'uwr': {
                        'mean': stats.uwr_mean,
                        'std': stats.uwr_std,
                        'range': stats.uwr_range
                    },
                    'center': stats.center
                }
                report['securities'][symbol]['clusters'].append(cluster_info)
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Exported cluster analysis to {output_path}")


def main():
    """Main function to demonstrate GMM Cluster Explorer functionality."""
    
    explorer = GMMClusterExplorer()
    
    # Generate individual security visualization
    symbols = list(explorer.available_analyses.keys())
    if symbols:
        # Create sample visualization for first security
        symbol = symbols[0]
        logger.info(f"Creating cluster sample visualization for {symbol}...")
        
        # Create symbol directory if it doesn't exist
        symbol_dir = explorer.output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        fig = explorer.create_cluster_sample_visualization(symbol)
        if fig:
            output_file = symbol_dir / f"{symbol}_gmm_cluster_samples.html"
            fig.write_html(str(output_file))
            logger.info(f"Saved to {output_file}")
        
        # Create transition heatmap
        fig = explorer.create_cluster_transition_heatmap(symbol)
        if fig:
            output_file = symbol_dir / f"{symbol}_gmm_transition_matrix.html"
            fig.write_html(str(output_file))
            logger.info(f"Saved to {output_file}")
    
    # Create cross-security comparison
    logger.info("Creating cross-security cluster comparison...")
    fig = explorer.create_cluster_properties_comparison(symbols[:10])
    if fig:
        output_file = explorer.output_dir / "gmm_cross_security_comparison.html"
        fig.write_html(str(output_file))
        logger.info(f"Saved to {output_file}")
    
    # Export comprehensive report
    explorer.export_cluster_analysis()
    
    # Launch interactive dashboard
    logger.info("\nTo launch the interactive dashboard, run:")
    logger.info("python gmm_cluster_explorer.py --dashboard")
    
    # Check if dashboard flag is set
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--dashboard':
        explorer.create_interactive_dashboard()


if __name__ == "__main__":
    main()