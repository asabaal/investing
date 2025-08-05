"""
Market Intervention Detection Visualization

Creates comprehensive visualizations for intervention detection including:
- Timeline of detected interventions
- Energy anomaly analysis
- Volume-energy correlation plots
- Pattern space trajectory analysis
- Intervention type classification
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from intervention_detector import *
from market_hamiltonian import analyze_market_energy
from curved_candle_geometry import create_candle_metrics_from_ohlc


def create_intervention_timeline(ohlc_data: pd.DataFrame, 
                               interventions: List[InterventionSignal]) -> go.Figure:
    """
    Create a timeline showing price, energy, and detected interventions.
    """
    # Get energy analysis
    analysis = analyze_market_energy(ohlc_data)
    energies = analysis['energies']
    
    dates = ohlc_data.index
    prices = ohlc_data['close']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Price & Interventions', 'Market Energy', 'Intervention Signals'),
        vertical_spacing=0.05
    )
    
    # 1. Price chart with intervention markers
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#c9d1d9', width=2)
        ),
        row=1, col=1
    )
    
    # Add intervention markers on price chart
    intervention_colors = {
        InterventionType.BULLISH_PUMP: '#00ff00',
        InterventionType.BEARISH_DUMP: '#ff0000',
        InterventionType.VOLUME_SURGE: '#ffff00',
        InterventionType.ENERGY_INJECTION: '#ff00ff',
        InterventionType.PATTERN_ANOMALY: '#00ffff',
        InterventionType.REVERSAL_FORCE: '#ffa500',
        InterventionType.VOLATILITY_SPIKE: '#ff69b4'
    }
    
    for intervention_type in InterventionType:
        type_interventions = [i for i in interventions if i.intervention_type == intervention_type]
        
        if type_interventions:
            x_coords = [dates.iloc[i.timestamp] if hasattr(dates, 'iloc') else dates[i.timestamp] 
                       for i in type_interventions]
            y_coords = [prices.iloc[i.timestamp] if hasattr(prices, 'iloc') else prices[i.timestamp] 
                       for i in type_interventions]
            sizes = [10 + 20 * i.magnitude for i in type_interventions]
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    name=intervention_type.value.replace('_', ' ').title(),
                    marker=dict(
                        color=intervention_colors[intervention_type],
                        size=sizes,
                        symbol='triangle-up' if 'bullish' in intervention_type.value else 
                               'triangle-down' if 'bearish' in intervention_type.value else 'circle',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Price: %{y:.2f}<br>' +
                                  'Magnitude: %{customdata[0]:.2f}<br>' +
                                  'Confidence: %{customdata[1]:.2f}<extra></extra>',
                    text=[i.description for i in type_interventions],
                    customdata=[[i.magnitude, i.confidence] for i in type_interventions]
                ),
                row=1, col=1
            )
    
    # 2. Energy chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=energies,
            mode='lines',
            name='Market Energy',
            line=dict(color='#e0aaff', width=2),
            fill='tozeroy',
            fillcolor='rgba(224, 170, 255, 0.2)'
        ),
        row=2, col=1
    )
    
    # Mark energy interventions
    energy_interventions = [i for i in interventions 
                           if i.intervention_type in [InterventionType.ENERGY_INJECTION, 
                                                     InterventionType.REVERSAL_FORCE]]
    
    if energy_interventions:
        x_coords = [dates.iloc[i.timestamp] if hasattr(dates, 'iloc') else dates[i.timestamp] 
                   for i in energy_interventions]
        y_coords = [energies[i.timestamp] for i in energy_interventions]
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                name='Energy Anomalies',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 3. Intervention signals over time
    # Create a signal strength time series
    signal_strength = np.zeros(len(dates))
    
    for intervention in interventions:
        start_idx = intervention.timestamp
        end_idx = min(len(signal_strength), start_idx + intervention.duration)
        
        for idx in range(start_idx, end_idx):
            signal_strength[idx] = max(signal_strength[idx], 
                                     intervention.magnitude * intervention.confidence)
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=signal_strength,
            mode='lines',
            name='Intervention Signal',
            line=dict(color='#f72585', width=2),
            fill='tozeroy',
            fillcolor='rgba(247, 37, 133, 0.3)'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Market Intervention Detection Timeline',
        template='plotly_dark',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Energy', row=2, col=1)
    fig.update_yaxes(title_text='Signal Strength', row=3, col=1, range=[0, 1])
    fig.update_xaxes(title_text='Time', row=3, col=1)
    
    return fig


def create_energy_anomaly_analysis(ohlc_data: pd.DataFrame,
                                 interventions: List[InterventionSignal]) -> go.Figure:
    """
    Create detailed analysis of energy anomalies.
    """
    # Get energy analysis
    analysis = analyze_market_energy(ohlc_data)
    energies = np.array(analysis['energies'])
    gradients = analysis['gradients']
    
    # Compute energy changes and gradient magnitudes
    energy_changes = np.diff(energies)
    gradient_mags = np.array([np.linalg.norm(g) if g is not None else 0 for g in gradients])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Energy Changes vs Gradient Magnitude',
            'Energy Distribution',
            'Intervention Magnitude vs Confidence',
            'Energy Anomaly Scatter'
        )
    )
    
    # 1. Energy changes vs gradient magnitude
    fig.add_trace(
        go.Scatter(
            x=gradient_mags[1:],  # Align with energy changes
            y=energy_changes,
            mode='markers',
            marker=dict(color='#c9d1d9', size=6, opacity=0.7),
            name='Normal Points',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Overlay intervention points
    energy_interventions = [i for i in interventions 
                           if i.intervention_type in [InterventionType.ENERGY_INJECTION, 
                                                     InterventionType.REVERSAL_FORCE]]
    
    if energy_interventions:
        intervention_gradients = [gradient_mags[i.timestamp] if i.timestamp < len(gradient_mags) else 0 
                                for i in energy_interventions]
        intervention_changes = [i.energy_delta for i in energy_interventions]
        
        fig.add_trace(
            go.Scatter(
                x=intervention_gradients,
                y=intervention_changes,
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name='Energy Interventions',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Energy distribution
    fig.add_trace(
        go.Histogram(
            x=energies,
            nbinsx=30,
            name='Energy Distribution',
            marker_color='#e0aaff',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Mark intervention energies
    if energy_interventions:
        intervention_energies = [energies[i.timestamp] for i in energy_interventions]
        fig.add_trace(
            go.Scatter(
                x=intervention_energies,
                y=[0] * len(intervention_energies),
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-up'),
                name='Intervention Energies',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Intervention magnitude vs confidence
    if interventions:
        magnitudes = [i.magnitude for i in interventions]
        confidences = [i.confidence for i in interventions]
        types = [i.intervention_type.value for i in interventions]
        
        fig.add_trace(
            go.Scatter(
                x=confidences,
                y=magnitudes,
                mode='markers',
                marker=dict(
                    size=10,
                    color=[hash(t) % 360 for t in types],
                    colorscale='HSV',
                    showscale=False,
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=types,
                name='Interventions',
                showlegend=False,
                hovertemplate='Type: %{text}<br>Confidence: %{x:.2f}<br>Magnitude: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Energy anomaly scatter (energy vs time)
    x_time = list(range(len(energies)))
    
    fig.add_trace(
        go.Scatter(
            x=x_time,
            y=energies,
            mode='markers',
            marker=dict(color='#c9d1d9', size=4, opacity=0.6),
            name='Energy Evolution',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Overlay interventions
    if interventions:
        intervention_times = [i.timestamp for i in interventions]
        intervention_energies = [energies[i.timestamp] for i in interventions]
        
        fig.add_trace(
            go.Scatter(
                x=intervention_times,
                y=intervention_energies,
                mode='markers',
                marker=dict(color='red', size=10, symbol='diamond'),
                name='Interventions',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Energy Anomaly Analysis',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Gradient Magnitude', row=1, col=1)
    fig.update_yaxes(title_text='Energy Change', row=1, col=1)
    
    fig.update_xaxes(title_text='Energy', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    
    fig.update_xaxes(title_text='Confidence', row=2, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=1)
    
    fig.update_xaxes(title_text='Time', row=2, col=2)
    fig.update_yaxes(title_text='Energy', row=2, col=2)
    
    return fig


def create_intervention_summary_dashboard(ohlc_data: pd.DataFrame,
                                        interventions: List[InterventionSignal]) -> go.Figure:
    """
    Create a summary dashboard of intervention statistics.
    """
    # Analyze intervention patterns
    stats = analyze_intervention_patterns(interventions)
    
    if stats['total_interventions'] == 0:
        # Create empty dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="No interventions detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(
            title='Intervention Summary Dashboard',
            template='plotly_dark',
            height=600
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Intervention Types',
            'Magnitude Distribution',
            'Confidence Distribution',
            'Duration Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )
    
    # 1. Intervention types
    type_dist = stats['type_distribution']
    types = list(type_dist.keys())
    counts = list(type_dist.values())
    
    fig.add_trace(
        go.Bar(
            x=types,
            y=counts,
            marker_color='#e0aaff',
            name='Intervention Types',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Magnitude distribution
    if interventions:
        magnitudes = [i.magnitude for i in interventions]
        
        fig.add_trace(
            go.Histogram(
                x=magnitudes,
                nbinsx=20,
                marker_color='#f72585',
                name='Magnitude',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Confidence distribution
        confidences = [i.confidence for i in interventions]
        
        fig.add_trace(
            go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color='#3a86ff',
                name='Confidence',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Duration distribution
        durations = [i.duration for i in interventions]
        
        fig.add_trace(
            go.Histogram(
                x=durations,
                nbinsx=10,
                marker_color='#06ffa5',
                name='Duration',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'Intervention Summary Dashboard - {stats["total_interventions"]} Total Interventions',
        template='plotly_dark',
        height=700,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Type', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    fig.update_xaxes(title_text='Magnitude', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    
    fig.update_xaxes(title_text='Confidence', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    
    fig.update_xaxes(title_text='Duration (periods)', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)
    
    return fig


def main():
    """Test the intervention detection and visualization system."""
    
    # Create test data with artificial interventions
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(50)]
    
    # Generate base OHLC data
    np.random.seed(42)
    price = 100
    prices = []
    volumes = []
    
    for i in range(50):
        # Natural evolution
        price_change = np.random.normal(0, 1)
        
        # Add artificial interventions
        if i == 10:  # Bullish intervention
            price_change += 5
        elif i == 25:  # Bearish intervention
            price_change -= 4
        elif i == 35:  # Volume spike
            price_change += np.random.normal(0, 0.5)
        
        price += price_change
        
        # Generate OHLC
        volatility = np.random.uniform(0.5, 2.0)
        open_price = price
        high = price + np.random.exponential(volatility)
        low = price - np.random.exponential(volatility)
        close = price + price_change
        
        prices.append([open_price, high, low, close])
        
        # Volume with spikes at interventions
        base_volume = np.random.uniform(1000, 5000)
        if i in [10, 25, 35]:
            base_volume *= np.random.uniform(3, 8)  # Volume spike
        
        volumes.append(base_volume)
    
    # Create DataFrame
    price_data = np.array(prices)
    ohlc_data = pd.DataFrame({
        'open': price_data[:, 0],
        'high': price_data[:, 1],
        'low': price_data[:, 2],
        'close': price_data[:, 3],
        'volume': volumes
    }, index=dates)
    
    print("🔍 Running intervention detection...")
    
    # Create detector and analyze
    detector = MarketInterventionDetector(
        energy_threshold=1.5,
        gradient_threshold=1.2,
        volume_threshold=2.0,
        confidence_min=0.5
    )
    
    interventions = detector.detect_all_interventions(ohlc_data)
    
    print(f"✅ Detected {len(interventions)} interventions")
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    timeline = create_intervention_timeline(ohlc_data, interventions)
    timeline.write_html("intervention_timeline.html")
    
    anomaly_analysis = create_energy_anomaly_analysis(ohlc_data, interventions)
    anomaly_analysis.write_html("intervention_anomaly_analysis.html")
    
    summary_dashboard = create_intervention_summary_dashboard(ohlc_data, interventions)
    summary_dashboard.write_html("intervention_summary_dashboard.html")
    
    print("\n✅ Intervention detection visualizations created:")
    print("   - intervention_timeline.html")
    print("   - intervention_anomaly_analysis.html")
    print("   - intervention_summary_dashboard.html")
    
    # Print intervention summary
    if interventions:
        print(f"\n📈 Intervention Summary:")
        for i, intervention in enumerate(interventions):
            print(f"   {i+1}. {intervention.description}")
            print(f"      Type: {intervention.intervention_type.value}")
            print(f"      Time: {intervention.timestamp}")
            print(f"      Magnitude: {intervention.magnitude:.2f}")
            print(f"      Confidence: {intervention.confidence:.2f}")
            print()


if __name__ == "__main__":
    main()