"""
Visualization module for the Trend Detection Algorithm
Separated from business logic with clear responsibilities

Uses existing Candle infrastructure from supply/demand refactoring
"""

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - visualization features disabled")
    print("   Install with: pip install plotly")

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from trend_core_models import (
    SwingPoint, SwingType, Trend, TrendDirection, TrendSignificance,
    TrendAnalysisResult, VisualizationData
)
from trend_utilities import TrendDataProcessor

# Use existing Candle from supply/demand refactoring
from utilities import Candle


class TrendColorScheme:
    """
    Responsibility: Define consistent color schemes for trend visualization.
    """
    
    # Trend direction colors
    UPTREND = '#51cf66'      # Green
    DOWNTREND = '#ff6b6b'    # Red  
    SIDEWAYS = '#ffa500'     # Orange
    
    # Significance colors
    MAJOR_ALPHA = 0.3
    MINOR_ALPHA = 0.2
    CONSOLIDATION_ALPHA = 0.1
    
    # Swing colors
    SWING_HIGH = '#ffd43b'   # Yellow
    SWING_LOW = '#4dabf7'    # Blue
    
    # Candle colors
    BULLISH_CANDLE = '#00ff88'
    BEARISH_CANDLE = '#ff4444'
    BULLISH_FILL = 'rgba(0, 255, 136, 0.8)'
    BEARISH_FILL = 'rgba(255, 68, 68, 0.8)'
    
    # Background colors
    CHART_BACKGROUND = '#2d2d2d'
    PAPER_BACKGROUND = '#1e1e1e'
    GRID_COLOR = '#444444'
    LINE_COLOR = '#666666'
    TEXT_COLOR = '#ffffff'
    
    @classmethod
    def get_trend_color(cls, direction: TrendDirection) -> str:
        """Get color for trend direction"""
        color_map = {
            TrendDirection.UP: cls.UPTREND,
            TrendDirection.DOWN: cls.DOWNTREND,
            TrendDirection.SIDEWAYS: cls.SIDEWAYS
        }
        return color_map.get(direction, cls.UPTREND)
    
    @classmethod
    def get_significance_alpha(cls, significance: TrendSignificance) -> float:
        """Get alpha value for trend significance"""
        alpha_map = {
            TrendSignificance.MAJOR: cls.MAJOR_ALPHA,
            TrendSignificance.MINOR: cls.MINOR_ALPHA,
            TrendSignificance.CONSOLIDATION: cls.CONSOLIDATION_ALPHA
        }
        return alpha_map.get(significance, cls.MINOR_ALPHA)


class CandlestickRenderer:
    """
    Responsibility: Render candlestick charts with proper formatting.
    """
    
    @staticmethod
    def create_candlestick_trace(candles: List[Candle], name: str = "Price Action") -> 'go.Candlestick':
        """Create candlestick trace from candle data"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization. Install with: pip install plotly")
            
        indices = list(range(len(candles)))
        
        return go.Candlestick(
            x=indices,
            open=[c.open for c in candles],
            high=[c.high for c in candles],
            low=[c.low for c in candles],
            close=[c.close for c in candles],
            name=name,
            increasing_line_color=TrendColorScheme.BULLISH_CANDLE,
            decreasing_line_color=TrendColorScheme.BEARISH_CANDLE,
            increasing_fillcolor=TrendColorScheme.BULLISH_FILL,
            decreasing_fillcolor=TrendColorScheme.BEARISH_FILL
        )
    
    @staticmethod
    def add_candle_annotations(fig: go.Figure, candles: List[Candle], 
                              classifications: Optional[List] = None, row: int = 1) -> None:
        """Add candle index and classification annotations"""
        if not classifications:
            return
        
        for i, (candle, classification) in enumerate(zip(candles, classifications)):
            ratio = candle.body_to_wick_ratio if candle.body_to_wick_ratio != float('inf') else 99
            color = '#00aaff' if classification.value == 'leg' else '#ffaa00'
            
            fig.add_annotation(
                x=i, y=candle.low - (candle.high - candle.low) * 0.05,
                text=f"<b>{i}</b><br>{ratio:.1f}",
                showarrow=False,
                font=dict(color=color, size=8),
                bgcolor=f"rgba({0 if classification.value == 'leg' else 255}, 170, {255 if classification.value == 'leg' else 0}, 0.3)",
                row=row, col=1
            )


class SwingPointRenderer:
    """
    Responsibility: Render swing points on charts.
    """
    
    @staticmethod
    def create_swing_traces(swings: List[SwingPoint]) -> List[go.Scatter]:
        """Create swing point traces"""
        traces = []
        
        for swing in swings:
            color = TrendColorScheme.SWING_HIGH if swing.swing_type == SwingType.HIGH else TrendColorScheme.SWING_LOW
            symbol = 'triangle-up' if swing.swing_type == SwingType.HIGH else 'triangle-down'
            text_pos = "top center" if swing.swing_type == SwingType.HIGH else "bottom center"
            
            trace = go.Scatter(
                x=[swing.candle_index],
                y=[swing.price],
                mode='markers+text',
                marker=dict(
                    symbol=symbol, 
                    size=16, 
                    color=color, 
                    line=dict(color='white', width=2)
                ),
                text=f"<b>{swing.swing_type.value[0].upper()}{swing.candle_index}</b>",
                textposition=text_pos,
                name=f"Swing {swing.swing_type.value.title()}",
                showlegend=False
            )
            traces.append(trace)
        
        return traces
    
    @staticmethod
    def add_swing_annotations(fig: go.Figure, swings: List[SwingPoint], row: int = 1) -> None:
        """Add detailed swing annotations"""
        for swing in swings:
            fig.add_annotation(
                x=swing.candle_index,
                y=swing.price,
                text=f"{swing.swing_type.value.upper()}<br>{swing.price:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=TrendColorScheme.SWING_HIGH if swing.swing_type == SwingType.HIGH else TrendColorScheme.SWING_LOW,
                font=dict(size=10, color=TrendColorScheme.TEXT_COLOR),
                row=row, col=1
            )


class TrendRenderer:
    """
    Responsibility: Render trend lines, regions, and annotations.
    """
    
    @staticmethod
    def create_trend_region_shapes(trends: List[Trend], candles: List['Candle']) -> List[Dict]:
        """Create trend region background shapes"""
        shapes = []
        
        for trend in trends:
            if not trend.is_active and trend.end_index is None:
                continue
                
            color = TrendColorScheme.get_trend_color(trend.direction)
            alpha = TrendColorScheme.get_significance_alpha(trend.significance)
            
            # Calculate trend boundaries
            start_idx = trend.start_index
            end_idx = trend.end_index if trend.end_index is not None else len(candles) - 1
            
            if start_idx < len(candles) and end_idx < len(candles):
                trend_candles = candles[start_idx:end_idx + 1]
                if trend_candles:
                    trend_high = max(c.high for c in trend_candles)
                    trend_low = min(c.low for c in trend_candles)
                    
                    shape = dict(
                        type="rect",
                        x0=start_idx - 0.5,
                        x1=end_idx + 0.5,
                        y0=trend_low,
                        y1=trend_high,
                        fillcolor=f"rgba{TrendRenderer._hex_to_rgb(color) + (alpha,)}",
                        line=dict(width=0)
                    )
                    shapes.append(shape)
        
        return shapes
    
    @staticmethod
    def create_trend_formation_traces(trends: List[Trend]) -> List[go.Scatter]:
        """Create trend formation line traces"""
        traces = []
        
        for trend in trends:
            color = TrendColorScheme.get_trend_color(trend.direction)
            
            # Draw formation pattern
            formation_swings = trend.formation_pattern.formation_swings
            if len(formation_swings) >= 2:
                formation_x = [s.candle_index for s in formation_swings]
                formation_y = [s.price for s in formation_swings]
                
                trace = go.Scatter(
                    x=formation_x,
                    y=formation_y,
                    mode='lines+markers',
                    line=dict(color=color, width=4),
                    marker=dict(size=8, color=color),
                    name=f"{trend.direction.value.title()} Trend {trend.trend_id}",
                    showlegend=False
                )
                traces.append(trace)
        
        return traces
    
    @staticmethod
    def add_trend_annotations(fig: go.Figure, trends: List[Trend], candles: List['Candle'], row: int = 1) -> None:
        """Add trend labels and controlling level annotations"""
        for trend in trends:
            color = TrendColorScheme.get_trend_color(trend.direction)
            
            # Calculate trend midpoint for label
            start_idx = trend.start_index
            end_idx = trend.end_index if trend.end_index is not None else len(candles) - 1
            
            if start_idx < len(candles) and end_idx < len(candles):
                trend_candles = candles[start_idx:end_idx + 1]
                if trend_candles:
                    mid_x = (start_idx + end_idx) / 2
                    trend_high = max(c.high for c in trend_candles)
                    trend_low = min(c.low for c in trend_candles)
                    mid_y = (trend_high + trend_low) / 2
                    
                    # Direction arrow
                    arrow_map = {
                        TrendDirection.UP: "↗",
                        TrendDirection.DOWN: "↘", 
                        TrendDirection.SIDEWAYS: "↔"
                    }
                    arrow = arrow_map.get(trend.direction, "")
                    
                    # Trend label
                    fig.add_annotation(
                        x=mid_x, y=mid_y,
                        text=f"<b>{arrow} {trend.direction.value.upper()}</b><br>T{trend.trend_id}",
                        showarrow=False,
                        bgcolor=f"rgba{TrendRenderer._hex_to_rgb(color) + (0.9,)}",
                        bordercolor=color,
                        borderwidth=2,
                        font=dict(color='white', size=11),
                        row=row, col=1
                    )
            
            # Add controlling level lines
            if trend.controlling_swing:
                fig.add_hline(
                    y=trend.controlling_swing.price,
                    line=dict(color=color, width=2, dash='dash'),
                    annotation_text=f"Control: {trend.controlling_swing.price:.1f}",
                    annotation_position="right",
                    row=row, col=1
                )
            
            # Add sideways range levels
            if trend.direction == TrendDirection.SIDEWAYS and trend.range_high and trend.range_low:
                fig.add_hline(
                    y=trend.range_high,
                    line=dict(color=color, width=2, dash='dash'),
                    annotation_text=f"Range High: {trend.range_high:.1f}",
                    annotation_position="right",
                    row=row, col=1
                )
                fig.add_hline(
                    y=trend.range_low,
                    line=dict(color=color, width=2, dash='dash'),
                    annotation_text=f"Range Low: {trend.range_low:.1f}",
                    annotation_position="right",
                    row=row, col=1
                )
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class TrendSummaryRenderer:
    """
    Responsibility: Render trend summary panels and statistics.
    """
    
    @staticmethod
    def create_classification_panel(classifications: List, row: int = 2) -> go.Bar:
        """Create classification bar chart"""
        indices = list(range(len(classifications)))
        colors = ['#00aaff' if c.value == 'leg' else '#ffaa00' for c in classifications]
        labels = [c.value.upper() for c in classifications]
        
        return go.Bar(
            x=indices,
            y=[1] * len(classifications),
            marker_color=colors,
            text=labels,
            textposition='inside',
            name="Classifications",
            showlegend=False
        )
    
    @staticmethod
    def create_trend_duration_panel(trends: List[Trend], row: int = 3) -> go.Bar:
        """Create trend duration bar chart"""
        if not trends:
            return go.Bar(x=[], y=[], name="No Trends")
        
        trend_names = [f"T{t.trend_id}: {t.direction.value.upper()}" for t in trends]
        durations = [t.duration for t in trends]
        colors = [TrendColorScheme.get_trend_color(t.direction) for t in trends]
        
        return go.Bar(
            x=trend_names,
            y=durations,
            marker_color=colors,
            text=[f"<b>{dur} candles</b>" for dur in durations],
            textposition='auto',
            name="Trend Durations"
        )
    
    @staticmethod
    def create_trend_summary_table(trends: List[Trend]) -> Dict:
        """Create trend summary data for display"""
        summary = TrendDataProcessor.extract_trend_summary(trends)
        
        return {
            'total_trends': summary['total_trends'],
            'avg_duration': f"{summary['avg_duration']:.1f}",
            'avg_range': f"{summary['avg_price_range']:.1f}",
            'directions': summary['direction_counts'],
            'significance': summary['significance_counts']
        }


class TrendVisualizationEngine:
    """
    Responsibility: Orchestrate the complete trend visualization process.
    Main interface for creating trend visualizations.
    """
    
    def __init__(self, color_scheme: Optional[TrendColorScheme] = None):
        self.color_scheme = color_scheme or TrendColorScheme()
        self.candle_renderer = CandlestickRenderer()
        self.swing_renderer = SwingPointRenderer()
        self.trend_renderer = TrendRenderer()
        self.summary_renderer = TrendSummaryRenderer()
    
    def create_comprehensive_visualization(self, analysis_result: TrendAnalysisResult, 
                                         candles: List[Candle],
                                         classifications: Optional[List] = None,
                                         title: str = "Trend Analysis Results") -> go.Figure:
        """
        Create comprehensive trend visualization with multiple panels.
        
        Args:
            analysis_result: Complete trend analysis results
            candles: Original candle data (using existing Candle from supply/demand)
            classifications: Optional candle classifications for display
            title: Chart title
            
        Returns:
            Complete Plotly figure with trend visualization
        """
        # Determine panel configuration
        panel_count = 2  # Main chart + duration panel
        if classifications:
            panel_count += 1  # Add classification panel
        
        row_heights = [0.7] + [0.3 / (panel_count - 1)] * (panel_count - 1)
        
        # Create subplot structure
        subplot_titles = [
            f"📈 {title}",
            "📊 Trend Durations"
        ]
        if classifications:
            subplot_titles.insert(1, "🎯 Candle Classifications")
        
        fig = make_subplots(
            rows=panel_count, cols=1,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08
        )
        
        # === MAIN CHART PANEL ===
        
        # Add candlestick chart
        candlestick_trace = self.candle_renderer.create_candlestick_trace(candles)
        fig.add_trace(candlestick_trace, row=1, col=1)
        
        # Add candle annotations if provided
        if classifications:
            self.candle_renderer.add_candle_annotations(fig, candles, classifications, row=1)
        
        # Add trend region backgrounds
        trend_shapes = self.trend_renderer.create_trend_region_shapes(list(analysis_result.trends), candles)
        for shape in trend_shapes:
            fig.add_shape(shape, row=1, col=1)
        
        # Add swing points
        swing_traces = self.swing_renderer.create_swing_traces(list(analysis_result.swings))
        for trace in swing_traces:
            fig.add_trace(trace, row=1, col=1)
        
        # Add trend formation lines
        trend_traces = self.trend_renderer.create_trend_formation_traces(list(analysis_result.trends))
        for trace in trend_traces:
            fig.add_trace(trace, row=1, col=1)
        
        # Add trend annotations
        self.trend_renderer.add_trend_annotations(fig, list(analysis_result.trends), candles, row=1)
        
        # === SECONDARY PANELS ===
        
        current_row = 2
        
        # Classification panel (if provided)
        if classifications:
            classification_trace = self.summary_renderer.create_classification_panel(classifications, current_row)
            fig.add_trace(classification_trace, row=current_row, col=1)
            current_row += 1
        
        # Duration panel
        duration_trace = self.summary_renderer.create_trend_duration_panel(list(analysis_result.trends), current_row)
        fig.add_trace(duration_trace, row=current_row, col=1)
        
        # === STYLING ===
        self._apply_chart_styling(fig, panel_count)
        
        return fig
    
    def create_simple_trend_chart(self, analysis_result: TrendAnalysisResult,
                                candles: List[Candle], 
                                title: str = "Trend Analysis") -> go.Figure:
        """
        Create simple single-panel trend chart.
        
        Args:
            analysis_result: Trend analysis results
            candles: Original candle data
            title: Chart title
            
        Returns:
            Simple trend visualization figure
        """
        fig = go.Figure()
        
        # Add candlestick chart
        candlestick_trace = self.candle_renderer.create_candlestick_trace(candles)
        fig.add_trace(candlestick_trace)
        
        # Add trend regions
        trend_shapes = self.trend_renderer.create_trend_region_shapes(list(analysis_result.trends), candles)
        for shape in trend_shapes:
            fig.add_shape(shape)
        
        # Add swing points
        swing_traces = self.swing_renderer.create_swing_traces(list(analysis_result.swings))
        for trace in swing_traces:
            fig.add_trace(trace)
        
        # Add trend formation lines
        trend_traces = self.trend_renderer.create_trend_formation_traces(list(analysis_result.trends))
        for trace in trend_traces:
            fig.add_trace(trace)
        
        # Add trend annotations
        self.trend_renderer.add_trend_annotations(fig, list(analysis_result.trends), candles)
        
        # Styling
        fig.update_layout(
            title={
                'text': f"📈 {title}",
                'x': 0.5,
                'font': {'size': 18, 'color': self.color_scheme.TEXT_COLOR}
            },
            paper_bgcolor=self.color_scheme.PAPER_BACKGROUND,
            plot_bgcolor=self.color_scheme.CHART_BACKGROUND,
            font=dict(color=self.color_scheme.TEXT_COLOR),
            showlegend=False,
            height=600
        )
        
        fig.update_xaxes(
            gridcolor=self.color_scheme.GRID_COLOR,
            linecolor=self.color_scheme.LINE_COLOR,
            title="Candle Index"
        )
        fig.update_yaxes(
            gridcolor=self.color_scheme.GRID_COLOR,
            linecolor=self.color_scheme.LINE_COLOR,
            title="Price"
        )
        
        return fig
    
    def create_trend_comparison_chart(self, multiple_results: List[Tuple[str, TrendAnalysisResult, List['Candle']]],
                                    title: str = "Trend Analysis Comparison") -> go.Figure:
        """
        Create comparison chart for multiple trend analysis results.
        
        Args:
            multiple_results: List of (label, analysis_result, candles) tuples
            title: Chart title
            
        Returns:
            Comparison visualization figure
        """
        rows = len(multiple_results)
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=[f"📈 {label}" for label, _, _ in multiple_results],
            vertical_spacing=0.05
        )
        
        for row, (label, result, candles) in enumerate(multiple_results, 1):
            # Add candlestick
            candlestick_trace = self.candle_renderer.create_candlestick_trace(candles, f"{label} Price")
            fig.add_trace(candlestick_trace, row=row, col=1)
            
            # Add trends
            trend_traces = self.trend_renderer.create_trend_formation_traces(list(result.trends))
            for trace in trend_traces:
                fig.add_trace(trace, row=row, col=1)
            
            # Add trend annotations
            self.trend_renderer.add_trend_annotations(fig, list(result.trends), candles, row=row)
        
        # Styling
        self._apply_chart_styling(fig, rows)
        fig.update_layout(
            title={
                'text': f"📊 {title}",
                'x': 0.5,
                'font': {'size': 20, 'color': self.color_scheme.TEXT_COLOR}
            },
            height=300 * rows
        )
        
        return fig
    
    def export_trend_summary_data(self, analysis_result: TrendAnalysisResult) -> Dict:
        """
        Export trend analysis data in format suitable for external use.
        
        Args:
            analysis_result: Complete trend analysis results
            
        Returns:
            Dictionary with exportable trend data
        """
        return TrendDataProcessor.convert_to_visualization_format(analysis_result, [])
    
    def _apply_chart_styling(self, fig: go.Figure, panel_count: int) -> None:
        """Apply consistent styling to the chart"""
        fig.update_layout(
            title={
                'x': 0.5,
                'font': {'size': 20, 'color': self.color_scheme.TEXT_COLOR}
            },
            paper_bgcolor=self.color_scheme.PAPER_BACKGROUND,
            plot_bgcolor=self.color_scheme.CHART_BACKGROUND,
            font=dict(color=self.color_scheme.TEXT_COLOR),
            height=200 + (400 * panel_count),
            showlegend=False
        )
        
        # Style all axes
        for row in range(1, panel_count + 1):
            fig.update_xaxes(
                gridcolor=self.color_scheme.GRID_COLOR,
                linecolor=self.color_scheme.LINE_COLOR,
                title_font_color='#cccccc',
                tickfont_color='#cccccc',
                row=row, col=1
            )
            fig.update_yaxes(
                gridcolor=self.color_scheme.GRID_COLOR,
                linecolor=self.color_scheme.LINE_COLOR,
                title_font_color='#cccccc',
                tickfont_color='#cccccc',
                row=row, col=1
            )
        
        # Set main chart y-axis title
        fig.update_yaxes(title_text="Price", row=1, col=1)


# Factory function for easy visualization creation
def create_trend_visualization(analysis_result: TrendAnalysisResult, 
                             candles: List[Candle],
                             style: str = "comprehensive",
                             **kwargs) -> 'go.Figure':
    """
    Factory function to create trend visualizations.
    
    Args:
        analysis_result: Trend analysis results
        candles: Candle data (using existing Candle from supply/demand)
        style: Visualization style ("comprehensive", "simple", "comparison")
        **kwargs: Additional arguments for specific visualization types
        
    Returns:
        Plotly figure with trend visualization
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install with: pip install plotly")
    
    engine = TrendVisualizationEngine()
    
    if style == "comprehensive":
        return engine.create_comprehensive_visualization(
            analysis_result, candles, **kwargs
        )
    elif style == "simple":
        return engine.create_simple_trend_chart(
            analysis_result, candles, **kwargs
        )
    else:
        raise ValueError(f"Unknown visualization style: {style}")