"""
Dark Mode HTML Report Generator for Trade Likelihood Backtests

Creates comprehensive, professional-looking dark mode reports showing:
- Trade setup and parameters
- Model predictions vs actual outcomes  
- Historical data sensitivity analysis
- Visual charts and metrics
- Recommendations and insights
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Set matplotlib style for dark theme
plt.style.use('dark_background')


class BacktestReportGenerator:
    """Generate comprehensive dark mode HTML reports for backtest results"""
    
    def __init__(self):
        self.dark_theme = {
            'bg_primary': '#0d1117',
            'bg_secondary': '#161b22', 
            'bg_tertiary': '#21262d',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e',
            'accent_blue': '#58a6ff',
            'accent_green': '#3fb950',
            'accent_red': '#f85149',
            'accent_yellow': '#d29922',
            'accent_purple': '#a5a5f5',
            'border': '#30363d'
        }
    
    def generate_single_trade_report(self, backtest_result, output_path: str = None) -> str:
        """
        Generate comprehensive HTML report for a single trade backtest
        
        Args:
            backtest_result: BacktestResult object
            output_path: Optional custom output path
            
        Returns:
            Path to generated HTML file
        """
        
        if output_path is None:
            # Use the actual trade date, not current timestamp!
            trade_date = backtest_result.trade.trade_date.strftime('%Y%m%d')
            symbol = backtest_result.trade.symbol
            direction = backtest_result.trade.direction
            output_path = f"backtest_{symbol}_{direction}_{trade_date}.html"
        
        # Generate report sections
        trade_summary = self._generate_trade_summary(backtest_result)
        model_analysis = self._generate_model_analysis(backtest_result) 
        optimization_analysis = self._generate_optimization_analysis(backtest_result)
        outcome_analysis = self._generate_outcome_analysis(backtest_result)
        charts = self._generate_charts(backtest_result)
        recommendations = self._generate_recommendations(backtest_result)
        
        # Compile full HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trade Likelihood Backtest Report - {backtest_result.trade.symbol}</title>
            {self._get_css_styles()}
        </head>
        <body>
            <div class="container">
                {self._generate_header(backtest_result)}
                {trade_summary}
                {model_analysis}
                {optimization_analysis}
                {outcome_analysis}
                {charts}
                {recommendations}
                {self._generate_footer()}
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Report generated: {output_path}")
        return output_path
    
    def _get_css_styles(self) -> str:
        """Return CSS styles for dark mode report"""
        return f"""
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
                background-color: {self.dark_theme['bg_primary']};
                color: {self.dark_theme['text_primary']};
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px 0;
                border-bottom: 2px solid {self.dark_theme['border']};
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(135deg, {self.dark_theme['accent_blue']}, {self.dark_theme['accent_purple']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .header .subtitle {{
                color: {self.dark_theme['text_secondary']};
                font-size: 1.1rem;
            }}
            
            .section {{
                background: {self.dark_theme['bg_secondary']};
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 30px;
                border: 1px solid {self.dark_theme['border']};
            }}
            
            .section h2 {{
                color: {self.dark_theme['accent_blue']};
                margin-bottom: 20px;
                font-size: 1.8rem;
                display: flex;
                align-items: center;
            }}
            
            .section h2::before {{
                content: '';
                width: 4px;
                height: 30px;
                background: {self.dark_theme['accent_blue']};
                margin-right: 15px;
                border-radius: 2px;
            }}
            
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            
            .metric-card {{
                background: {self.dark_theme['bg_tertiary']};
                padding: 20px;
                border-radius: 8px;
                border: 1px solid {self.dark_theme['border']};
                text-align: center;
            }}
            
            .metric-card .label {{
                color: {self.dark_theme['text_secondary']};
                font-size: 0.9rem;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .metric-card .value {{
                font-size: 1.8rem;
                font-weight: bold;
                color: {self.dark_theme['text_primary']};
            }}
            
            .metric-card.positive .value {{
                color: {self.dark_theme['accent_green']};
            }}
            
            .metric-card.negative .value {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .metric-card.neutral .value {{
                color: {self.dark_theme['accent_blue']};
            }}
            
            .analysis-table {{
                width: 100%;
                border-collapse: collapse;
                background: {self.dark_theme['bg_tertiary']};
                border-radius: 8px;
                overflow: hidden;
            }}
            
            .analysis-table th,
            .analysis-table td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid {self.dark_theme['border']};
            }}
            
            .analysis-table th {{
                background: {self.dark_theme['bg_primary']};
                color: {self.dark_theme['accent_blue']};
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9rem;
                letter-spacing: 0.5px;
            }}
            
            .analysis-table tr:hover {{
                background: {self.dark_theme['bg_primary']};
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .status-badge.win {{
                background: {self.dark_theme['accent_green']}33;
                color: {self.dark_theme['accent_green']};
                border: 1px solid {self.dark_theme['accent_green']};
            }}
            
            .status-badge.loss {{
                background: {self.dark_theme['accent_red']}33;
                color: {self.dark_theme['accent_red']};
                border: 1px solid {self.dark_theme['accent_red']};
            }}
            
            .status-badge.pending {{
                background: {self.dark_theme['accent_yellow']}33;
                color: {self.dark_theme['accent_yellow']};
                border: 1px solid {self.dark_theme['accent_yellow']};
            }}
            
            .status-badge.attractive {{
                background: {self.dark_theme['accent_green']}33;
                color: {self.dark_theme['accent_green']};
                border: 1px solid {self.dark_theme['accent_green']};
            }}
            
            .status-badge.not-attractive {{
                background: {self.dark_theme['accent_red']}33;
                color: {self.dark_theme['accent_red']};
                border: 1px solid {self.dark_theme['accent_red']};
            }}
            
            .chart-container {{
                background: {self.dark_theme['bg_tertiary']};
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: center;
            }}
            
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 6px;
            }}
            
            .recommendation {{
                background: linear-gradient(135deg, {self.dark_theme['accent_blue']}22, {self.dark_theme['accent_purple']}22);
                border: 1px solid {self.dark_theme['accent_blue']};
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
            }}
            
            .recommendation h3 {{
                color: {self.dark_theme['accent_blue']};
                margin-bottom: 15px;
                font-size: 1.3rem;
            }}
            
            .recommendation ul {{
                list-style: none;
                padding-left: 0;
            }}
            
            .recommendation li {{
                margin-bottom: 10px;
                padding-left: 20px;
                position: relative;
            }}
            
            .recommendation li::before {{
                content: '▶';
                color: {self.dark_theme['accent_blue']};
                position: absolute;
                left: 0;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 30px 0;
                border-top: 2px solid {self.dark_theme['border']};
                color: {self.dark_theme['text_secondary']};
            }}
            
            .footer .timestamp {{
                font-size: 0.9rem;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                
                .header h1 {{
                    font-size: 2rem;
                }}
                
                .metric-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .section {{
                    padding: 20px;
                }}
            }}
            
            /* Bidirectional Strategy Specific Styles */
            .setups-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 25px;
            }}
            
            .setup-card {{
                background: {self.dark_theme['bg_tertiary']};
                border-radius: 12px;
                padding: 25px;
                border: 2px solid {self.dark_theme['border']};
            }}
            
            .setup-card.long-setup {{
                border-color: {self.dark_theme['accent_green']};
            }}
            
            .setup-card.short-setup {{
                border-color: {self.dark_theme['accent_red']};
            }}
            
            .setup-card h3 {{
                color: {self.dark_theme['text_primary']};
                margin-bottom: 20px;
                font-size: 1.4rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .setup-details {{
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}
            
            .setup-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid {self.dark_theme['border']};
            }}
            
            .setup-row.highlight {{
                background: {self.dark_theme['bg_primary']};
                padding: 12px;
                border-radius: 6px;
                border: none;
                margin-top: 8px;
            }}
            
            .setup-row .price {{
                font-weight: bold;
                font-size: 1.1rem;
            }}
            
            .setup-row .price.entry {{
                color: {self.dark_theme['accent_blue']};
            }}
            
            .setup-row .price.stop {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .setup-row .price.target {{
                color: {self.dark_theme['accent_green']};
            }}
            
            .setup-row .rr-ratio {{
                color: {self.dark_theme['accent_yellow']};
                font-weight: bold;
                font-size: 1.2rem;
            }}
            
            .performance-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 25px;
            }}
            
            .performance-section {{
                background: {self.dark_theme['bg_tertiary']};
                padding: 20px;
                border-radius: 8px;
                border: 1px solid {self.dark_theme['border']};
            }}
            
            .performance-section h3 {{
                color: {self.dark_theme['accent_blue']};
                margin-bottom: 15px;
                font-size: 1.2rem;
            }}
            
            .mini-metrics {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            
            .mini-metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: {self.dark_theme['bg_primary']};
                border-radius: 5px;
            }}
            
            .mini-metric.highlight {{
                border: 2px solid {self.dark_theme['accent_blue']};
                background: {self.dark_theme['bg_secondary']};
            }}
            
            .mini-metric .value {{
                font-weight: bold;
            }}
            
            .mini-metric .value.negative {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .sensitivity-analysis {{
                background: {self.dark_theme['bg_tertiary']};
                border-radius: 8px;
                padding: 20px;
            }}
            
            .sensitivity-header {{
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
                padding: 15px;
                background: {self.dark_theme['bg_primary']};
                border-radius: 5px;
                margin-bottom: 15px;
                font-weight: bold;
                color: {self.dark_theme['accent_blue']};
            }}
            
            .sensitivity-row {{
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
                padding: 12px 15px;
                border-bottom: 1px solid {self.dark_theme['border']};
                align-items: center;
            }}
            
            .window-label {{
                font-weight: bold;
                color: {self.dark_theme['text_primary']};
            }}
            
            .window-metrics {{
                display: flex;
                gap: 15px;
                justify-content: space-around;
            }}
            
            .window-metrics .metric {{
                font-size: 0.9rem;
                padding: 4px 8px;
                background: {self.dark_theme['bg_primary']};
                border-radius: 4px;
            }}
            
            .window-metrics .metric.combined {{
                font-weight: bold;
                background: {self.dark_theme['bg_secondary']};
                border: 1px solid {self.dark_theme['border']};
            }}
            
            .window-metrics .metric.negative {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .assessment-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }}
            
            .assessment-card {{
                background: {self.dark_theme['bg_tertiary']};
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                border: 2px solid {self.dark_theme['border']};
            }}
            
            .assessment-card.attractive {{
                border-color: {self.dark_theme['accent_green']};
                background: linear-gradient(135deg, {self.dark_theme['bg_tertiary']}, rgba(63, 185, 80, 0.1));
            }}
            
            .assessment-card.not-attractive {{
                border-color: {self.dark_theme['accent_red']};
                background: linear-gradient(135deg, {self.dark_theme['bg_tertiary']}, rgba(248, 81, 73, 0.1));
            }}
            
            .assessment-card h3 {{
                color: {self.dark_theme['text_primary']};
                margin-bottom: 15px;
                font-size: 1.1rem;
            }}
            
            .assessment-value {{
                font-size: 1.4rem;
                font-weight: bold;
                margin-bottom: 10px;
                color: {self.dark_theme['accent_blue']};
            }}
            
            .assessment-value.risk-high {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .assessment-value.risk-moderate {{
                color: {self.dark_theme['accent_yellow']};
            }}
            
            .assessment-value.risk-low {{
                color: {self.dark_theme['accent_green']};
            }}
            
            .assessment-reason {{
                color: {self.dark_theme['text_secondary']};
                font-size: 0.9rem;
                line-height: 1.4;
            }}
            
            .recommendations {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
            }}
            
            .recommendation-category {{
                background: {self.dark_theme['bg_tertiary']};
                padding: 25px;
                border-radius: 8px;
                border: 1px solid {self.dark_theme['border']};
            }}
            
            .recommendation-category h3 {{
                color: {self.dark_theme['accent_blue']};
                margin-bottom: 15px;
                font-size: 1.2rem;
            }}
            
            .recommendation-category ul {{
                list-style: none;
                padding-left: 0;
            }}
            
            .recommendation-category li {{
                margin-bottom: 12px;
                padding-left: 20px;
                position: relative;
                line-height: 1.5;
            }}
            
            .recommendation-category li::before {{
                content: '💡';
                position: absolute;
                left: 0;
                top: 0;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }}
            
            .metric-card .metric-label {{
                color: {self.dark_theme['text_secondary']};
                font-size: 0.9rem;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .metric-card .metric-value {{
                font-size: 1.8rem;
                font-weight: bold;
                color: {self.dark_theme['text_primary']};
            }}
            
            .metric-card .metric-note {{
                color: {self.dark_theme['text_secondary']};
                font-size: 0.8rem;
                margin-top: 5px;
                font-style: italic;
            }}
            
            .metric-card.highlight {{
                border: 2px solid {self.dark_theme['accent_blue']};
                background: linear-gradient(135deg, {self.dark_theme['bg_tertiary']}, rgba(88, 166, 255, 0.1));
            }}
            
            .metric-value.probability {{
                color: {self.dark_theme['accent_purple']};
            }}
            
            .metric-value.positive {{
                color: {self.dark_theme['accent_green']};
            }}
            
            .metric-value.negative {{
                color: {self.dark_theme['accent_red']};
            }}
            
            .subtitle .symbol {{
                color: {self.dark_theme['accent_blue']};
                font-weight: bold;
                font-size: 1.3rem;
            }}
            
            .subtitle .date {{
                color: {self.dark_theme['accent_yellow']};
            }}
            
            .description {{
                color: {self.dark_theme['text_secondary']};
                font-size: 1.1rem;
                margin-top: 15px;
                font-style: italic;
            }}
            
            @media (max-width: 768px) {{
                .setups-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .performance-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .assessment-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .recommendations {{
                    grid-template-columns: 1fr;
                }}
                
                .sensitivity-header,
                .sensitivity-row {{
                    grid-template-columns: 1fr;
                    gap: 10px;
                }}
                
                .window-metrics {{
                    flex-direction: column;
                    gap: 8px;
                }}
            }}
        </style>
        """
    
    def _generate_header(self, backtest_result) -> str:
        """Generate report header"""
        trade = backtest_result.trade
        return f"""
        <div class="header">
            <h1>📊 Trade Likelihood Backtest Report</h1>
            <div class="subtitle">
                {trade.symbol} • {trade.direction.upper()} • {trade.trade_date.strftime('%B %d, %Y')}
            </div>
        </div>
        """
    
    def _generate_trade_summary(self, backtest_result) -> str:
        """Generate trade summary section"""
        trade = backtest_result.trade
        
        # Calculate key metrics
        risk = abs(trade.entry_price - trade.stop_loss)
        reward = abs(trade.target_price - trade.entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Outcome styling
        outcome_class = ""
        if backtest_result.actual_outcome:
            if backtest_result.actual_outcome == 'win':
                outcome_class = "positive"
            elif backtest_result.actual_outcome == 'loss':
                outcome_class = "negative"
            else:
                outcome_class = "neutral"
        
        return f"""
        <div class="section">
            <h2>🎯 Trade Summary</h2>
            <div class="metric-grid">
                <div class="metric-card neutral">
                    <div class="label">Symbol</div>
                    <div class="value">{trade.symbol}</div>
                </div>
                <div class="metric-card neutral">
                    <div class="label">Direction</div>
                    <div class="value">{trade.direction.upper()}</div>
                </div>
                <div class="metric-card neutral">
                    <div class="label">Entry Price</div>
                    <div class="value">${trade.entry_price:.2f}</div>
                </div>
                <div class="metric-card negative">
                    <div class="label">Stop Loss</div>
                    <div class="value">${trade.stop_loss:.2f}</div>
                </div>
                <div class="metric-card positive">
                    <div class="label">Target</div>
                    <div class="value">${trade.target_price:.2f}</div>
                </div>
                <div class="metric-card {"positive" if risk_reward > 2 else "neutral"}">
                    <div class="label">Risk:Reward</div>
                    <div class="value">{risk_reward:.2f}:1</div>
                </div>
            </div>
            
            {self._generate_actual_outcome_section(backtest_result)}
        </div>
        """
    
    def _generate_actual_outcome_section(self, backtest_result) -> str:
        """Generate actual outcome section if available"""
        if not backtest_result.actual_outcome or backtest_result.actual_outcome == 'unknown':
            return """
            <div class="recommendation">
                <h3>📊 Actual Outcome</h3>
                <p>Outcome data not available - trade may still be pending or data insufficient.</p>
            </div>
            """
        
        outcome = backtest_result.actual_outcome
        badge_class = outcome
        
        outcome_html = f"""
        <div class="recommendation">
            <h3>📊 Actual Outcome</h3>
            <div style="margin-bottom: 15px;">
                <span class="status-badge {badge_class}">{outcome.upper()}</span>
            </div>
        """
        
        # Entry trigger information
        if hasattr(backtest_result, 'entry_triggered') and backtest_result.entry_triggered is not None:
            trigger_status = "✅ YES" if backtest_result.entry_triggered else "❌ NO"
            outcome_html += f"<p><strong>Entry Triggered:</strong> {trigger_status}</p>"
        
        # Timing information
        if hasattr(backtest_result, 'days_to_entry') and backtest_result.days_to_entry is not None:
            outcome_html += f"<p><strong>Days to Entry:</strong> {backtest_result.days_to_entry} days</p>"
        
        if backtest_result.days_to_exit is not None:
            outcome_html += f"<p><strong>Days to Exit:</strong> {backtest_result.days_to_exit} days</p>"
        
        if backtest_result.actual_exit_price:
            outcome_html += f"<p><strong>Exit Price:</strong> ${backtest_result.actual_exit_price:.2f}</p>"
        
        if backtest_result.actual_exit_date:
            outcome_html += f"<p><strong>Exit Date:</strong> {backtest_result.actual_exit_date.strftime('%Y-%m-%d')}</p>"
        
        outcome_html += "</div>"
        return outcome_html
    
    def _generate_model_analysis(self, backtest_result) -> str:
        """Generate model predictions analysis section"""
        predictions = backtest_result.model_predictions
        
        if not predictions:
            return """
            <div class="section">
                <h2>🤖 Model Predictions</h2>
                <p>No model predictions available - insufficient historical data.</p>
            </div>
            """
        
        # Create comprehensive table of predictions
        table_rows = ""
        for window, analysis in predictions.items():
            
            return_rate = analysis.return_rate_total or 0
            entry_prob = analysis.get_combined_entry_probability() or 0
            attractive = analysis.is_attractive_setup()
            best_dir = analysis.get_best_direction() or 'None'
            regime = analysis.market_params.regime
            
            # Get detailed probabilities and times
            win_prob = "N/A"
            expected_entry_time = "N/A"
            expected_trade_duration = "N/A"
            
            if best_dir == 'long':
                if analysis.prob_win_long:
                    win_prob = f"{analysis.prob_win_long:.1%}"
                if analysis.expected_entry_time_long:
                    expected_entry_time = f"{analysis.expected_entry_time_long:.1f}d"
                if analysis.expected_trade_duration_long:
                    expected_trade_duration = f"{analysis.expected_trade_duration_long:.1f}d"
            elif best_dir == 'short':
                if analysis.prob_win_short:
                    win_prob = f"{analysis.prob_win_short:.1%}"
                if analysis.expected_entry_time_short:
                    expected_entry_time = f"{analysis.expected_entry_time_short:.1f}d"
                if analysis.expected_trade_duration_short:
                    expected_trade_duration = f"{analysis.expected_trade_duration_short:.1f}d"
            
            attractive_badge = f'<span class="status-badge {"attractive" if attractive else "not-attractive"}">{"✅ YES" if attractive else "❌ NO"}</span>'
            
            table_rows += f"""
            <tr>
                <td><strong>{window}</strong></td>
                <td>{return_rate:.4f}</td>
                <td>{entry_prob:.1%}</td>
                <td>{win_prob}</td>
                <td>{expected_entry_time}</td>
                <td>{expected_trade_duration}</td>
                <td>{best_dir}</td>
                <td>{regime}</td>
                <td>{attractive_badge}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>🤖 Original Setup Analysis</h2>
            <p style="margin-bottom: 25px; color: {self.dark_theme['text_secondary']};">
                Analysis using different historical data windows to test model sensitivity and consistency.
            </p>
            
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Historical Window</th>
                        <th>Return Rate/Day</th>
                        <th>Entry Probability</th>
                        <th>Win Probability</th>
                        <th>Expected Entry Time</th>
                        <th>Expected Trade Duration</th>
                        <th>Best Direction</th>
                        <th>Market Regime</th>
                        <th>Attractive Setup</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_optimization_analysis(self, backtest_result) -> str:
        """Generate trade setup optimization analysis section"""
        
        if not backtest_result.optimization_results:
            return """
            <div class="section">
                <h2>🎯 Trade Setup Optimization</h2>
                <p style="color: #8b949e; font-style: italic;">
                    Optimization was not performed or failed for this trade.
                </p>
            </div>
            """
        
        opt_result = backtest_result.optimization_results
        opt_predictions = backtest_result.optimized_trade_predictions or {}
        
        # Generate setup comparison
        orig_setup = opt_result.original_setup
        opt_setup = opt_result.optimized_setup
        
        if orig_setup.has_long_setup():
            direction = "LONG"
            orig_entry = orig_setup.entry_long
            orig_stop = orig_setup.stop_long
            orig_target = orig_setup.target_long
            opt_entry = opt_setup.entry_long
            opt_stop = opt_setup.stop_long
            opt_target = opt_setup.target_long
        else:
            direction = "SHORT"
            orig_entry = orig_setup.entry_short
            orig_stop = orig_setup.stop_short
            orig_target = orig_setup.target_short
            opt_entry = opt_setup.entry_short
            opt_stop = opt_setup.stop_short
            opt_target = opt_setup.target_short
        
        orig_rr = opt_result.original_analysis.risk_reward_long or opt_result.original_analysis.risk_reward_short or 0
        opt_rr = opt_result.optimized_analysis.risk_reward_long or opt_result.optimized_analysis.risk_reward_short or 0
        
        # Generate optimization metrics comparison
        orig_analysis = opt_result.original_analysis
        opt_analysis = opt_result.optimized_analysis
        
        orig_return_rate = orig_analysis.return_rate_total or 0
        opt_return_rate = opt_analysis.return_rate_total or 0
        return_improvement = ((opt_return_rate - orig_return_rate) / abs(orig_return_rate) * 100) if orig_return_rate != 0 else 0
        
        orig_win_prob = (orig_analysis.prob_win_long or 0) + (orig_analysis.prob_win_short or 0)
        opt_win_prob = (opt_analysis.prob_win_long or 0) + (opt_analysis.prob_win_short or 0)
        win_improvement = ((opt_win_prob - orig_win_prob) / orig_win_prob * 100) if orig_win_prob != 0 else 0
        
        orig_entry_prob = orig_analysis.get_combined_entry_probability() or 0
        opt_entry_prob = opt_analysis.get_combined_entry_probability() or 0
        entry_improvement = ((opt_entry_prob - orig_entry_prob) / orig_entry_prob * 100) if orig_entry_prob != 0 else 0
        
        # Generate optimized predictions table if available
        opt_table_rows = ""
        if opt_predictions:
            for window, analysis in opt_predictions.items():
                return_rate = analysis.return_rate_total or 0
                entry_prob = analysis.get_combined_entry_probability() or 0
                attractive = analysis.is_attractive_setup()
                best_dir = analysis.get_best_direction() or 'None'
                regime = analysis.market_params.regime
                
                # Get detailed probabilities and times
                win_prob = "N/A"
                expected_entry_time = "N/A"
                expected_trade_duration = "N/A"
                
                if best_dir == 'long':
                    if analysis.prob_win_long:
                        win_prob = f"{analysis.prob_win_long:.1%}"
                    if analysis.expected_entry_time_long:
                        expected_entry_time = f"{analysis.expected_entry_time_long:.1f}d"
                    if analysis.expected_trade_duration_long:
                        expected_trade_duration = f"{analysis.expected_trade_duration_long:.1f}d"
                elif best_dir == 'short':
                    if analysis.prob_win_short:
                        win_prob = f"{analysis.prob_win_short:.1%}"
                    if analysis.expected_entry_time_short:
                        expected_entry_time = f"{analysis.expected_entry_time_short:.1f}d"
                    if analysis.expected_trade_duration_short:
                        expected_trade_duration = f"{analysis.expected_trade_duration_short:.1f}d"
                
                attractive_badge = f'<span class="status-badge {"attractive" if attractive else "not-attractive"}">{"✅ YES" if attractive else "❌ NO"}</span>'
                
                opt_table_rows += f"""
                <tr>
                    <td><strong>{window}</strong></td>
                    <td>{return_rate:.4f}</td>
                    <td>{entry_prob:.1%}</td>
                    <td>{win_prob}</td>
                    <td>{expected_entry_time}</td>
                    <td>{expected_trade_duration}</td>
                    <td>{best_dir}</td>
                    <td>{regime}</td>
                    <td>{attractive_badge}</td>
                </tr>
                """
        
        return f"""
        <div class="section">
            <h2>🎯 Trade Setup Optimization</h2>
            <p style="margin-bottom: 25px; color: {self.dark_theme['text_secondary']};">
                Mathematically optimized setup to maximize {opt_result.objective_function} while respecting R:R constraints.
            </p>
            
            <div class="metric-grid" style="margin-bottom: 30px;">
                <div class="metric-card neutral">
                    <div class="label">Original Entry</div>
                    <div class="value">${orig_entry:.2f}</div>
                </div>
                <div class="metric-card {"positive" if opt_entry != orig_entry else "neutral"}">
                    <div class="label">Optimized Entry</div>
                    <div class="value">${opt_entry:.2f}</div>
                </div>
                <div class="metric-card neutral">
                    <div class="label">Original Stop</div>
                    <div class="value">${orig_stop:.2f}</div>
                </div>
                <div class="metric-card {"positive" if opt_stop != orig_stop else "neutral"}">
                    <div class="label">Optimized Stop</div>
                    <div class="value">${opt_stop:.2f}</div>
                </div>
                <div class="metric-card neutral">
                    <div class="label">Original Target</div>
                    <div class="value">${orig_target:.2f}</div>
                </div>
                <div class="metric-card {"positive" if opt_target != orig_target else "neutral"}">
                    <div class="label">Optimized Target</div>
                    <div class="value">${opt_target:.2f}</div>
                </div>
                <div class="metric-card neutral">
                    <div class="label">Original R:R</div>
                    <div class="value">{orig_rr:.2f}:1</div>
                </div>
                <div class="metric-card {"positive" if opt_rr >= 3.0 else "negative"}">
                    <div class="label">Optimized R:R</div>
                    <div class="value">{opt_rr:.2f}:1</div>
                </div>
            </div>
            
            <div class="recommendation">
                <h3>📊 Performance Improvements</h3>
                <div class="metric-grid">
                    <div class="metric-card {"positive" if return_improvement > 0 else "negative"}">
                        <div class="label">Return Rate Change</div>
                        <div class="value">{return_improvement:+.1f}%</div>
                    </div>
                    <div class="metric-card {"positive" if win_improvement > 0 else "negative"}">
                        <div class="label">Win Probability Change</div>
                        <div class="value">{win_improvement:+.1f}%</div>
                    </div>
                    <div class="metric-card {"positive" if entry_improvement > 0 else "negative"}">
                        <div class="label">Entry Probability Change</div>
                        <div class="value">{entry_improvement:+.1f}%</div>
                    </div>
                </div>
            </div>
            
            {f'''
            <h3 style="margin-top: 30px; color: {self.dark_theme['accent_blue']};">Optimized Setup Analysis</h3>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Historical Window</th>
                        <th>Return Rate/Day</th>
                        <th>Entry Probability</th>
                        <th>Win Probability</th>
                        <th>Expected Entry Time</th>
                        <th>Expected Trade Duration</th>
                        <th>Best Direction</th>
                        <th>Market Regime</th>
                        <th>Attractive Setup</th>
                    </tr>
                </thead>
                <tbody>
                    {opt_table_rows}
                </tbody>
            </table>
            ''' if opt_table_rows else '<p style="color: #8b949e; margin-top: 20px;">Detailed optimized analysis not available.</p>'}
        </div>
        """
    
    def _generate_outcome_analysis(self, backtest_result) -> str:
        """Generate model vs actual outcome analysis"""
        
        if not backtest_result.model_predictions or backtest_result.actual_outcome == 'unknown':
            return ""
        
        # Calculate prediction accuracy
        actual_success = backtest_result.actual_outcome == 'win'
        
        analysis_rows = ""
        for window, prediction in backtest_result.model_predictions.items():
            predicted_attractive = prediction.is_attractive_setup()
            
            # Simple accuracy: did model correctly predict if trade would be successful?
            if actual_success and predicted_attractive:
                accuracy = "✅ Correct - Predicted attractive, trade won"
                accuracy_class = "positive"
            elif not actual_success and not predicted_attractive:
                accuracy = "✅ Correct - Predicted unattractive, trade lost"
                accuracy_class = "positive"
            elif actual_success and not predicted_attractive:
                accuracy = "❌ Miss - Predicted unattractive, but trade won"
                accuracy_class = "negative"
            else:
                accuracy = "❌ False positive - Predicted attractive, but trade lost"
                accuracy_class = "negative"
            
            analysis_rows += f"""
            <tr>
                <td><strong>{window}</strong></td>
                <td><span class="metric-card {accuracy_class}" style="display: inline; padding: 5px 10px; font-size: 0.9rem;">{accuracy}</span></td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>📈 Model Accuracy Analysis</h2>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Historical Window</th>
                        <th>Prediction Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {analysis_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_charts(self, backtest_result) -> str:
        """Generate comprehensive visual analysis section"""
        
        from .chart_generator import TradeChartGenerator
        
        # Initialize chart generator
        chart_gen = TradeChartGenerator(figsize=(14, 10))
        
        charts_html = f"""
        <div class="section">
            <h2>📊 Visual Analysis</h2>
            <div class="chart-container">
        """
        
        try:
            # Get trade details and best prediction
            trade = backtest_result.trade
            predictions = backtest_result.model_predictions
            
            if not predictions:
                charts_html += f"""
                <p style="color: {self.dark_theme['text_secondary']}; font-style: italic;">
                    No predictions available for chart generation.
                </p>
                """
            else:
                # Use best prediction window for chart parameters
                best_window = max(predictions.keys(), key=lambda k: predictions[k].return_rate_total or -999)
                best_prediction = predictions[best_window]
                market_params = best_prediction.market_params
                
                # Estimate current price (entry price adjusted by reasonable market move)
                if trade.direction == 'long':
                    current_price = trade.entry_price * 0.98  # Assume 2% below entry for long
                    entry = trade.entry_price
                    stop = trade.stop_loss
                    target = trade.target_price
                else:  # short
                    current_price = trade.entry_price * 1.02  # Assume 2% above entry for short
                    entry = trade.entry_price  
                    stop = trade.stop_loss
                    target = trade.target_price
                
                # Generate Monte Carlo simulation chart
                print(f"🎨 Generating visual analysis charts for {trade.symbol}...")
                simulation_chart = chart_gen.generate_price_simulation_chart(
                    current_price=current_price,
                    entry_price=entry,
                    stop_loss=stop,
                    target_price=target,
                    drift=market_params.drift,
                    volatility=market_params.volatility,
                    direction=trade.direction,
                    days_ahead=45,
                    num_simulations=1000
                )
                
                charts_html += f"""
                <h3 style="color: {self.dark_theme['accent_blue']}; margin-bottom: 15px;">
                    📈 Monte Carlo Price Simulation & Probability Analysis
                </h3>
                <p style="color: {self.dark_theme['text_secondary']}; margin-bottom: 20px;">
                    Geometric Brownian Motion simulation showing expected price paths and entry/exit probabilities.
                    Based on {best_window} historical data window with drift: {market_params.drift:.3f}, volatility: {market_params.volatility:.3f}
                </p>
                <img src="data:image/png;base64,{simulation_chart}" 
                     alt="Monte Carlo Simulation Chart" 
                     style="max-width: 100%; height: auto; border-radius: 6px; margin-bottom: 20px;">
                """
                
                # Generate validation chart  
                validation_chart = chart_gen.generate_trade_outcome_validation_chart(
                    backtest_result=backtest_result,
                    num_simulations=5000
                )
                
                charts_html += f"""
                <h3 style="color: {self.dark_theme['accent_blue']}; margin-bottom: 15px; margin-top: 30px;">
                    🔬 Mathematical Model Validation
                </h3>
                <p style="color: {self.dark_theme['text_secondary']}; margin-bottom: 20px;">
                    Comparison of mathematical predictions vs Monte Carlo simulation results to validate model accuracy.
                </p>
                <img src="data:image/png;base64,{validation_chart}" 
                     alt="Model Validation Chart" 
                     style="max-width: 100%; height: auto; border-radius: 6px;">
                """
                
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            charts_html += f"""
            <p style="color: {self.dark_theme['accent_red']};">
                ⚠️ Chart generation failed: {str(e)}
            </p>
            """
        
        charts_html += """
            </div>
        </div>
        """
        
        return charts_html
    
    def _generate_recommendations(self, backtest_result) -> str:
        """Generate recommendations and insights"""
        recommendations = []
        
        # Analyze model consistency
        if backtest_result.model_predictions:
            attractive_count = sum(1 for analysis in backtest_result.model_predictions.values() 
                                 if analysis.is_attractive_setup())
            total_predictions = len(backtest_result.model_predictions)
            consistency = attractive_count / total_predictions
            
            if consistency >= 0.75:
                recommendations.append("Model shows high consistency across different historical windows")
            elif consistency <= 0.25:
                recommendations.append("Model consistently predicts this setup as unattractive")
            else:
                recommendations.append("Model predictions vary significantly with historical window size")
        
        # Analyze actual outcome
        if backtest_result.actual_outcome == 'win':
            recommendations.append("Trade was successful - validate if model correctly identified opportunity")
        elif backtest_result.actual_outcome == 'loss':
            recommendations.append("Trade was unsuccessful - analyze if model provided adequate warning")
        
        # Risk-reward analysis
        rr_ratio = backtest_result.trade.get_risk_reward_ratio()
        if rr_ratio > 3:
            recommendations.append(f"Excellent risk-reward ratio ({rr_ratio:.2f}:1) - high profit potential setup")
        elif rr_ratio < 1.5:
            recommendations.append(f"Poor risk-reward ratio ({rr_ratio:.2f}:1) - consider tighter stops or distant targets")
        
        # Model-specific insights
        if backtest_result.model_predictions:
            avg_return_rate = np.mean([analysis.return_rate_total or 0 
                                     for analysis in backtest_result.model_predictions.values()])
            if avg_return_rate > 0.05:
                recommendations.append("Model predicts high daily return rate - very attractive opportunity")
            elif avg_return_rate < 0:
                recommendations.append("Model predicts negative returns - setup should be avoided")
        
        if not recommendations:
            recommendations.append("Insufficient data for detailed recommendations")
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        
        return f"""
        <div class="section">
            <h2>💡 Key Insights & Recommendations</h2>
            <div class="recommendation">
                <h3>Model Performance Insights</h3>
                <ul>
                    {recommendations_html}
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Next Steps</h3>
                <ul>
                    <li>Compare these results with other trades in your journal</li>
                    <li>Analyze if model accuracy varies by symbol or market conditions</li>
                    <li>Consider adjusting historical window size based on model consistency</li>
                    <li>Use these insights to refine your paper trading strategy</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""
        <div class="footer">
            <div class="timestamp">
                Report generated on {timestamp} by Trade Likelihood Estimator
            </div>
            <div style="margin-top: 10px; color: {self.dark_theme['text_secondary']};">
                🤖 Powered by quantitative analysis and geometric Brownian motion
            </div>
        </div>
        """
    
    def generate_bidirectional_report(self, long_analysis, short_analysis, bidirectional_setup, 
                                    actual_outcomes=None, output_path: str = None) -> str:
        """
        Generate comprehensive HTML report for bidirectional trading strategy
        
        Args:
            long_analysis: Dict[str, TradeAnalysis] for different historical windows (long side)
            short_analysis: Dict[str, TradeAnalysis] for different historical windows (short side)  
            bidirectional_setup: BidirectionalSetup object with trade parameters
            actual_outcomes: Optional dict with actual trade outcomes
            output_path: Optional custom output path
            
        Returns:
            str: Path to generated HTML file
        """
        
        # Generate filename
        if output_path is None:
            trade_date_str = bidirectional_setup.trade_date.strftime('%Y%m%d')
            output_path = f"bidirectional_{bidirectional_setup.symbol}_{trade_date_str}.html"
        
        print(f"📄 Generating bidirectional report: {output_path}")
        
        # Get best analysis windows
        best_long_window, best_long = self._get_best_analysis(long_analysis)
        best_short_window, best_short = self._get_best_analysis(short_analysis)
        
        # Calculate combined metrics
        long_entry_prob = best_long.get_combined_entry_probability() if best_long else 0
        short_entry_prob = best_short.get_combined_entry_probability() if best_short else 0
        any_entry_prob = long_entry_prob + short_entry_prob - (long_entry_prob * short_entry_prob)
        both_entries_prob = long_entry_prob * short_entry_prob
        
        long_ev = best_long.expected_value_total if best_long else 0
        short_ev = best_short.expected_value_total if best_short else 0
        total_ev = (long_ev or 0) + (short_ev or 0)
        
        long_return = best_long.return_rate_total if best_long else 0
        short_return = best_short.return_rate_total if best_short else 0
        total_return = (long_return or 0) + (short_return or 0)
        
        # Build HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bidirectional Strategy Analysis - {bidirectional_setup.symbol}</title>
    {self._get_css_styles()}
</head>
<body>
    <div class="container">
        {self._generate_bidirectional_header(bidirectional_setup)}
        {self._generate_market_context_section(best_long, best_short)}
        {self._generate_trade_setups_section(bidirectional_setup)}
        {self._generate_probability_metrics_section(long_entry_prob, short_entry_prob, any_entry_prob, both_entries_prob)}
        {self._generate_performance_analysis_section(best_long, best_short, total_ev, total_return)}
        {self._generate_detailed_setup_analysis_section(long_analysis, short_analysis)}
        {self._generate_sensitivity_analysis_section(long_analysis, short_analysis)}
        {self._generate_strategy_assessment_section(bidirectional_setup, total_return, any_entry_prob, total_ev)}
        {self._generate_bidirectional_recommendations(bidirectional_setup, best_long, best_short, total_return)}
        {self._generate_footer()}
    </div>
</body>
</html>
        """
        
        # Write to file  
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Bidirectional report saved: {output_path}")
        return output_path
    
    def _get_best_analysis(self, analyses: Dict[str, Any]) -> tuple:
        """Get the best analysis window based on return rate"""
        if not analyses:
            return None, None
            
        best_window = max(analyses.keys(), 
                         key=lambda k: analyses[k].return_rate_total or -999)
        return best_window, analyses[best_window]
    
    def _generate_bidirectional_header(self, setup) -> str:
        """Generate header section for bidirectional report"""
        return f"""
        <div class="header">
            <h1>🎯 Bidirectional Trading Strategy Analysis</h1>
            <div class="subtitle">
                <span class="symbol">{setup.symbol}</span>
                <span class="date">Trade Date: {setup.trade_date.strftime('%B %d, %Y')}</span>
            </div>
            <div class="description">
                Comprehensive analysis of simultaneous long and short setups designed to capture volatility in either direction
            </div>
        </div>
        """
    
    def _generate_market_context_section(self, best_long, best_short) -> str:
        """Generate market context section"""
        regime = best_long.market_params.regime if best_long else "Unknown"
        drift = best_long.market_params.drift if best_long else 0
        volatility = best_long.market_params.volatility if best_long else 0
        
        return f"""
        <div class="section">
            <h2>📊 Market Context</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Market Regime</div>
                    <div class="metric-value regime-{regime.lower()}">{regime.title()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Estimated Drift</div>
                    <div class="metric-value {'positive' if drift > 0 else 'negative'}">{drift:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{volatility:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{'High' if best_long and best_long.market_params.is_high_confidence() else 'Moderate'}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_trade_setups_section(self, setup) -> str:
        """Generate trade setups comparison section"""
        return f"""
        <div class="section">
            <h2>⚡ Trade Setups</h2>
            <div class="setups-grid">
                <div class="setup-card long-setup">
                    <h3>🔺 Long Setup</h3>
                    <div class="setup-details">
                        <div class="setup-row">
                            <span>Current Price</span>
                            <span class="price">${setup.current_price:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Entry</span>
                            <span class="price entry">${setup.long_entry:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Stop Loss</span>
                            <span class="price stop">${setup.long_stop:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Target</span>
                            <span class="price target">${setup.long_target:.2f}</span>
                        </div>
                        <div class="setup-row highlight">
                            <span>Risk:Reward</span>
                            <span class="rr-ratio">{setup.get_long_rr():.2f}:1</span>
                        </div>
                    </div>
                </div>
                
                <div class="setup-card short-setup">
                    <h3>🔻 Short Setup</h3>
                    <div class="setup-details">
                        <div class="setup-row">
                            <span>Current Price</span>
                            <span class="price">${setup.current_price:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Entry</span>
                            <span class="price entry">${setup.short_entry:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Stop Loss</span>
                            <span class="price stop">${setup.short_stop:.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Target</span>
                            <span class="price target">${setup.short_target:.2f}</span>
                        </div>
                        <div class="setup-row highlight">
                            <span>Risk:Reward</span>
                            <span class="rr-ratio">{setup.get_short_rr():.2f}:1</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_probability_metrics_section(self, long_entry_prob, short_entry_prob, 
                                            any_entry_prob, both_entries_prob) -> str:
        """Generate probability metrics section"""
        return f"""
        <div class="section">
            <h2>📈 Entry & Execution Probabilities</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Long Entry Probability</div>
                    <div class="metric-value probability">{long_entry_prob:.1%}</div>
                    <div class="metric-note">Chance long setup triggers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Short Entry Probability</div>
                    <div class="metric-value probability">{short_entry_prob:.1%}</div>
                    <div class="metric-note">Chance short setup triggers</div>
                </div>
                <div class="metric-card highlight">
                    <div class="metric-label">Any Entry Probability</div>
                    <div class="metric-value probability">{any_entry_prob:.1%}</div>
                    <div class="metric-note">Chance at least one triggers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Both Entries Probability</div>
                    <div class="metric-value probability">{both_entries_prob:.1%}</div>
                    <div class="metric-note">Chance both trigger</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_performance_analysis_section(self, best_long, best_short, total_ev, total_return) -> str:
        """Generate performance analysis section"""
        long_win_prob = best_long.prob_win_long if best_long else 0
        short_win_prob = best_short.prob_win_short if best_short else 0
        long_ev = best_long.expected_value_total if best_long else 0
        short_ev = best_short.expected_value_total if best_short else 0
        long_return_rate = best_long.return_rate_total if best_long else 0
        short_return_rate = best_short.return_rate_total if best_short else 0
        
        return f"""
        <div class="section">
            <h2>💰 Performance Analysis</h2>
            <div class="performance-grid">
                <div class="performance-section">
                    <h3>Win Probabilities</h3>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <span>Long Win</span>
                            <span class="value">{(long_win_prob or 0):.1%}</span>
                        </div>
                        <div class="mini-metric">
                            <span>Short Win</span>
                            <span class="value">{(short_win_prob or 0):.1%}</span>
                        </div>
                    </div>
                </div>
                
                <div class="performance-section">
                    <h3>Expected Values</h3>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <span>Long EV</span>
                            <span class="value {'negative' if (long_ev or 0) < 0 else ''}">${(long_ev or 0):.2f}</span>
                        </div>
                        <div class="mini-metric">
                            <span>Short EV</span>
                            <span class="value {'negative' if (short_ev or 0) < 0 else ''}">${(short_ev or 0):.2f}</span>
                        </div>
                        <div class="mini-metric highlight">
                            <span>Total EV</span>
                            <span class="value {'negative' if total_ev < 0 else ''}">${total_ev:.2f}</span>
                        </div>
                    </div>
                </div>
                
                <div class="performance-section">
                    <h3>Return Rates</h3>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <span>Long Rate</span>
                            <span class="value {'negative' if (long_return_rate or 0) < 0 else ''}">{(long_return_rate or 0):.4f}/day</span>
                        </div>
                        <div class="mini-metric">
                            <span>Short Rate</span>
                            <span class="value {'negative' if (short_return_rate or 0) < 0 else ''}">{(short_return_rate or 0):.4f}/day</span>
                        </div>
                        <div class="mini-metric highlight">
                            <span>Total Rate</span>
                            <span class="value {'negative' if total_return < 0 else ''}">{total_return:.4f}/day</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_sensitivity_analysis_section(self, long_analysis, short_analysis) -> str:
        """Generate historical window sensitivity analysis"""
        sensitivity_html = ""
        
        all_windows = set(long_analysis.keys()) | set(short_analysis.keys())
        
        for window in sorted(all_windows, key=lambda x: int(x.replace('d', ''))):
            long_data = long_analysis.get(window)
            short_data = short_analysis.get(window)
            
            long_return = long_data.return_rate_total if long_data else 0
            short_return = short_data.return_rate_total if short_data else 0
            combined_return = (long_return or 0) + (short_return or 0)
            
            sensitivity_html += f"""
            <div class="sensitivity-row">
                <div class="window-label">{window} Window</div>
                <div class="window-metrics">
                    <span class="metric">Long: {(long_return or 0):.4f}</span>
                    <span class="metric">Short: {(short_return or 0):.4f}</span>
                    <span class="metric combined {'negative' if combined_return < 0 else ''}">Combined: {combined_return:.4f}</span>
                </div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>🔍 Historical Window Sensitivity</h2>
            <div class="sensitivity-analysis">
                <div class="sensitivity-header">
                    <span>Analysis Window</span>
                    <span>Return Rate Comparison (per day)</span>
                </div>
                {sensitivity_html}
            </div>
        </div>
        """

    def _generate_detailed_setup_analysis_section(self, long_analysis, short_analysis) -> str:
        """Generate detailed setup analysis table with timing metrics"""
        
        table_rows = ""
        all_windows = set(long_analysis.keys()) | set(short_analysis.keys())
        
        for window in sorted(all_windows, key=lambda x: int(x.replace('d', ''))):
            long_data = long_analysis.get(window)
            short_data = short_analysis.get(window)
            
            # Get metrics for long and short sides
            long_return = long_data.return_rate_total if long_data else 0
            short_return = short_data.return_rate_total if short_data else 0
            combined_return = (long_return or 0) + (short_return or 0)
            
            # Entry probabilities
            long_entry_prob = 0
            short_entry_prob = 0
            if long_data:
                long_entry_prob = long_data.get_combined_entry_probability() or (long_data.prob_entry_long or 0)
            if short_data:
                short_entry_prob = short_data.get_combined_entry_probability() or (short_data.prob_entry_short or 0)
            combined_entry_prob = long_entry_prob + short_entry_prob - (long_entry_prob * short_entry_prob)
            
            # Win probabilities
            long_win = (long_data.prob_win_long or 0) if long_data else 0
            short_win = (short_data.prob_win_short or 0) if short_data else 0
            combined_win = (long_win + short_win) / 2 if (long_win > 0 or short_win > 0) else 0
            
            # Expected timing metrics (convert from days to readable format)
            expected_entry_time = "N/A"
            expected_duration = "N/A"
            
            if long_data and (long_data.expected_entry_time_long or long_data.expected_entry_time_short):
                entry_times = [t for t in [long_data.expected_entry_time_long, long_data.expected_entry_time_short] if t is not None]
                if entry_times:
                    avg_entry_time = sum(entry_times) / len(entry_times)
                    expected_entry_time = f"{avg_entry_time:.1f} days"
                    
            if long_data and (long_data.expected_trade_duration_long or long_data.expected_trade_duration_short):
                durations = [d for d in [long_data.expected_trade_duration_long, long_data.expected_trade_duration_short] if d is not None]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    expected_duration = f"{avg_duration:.1f} days"
            
            # Determine best direction
            best_direction = "Long" if long_return > short_return else "Short" if short_return > 0 else "Neither"
            
            # Market regime
            regime = long_data.market_params.regime if long_data else (short_data.market_params.regime if short_data else "Unknown")
            regime_display = regime.title() if regime else "Unknown"
            
            # Attractive setup
            attractive = combined_return > 0.05  # 5% threshold
            attractive_badge = f'<span class="status-badge {"attractive" if attractive else "not-attractive"}">{"Attractive" if attractive else "Not Attractive"}</span>'
            
            # Color coding for return rates
            return_class = "positive" if combined_return > 0 else "negative" if combined_return < 0 else "neutral"
            
            table_rows += f"""
            <tr>
                <td><strong>{window}</strong></td>
                <td class="{return_class}"><strong>{combined_return:.4f}</strong></td>
                <td>{combined_entry_prob:.1%}</td>
                <td>{combined_win:.1%}</td>
                <td>{expected_entry_time}</td>
                <td>{expected_duration}</td>
                <td><strong>{best_direction}</strong></td>
                <td>{regime_display}</td>
                <td>{attractive_badge}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>🤖 Detailed Setup Analysis</h2>
            <p style="margin-bottom: 25px; color: {self.dark_theme['text_secondary']};">
                Comprehensive analysis across different historical data windows showing timing metrics and model predictions.
            </p>
            
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Historical Window</th>
                        <th>Return Rate/Day</th>
                        <th>Entry Probability</th>
                        <th>Win Probability</th>
                        <th>Expected Entry Time</th>
                        <th>Expected Trade Duration</th>
                        <th>Best Direction</th>
                        <th>Market Regime</th>
                        <th>Setup Attractiveness</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_strategy_assessment_section(self, setup, total_return, any_entry_prob, total_ev) -> str:
        """Generate overall strategy assessment"""
        avg_rr = (setup.get_long_rr() + setup.get_short_rr()) / 2
        is_attractive = total_return > 0.05 and any_entry_prob > 0.3
        risk_level = "High" if total_return < 0 else ("Moderate" if total_return < 0.05 else "Low")
        
        return f"""
        <div class="section">
            <h2>✅ Strategy Assessment</h2>
            <div class="assessment-grid">
                <div class="assessment-card {'attractive' if is_attractive else 'not-attractive'}">
                    <h3>Overall Attractiveness</h3>
                    <div class="assessment-value">{('✅ Attractive' if is_attractive else '❌ Not Attractive')}</div>
                    <div class="assessment-reason">
                        {'Strong return rate and high entry probability' if is_attractive else 'Low return rate or poor entry probability'}
                    </div>
                </div>
                
                <div class="assessment-card">
                    <h3>Risk Profile</h3>
                    <div class="assessment-value risk-{risk_level.lower()}">{risk_level} Risk</div>
                    <div class="assessment-reason">
                        Based on return volatility and expected values
                    </div>
                </div>
                
                <div class="assessment-card">
                    <h3>Setup Quality</h3>
                    <div class="assessment-value">{avg_rr:.2f}:1 Avg R:R</div>
                    <div class="assessment-reason">
                        {'Excellent' if avg_rr > 3 else ('Good' if avg_rr > 2 else 'Fair')} risk-reward structure
                    </div>
                </div>
                
                <div class="assessment-card">
                    <h3>Execution Certainty</h3>
                    <div class="assessment-value">{any_entry_prob:.1%} Entry Chance</div>
                    <div class="assessment-reason">
                        {'High' if any_entry_prob > 70 else ('Moderate' if any_entry_prob > 40 else 'Low')} probability of execution
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_bidirectional_recommendations(self, setup, best_long, best_short, total_return) -> str:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if total_return > 0.05:
            recommendations.append("Strategy shows strong positive expected returns - consider deployment")
        elif total_return > 0:
            recommendations.append("Strategy shows modest positive returns - monitor for better opportunities")
        else:
            recommendations.append("Strategy shows negative expected returns - avoid or wait for better market conditions")
        
        # Setup-specific recommendations
        if best_long and best_short:
            long_better = (best_long.return_rate_total or 0) > (best_short.return_rate_total or 0)
            if long_better:
                recommendations.append("Long setup significantly outperforms short - consider long-only approach")
            else:
                recommendations.append("Short setup outperforms long - consider short-bias or short-only approach")
        
        # Risk management
        avg_rr = (setup.get_long_rr() + setup.get_short_rr()) / 2
        if avg_rr > 4:
            recommendations.append("Excellent risk-reward ratios provide good downside protection")
        elif avg_rr < 2:
            recommendations.append("Low risk-reward ratios require high win rates - consider setup adjustment")
        
        rec_html = ""
        for rec in recommendations:
            rec_html += f"<li>{rec}</li>"
        
        return f"""
        <div class="section">
            <h2>💡 Strategic Recommendations</h2>
            <div class="recommendations">
                <div class="recommendation-category">
                    <h3>Key Insights</h3>
                    <ul>{rec_html}</ul>
                </div>
                
                <div class="recommendation-category">
                    <h3>Next Steps</h3>
                    <ul>
                        <li>Backtest this strategy on historical data to validate model predictions</li>
                        <li>Consider paper trading smaller position sizes to test execution</li>
                        <li>Monitor market regime changes that might affect strategy performance</li>
                        <li>Compare with single-direction setups to assess bidirectional benefit</li>
                    </ul>
                </div>
            </div>
        </div>
        """