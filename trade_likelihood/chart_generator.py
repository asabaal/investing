"""
Chart Generator for Trade Likelihood Analysis

Creates visual representations of:
- Expected price movements using geometric Brownian motion
- Entry/exit probability timelines 
- Monte Carlo simulations for validation
- Trade setup visualization with actual vs predicted outcomes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import base64
from io import BytesIO
import warnings

# Set dark theme for charts
plt.style.use('dark_background')
sns.set_palette("husl")

class TradeChartGenerator:
    """Generate comprehensive charts for trade likelihood analysis"""
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """
        Initialize chart generator
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Resolution for charts
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Dark theme colors
        self.colors = {
            'bg': '#0d1117',
            'grid': '#30363d',
            'text': '#f0f6fc',
            'accent_blue': '#58a6ff',
            'accent_green': '#3fb950',
            'accent_red': '#f85149',
            'accent_yellow': '#d29922',
            'entry': '#58a6ff',
            'stop': '#f85149',
            'target': '#3fb950',
            'current': '#d29922'
        }
    
    def generate_price_simulation_chart(self, 
                                       current_price: float,
                                       entry_price: float,
                                       stop_loss: float,
                                       target_price: float,
                                       drift: float,
                                       volatility: float,
                                       direction: str,
                                       days_ahead: int = 60,
                                       num_simulations: int = 1000) -> str:
        """
        Generate Monte Carlo price simulation chart showing expected paths
        
        Args:
            current_price: Starting price
            entry_price: Trade entry level
            stop_loss: Stop loss level  
            target_price: Target price level
            drift: Annual drift rate
            volatility: Annual volatility
            direction: 'long' or 'short'
            days_ahead: Number of days to simulate
            num_simulations: Number of Monte Carlo paths
            
        Returns:
            Base64 encoded PNG image
        """
        
        # Create figure with dark theme
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['bg'])
        
        # Time array - ensure consistent dimensions
        dates = [datetime.now() + timedelta(days=i) for i in range(days_ahead + 1)]
        
        # Generate Monte Carlo paths
        paths = self._simulate_gbm_paths(
            S0=current_price,
            mu=drift,
            sigma=volatility,
            T=days_ahead/365.25,
            num_steps=days_ahead,
            num_paths=num_simulations
        )
        
        # Plot 1: Price simulation with entry/exit levels
        ax1.set_facecolor(self.colors['bg'])
        
        # Plot sample paths (subset for visibility)
        sample_indices = np.random.choice(num_simulations, min(50, num_simulations), replace=False)
        for i in sample_indices:
            ax1.plot(dates, paths[i, :], alpha=0.1, color='lightblue', linewidth=0.5)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        ax1.plot(dates, mean_path, color=self.colors['accent_blue'], linewidth=2, label='Expected Path')
        
        # Plot percentile bands
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        
        ax1.fill_between(dates, p5, p95, alpha=0.2, color=self.colors['accent_blue'], label='90% Range')
        ax1.fill_between(dates, p25, p75, alpha=0.3, color=self.colors['accent_blue'], label='50% Range')
        
        # Add trade levels
        ax1.axhline(y=current_price, color=self.colors['current'], linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')
        ax1.axhline(y=entry_price, color=self.colors['entry'], linestyle='--', linewidth=2, label=f'Entry: ${entry_price:.2f}')
        ax1.axhline(y=stop_loss, color=self.colors['stop'], linestyle='--', linewidth=2, label=f'Stop: ${stop_loss:.2f}')
        ax1.axhline(y=target_price, color=self.colors['target'], linestyle='--', linewidth=2, label=f'Target: ${target_price:.2f}')
        
        ax1.set_title(f'Monte Carlo Price Simulation - {direction.upper()} Trade', 
                     color=self.colors['text'], fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', color=self.colors['text'])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Entry/Exit probability timeline
        ax2.set_facecolor(self.colors['bg'])
        
        entry_probs, exit_probs = self._calculate_probability_timeline(
            paths, entry_price, stop_loss, target_price, direction
        )
        
        ax2.plot(dates, entry_probs * 100, color=self.colors['entry'], linewidth=2, 
                label=f'Cumulative Entry Probability')
        ax2.plot(dates, exit_probs * 100, color=self.colors['accent_red'], linewidth=2, 
                label=f'Cumulative Exit Probability (given entry)')
        
        ax2.set_title('Entry and Exit Probability Timeline', 
                     color=self.colors['text'], fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', color=self.colors['text'])
        ax2.set_ylabel('Probability (%)', color=self.colors['text'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Style both axes
        for ax in [ax1, ax2]:
            ax.tick_params(colors=self.colors['text'])
            ax.spines['bottom'].set_color(self.colors['grid'])
            ax.spines['top'].set_color(self.colors['grid'])
            ax.spines['left'].set_color(self.colors['grid'])
            ax.spines['right'].set_color(self.colors['grid'])
        
        plt.tight_layout()
        
        # Convert to base64
        return self._fig_to_base64(fig)
    
    def generate_trade_outcome_validation_chart(self,
                                              backtest_result,
                                              num_simulations: int = 10000) -> str:
        """
        Generate validation chart comparing mathematical predictions vs Monte Carlo results
        
        Args:
            backtest_result: BacktestResult object with predictions and actual outcome
            num_simulations: Number of Monte Carlo simulations for validation
            
        Returns:
            Base64 encoded PNG image  
        """
        
        # Get trade details
        trade = backtest_result.trade
        predictions = backtest_result.model_predictions
        
        if not predictions:
            return self._create_no_data_chart("No model predictions available")
        
        # Use best prediction window
        best_window = max(predictions.keys(), key=lambda k: predictions[k].return_rate_total or -999)
        best_prediction = predictions[best_window]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['bg'])
        fig.suptitle(f'Trade Outcome Validation: {trade.symbol} {trade.direction.upper()}', 
                    color=self.colors['text'], fontsize=16, fontweight='bold')
        
        # Get market parameters
        market_params = best_prediction.market_params
        
        # Determine trade levels
        if trade.direction == 'long':
            entry = trade.entry_price
            stop = trade.stop_loss  
            target = trade.target_price
        else:
            entry = trade.entry_price
            stop = trade.stop_loss
            target = trade.target_price
        
        # Current price (estimate from trade date)
        current_price = entry - 5 if trade.direction == 'short' else entry - 5  # Rough estimate
        
        # Run Monte Carlo validation
        validation_results = self._run_monte_carlo_validation(
            current_price=current_price,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            drift=market_params.drift,
            volatility=market_params.volatility,
            direction=trade.direction,
            max_days=30,  # Reasonable timeframe
            num_simulations=num_simulations
        )
        
        # Chart 1: Entry probability comparison
        ax1 = axes[0, 0]
        ax1.set_facecolor(self.colors['bg'])
        
        model_entry_prob = best_prediction.get_combined_entry_probability() or 0
        mc_entry_prob = validation_results['entry_probability']
        
        categories = ['Mathematical\nModel', 'Monte Carlo\nSimulation']
        values = [model_entry_prob * 100, mc_entry_prob * 100]
        colors = [self.colors['accent_blue'], self.colors['accent_green']]
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_title('Entry Probability Comparison', color=self.colors['text'], fontweight='bold')
        ax1.set_ylabel('Probability (%)', color=self.colors['text'])
        ax1.tick_params(colors=self.colors['text'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', color=self.colors['text'])
        
        # Chart 2: Win probability comparison  
        ax2 = axes[0, 1]
        ax2.set_facecolor(self.colors['bg'])
        
        if trade.direction == 'long':
            model_win_prob = best_prediction.prob_win_long or 0
        else:
            model_win_prob = best_prediction.prob_win_short or 0
            
        mc_win_prob = validation_results['win_probability']
        
        values = [model_win_prob * 100, mc_win_prob * 100]
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.8)
        ax2.set_title('Win Probability Comparison', color=self.colors['text'], fontweight='bold')
        ax2.set_ylabel('Probability (%)', color=self.colors['text'])
        ax2.tick_params(colors=self.colors['text'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', color=self.colors['text'])
        
        # Chart 3: Expected timing comparison
        ax3 = axes[1, 0]
        ax3.set_facecolor(self.colors['bg'])
        
        if trade.direction == 'long':
            model_entry_time = best_prediction.expected_entry_time_long or 0
            model_trade_duration = best_prediction.expected_trade_duration_long or 0
        else:
            model_entry_time = best_prediction.expected_entry_time_short or 0
            model_trade_duration = best_prediction.expected_trade_duration_short or 0
        
        mc_entry_time = validation_results['expected_entry_time']
        mc_trade_duration = validation_results['expected_trade_duration']
        
        metrics = ['Entry Time\n(days)', 'Trade Duration\n(days)']
        model_values = [model_entry_time, model_trade_duration]
        mc_values = [mc_entry_time, mc_trade_duration]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, model_values, width, label='Mathematical Model', 
                       color=self.colors['accent_blue'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, mc_values, width, label='Monte Carlo Simulation',
                       color=self.colors['accent_green'], alpha=0.8)
        
        ax3.set_title('Expected Timing Comparison', color=self.colors['text'], fontweight='bold')
        ax3.set_ylabel('Days', color=self.colors['text'])
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.tick_params(colors=self.colors['text'])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}', ha='center', va='bottom', color=self.colors['text'])
        
        # Chart 4: Distribution of outcomes
        ax4 = axes[1, 1]
        ax4.set_facecolor(self.colors['bg'])
        
        outcomes = validation_results['outcomes']
        outcome_counts = {
            'Win': outcomes.count('win'),
            'Loss': outcomes.count('loss'), 
            'No Entry': outcomes.count('no_entry'),
            'Timeout': outcomes.count('timeout')
        }
        
        # Filter out zero counts for cleaner chart
        outcome_counts = {k: v for k, v in outcome_counts.items() if v > 0}
        
        labels = list(outcome_counts.keys())
        sizes = list(outcome_counts.values())
        colors_pie = [self.colors['accent_green'] if l == 'Win' 
                     else self.colors['accent_red'] if l == 'Loss'
                     else self.colors['accent_yellow'] if l == 'No Entry'
                     else self.colors['accent_blue'] for l in labels]
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90, textprops={'color': self.colors['text']})
        ax4.set_title('Monte Carlo Outcome Distribution', color=self.colors['text'], fontweight='bold')
        
        # Style all axes
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _simulate_gbm_paths(self, S0: float, mu: float, sigma: float, 
                           T: float, num_steps: int, num_paths: int) -> np.ndarray:
        """Simulate Geometric Brownian Motion paths"""
        
        dt = T / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        
        for i in range(1, num_steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    def _calculate_probability_timeline(self, paths: np.ndarray, entry_price: float,
                                      stop_loss: float, target_price: float, 
                                      direction: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cumulative entry and exit probabilities over time"""
        
        num_paths, num_steps = paths.shape
        entry_probs = np.zeros(num_steps)
        exit_probs = np.zeros(num_steps)
        
        for t in range(num_steps):
            # Entry probability: fraction of paths that have hit entry by time t
            if direction == 'long':
                entries = np.any(paths[:, :t+1] >= entry_price, axis=1)
            else:  # short
                entries = np.any(paths[:, :t+1] <= entry_price, axis=1)
            
            entry_probs[t] = np.mean(entries)
            
            # Exit probability: of paths that entered, fraction that hit stop or target
            if entry_probs[t] > 0:
                entered_paths = paths[entries, :t+1]
                if len(entered_paths) > 0:
                    if direction == 'long':
                        exits = np.any((entered_paths <= stop_loss) | (entered_paths >= target_price), axis=1)
                    else:  # short  
                        exits = np.any((entered_paths >= stop_loss) | (entered_paths <= target_price), axis=1)
                    
                    exit_probs[t] = np.mean(exits)
        
        return entry_probs, exit_probs
    
    def _run_monte_carlo_validation(self, current_price: float, entry_price: float,
                                   stop_loss: float, target_price: float,
                                   drift: float, volatility: float, direction: str,
                                   max_days: int, num_simulations: int) -> Dict[str, Any]:
        """Run Monte Carlo simulation to validate mathematical calculations"""
        
        # Generate paths
        paths = self._simulate_gbm_paths(
            S0=current_price,
            mu=drift,
            sigma=volatility,
            T=max_days/365.25,
            num_steps=max_days,
            num_paths=num_simulations
        )
        
        # Analyze each path
        entry_times = []
        trade_durations = []
        outcomes = []
        
        for path in paths:
            path_result = self._analyze_single_path(
                path, entry_price, stop_loss, target_price, direction, max_days
            )
            
            if path_result['entered']:
                entry_times.append(path_result['entry_time'])
                if path_result['outcome'] in ['win', 'loss']:
                    trade_durations.append(path_result['trade_duration'])
                    
            outcomes.append(path_result['outcome'])
        
        # Calculate statistics
        entry_probability = len(entry_times) / num_simulations
        
        if entry_times:
            expected_entry_time = np.mean(entry_times)
            win_outcomes = [o for o in outcomes if o == 'win']
            win_probability = len(win_outcomes) / len(entry_times) if entry_times else 0
        else:
            expected_entry_time = 0
            win_probability = 0
            
        if trade_durations:
            expected_trade_duration = np.mean(trade_durations)
        else:
            expected_trade_duration = 0
        
        return {
            'entry_probability': entry_probability,
            'win_probability': win_probability,
            'expected_entry_time': expected_entry_time,
            'expected_trade_duration': expected_trade_duration,
            'outcomes': outcomes
        }
    
    def _analyze_single_path(self, path: np.ndarray, entry_price: float,
                           stop_loss: float, target_price: float, 
                           direction: str, max_days: int) -> Dict[str, Any]:
        """Analyze a single price path for entry/exit events"""
        
        entered = False
        entry_time = None
        outcome = 'timeout'
        trade_duration = None
        
        for day, price in enumerate(path):
            if not entered:
                # Check for entry
                if direction == 'long' and price >= entry_price:
                    entered = True
                    entry_time = day
                elif direction == 'short' and price <= entry_price:
                    entered = True
                    entry_time = day
            else:
                # Check for exit
                if direction == 'long':
                    if price <= stop_loss:
                        outcome = 'loss'
                        trade_duration = day - entry_time
                        break
                    elif price >= target_price:
                        outcome = 'win'
                        trade_duration = day - entry_time
                        break
                else:  # short
                    if price >= stop_loss:
                        outcome = 'loss'
                        trade_duration = day - entry_time
                        break
                    elif price <= target_price:
                        outcome = 'win'
                        trade_duration = day - entry_time
                        break
        
        if not entered:
            outcome = 'no_entry'
        
        return {
            'entered': entered,
            'entry_time': entry_time,
            'outcome': outcome,
            'trade_duration': trade_duration
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', facecolor=self.colors['bg'], 
                   edgecolor='none', bbox_inches='tight', dpi=self.dpi)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_str
    
    def _create_no_data_chart(self, message: str) -> str:
        """Create a simple chart with a message when data is unavailable"""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        ax.text(0.5, 0.5, message, ha='center', va='center',
               color=self.colors['text'], fontsize=14, 
               transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return self._fig_to_base64(fig)


def quick_chart_test():
    """Quick test of chart generation"""
    
    print("🎨 Testing chart generation...")
    
    generator = TradeChartGenerator()
    
    # DPZ short trade parameters
    current_price = 458.30
    entry_price = 464.80
    stop_loss = 468.46
    target_price = 441.47
    drift = -0.05  # -5% annual drift
    volatility = 0.25  # 25% annual volatility
    direction = 'short'
    
    print("📊 Generating Monte Carlo simulation chart...")
    chart_b64 = generator.generate_price_simulation_chart(
        current_price=current_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        drift=drift,
        volatility=volatility,
        direction=direction,
        days_ahead=30,
        num_simulations=1000
    )
    
    print(f"✅ Chart generated: {len(chart_b64)} characters")
    return chart_b64


if __name__ == "__main__":
    quick_chart_test()