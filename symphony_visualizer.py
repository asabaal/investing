"""
Symphony Backtest Visualizer

Creates comprehensive visualizations for symphony backtest results including:
- Performance charts
- Drawdown analysis  
- Allocation evolution
- Risk metrics
- Benchmark comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

class SymphonyVisualizer:
    """Create visualizations for symphony backtest results"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style ('seaborn-v0_8', 'ggplot', 'classic')
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            print(f"Warning: Style '{style}' not available, using default")
        
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_performance_dashboard(self, results: pd.DataFrame, analysis: Dict, 
                                  benchmark_data: pd.DataFrame = None, 
                                  save_path: str = None) -> None:
        """
        Create comprehensive performance dashboard
        
        Args:
            results: Backtest results DataFrame
            analysis: Performance analysis dictionary  
            benchmark_data: Optional benchmark comparison data
            save_path: Path to save the dashboard image
        """
        
        print("📊 Creating performance dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Calculate cumulative returns
        results = results.copy()
        results['cumulative_return'] = (1 + results['portfolio_return']).cumprod() - 1
        results['date'] = pd.to_datetime(results['date'])
        
        # 1. Cumulative Returns (top left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_cumulative_returns(ax1, results, benchmark_data)
        
        # 2. Drawdown Chart (top middle)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_drawdown(ax2, results)
        
        # 3. Monthly Returns Heatmap (top right)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_monthly_returns_heatmap(ax3, results)
        
        # 4. Allocation Evolution (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_allocation_evolution(ax4, results)
        
        # 5. Performance Metrics (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_performance_metrics(ax5, analysis, benchmark_data)
        
        # 6. Return Distribution (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        self._plot_return_distribution(ax6, results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dashboard saved: {save_path}")
        
        plt.show()
    
    def _plot_cumulative_returns(self, ax, results: pd.DataFrame, benchmark_data: pd.DataFrame = None):
        """Plot cumulative returns over time"""
        
        ax.plot(results['date'], results['cumulative_return'] * 100, 
                linewidth=2, label='Symphony Strategy', color=self.colors[0])
        
        if benchmark_data is not None:
            benchmark_data = benchmark_data.copy()
            benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod() - 1
            ax.plot(benchmark_data['date'], benchmark_data['cumulative_return'] * 100,
                   linewidth=2, label='Benchmark', color=self.colors[1], alpha=0.7)
        
        ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_drawdown(self, ax, results: pd.DataFrame):
        """Plot drawdown over time"""
        
        cumulative_returns = 1 + results['cumulative_return']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        ax.fill_between(results['date'], drawdown, 0, alpha=0.3, color='red')
        ax.plot(results['date'], drawdown, color='darkred', linewidth=1)
        
        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add max drawdown annotation
        max_dd_idx = drawdown.idxmin()
        max_dd_date = results.loc[max_dd_idx, 'date']
        max_dd_value = drawdown.iloc[max_dd_idx]
        
        ax.annotate(f'Max DD: {max_dd_value:.1f}%', 
                   xy=(max_dd_date, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    def _plot_monthly_returns_heatmap(self, ax, results: pd.DataFrame):
        """Plot monthly returns as heatmap"""
        
        # Calculate monthly returns
        results_monthly = results.set_index('date').resample('M')['portfolio_return'].apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        if len(results_monthly) == 0:
            ax.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns', fontsize=12, fontweight='bold')
            return
        
        # Create pivot table for heatmap
        monthly_df = results_monthly.reset_index()
        monthly_df['year'] = monthly_df['date'].dt.year
        monthly_df['month'] = monthly_df['date'].dt.month
        
        pivot_table = monthly_df.pivot(index='year', columns='month', values='portfolio_return')
        
        # Plot heatmap
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels([f'{int(m):02d}' for m in pivot_table.columns])
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(int))
        
        ax.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Return (%)')
    
    def _plot_allocation_evolution(self, ax, results: pd.DataFrame):
        """Plot how allocations change over time"""
        
        # Extract allocations over time
        all_symbols = set()
        for allocations in results['allocation']:
            all_symbols.update(allocations.keys())
        
        allocation_data = []
        for _, row in results.iterrows():
            for symbol in all_symbols:
                allocation_data.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'weight': row['allocation'].get(symbol, 0)
                })
        
        allocation_df = pd.DataFrame(allocation_data)
        pivot_allocations = allocation_df.pivot(index='date', columns='symbol', values='weight').fillna(0)
        
        # Create stacked area chart
        ax.stackplot(pivot_allocations.index, *[pivot_allocations[col] for col in pivot_allocations.columns],
                    labels=pivot_allocations.columns, alpha=0.7)
        
        ax.set_title('Portfolio Allocation Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_metrics(self, ax, analysis: Dict, benchmark_data: pd.DataFrame = None):
        """Plot key performance metrics as bar chart"""
        
        metrics = {
            'Annual Return': analysis['annual_return'] * 100,
            'Volatility': analysis['volatility'] * 100,
            'Sharpe Ratio': analysis['sharpe_ratio'],
            'Max Drawdown': abs(analysis['max_drawdown']) * 100,
            'Win Rate': analysis['win_rate'] * 100
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=self.colors[:len(metrics)])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}{"%" if "Rate" in bar.get_label() or "Return" in bar.get_label() or "Volatility" in bar.get_label() or "Drawdown" in bar.get_label() else ""}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_return_distribution(self, ax, results: pd.DataFrame):
        """Plot distribution of returns"""
        
        returns = results['portfolio_return'] * 100
        
        # Histogram
        ax.hist(returns, bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        
        # Add statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
        ax.axvline(mean_return + std_return, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_return + std_return:.2f}%')
        ax.axvline(mean_return - std_return, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_return - std_return:.2f}%')
        
        ax.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_interactive_dashboard(self, results: pd.DataFrame, analysis: Dict,
                                   benchmark_data: pd.DataFrame = None,
                                   save_path: str = None) -> None:
        """Create interactive dashboard using Plotly"""
        
        print("📊 Creating interactive dashboard...")
        
        results = results.copy()
        results['cumulative_return'] = (1 + results['portfolio_return']).cumprod() - 1
        results['date'] = pd.to_datetime(results['date'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Drawdown', 
                          'Allocation Evolution', 'Monthly Performance',
                          'Risk-Return Scatter', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Cumulative Returns
        fig.add_trace(
            go.Scatter(x=results['date'], y=results['cumulative_return'] * 100,
                      mode='lines', name='Symphony Strategy',
                      line=dict(color=self.colors[0], width=2)),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            benchmark_data = benchmark_data.copy()
            benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod() - 1
            fig.add_trace(
                go.Scatter(x=benchmark_data['date'], y=benchmark_data['cumulative_return'] * 100,
                          mode='lines', name='Benchmark',
                          line=dict(color=self.colors[1], width=2)),
                row=1, col=1
            )
        
        # 2. Drawdown
        cumulative_returns = 1 + results['cumulative_return']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(x=results['date'], y=drawdown,
                      fill='tonexty', mode='lines', name='Drawdown',
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=1, col=2
        )
        
        # 3. Allocation Evolution (stacked area chart)
        all_symbols = set()
        for allocations in results['allocation']:
            all_symbols.update(allocations.keys())
        
        allocation_data = []
        for _, row in results.iterrows():
            for symbol in all_symbols:
                allocation_data.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'weight': row['allocation'].get(symbol, 0)
                })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        for i, symbol in enumerate(all_symbols):
            symbol_data = allocation_df[allocation_df['symbol'] == symbol]
            fig.add_trace(
                go.Scatter(x=symbol_data['date'], y=symbol_data['weight'],
                          mode='lines', stackgroup='one', name=symbol,
                          line=dict(color=self.colors[i % len(self.colors)])),
                row=2, col=1
            )
        
        # 4. Monthly Performance
        results_monthly = results.set_index('date').resample('M')['portfolio_return'].apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        fig.add_trace(
            go.Bar(x=results_monthly.index, y=results_monthly.values,
                   name='Monthly Returns', marker_color=self.colors[2]),
            row=3, col=1
        )
        
        # 5. Performance Metrics
        metrics = {
            'Annual Return': analysis['annual_return'] * 100,
            'Volatility': analysis['volatility'] * 100,
            'Sharpe Ratio': analysis['sharpe_ratio'],
            'Max Drawdown': abs(analysis['max_drawdown']) * 100,
            'Win Rate': analysis['win_rate'] * 100
        }
        
        fig.add_trace(
            go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                   name='Metrics', marker_color=self.colors[3]),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Symphony Performance Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Weight", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Interactive dashboard saved: {save_path}")
        
        fig.show()
    
    def create_allocation_sunburst(self, results: pd.DataFrame, save_path: str = None):
        """Create sunburst chart showing allocation patterns"""
        
        print("🌅 Creating allocation sunburst chart...")
        
        # Get the most recent allocation
        latest_allocation = results.iloc[-1]['allocation']
        latest_condition = results.iloc[-1]['triggered_condition']
        
        # Prepare data for sunburst
        labels = ['Portfolio'] + list(latest_allocation.keys())
        parents = [''] + ['Portfolio'] * len(latest_allocation)
        values = [1.0] + list(latest_allocation.values())
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title=f"Current Portfolio Allocation<br><sub>Triggered Condition: {latest_condition}</sub>",
            title_x=0.5,
            font_size=12
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Sunburst chart saved: {save_path}")
        
        fig.show()
    
    def create_rolling_metrics_chart(self, results: pd.DataFrame, window: int = 12,
                                   save_path: str = None):
        """Create rolling performance metrics chart"""
        
        print("📈 Creating rolling metrics chart...")
        
        results = results.copy()
        results['date'] = pd.to_datetime(results['date'])
        
        # Calculate rolling metrics
        results['rolling_return'] = results['portfolio_return'].rolling(window=window).mean() * 252
        results['rolling_volatility'] = results['portfolio_return'].rolling(window=window).std() * np.sqrt(252)
        results['rolling_sharpe'] = results['rolling_return'] / results['rolling_volatility']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{window}-Period Rolling Annual Return',
                          f'{window}-Period Rolling Volatility',
                          f'{window}-Period Rolling Sharpe Ratio'),
            vertical_spacing=0.08
        )
        
        # Rolling Return
        fig.add_trace(
            go.Scatter(x=results['date'], y=results['rolling_return'] * 100,
                      mode='lines', name='Rolling Return',
                      line=dict(color=self.colors[0], width=2)),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(x=results['date'], y=results['rolling_volatility'] * 100,
                      mode='lines', name='Rolling Volatility',
                      line=dict(color=self.colors[1], width=2)),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=results['date'], y=results['rolling_sharpe'],
                      mode='lines', name='Rolling Sharpe',
                      line=dict(color=self.colors[2], width=2)),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Rolling Performance Metrics",
            title_x=0.5,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Rolling metrics chart saved: {save_path}")
        
        fig.show()
    
    def export_all_charts(self, results: pd.DataFrame, analysis: Dict,
                         output_dir: str = "./backtest_charts",
                         benchmark_data: pd.DataFrame = None):
        """Export all visualization charts"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Exporting all charts to {output_dir}...")
        
        # Performance Dashboard
        self.create_performance_dashboard(
            results, analysis, benchmark_data,
            save_path=os.path.join(output_dir, "performance_dashboard.png")
        )
        
        # Interactive Dashboard
        self.create_interactive_dashboard(
            results, analysis, benchmark_data,
            save_path=os.path.join(output_dir, "interactive_dashboard.html")
        )
        
        # Allocation Sunburst
        self.create_allocation_sunburst(
            results,
            save_path=os.path.join(output_dir, "allocation_sunburst.html")
        )
        
        # Rolling Metrics
        self.create_rolling_metrics_chart(
            results,
            save_path=os.path.join(output_dir, "rolling_metrics.html")
        )
        
        print("✅ All charts exported successfully!")


# Benchmark data fetcher
def fetch_benchmark_data(symbol: str = 'SPY', start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
    """Fetch benchmark data for comparison"""
    
    try:
        from data_pipeline import MarketDataPipeline, DataConfig
        
        config = DataConfig(rate_limit_delay=1.0)
        pipeline = MarketDataPipeline(config)
        
        print(f"📊 Fetching benchmark data for {symbol}...")
        benchmark_raw = pipeline.get_daily_data(symbol, start_date, end_date)
        
        if benchmark_raw.empty:
            print(f"❌ No benchmark data available for {symbol}")
            return pd.DataFrame()
        
        # Calculate returns
        benchmark_data = pd.DataFrame({
            'date': benchmark_raw.index,
            'price': benchmark_raw['Close'],
            'return': benchmark_raw['Close'].pct_change().fillna(0)
        })
        
        print(f"✅ Fetched {len(benchmark_data)} days of {symbol} data")
        return benchmark_data
        
    except Exception as e:
        print(f"❌ Error fetching benchmark data: {e}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    
    print("🎨 Symphony Visualizer Test")
    print("=" * 40)
    
    # Create sample backtest data for testing
    dates = pd.date_range('2023-01-01', periods=50, freq='W')
    
    # Generate realistic backtest results
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.04, len(dates))  # Weekly returns
    
    sample_results = []
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for i, date in enumerate(dates):
        # Simulate changing allocations
        if i % 4 == 0:  # Monthly rebalancing
            weights = np.random.dirichlet([1, 1, 1])  # Random allocation
            allocation = {symbol: weight for symbol, weight in zip(symbols, weights)}
            condition = 'momentum_allocation' if np.random.random() > 0.3 else 'defensive_allocation'
        
        sample_results.append({
            'date': date,
            'portfolio_return': returns[i],
            'allocation': allocation,
            'triggered_condition': condition,
            'metrics': {'spy_momentum': np.random.normal(0.05, 0.1)}
        })
    
    sample_df = pd.DataFrame(sample_results)
    
    # Sample analysis
    cumulative_return = (1 + sample_df['portfolio_return']).prod() - 1
    sample_analysis = {
        'total_return': cumulative_return,
        'annual_return': (1 + cumulative_return) ** (52/len(sample_df)) - 1,
        'volatility': sample_df['portfolio_return'].std() * np.sqrt(52),
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'num_trades': len(sample_df)
    }
    
    # Test visualizer
    visualizer = SymphonyVisualizer()
    
    try:
        print("\n📊 Testing Performance Dashboard...")
        visualizer.create_performance_dashboard(sample_df, sample_analysis)
        
        print("\n🌐 Testing Interactive Dashboard...")
        visualizer.create_interactive_dashboard(sample_df, sample_analysis)
        
        print("\n🌅 Testing Allocation Sunburst...")
        visualizer.create_allocation_sunburst(sample_df)
        
        print("\n📈 Testing Rolling Metrics...")
        visualizer.create_rolling_metrics_chart(sample_df, window=8)
        
        print("\n✅ All visualization tests completed!")
        
    except Exception as e:
        print(f"❌ Error during visualization test: {e}")
        import traceback
        traceback.print_exc()
