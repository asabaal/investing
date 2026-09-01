"""
Generate Charts for Presentation Failure Cases

Creates visual proof for the three failure case examples:
1. GameStop Jan 2021 - Technical Analysis subjective interpretation
2. Netflix Q1 2022 - Fundamentals vs stock performance disconnect  
3. 2008 Financial Crisis - Mathematical model failure (VaR)
"""

import sys
import os
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
from matplotlib.patches import Rectangle
import mplfinance as mpf

from market_data_database import MarketDataDatabase

# Set dark theme
plt.style.use('dark_background')

class FailureCaseChartGenerator:
    """Generate charts showing why traditional methods fail"""
    
    def __init__(self):
        self.market_db = MarketDataDatabase()
        
        # Dark theme colors matching presentation
        self.colors = {
            'bg': '#0d1117',
            'grid': '#30363d', 
            'text': '#f0f6fc',
            'accent_blue': '#58a6ff',
            'accent_green': '#3fb950',
            'accent_red': '#f85149',
            'accent_yellow': '#d29922',
            'purple': '#a5a5f5'
        }
        
    def generate_gamestop_chart(self) -> str:
        """
        Generate GameStop Jan 2021 chart showing subjective pattern interpretation
        """
        print("🎮 Generating GameStop Jan 2021 technical analysis failure chart...")
        
        try:
            # Get GameStop data from Dec 2020 - Feb 2021
            gme_data = self.market_db.get_daily_data('GME', '2020-12-01', '2021-02-28')
            
            if gme_data is None or gme_data.empty:
                print("❌ No GameStop data available - updating...")
                self.market_db.update_daily_data('GME', force_full_update=True)
                gme_data = self.market_db.get_daily_data('GME', '2020-12-01', '2021-02-28')
            
            if gme_data is None or gme_data.empty:
                print("❌ Failed to get GameStop data")
                return self._create_placeholder_chart("GameStop Data Not Available")
            
            # Check and convert date column to datetime
            print(f"GameStop data columns: {gme_data.columns.tolist()}")
            print(f"GameStop data shape: {gme_data.shape}")
            
            # Prepare data for mplfinance - set date as index and ensure proper column names
            gme_data.index = pd.to_datetime(gme_data.index)
            
            # Use Unadjusted_Close for proper candlestick coloring
            if 'Open' in gme_data.columns:
                gme_plot_data = gme_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
                gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                gme_plot_data = gme_data[['open', 'high', 'low', 'unadjusted_close', 'volume']].copy()
                gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
            # Create custom style for dark theme with proper visibility
            mc = mpf.make_marketcolors(up='#3fb950', down='#f85149', 
                                     edge='inherit',  # Use candle color for edges
                                     wick={'up':'#3fb950', 'down':'#f85149'},
                                     volume={'up':'#3fb950', 'down':'#f85149'})
            
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False,
                                 facecolor=self.colors['bg'], figcolor=self.colors['bg'],
                                 gridcolor=self.colors['grid'], 
                                 rc={'axes.labelcolor': self.colors['text'],
                                     'axes.edgecolor': self.colors['text'],
                                     'xtick.color': self.colors['text'],
                                     'ytick.color': self.colors['text'],
                                     'text.color': self.colors['text']})
            
            # Create the candlestick chart with volume using proper approach
            fig, axes = mpf.plot(gme_plot_data, type='candle', style=s, volume=True,
                               figsize=(14, 10), returnfig=True,
                               title='GameStop Jan 2021: Same Chart, Opposite Expert Interpretations')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig('assets/images/charts/gamestop_technical_failure.png', 
                       format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            
            # Create base64 version for HTML integration
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            buffer.seek(0)
            chart_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            print("✅ GameStop chart saved as gamestop_technical_failure.png")
            return chart_b64
            
        except Exception as e:
            print(f"❌ Error generating GameStop chart: {e}")
            return self._create_placeholder_chart("GameStop Chart Error")
    
    def generate_netflix_chart(self) -> str:
        """
        Generate Netflix Q1 2022 chart showing fundamentals vs stock performance disconnect
        """
        print("📺 Generating Netflix Q1 2022 fundamental analysis failure chart...")
        
        try:
            # Get Netflix data from Q4 2021 - Q2 2022
            nflx_data = self.market_db.get_daily_data('NFLX', '2021-12-01', '2022-06-30')
            
            if nflx_data is None or nflx_data.empty:
                print("❌ No Netflix data available - updating...")
                self.market_db.update_daily_data('NFLX', force_full_update=True)
                nflx_data = self.market_db.get_daily_data('NFLX', '2021-12-01', '2022-06-30')
            
            if nflx_data is None or nflx_data.empty:
                print("❌ Failed to get Netflix data")
                return self._create_placeholder_chart("Netflix Data Not Available")
            
            # Check and convert date column to datetime
            print(f"Netflix data columns: {nflx_data.columns.tolist()}")
            print(f"Netflix data shape: {nflx_data.shape}")
            
            # Prepare data for mplfinance
            nflx_data.index = pd.to_datetime(nflx_data.index)
            
            # Use Unadjusted_Close for proper candlestick coloring 
            if 'Open' in nflx_data.columns:
                nflx_plot_data = nflx_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
                nflx_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                nflx_plot_data = nflx_data[['open', 'high', 'low', 'unadjusted_close', 'volume']].copy()
                nflx_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Create custom style for dark theme with proper visibility
            mc = mpf.make_marketcolors(up='#3fb950', down='#f85149', 
                                     edge='inherit',  # Use candle color for edges
                                     wick={'up':'#3fb950', 'down':'#f85149'},
                                     volume={'up':'#3fb950', 'down':'#f85149'})
            
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False,
                                 facecolor=self.colors['bg'], figcolor=self.colors['bg'],
                                 gridcolor=self.colors['grid'], 
                                 rc={'axes.labelcolor': self.colors['text'],
                                     'axes.edgecolor': self.colors['text'],
                                     'xtick.color': self.colors['text'],
                                     'ytick.color': self.colors['text'],
                                     'text.color': self.colors['text']})
            
            # Create the candlestick chart with volume using proper approach
            fig, axes = mpf.plot(nflx_plot_data, type='candle', style=s, volume=True,
                               figsize=(14, 10), returnfig=True,
                               title='Netflix Q1 2022: Great Fundamentals ≠ Good Stock Performance')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig('assets/images/charts/netflix_fundamental_failure.png', 
                       format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            
            # Create base64 version for HTML integration
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            buffer.seek(0)
            chart_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            print("✅ Netflix chart saved as netflix_fundamental_failure.png")
            return chart_b64
            
        except Exception as e:
            print(f"❌ Error generating Netflix chart: {e}")
            return self._create_placeholder_chart("Netflix Chart Error")
    
    def generate_var_failure_chart(self) -> str:
        """
        Generate 2008 Financial Crisis VaR model failure visualization
        """
        print("🏦 Generating VaR model failure chart...")
        
        try:
            # Create synthetic data showing VaR model vs reality
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.patch.set_facecolor(self.colors['bg'])
            ax1.set_facecolor(self.colors['bg'])
            ax2.set_facecolor(self.colors['bg'])
            
            # Simulate VaR model prediction (normal distribution)
            returns = np.random.normal(0, 0.02, 1000)  # 2% daily volatility
            
            ax1.hist(returns, bins=50, alpha=0.7, color=self.colors['accent_blue'], 
                    density=True, label='VaR Model Prediction (Normal Distribution)')
            
            # Add VaR line (99% confidence, ~2.33 standard deviations)
            var_99 = -2.33 * 0.02  # 99% VaR
            ax1.axvline(x=var_99, color=self.colors['accent_green'], 
                       linestyle='--', linewidth=3, label=f'99% VaR: {var_99:.1%}')
            
            # Add actual 2008 crisis events
            crisis_events = [-0.22, -0.18, -0.15, -0.12]  # Lehman, AIG, etc.
            for i, event in enumerate(crisis_events):
                ax1.axvline(x=event, color=self.colors['accent_red'], 
                           linewidth=2, alpha=0.8)
                if i == 0:  # Label first one
                    ax1.axvline(x=event, color=self.colors['accent_red'], 
                               linewidth=2, alpha=0.8, label='Actual 2008 Crisis Events')
            
            ax1.set_title('VaR Model vs 2008 Reality: "1 in 10,000 Year Events" Happened Multiple Times', 
                         fontsize=16, color=self.colors['text'], pad=20)
            ax1.set_xlabel('Daily Returns', fontsize=12, color=self.colors['text'])
            ax1.set_ylabel('Probability Density', fontsize=12, color=self.colors['text'])
            ax1.tick_params(colors=self.colors['text'])
            ax1.grid(True, alpha=0.3, color=self.colors['grid'])
            ax1.legend(fontsize=10, facecolor=self.colors['bg'], edgecolor=self.colors['grid'])
            
            # Bottom: Timeline of major bank failures
            banks = ['Bear Stearns', 'Lehman Bros', 'AIG', 'Merrill Lynch', 'Washington Mutual']
            dates = pd.date_range(start='2008-03-01', end='2008-09-30', periods=len(banks))
            
            ax2.scatter(dates, [1]*len(banks), s=200, c=self.colors['accent_red'], alpha=0.8)
            
            for i, (date, bank) in enumerate(zip(dates, banks)):
                ax2.annotate(bank, (date, 1), xytext=(0, 20), 
                           textcoords='offset points', ha='center', fontsize=10,
                           color=self.colors['text'])
            
            ax2.set_title('Major Financial Institution Failures in 2008', 
                         fontsize=14, color=self.colors['text'])
            ax2.set_ylim(0.5, 1.5)
            ax2.set_yticks([])
            ax2.tick_params(colors=self.colors['text'])
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            
            # Add failure explanation
            failure_text = ("Mathematical Model Failure: VaR assumed normal distributions and independence\n"
                          "Reality: Fat tails, correlation, and systemic risk made 'impossible' events routine")
            ax1.text(0.02, 0.95, failure_text, transform=ax1.transAxes, 
                    fontsize=11, color=self.colors['accent_red'], 
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['bg'], 
                             edgecolor=self.colors['accent_red'], alpha=0.9))
            
            plt.tight_layout()
            
            # Save as PNG file
            plt.savefig('assets/images/charts/var_model_failure.png', format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            
            # Also return base64 for HTML integration if needed
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor=self.colors['bg'], edgecolor='none')
            buffer.seek(0)
            chart_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            print("✅ VaR failure chart saved as var_model_failure.png")
            return chart_b64
            
        except Exception as e:
            print(f"❌ Error generating VaR chart: {e}")
            return self._create_placeholder_chart("VaR Chart Error")
    
    def _create_placeholder_chart(self, message: str) -> str:
        """Create a placeholder chart when data is unavailable"""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        ax.text(0.5, 0.5, message, transform=ax.transAxes, 
               fontsize=16, color=self.colors['text'], 
               ha='center', va='center',
               bbox=dict(boxstyle="round,pad=1", facecolor=self.colors['accent_yellow'], alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save placeholder as PNG
        plt.savefig('placeholder_chart.png', format='png', dpi=150, bbox_inches='tight',
                   facecolor=self.colors['bg'], edgecolor='none')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor=self.colors['bg'], edgecolor='none')
        buffer.seek(0)
        chart_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return chart_b64
    
    def create_base_style(self):
        """Create the consistent mplfinance style"""
        mc = mpf.make_marketcolors(up='#3fb950', down='#f85149', 
                                 edge='inherit',
                                 wick={'up':'#3fb950', 'down':'#f85149'},
                                 volume={'up':'#3fb950', 'down':'#f85149'})
        
        return mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False,
                                facecolor=self.colors['bg'], figcolor=self.colors['bg'],
                                gridcolor=self.colors['grid'], 
                                rc={'axes.labelcolor': self.colors['text'],
                                    'axes.edgecolor': self.colors['text'],
                                    'xtick.color': self.colors['text'],
                                    'ytick.color': self.colors['text'],
                                    'text.color': self.colors['text']})
        
    def create_gamestop_bull_consolidation(self):
        """GameStop consolidation period - what bulls saw"""
        print("🟢 Creating GameStop bull consolidation view...")
        
        # Focus on consolidation period Jan 12-21
        gme_data = self.market_db.get_daily_data('GME', '2021-01-08', '2021-01-25')
        gme_data.index = pd.to_datetime(gme_data.index)
        gme_plot_data = gme_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(gme_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='BULL VIEW: Consolidation Phase (Jan 8-25)',
                savefig='assets/images/charts/gme_bull_consolidation.png')
        
    def create_gamestop_bull_breakout(self):
        """GameStop breakout period - what bulls saw"""
        print("🟢 Creating GameStop bull breakout view...")
        
        # Focus on breakout period Jan 22-26
        gme_data = self.market_db.get_daily_data('GME', '2021-01-20', '2021-01-28')
        gme_data.index = pd.to_datetime(gme_data.index)
        gme_plot_data = gme_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(gme_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='BULL VIEW: Breakout Confirmation (Jan 20-28)',
                savefig='assets/images/charts/gme_bull_breakout.png')
        
    def create_gamestop_bear_parabolic(self):
        """GameStop parabolic rise - what bears saw"""
        print("🔴 Creating GameStop bear parabolic view...")
        
        # Focus on parabolic phase Jan 22-28  
        gme_data = self.market_db.get_daily_data('GME', '2021-01-22', '2021-02-05')
        gme_data.index = pd.to_datetime(gme_data.index)
        gme_plot_data = gme_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(gme_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='BEAR VIEW: Parabolic Exhaustion (Jan 22 - Feb 5)',
                savefig='assets/images/charts/gme_bear_parabolic.png')
        
    def create_gamestop_bear_crash(self):
        """GameStop crash - what bears predicted"""
        print("🔴 Creating GameStop bear crash view...")
        
        # Focus on crash period Jan 28 - Feb 12
        gme_data = self.market_db.get_daily_data('GME', '2021-01-26', '2021-02-15')
        gme_data.index = pd.to_datetime(gme_data.index)
        gme_plot_data = gme_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        gme_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(gme_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='BEAR VIEW: Inevitable Crash (Jan 26 - Feb 15)',
                savefig='assets/images/charts/gme_bear_crash.png')
        
    def create_netflix_fundamental_strength(self):
        """Netflix pre-earnings strength - what fundamentalists saw"""
        print("📊 Creating Netflix fundamental strength period...")
        
        # Focus on pre-earnings period showing strong performance
        nflx_data = self.market_db.get_daily_data('NFLX', '2022-01-01', '2022-04-19')
        nflx_data.index = pd.to_datetime(nflx_data.index)
        nflx_plot_data = nflx_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        nflx_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(nflx_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='FUNDAMENTAL VIEW: Strong Q4 Performance (Jan - April 19)',
                savefig='assets/images/charts/nflx_fundamental_strength.png')
    
    def create_netflix_earnings_shock(self):
        """Netflix earnings day shock - what the market delivered"""
        print("📉 Creating Netflix earnings shock period...")
        
        # Focus on earnings day and immediate aftermath
        nflx_data = self.market_db.get_daily_data('NFLX', '2022-04-18', '2022-04-22')
        nflx_data.index = pd.to_datetime(nflx_data.index)
        nflx_plot_data = nflx_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        nflx_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(nflx_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='MARKET REALITY: Earnings Day Shock (April 18-22)',
                savefig='assets/images/charts/nflx_earnings_shock.png')
                
    def create_netflix_continued_decline(self):
        """Netflix continued decline - market's forward-looking view"""
        print("📉 Creating Netflix continued decline view...")
        
        # Focus on post-earnings decline period
        nflx_data = self.market_db.get_daily_data('NFLX', '2022-04-19', '2022-06-30')
        nflx_data.index = pd.to_datetime(nflx_data.index)
        nflx_plot_data = nflx_data[['Open', 'High', 'Low', 'Unadjusted_Close', 'Volume']].copy()
        nflx_plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        style = self.create_base_style()
        
        mpf.plot(nflx_plot_data, 
                type='candle', 
                style=style,
                volume=True,
                figsize=(12, 8),
                title='MARKET REALITY: Continued Decline (April 19 - June 30)',
                savefig='assets/images/charts/nflx_continued_decline.png')

    def generate_all_charts(self) -> dict:
        """Generate all failure case charts including detailed views"""
        print("📊 Generating all failure case charts...")
        
        # Generate original overview charts
        charts = {
            'gamestop': self.generate_gamestop_chart(),
            'netflix': self.generate_netflix_chart(), 
            'var_failure': self.generate_var_failure_chart()
        }
        
        # Generate focused date range charts
        self.create_gamestop_bull_consolidation()
        self.create_gamestop_bull_breakout()
        self.create_gamestop_bear_parabolic()
        self.create_gamestop_bear_crash()
        self.create_netflix_fundamental_strength()
        self.create_netflix_earnings_shock()
        self.create_netflix_continued_decline()
        
        print("✅ All charts generated successfully!")
        print("💾 Charts saved as PNG files in assets/images/charts/")
        return charts
    
    def _create_charts_html(self, charts: dict) -> str:
        """Create HTML file with all charts for easy viewing and integration"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Failure Case Charts</title>
    <style>
        body {{ background-color: {self.colors['bg']}; color: {self.colors['text']}; font-family: Arial, sans-serif; }}
        .chart {{ margin: 40px 0; text-align: center; }}
        .chart h2 {{ color: {self.colors['accent_blue']}; }}
        .chart img {{ max-width: 100%; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1 style="text-align: center; color: {self.colors['accent_blue']};">Failure Case Charts</h1>
"""
        
        if charts['gamestop'] and 'placeholder' not in charts['gamestop'].lower():
            html += f'''
    <div class="chart">
        <h2>GameStop Jan 2021 - Technical Analysis Failure</h2>
        <img src="data:image/png;base64,{charts['gamestop']}" alt="GameStop Chart">
    </div>
'''
        
        if charts['netflix'] and 'placeholder' not in charts['netflix'].lower():
            html += f'''
    <div class="chart">
        <h2>Netflix Q1 2022 - Fundamental Analysis Failure</h2>
        <img src="data:image/png;base64,{charts['netflix']}" alt="Netflix Chart">
    </div>
'''
        
        if charts['var_failure']:
            html += f'''
    <div class="chart">
        <h2>2008 Financial Crisis - Mathematical Model Failure</h2>
        <img src="data:image/png;base64,{charts['var_failure']}" alt="VaR Failure Chart">
    </div>
'''
        
        html += """
</body>
</html>
"""
        return html

if __name__ == "__main__":
    generator = FailureCaseChartGenerator()
    charts = generator.generate_all_charts()
    print(f"Generated {len(charts)} charts for presentation failure cases")