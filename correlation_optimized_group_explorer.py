#!/usr/bin/env python3
"""
CORRELATION-OPTIMIZED GROUP EXPLORER

Interactive visualization tool to explore individual groups from the correlation-optimized portfolio.
Features:
- Group selector dropdown
- Securities overview table  
- Candlestick chart with scroll wheel zoom and click/drag pan
- Correlation matrix heatmap
- Performance metrics
- Dark mode design
"""

import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from market_data_database import get_default_database_path
import json
import warnings
warnings.filterwarnings('ignore')

class CorrelationOptimizedGroupExplorer:
    def __init__(self, groups_csv_path, db_path):
        """Initialize with correlation-optimized groups data"""
        self.groups_csv_path = groups_csv_path
        self.db_path = db_path
        self.groups_df = None
        self.conn = None
        
    def load_groups_data(self):
        """Load correlation-optimized groups"""
        print("Loading correlation-optimized groups...")
        
        self.groups_df = pd.read_csv(self.groups_csv_path)
        
        # Connect to database
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
            
        print(f"✅ Loaded {len(self.groups_df)} securities across {self.groups_df['group_number'].nunique()} groups")
        return True
    
    def get_group_data(self, group_number):
        """Get all data for a specific group"""
        group_securities = self.groups_df[self.groups_df['group_number'] == group_number].copy()
        
        if group_securities.empty:
            return None
            
        # Get price data for all securities in the group
        tickers = group_securities['ticker'].tolist()
        
        # Get price data for correlation calculation (30 days to match optimization)
        end_date = datetime.now()
        correlation_start_date = end_date - timedelta(days=30)
        
        # Get all available price data for charting
        placeholders = ','.join(['?' for _ in tickers])
        query_all = f"""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        ORDER BY symbol, datetime
        """
        
        # Get recent data for correlation (15-minute data, matching optimization exactly)
        query_corr = f"""
        SELECT symbol, datetime as timestamp, close 
        FROM intraday_data 
        WHERE symbol IN ({placeholders})
        AND DATE(datetime) >= DATE(?)
        ORDER BY symbol, datetime
        """
        
        try:
            params = tickers
            correlation_params = tickers + [correlation_start_date.strftime('%Y-%m-%d')]
            
            # Get full data for charting
            price_data_all = pd.read_sql_query(query_all, self.conn, params=params)
            daily_data = self._aggregate_to_daily(price_data_all)
            
            # Get 30-day 15-minute data for correlations (matching optimization exactly)
            price_data_corr = pd.read_sql_query(query_corr, self.conn, params=correlation_params)
            
            # Calculate correlations using 15-minute returns (matching optimization exactly)
            correlation_matrix = self._calculate_15min_correlations(price_data_corr)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(daily_data, group_securities)
            
            return {
                'group_info': group_securities,
                'daily_data': daily_data,
                'correlation_matrix': correlation_matrix,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            print(f"Error loading group {group_number} data: {e}")
            return None
    
    def _aggregate_to_daily(self, price_data):
        """Aggregate 15-minute data to daily OHLCV"""
        if price_data.empty:
            return pd.DataFrame()
            
        price_data['date'] = pd.to_datetime(price_data['datetime']).dt.date
        
        daily_data = price_data.groupby(['symbol', 'date']).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        return daily_data.sort_values(['symbol', 'date'])
    
    def _calculate_group_correlations(self, daily_data):
        """Calculate correlation matrix for the group using daily data"""
        if daily_data.empty:
            return pd.DataFrame()
            
        # Pivot to get returns matrix
        price_pivot = daily_data.pivot(index='date', columns='symbol', values='close')
        returns = price_pivot.pct_change().dropna()
        
        if returns.empty:
            return pd.DataFrame()
            
        return returns.corr()
    
    def _calculate_15min_correlations(self, price_data):
        """Calculate correlation matrix using 15-minute returns (matching optimization exactly)"""
        if price_data.empty:
            return pd.DataFrame()
        
        # Clean and pivot data (matching optimization exactly)
        price_data_clean = price_data.drop_duplicates(subset=['timestamp', 'symbol'])
        price_pivot = price_data_clean.pivot(index='timestamp', columns='symbol', values='close')
        
        # Calculate returns (matching optimization exactly)  
        returns = price_pivot.pct_change().dropna()
        
        if returns.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def _calculate_performance_metrics(self, daily_data, group_securities):
        """Calculate performance metrics for each security"""
        metrics = []
        
        for _, security in group_securities.iterrows():
            ticker = security['ticker']
            security_data = daily_data[daily_data['symbol'] == ticker].copy()
            
            if security_data.empty:
                continue
                
            security_data = security_data.sort_values('date')
            
            # Calculate metrics
            start_price = security_data['close'].iloc[0] if len(security_data) > 0 else 0
            end_price = security_data['close'].iloc[-1] if len(security_data) > 0 else 0
            
            total_return = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
            
            # Calculate volatility (std of daily returns)
            returns = security_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
            
            # Average volume
            avg_volume = security_data['volume'].mean()
            
            metrics.append({
                'ticker': ticker,
                'company': security['company'],
                'sector': security['sector'],
                'total_score': security['total_score'],
                'total_return': total_return,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'current_price': end_price
            })
        
        return pd.DataFrame(metrics)
    
    def create_group_explorer(self, output_file='correlation_optimized_group_explorer.html'):
        """Create interactive group explorer"""
        print("Creating interactive group explorer...")
        
        if not self.load_groups_data():
            return
        
        unique_groups = sorted(self.groups_df['group_number'].unique())
        
        # Create the HTML with embedded JavaScript
        html_content = self._generate_explorer_html(unique_groups)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"✅ Interactive Group Explorer saved: {output_file}")
        return output_file
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    def _generate_explorer_html(self, unique_groups):
        """Generate HTML with embedded Plotly and group data"""
        
        # Prepare group data for all groups
        all_group_data = {}
        
        print("Loading data for all groups...")
        for group_num in unique_groups:
            group_data = self.get_group_data(group_num)
            if group_data:
                all_group_data[str(group_num)] = {
                    'securities': self._convert_numpy_types(group_data['group_info'].to_dict('records')),
                    'daily_data': self._convert_numpy_types(group_data['daily_data'].to_dict('records')),
                    'correlation_matrix': self._convert_numpy_types(group_data['correlation_matrix'].to_dict() if not group_data['correlation_matrix'].empty else {}),
                    'performance_metrics': self._convert_numpy_types(group_data['performance_metrics'].to_dict('records'))
                }
        
        html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Correlation-Optimized Group Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1e1e1e;
            color: white;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .controls {{
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .group-selector {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        select {{
            padding: 10px;
            background-color: #404040;
            color: white;
            border: 1px solid #555;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .chart-container {{
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 15px;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .securities-table {{
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #404040;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #555;
        }}
        
        th {{
            background-color: #333;
            font-weight: bold;
        }}
        
        .metric-positive {{ color: #4CAF50; }}
        .metric-negative {{ color: #f44336; }}
        .metric-neutral {{ color: #FFC107; }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #888;
        }}
        
        .group-stats {{
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            justify-content: space-around;
            text-align: center;
        }}
        
        .stat-item {{
            flex: 1;
        }}
        
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Correlation-Optimized Group Explorer</h1>
        <p>Interactive analysis of correlation-optimized trading groups</p>
    </div>
    
    <div class="controls">
        <div class="group-selector">
            <label for="groupSelect"><strong>Select Group:</strong></label>
            <select id="groupSelect" onchange="updateGroup()">
                <option value="">Choose a group...</option>
                {self._generate_group_options(unique_groups)}
            </select>
        </div>
        <div class="group-selector">
            <label for="securitySelect"><strong>Select Security for Chart:</strong></label>
            <select id="securitySelect" onchange="updateSingleCandlestickChart()" disabled>
                <option value="">Choose a security...</option>
            </select>
        </div>
        <div id="groupInfo" style="flex: 1; margin-left: 20px;"></div>
    </div>
    
    <div id="groupStats" class="group-stats" style="display: none;">
        <div class="stat-item">
            <div class="stat-value" id="statSecurities">-</div>
            <div>Securities</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="statSectors">-</div>
            <div>Sectors</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="statMaxCorr">-</div>
            <div>Max Correlation</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="statAvgScore">-</div>
            <div>Avg Score</div>
        </div>
    </div>
    
    <div id="mainContent" style="display: none;">
        <div class="securities-table">
            <h3>📊 Securities in Group</h3>
            <div id="securitiesTable"></div>
        </div>
        
        <div class="dashboard">
            <div class="chart-container">
                <h3 id="candlestickTitle">📈 Price Charts (Candlestick)</h3>
                <div id="candlestickChart"></div>
            </div>
            
            <div class="chart-container">
                <h3>🔥 Correlation Matrix (30-Day Period)</h3>
                <div id="correlationHeatmap"></div>
            </div>
        </div>
        
        <div class="chart-container full-width">
            <h3>📊 Performance Comparison</h3>
            <div id="performanceChart"></div>
        </div>
    </div>
    
    <div id="loadingMessage" class="loading">
        <h3>Select a group to begin exploration...</h3>
    </div>

    <script>
        // Embed group data
        const groupData = {json.dumps(all_group_data, default=str)};
        
        function updateGroup() {{
            const groupSelect = document.getElementById('groupSelect');
            const securitySelect = document.getElementById('securitySelect');
            const selectedGroup = groupSelect.value;
            
            if (!selectedGroup || !groupData[selectedGroup]) {{
                document.getElementById('mainContent').style.display = 'none';
                document.getElementById('groupStats').style.display = 'none';
                document.getElementById('loadingMessage').style.display = 'block';
                securitySelect.disabled = true;
                securitySelect.innerHTML = '<option value="">Choose a security...</option>';
                return;
            }}
            
            const data = groupData[selectedGroup];
            
            // Update security selector
            updateSecuritySelector(data.securities);
            
            // Update group info
            updateGroupInfo(selectedGroup, data);
            
            // Show main content
            document.getElementById('mainContent').style.display = 'block';
            document.getElementById('groupStats').style.display = 'flex';
            document.getElementById('loadingMessage').style.display = 'none';
            
            // Update all visualizations
            updateSecuritiesTable(data.securities);
            updateCandlestickChart(data.daily_data, data.securities);
            updateCorrelationHeatmap(data.correlation_matrix);
            updatePerformanceChart(data.performance_metrics);
        }}
        
        function updateSecuritySelector(securities) {{
            const securitySelect = document.getElementById('securitySelect');
            
            // Clear and populate security options
            securitySelect.innerHTML = '<option value="">All Securities</option>';
            
            securities.forEach(security => {{
                const option = document.createElement('option');
                option.value = security.ticker;
                option.textContent = `${{security.ticker}} - ${{security.company.substring(0, 30)}}${{security.company.length > 30 ? '...' : ''}}`;
                securitySelect.appendChild(option);
            }});
            
            securitySelect.disabled = false;
        }}
        
        function updateGroupInfo(groupNumber, data) {{
            const securities = data.securities;
            const sectors = [...new Set(securities.map(s => s.sector))];
            const avgScore = securities.reduce((sum, s) => sum + s.total_score, 0) / securities.length;
            
            // Calculate max correlation
            const corrMatrix = data.correlation_matrix;
            let maxCorr = 0;
            if (corrMatrix && Object.keys(corrMatrix).length > 0) {{
                Object.keys(corrMatrix).forEach(ticker1 => {{
                    Object.keys(corrMatrix[ticker1]).forEach(ticker2 => {{
                        if (ticker1 !== ticker2) {{
                            maxCorr = Math.max(maxCorr, Math.abs(corrMatrix[ticker1][ticker2] || 0));
                        }}
                    }});
                }});
            }}
            
            document.getElementById('groupInfo').innerHTML = 
                `<strong>Group ${{groupNumber}}</strong> - ${{securities.length}} securities from ${{sectors.length}} sectors`;
            
            document.getElementById('statSecurities').textContent = securities.length;
            document.getElementById('statSectors').textContent = sectors.length;
            document.getElementById('statMaxCorr').textContent = (maxCorr * 100).toFixed(1) + '%';
            document.getElementById('statAvgScore').textContent = avgScore.toFixed(0);
        }}
        
        function updateSecuritiesTable(securities) {{
            let tableHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Company</th>
                            <th>Sector</th>
                            <th>Score</th>
                            <th>Selection Reason</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            securities.forEach(security => {{
                tableHTML += `
                    <tr>
                        <td><strong>${{security.ticker}}</strong></td>
                        <td>${{security.company.substring(0, 30)}}${{security.company.length > 30 ? '...' : ''}}</td>
                        <td>${{security.sector}}</td>
                        <td class="metric-positive">${{security.total_score.toFixed(0)}}</td>
                        <td>${{security.selection_reason}}</td>
                    </tr>
                `;
            }});
            
            tableHTML += '</tbody></table>';
            document.getElementById('securitiesTable').innerHTML = tableHTML;
        }}
        
        function updateCandlestickChart(dailyData, securities) {{
            const traces = [];
            const tickers = securities.map(s => s.ticker);
            
            tickers.forEach(ticker => {{
                const tickerData = dailyData.filter(d => d.symbol === ticker);
                
                if (tickerData.length === 0) return;
                
                const trace = {{
                    type: 'candlestick',
                    x: tickerData.map(d => d.date),
                    open: tickerData.map(d => d.open),
                    high: tickerData.map(d => d.high),
                    low: tickerData.map(d => d.low),
                    close: tickerData.map(d => d.close),
                    name: ticker,
                    visible: traces.length < 3 ? true : 'legendonly'  // Show first 3 by default
                }};
                
                traces.push(trace);
            }});
            
            const layout = {{
                height: 400,
                paper_bgcolor: '#2d2d2d',
                plot_bgcolor: '#404040',
                font: {{ color: 'white' }},
                xaxis: {{
                    gridcolor: '#555',
                    showgrid: true,
                    type: 'date'
                }},
                yaxis: {{
                    gridcolor: '#555',
                    showgrid: true,
                    title: 'Price ($)'
                }},
                legend: {{
                    orientation: 'h',
                    y: -0.2
                }},
                margin: {{ l: 50, r: 50, t: 30, b: 60 }}
            }};
            
            const config = {{
                scrollZoom: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['select2d', 'lasso2d']
            }};
            
            Plotly.newPlot('candlestickChart', traces, layout, config);
        }}
        
        function updateCorrelationHeatmap(correlationMatrix) {{
            if (!correlationMatrix || Object.keys(correlationMatrix).length === 0) {{
                document.getElementById('correlationHeatmap').innerHTML = 
                    '<p style="text-align: center; color: #888;">No correlation data available</p>';
                return;
            }}
            
            const tickers = Object.keys(correlationMatrix);
            const zValues = [];
            
            tickers.forEach(ticker1 => {{
                const row = [];
                tickers.forEach(ticker2 => {{
                    row.push(correlationMatrix[ticker1][ticker2] || 0);
                }});
                zValues.push(row);
            }});
            
            const trace = {{
                type: 'heatmap',
                x: tickers,
                y: tickers,
                z: zValues,
                colorscale: 'RdYlBu',
                reversescale: true,
                zmin: -1,
                zmax: 1,
                texttemplate: "%{{z:.2f}}",
                textfont: {{ color: 'white', size: 10 }},
                showscale: true,
                colorbar: {{
                    title: 'Correlation',
                    titlefont: {{ color: 'white' }},
                    tickfont: {{ color: 'white' }}
                }}
            }};
            
            const layout = {{
                height: 400,
                paper_bgcolor: '#2d2d2d',
                plot_bgcolor: '#404040',
                font: {{ color: 'white' }},
                xaxis: {{ 
                    tickfont: {{ size: 10 }},
                    side: 'bottom'
                }},
                yaxis: {{ 
                    tickfont: {{ size: 10 }}
                }},
                margin: {{ l: 60, r: 60, t: 30, b: 60 }}
            }};
            
            const config = {{
                displayModeBar: false
            }};
            
            Plotly.newPlot('correlationHeatmap', [trace], layout, config);
        }}
        
        function updatePerformanceChart(performanceMetrics) {{
            if (!performanceMetrics || performanceMetrics.length === 0) {{
                document.getElementById('performanceChart').innerHTML = 
                    '<p style="text-align: center; color: #888;">No performance data available</p>';
                return;
            }}
            
            const tickers = performanceMetrics.map(m => m.ticker);
            
            const trace1 = {{
                type: 'bar',
                x: tickers,
                y: performanceMetrics.map(m => m.total_return),
                name: 'Total Return (%)',
                marker: {{ 
                    color: performanceMetrics.map(m => m.total_return >= 0 ? '#4CAF50' : '#f44336')
                }},
                yaxis: 'y'
            }};
            
            const trace2 = {{
                type: 'scatter',
                mode: 'markers+lines',
                x: tickers,
                y: performanceMetrics.map(m => m.volatility),
                name: 'Volatility (%)',
                marker: {{ color: '#FFC107', size: 8 }},
                yaxis: 'y2'
            }};
            
            const layout = {{
                height: 400,
                paper_bgcolor: '#2d2d2d',
                plot_bgcolor: '#404040',
                font: {{ color: 'white' }},
                xaxis: {{
                    gridcolor: '#555',
                    tickfont: {{ size: 10 }}
                }},
                yaxis: {{
                    title: 'Return (%)',
                    gridcolor: '#555',
                    side: 'left'
                }},
                yaxis2: {{
                    title: 'Volatility (%)',
                    side: 'right',
                    overlaying: 'y',
                    gridcolor: '#555'
                }},
                legend: {{
                    orientation: 'h',
                    y: -0.2
                }},
                margin: {{ l: 50, r: 50, t: 30, b: 80 }}
            }};
            
            const config = {{
                displayModeBar: false
            }};
            
            Plotly.newPlot('performanceChart', [trace1, trace2], layout, config);
        }}
        
        function updateSingleCandlestickChart() {{
            const groupSelect = document.getElementById('groupSelect');
            const securitySelect = document.getElementById('securitySelect');
            const selectedGroup = groupSelect.value;
            const selectedSecurity = securitySelect.value;
            
            if (!selectedGroup || !groupData[selectedGroup]) {{
                return;
            }}
            
            const data = groupData[selectedGroup];
            
            if (!selectedSecurity) {{
                // Show all securities if no specific security selected
                document.getElementById('candlestickTitle').textContent = '📈 Price Charts (All Securities)';
                updateCandlestickChart(data.daily_data, data.securities);
                return;
            }}
            
            // Filter data for selected security only
            const filteredSecurities = data.securities.filter(s => s.ticker === selectedSecurity);
            
            if (filteredSecurities.length === 0) {{
                return;
            }}
            
            const tickerData = data.daily_data.filter(d => d.symbol === selectedSecurity);
            
            if (tickerData.length === 0) {{
                document.getElementById('candlestickChart').innerHTML = 
                    '<p style="text-align: center; color: #888;">No price data available for this security</p>';
                return;
            }}
            
            const trace = {{
                type: 'candlestick',
                x: tickerData.map(d => d.date),
                open: tickerData.map(d => d.open),
                high: tickerData.map(d => d.high),
                low: tickerData.map(d => d.low),
                close: tickerData.map(d => d.close),
                name: selectedSecurity,
                increasing: {{ line: {{ color: '#4CAF50' }} }},
                decreasing: {{ line: {{ color: '#f44336' }} }}
            }};
            
            // Update chart title
            document.getElementById('candlestickTitle').textContent = `📈 ${{selectedSecurity}} - ${{filteredSecurities[0].company.substring(0, 40)}}${{filteredSecurities[0].company.length > 40 ? '...' : ''}}`;
            
            const layout = {{
                height: 400,
                paper_bgcolor: '#2d2d2d',
                plot_bgcolor: '#404040',
                font: {{ color: 'white' }},
                xaxis: {{
                    gridcolor: '#555',
                    showgrid: true,
                    type: 'date',
                    title: 'Date'
                }},
                yaxis: {{
                    gridcolor: '#555',
                    showgrid: true,
                    title: 'Price ($)'
                }},
                showlegend: false,
                margin: {{ l: 50, r: 50, t: 50, b: 50 }}
            }};
            
            const config = {{
                scrollZoom: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['select2d', 'lasso2d']
            }};
            
            Plotly.newPlot('candlestickChart', [trace], layout, config);
        }}
    </script>
</body>
</html>
        '''
        
        return html_template
    
    def _generate_group_options(self, unique_groups):
        """Generate HTML options for group selector"""
        options = []
        
        for group_num in unique_groups:
            group_data = self.groups_df[self.groups_df['group_number'] == group_num]
            securities_count = len(group_data)
            sectors_count = group_data['sector'].nunique()
            avg_score = group_data['total_score'].mean()
            max_corr = group_data['estimated_max_correlation'].iloc[0] if 'estimated_max_correlation' in group_data.columns else 0
            
            option_text = f"Group {group_num} - {securities_count} securities, {sectors_count} sectors, {avg_score:.0f} avg score, {max_corr:.1%} max corr"
            options.append(f'<option value="{group_num}">{option_text}</option>')
        
        return '\n'.join(options)

def main():
    """Create the correlation-optimized group explorer"""
    groups_csv_path = "/home/asabaal/asabaal_ventures/repos/investing/correlation_optimized_trading_groups.csv"
    db_path = get_default_database_path()
    
    explorer = CorrelationOptimizedGroupExplorer(groups_csv_path, db_path)
    explorer.create_group_explorer()

if __name__ == "__main__":
    main()