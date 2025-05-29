#!/usr/bin/env python3
"""
Market Data Admin Dashboard

A web-based interface for exploring and managing market data:
- Live SQL query execution
- Interactive visualizations  
- Database management tools
- Data exploration and analysis

Run with: python market_data_dashboard.py
Then visit: http://localhost:5000
"""

import os
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, Response
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from market_data_database import MarketDataDatabase

app = Flask(__name__)
app.secret_key = 'market_data_dashboard_2025'

# Initialize database connection
db = MarketDataDatabase()

@app.route('/')
def index():
    """Main dashboard page"""
    stats = db.get_database_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/api/stats')
def api_stats():
    """Get database statistics"""
    return jsonify(db.get_database_stats())

@app.route('/api/query', methods=['POST'])
def api_query():
    """Execute SQL query safely"""
    try:
        query = request.json.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Safety checks
        if not _is_safe_query(query):
            return jsonify({'error': 'Query contains potentially unsafe operations'}), 400
        
        # Execute query
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get results
            results = cursor.fetchall()
            
            return jsonify({
                'columns': columns,
                'data': results,
                'row_count': len(results)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbols')
def api_symbols():
    """Get all symbols with data info"""
    try:
        with sqlite3.connect(db.db_path) as conn:
            # Get symbol info from both tables (SQLite-compatible)
            # First get all symbols from both tables
            symbols_query = """
            SELECT DISTINCT symbol FROM (
                SELECT symbol FROM daily_data
                UNION
                SELECT symbol FROM intraday_data
            ) ORDER BY symbol
            """
            
            symbols = pd.read_sql_query(symbols_query, conn)['symbol'].tolist()
            
            # Get detailed info for each symbol
            symbol_info = []
            for symbol in symbols:
                # Daily data info
                daily_info = conn.execute(
                    'SELECT COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest FROM daily_data WHERE symbol = ?',
                    [symbol]
                ).fetchone()
                
                # Intraday data info  
                intraday_info = conn.execute(
                    'SELECT COUNT(*) as count, MIN(datetime) as earliest, MAX(datetime) as latest FROM intraday_data WHERE symbol = ?',
                    [symbol]
                ).fetchone()
                
                symbol_info.append({
                    'symbol': symbol,
                    'has_daily': 1 if daily_info[0] > 0 else 0,
                    'has_intraday': 1 if intraday_info[0] > 0 else 0,
                    'daily_count': daily_info[0] or 0,
                    'intraday_count': intraday_info[0] or 0,
                    'earliest_date': daily_info[1] or intraday_info[1],
                    'latest_date': daily_info[2] or intraday_info[2]
                })
            
            return jsonify(symbol_info)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Generate price chart for symbol"""
    try:
        interval = request.args.get('interval', 'daily')
        days = int(request.args.get('days', 30))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get data using unified system
        data = db.get_data(
            symbol, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval=interval
        )
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
        
        fig.update_layout(
            title=f'{symbol} - {interval.title()} Data ({days} days)',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        return Response(
            json.dumps(fig, cls=PlotlyJSONEncoder),
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comparison')
def api_comparison():
    """Compare multiple symbols"""
    try:
        symbols = request.args.get('symbols', '').split(',')
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        days = int(request.args.get('days', 30))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        fig = go.Figure()
        
        for symbol in symbols:
            try:
                data = db.get_data(
                    symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval='daily'
                )
                
                if not data.empty:
                    # Calculate normalized returns (percentage change from first day)
                    normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
                    
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                continue
        
        fig.update_layout(
            title=f'Symbol Comparison - Normalized Returns ({days} days)',
            yaxis_title='Return (%)',
            xaxis_title='Date',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return Response(
            json.dumps(fig, cls=PlotlyJSONEncoder),
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heatmap')
def api_heatmap():
    """Generate performance heatmap"""
    try:
        days = int(request.args.get('days', 7))
        
        # Get recent performance for top symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'TLT']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        performance_data = []
        
        for symbol in symbols:
            try:
                data = db.get_data(
                    symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval='daily'
                )
                
                if len(data) >= 2:
                    return_pct = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    performance_data.append({
                        'symbol': symbol,
                        'return': return_pct,
                        'volume': data['Volume'].sum() if 'Volume' in data else 0
                    })
                    
            except Exception as e:
                print(f"Error calculating performance for {symbol}: {e}")
                continue
        
        if not performance_data:
            return jsonify({'error': 'No performance data available'}), 404
        
        df = pd.DataFrame(performance_data)
        
        # Create heatmap-style bar chart
        fig = px.bar(
            df.sort_values('return', ascending=True),
            x='return',
            y='symbol',
            orientation='h',
            color='return',
            color_continuous_scale='RdYlGn',
            title=f'Performance Heatmap ({days} days)',
            labels={'return': 'Return (%)', 'symbol': 'Symbol'}
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return Response(
            json.dumps(fig, cls=PlotlyJSONEncoder),
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _is_safe_query(query):
    """Check if SQL query is safe to execute"""
    query_lower = query.lower().strip()
    
    # Allow only SELECT queries
    if not query_lower.startswith('select'):
        return False
    
    # Block dangerous keywords
    dangerous_keywords = [
        'drop', 'delete', 'insert', 'update', 'alter', 'create',
        'truncate', 'replace', 'merge', 'exec', 'execute'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            return False
    
    return True

if __name__ == '__main__':
    print("🚀 Starting Market Data Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("💡 Use Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)