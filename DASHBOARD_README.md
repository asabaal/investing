# Market Data Admin Dashboard

A comprehensive web-based interface for exploring and managing your market data database.

## 🚀 Quick Start

```bash
# Install dependencies (if needed)
pip install flask plotly

# Launch dashboard
python launch_dashboard.py

# Or run directly
python market_data_dashboard.py
```

Visit: **http://localhost:5000**

## 📊 Features

### 1. **Database Overview**
- Real-time statistics and metrics
- Data breakdown visualizations
- Performance heatmaps
- Date coverage analysis

### 2. **SQL Query Explorer**
- Live SQL query execution
- Safety restrictions (SELECT only)
- Formatted result tables
- Query examples and templates

### 3. **Interactive Visualizations**
- **Price Charts**: Candlestick charts with zoom/pan
- **Symbol Comparison**: Multi-symbol normalized returns
- **Performance Heatmap**: Recent performance across symbols
- **Custom Time Ranges**: 7 days to 1 year

### 4. **Symbol Explorer**
- Complete symbol inventory (79 symbols)
- Data source breakdown (daily/intraday/both)
- Symbol-specific statistics
- Quick chart generation

### 5. **Database Management**
- Data quality checks
- Export capabilities
- Update controls
- System monitoring

## 🎯 Key Capabilities

### Unified Data Access
- **Smart Data Retrieval**: Automatically derives daily data from intraday when available
- **Source Transparency**: Shows which symbols come from which tables
- **Comprehensive Coverage**: Access to all 79 symbols in your database

### SQL Exploration
```sql
-- Example queries you can run:

-- Top symbols by data volume
SELECT symbol, COUNT(*) as records 
FROM daily_data 
GROUP BY symbol 
ORDER BY records DESC;

-- Recent performance
SELECT symbol, date, close 
FROM daily_data 
WHERE date >= '2025-05-01' 
ORDER BY date DESC;

-- Intraday volume analysis
SELECT symbol, interval, AVG(volume) as avg_volume
FROM intraday_data 
GROUP BY symbol, interval;
```

### Interactive Charts
- **Candlestick Charts**: OHLC data with volume
- **Zoom & Pan**: Explore different time periods
- **Multi-Symbol Comparison**: Normalized returns overlay
- **Real-time Updates**: Charts update as you change parameters

## 📁 File Structure

```
investing/
├── market_data_dashboard.py    # Main dashboard application
├── launch_dashboard.py         # Quick launcher script
├── templates/
│   └── dashboard.html          # Web interface template
├── market_data.db             # Your SQLite database
└── market_data_database.py    # Database interface
```

## 🔧 Technical Details

**Backend**: Flask (Python web framework)
**Frontend**: Bootstrap + Chart.js + Plotly.js
**Database**: SQLite with unified data access
**Safety**: SQL injection protection, read-only queries

## 🎨 Dashboard Sections

### Overview Tab
- Database statistics and health
- Visual data breakdown charts
- Performance summaries

### SQL Explorer Tab
- Live query execution
- Safety-checked SQL commands
- Formatted result display

### Visualizations Tab
- Interactive price charts
- Multi-symbol comparisons
- Custom time ranges

### Symbol Explorer Tab
- Complete symbol inventory
- Data source indicators
- Quick chart access

### Management Tab
- Quality checks
- Export tools
- System controls

## 🛡️ Security

- **Read-Only Access**: Only SELECT queries allowed
- **SQL Injection Protection**: Input validation and parameterized queries
- **Local Only**: Dashboard runs on localhost by default

## 🚀 Performance

- **Fast Queries**: Direct SQLite access with indexing
- **Cached Visualizations**: Charts update efficiently
- **Responsive Design**: Works on desktop and tablet

## 💡 Tips

1. **Start with Overview**: Get familiar with your data structure
2. **Use SQL Explorer**: Run custom queries to understand patterns
3. **Try Visualizations**: Compare symbols and explore trends
4. **Check Symbol Explorer**: See all available data sources
5. **Bookmark Queries**: Save useful SQL queries for later

## 🔄 Future Enhancements

- Export to CSV/Excel
- Custom dashboard creation
- Alert systems
- Data update scheduling
- Symphony strategy testing

---

**Dashboard URL**: http://localhost:5000  
**Created**: May 2025  
**Purpose**: Market data exploration and management