# Symphony Trading System v2 - Quick Start Guide

## Overview

This is a complete rewrite of your symphony trading system with a **clean, scalable architecture** that eliminates circular imports and provides a robust foundation for algorithmic trading.

### ✅ **Fixed Architecture Issues**
- **No circular imports** - Uses factory pattern and dependency injection
- **Modular components** - Each system is independent and testable  
- **Scalable design** - Easy to add new features without breaking existing code
- **Clean separation** - Core, service, and presentation layers properly separated

1. **Metrics Module** - Calculates all technical indicators
2. **Symphony Engine** - Executes conditional logic and portfolio allocation  
3. **Data Pipeline** - Fetches and manages market data
4. **Backtesting Engine** - Tests strategies over historical periods
5. **Symphony Runner** - Main orchestration script

## Installation

### 1. Set up Python Environment
```

## Prerequisites

- Python 3.8 or higher
- Alpha Vantage API Key (set as environment variable `ALPHA_VANTAGE_API_KEY`)
- Required packages (install with `pip install -r requirements.txt`)

## Getting Startedbash
# Create new branch for clean implementation
git checkout -b symphony-v2-clean

# Create virtual environment (recommended)
python -m venv symphony_env
source symphony_env/bin/activate  # On Windows: symphony_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create requirements.txt
```txt
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
python-dateutil>=2.8.0

# Market data and APIs
requests>=2.28.0

# Visualization
matplotlib>=3.5.0
plotly>=5.0.0
seaborn>=0.11.0

# Optional forecasting (install if needed)
# prophet>=1.1.0

# Development and testing
pytest>=7.0.0
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Optional: Install Prophet for advanced forecasting
pip install prophet
```

### 3. Set Environment Variables
```bash
# Set your Alpha Vantage API key
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# On Windows:
# set ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Quick Start

### 1. Verify Setup
```bash
python setup_verification.py
```
This checks all dependencies and system components.

### 2. Create Sample Symphony
```bash
python symphony_runner.py --create-sample
```
This creates `sample_symphony_v2.json` with a working strategy.

### 3. Run Complete Analysis (NEW!)
```bash
python symphony_runner.py --config sample_symphony_v2.json --full-analysis
```
This runs backtest AND creates beautiful visualizations!

### 4. Open Interactive Dashboard
```bash
# After running analysis, open in your browser:
open results/charts/interactive_dashboard.html
```

### 5. Run Just Backtest
```bash
python symphony_runner.py --config sample_symphony_v2.json --backtest --start-date 2023-01-01
```

### 6. Get Current Portfolio
```bash
python symphony_runner.py --config sample_symphony_v2.json --execute
```

## Symphony Configuration Format

The new format properly represents Composer.trade logic:

```json
{
  "name": "Strategy Name",
  "universe": ["AAPL", "MSFT", "SPY"],
  "rebalance_frequency": "monthly",
  
  "logic": {
    "conditions": [
      {
        "id": "market_check",
        "type": "if_statement",
        "condition": {
          "metric": "cumulative_return",
          "asset_1": "SPY",
          "operator": "greater_than", 
          "asset_2": {"type": "fixed_value", "value": 0.0},
          "lookback_days": 60
        },
        "if_true": "aggressive_allocation",
        "if_false": "defensive_allocation"
      }
    ],
    
    "allocations": {
      "aggressive_allocation": {
        "type": "sort_and_weight",
        "sort": {
          "metric": "cumulative_return",
          "lookback_days": 90,
          "direction": "top",
          "count": 3
        },
        "weighting": {
          "method": "equal_weight"
        }
      },
      
      "defensive_allocation": {
        "type": "fixed_allocation", 
        "weights": {
          "SPY": 0.6,
          "TLT": 0.4
        }
      }
    }
  }
}
```

## Available Metrics

- `current_price` - Latest closing price
- `cumulative_return` - Total return over period
- `ema_price` - Exponential moving average
- `max_drawdown` - Maximum drawdown
- `moving_average_price` - Simple moving average
- `moving_average_return` - Average of daily returns
- `rsi` - Relative Strength Index
- `standard_deviation_price` - Price volatility
- `standard_deviation_return` - Return volatility

## Available Operators

- `greater_than`, `less_than`
- `greater_than_or_equal`, `less_than_or_equal`
- `equal`, `not_equal`

## Weighting Methods

- `equal_weight` - Equal allocation across selected assets
- `inverse_volatility` - Weight inversely to volatility
- `specified_weight` - Use exact weights provided

## File Structure

```
symphony-v2-clean/
├── symphony_runner.py          # Main entry point
├── metrics_module.py           # Technical indicators
├── symphony_engine.py          # Logic execution engine
├── data_pipeline.py           # Data fetching and management
├── requirements.txt           # Python dependencies
├── sample_symphony_v2.json    # Example configuration
├── data_cache/               # Cached market data
└── backtest_results/         # Output files
    ├── backtest_results.csv
    ├── performance_analysis.json
    └── allocation_history.csv
```

## Usage Examples

### Basic Backtest
```bash
# 1-year backtest with monthly rebalancing
python symphony_runner.py --config my_strategy.json --backtest \
  --start-date 2023-01-01 --end-date 2024-01-01 --frequency monthly
```

### Weekly Rebalancing
```bash
python symphony_runner.py --config my_strategy.json --backtest \
  --frequency weekly --start-date 2024-01-01
```

### Current Execution
```bash
# See what the strategy would do today
python symphony_runner.py --config my_strategy.json --execute
```

### Custom Output Directory
```bash
python symphony_runner.py --config my_strategy.json --backtest \
  --output-dir ./my_strategy_results
```

## Sample Symphony Strategies

### 1. Momentum Strategy
Buys top 3 momentum stocks when market is positive, bonds when negative.

### 2. Low Volatility Strategy
```json
{
  "name": "Low Vol Quality",
  "universe": ["AAPL", "MSFT", "JNJ", "PG", "KO"],
  "logic": {
    "conditions": [
      {
        "condition": {
          "metric": "rsi",
          "asset_1": "SPY", 
          "operator": "less_than",
          "asset_2": {"type": "fixed_value", "value": 70},
          "lookback_days": 14
        },
        "if_true": "low_vol_allocation",
        "if_false": "cash"
      }
    ],
    "allocations": {
      "low_vol_allocation": {
        "type": "sort_and_weight",
        "sort": {
          "metric": "standard_deviation_return",
          "direction": "bottom",
          "count": 3
        },
        "weighting": {
          "method": "inverse_volatility"
        }
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: System automatically handles rate limiting (12 seconds between calls)
2. **Missing Data**: Check symbols are valid and API key is set
3. **Insufficient Data**: Ensure symbols have at least 100 days of history

### Data Caching

- Data is cached in `./data_cache/` directory
- Cache expires after 24 hours for daily data
- Use `use_cache=False` to force fresh data fetch

### Performance Tips

1. Use monthly rebalancing for longer backtests
2. Limit universe to 10-15 symbols for faster execution
3. Cache is automatically used for repeated runs
4. Start with shorter time periods for testing

## Next Steps

1. **Test the Sample**: Run the included sample strategy
2. **Create Your Strategy**: Modify the sample or create new configurations  
3. **Backtest Multiple Periods**: Test across different market conditions
4. **Paper Trading**: Use execute mode to track real-time signals
5. **Production Integration**: Connect to broker APIs for live trading

## Architecture Benefits

- **Modular Design**: Each component can be used independently
- **Extensible**: Easy to add new metrics or weighting methods
- **Testable**: Clean separation allows unit testing
- **Cacheable**: Efficient data management reduces API calls
- **Scalable**: Can handle multiple symphonies and asset classes

This system gives you the foundation to build sophisticated trading strategies while maintaining the flexibility to customize every aspect of the logic.