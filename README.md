# Symphony Trading System

A comprehensive system for algorithmic trading using Composer symphonies with backtesting, forecasting, and strategy optimization capabilities.

## Overview

This system provides tools for:

1. **Analyzing** Composer symphonies through backtesting and forecasting
2. **Optimizing** symphonies by testing parameter variations
3. **Monitoring** the health of symphonies with a watchlist dashboard
4. **Testing** symphonies under different market scenarios
5. **Forecasting** security prices using Prophet models

## Components

The system consists of several modules:

- **alpha_vantage_api.py**: Client for fetching market data
- **composer_symphony.py**: Core functionality for parsing and executing symphonies
- **prophet_forecasting.py**: Time series forecasting using Prophet
- **symphony_analyzer.py**: Comprehensive symphony analysis
- **symphony_simulation.py**: Testing symphonies under different scenarios
- **symphony_watchlist.py**: GUI for monitoring symphonies and their components
- **symphony_cli.py**: Command-line interface for the system

## Prerequisites

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - prophet
  - plotly
  - tkinter
  - requests

- Alpha Vantage API Key (set as environment variable `ALPHA_VANTAGE_API_KEY`)

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/asabaal/investing.git
   cd investing
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Alpha Vantage API key:
   ```
   export ALPHA_VANTAGE_API_KEY=your_api_key
   ```

### Usage

#### Command Line Interface

The system provides a command-line interface through `symphony_cli.py`:

```
python symphony_cli.py [command] [options]
```

Available commands:

- **analyze**: Analyze a symphony's performance and forecast
- **optimize**: Generate variations of a symphony to find optimal parameters
- **test**: Test a symphony under different market scenarios
- **forecast**: Generate price forecasts for symbols
- **create**: Create a new symphony configuration
- **watchlist**: Launch the Symphony Watchlist GUI

#### Sample Commands

1. **Analyze a symphony**:
   ```
   python symphony_cli.py analyze sample_symphony.json --start-date 2023-01-01 --end-date 2023-12-31 --benchmark SPY
   ```

2. **Optimize a symphony**:
   ```
   python symphony_cli.py optimize sample_symphony.json --variations 10 --metric sharpe_ratio --output optimized_symphony.json
   ```

3. **Generate forecasts**:
   ```
   python symphony_cli.py forecast --symbols SPY,QQQ,IWM --days 30 --ensemble
   ```

4. **Launch the watchlist**:
   ```
   python symphony_cli.py watchlist
   ```

### Symphony Configuration

Symphonies are defined in JSON format with the following structure:

```json
{
  "name": "Strategy Name",
  "description": "Strategy description",
  "universe": ["SPY", "QQQ", "IWM"],
  "operators": [
    {
      "type": "Momentum",
      "name": "Momentum Filter",
      "condition": {
        "lookback_days": 90,
        "top_n": 3
      }
    }
  ],
  "allocator": {
    "type": "InverseVolatilityAllocator",
    "name": "Inverse Volatility",
    "method": "inverse_volatility",
    "lookback_days": 30
  }
}
```

Available operator types:
- `Momentum`: Selects top performing symbols based on momentum
- `RSIFilter`: Filters symbols based on RSI values
- `MovingAverageCrossover`: Selects symbols based on moving average crossovers

Available allocator types:
- `EqualWeightAllocator`: Assigns equal weight to all symbols
- `InverseVolatilityAllocator`: Weights inversely proportional to volatility

## Watchlist Dashboard

The watchlist dashboard provides a GUI for monitoring symphonies and their components:

- **Dashboard Tab**: Overview of symphony health and alerts
- **Symbols Tab**: Detailed information on all securities
- **Symphonies Tab**: Symphony composition and performance
- **Analysis Tab**: Tools for detailed symphony analysis

Launch the dashboard with:
```
python symphony_cli.py watchlist
```

## Advanced Features

### Symphony Variations

Generate variations of a symphony to test parameter sensitivity:

```
python symphony_cli.py optimize sample_symphony.json --variations 10
```

### Market Scenario Testing

Test symphonies under different market conditions:

```
python symphony_cli.py test sample_symphony.json
```

### Ensemble Forecasting

Use multiple Prophet models for more robust forecasts:

```
python symphony_cli.py forecast --symbols SPY --ensemble --models 5
```

## Examples

### Creating a Symphony

1. Create a basic symphony:
   ```
   python symphony_cli.py create "My Strategy" --description "A simple strategy" --universe SPY,QQQ,IWM --output my_symphony.json
   ```

2. Edit the JSON file to add operators and an allocator

3. Analyze the symphony:
   ```
   python symphony_cli.py analyze my_symphony.json
   ```

### Optimizing a Symphony

1. Create parameter space variations:
   ```
   python symphony_cli.py optimize sample_symphony.json --variations 20 --output optimized.json
   ```

2. Analyze the optimized symphony:
   ```
   python symphony_cli.py analyze optimized.json
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
