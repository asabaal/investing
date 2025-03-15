# Market Analysis Engine

A comprehensive market analysis engine for technical analysis and options prediction. This package provides tools for pattern recognition, regime analysis, risk assessment, and market relationship analysis.

## Features

- Technical Pattern Recognition
  - Head and Shoulders
  - Double Bottom
  - Volume-Price Patterns
  - Support/Resistance Analysis

- Market Regime Analysis
  - Hidden Markov Models
  - Volatility Regimes
  - Structural Breaks

- Risk Analysis
  - Value at Risk (VaR)
  - Expected Shortfall
  - Stress Testing
  - Volatility Analysis

- Market Relationships
  - Lead-Lag Analysis
  - Cross-correlations
  - Network Analysis
  - Granger Causality

## Installation

```bash
pip install market_analyzer
```

For development installation:

```bash
git clone https://github.com/yourusername/investing.git
cd investing
pip install -e ".[dev]"
```

## Quick Start

```python
from market_analyzer import MarketAnalyzer

# Initialize analyzer with your data
analyzer = MarketAnalyzer(data={
    'AAPL': apple_data_df,
    '^GSPC': sp500_data_df
})

# Detect patterns
patterns = analyzer.analyze_patterns('AAPL')

# Analyze regimes
regimes = analyzer.analyze_regimes('AAPL')

# Calculate risk metrics
risk = analyzer.analyze_risk('AAPL')
```

## Documentation

Full documentation is available at [readthedocs link].

## Development

To contribute to the project:

1. Fork the repository
2. Create a virtual environment
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.