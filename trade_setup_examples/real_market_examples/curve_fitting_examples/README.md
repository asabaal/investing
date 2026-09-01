# Curve Fitting Supply/Demand Zone Detection

This directory contains **real market data examples** demonstrating how **mathematical curve fitting** can be used to identify **supply/demand zones** and **trend changes** in financial markets.

## 🎯 Core Concept

Instead of manually identifying support/resistance levels, this approach uses:

- **📈 Curve Fitting**: Fits mathematical curves (cubic splines, polynomials, Prophet) to price data
- **🔍 Extrema Detection**: Uses analytical differentiation to find local maxima (supply zones) and minima (demand zones)  
- **💫 Inflection Points**: Identifies curvature changes that signal trend shifts
- **🌌 Spacetime Coordinates**: Analyzes both traditional OHLC and transformed coordinates (low, range, body_ratio, upper_wick_ratio)

## 📊 Examples

### Real Market Data Analysis

| File | Symbol | Period | Description |
|------|--------|---------|-------------|
| `spy_supply_demand_zones.html` | SPY | Mar-Jul 2024 | S&P 500 ETF showing uptrend with supply zone resistance |
| `aapl_supply_demand_zones.html` | AAPL | Sep 2023-Jan 2024 | Apple stock with tech volatility and clear demand levels |
| `nvda_supply_demand_zones.html` | NVDA | May-Sep 2023 | High-volatility AI stock with strong zones during AI boom |
| `tsla_supply_demand_zones.html` | TSLA | Aug-Dec 2023 | High-beta Tesla with frequent zone tests |

### Visual Elements

- 🔴 **Red Triangles** = Supply zones (resistance levels from maxima)
- 🟢 **Green Triangles** = Demand zones (support levels from minima) 
- 🟡 **Yellow Diamonds** = Inflection points (trend change signals)
- 🟡 **Yellow Curve** = Mathematical curve fit showing underlying trend

## ⚙️ Technical Implementation

### Curve Fitting Methods
1. **Cubic Spline** - Smooth curves through data points with continuous derivatives
2. **Univariate Spline** - Smoothing splines with adjustable tension
3. **Polynomial** - 5th-8th degree polynomials for trend approximation
4. **Facebook Prophet** - Time series forecasting with trend detection

### Extrema Detection
- **Analytical Method**: Find roots of first derivative using scipy optimization
- **Numerical Method**: Peak detection with scipy.signal.find_peaks
- **Classification**: Second derivative test distinguishes maxima from minima

### Coordinate Systems
- **Traditional**: Raw OHLC price data
- **Spacetime**: Transformed coordinates emphasizing candle structure:
  - `low` = Candle low price
  - `range` = High - Low 
  - `body_ratio` = Body size / Range
  - `upper_wick_ratio` = Upper wick / Range

## 🚀 Key Benefits

1. **Mathematical Precision**: Uses calculus to find exact turning points
2. **Automated Detection**: No manual zone drawing required
3. **Multiple Timeframes**: Works across different time scales
4. **Objective Results**: Removes subjective interpretation
5. **Early Signals**: Inflection points can predict trend changes before they're obvious

## 💡 Trading Applications

### Supply Zones (Red Triangles)
- **Entry**: Short positions when price approaches from below
- **Stop Loss**: Above the zone high
- **Target**: Previous support levels or demand zones

### Demand Zones (Green Triangles)  
- **Entry**: Long positions when price approaches from above
- **Stop Loss**: Below the zone low
- **Target**: Previous resistance levels or supply zones

### Inflection Points (Yellow Diamonds)
- **Trend Changes**: Early warning of momentum shifts
- **Position Management**: Consider reducing exposure or taking profits
- **New Opportunities**: Setup for trend-following entries

## 🔧 Files

- `simple_curve_example.py` - Basic example generator
- `create_multiple_examples.py` - Batch example creator for multiple symbols
- `supply_demand_curve_detector.py` - Full implementation with zone objects
- `README.md` - This documentation

## 📈 Results Summary

Across all examples, the system detected:

| Symbol | Supply Zones | Demand Zones | Inflection Points | Key Insight |
|--------|--------------|--------------|-------------------|-------------|
| SPY | 5 | 4 | 26 | Stable uptrend with clear resistance |
| AAPL | 3 | 5 | 26 | Tech volatility with strong support |
| NVDA | 5 | 3 | 26 | High volatility during AI hype |
| TSLA | 3 | 6 | 23 | High-beta with frequent zone tests |

## 🎓 Educational Value

This approach demonstrates:
- **Real-world application** of mathematical concepts to trading
- **Integration** of multiple curve fitting techniques
- **Practical implementation** of supply/demand analysis
- **Objective methodology** for zone identification
- **Advanced coordinate systems** for market structure analysis

---

*This is a practical implementation of the "GOOD ENOUGH and FAST" curve fitting system for supply/demand zone detection as requested.*