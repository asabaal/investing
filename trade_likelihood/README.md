# Trade Likelihood Estimator

A comprehensive system for calculating trading probabilities and expected returns based on geometric Brownian motion and first passage time theory. **Perfect for paper trading decisions!**

## 🎯 What This Does

This system helps you make better paper trading decisions by:

- **Calculating entry probabilities** - How likely is your trade to trigger?
- **Estimating win rates** - What's the probability of hitting your target vs. stop loss?
- **Computing expected values** - Risk-adjusted returns for each setup
- **Measuring return rates** - Expected profit per unit time (the key innovation!)
- **Providing recommendations** - Clear buy/sell/avoid signals

## 🚀 Quick Start

### Simple 30-second analysis:
```python
from trade_likelihood import quick_analysis

# Your price history (replace with real data from your broker)
prices = [98.5, 99.1, 100.0, 99.8, 100.2, 101.1, ...]  

# Analyze current setup
result = quick_analysis(prices, current_price=100.0)

print(f"Return rate: {result['return_rate']:.4f} per day")
print(f"Best direction: {result['best_direction']}")  
print(f"Recommendation: {result['recommendation']}")
```

### Full analysis workflow:
```python
from trade_likelihood import TradeAnalyzer

# Initialize analyzer
analyzer = TradeAnalyzer()

# Load your price data
analyzer.load_price_data(your_price_series)

# Create trade setups (2% entry, 1% stop, 3% target)
setups = analyzer.create_simple_setups(current_price=100.0)

# Analyze the bidirectional setup
analysis = analyzer.analyze_trade_setup(setups['bidirectional'])

# Check if it's attractive for paper trading
if analysis.is_attractive_setup():
    print("✅ This is a good paper trade!")
    print(f"Expected return: {analysis.return_rate_total:.4f} per day")
    print(f"Best direction: {analysis.get_best_direction()}")
else:
    print("❌ Wait for better conditions")
```

### Compare multiple setups:
```python
# Compare different configurations
setups = analyzer.create_simple_setups(current_price=100.0)
comparison = analyzer.compare_setups(list(setups.values()), list(setups.keys()))

print(comparison.head())  # Shows best setups first
```

## 📊 Key Features

### 🎲 Probability Calculations
- Entry probability (will the trade trigger?)
- Win probability (will it hit target before stop?)
- Based on proven mathematical models (GBM + first passage time)

### ⏱️ Time Analysis
- Expected time to entry
- Expected trade duration
- Return rate per unit time

### 🧠 Market Intelligence
- Automatic parameter estimation (drift & volatility)
- Regime detection (trending, ranging, high/low vol)
- Confidence scoring for all estimates

### 🔧 Flexible Configuration
- Conservative vs. aggressive parameter estimation
- Customizable trade setups
- Risk management controls

## 📈 Perfect for Paper Trading

This system excels at paper trading because it:

1. **Quantifies opportunities** - No more gut feelings, get actual probabilities
2. **Compares setups objectively** - Find the best risk-adjusted opportunities
3. **Sets realistic expectations** - Know what return rate to expect
4. **Tracks model accuracy** - Compare predictions vs. actual results
5. **Scales to any timeframe** - Works for scalping to swing trading

## 🔬 The Math Behind It

Based on **geometric Brownian motion** (GBM) and **first passage time theory**:

- **Entry Probability**: P(price hits entry level within time window)
- **Win Probability**: P(hits take-profit before stop-loss | entry)
- **Expected Times**: Mean time to entry and exit
- **Return Rate**: Expected Value ÷ Expected Time (**key innovation!**)

This combines classic quantitative finance with practical trading needs.

## 📁 Project Structure

```
trade_likelihood/
├── __init__.py              # Main exports and quick_analysis()
├── trade_analyzer.py        # Primary interface (TradeAnalyzer)
├── core_math.py            # Mathematical functions
├── data_structures.py      # TradeSetup, TradeAnalysis, etc.
├── market_data.py          # Price data management
├── parameter_estimation.py # EWMA volatility, adaptive drift
├── probability_calculator.py # High-level probability calculations
└── examples/
    └── paper_trading_workflow.py # Complete examples
```

## 🏃‍♂️ Running the Examples

```bash
cd trade_likelihood/examples
python paper_trading_workflow.py
```

This runs complete examples showing:
- Basic workflow
- Quick analysis
- Advanced configurations
- Real-time monitoring simulation

## ⚙️ Configuration Options

### Conservative (for stable markets):
```python
from trade_likelihood import get_conservative_config, TradeAnalyzer

config = get_conservative_config()
analyzer = TradeAnalyzer(estimation_config=config)
```

### Aggressive (for volatile markets):
```python
from trade_likelihood import get_aggressive_config, TradeAnalyzer

config = get_aggressive_config()
analyzer = TradeAnalyzer(estimation_config=config)
```

## 🎯 Paper Trading Workflow

1. **Load Data**: Connect your broker API or load historical data
2. **Analyze Setups**: Use `analyzer.analyze_trade_setup()` for each configuration
3. **Compare Options**: Use `analyzer.compare_setups()` to find the best
4. **Execute**: When return rate > threshold, execute in paper account
5. **Monitor**: Track actual vs. predicted performance
6. **Validate**: Use results to refine parameters and strategies

## 🔍 Understanding the Output

### Key Metrics:
- **Return Rate**: Expected profit per day (most important metric)
- **Entry Probability**: Likelihood trade will trigger (0-100%)
- **Win Probability**: Likelihood of profit if triggered (0-100%)
- **Expected Value**: Average profit/loss per trade
- **Attractive Setup**: Boolean recommendation

### Interpreting Return Rates:
- `> 0.05` per day: Excellent setup (>18% annual)
- `> 0.01` per day: Good setup (>3.6% annual)
- `> 0.00` per day: Marginal setup
- `< 0.00` per day: Avoid

## ⚠️ Important Notes

1. **This is for PAPER TRADING** - Test thoroughly before risking real money
2. **Model assumptions** - Based on GBM (no jumps, constant parameters)
3. **Data quality matters** - Need clean, sufficient price data
4. **Parameter uncertainty** - Check confidence scores
5. **Market regime changes** - Monitor regime detection

## 🚨 Risk Disclaimer

This tool is for educational and paper trading purposes only. Past performance does not guarantee future results. Always practice proper risk management and never risk more than you can afford to lose.

## 📧 Next Steps

1. **Replace sample data** with real market data from your broker
2. **Adjust parameters** to match your trading style
3. **Start paper trading** with small position sizes
4. **Track performance** vs. model predictions
5. **Refine the model** based on actual results

---

**Happy paper trading! 📊✨**