# Symphony Visualization Guide 🎨

Your Symphony Trading System now includes comprehensive visualization capabilities! Here's how to create beautiful, insightful charts to analyze your trading strategies.

## Quick Start

### 1. Full Analysis (Backtest + Visualizations)
```bash
# Create sample and run complete analysis
python symphony_runner.py --create-sample
python symphony_runner.py --config sample_symphony_v2.json --full-analysis
```

This creates:
- 📊 **Performance Dashboard** (static PNG)
- 🌐 **Interactive Dashboard** (HTML with Plotly)
- 🌅 **Allocation Sunburst Chart** (interactive HTML)
- 📈 **Rolling Metrics Chart** (interactive HTML)

### 2. Backtest with Custom Settings
```bash
# 2-year backtest with weekly rebalancing and benchmark comparison
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2022-01-01 --end-date 2024-01-01 \
  --frequency weekly --benchmark QQQ
```

### 3. Visualize Existing Results
```bash
# Create charts from previously run backtest
python symphony_runner.py --visualize-only --output-dir ./my_results --benchmark SPY
```

## Available Charts

### 📊 Performance Dashboard
**File**: `charts/performance_dashboard.png`

A comprehensive 6-panel static dashboard showing:
1. **Cumulative Returns** - Strategy vs benchmark over time
2. **Drawdown Chart** - Maximum drawdown periods highlighted  
3. **Monthly Returns Heatmap** - Heat map of monthly performance
4. **Allocation Evolution** - How portfolio weights change over time
5. **Performance Metrics** - Key statistics as bar chart
6. **Return Distribution** - Histogram of returns with statistics

### 🌐 Interactive Dashboard  
**File**: `charts/interactive_dashboard.html`

Interactive Plotly dashboard with:
- **Zoomable charts** - Click and drag to zoom
- **Hover details** - Mouse over for exact values
- **Toggle series** - Click legend to show/hide data
- **Responsive design** - Works on desktop and mobile

### 🌅 Allocation Sunburst
**File**: `charts/allocation_sunburst.html`

Beautiful sunburst chart showing:
- **Current portfolio allocation**
- **Hierarchical view** of portfolio structure
- **Interactive hover** with exact percentages
- **Triggered condition** displayed in title

### 📈 Rolling Metrics
**File**: `charts/rolling_metrics.html`

Three-panel rolling analysis:
- **Rolling Annual Return** - Performance over time
- **Rolling Volatility** - Risk changes over time  
- **Rolling Sharpe Ratio** - Risk-adjusted performance

## Chart Customization

### Benchmark Comparison
```bash
# Compare against different benchmarks
python symphony_runner.py --config my_strategy.json --full-analysis --benchmark QQQ
python symphony_runner.py --config my_strategy.json --full-analysis --benchmark IWM
python symphony_runner.py --config my_strategy.json --full-analysis --benchmark VTI
```

### Custom Time Periods
```bash
# Bull market period
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2020-04-01 --end-date 2022-01-01

# Bear market period  
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2022-01-01 --end-date 2022-12-31

# Recent performance
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2024-01-01
```

### Rebalancing Frequency Analysis
```bash
# Compare different rebalancing frequencies
python symphony_runner.py --config my_strategy.json --full-analysis \
  --frequency daily --output-dir ./results_daily

python symphony_runner.py --config my_strategy.json --full-analysis \
  --frequency weekly --output-dir ./results_weekly

python symphony_runner.py --config my_strategy.json --full-analysis \
  --frequency monthly --output-dir ./results_monthly
```

## Understanding the Charts

### 📊 Key Metrics Explained

**Total Return**: Overall strategy performance
- Formula: `(Final Value - Initial Value) / Initial Value`
- Good: > 10% annually
- Excellent: > 15% annually

**Sharpe Ratio**: Risk-adjusted returns
- Formula: `(Return - Risk Free Rate) / Volatility`
- Good: > 1.0
- Excellent: > 2.0

**Max Drawdown**: Worst peak-to-trough decline
- Shows maximum pain endured
- Lower (less negative) is better
- Acceptable: < -20%

**Volatility**: Standard deviation of returns
- Measures consistency
- Lower is generally better
- Compare to benchmark for context

**Win Rate**: Percentage of profitable periods
- Higher is better
- 60%+ is good for most strategies

### 🎯 Reading the Charts

#### Cumulative Returns Chart
- **Upward slope**: Strategy making money
- **Steeper = better**: Higher returns
- **Smooth line**: Consistent performance
- **Jagged line**: Volatile performance

#### Drawdown Chart
- **Red areas**: Losing periods
- **Deeper red**: Larger losses
- **Recovery speed**: How quickly strategy recovers

#### Allocation Chart
- **Color bands**: Different assets
- **Width**: Portfolio weight
- **Changes**: When rebalancing occurs

## Advanced Usage

### Batch Analysis
Create a script to analyze multiple strategies:

```bash
#!/bin/bash
# analyze_all_strategies.sh

strategies=("momentum_strategy.json" "value_strategy.json" "growth_strategy.json")

for strategy in "${strategies[@]}"; do
    echo "Analyzing $strategy..."
    python symphony_runner.py --config "$strategy" --full-analysis \
      --output-dir "./results_$(basename $strategy .json)"
done
```

### Custom Output Organization
```bash
# Organize results by date
today=$(date +%Y-%m-%d)
python symphony_runner.py --config my_strategy.json --full-analysis \
  --output-dir "./backtest_results_$today"
```

### Performance Comparison
```bash
# Create multiple backtests for comparison
python symphony_runner.py --config strategy_v1.json --full-analysis \
  --output-dir ./v1_results

python symphony_runner.py --config strategy_v2.json --full-analysis \
  --output-dir ./v2_results

# Then manually compare the HTML dashboards
```

## Troubleshooting

### Common Issues

**"No benchmark data available"**
- Check benchmark symbol is valid (SPY, QQQ, IWM, etc.)
- Ensure you have valid Alpha Vantage API key
- Try with `--benchmark ""` to skip benchmark

**"Insufficient data for monthly heatmap"**
- Backtest period too short for monthly analysis
- Use longer time period or weekly frequency

**Charts not displaying**
- For PNG files: Check file exists and try different image viewer
- For HTML files: Open in web browser (Chrome, Firefox, Safari)
- For interactive issues: Try different browser or disable ad blockers

**Slow chart generation**
- Large datasets take time to process
- Use `--no-charts` flag to skip visualization temporarily
- Consider shorter time periods for testing

### Performance Tips

1. **Use monthly rebalancing** for longer backtests (faster)
2. **Limit universe size** to 10-15 symbols for speed
3. **Cache is your friend** - repeated runs use cached data
4. **Interactive charts load faster** than static dashboards

### File Organization

```
backtest_results/
├── backtest_results.csv           # Raw backtest data
├── performance_analysis.json      # Performance metrics
├── allocation_history.csv         # Allocation changes
├── latest_execution.json          # Most recent execution
└── charts/
    ├── performance_dashboard.png   # Static dashboard
    ├── interactive_dashboard.html  # Interactive dashboard  
    ├── allocation_sunburst.html    # Portfolio allocation
    └── rolling_metrics.html        # Rolling analysis
```

## Tips for Strategy Analysis

### 1. Always Compare to Benchmark
- Use relevant benchmark (SPY for large cap, IWM for small cap)
- Look for **excess return** above benchmark
- Check if **risk-adjusted returns** beat benchmark

### 2. Analyze Different Market Conditions
```bash
# Bull market (2020-2021)
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2020-01-01 --end-date 2022-01-01

# Bear market (2022)
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2022-01-01 --end-date 2023-01-01

# Recovery (2023)
python symphony_runner.py --config my_strategy.json --full-analysis \
  --start-date 2023-01-01 --end-date 2024-01-01
```

### 3. Look for Consistent Patterns
- **Steady upward trajectory** in cumulative returns
- **Quick recovery** from drawdowns  
- **Reasonable volatility** for the return achieved
- **Logical allocation changes** based on market conditions

### 4. Red Flags to Watch
- **Excessive drawdowns** (> -30%)
- **Long recovery periods** from losses
- **Extreme concentration** in single assets
- **Poor performance vs benchmark** consistently

The visualization system gives you powerful tools to understand your trading strategies. Use them to refine your symphonies and build confidence in your trading decisions! 🚀