# Complete Bidirectional Trading Probability Model

## 1. Price Dynamics Foundation

We model the asset price as geometric Brownian motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:
- $S_t$ = asset price at time t
- $\mu$ = drift rate (expected return)
- $\sigma$ = volatility
- $W_t$ = Wiener process (random walk)

## 2. Trade Setup Parameters

**Current State:**
- $S_0$ = current price
- $T$ = maximum time window for trade entry

**Long Trade Setup:**
- $E_L$ = long entry price ($E_L > S_0$)
- $SL_L$ = long stop loss ($SL_L < E_L$) 
- $TP_L$ = long take profit ($TP_L > E_L$)
- $R_L = \frac{TP_L - E_L}{E_L - SL_L}$ = long risk-reward ratio

**Short Trade Setup:**
- $E_S$ = short entry price ($E_S < S_0$)
- $SL_S$ = short stop loss ($SL_S > E_S$)
- $TP_S$ = short take profit ($TP_S < E_S$)
- $R_S = \frac{E_S - TP_S}{SL_S - E_S}$ = short risk-reward ratio

## 3. Entry Probability Calculations

### 3.1 First Passage Time Probability

For geometric Brownian motion, the probability of hitting level $B$ before time $T$ starting from $S_0$:

$$P(\tau_B \leq T) = \Phi(d_1) + \left(\frac{S_0}{B}\right)^{2\alpha} \Phi(d_2)$$

Where:
- $\alpha = \frac{\mu}{\sigma^2} - \frac{1}{2}$
- $d_1 = \frac{\ln(S_0/B) + (\mu + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}$
- $d_2 = \frac{\ln(S_0/B) + (\mu - \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}$
- $\Phi$ = standard normal cumulative distribution function

### 3.2 Specific Entry Probabilities

**Long Entry Probability:**
$$P_{entry,L} = P(\text{price hits } E_L \text{ before time } T)$$

**Short Entry Probability:**
$$P_{entry,S} = P(\text{price hits } E_S \text{ before time } T)$$

## 4. Win Probability Given Entry

Once a trade is entered, we calculate the probability of hitting take profit before stop loss using the barrier race formula.

### 4.1 General Barrier Race Formula

For a trade entered at price $E$, probability of hitting $TP$ before $SL$:

$$P_{win|entry} = \frac{1 - \left(\frac{SL}{E}\right)^{2\alpha}}{1 - \left(\frac{SL}{TP}\right)^{2\alpha}}$$

For zero drift ($\mu = 0$), this simplifies to:
$$P_{win|entry} = \frac{\ln(E/SL)}{\ln(TP/SL)}$$

### 4.2 Trade-Specific Win Probabilities

**Long Win Probability (given entry at $E_L$):**
$$P_{win,L|entry} = \frac{1 - \left(\frac{SL_L}{E_L}\right)^{2\alpha}}{1 - \left(\frac{SL_L}{TP_L}\right)^{2\alpha}}$$

**Short Win Probability (given entry at $E_S$):**
$$P_{win,S|entry} = \frac{1 - \left(\frac{E_S}{SL_S}\right)^{2\alpha}}{1 - \left(\frac{TP_S}{SL_S}\right)^{2\alpha}}$$

## 5. Expected Time Analysis

### 5.1 Expected Time to Entry

Expected first passage time from $S_0$ to barrier $B$:

$$E[\tau_B] = \begin{cases}
\frac{\ln(B/S_0)}{\mu} & \text{if } \mu \neq 0 \text{ and drift toward } B \\
\infty & \text{if } \mu \neq 0 \text{ and drift away from } B \\
\frac{(\ln(B/S_0))^2}{\sigma^2} & \text{if } \mu = 0
\end{cases}$$

**Specific Entry Times:**
- $E[T_{entry,L}]$ = expected time to hit long entry $E_L$
- $E[T_{entry,S}]$ = expected time to hit short entry $E_S$

### 5.2 Expected Trade Duration Given Entry

Expected time to exit (hit either TP or SL) once trade is entered:

$$E[T_{exit}|entry] = \frac{1}{\mu^2 + \frac{\sigma^2}{2}} \left[ \frac{TP - E}{TP - SL} \ln\left(\frac{TP}{E}\right) + \frac{E - SL}{TP - SL} \ln\left(\frac{E}{SL}\right) \right]$$

For zero drift ($\mu = 0$):
$$E[T_{exit}|entry] = \frac{2}{\sigma^2} \frac{(TP-E)(E-SL)}{TP-SL}$$

### 5.3 Conditional Expected Times

**Expected time to win (given entry and eventual win):**
$$E[T_{win}|entry,win] = \frac{E[T_{exit}|entry] \cdot P_{win|entry}}{\text{probability weighted}}$$

**Expected time to loss (given entry and eventual loss):**
$$E[T_{loss}|entry,loss] = \frac{E[T_{exit}|entry] \cdot (1-P_{win|entry})}{\text{probability weighted}}$$

## 6. Dynamic Parameter Estimation

### 6.1 Exponentially Weighted Moving Average (EWMA) Volatility

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$$

Where:
- $\lambda$ = decay factor (typically 0.94-0.97 for daily data)
- $r_{t-1} = \ln(S_{t-1}/S_{t-2})$ = previous log return

### 6.2 Adaptive Drift Estimation

Recent-weighted drift estimation:
$$\mu_t = \sum_{i=1}^{n} w_i r_{t-i}$$

Where $w_i = \frac{e^{-\alpha i}}{\sum_{j=1}^{n} e^{-\alpha j}}$ are exponential weights with decay rate $\alpha$.

### 6.3 Regime Detection

Monitor for volatility regime changes:
- **Low volatility regime**: $\sigma_{low}$, probability $p_{low}$  
- **High volatility regime**: $\sigma_{high}$, probability $p_{high}$
- **Regime switching**: Use Markov models or threshold detection

## 7. Expected Value Framework

### 7.1 Individual Trade Expected Values

**Long Trade Expected Value:**
$$EV_L = P_{entry,L} \times P_{win,L|entry} \times R_L - P_{entry,L} \times (1-P_{win,L|entry}) \times 1$$

**Short Trade Expected Value:**
$$EV_S = P_{entry,S} \times P_{win,S|entry} \times R_S - P_{entry,S} \times (1-P_{win,S|entry}) \times 1$$

### 7.2 Combined Strategy Expected Value

$$EV_{total} = EV_L + EV_S$$

Note: This assumes independence; adjust for correlation if both trades can be active simultaneously.

## 8. Return Rate Analysis (Key Innovation)

### 8.1 Expected Return Per Unit Time

**Long Trade Return Rate:**
$$RR_L = \frac{EV_L}{E[T_{entry,L}] + P_{entry,L} \times E[T_{exit,L}|entry]}$$

**Short Trade Return Rate:**
$$RR_S = \frac{EV_S}{E[T_{entry,S}] + P_{entry,S} \times E[T_{exit,S}|entry]}$$

**Combined Strategy Return Rate:**
$$RR_{total} = \frac{EV_{total}}{E[T_{total\_strategy}]}$$

### 8.2 Trade Frequency Metrics

**Expected number of setups per time period:**
$$\text{Setup Rate} = \frac{P_{entry,L} + P_{entry,S}}{E[T_{between\_opportunities}]}$$

**Capital utilization efficiency:**
$$\text{Capital Efficiency} = \frac{\text{Expected Active Capital}}{\text{Total Allocated Capital}}$$

### 8.3 Risk-Adjusted Return Rate

$$\text{Sharpe}_{strategy} = \frac{RR_{total} - r_f}{\sqrt{\text{Var}[Returns/Time]}}$$

Where $r_f$ is the risk-free rate.

## 9. Implementation Functions

### 9.1 Core Calculation Functions

```python
def first_passage_probability(S0, barrier, mu, sigma, T):
    """Calculate probability of hitting barrier before time T"""
    
def barrier_race_probability(entry, tp, sl, mu, sigma):
    """Calculate P(hit TP before SL) given entry"""
    
def expected_entry_time(S0, entry_level, mu, sigma):
    """Calculate expected time to hit entry level"""
    
def expected_exit_time(entry, tp, sl, mu, sigma):
    """Calculate expected trade duration given entry"""
```

### 9.2 Parameter Estimation Functions

```python
def update_volatility_ewma(returns, lambda_decay=0.94):
    """Update volatility using EWMA"""
    
def estimate_drift_adaptive(returns, alpha=0.1):
    """Estimate drift with recency weighting"""
    
def detect_regime_change(volatility_series, threshold=2.0):
    """Detect volatility regime shifts"""
```

### 9.3 Strategy Optimization Functions

```python
def optimize_entry_levels(S0, mu, sigma, T, min_prob=0.3):
    """Find optimal entry levels given constraints"""
    
def calculate_return_rates(probabilities, expected_times, payoffs):
    """Calculate return per unit time for all trades"""
    
def position_sizing(return_rates, correlations, risk_budget):
    """Determine position sizes based on return rates"""
```

## 10. Implementation Workflow

### 10.1 Real-Time Execution Flow

1. **Update Parameters**: Calculate current $\mu_t$ and $\sigma_t$ using EWMA
2. **Calculate Entry Probabilities**: For given entry levels and time window
3. **Calculate Expected Times**: Entry times and trade durations  
4. **Compute Return Rates**: Expected value per unit time
5. **Optimize if Needed**: Adjust entry levels if return rates are suboptimal
6. **Execute Trades**: Place orders with appropriate position sizing
7. **Monitor**: Track actual vs expected times and probabilities

### 10.2 Backtesting Framework

- **Rolling Estimation**: Update parameters with each new data point
- **Out-of-Sample Testing**: Test on unseen data with time-aware metrics
- **Regime Analysis**: Performance across different volatility regimes
- **Transaction Costs**: Include realistic spreads and commissions

## 11. Key Practical Considerations

### 11.1 Model Limitations
- **GBM Assumptions**: No jumps, mean-reverting volatility not captured
- **Parameter Stability**: Regime changes can invalidate recent estimates
- **Market Microstructure**: Bid-ask spreads, liquidity not modeled

### 11.2 Risk Management
- **Correlation Risk**: Both long and short may trigger simultaneously
- **Time Risk**: Trades taking longer than expected tie up capital
- **Parameter Risk**: Model sensitivity to estimation errors
- **Regime Risk**: Performance degradation in different market conditions

### 11.3 Advanced Enhancements
- **Jump-Diffusion Models**: Capture sudden price movements
- **Stochastic Volatility**: Allow time-varying volatility
- **Machine Learning**: Use ML for parameter estimation and regime detection
- **Multi-Asset**: Extend to portfolio of correlated assets

This framework provides the mathematical foundation for automated identification of high-probability, time-efficient trading opportunities using bidirectional strategies.