# Trading Research, Paper Trading & Live Execution Platform

## Executive + Technical Strategy & Vision

**Status:** Initial Architecture Vision
**Primary cadence:** 15-minute market intervals
**Operating principle:** Deterministic trading models first; AI reasoning optional, never required for routine execution.

---

## 1. Executive Vision

Build a private trading research platform that can:

1. Define a curated universe of fundamentally acceptable companies.
2. Pull current and historical market data programmatically.
3. Test multiple trading strategies against historical data.
4. Forward-test those same strategies against live markets using simulated capital.
5. Compare model performance objectively.
6. Promote successful strategies through increasingly realistic validation stages.
7. Eventually support controlled live trading.
8. Maintain strong separation between:

   * market data,
   * strategy logic,
   * risk management,
   * portfolio state,
   * execution,
   * notifications.

The system should not depend on an AI model continuously reasoning about the market.

The trading rules themselves should be normal software:

```text
market data
    ↓
deterministic model
    ↓
signal
    ↓
risk engine
    ↓
paper / shadow / live execution
```

AI can later assist with research, strategy development, debugging, analysis, and reporting without becoming a required component of the live trading loop.

---

# 2. Core Strategic Principle

## One Strategy, Four Operating Modes

A strategy should be written **once**.

The same strategy code should then operate in:

```text
BACKTEST
   ↓
FORWARD PAPER TRADING
   ↓
SHADOW LIVE
   ↓
LIVE TRADING
```

Only the adapters around it should change.

### Backtest

Historical market data is replayed through the model.

### Paper trading

Current live market data is used, but positions and cash exist only inside our simulator.

### Shadow live

The system generates exactly the trades it would execute with real money but does not submit them.

### Live

The same signal and risk pipeline ultimately reaches a brokerage execution adapter.

This prevents a major quantitative-development failure:

> building one system for backtesting and an entirely different system for actual trading.

---

# 3. Investment Universe Strategy

The project should **not begin by indiscriminately trading the entire stock market.**

Instead, maintain a curated universe of companies that have already passed a fundamental or qualitative investment screen.

Conceptually:

```text
Entire market
    ↓
Fundamental / quality screening
    ↓
Approved research universe
    ↓
Trading models
    ↓
Positions
```

This separates two very different questions.

### Question A — Should this company be considered at all?

Handled by the **universe selection process**.

Possible criteria:

* market capitalization
* liquidity
* profitability
* revenue growth
* earnings growth
* debt
* free cash flow
* margins
* return on capital
* valuation
* business quality
* industry attractiveness
* financial stability

### Question B — When should capital enter or leave?

Handled by the **trading model**.

Examples:

* momentum
* moving-average systems
* mean reversion
* volatility breakouts
* RSI systems
* trend following
* relative strength
* multi-factor models

That gives us:

> **good-company filtering + systematic timing**

rather than asking one algorithm to discover everything simultaneously.

---

# 4. Universe Size

The infrastructure should not artificially constrain the universe to 20 symbols merely because Robinhood quote requests are batched.

A reasonable development progression is:

| Phase                  | Universe |
| ---------------------- | -------: |
| Prototype              |    25–50 |
| Early system test      |      100 |
| Scale validation       |      250 |
| Quant research         |      500 |
| Target production test |    1,000 |
| Experimental expansion |   2,000+ |

The exact practical ceiling should be discovered through benchmarking.

For 1,000 stocks with 20 symbols per quote request:

```text
1,000 / 20 = 50 requests
```

Over a 15-minute window:

```text
50 calls / 15 minutes
≈ 3.3 calls per minute
≈ one call every 18 seconds
```

That is operationally modest.

---

# 5. High-Level Architecture

```text
                ┌────────────────────┐
                │  Universe Manager  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Market Data Layer  │
                │ Robinhood MCP etc. │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Historical / Live  │
                │   Data Database    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Feature / Indicator│
                │      Engine        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Strategy Engine    │
                │ Model A / B / C    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    Risk Engine     │
                └─────────┬──────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Backtest      Paper       Shadow
                                      │
                                      ▼
                                Live Execution
                                      │
                                      ▼
                                 Brokerage
```

---

# 6. Technology Philosophy

The first version should be deliberately boring.

Recommended starting stack:

```text
Python
SQLite
cron / systemd timer
Robinhood MCP client
Pandas / NumPy
basic notification service
Git
```

Later:

```text
PostgreSQL
FastAPI
dashboard
Docker
Redis / task queue
cloud or dedicated server
```

Do **not** begin with distributed infrastructure.

The amount of data involved in 15-minute equity research is tiny by modern computing standards.

---

# 7. Scheduling

The production monitor should run approximately every 15 minutes while relevant markets are open.

Conceptually:

```text
scheduler wakes
       ↓
retrieve latest prices
       ↓
store observations
       ↓
update indicators
       ↓
run strategies
       ↓
evaluate risk
       ↓
simulate / notify / execute
       ↓
save complete audit trail
       ↓
process exits
```

No continuously running AI session is required.

A cron job, systemd timer, or equivalent scheduler is enough.

---

# 8. Market Data Layer

The market-data adapter should abstract Robinhood away from the strategy code.

Example interface:

```python
class MarketDataProvider:
    def get_quote(symbol):
        ...

    def get_quotes(symbols):
        ...

    def get_bars(symbol, timeframe, start, end):
        ...

    def get_positions():
        ...
```

Then:

```python
RobinhoodMarketDataProvider
```

could later be replaced or complemented by:

```python
PolygonMarketDataProvider
AlpacaMarketDataProvider
IBKRMarketDataProvider
HistoricalCSVProvider
```

without rewriting trading models.

---

# 9. Local Data Strategy

Robinhood should be treated as a **source**, not as the long-term research database.

Every relevant market observation should be stored locally.

Possible tables:

```text
securities
quotes
bars
fundamentals
universe_membership
strategies
strategy_versions
signals
paper_orders
paper_fills
paper_positions
live_orders
live_fills
portfolio_snapshots
risk_events
alerts
system_runs
```

---

# 10. Universe Membership

Universe selection needs its own history.

Example:

```text
ticker: SNDR
approved: true
approval_date: 2026-08-24
screen_version: quality_v1
reason: passes fundamental screen
```

This becomes important for avoiding historical research mistakes.

We should know:

> Was this company actually considered investable at the time?

rather than silently using today's successful companies in old backtests.

---

# 11. Strategy Interface

Each strategy should follow a standardized contract.

Example:

```python
class Strategy:

    def universe(self):
        pass

    def required_history(self):
        pass

    def calculate_features(self, data):
        pass

    def generate_signal(self, context):
        pass

    def size_position(self, context):
        pass

    def exit_signal(self, context):
        pass
```

A signal might look like:

```json
{
  "ticker": "SNDR",
  "strategy": "momentum_v1",
  "timestamp": "2026-08-24T10:15:00",
  "action": "BUY",
  "confidence": 0.71,
  "target_weight": 0.015
}
```

The strategy should **not directly place orders**.

---

# 12. Risk Engine

Every strategy decision should pass through an independent risk layer.

```text
Strategy says BUY
      ↓
Risk engine checks:
      ↓
position limits
portfolio limits
capital availability
drawdown limits
duplicate orders
liquidity
market status
system health
      ↓
approved / rejected
```

This protects the portfolio from strategy bugs.

Potential controls:

* maximum position size
* maximum portfolio exposure
* maximum sector exposure
* maximum new capital per day
* maximum daily loss
* maximum strategy drawdown
* maximum simultaneous positions
* cooldown after losses
* minimum liquidity
* spread limits
* stale-data detection
* emergency kill switch

---

# 13. Backtesting Engine

Historical testing should simulate:

```text
historical bar
    ↓
strategy sees only information available then
    ↓
signal
    ↓
simulated order
    ↓
simulated fill
    ↓
portfolio update
```

Important protections:

### No look-ahead bias

A strategy cannot use information that was unavailable at that timestamp.

### No impossible fills

If a signal depends on the completed 10:00–10:15 bar, it should not magically trade at information contained inside that same bar before the signal became knowable.

### Slippage

Simulated execution should include realistic friction.

### Bid/ask effects

Where possible:

```text
buy ≈ ask + slippage
sell ≈ bid - slippage
```

### Survivorship bias

Historical tests should not unknowingly use only companies that survived until today.

---

# 14. Backtest Metrics

Every strategy version should automatically calculate:

* total return
* CAGR
* volatility
* maximum drawdown
* Sharpe ratio
* Sortino ratio
* profit factor
* win rate
* average winner
* average loser
* expectancy
* number of trades
* average holding period
* turnover
* time in market
* sector exposure
* largest loss
* consecutive losses
* performance by year
* performance by volatility regime
* benchmark comparison

A model should not be judged primarily on headline return.

---

# 15. Forward Paper Trading

Historical testing is only the first gate.

Successful models should enter **forward paper trading**.

Every 15 minutes:

```text
live market data
      ↓
real strategy
      ↓
paper signal
      ↓
paper risk engine
      ↓
paper order
      ↓
simulated fill
      ↓
paper portfolio
```

This creates an out-of-sample test against a future the model could not have seen.

---

# 16. Paper Portfolio Engine

The paper account should behave like a real brokerage account.

Track:

```text
cash
positions
average cost
realized P&L
unrealized P&L
buying power
open orders
fills
fees
slippage
portfolio value
```

Example:

```text
Starting capital: $100,000

Cash: $81,240
Positions: $18,760
Total equity: $100,430
Realized P&L: +$310
Unrealized P&L: +$120
```

Multiple paper portfolios can coexist.

Example:

```text
momentum_v1
momentum_v2
mean_reversion_v1
trend_v3
ensemble_v1
```

That allows genuine strategy competition.

---

# 17. Strategy Versioning

Every material model change creates a new strategy version.

Never silently modify a running model.

Example:

```text
momentum_v1
momentum_v2
momentum_v3
```

Store:

```text
strategy name
version
parameters
Git commit
creation date
backtest period
paper start date
status
```

A strategy that underperforms should not be retroactively altered and presented as though the new parameters produced the old results.

---

# 18. Promotion Pipeline

Strategies advance through explicit gates.

```text
IDEA
 ↓
BACKTEST
 ↓
OUT-OF-SAMPLE TEST
 ↓
FORWARD PAPER
 ↓
SHADOW LIVE
 ↓
SMALL-CAPITAL LIVE
 ↓
NORMAL LIVE
```

No strategy should jump directly from an attractive backtest into meaningful live capital.

---

# 19. Suggested Promotion Standards

These should eventually become configurable, but conceptually:

### Backtest → Paper

Require:

* sufficient number of trades
* reasonable drawdown
* positive expectancy
* stability across multiple periods
* robustness to parameter variation
* realistic execution assumptions

### Paper → Shadow

Require:

* meaningful forward-testing period
* model behaving similarly to expectations
* no severe operational failures
* execution simulation behaving properly

### Shadow → Small Live

Require:

* stable signals
* stable infrastructure
* correct risk behavior
* no unexplained divergences
* manual review

### Small Live → Scale

Require:

* real execution confirms paper assumptions
* slippage acceptable
* drawdowns inside expected range
* operational reliability established

---

# 20. Shadow Trading

Shadow mode is one of the most important validation stages.

The system does everything except submit the order.

Example:

```text
10:15

Strategy:
BUY XYZ

Desired quantity:
17.3 shares

Expected execution:
$42.18

Risk check:
PASS

Action:
LOG ONLY
```

Later compare:

```text
model expectation
vs.
actual market outcome
```

This helps detect production differences before capital is exposed.

---

# 21. Live Execution Architecture

Live trading should use another adapter.

```python
class ExecutionBroker:

    def review_order(order):
        ...

    def submit_order(order):
        ...

    def cancel_order(order_id):
        ...

    def get_order_status(order_id):
        ...
```

Implementations:

```text
PaperExecutionBroker
RobinhoodExecutionBroker
ManualNotificationBroker
```

This allows the exact same strategy and risk engine to support:

```text
simulation
manual execution
automated execution
```

---

# 22. Human-in-the-Loop Mode

Automated brokerage execution should not be mandatory.

One valid production mode is:

```text
model
 ↓
signal
 ↓
risk approval
 ↓
notification
 ↓
human reviews
 ↓
human trades manually
```

Example notification:

```text
MODEL SIGNAL

Ticker: XYZ
Strategy: trend_v3
Action: SELL
Reason: trailing exit condition triggered

Current price: $48.72
Model threshold: $49.10
Position: 26.4 shares

Open Robinhood to review.
```

This lets us deploy sophisticated monitoring without granting software trading authority.

---

# 23. Automated Live Mode

If later enabled:

```text
signal
 ↓
risk engine
 ↓
broker order review
 ↓
execution
 ↓
verify broker acknowledgement
 ↓
monitor order
 ↓
record fill
 ↓
reconcile position
```

Live trading should never rely on:

```text
send order
assume success
```

Every order must be reconciled against the brokerage.

---

# 24. System Kill Switch

The system needs an immediate global shutdown mechanism.

Example:

```text
TRADING_ENABLED=false
```

When false:

```text
NO NEW LIVE ORDERS
```

Additional automatic triggers:

* stale data
* repeated API failures
* brokerage mismatch
* unexpected account state
* maximum daily loss breached
* model error
* database error
* duplicate-order detection
* excessive latency
* unknown position

Default behavior should be:

> **When uncertain, stop trading.**

---

# 25. Idempotency

A 15-minute scheduler must not accidentally execute the same signal twice.

Every trading decision should have a unique identity.

Example:

```text
strategy + ticker + bar_timestamp + action
```

Before execution:

```text
Has this signal already been processed?
```

If yes:

```text
DO NOTHING
```

This protects against cron retries, crashes, and duplicate runs.

---

# 26. Observability

Every scheduled cycle should generate telemetry.

Example:

```text
Run: 2026-08-24 10:15

Universe: 500
Quotes requested: 500
Quotes received: 500
API calls: 25
Errors: 0
429 responses: 0

Data retrieval: 7.4 sec
Feature calculation: 0.8 sec
Strategy evaluation: 0.4 sec
Database write: 0.3 sec

Signals: 4
Risk approved: 3
Rejected: 1
Orders: 0
Alerts: 3

Total runtime: 9.1 sec
```

This is also how we discover Robinhood's practical market-data limits.

---

# 27. Benchmarking Robinhood MCP

Do not assume the limit.

Measure it.

Test sequence:

```text
100 symbols
↓
250
↓
500
↓
1,000
↓
2,000
↓
higher only if useful
```

Measure:

* request latency
* cycle duration
* throttling
* failed symbols
* authentication failures
* server errors
* rate-limit behavior
* response consistency

Automatic exponential backoff should handle throttling.

---

# 28. Notifications

Notifications should distinguish severity.

### Information

```text
Paper trade executed.
```

### Action

```text
Manual trade review required.
```

### Warning

```text
Market data delayed.
```

### Critical

```text
Trading disabled automatically.
Broker position mismatch detected.
```

Potential channels:

```text
desktop
email
push notification
SMS
Discord
Slack
Telegram
```

One channel can be selected initially.

---

# 29. Security Model

Secrets should never be stored in source code.

Use:

```text
environment variables
OS credential store
secret manager
encrypted token storage
```

Protect:

* Robinhood authentication tokens
* MCP credentials
* notification credentials
* database credentials
* market-data API keys

Never commit secrets to Git.

---

# 30. Financial Data vs Trading Data

Keep the broader personal-finance project separate.

### Robinhood MCP

Primary role:

```text
investment positions
market quotes
historical prices
research data
trading
```

### Plaid

Primary role:

```text
banking
mortgage
business accounts
transactions
cash flow
net worth
financial aggregation
```

Eventually they can feed a shared financial dashboard, but they should remain logically separate services.

---

# 31. AI's Role

Routine model operation:

```text
NO AI REQUIRED
```

AI is useful for:

* developing strategies
* reviewing strategy code
* analyzing failures
* researching companies
* interpreting results
* generating experiment ideas
* examining anomalies
* producing performance reports
* debugging infrastructure
* comparing model versions

This keeps the critical trading loop deterministic, testable, auditable, and inexpensive.

---

# 32. Research Discipline

Every experiment should record:

```text
hypothesis
strategy version
universe version
parameters
training period
validation period
results
decision
```

Example:

```text
Experiment:
EXP-0042

Hypothesis:
20/80 EMA cross performs better when restricted
to companies with positive free cash flow.

Universe:
quality_screen_v3

Strategy:
ema_cross_v6

Result:
Rejected

Reason:
Return improved but drawdown and turnover became unacceptable.
```

This prevents random strategy tweaking.

---

# 33. Project Directory

Initial repository:

```text
trading-platform/
│
├── config/
│
├── data/
│
├── db/
│
├── market_data/
│   ├── base.py
│   └── robinhood.py
│
├── universe/
│   ├── screens/
│   └── manager.py
│
├── indicators/
│
├── strategies/
│   ├── base.py
│   ├── momentum_v1.py
│   └── mean_reversion_v1.py
│
├── risk/
│
├── execution/
│   ├── base.py
│   ├── paper.py
│   ├── manual.py
│   └── robinhood.py
│
├── backtest/
│
├── portfolio/
│
├── notifications/
│
├── scheduler/
│
├── analytics/
│
├── tests/
│
└── logs/
```

---

# 34. Phase 1 — Connectivity Proof

Goal:

> Prove that a normal non-AI program can retrieve Robinhood MCP data reliably.

Build:

```text
authenticate
get arbitrary ticker quote
get positions
exit
```

Then execute again from a fresh process.

Success criterion:

> Subsequent runs work unattended with an acceptable authentication lifecycle.

---

# 35. Phase 2 — Market Data Collector

Build:

```text
universe loader
20-symbol batching
quote fetcher
retry logic
rate-limit detection
SQLite storage
telemetry
```

Test at:

```text
100
250
500
1,000
2,000
```

Determine the practical operating ceiling.

---

# 36. Phase 3 — Historical Dataset

Collect and/or retrieve historical OHLCV data.

Normalize it into a consistent schema.

Example:

```text
timestamp
ticker
open
high
low
close
volume
source
```

All strategies use the same normalized format.

---

# 37. Phase 4 — First Backtesting Engine

Support:

```text
initial capital
positions
cash
market orders
slippage
commissions / fees
signals
fills
portfolio valuation
metrics
```

Begin with one deliberately simple model to validate the engine.

The objective is not initially to discover alpha.

The objective is to prove:

> **the simulator behaves correctly.**

---

# 38. Phase 5 — Live Paper Trader

Connect:

```text
scheduler
    ↓
Robinhood market data
    ↓
strategy
    ↓
paper execution
    ↓
paper portfolio
```

Run without human intervention during market hours.

Every decision gets stored.

---

# 39. Phase 6 — Research Dashboard

Display:

```text
Universe
Strategies
Paper portfolios
Current signals
Positions
Trades
Returns
Drawdowns
Model rankings
System health
```

This can initially be a simple local web application.

---

# 40. Phase 7 — Human Alert Trading

Introduce:

```text
signal
 ↓
risk engine
 ↓
manual-execution notification
```

No automated brokerage authority yet.

This is the first production use of model recommendations.

---

# 41. Phase 8 — Shadow Brokerage Integration

Connect to the real brokerage state.

Generate would-be orders using actual account conditions.

Do not submit them.

Compare:

```text
paper assumptions
vs.
real brokerage reality
```

---

# 42. Phase 9 — Tiny-Capital Live Validation

Only after passing previous stages.

Use intentionally small capital.

Objective:

> Validate execution mechanics, not maximize return.

Measure:

* slippage
* latency
* broker behavior
* fractional-share behavior
* order rejection
* partial fills
* position reconciliation
* operational reliability

---

# 43. Phase 10 — Controlled Scaling

Capital allocation should increase based on **demonstrated system performance**, not confidence.

Possible framework:

```text
Paper only
    ↓
$100 live allocation
    ↓
$500
    ↓
$1,000
    ↓
portfolio percentage allocation
```

Scaling rules should themselves be systematic.

---

# 44. Initial 30-Day Development Roadmap

## Week 1 — Connectivity + data

Build:

* repository
* Robinhood MCP client
* authentication proof
* arbitrary ticker retrieval
* quote batching
* SQLite schema
* logging

Deliverable:

> Reliable scheduled quote collector.

---

## Week 2 — Universe + historical research

Build:

* reconstructed fundamental screen
* approved universe
* historical price loader
* indicator library
* first strategy interface

Deliverable:

> Research-ready dataset.

---

## Week 3 — Backtest engine

Build:

* portfolio simulator
* signal processing
* fill model
* slippage
* metrics
* experiment logging

Deliverable:

> Reproducible strategy backtests.

---

## Week 4 — Forward paper trader

Build:

* 15-minute scheduler
* live data retrieval
* paper execution
* notifications
* dashboard/report
* system health telemetry

Deliverable:

> Autonomous live-market paper-trading system.

---

# 45. Immediate MVP

The first useful version does **not** need live brokerage execution.

MVP:

```text
Curated universe
        ↓
Robinhood market data
        ↓
15-minute local collector
        ↓
SQLite database
        ↓
Trading strategy
        ↓
Paper portfolio
        ↓
Performance report
```

If this works reliably, the foundational engineering problem is solved.

Everything after that is controlled expansion.

---

# 46. Definition of Success

The system succeeds when we can say:

> We can define an investment universe, create a trading hypothesis, encode it as deterministic software, backtest it without obvious methodological errors, forward-test it against live unseen data, measure its performance objectively, and promote it toward real capital through explicit risk gates.

The goal is **not**:

> Build a bot that trades.

The goal is:

> **Build a scientific trading research and execution platform capable of determining which strategies deserve capital.**

---

# 47. Long-Term Vision

Eventually:

```text
Fundamental research
        +
Quantitative research
        +
Market data
        +
Strategy laboratory
        +
Paper trading
        +
Live portfolio management
        +
Personal financial intelligence
```

becomes one integrated private investment operating system.

The system should accumulate its own proprietary history:

```text
our screened universes
our market observations
our experiments
our failed models
our successful models
our paper results
our live results
```

Over time, the value is no longer merely the Robinhood MCP connection.

The value becomes the **research dataset, experiment history, model library, and disciplined promotion framework that we build on top of it.**

---

# Architectural North Star

> **Use AI to help build the laboratory.
> Use deterministic software to run the experiments.
> Use paper markets to prove the models.
> Use strict risk controls before exposing real capital.
> Let demonstrated results—not excitement—determine what gets deployed.**
