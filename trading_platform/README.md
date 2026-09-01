# Trading Platform

Implementation of `NEW_PLAN/trading-plan.md` — a trading research, paper
trading and (eventually) live execution platform. One strategy codebase runs
unchanged across all four operating modes; only the adapters change (§2).

```text
market data → deterministic model → signal → risk engine → paper / shadow / live execution
```

## Status vs. the plan

| Plan section | Status | Where |
|---|---|---|
| §2 One strategy, four modes | ✅ implemented | `backtest/engine.py` + `scheduler/cycle.py` share strategy, risk & portfolio code; only brokers differ |
| §3/§4 Curated universe, sizing | ✅ implemented | `universe/manager.py` (quality_v1 screen; seeded from `watchlist/WATCHLIST - Sheet1.csv`) |
| §5/§6 Architecture, boring stack | ✅ implemented | Python + SQLite + pandas + cron. No distributed anything. |
| §7 Scheduler | ✅ implemented | `scheduler/cycle.py` one-shot cycle process; cron below |
| §8 Market data abstraction | ✅ implemented, **Robinhood pending** | `market_data/` — `alpha_vantage.py` works today; `robinhood_mcp.py` is a stub until Robinhood MCP access exists |
| §9 Local data strategy | ✅ implemented | `db/schema.sql` (all §9 tables) |
| §10 Universe membership history | ✅ implemented | `universe_membership` table, point-in-time queries (`universe-show --as-of`) |
| §11 Strategy interface | ✅ implemented | `strategies/base.py` + 3 example strategies |
| §12 Risk engine | ✅ implemented | `risk/engine.py` — position/exposure/sector caps, daily deployment cap, drawdown breakers, cooldown, liquidity, spread, stale data, kill switch |
| §13 Backtest discipline | ✅ implemented + **unit-tested** | `backtest/engine.py`: decisions on completed bar *i*, fills at bar *i+1* OPEN; slippage charged against trade direction |
| §14 Metrics | ✅ implemented | `backtest/metrics.py` (return, CAGR, vol, maxDD, Sharpe, Sortino, PF, win rate, expectancy, holding period, by-year, by-vol-regime, benchmark) |
| §15/§16 Paper trading + portfolios | ✅ implemented | `portfolio/paper.py`, `scheduler/cycle.py`, multiple named portfolios |
| §17 Strategy versioning | ✅ implemented | `strategy_versions` table; params + git commit recorded on every backtest |
| §18/§19 Promotion pipeline | ⚙️ tracked | `strategy_versions.status` (IDEA→BACKTEST→…); gate criteria reviewed manually, not yet auto-evaluated |
| §20 Shadow trading | ✅ implemented | `run-cycle --mode shadow` logs would-be orders; `shadow-report` compares expectations |
| §21 Execution adapters | ✅ paper + manual; ❌ live | `execution/` — `PaperExecutionBroker`, `ManualNotificationBroker`; `RobinhoodExecutionBroker` deliberately absent |
| §22 Human-in-the-loop | ✅ implemented | `execution/manual.py` sends §22-style notifications |
| §23/§43 Automated live + scaling | ❌ not built (by design) | Requires real brokerage access; the interfaces are ready |
| §24 Kill switch | ✅ implemented | `TRADING_ENABLED` env **and** persistent flag file; default = stop trading; `cli kill-switch --on/--off/--status` |
| §25 Idempotency | ✅ implemented + **unit-tested** | `signal_id = strategy+ticker+bar_timestamp+action`, UNIQUE in DB |
| §26 Observability | ✅ implemented | `system_runs` telemetry per cycle; `report` shows recent runs |
| §27 Robinhood benchmarking | ⏸ blocked | No Robinhood access yet; harness exists once a provider lands |
| §28 Notifications | ✅ implemented | Severity levels; log + file inbox always, webhook via `ALERT_WEBHOOK_URL`, desktop best-effort |
| §29 Security | ✅ implemented | Secrets only via env vars (`ALPHA_VANTAGE_API_KEY`, `ALERT_WEBHOOK_URL`) |
| §30 Financial vs trading data | n/a | Nothing built here touches Plaid/banking |
| §31 AI's role | ✅ honored | The trading loop is 100% deterministic software |
| §32 Research discipline | ✅ implemented | `experiments` table; every backtest records hypothesis/params/results; `cli experiments` |
| §33 Directory layout | ✅ implemented | Mirrors the plan inside this package |
| §34 Phase 1 connectivity proof | ✅ (adapted) | `cli prove-connectivity` — works today via Alpha Vantage; Robinhood pending |
| §35 Phase 2 collector | ✅ (adapted) | `cli collect` (Alpha Vantage free tier = sequential calls w/ rate limiter; batching arrives with Robinhood's 20-symbol quotes) |
| §36 Phase 3 historical dataset | ✅ (adapted) | `cli import-legacy` imported the repo's existing daily OHLCV; normalized `bars` schema |
| §37 Phase 4 backtest engine | ✅ implemented | Validated with the deliberately simple `sma_cross:v1` first |
| §38 Phase 5 live paper trader | ✅ implemented | `cli run-cycle --mode paper` — runs unattended, stores every decision |
| §39 Phase 6 dashboard | ◐ minimal | `cli report` (text). Rich web dashboard deferred |
| §40-§42 Human alert / shadow broker / tiny-capital live | ◐ / ⏸ / ⏸ | Manual-notification mode works; brokerage stages need Robinhood access |

**Bottom line:** the entire MVP of §45 is implemented and tested. The only
unimplementable-today pieces are the Robinhood-dependent stages (live
execution, MCP benchmarking) — exactly the parts that require external
credentials, which the plan itself gates behind Phases 1–2.

## Quick start

```bash
# one-time
python3 -m trading_platform.cli init-db
python3 -m trading_platform.cli import-legacy --source trading_data.db
python3 -m trading_platform.cli universe-import --csv "watchlist/WATCHLIST - Sheet1.csv"

# research
python3 -m trading_platform.cli backtest --strategy sma_cross --symbols AAPL,GOOGL,META,MSFT

# operations (cron / manual)
python3 -m trading_platform.cli prove-connectivity           # §34 health check
python3 -m trading_platform.cli collect --symbols SPY,QQQ    # §35 collector (needs API key)
DATA_PROVIDER=local_db python3 -m trading_platform.cli run-cycle --mode paper
python3 -m trading_platform.cli run-cycle --mode shadow      # §20 log-only
python3 -m trading_platform.cli report
```

### The 15-minute schedule (§7)

```cron
# paper trading cycle, every 15 minutes, US market hours only (ET approximation)
*/15 9-15 * * 1-5  cd /mnt/storage/repos/investing && DATA_PROVIDER=alpha_vantage \
                   ALPHA_VANTAGE_API_KEY=... python3 -m trading_platform.cli run-cycle --mode paper
```

The cycle process fetches quotes → stores them → runs strategies → risk engine
→ paper execution → telemetry → **exits**. systemd timers work identically.

## Kill switch (§24)

Both must be true for shadow/live orders:
1. env `TRADING_ENABLED=true`
2. flag file present: `python3 -m trading_platform.cli kill-switch --on`

Everything else defaults to **stop trading**.

## Tests

```bash
python3 -m pytest trading_platform/tests/ -q
```

33 tests cover the invariants that matter: no look-ahead fills (decision bar
*i* → fill at bar *i+1* open), slippage direction, portfolio accounting,
every risk gate, kill switch, §25 idempotency, metrics math, and a full
end-to-end scheduler cycle.

## Adding a Robinhood provider later (§8, §34)

Implement `trading_platform/market_data/robinhood_mcp.py` (quotes + bars +
positions), add `RobinhoodExecutionBroker` in `execution/`, register both in
`cli.make_provider()`. No strategy or risk code changes.
