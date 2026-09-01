"""Backtesting engine (plan §13).

Timeline discipline (the "no impossible fills" rule):

    completed bar i  ->  strategy decides  ->  order queued
    bar i+1 OPENS    ->  order fills at open +/- slippage

A strategy can never see bar i+1 before trading on bar i's information.
The SAME risk engine and portfolio code run here and in live paper mode (§2).
"""

from __future__ import annotations

import pandas as pd

from ..config import Config, load_config
from ..db.database import PlatformDatabase, Signal
from ..execution.base import Order, make_order_id
from ..execution.paper import PaperExecutionBroker
from ..portfolio.paper import PaperPortfolio
from ..risk.engine import RiskContext, RiskEngine
from ..strategies.base import Strategy, StrategyContext


class BacktestEngine:
    def __init__(self, strategy: Strategy, db: PlatformDatabase | None = None,
                 config: Config | None = None, initial_capital: float = 100_000.0,
                 benchmark: str | None = None, store_signals: bool = False):
        self.strategy = strategy
        self.db = db
        self.config = config or load_config()
        self.initial_capital = initial_capital
        self.benchmark = benchmark or self.config.benchmark
        self.store_signals = store_signals

    def run(self, data: dict[str, pd.DataFrame], start: str | None = None,
            end: str | None = None) -> "BacktestResult":
        strat = self.strategy
        feat = {sym: strat.calculate_features(df) for sym, df in data.items()}
        need = strat.required_history()

        # Global timeline: union of timestamps, only bars where the oldest
        # symbol has enough history to be tradeable.
        timeline = sorted(set().union(*[set(df["timestamp"]) for df in feat.values()
                                        if len(df) > need]))
        if start:
            timeline = [t for t in timeline if t >= pd.Timestamp(start)]
        if end:
            timeline = [t for t in timeline if t <= pd.Timestamp(end)]

        portfolio = PaperPortfolio(starting_capital=self.initial_capital,
                                   name=f"backtest_{strat.name}_{strat.version}",
                                   execution=self.config.execution)
        broker = PaperExecutionBroker(portfolio, self.config.execution,
                                      db=self.db, mode="backtest")
        risk = RiskEngine(self.config, self.db) if self.db else None

        rows = []  # equity curve rows: (timestamp, equity, exposure)
        pending: list[tuple[Signal, float]] = []  # signals awaiting next-open fill
        day_start_equity = self.initial_capital
        cur_day = None

        for t in timeline:
            prices_at_close: dict[str, float] = {}
            opens_now: dict[str, float] = {}
            for sym, df in feat.items():
                hist = df[df["timestamp"] <= t]
                if len(hist) == 0:
                    continue
                if hist.iloc[-1]["timestamp"] == t:
                    prices_at_close[sym] = float(hist.iloc[-1]["close"])
                    opens_now[sym] = float(hist.iloc[-1]["open"])
                else:
                    opens_now[sym] = float(hist.iloc[-1]["close"])

            # new day -> reset daily-loss baseline (knowable prices only)
            day = t.strftime("%Y-%m-%d")
            if day != cur_day:
                cur_day = day
                day_start_equity = portfolio.equity(opens_now) or day_start_equity

            # ---- 1. execute orders queued from the previous bar at THIS open.
            # Only the open is knowable now, so risk checks and fills use it.
            for sig, _decision_price in pending:
                fill_ref = opens_now.get(sig.ticker, _decision_price)
                self._execute(sig, fill_ref, t, portfolio, broker,
                              risk, opens_now, feat, need)
            pending = []

            # ---- 2. observe completed bar t, queue new decisions
            for sym, df in feat.items():
                hist = df[df["timestamp"] <= t]
                if len(hist) < need or hist.iloc[-1]["timestamp"] != t:
                    continue
                ctx = StrategyContext(ticker=sym, timestamp=t, data=hist,
                                      portfolio=portfolio, prices=prices_at_close)
                pos = portfolio.positions.get(sym)
                if pos is not None and pos.is_open:
                    if strat.exit_signal(ctx):
                        pending.append((strat.signal(ctx, "EXIT"), prices_at_close[sym]))
                else:
                    sig = strat.generate_signal(ctx)
                    if sig is not None:
                        pending.append((sig, prices_at_close[sym]))

            # ---- 3. mark to market at bar close
            equity = portfolio.equity(prices_at_close) if prices_at_close else portfolio.cash
            exposure = (portfolio.positions_value(prices_at_close) / equity
                        if equity > 0 and prices_at_close else 0.0)
            rows.append((t, equity, exposure))

        curve = pd.DataFrame(rows, columns=["timestamp", "equity", "exposure"]) \
            .set_index("timestamp")
        return BacktestResult(strategy=strat, portfolio=portfolio, equity_curve=curve,
                              engine=self)

    # ---- order execution through the SAME risk layer -----------------------
    def _execute(self, sig: Signal, decision_price: float, fill_time: pd.Timestamp,
                 portfolio: PaperPortfolio, broker: PaperExecutionBroker,
                 risk: RiskEngine | None, prices: dict, feat: dict, need: int) -> None:
        # Idempotency (§25): skip signals already processed (cron retries etc.)
        if self.db is not None:
            if not self.db.insert_signal(sig, status="NEW"):
                return  # duplicate

        ctx = StrategyContext(ticker=sig.ticker, timestamp=fill_time,
                              data=feat[sig.ticker][feat[sig.ticker]["timestamp"] <= fill_time]
                              if sig.ticker in feat else pd.DataFrame(),
                              portfolio=portfolio, prices=prices)
        ctx.prices = dict(prices)
        ctx.prices.setdefault(sig.ticker, decision_price)

        if risk is not None:
            dd = self._strategy_drawdowns(portfolio).get(sig.strategy)
            rc = RiskContext(prices=ctx.prices, now=fill_time,
                             strategy_drawdowns={sig.strategy: dd} if dd is not None else {})
            decision = risk.approve(sig, portfolio, rc, mode="backtest")
            if not decision.approved:
                if self.db is not None:
                    self.db.update_signal_status(sig.signal_id or sig.identity(),
                                                 "REJECTED", "; ".join(decision.reasons))
                return

        side = "BUY" if sig.action == "BUY" else "SELL"
        qty = sig.quantity
        if qty is None:
            equity = portfolio.equity(ctx.prices)
            weight = sig.target_weight or self.config.risk.max_position_pct
            qty = equity * weight / decision_price if decision_price > 0 else 0.0
        order = Order(order_id=make_order_id("backtest"), ticker=sig.ticker, side=side,
                      quantity=float(qty), mode="backtest",
                      portfolio=portfolio.name, signal_id=sig.signal_id or sig.identity(),
                      expected_price=decision_price, strategy=f"{sig.strategy}:{sig.version}",
                      created_at=str(fill_time))
        result = broker.submit_order(order, decision_price)
        if self.db is not None:
            self.db.update_signal_status(
                sig.signal_id or sig.identity(),
                "FILLED" if result.filled else "REJECTED",
                None if result.filled else result.note)

    def _strategy_drawdowns(self, portfolio: PaperPortfolio) -> dict[str, float]:
        """Realized-only per-strategy drawdown - a coarse circuit breaker."""
        cum: dict[str, list[float]] = {}
        peak: dict[str, float] = {}
        dd: dict[str, float] = {}
        for tr in portfolio.trades:
            cum.setdefault(tr.strategy, []).append(tr.pnl)
        for s, pnls in cum.items():
            eq = 0.0
            peak_s = 0.0
            worst = 0.0
            base = max(sum(pnls), 0.0) + 1.0
            for p in pnls:
                eq += p
                peak_s = max(peak_s, eq)
                if base > 0:
                    worst = min(worst, (eq - peak_s) / base)
            dd[s] = worst
        return dd


class BacktestResult:
    def __init__(self, strategy: Strategy, portfolio: PaperPortfolio,
                 equity_curve: pd.DataFrame, engine: BacktestEngine):
        self.strategy = strategy
        self.portfolio = portfolio
        self.equity_curve = equity_curve
        self.engine = engine

    def metrics(self, benchmark_data: pd.DataFrame | None = None) -> dict:
        from .metrics import compute_metrics

        bench_equity = None
        if benchmark_data is not None and not self.equity_curve.empty and not benchmark_data.empty:
            b = benchmark_data.copy()
            b = b[(b["timestamp"] >= self.equity_curve.index[0])
                  & (b["timestamp"] <= self.equity_curve.index[-1])]
            if not b.empty:
                bench_equity = pd.Series(b["close"].values, index=b["timestamp"])
        return compute_metrics(
            equity=self.equity_curve["equity"],
            exposure=self.equity_curve["exposure"],
            trades=self.portfolio.trades,
            benchmark_equity=bench_equity,
        )

    def summary(self, benchmark_data: pd.DataFrame | None = None) -> str:
        m = self.metrics(benchmark_data)
        lines = [
            f"Strategy: {self.strategy.name}:{self.strategy.version}",
            f"Period: {m.get('start')} -> {m.get('end')} ({m.get('days')} days)",
            f"Total return: {m.get('total_return', 0):+.2%}   CAGR: {m.get('cagr', 0):+.2%}",
            f"Max drawdown: {m.get('max_drawdown', 0):.2%}   Vol: {m.get('volatility', 0):.2%}",
            f"Sharpe: {m.get('sharpe', 0):.2f}   Sortino: {m.get('sortino', 0):.2f}",
            f"Trades: {m.get('number_of_trades', 0)}   Win rate: {m.get('win_rate', 0):.1%}   "
            f"Profit factor: {m.get('profit_factor', 0):.2f}",
            f"Expectancy/trade: ${m.get('expectancy', 0):,.2f}   Avg hold: "
            f"{m.get('avg_holding_days', 0):.1f}d",
            f"Time in market: {m.get('time_in_market', 0):.1%}",
        ]
        if "benchmark" in m:
            b = m["benchmark"]
            lines.append(f"Benchmark ({self.engine.benchmark}): {b['total_return']:+.2%} "
                         f"(sharpe {b['sharpe']:.2f}, maxdd {b['max_drawdown']:.2%})  "
                         f"excess: {m['excess_return_vs_benchmark']:+.2%}")
        return "\n".join(lines)
