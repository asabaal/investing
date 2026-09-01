"""One 15-minute production cycle (plan §7, §15, §45).

    scheduler wakes
        -> retrieve latest prices
        -> store observations
        -> update indicators
        -> run strategies
        -> evaluate risk
        -> simulate / notify
        -> save complete audit trail + telemetry
        -> process exits

No continuously running AI session; a cron/systemd timer is enough (§7).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from ..config import Config
from ..db.database import PlatformDatabase, Signal
from ..execution.base import Order, make_order_id
from ..execution.manual import ManualNotificationBroker
from ..execution.paper import PaperExecutionBroker
from ..market_data.base import MarketDataProvider
from ..notifications.channels import NotificationRouter
from ..portfolio.paper import PaperPortfolio
from ..risk.engine import RiskContext, RiskEngine
from ..strategies.base import Strategy, StrategyContext


def is_market_open(now: pd.Timestamp | None = None) -> bool:
    """US equity regular hours, naive ET approximation (v1)."""
    now = now or pd.Timestamp.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


class PaperTradingCycle:
    def __init__(self, config: Config, db: PlatformDatabase, provider: MarketDataProvider,
                 strategies: list[Strategy], portfolio: PaperPortfolio | None = None,
                 router: NotificationRouter | None = None, timeframe: str = "15min"):
        self.config = config
        self.db = db
        self.provider = provider
        self.strategies = strategies
        self.portfolio = portfolio or PaperPortfolio(
            starting_capital=100_000.0, name="paper_default", execution=config.execution)
        self.router = router or NotificationRouter(alert_dir=config.alert_dir)
        self.router.attach_db(db)
        self.timeframe = timeframe

    # ---- main entry ---------------------------------------------------------
    def run_cycle(self, force: bool = False, now: pd.Timestamp | None = None) -> dict:
        t0 = time.time()
        timings: dict[str, float] = {}
        now = now or pd.Timestamp.now()
        result: dict = {"mode": self.config.mode, "timestamp": str(now)}

        market_open = is_market_open(now)
        if not market_open and not force:
            result["skipped"] = "market closed"
            self._telemetry(t0, timings, result, status="OK", notes="market closed")
            return result

        # -- universe (§3): approved symbols intersected with strategy universes
        universe: list[str] = sorted({s for st in self.strategies for s in st.universe()})
        result["universe_size"] = len(universe)

        # -- 1. retrieve + store latest prices (§7)
        t = time.time()
        quotes = self.provider.get_quotes(universe)
        timings["data_retrieval"] = time.time() - t
        valid = {s: q for s, q in quotes.items() if ":error" not in q.source}
        rows = [{
            "symbol": s, "timestamp": q.timestamp, "bid": q.bid, "ask": q.ask,
            "last": q.price, "volume": q.volume, "source": q.source,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        } for s, q in valid.items()]
        if rows:
            self.db.store_quotes(rows)
        result["quotes_requested"] = len(universe)
        result["quotes_received"] = len(valid)
        result["errors"] = len(quotes) - len(valid)

        prices = {s: q.price for s, q in valid.items()}
        qts = {s: pd.Timestamp(q.timestamp) for s, q in valid.items()}
        spreads = {s: q.spread_pct for s, q in valid.items()}

        # -- 2. features + strategies (§7) -- decisions use the latest history
        t = time.time()
        signals: list[Signal] = []
        for strat in self.strategies:
            for sym in strat.universe():
                if sym not in prices:
                    continue
                hist = self.db.get_bars(sym, timeframe=self.timeframe)
                if hist.empty:
                    hist = self.db.get_bars(sym, timeframe="daily")
                if hist.empty:
                    continue
                hist = pd.concat([hist, _quote_bar(hist, prices[sym], qts[sym])],
                                 ignore_index=True)
                feats = strat.calculate_features(hist)
                need = strat.required_history()
                if len(feats) < need:
                    continue
                ctx = StrategyContext(sym, feats.iloc[-1]["timestamp"], feats,
                                      self.portfolio, prices)
                pos = self.portfolio.positions.get(sym)
                if pos is not None and pos.is_open:
                    if strat.exit_signal(ctx):
                        signals.append(strat.signal(ctx, "EXIT"))
                else:
                    sig = strat.generate_signal(ctx)
                    if sig is not None:
                        signals.append(sig)
        timings["strategy_evaluation"] = time.time() - t

        # -- 3. risk evaluation + execution (§12, §15, §20)
        t = time.time()
        risk = RiskEngine(self.config, self.db)
        approved = rejected = 0
        orders_placed = 0
        kill_switch = not self.config.trading_enabled

        broker: PaperExecutionBroker | ManualNotificationBroker
        if self.config.mode == "shadow":
            # §20: does everything except submit; records the would-be order.
            broker = PaperExecutionBroker(self.portfolio, self.config.execution,
                                          db=self.db, mode="shadow",
                                          kill_switch_on=True)
        else:
            broker = PaperExecutionBroker(self.portfolio, self.config.execution,
                                          db=self.db, mode="paper",
                                          kill_switch_on=kill_switch)

        dollar_vols = {}
        for sym, p in prices.items():
            bars = self.db.get_bars(sym, timeframe="daily")
            if len(bars) >= 2:
                dollar_vols[sym] = float((bars["close"] * bars["volume"]).tail(20).mean())

        for sig in signals:
            # idempotency (§25): strategy+ticker+bar_timestamp+action
            if not self.db.insert_signal(sig, status="NEW"):
                continue  # already processed - DO NOTHING
            rc = RiskContext(prices=prices, quote_timestamps=qts, dollar_volumes=dollar_vols,
                             spreads=spreads, now=now)
            decision = risk.approve(sig, self.portfolio, rc, mode=self.config.mode)
            if not decision.approved:
                rejected += 1
                self.db.update_signal_status(sig.signal_id or sig.identity(), "REJECTED",
                                             "; ".join(decision.reasons))
                continue
            approved += 1
            price = prices.get(sig.ticker, sig.price_hint or 0.0)
            qty = sig.quantity
            if not qty:
                equity = self.portfolio.equity(prices)
                weight = sig.target_weight or self.config.risk.max_position_pct
                qty = equity * weight / price if price > 0 else 0.0
            order = Order(order_id=make_order_id(self.config.mode), ticker=sig.ticker,
                          side="BUY" if sig.action == "BUY" else "SELL", quantity=float(qty),
                          mode=self.config.mode, portfolio=self.portfolio.name,
                          signal_id=sig.signal_id or sig.identity(),
                          expected_price=price,
                          strategy=f"{sig.strategy}:{sig.version}",
                          created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
            res = broker.submit_order(order, price)
            self.db.update_signal_status(sig.signal_id or sig.identity(),
                                         "FILLED" if res.filled else "SHADOW"
                                         if self.config.mode == "shadow" else
                                         ("FILLED" if res.filled else "REJECTED"),
                                         None if res.filled else res.note)
            orders_placed += 1
            self.router.send(
                "INFORMATION",
                f"{self.config.mode} order {'executed' if res.filled else 'logged'}: "
                f"{sig.action} {sig.ticker}",
                f"Strategy: {sig.strategy}:{sig.version}\nQty: {qty:.4g} @ ~${price:.2f}"
                + (f"\nNote: {res.note}" if res.note else ""),
                signal_id=sig.signal_id)

        timings["risk_and_execution"] = time.time() - t

        # -- 4. audit trail: snapshot + persist positions (§16)
        t = time.time()
        snapshot_prices = prices or {}
        equity = self.portfolio.equity(snapshot_prices)
        self.db.save_snapshot(self.portfolio.name, now.strftime("%Y-%m-%dT%H:%M:%S"),
                              self.portfolio.cash,
                              self.portfolio.positions_value(snapshot_prices), equity,
                              self.portfolio.realized_pl(),
                              self.portfolio.unrealized_pl(snapshot_prices))
        self.portfolio.save_positions(self.db)
        timings["database_write"] = time.time() - t

        result.update({"signals": len(signals), "risk_approved": approved,
                       "risk_rejected": rejected, "orders": orders_placed})
        self._telemetry(t0, timings, result, status="OK")
        return result

    # ---- observability (§26) -------------------------------------------------
    def _telemetry(self, t0: float, timings: dict, result: dict, status: str,
                   notes: str | None = None) -> None:
        runtime = time.time() - t0
        self.db.record_run(
            run_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            mode=self.config.mode, status=status,
            universe_size=result.get("universe_size"),
            quotes_requested=result.get("quotes_requested"),
            quotes_received=result.get("quotes_received"),
            api_calls=getattr(self.provider, "api_calls", None),
            errors=result.get("errors"), rate_limited=None,
            signals=result.get("signals"), risk_approved=result.get("risk_approved"),
            risk_rejected=result.get("risk_rejected"), orders=result.get("orders"),
            runtime_sec=round(runtime, 3),
            timings_json=str({k: round(v, 3) for k, v in timings.items()}),
            notes=notes,
        )
        result["runtime_sec"] = round(runtime, 3)
        result["timings"] = {k: round(v, 3) for k, v in timings.items()}


def _quote_bar(hist: pd.DataFrame, price: float, ts: pd.Timestamp) -> pd.DataFrame:
    """Latest live price as a provisional bar so strategies see current data."""
    timeframe = "1min"
    row = pd.DataFrame([{"timestamp": ts, "open": price, "high": price,
                         "low": price, "close": price, "volume": 0}])
    last_hist_ts = hist.iloc[-1]["timestamp"] if len(hist) else None
    if last_hist_ts is not None and ts <= last_hist_ts:
        row["timestamp"] = last_hist_ts + pd.Timedelta(minutes=1)
    return row
