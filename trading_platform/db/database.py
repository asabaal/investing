"""SQLite access layer for the platform (plan §9, §25, §26)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from ..config import Config, load_config

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass
class Signal:
    """Plan §11 signal envelope."""

    ticker: str
    strategy: str
    version: str
    bar_timestamp: str
    action: str                 # BUY | SELL | EXIT
    confidence: float | None = None
    target_weight: float | None = None
    quantity: float | None = None
    price_hint: float | None = None
    signal_id: str | None = None

    def identity(self) -> str:
        """§25 idempotency identity: strategy + ticker + bar_timestamp + action."""
        raw = f"{self.strategy}|{self.ticker}|{self.bar_timestamp}|{self.action}"
        import hashlib

        return hashlib.sha1(raw.encode()).hexdigest()[:24]


class PlatformDatabase:
    def __init__(self, db_path: str | Path | None = None, config: Config | None = None):
        self.config = config or load_config()
        self.db_path = Path(db_path) if db_path else self.config.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # -- setup -------------------------------------------------------------
    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- securities --------------------------------------------------------
    def upsert_security(self, symbol: str, name: str | None = None,
                        sector: str | None = None, industry: str | None = None,
                        active: bool = True) -> None:
        self._conn.execute(
            """INSERT INTO securities (symbol, name, sector, industry, active, first_seen)
               VALUES (?, ?, ?, ?, ?, COALESCE((SELECT first_seen FROM securities WHERE symbol = ?), ?))
               ON CONFLICT(symbol) DO UPDATE SET
                 name=COALESCE(excluded.name, name),
                 sector=COALESCE(excluded.sector, sector),
                 industry=COALESCE(excluded.industry, industry),
                 active=excluded.active""",
            (symbol, name, sector, industry, int(active), symbol, pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._conn.commit()

    def securities(self, active_only: bool = True) -> list[dict]:
        q = "SELECT * FROM securities" + (" WHERE active = 1" if active_only else "")
        return [dict(r) for r in self._conn.execute(q).fetchall()]

    def set_sector(self, symbol: str, sector: str) -> None:
        self._conn.execute("UPDATE securities SET sector = ? WHERE symbol = ?", (sector, symbol))
        self._conn.commit()

    # -- quotes / bars (§36 normalized OHLCV) -------------------------------
    def store_quotes(self, rows: Iterable[dict]) -> int:
        n = 0
        for r in rows:
            self._conn.execute(
                """INSERT OR IGNORE INTO quotes (symbol, timestamp, bid, ask, last, volume, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["symbol"], r["timestamp"], r.get("bid"), r.get("ask"), r.get("last"),
                 r.get("volume"), r["source"], r["created_at"]),
            )
            n += 1
        self._conn.commit()
        return n

    def store_bars(self, df: pd.DataFrame, symbol: str, timeframe: str, source: str) -> int:
        """Store a normalized OHLCV frame indexed (or columned) by timestamp."""
        d = df.copy()
        if "timestamp" not in d.columns:
            d = d.reset_index().rename(columns={d.columns[0]: "timestamp"})
        d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        rows = [
            (symbol, timeframe, r.timestamp, r.open, r.high, r.low, r.close,
             None if pd.isna(r.volume) else int(r.volume), source)
            for r in d[cols].itertuples(index=False)
        ]
        before = self._conn.total_changes
        self._conn.executemany(
            """INSERT OR IGNORE INTO bars
                 (symbol, timeframe, timestamp, open, high, low, close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return self._conn.total_changes - before

    def get_bars(self, symbol: str, timeframe: str = "daily",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
        q = ("SELECT timestamp, open, high, low, close, volume, source FROM bars "
             "WHERE symbol = ? AND timeframe = ?")
        params: list[Any] = [symbol, timeframe]
        if start:
            q += " AND timestamp >= ?"
            params.append(start)
        if end:
            q += " AND timestamp <= ?"
            params.append(end)
        df = pd.read_sql_query(q + " ORDER BY timestamp", self._conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_symbols_with_bars(self, timeframe: str = "daily") -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM bars WHERE timeframe = ? ORDER BY symbol", (timeframe,)
        ).fetchall()
        return [r["symbol"] for r in rows]

    # -- universe membership (§10) ------------------------------------------
    def record_membership(self, ticker: str, approved: bool, approval_date: str,
                          screen_version: str, reason: str) -> bool:
        """Returns True if a new membership row was written (idempotent)."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO universe_membership
                 (ticker, approved, approval_date, screen_version, reason)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker, int(approved), approval_date, screen_version, reason),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def approved_universe(self, as_of: str | None = None,
                          screen_version: str | None = None) -> list[str]:
        """Symbols whose latest membership record as of `as_of` is approved."""
        q = "SELECT ticker, approved, approval_date, screen_version FROM universe_membership"
        conds, params = [], []
        if as_of:
            conds.append("approval_date <= ?")
            params.append(as_of)
        if screen_version:
            conds.append("screen_version = ?")
            params.append(screen_version)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        df = pd.read_sql_query(q, self._conn, params=params)
        if df.empty:
            return []
        df = df.sort_values("approval_date").groupby("ticker", as_index=False).last()
        return sorted(df.loc[df["approved"] == 1, "ticker"].tolist())

    # -- strategy registry (§17) ---------------------------------------------
    def register_strategy_version(self, name: str, version: str, params: dict,
                                  git_commit: str | None = None,
                                  backtest_period: str | None = None,
                                  status: str = "BACKTEST") -> None:
        now = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute(
            """INSERT INTO strategies (name, description, created_at) VALUES (?, NULL, ?)
               ON CONFLICT(name) DO NOTHING""",
            (name, now),
        )
        self._conn.execute(
            """INSERT INTO strategy_versions
                 (strategy_name, version, params_json, git_commit, created_at, backtest_period, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(strategy_name, version) DO UPDATE SET
                 params_json=excluded.params_json, status=excluded.status""",
            (name, version, json.dumps(params, sort_keys=True), git_commit, now, backtest_period, status),
        )
        self._conn.commit()

    def set_strategy_status(self, name: str, version: str, status: str) -> None:
        """Promotion pipeline transitions (§18).  Never mutate parameters in place."""
        self._conn.execute(
            "UPDATE strategy_versions SET status = ? WHERE strategy_name = ? AND version = ?",
            (status, name, version),
        )
        self._conn.commit()

    # -- signals (§11, §25) ---------------------------------------------------
    def insert_signal(self, sig: Signal, status: str = "NEW") -> bool:
        """Insert signal; returns False when the idempotency key already exists."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO signals
                 (signal_id, ticker, strategy, version, bar_timestamp, action,
                  confidence, target_weight, quantity, price_hint, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sig.signal_id or sig.identity(), sig.ticker, sig.strategy, sig.version,
             sig.bar_timestamp, sig.action, sig.confidence, sig.target_weight,
             sig.quantity, sig.price_hint, status,
             pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def signal_seen(self, sig: Signal) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM signals WHERE signal_id = ?",
            (sig.signal_id or sig.identity(),),
        ).fetchone()
        return row is not None

    def update_signal_status(self, signal_id: str, status: str,
                             reject_reason: str | None = None) -> None:
        self._conn.execute(
            "UPDATE signals SET status = ?, reject_reason = ? WHERE signal_id = ?",
            (status, reject_reason, signal_id),
        )
        self._conn.commit()

    # -- orders / fills (§21) ---------------------------------------------------
    def insert_order(self, order: dict) -> bool:
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO orders
                 (order_id, signal_id, mode, portfolio, ticker, side, quantity,
                  order_type, limit_price, expected_price, status, kill_switch_hold,
                  created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order["order_id"], order.get("signal_id"), order["mode"], order.get("portfolio", "default"),
             order["ticker"], order["side"], order["quantity"], order.get("order_type", "MARKET"),
             order.get("limit_price"), order.get("expected_price"), order["status"],
             int(order.get("kill_switch_hold", 0)), order["created_at"], order["created_at"]),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_order(self, order_id: str, **fields: Any) -> None:
        allowed = ("status", "expected_price", "kill_switch_hold", "updated_at")
        sets, params = [], []
        for k in allowed:
            if k in fields:
                sets.append(f"{k} = ?")
                params.append(fields[k])
        if not sets:
            return
        params.append(order_id)
        self._conn.execute(f"UPDATE orders SET {', '.join(sets)} WHERE order_id = ?", params)
        self._conn.commit()

    def insert_fill(self, order_id: str, timestamp: str, price: float, quantity: float,
                    commission: float = 0.0, slippage: float = 0.0) -> bool:
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO fills (order_id, timestamp, price, quantity, commission, slippage)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (order_id, timestamp, price, quantity, commission, slippage),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_order(self, order_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def open_orders(self, mode: str, portfolio: str = "default") -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE mode = ? AND portfolio = ? AND status IN ('NEW','REVIEWED','SUBMITTED','PENDING_MANUAL')",
            (mode, portfolio),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- positions / snapshots (§16) ---------------------------------------------
    def upsert_position(self, portfolio: str, ticker: str, quantity: float,
                        avg_cost: float, realized_pl: float) -> None:
        self._conn.execute(
            """INSERT INTO paper_positions (portfolio, ticker, quantity, avg_cost, realized_pl, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(portfolio, ticker) DO UPDATE SET
                 quantity=excluded.quantity, avg_cost=excluded.avg_cost,
                 realized_pl=excluded.realized_pl, updated_at=excluded.updated_at""",
            (portfolio, ticker, quantity, avg_cost, realized_pl,
             pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._conn.commit()

    def load_positions(self, portfolio: str) -> dict[str, tuple[float, float, float]]:
        rows = self._conn.execute(
            "SELECT ticker, quantity, avg_cost, realized_pl FROM paper_positions WHERE portfolio = ?",
            (portfolio,),
        ).fetchall()
        return {r["ticker"]: (r["quantity"], r["avg_cost"], r["realized_pl"]) for r in rows}

    def save_snapshot(self, portfolio: str, timestamp: str, cash: float,
                      positions_value: float, equity: float,
                      realized_pl: float, unrealized_pl: float) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
                 (portfolio, timestamp, cash, positions_value, equity, realized_pl, unrealized_pl)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (portfolio, timestamp, cash, positions_value, equity, realized_pl, unrealized_pl),
        )
        self._conn.commit()

    def snapshots(self, portfolio: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM portfolio_snapshots WHERE portfolio = ? ORDER BY timestamp",
            self._conn, params=(portfolio,),
        )

    # -- risk events / alerts / telemetry --------------------------------------
    def record_risk_event(self, check: str, severity: str, detail: str,
                          signal_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO risk_events (timestamp, check_name, severity, detail, signal_id) VALUES (?, ?, ?, ?, ?)",
            (pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), check, severity, detail, signal_id),
        )
        self._conn.commit()

    def record_alert(self, severity: str, channel: str, subject: str,
                     body: str, signal_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO alerts (timestamp, severity, channel, subject, body, signal_id) VALUES (?, ?, ?, ?, ?, ?)",
            (pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), severity, channel, subject, body, signal_id),
        )
        self._conn.commit()

    def record_run(self, **fields: Any) -> None:
        cols = ("run_timestamp", "mode", "status", "universe_size", "quotes_requested",
                "quotes_received", "api_calls", "errors", "rate_limited", "signals",
                "risk_approved", "risk_rejected", "orders", "runtime_sec", "timings_json", "notes")
        vals = [fields.get(c) for c in cols]
        self._conn.execute(
            f"INSERT INTO system_runs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            vals,
        )
        self._conn.commit()

    # -- experiments (§32) ---------------------------------------------------------
    def record_experiment(self, experiment_id: str, hypothesis: str, strategy: str,
                          strategy_version: str, universe_version: str, params: dict,
                          train_period: str, validation_period: str, results: dict,
                          decision: str, reason: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO experiments
                 (experiment_id, hypothesis, strategy, strategy_version, universe_version,
                  params_json, train_period, validation_period, results_json, decision, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, hypothesis, strategy, strategy_version, universe_version,
             json.dumps(params, sort_keys=True), train_period, validation_period,
             json.dumps(results, sort_keys=True), decision, reason,
             pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._conn.commit()

    def kill_switch_events(self, since: str | None = None) -> list[dict]:
        q = "SELECT * FROM risk_events WHERE severity = 'CRITICAL'"
        params: Sequence = ()
        if since:
            q += " AND timestamp >= ?"
            params = (since,)
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]
