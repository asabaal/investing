"""Command-line entry points for the platform (plan §45 MVP)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DB_PATH, Config, load_config
from .db.database import PlatformDatabase
from .backtest.engine import BacktestEngine
from .market_data.alpha_vantage import AlphaVantageProvider
from .market_data.local import LocalDatabaseProvider
from .notifications.channels import NotificationRouter
from .portfolio.paper import PaperPortfolio
from .scheduler.cycle import PaperTradingCycle
from .strategies import MeanReversionV1, MomentumV1, SmaCrossV1
from .universe.manager import QualityScreenV1, UniverseManager

STRATEGY_FACTORIES = {
    "sma_cross": SmaCrossV1,
    "momentum": MomentumV1,
    "mean_reversion": MeanReversionV1,
}


def make_provider(config: Config, db: PlatformDatabase):
    name = config.data_provider
    if name == "alpha_vantage":
        return AlphaVantageProvider(api_key=config.alpha_vantage_api_key)
    if name == "local_db":
        return LocalDatabaseProvider(db)
    if name == "robinhood_mcp":
        from .market_data.robinhood_mcp import RobinhoodMCPProvider
        return RobinhoodMCPProvider()
    raise ValueError(f"unknown data provider {name!r}")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- subcommands
def cmd_init_db(args, config, db):
    db._init_schema()
    print(f"database ready: {db.db_path}")


def cmd_import_legacy(args, config, db):
    """Pull the legacy repo DB's daily OHLCV into normalized platform bars (§36)."""
    import sqlite3

    src = sqlite3.connect(str(args.source))
    rows = src.execute(
        """SELECT symbol, timestamp, open, high, low, close, volume FROM market_data
           ORDER BY symbol, timestamp"""
    ).fetchall()
    src.close()
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low",
                                     "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    n = 0
    for sym, g in df.groupby("symbol"):
        daily = g.groupby("date").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
        ).reset_index().rename(columns={"date": "timestamp"})
        stored = db.store_bars(daily, sym, timeframe="daily", source="legacy_import")
        db.upsert_security(sym)
        n += stored
        print(f"  {sym}: {stored} new daily bars")
    print(f"imported {n} bars from {args.source}")


def cmd_universe_import(args, config, db):
    um = UniverseManager(db)
    stats = um.import_watchlist(args.csv, screen=QualityScreenV1())
    print(json.dumps(stats, indent=2))


def cmd_universe_show(args, config, db):
    um = UniverseManager(db)
    as_of = args.as_of or _now()[:10]
    tickers = um.universe_as_of(as_of)
    print(f"approved universe as of {as_of}: {len(tickers)} symbols")
    for t in tickers:
        print(f"  {t}")


def cmd_prove_connectivity(args, config, db):
    """Plan §34 Phase 1 (adapted): authenticate, quote an arbitrary ticker, exit."""
    try:
        provider = make_provider(config, db)
        quote = provider.get_quote(args.symbol)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    db.store_quotes([{
        "symbol": quote.symbol, "timestamp": quote.timestamp, "bid": quote.bid,
        "ask": quote.ask, "last": quote.price, "volume": quote.volume,
        "source": quote.source, "created_at": _now(),
    }])
    print(f"OK [{provider.name}] {quote.symbol} = ${quote.price} "
          f"at {quote.timestamp} (stored locally)")
    print("run again from a fresh process to confirm unattended operation (plan §34)")
    return 0


def cmd_collect(args, config, db):
    """Plan §35: fetch + store bars for a symbol list (rate limits honored)."""
    provider = make_provider(config, db)
    for sym in args.symbols:
        try:
            df = provider.get_bars(sym, timeframe=args.timeframe)
        except Exception as exc:
            print(f"  {sym}: ERROR {exc}")
            continue
        if df.empty:
            print(f"  {sym}: no data")
            continue
        n = db.store_bars(df, sym, timeframe=args.timeframe, source=provider.name)
        db.upsert_security(sym)
        print(f"  {sym}: {n} new {args.timeframe} bars ({len(df)} fetched)")
    print("collection complete")


def cmd_backtest(args, config, db):
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data = {}
    for sym in symbols:
        df = db.get_bars(sym, timeframe=args.timeframe, start=args.start, end=args.end)
        if df.empty:
            print(f"WARNING: no {args.timeframe} bars for {sym} - skipped")
            continue
        data[sym] = df
    if not data:
        print("no data for backtest; run `collect` or `import-legacy` first")
        return 1
    factory = STRATEGY_FACTORIES[args.strategy]
    strat = factory(symbols=list(data))
    engine = BacktestEngine(strat, db=db if args.store else None, config=config,
                            initial_capital=args.capital,
                            benchmark=args.benchmark)
    result = engine.run(data, start=args.start, end=args.end)
    bench = db.get_bars(args.benchmark, timeframe=args.timeframe) if args.benchmark else None
    print(result.summary(bench))

    # §17 versioning + §32 experiment discipline: every run is recorded
    db.register_strategy_version(strat.name, strat.version, strat.params,
                                 git_commit=_git_commit(), backtest_period=f"{args.start or 'min'}..{args.end or 'max'}")
    db.record_experiment(
        experiment_id=f"EXP-{_next_exp_num(db):04d}",
        hypothesis=f"{strat.name}:{strat.version} baseline run on {sorted(data)}",
        strategy=strat.name, strategy_version=strat.version,
        universe_version="unmanaged", params=strat.params,
        train_period=f"{args.start or 'min'}..{args.end or 'max'}",
        validation_period="n/a", results=result.metrics(bench), decision="INCONCLUSIVE",
        reason="baseline measurement; set decision after review")
    if args.store:
        result.portfolio.save_positions(db)
    return 0


def cmd_run_cycle(args, config, db):
    """Plan §38: one scheduled live-market paper-trading cycle."""
    config.mode = args.mode
    provider = make_provider(config, db)
    strategies = []
    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if args.strategy == "all":
        strategies = [f(symbols=symbols or db.get_symbols_with_bars(args.timeframe))
                      for f in STRATEGY_FACTORIES.values()]
    else:
        strategies = [STRATEGY_FACTORIES[args.strategy](
            symbols=symbols or db.get_symbols_with_bars(args.timeframe))]
    portfolio = PaperPortfolio(starting_capital=args.capital, name=f"{args.mode}_default",
                               execution=config.execution)
    cycle = PaperTradingCycle(config, db, provider, strategies, portfolio,
                              router=NotificationRouter(alert_dir=config.alert_dir),
                              timeframe=args.timeframe)
    result = cycle.run_cycle(force=args.force)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_kill_switch(args, config, db):
    flag = DEFAULT_DB_PATH.parent / "TRADING_ENABLED"
    if args.on:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("trading enabled - remove this file to kill switch\n")
        print("KILL SWITCH DISENGAGED: trading enabled (env TRADING_ENABLED must also be true)")
    elif args.off:
        flag.unlink(missing_ok=True)
        print("KILL SWITCH ENGAGED: NO NEW LIVE ORDERS")
    else:
        state = config.trading_enabled
        print(f"trading_enabled: {state}\nflag file: {flag} "
              f"({'present' if flag.exists() else 'absent'})\n"
              f"env TRADING_ENABLED: {__import__('os').environ.get('TRADING_ENABLED', 'unset')}")


def cmd_report(args, config, db):
    lines = [f"# Platform report - {_now()}"]
    for pf in db._conn.execute("SELECT DISTINCT portfolio FROM portfolio_snapshots").fetchall():
        name = pf["portfolio"]
        snaps = db.snapshots(name)
        if snaps.empty:
            continue
        first, last = snaps.iloc[0], snaps.iloc[-1]
        lines += [f"\n## Portfolio: {name}",
                  f"- snapshots: {len(snaps)} ({first['timestamp']} .. {last['timestamp']})",
                  f"- equity: ${last['equity']:,.2f} "
                  f"({last['equity'] / first['equity'] - 1:+.2%} since first snapshot)",
                  f"- cash: ${last['cash']:,.2f}  realized P&L: ${last['realized_pl']:,.2f}  "
                  f"unrealized P&L: ${last['unrealized_pl']:,.2f}"]
        positions = db.load_positions(name)
        open_pos = {t: p for t, p in positions.items() if p[0] > 1e-9}
        if open_pos:
            lines.append(f"- open positions: {len(open_pos)}")
            for t, (qty, avg, rlz) in sorted(open_pos.items()):
                lines.append(f"    {t}: {qty:.4g} @ ${avg:.2f}")
    runs = db._conn.execute(
        "SELECT * FROM system_runs ORDER BY id DESC LIMIT 5").fetchall()
    lines.append("\n## Recent system runs (§26 telemetry)")
    for r in runs:
        lines.append(f"- {r['run_timestamp']} mode={r['mode']} status={r['status']} "
                     f"signals={r['signals']} approved={r['risk_approved']} "
                     f"rejected={r['risk_rejected']} runtime={r['runtime_sec']}s")
    versions = db._conn.execute(
        "SELECT * FROM strategy_versions ORDER BY created_at DESC LIMIT 10").fetchall()
    lines.append("\n## Strategy versions (§17)")
    for v in versions:
        lines.append(f"- {v['strategy_name']}:{v['version']} status={v['status']} "
                     f"created={v['created_at']} params={v['params_json'][:80]}")
    report = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(report)
        print(f"report written to {args.out}")
    else:
        print(report)


def cmd_shadow_report(args, config, db):
    """Plan §20: compare model expectation vs. later market outcome."""
    rows = db._conn.execute(
        """SELECT o.*, f.price AS fill_price FROM orders o
           LEFT JOIN fills f ON f.order_id = o.order_id
           WHERE o.mode = 'shadow' ORDER BY o.created_at DESC LIMIT 200""").fetchall()
    if not rows:
        print("no shadow orders recorded yet")
        return 0
    print(f"{'created':<21} {'ticker':<6} {'side':<5} {'qty':>10} {'expected':>10} {'filled':>10}")
    for r in rows:
        print(f"{r['created_at']:<21} {r['ticker']:<6} {r['side']:<5} "
              f"{r['quantity']:>10.4g} {r['expected_price'] or 0:>10.2f} "
              f"{r['fill_price'] if r['fill_price'] is not None else 'LOG ONLY':>10}")
    return 0


def cmd_experiments(args, config, db):
    rows = db._conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    if not rows:
        print("no experiments recorded yet (plan §32)")
        return 0
    for r in rows:
        print(f"{r['experiment_id']}  {r['strategy']}:{r['strategy_version']}  "
              f"decision={r['decision']}  {r['hypothesis'][:70]}")
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _next_exp_num(db) -> int:
    row = db._conn.execute(
        "SELECT experiment_id FROM experiments ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return 1
    try:
        return int(row["experiment_id"].split("-")[1]) + 1
    except (IndexError, ValueError):
        return 1


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_platform",
                                description="Trading research & paper trading platform")
    p.add_argument("--db", default=None, help="override platform DB path")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create/upgrade schema")

    leg = sub.add_parser("import-legacy", help="import daily bars from legacy trading_data.db")
    leg.add_argument("--source", default="trading_data.db")

    ui = sub.add_parser("universe-import", help="seed universe from screening CSV (§3, §10)")
    ui.add_argument("--csv", default="watchlist/WATCHLIST - Sheet1.csv")
    sub.add_parser("universe-show").add_argument("--as-of", default=None)

    pc = sub.add_parser("prove-connectivity", help="Phase 1 proof (§34)")
    pc.add_argument("--symbol", default="AAPL")

    col = sub.add_parser("collect", help="fetch + store bars (§35)")
    col.add_argument("--symbols", required=True)
    col.add_argument("--timeframe", default="daily")

    bt = sub.add_parser("backtest", help="run a strategy backtest (§13-14)")
    bt.add_argument("--strategy", choices=sorted(STRATEGY_FACTORIES), default="sma_cross")
    bt.add_argument("--symbols", required=True)
    bt.add_argument("--timeframe", default="daily")
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--capital", type=float, default=100_000.0)
    bt.add_argument("--benchmark", default="SPY")
    bt.add_argument("--store", action="store_true", help="record signals/orders in DB")

    rc = sub.add_parser("run-cycle", help="one scheduled paper/shadow cycle (§7, §38)")
    rc.add_argument("--mode", choices=["paper", "shadow"], default="paper")
    rc.add_argument("--strategy", default="all")
    rc.add_argument("--symbols", default=None)
    rc.add_argument("--timeframe", default="daily")
    rc.add_argument("--capital", type=float, default=100_000.0)
    rc.add_argument("--force", action="store_true", help="run even if market closed")

    ks = sub.add_parser("kill-switch", help="§24 global kill switch")
    g = ks.add_mutually_exclusive_group()
    g.add_argument("--on", action="store_true", help="disengage (allow trading)")
    g.add_argument("--off", action="store_true", help="engage (NO NEW ORDERS)")

    rep = sub.add_parser("report", help="performance & health report (§39 lite)")
    rep.add_argument("--out", default=None)
    sub.add_parser("shadow-report", help="§20 shadow expectation log")
    sub.add_parser("experiments", help="§32 experiment log")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    db = PlatformDatabase(args.db) if args.db else PlatformDatabase()
    try:
        handler = globals()[f"cmd_{args.command.replace('-', '_')}"]
        return handler(args, config, db) or 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
