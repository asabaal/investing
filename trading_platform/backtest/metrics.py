"""Backtest metrics (plan §14) - a model is never judged on headline return."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, exposure: pd.Series, trades: list,
                    bars_per_year: int = 252, risk_free: float = 0.0,
                    benchmark_equity: pd.Series | None = None) -> dict:
    out: dict = {}
    eq = equity.dropna()
    if len(eq) < 2:
        return {"error": "not enough data points"}

    rets = eq.pct_change().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = n_days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

    vol = float(rets.std() * np.sqrt(bars_per_year)) if len(rets) > 1 else 0.0
    rf_per_bar = risk_free / bars_per_year
    excess = rets - rf_per_bar
    sharpe = float(excess.mean() / excess.std() * np.sqrt(bars_per_year)) \
        if len(excess) > 1 and excess.std() > 0 else 0.0
    downside = excess[excess < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(bars_per_year)) \
        if len(downside) > 1 and downside.std() > 0 else 0.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())

    out.update({
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "days": n_days,
    })

    # Trade statistics (§14)
    if trades:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        hold_days = [t.holding_bars for t in trades if t.holding_bars is not None]
        consec = _max_consecutive_losses(pnls)
        out.update({
            "number_of_trades": int(len(pnls)),
            "win_rate": float(len(wins) / len(pnls)),
            "avg_winner": float(wins.mean()) if len(wins) else 0.0,
            "avg_loser": float(losses.mean()) if len(losses) else 0.0,
            "largest_loss": float(losses.min()) if len(losses) else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "expectancy": float(pnls.mean()),
            "max_consecutive_losses": int(consec),
            "avg_holding_days": float(np.mean(hold_days)) if hold_days else 0.0,
        })
        # expectancy in R-multiples (avg win / |avg loss|)
        if len(wins) and len(losses) and losses.mean() != 0:
            r_multiple = abs(wins.mean() / losses.mean())
            p = len(wins) / len(pnls)
            out["expectancy_r"] = float(p * r_multiple - (1 - p))
    else:
        out.update({"number_of_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                    "expectancy": 0.0})

    # Turnover & time in market
    if len(eq) and eq.iloc[0] > 0:
        out["time_in_market"] = float((exposure > 1e-9).mean())
        out["exposure_avg"] = float(exposure.mean())

    # Performance by year (§14)
    by_year = {}
    yearly = eq.resample("YE").last().dropna()
    yearly_first = eq.resample("YE").first().dropna()
    prev = eq.iloc[0]
    for ts, endv in yearly.items():
        startv = yearly_first.get(ts, prev)
        by_year[str(ts.year)] = float(endv / prev - 1) if prev else 0.0
        prev = endv
    out["performance_by_year"] = by_year

    # Performance by volatility regime: terciles of trailing 20-bar vol
    if len(rets) > 60:
        rv = rets.rolling(20).std().dropna()
        q1, q2 = rv.quantile(1 / 3), rv.quantile(2 / 3)
        regime_of = pd.qcut(rv.rank(method="first"), 3, labels=["low", "mid", "high"])
        grouped = rets[rv.index].groupby(regime_of, observed=True)
        out["performance_by_vol_regime"] = {
            str(k): {"return": float((1 + g).prod() - 1), "bars": int(len(g))}
            for k, g in grouped
        }

    # Benchmark comparison (§14)
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        b = benchmark_equity.dropna()
        b_rets = b.pct_change().dropna()
        b_dd = float((b / b.cummax() - 1).min())
        b_sharpe = float(b_rets.mean() / b_rets.std() * np.sqrt(bars_per_year)) \
            if b_rets.std() > 0 else 0.0
        out["benchmark"] = {
            "total_return": float(b.iloc[-1] / b.iloc[0] - 1),
            "sharpe": b_sharpe,
            "max_drawdown": b_dd,
        }
        out["excess_return_vs_benchmark"] = out["total_return"] - out["benchmark"]["total_return"]

    return out


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    best = cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
