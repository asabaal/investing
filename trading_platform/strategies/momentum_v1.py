"""Momentum v1 (plan §3 example family): ride established strength.

ENTRY: N-day return above threshold, price above its EMA trend filter, RSI not
overbought.
EXIT: momentum turns negative or a trailing-stop on the highest close since
entry is hit.
"""

from __future__ import annotations

import pandas as pd

from ..db.database import Signal
from ..indicators.library import ema, rsi
from .base import Strategy, StrategyContext


class MomentumV1(Strategy):
    name = "momentum"
    version = "v1"
    params = {
        "lookback": 63,             # ~1 quarter of daily bars
        "min_return": 0.05,
        "trend_ema": 100,
        "rsi_window": 14,
        "rsi_max": 75.0,
        "trail_stop_pct": 0.08,
        "target_weight": 0.04,
    }

    def __init__(self, symbols: list[str], **overrides):
        self.symbols = symbols
        self.params = {**self.params, **overrides}

    def universe(self) -> list[str]:
        return list(self.symbols)

    def required_history(self) -> int:
        return max(self.params["lookback"], self.params["trend_ema"]) + 5

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["momentum"] = df["close"].pct_change(self.params["lookback"])
        df["trend_ema"] = ema(df["close"], self.params["trend_ema"])
        df["rsi"] = rsi(df["close"], self.params["rsi_window"])
        df["trail_high"] = df["close"].cummax()
        return df

    def generate_signal(self, context: StrategyContext) -> Signal | None:
        row = context.last_row
        if pd.isna(row["momentum"]) or pd.isna(row["trend_ema"]) or pd.isna(row["rsi"]):
            return None
        pos = context.portfolio.positions.get(context.ticker) if context.portfolio else None
        if pos is not None and pos.is_open:
            return None
        if (row["momentum"] >= self.params["min_return"]
                and row["close"] > row["trend_ema"]
                and row["rsi"] < self.params["rsi_max"]):
            conf = min(0.95, 0.4 + row["momentum"])
            return self.signal(context, "BUY", confidence=round(float(conf), 3))
        return None

    def exit_signal(self, context: StrategyContext) -> bool:
        row = context.last_row
        recent = context.data["close"].iloc[-self.params["lookback"]:]
        trail_high = float(recent.max())
        if row["close"] < trail_high * (1 - self.params["trail_stop_pct"]):
            return True
        if not pd.isna(row["momentum"]) and row["momentum"] < 0:
            return True
        return False
