"""Mean reversion v1: fade stretched moves back to the band midline.

ENTRY: close below lower Bollinger band with RSI oversold.
EXIT: close back at the band midline, RSI recovered, or stop on further
breakdown.
"""

from __future__ import annotations

import pandas as pd

from ..db.database import Signal
from ..indicators.library import bollinger, rsi
from .base import Strategy, StrategyContext


class MeanReversionV1(Strategy):
    name = "mean_reversion"
    version = "v1"
    params = {
        "bb_window": 20,
        "bb_num_std": 2.0,
        "rsi_window": 14,
        "rsi_entry_max": 30.0,
        "rsi_exit_min": 55.0,
        "stop_pct": 0.10,
        "target_weight": 0.04,
    }

    def __init__(self, symbols: list[str], **overrides):
        self.symbols = symbols
        self.params = {**self.params, **overrides}

    def universe(self) -> list[str]:
        return list(self.symbols)

    def required_history(self) -> int:
        return self.params["bb_window"] + self.params["rsi_window"] + 5

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        lower, mid, upper = bollinger(df["close"], self.params["bb_window"],
                                      self.params["bb_num_std"])
        df["bb_lower"], df["bb_mid"], df["bb_upper"] = lower, mid, upper
        df["rsi"] = rsi(df["close"], self.params["rsi_window"])
        return df

    def generate_signal(self, context: StrategyContext) -> Signal | None:
        row = context.last_row
        if pd.isna(row["bb_lower"]) or pd.isna(row["rsi"]):
            return None
        pos = context.portfolio.positions.get(context.ticker) if context.portfolio else None
        if pos is not None and pos.is_open:
            return None
        if row["close"] < row["bb_lower"] and row["rsi"] < self.params["rsi_entry_max"]:
            return self.signal(context, "BUY", confidence=0.55)
        return None

    def exit_signal(self, context: StrategyContext) -> bool:
        row = context.last_row
        entry = self._entry_price(context)
        if entry is not None and row["close"] < entry * (1 - self.params["stop_pct"]):
            return True
        if not pd.isna(row["bb_mid"]) and row["close"] >= row["bb_mid"]:
            return True
        if not pd.isna(row["rsi"]) and row["rsi"] > self.params["rsi_exit_min"]:
            return True
        return False

    @staticmethod
    def _entry_price(context: StrategyContext) -> float | None:
        pos = context.portfolio.positions.get(context.ticker) if context.portfolio else None
        return pos.avg_cost if pos is not None and pos.is_open else None
