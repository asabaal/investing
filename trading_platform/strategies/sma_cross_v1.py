"""Deliberately simple first model (plan §37): validate the ENGINE, not alpha.

LONG when SMA(short) > SMA(long); exit when it flips.  If the simulator cannot
run this correctly, nothing else matters.
"""

from __future__ import annotations

import pandas as pd

from ..indicators.library import sma
from ..db.database import Signal
from .base import Strategy, StrategyContext


class SmaCrossV1(Strategy):
    name = "sma_cross"
    version = "v1"
    params = {"short_window": 20, "long_window": 60, "target_weight": 0.04}

    def __init__(self, symbols: list[str], **overrides):
        self.symbols = symbols
        self.params = {**self.params, **overrides}

    def universe(self) -> list[str]:
        return list(self.symbols)

    def required_history(self) -> int:
        return self.params["long_window"] + 5

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["sma_short"] = sma(df["close"], self.params["short_window"])
        df["sma_long"] = sma(df["close"], self.params["long_window"])
        return df

    def generate_signal(self, context: StrategyContext) -> Signal | None:
        row = context.last_row
        if pd.isna(row["sma_short"]) or pd.isna(row["sma_long"]):
            return None
        if row["sma_short"] > row["sma_long"]:
            pos = context.portfolio.positions.get(context.ticker) if context.portfolio else None
            if pos is None or not pos.is_open:
                return self.signal(context, "BUY", confidence=0.5)
        return None

    def exit_signal(self, context: StrategyContext) -> bool:
        row = context.last_row
        if pd.isna(row["sma_short"]) or pd.isna(row["sma_long"]):
            return False
        return bool(row["sma_short"] < row["sma_long"])
