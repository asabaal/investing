"""Strategy contract (plan §11).

A strategy is written ONCE and runs unchanged across BACKTEST, FORWARD PAPER,
SHADOW and LIVE (plan §2).  Strategies never place orders - they emit Signals;
the risk engine and execution adapters do the rest.

`required_history` is expressed in bars so the same code works for any
timeframe (daily backtests, 15-minute paper cycles).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from ..db.database import Signal

if TYPE_CHECKING:  # avoid runtime import cycle
    from ..portfolio.paper import PaperPortfolio


@dataclass
class StrategyContext:
    ticker: str
    timestamp: pd.Timestamp          # close time of the last COMPLETED bar
    data: pd.DataFrame               # feature frame, history up to and incl. this bar
    portfolio: "PaperPortfolio | None" = None
    prices: dict[str, float] = field(default_factory=dict)

    @property
    def last_row(self) -> pd.Series:
        return self.data.iloc[-1]

    @property
    def price(self) -> float:
        return float(self.last_row["close"])


class Strategy(ABC):
    """Plan §11 standardized strategy contract."""

    name: str = "strategy"
    version: str = "v1"
    params: dict = {}

    @abstractmethod
    def universe(self) -> list[str]:
        """Symbols this strategy trades."""

    @abstractmethod
    def required_history(self) -> int:
        """Bars of history needed before signals are trustworthy."""

    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to a normalized OHLCV frame."""

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal | None:
        """Entry decision for one symbol, or None.  No order placement."""

    @abstractmethod
    def exit_signal(self, context: StrategyContext) -> bool:
        """True when an open position in this symbol should be exited."""

    def size_position(self, context: StrategyContext) -> float | None:
        """Optional target weight override; None defers to risk-engine sizing."""
        return None

    # -- convenience ---------------------------------------------------------
    def signal(self, context: StrategyContext, action: str = "BUY",
               confidence: float | None = None) -> Signal:
        weight = self.size_position(context)
        price = context.price
        qty = None
        if weight and context.portfolio is not None and context.portfolio.starting_capital:
            equity = context.portfolio.equity(context.prices)
            qty = equity * weight / price if price > 0 and equity > 0 else None
        return Signal(
            ticker=context.ticker,
            strategy=self.name,
            version=self.version,
            bar_timestamp=str(context.timestamp),
            action=action,
            confidence=confidence,
            target_weight=weight,
            quantity=qty,
            price_hint=price,
        )
