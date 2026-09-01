"""Abstract market data interface (plan §8)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Quote:
    symbol: str
    timestamp: str
    last: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    source: str = "unknown"

    @property
    def price(self) -> float:
        if self.last is not None:
            return self.last
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        raise ValueError(f"Quote for {self.symbol} has no usable price")

    @property
    def spread_pct(self) -> float | None:
        if self.bid and self.ask and self.bid > 0:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2.0)
        return None


class MarketDataProvider(ABC):
    """All market data flows through this interface (plan §8)."""

    name: str = "base"

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        ...

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batched quotes; implementations should batch where the vendor allows."""

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str = "daily",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Normalized OHLCV frame: timestamp, open, high, low, close, volume.

        `timestamp` is the bar OPEN time.
        """

    def get_positions(self) -> dict:
        """Brokerage positions; only meaningful for live/shadow providers."""
        return {}


def quote_to_bars(quote: Quote, timeframe: str = "15min") -> pd.DataFrame:
    """Convert a live quote into a single synthetic bar row (open=close=price).

    Used by the live paper trader to append the freshest observation to history.
    """
    ts = pd.Timestamp(quote.timestamp)
    if timeframe == "daily":
        ts = ts.normalize()
    price = quote.price
    return pd.DataFrame([{
        "timestamp": ts,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": quote.volume or 0,
    }])
