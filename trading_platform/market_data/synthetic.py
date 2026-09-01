"""Deterministic synthetic provider - TESTS ONLY, never for research results.

A fixed-seed geometric random walk so unit tests exercise the same code paths
as real data without touching the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MarketDataProvider, Quote


class SyntheticRandomProvider(MarketDataProvider):
    name = "synthetic"

    def __init__(self, symbols: list[str], n_bars: int = 300, seed: int = 42,
                 start: str = "2024-01-01", drift: float = 0.0004, vol: float = 0.02):
        self.symbols = list(symbols)
        self.n_bars = n_bars
        self._data = {s: self._walk(seed + i, drift, vol, start) for i, s in enumerate(symbols)}

    def _walk(self, seed: int, drift: float, vol: float, start: str) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets = rng.normal(drift, vol, self.n_bars)
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[100.0], close[:-1]])
        spread = np.abs(rng.normal(0, vol / 2, self.n_bars)) * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.integers(500_000, 5_000_000, self.n_bars)
        ts = pd.date_range(start, periods=self.n_bars, freq="D")
        return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                             "low": low, "close": close, "volume": volume})

    def get_quote(self, symbol: str) -> Quote:
        bar = self._data[symbol].iloc[-1]
        return Quote(symbol=symbol, timestamp=str(bar["timestamp"]), last=float(bar["close"]),
                     volume=int(bar["volume"]), source=self.name)

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    def get_bars(self, symbol: str, timeframe: str = "daily",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
        df = self._data[symbol]
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)
