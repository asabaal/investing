"""Alpha Vantage adapter - the working data source until Robinhood MCP lands.

Wraps the repo-root ``alpha_vantage_api.AlphaVantageClient`` (rate limiting and
file cache included) so strategies never see vendor details (plan §8).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base import MarketDataProvider, Quote

# The platform package lives inside the investing repo; make the legacy
# client importable regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class AlphaVantageProvider(MarketDataProvider):
    name = "alpha_vantage"

    def __init__(self, api_key: str, premium: bool = False):
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key required (set ALPHA_VANTAGE_API_KEY; plan §29: env vars only)")
        from alpha_vantage_api import AlphaVantageClient  # lazy: legacy repo module

        self._client = AlphaVantageClient(api_key=api_key, premium=premium)
        self.api_calls = 0

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def get_quote(self, symbol: str) -> Quote:
        data = self._client.get_quote(symbol)
        self.api_calls += 1
        return Quote(
            symbol=symbol,
            timestamp=self._now_iso(),
            last=float(data.get("05. price", 0) or 0) or None,
            volume=int(float(data.get("06. volume", 0) or 0)) or None,
            source=self.name,
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        # Alpha Vantage has no batch endpoint; honor per-symbol calls and let
        # the underlying RateLimiter police 5/min on the free tier.
        out: dict[str, Quote] = {}
        for sym in symbols:
            try:
                out[sym] = self.get_quote(sym)
            except Exception as exc:  # keep the cycle alive on single-symbol failure
                out[sym] = Quote(symbol=sym, timestamp=self._now_iso(), source=f"{self.name}:error:{exc}")
        return out

    def get_bars(self, symbol: str, timeframe: str = "daily",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
        if timeframe != "daily":
            raise NotImplementedError(
                "AlphaVantageProvider v1 supports daily bars only; intraday collection "
                "is Phase 2 (plan §35).")
        raw = self._client.get_daily(symbol, outputsize="full")
        self.api_calls += 1
        df = self._client.clean_data(raw)
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.reset_index()
        ts_col = df.columns[0]
        df = df.rename(columns={
            ts_col: "timestamp", "1. open": "open", "2. high": "high",
            "3. low": "low", "4. close": "close", "5. volume": "volume",
            "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.sort_values("timestamp").reset_index(drop=True)
