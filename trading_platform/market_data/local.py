"""Provider that reads bars/quotes already stored in the platform database.

This is the workhorse for backtests (plan §13 historical replay) and lets the
paper-trading cycle run end-to-end without any vendor connectivity.
"""

from __future__ import annotations

import pandas as pd

from .base import MarketDataProvider, Quote


class LocalDatabaseProvider(MarketDataProvider):
    name = "local_db"

    def __init__(self, db):
        self._db = db  # trading_platform.db.database.PlatformDatabase

    def get_quote(self, symbol: str) -> Quote:
        row = self._db._conn.execute(
            """SELECT symbol, timestamp, last, bid, ask, volume, source FROM quotes
               WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        if row is None:
            bars = self.get_bars(symbol)
            if bars.empty:
                raise KeyError(f"No local data for {symbol}")
            last = bars.iloc[-1]
            return Quote(symbol=symbol, timestamp=str(last["timestamp"]),
                         last=float(last["close"]), source=self.name)
        return Quote(symbol=row["symbol"], timestamp=row["timestamp"], last=row["last"],
                     bid=row["bid"], ask=row["ask"], volume=row["volume"], source=self.name)

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out = {}
        for sym in symbols:
            try:
                out[sym] = self.get_quote(sym)
            except KeyError:
                continue
        return out

    def get_bars(self, symbol: str, timeframe: str = "daily",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
        return self._db.get_bars(symbol, timeframe=timeframe, start=start, end=end)
