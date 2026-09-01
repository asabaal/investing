"""Universe management (plan §3, §10): good-company filtering with history.

Question A ("should this company be considered at all?") lives here.
Question B ("when to enter/exit?") lives in strategies.  Every membership
decision is recorded with an approval date + screen version so backtests can
answer: *was this company actually considered investable at the time?*
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..db.database import PlatformDatabase


class QualityScreenV1:
    """A deliberately simple fundamental screen (screen_version='quality_v1').

    Operates on a DataFrame of fundamentals with columns similar to the repo's
    watchlist snapshot: Ticker, Company, Sector, Industry, Market Cap, P/E,
    Price, Volume.  Criteria are intentionally transparent.
    """

    screen_version = "quality_v1"

    def __init__(self, min_market_cap: float = 2e9, min_price: float = 5.0,
                 require_positive_earnings: bool = True, min_volume: float = 100_000.0):
        self.min_market_cap = min_market_cap
        self.min_price = min_price
        self.require_positive_earnings = require_positive_earnings
        self.min_volume = min_volume

    @staticmethod
    def _parse_number(s: pd.Series) -> pd.Series:
        """Parse '3.20B' / '565.31M' / '13,372,791' style values to floats."""
        def one(v):
            if pd.isna(v):
                return float("nan")
            if isinstance(v, (int, float)):
                return float(v)
            t = str(v).strip().replace(",", "").replace("$", "")
            mult = 1.0
            if t and t[-1].upper() in "KMBT":
                mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[t[-1].upper()]
                t = t[:-1]
            try:
                return float(t) * mult
            except ValueError:
                return float("nan")
        return s.map(one)

    def apply(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Returns the input frame plus 'approved' bool and 'reason' str."""
        f = fundamentals.copy()
        cap = self._parse_number(f.get("Market Cap"))
        price = self._parse_number(f.get("Price"))
        pe = self._parse_number(f.get("P/E"))
        vol = self._parse_number(f.get("Volume"))

        approved = (
            (cap >= self.min_market_cap)
            & (price >= self.min_price)
            & (vol >= self.min_volume)
            & ((pe > 0) if self.require_positive_earnings else pe.notna())
        )
        reasons = []
        for i in f.index:
            if approved.loc[i]:
                reasons.append("passes quality_v1 fundamental screen")
            elif not (cap.loc[i] >= self.min_market_cap):
                reasons.append("market cap below minimum")
            elif not (price.loc[i] >= self.min_price):
                reasons.append("price below minimum")
            elif self.require_positive_earnings and not (pe.loc[i] > 0):
                reasons.append("non-positive earnings")
            elif not (vol.loc[i] >= self.min_volume):
                reasons.append("volume below minimum")
            else:
                reasons.append("failed screen")
        f["approved"] = approved.fillna(False)
        f["reason"] = reasons
        return f


class UniverseManager:
    def __init__(self, db: PlatformDatabase):
        self.db = db

    def import_watchlist(self, csv_path: str | Path, screen: QualityScreenV1 | None = None,
                         approval_date: str | None = None) -> dict:
        """Seed the universe from the repo's fundamental screening snapshot."""
        screen = screen or QualityScreenV1()
        raw = pd.read_csv(csv_path)
        raw = raw.rename(columns={c: c.strip() for c in raw.columns})
        scored = screen.apply(raw)
        date = approval_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        added = 0
        for row in scored.itertuples(index=False):
            ticker = str(getattr(row, "Ticker", "")).strip().upper()
            if not ticker:
                continue
            self.db.upsert_security(
                ticker,
                name=str(getattr(row, "Company", "") or "") or None,
                sector=str(getattr(row, "Sector", "") or "") or None,
                industry=str(getattr(row, "Industry", "") or "") or None,
            )
            if self.db.record_membership(
                ticker, bool(row.approved), date, screen.screen_version, row.reason,
            ):
                added += 1
        return {"date": date, "screen_version": screen.screen_version,
                "rows": int(len(scored)), "new_membership_rows": added,
                "approved_total": len(self.db.approved_universe(screen_version=screen.screen_version))}

    def approve(self, ticker: str, screen_version: str, reason: str,
                approval_date: str | None = None) -> None:
        date = approval_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        self.db.record_membership(ticker, True, date, screen_version, reason)

    def revoke(self, ticker: str, screen_version: str, reason: str,
               approval_date: str | None = None) -> None:
        date = approval_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        self.db.record_membership(ticker, False, date, screen_version, reason)

    def current(self, screen_version: str | None = None) -> list[str]:
        return self.db.approved_universe(screen_version=screen_version)

    def universe_as_of(self, as_of: str, screen_version: str | None = None) -> list[str]:
        """Membership as it existed on a past date (anti-survivorship bias, §13)."""
        return self.db.approved_universe(as_of=as_of, screen_version=screen_version)
