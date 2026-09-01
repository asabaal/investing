"""Paper portfolio engine (plan §16).

Behaves like a real brokerage cash account: cash, positions with average cost,
realized/unrealized P&L, buying power, fills with fees & slippage.  Multiple
named portfolios coexist so strategies can compete.

Used identically by the backtester (replaying history) and the live paper
trader - the §2 "one strategy, four modes" principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ExecutionConfig


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pl: float = 0.0

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-9


@dataclass
class FillRecord:
    timestamp: str
    ticker: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    strategy: str = ""


@dataclass
class TradeRecord:
    """Completed round trip (position fully or partially closed)."""
    ticker: str
    strategy: str
    entry_time: str
    exit_time: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_bars: int = 0


@dataclass
class PaperPortfolio:
    starting_capital: float
    name: str = "default"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[FillRecord] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    # entry metadata for holding-period / per-strategy attribution
    _entries: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)  # ticker -> [(time, price, qty)]

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.starting_capital

    # -- valuation -----------------------------------------------------------
    def positions_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for tkr, pos in self.positions.items():
            if pos.is_open and tkr in prices:
                total += pos.quantity * prices[tkr]
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def buying_power(self) -> float:
        return self.cash  # cash account, no margin

    def realized_pl(self) -> float:
        return sum(p.realized_pl for p in self.positions.values())

    def unrealized_pl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for tkr, pos in self.positions.items():
            if pos.is_open and tkr in prices:
                total += (prices[tkr] - pos.avg_cost) * pos.quantity
        return total

    def exposure(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        return self.positions_value(prices) / eq if eq > 0 else 0.0

    # -- order application -----------------------------------------------------
    def apply_fill(self, ticker: str, side: str, quantity: float, price: float,
                   timestamp: str, strategy: str = "",
                   reference_price: float | None = None) -> FillRecord:
        """Apply an executed fill to cash/positions with friction (§13).

        `reference_price` is the pre-slippage decision price; slippage is
        charged against the trade direction (buy ≈ ask+slip, sell ≈ bid-slip).
        """
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        slip_bps = self.execution.slippage_bps / 10_000.0
        if reference_price is None:
            reference_price = price
            fill_price = price
        else:
            fill_price = (reference_price * (1 + slip_bps) if side == "BUY"
                          else reference_price * (1 - slip_bps))
        commission = max(self.execution.commission_per_share * quantity,
                         self.execution.min_commission)
        slippage_cost = abs(fill_price - reference_price) * quantity
        gross = fill_price * quantity

        if side == "BUY":
            if gross + commission > self.cash + 1e-6:
                raise ValueError(
                    f"insufficient cash for BUY {quantity} {ticker} @ {fill_price:.2f} "
                    f"(need {gross + commission:.2f}, have {self.cash:.2f})")
            self.cash -= gross + commission
            pos = self.positions.setdefault(ticker, Position(ticker))
            total_cost = pos.avg_cost * pos.quantity + fill_price * quantity
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0.0
            self._entries.setdefault(ticker, []).append((timestamp, fill_price, quantity))
        elif side == "SELL":
            pos = self.positions.get(ticker)
            if pos is None or pos.quantity < quantity - 1e-9:
                have = pos.quantity if pos else 0.0
                raise ValueError(f"cannot SELL {quantity} {ticker}, position is {have}")
            self.cash += gross - commission
            pnl = (fill_price - pos.avg_cost) * quantity - commission
            pos.realized_pl += pnl
            pos.quantity -= quantity
            if pos.quantity <= 1e-9:
                pos.quantity = 0.0
                pos.avg_cost = 0.0
            self._record_trade(ticker, strategy, timestamp, fill_price, quantity, pnl)
        else:
            raise ValueError(f"unknown side {side!r}")

        rec = FillRecord(timestamp, ticker, side, quantity, fill_price, commission,
                         slippage_cost, strategy)
        self.fills.append(rec)
        return rec

    def _record_trade(self, ticker: str, strategy: str, exit_time: str,
                      exit_price: float, qty: float, pnl: float) -> None:
        entries = self._entries.get(ticker, [])
        if not entries:
            return
        # FIFO round-trip attribution
        entry_time, entry_price, _ = entries.pop(0)
        holding_bars = 0
        try:
            import pandas as pd
            holding_bars = max(0, int((pd.Timestamp(exit_time) - pd.Timestamp(entry_time))
                                      / pd.Timedelta(days=1)))
        except Exception:
            pass
        cost = entry_price * qty
        self.trades.append(TradeRecord(
            ticker=ticker, strategy=strategy, entry_time=entry_time, exit_time=exit_time,
            quantity=qty, entry_price=entry_price, exit_price=exit_price, pnl=pnl,
            pnl_pct=pnl / cost if cost > 0 else 0.0, holding_bars=holding_bars,
        ))

    # -- persistence ------------------------------------------------------------
    def save_positions(self, db) -> None:
        for tkr, pos in self.positions.items():
            if pos.is_open or pos.realized_pl != 0.0:
                db.upsert_position(self.name, tkr, pos.quantity, pos.avg_cost, pos.realized_pl)
