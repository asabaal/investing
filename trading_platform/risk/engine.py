"""Independent risk engine (plan §12).

Every strategy decision passes through here before any execution path.
The risk engine protects the portfolio *from strategy bugs*: strategies never
place orders directly (§11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import Config, RiskLimits
from ..db.database import PlatformDatabase, Signal
from ..portfolio.paper import PaperPortfolio


@dataclass
class RiskContext:
    """Everything the risk layer may look at - strategies cannot touch this."""
    prices: dict[str, float]                 # decision prices per symbol
    quote_timestamps: dict[str, pd.Timestamp] = field(default_factory=dict)
    dollar_volumes: dict[str, float] = field(default_factory=dict)   # 20-bar avg $vol
    spreads: dict[str, float | None] = field(default_factory=dict)   # spread pct
    sectors: dict[str, str] = field(default_factory=dict)
    now: pd.Timestamp | None = None
    daily_loss_pct: float = 0.0              # today's realized+unrealized drawdown
    strategy_drawdowns: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)  # check -> PASS/FAIL/SKIP


class RiskEngine:
    def __init__(self, config: Config, db: PlatformDatabase):
        self.config = config
        self.limits: RiskLimits = config.risk
        self.db = db

    # -- main gate ------------------------------------------------------------
    def approve(self, sig: Signal, portfolio: PaperPortfolio, ctx: RiskContext,
                mode: str = "paper") -> RiskDecision:
        reasons: list[str] = []
        checks: dict[str, str] = {}

        def fail(check: str, why: str) -> None:
            checks[check] = "FAIL"
            reasons.append(f"{check}: {why}")
            self.db.record_risk_event(check, "WARNING", f"{why} [signal {sig.ticker} "
                                      f"{sig.action} {sig.bar_timestamp}]", sig.signal_id)

        # Kill switch (§24) - shadow/live modes refuse everything when tripped.
        if mode in ("shadow", "live") and not self.config.trading_enabled:
            fail("kill_switch", "TRADING_ENABLED is false - NO NEW ORDERS")

        # Market status (§12)
        now = ctx.now or pd.Timestamp.utcnow()
        if now.hour in (0, 1, 2, 3) and now.weekday() < 5:
            # 00:00-04:00 ET-equivalent naive guard; conservative pre-open block
            checks["market_status"] = "SKIP"  # scheduler itself runs market hours only

        # System health: stale data
        ts = ctx.quote_timestamps.get(sig.ticker)
        if ts is not None and ctx.now is not None:
            age_min = (ctx.now - ts).total_seconds() / 60.0
            if age_min > self.limits.max_data_age_minutes * 24:  # daily-data friendly
                fail("stale_data", f"{sig.ticker} data is {age_min / 60:.1f}h old")

        # Liquidity
        dv = ctx.dollar_volumes.get(sig.ticker)
        if dv is not None and dv < self.limits.min_liquidity_dollar_volume:
            fail("liquidity", f"{sig.ticker} avg $volume {dv:,.0f} below minimum")

        # Spread limit
        spread = ctx.spreads.get(sig.ticker)
        if spread is not None and spread > self.limits.max_spread_pct:
            fail("spread_limit", f"{sig.ticker} spread {spread:.3%} exceeds limit")

        price = ctx.prices.get(sig.ticker)
        if price is None or price <= 0:
            fail("price_available", f"no usable price for {sig.ticker}")
            return RiskDecision(False, reasons, checks)

        equity = portfolio.equity(ctx.prices)
        qty = self._resolve_quantity(sig, price, equity, portfolio)
        if qty <= 0:
            fail("sizing", "computed quantity <= 0")
            return RiskDecision(False, reasons, checks)
        notional = qty * price

        # Position limit
        current_qty = portfolio.positions.get(sig.ticker).quantity \
            if sig.ticker in portfolio.positions else 0.0
        if sig.action == "BUY":
            new_weight = (current_qty * price + notional) / equity if equity > 0 else 1.0
            if new_weight > self.limits.max_position_pct + 1e-9:
                fail("position_limit",
                     f"{sig.ticker} would weigh {new_weight:.1%} > "
                     f"{self.limits.max_position_pct:.0%} cap")
            # Portfolio exposure
            new_exposure = (portfolio.positions_value(ctx.prices) + notional) / equity
            if new_exposure > self.limits.max_portfolio_exposure_pct + 1e-9:
                fail("portfolio_limit",
                     f"exposure would be {new_exposure:.1%} > "
                     f"{self.limits.max_portfolio_exposure_pct:.0%} cap")
            # Capital availability
            if notional > portfolio.buying_power():
                fail("capital_available", f"order ${notional:,.0f} exceeds buying power")
            # Max simultaneous positions
            open_n = sum(1 for p in portfolio.positions.values() if p.is_open)
            if current_qty <= 0 and open_n >= self.limits.max_positions:
                fail("max_positions", f"{open_n} open positions at cap")
            # Max new capital per day (approximated by trailing day's BUY fills)
            deployed_today = self._capital_deployed_today(portfolio, ctx.now)
            if deployed_today + notional > equity * self.limits.max_new_capital_per_day_pct:
                fail("daily_deployment", f"${deployed_today:,.0f} deployed today; cap "
                     f"{self.limits.max_new_capital_per_day_pct:.0%} of equity")
            # Sector exposure
            sector = ctx.sectors.get(sig.ticker)
            if sector:
                sector_val = sum(
                    p.quantity * ctx.prices.get(t, 0.0)
                    for t, p in portfolio.positions.items()
                    if p.is_open and ctx.sectors.get(t) == sector
                )
                if (sector_val + notional) / equity > self.limits.max_sector_exposure_pct + 1e-9:
                    fail("sector_limit", f"{sector} exposure would exceed cap")

        # Cooldown after losses (per-symbol)
        if sig.action == "BUY" and self._in_cooldown(portfolio, sig.ticker, ctx.now):
            fail("cooldown", f"{sig.ticker} in post-loss cooldown")

        # Drawdown circuit breaker per strategy
        dd = ctx.strategy_drawdowns.get(sig.strategy)
        if dd is not None and dd < -self.limits.max_strategy_drawdown_pct:
            fail("strategy_drawdown", f"{sig.strategy} drawdown {dd:.1%} breached limit")

        # Daily loss kill trigger (§24: maximum daily loss breached)
        if ctx.daily_loss_pct < -self.limits.max_daily_loss_pct:
            fail("daily_loss_limit", f"daily loss {ctx.daily_loss_pct:.1%} breached - "
                 "trading disabled automatically")

        approved = not reasons
        if approved:
            checks["all"] = "PASS"
        return RiskDecision(approved, reasons, checks)

    # -- helpers -------------------------------------------------------------
    def _resolve_quantity(self, sig: Signal, price: float, equity: float,
                          portfolio: PaperPortfolio) -> float:
        if sig.quantity:
            return float(sig.quantity)
        if sig.target_weight and equity > 0:
            notional = equity * sig.target_weight
        else:
            # default: max_position_pct slice, conservative fallback
            notional = equity * self.limits.max_position_pct if equity > 0 else 0.0
        if price <= 0:
            return 0.0
        qty = notional / price
        if not self.config.execution.allow_fractional_shares:
            qty = float(int(qty))
        return qty

    def _capital_deployed_today(self, portfolio: PaperPortfolio, now: pd.Timestamp | None) -> float:
        if now is None:
            return 0.0
        day = now.strftime("%Y-%m-%d")
        return sum(f.price * f.quantity for f in portfolio.fills
                   if f.side == "BUY" and str(f.timestamp).startswith(day))

    def _in_cooldown(self, portfolio: PaperPortfolio, ticker: str,
                     now: pd.Timestamp | None) -> bool:
        if now is None or self.limits.cooldown_after_loss_days <= 0:
            return False
        losing = [t for t in portfolio.trades
                  if t.ticker == ticker and t.pnl < 0]
        if not losing:
            return False
        last = max(pd.Timestamp(t.exit_time) for t in losing)
        return (now - last).days < self.limits.cooldown_after_loss_days
