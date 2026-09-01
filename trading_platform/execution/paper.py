"""Simulated broker: fills market orders at decision price +/- slippage (§13, §15).

Also powers SHADOW mode's bookkeeping via `expected_price` - the shadow record
of "what we would have done" (plan §20).
"""

from __future__ import annotations

from ..config import ExecutionConfig
from ..portfolio.paper import PaperPortfolio
from .base import ExecutionBroker, FillResult, Order


class PaperExecutionBroker(ExecutionBroker):
    def __init__(self, portfolio: PaperPortfolio, execution: ExecutionConfig,
                 db=None, mode: str = "paper", kill_switch_on: bool = False):
        self.portfolio = portfolio
        self.execution = execution
        self.db = db
        self.mode = mode
        self.kill_switch_on = kill_switch_on
        self._orders: dict[str, Order] = {}

    def review_order(self, order: Order) -> tuple[bool, str]:
        if self.kill_switch_on and self.mode in ("shadow", "live"):
            return False, "kill switch engaged - order not submitted"
        if order.quantity <= 0:
            return False, "non-positive quantity"
        return True, "ok"

    def submit_order(self, order: Order, decision_price: float) -> FillResult:
        """Fill immediately at decision price frictioned toward the trade."""
        ok, note = self.review_order(order)
        self._orders[order.order_id] = order
        if not ok:
            order.status = "REJECTED"
            self._persist(order, "REJECTED")
            return FillResult(order.order_id, filled=False, note=note)
        slip_bps = self.execution.slippage_bps / 10_000.0
        fill_price = decision_price * (1 + slip_bps if order.side == "BUY" else 1 - slip_bps)
        try:
            rec = self.portfolio.apply_fill(
                ticker=order.ticker, side=order.side, quantity=order.quantity,
                price=fill_price, timestamp=order.created_at, strategy=order.strategy,
                reference_price=decision_price,
            )
        except ValueError as exc:
            order.status = "REJECTED"
            self._persist(order, "REJECTED")
            return FillResult(order.order_id, filled=False, note=str(exc))
        order.status = "FILLED"
        self._persist(order, "FILLED", rec.price)
        if self.db is not None:
            self.db.insert_fill(order.order_id, order.created_at, rec.price,
                                rec.quantity, rec.commission, rec.slippage_cost)
        return FillResult(order.order_id, filled=True, price=rec.price,
                          quantity=rec.quantity, commission=rec.commission,
                          slippage=rec.slippage_cost)

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status in ("NEW", "REVIEWED", "SUBMITTED", "PENDING_MANUAL"):
            order.status = "CANCELLED"
            self._persist(order, "CANCELLED")
            return True
        return False

    def get_order_status(self, order_id: str) -> str:
        order = self._orders.get(order_id)
        return order.status if order else "UNKNOWN"

    def _persist(self, order: Order, status: str, fill_price: float | None = None) -> None:
        if self.db is None:
            return
        self.db.insert_order({
            "order_id": order.order_id, "signal_id": order.signal_id, "mode": order.mode,
            "portfolio": order.portfolio, "ticker": order.ticker, "side": order.side,
            "quantity": order.quantity, "expected_price": fill_price or order.expected_price,
            "status": status, "created_at": order.created_at,
        })
