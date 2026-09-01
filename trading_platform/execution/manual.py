"""Manual-notification broker (plan §22 human-in-the-loop mode).

Generates exactly the order it would execute, formats the §22 notification
template, and requires a human to trade manually.  No automated brokerage
authority.
"""

from __future__ import annotations

from ..notifications.channels import NotificationRouter
from .base import ExecutionBroker, FillResult, Order


class ManualNotificationBroker(ExecutionBroker):
    def __init__(self, router: NotificationRouter):
        self.router = router
        self._orders: dict[str, Order] = {}

    def review_order(self, order: Order) -> tuple[bool, str]:
        if order.quantity <= 0:
            return False, "non-positive quantity"
        return True, "manual review required"

    def submit_order(self, order: Order, decision_price: float) -> FillResult:
        ok, _ = self.review_order(order)
        self._orders[order.order_id] = order
        if not ok:
            return FillResult(order.order_id, filled=False, note="rejected at review")
        order.status = "PENDING_MANUAL"
        body = (
            "MODEL SIGNAL\n\n"
            f"Ticker: {order.ticker}\n"
            f"Strategy: {order.strategy}\n"
            f"Action: {order.side}\n"
            f"Quantity: {order.quantity:.4g} shares\n\n"
            f"Current price: ${decision_price:.2f}\n\n"
            "Open Robinhood to review."
        )
        self.router.send("ACTION", f"Manual trade review required: {order.side} {order.ticker}",
                         body, signal_id=order.signal_id)
        return FillResult(order.order_id, filled=False,
                          note="LOG ONLY - awaiting human execution (plan §22)")

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order:
            order.status = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> str:
        order = self._orders.get(order_id)
        return order.status if order else "UNKNOWN"
