"""Execution broker interface (plan §21)."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str
    quantity: float
    mode: str
    portfolio: str = "default"
    signal_id: str | None = None
    expected_price: float | None = None
    strategy: str = ""
    status: str = "NEW"
    created_at: str = ""


@dataclass
class FillResult:
    order_id: str
    filled: bool
    price: float | None = None
    quantity: float | None = None
    commission: float = 0.0
    slippage: float = 0.0
    note: str = ""


def make_order_id(mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{mode}-{ts}-{uuid.uuid4().hex[:8]}"


class ExecutionBroker(ABC):
    """review_order / submit_order / cancel_order / get_order_status (§21)."""

    @abstractmethod
    def review_order(self, order: Order) -> tuple[bool, str]:
        """Broker-side pre-submission review; returns (ok, note)."""

    @abstractmethod
    def submit_order(self, order: Order, decision_price: float) -> FillResult:
        """Submit and return the fill result (immediate for market orders)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        ...
