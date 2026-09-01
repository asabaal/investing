"""Execution adapters (plan §21).

The same signal + risk pipeline feeds simulation, manual execution or
automated execution purely by swapping the broker.
"""

from .base import ExecutionBroker, make_order_id
from .paper import PaperExecutionBroker
from .manual import ManualNotificationBroker

__all__ = ["ExecutionBroker", "make_order_id", "PaperExecutionBroker",
           "ManualNotificationBroker"]
