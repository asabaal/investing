"""Trading research, paper trading & live execution platform.

Implements NEW_PLAN/trading-plan.md: one strategy codebase operating across
BACKTEST -> FORWARD PAPER TRADING -> SHADOW LIVE -> LIVE modes, with only the
adapters around it changing.  The critical loop is deterministic software;
AI is never required for routine execution.
"""

__version__ = "0.1.0"
