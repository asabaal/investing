"""Platform configuration.

Secrets come exclusively from the environment (plan §29) - never from source
control.  The kill switch (plan §24) defaults to *disabled* trading: when
uncertain, stop trading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PLATFORM_DIR / "data" / "platform.db"
DEFAULT_LOG_DIR = PLATFORM_DIR / "logs"
DEFAULT_ALERT_DIR = PLATFORM_DIR / "alerts"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class RiskLimits:
    """Plan §12 portfolio protection controls (all configurable)."""

    max_position_pct: float = 0.05          # max single-position weight of equity
    max_portfolio_exposure_pct: float = 0.95  # max invested fraction of equity
    max_sector_exposure_pct: float = 0.25   # max weight per sector
    max_new_capital_per_day_pct: float = 0.10  # max fresh capital deployed per day
    max_daily_loss_pct: float = 0.03        # daily portfolio loss that trips the kill switch
    max_strategy_drawdown_pct: float = 0.20  # per-strategy drawdown circuit breaker
    max_positions: int = 25                 # max simultaneous open positions
    cooldown_after_loss_days: int = 2       # per-symbol cooldown after a losing exit
    min_liquidity_dollar_volume: float = 250_000.0  # min 20-bar avg dollar volume
    max_spread_pct: float = 0.01            # max bid/ask spread as fraction of mid
    max_data_age_minutes: float = 30.0      # stale-data detection window


@dataclass
class ExecutionConfig:
    """Fill simulation assumptions (plan §13 slippage / friction)."""

    slippage_bps: float = 5.0               # friction applied against the trade
    commission_per_share: float = 0.0       # Robinhood-like zero commission
    min_commission: float = 0.0
    default_half_spread_bps: float = 5.0    # used when no bid/ask available
    allow_fractional_shares: bool = True


@dataclass
class Config:
    db_path: Path = field(default_factory=lambda: Path(
        os.environ.get("TRADING_PLATFORM_DB", DEFAULT_DB_PATH)))
    log_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("TRADING_PLATFORM_LOG_DIR", DEFAULT_LOG_DIR)))
    alert_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("TRADING_PLATFORM_ALERT_DIR", DEFAULT_ALERT_DIR)))

    # Market data
    data_provider: str = field(default_factory=lambda: os.environ.get("DATA_PROVIDER", "alpha_vantage"))
    alpha_vantage_api_key: str = field(default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY", ""))
    quote_batch_size: int = _env_int("QUOTE_BATCH_SIZE", 20)  # plan §4 batching

    # Modes: backtest | paper | shadow | live
    mode: str = field(default_factory=lambda: os.environ.get("TRADING_MODE", "paper"))
    benchmark: str = "SPY"

    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @property
    def trading_enabled(self) -> bool:
        """Plan §24 global kill switch.  Live/shadow orders require it.

        Two switches must agree: the TRADING_ENABLED environment variable and
        the persistent state file data/TRADING_ENABLED (so the switch survives
        across cron-spawned processes).  Default when uncertain: disabled.
        """
        env_on = os.environ.get("TRADING_ENABLED", "false").strip().lower() == "true"
        file_flag = DEFAULT_DB_PATH.parent / "TRADING_ENABLED"
        file_on = file_flag.exists()
        return env_on and file_on

    @property
    def live_trading_enabled(self) -> bool:
        """Separate, stricter gate for actual brokerage submission (§23)."""
        return self.trading_enabled and os.environ.get(
            "LIVE_TRADING_ENABLED", "false").strip().lower() == "true"


def load_config() -> Config:
    return Config()
