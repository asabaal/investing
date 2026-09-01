"""Market data provider abstraction (plan §8).

Strategy code never knows which vendor it is talking to.  Implementations:

- AlphaVantageProvider  - wraps the repo's existing alpha_vantage_api client
- LocalDatabaseProvider  - replays bars already stored in the platform DB
- RobinhoodMCPProvider   - planned primary source; stub until MCP access exists
- SyntheticRandomProvider- deterministic random-walk, used by tests only
"""

from .base import MarketDataProvider, Quote, quote_to_bars

__all__ = ["MarketDataProvider", "Quote", "quote_to_bars"]
