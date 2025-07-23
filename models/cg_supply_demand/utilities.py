"""
Production utilities module for the Supply & Demand Zone Detection Algorithm
Contains reusable components for production use, data processing, and simulations
"""

from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import replace

from core_models import Candle, Zone, ZoneType, AlgorithmConfig


class CandleBuilder:
    """
    Builder pattern for creating Candle objects.
    Useful for both production data processing and testing.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'CandleBuilder':
        """Reset builder to default values"""
        self._timestamp = datetime(2024, 1, 1, 9, 0)
        self._open = 100.0
        self._high = 105.0
        self._low = 95.0
        self._close = 102.0
        self._volume = 1000.0
        return self
    
    def with_timestamp(self, timestamp: datetime) -> 'CandleBuilder':
        self._timestamp = timestamp
        return self
    
    def with_ohlc(self, open_price: float, high: float, 
                  low: float, close: float) -> 'CandleBuilder':
        self._open = open_price
        self._high = high
        self._low = low
        self._close = close
        return self
    
    def with_volume(self, volume: float) -> 'CandleBuilder':
        self._volume = volume
        return self
    
    def bullish(self, body_size: float = 5.0) -> 'CandleBuilder':
        """Create bullish candle with specified body size"""
        self._close = self._open + body_size
        if self._high < self._close:
            self._high = self._close + 1.0
        return self
    
    def bearish(self, body_size: float = 5.0) -> 'CandleBuilder':
        """Create bearish candle with specified body size"""
        self._close = self._open - body_size
        if self._low > self._close:
            self._low = self._close - 1.0
        return self
    
    def doji(self, wick_size: float = 2.0) -> 'CandleBuilder':
        """Create doji candle (open = close) with specified wick size"""
        self._close = self._open
        self._high = self._open + wick_size
        self._low = self._open - wick_size
        return self
    
    def large_body(self, body_size: float = 10.0, small_wicks: float = 0.5) -> 'CandleBuilder':
        """Create candle with large body and small wicks (LEG candle) - preserves existing direction"""
        # Determine current direction or default to bullish
        current_direction = self._close - self._open
        is_bullish = current_direction >= 0
        
        # Set body size while preserving direction
        if is_bullish:
            self._close = self._open + abs(body_size)
        else:
            self._close = self._open - abs(body_size)
        
        # Set small wicks
        self._high = max(self._open, self._close) + small_wicks
        self._low = min(self._open, self._close) - small_wicks
        return self
    
    def large_wicks(self, body_size: float = 1.0, wick_size: float = 5.0) -> 'CandleBuilder':
        """Create candle with small body and large wicks (BASE candle) - preserves existing direction"""
        # Determine current direction or default to bullish
        current_direction = self._close - self._open
        is_bullish = current_direction >= 0
        
        # Set small body size while preserving direction
        if is_bullish:
            self._close = self._open + abs(body_size)
        else:
            self._close = self._open - abs(body_size)
        
        # Set large wicks
        self._high = max(self._open, self._close) + wick_size
        self._low = min(self._open, self._close) - wick_size
        return self
    
    def build(self) -> Candle:
        """Build the candle"""
        return Candle(
            timestamp=self._timestamp,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume
        )


class ZoneBuilder:
    """
    Builder pattern for creating Zone objects.
    Useful for programmatic zone creation, backtesting, and visualization.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'ZoneBuilder':
        """Reset builder to default values"""
        self._zone_type = ZoneType.SUPPLY
        self._start_index = 0
        self._end_index = 2
        self._high = 110.0
        self._low = 105.0
        self._base_candles = (1,)
        self._entry_leg_index = 0
        self._exit_leg_index = 2
        return self
    
    def supply_zone(self) -> 'ZoneBuilder':
        """Set zone type to SUPPLY"""
        self._zone_type = ZoneType.SUPPLY
        return self
    
    def demand_zone(self) -> 'ZoneBuilder':
        """Set zone type to DEMAND"""
        self._zone_type = ZoneType.DEMAND
        return self
    
    def with_indices(self, start: int, end: int) -> 'ZoneBuilder':
        """Set start and end indices"""
        self._start_index = start
        self._end_index = end
        return self
    
    def with_price_range(self, high: float, low: float) -> 'ZoneBuilder':
        """Set price range for the zone"""
        if high <= low:
            raise ValueError(f"High price {high} must be greater than low price {low}")
        self._high = high
        self._low = low
        return self
    
    def with_base_candles(self, *indices: int) -> 'ZoneBuilder':
        """Set base candle indices"""
        if not indices:
            raise ValueError("Must provide at least one base candle index")
        self._base_candles = tuple(sorted(indices))
        return self
    
    def with_legs(self, entry: int, exit: int) -> 'ZoneBuilder':
        """Set entry and exit leg indices"""
        self._entry_leg_index = entry
        self._exit_leg_index = exit
        return self
    
    def from_candles(self, candles: List[Candle], entry_idx: int, 
                    base_indices: List[int], exit_idx: int, 
                    zone_type: ZoneType) -> 'ZoneBuilder':
        """
        Build zone from actual candle data.
        Useful for creating zones from detected patterns.
        """
        if not base_indices:
            raise ValueError("Must provide at least one base candle index")
        
        # Calculate price range from base candles
        base_candles = [candles[i] for i in base_indices]
        high = max(candle.high for candle in base_candles)
        low = min(candle.low for candle in base_candles)
        
        self._zone_type = zone_type
        self._start_index = entry_idx
        self._end_index = exit_idx
        self._high = high
        self._low = low
        self._base_candles = tuple(sorted(base_indices))
        self._entry_leg_index = entry_idx
        self._exit_leg_index = exit_idx
        
        return self
    
    def build(self) -> Zone:
        """Build the zone with validation"""
        # Validate indices make sense
        if self._start_index >= self._end_index:
            raise ValueError(f"Start index {self._start_index} must be < end index {self._end_index}")
        
        if not self._base_candles:
            raise ValueError("Zone must have at least one base candle")
        
        # Validate leg positions relative to base candles
        min_base = min(self._base_candles)
        max_base = max(self._base_candles)
        
        if self._entry_leg_index >= min_base:
            raise ValueError(f"Entry leg index {self._entry_leg_index} must be before base candles")
        
        if self._exit_leg_index <= max_base:
            raise ValueError(f"Exit leg index {self._exit_leg_index} must be after base candles")
        
        return Zone(
            zone_type=self._zone_type,
            start_index=self._start_index,
            end_index=self._end_index,
            high=self._high,
            low=self._low,
            base_candles=self._base_candles,
            entry_leg_index=self._entry_leg_index,
            exit_leg_index=self._exit_leg_index
        )


class MarketDataProcessor:
    """
    Utility class for processing raw market data into Candle objects.
    Useful for production data ingestion from APIs.
    """
    
    @staticmethod
    def from_ohlc_dict(data: dict) -> Candle:
        """Create candle from dictionary with OHLC data"""
        return Candle(
            timestamp=data.get('timestamp', datetime.now()),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data.get('volume', 0)) if data.get('volume') else None
        )
    
    @staticmethod
    def from_csv_row(row: dict) -> Candle:
        """Create candle from CSV row data"""
        timestamp = datetime.fromisoformat(row['timestamp']) if 'timestamp' in row else datetime.now()
        
        return Candle(
            timestamp=timestamp,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']) if row.get('volume') else None
        )
    
    @staticmethod
    def batch_from_dicts(data_list: List[dict]) -> List[Candle]:
        """Create multiple candles from list of dictionaries"""
        return [MarketDataProcessor.from_ohlc_dict(data) for data in data_list]


class RandomDataGenerator:
    """
    Generates random valid data for simulations, backtesting, and performance testing.
    Useful for Monte Carlo simulations and stress testing.
    """
    
    @staticmethod
    def generate_random_candle(
        min_price: float = 50.0,
        max_price: float = 150.0,
        timestamp: Optional[datetime] = None
    ) -> Candle:
        """Generate a random but valid candle"""
        import random
        
        if timestamp is None:
            timestamp = datetime(2024, 1, 1, 9, 0)
        
        # Generate OHLC ensuring validity constraints
        open_price = random.uniform(min_price, max_price)
        close_price = random.uniform(min_price, max_price)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_price, close_price) + random.uniform(0, 5)
        low = min(open_price, close_price) - random.uniform(0, 5)
        
        volume = random.uniform(100, 10000)
        
        return Candle(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume
        )
    
    @staticmethod
    def generate_random_candles(
        count: int,
        min_price: float = 50.0,
        max_price: float = 150.0,
        interval_minutes: int = 5
    ) -> List[Candle]:
        """Generate a list of random valid candles"""
        base_time = datetime(2024, 1, 1, 9, 0)
        candles = []
        
        for i in range(count):
            timestamp = base_time + timedelta(minutes=i * interval_minutes)
            candle = RandomDataGenerator.generate_random_candle(
                min_price, max_price, timestamp
            )
            candles.append(candle)
        
        return candles
    
    @staticmethod
    def generate_trending_data(count: int, trend_strength: float = 0.5) -> List[Candle]:
        """
        Generate trending market data.
        trend_strength: 0.0 = random walk, 1.0 = strong trend
        """
        import random
        
        candles = []
        current_price = 100.0
        builder = CandleBuilder()
        base_time = datetime(2024, 1, 1, 9, 0)
        
        for i in range(count):
            # Add trend component
            trend_move = random.uniform(-1, 1) * trend_strength
            random_move = random.uniform(-2, 2) * (1 - trend_strength)
            price_change = trend_move + random_move
            
            timestamp = base_time + timedelta(minutes=i * 5)
            
            candle = builder.reset().with_timestamp(timestamp).with_ohlc(
                current_price,
                current_price + abs(price_change) + random.uniform(0.5, 2),
                current_price - abs(price_change) - random.uniform(0.5, 2),
                current_price + price_change
            ).build()
            
            candles.append(candle)
            current_price += price_change
        
        return candles
    
    @staticmethod
    def generate_random_config() -> AlgorithmConfig:
        """Generate random but valid algorithm configuration for testing/simulation"""
        import random
        
        body_ratio_threshold = random.uniform(0.1, 5.0)
        min_base_candles = random.randint(1, 3)
        max_base_candles = random.randint(min_base_candles, 10)
        
        return AlgorithmConfig(
            body_ratio_threshold=body_ratio_threshold,
            min_base_candles=min_base_candles,
            max_base_candles=max_base_candles
        )


class ValidationUtils:
    """
    Utility functions for data validation.
    """
    
    @staticmethod
    def validate_candle_data(candle: Candle) -> List[str]:
        """
        Validate candle data and return list of issues found.
        Returns empty list if no issues.
        """
        issues = []
        
        if candle.high < max(candle.open, candle.close):
            issues.append(f"High {candle.high} is less than max(open {candle.open}, close {candle.close})")
        
        if candle.low > min(candle.open, candle.close):
            issues.append(f"Low {candle.low} is greater than min(open {candle.open}, close {candle.close})")
        
        if candle.volume is not None and candle.volume < 0:
            issues.append(f"Volume {candle.volume} is negative")
        
        return issues
    
    @staticmethod
    def validate_candle_sequence(candles: List[Candle]) -> List[str]:
        """
        Validate a sequence of candles for common issues.
        Returns list of issues found.
        """
        issues = []
        
        if not candles:
            return issues
        
        # Check individual candles
        for i, candle in enumerate(candles):
            candle_issues = ValidationUtils.validate_candle_data(candle)
            for issue in candle_issues:
                issues.append(f"Candle {i}: {issue}")
        
        # Check sequence ordering (timestamps should be ascending)
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i-1].timestamp:
                issues.append(f"Timestamp ordering issue at index {i}: {candles[i].timestamp} <= {candles[i-1].timestamp}")
        
        return issues
    
    @staticmethod
    def is_valid_candle_sequence(candles: List[Candle]) -> bool:
        """Check if candle sequence is valid (no issues found)"""
        return len(ValidationUtils.validate_candle_sequence(candles)) == 0


class TimeSeriesUtils:
    """
    Utility functions for working with time series data.
    """
    
    @staticmethod
    def generate_time_series(start_time: datetime, interval_minutes: int, count: int) -> List[datetime]:
        """Generate a series of timestamps"""
        return [
            start_time + timedelta(minutes=i * interval_minutes)
            for i in range(count)
        ]
    
    @staticmethod
    def align_candles_to_timeframe(candles: List[Candle], 
                                  target_interval_minutes: int) -> List[Candle]:
        """
        Align candles to a specific timeframe (basic implementation).
        This is a simplified version - production would need more sophisticated logic.
        """
        if not candles:
            return []
        
        # This is a placeholder for actual timeframe alignment logic
        # In production, you'd implement proper OHLC aggregation
        return candles
    
    @staticmethod
    def filter_market_hours(candles: List[Candle], 
                           start_hour: int = 9, end_hour: int = 16) -> List[Candle]:
        """Filter candles to only include market hours"""
        return [
            candle for candle in candles
            if start_hour <= candle.timestamp.hour < end_hour
        ]