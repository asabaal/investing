"""
Usage example for the refactored Supply & Demand Zone Detection Algorithm
Demonstrates the benefits of Single Responsibility Principle refactoring
"""

from datetime import datetime, timedelta
from typing import List

# Import the refactored components
from core_models import Candle, AlgorithmConfig, ZoneType
from business_logic import SupplyDemandAlgorithm
from test_utilities import TestDataFactory, CandleBuilder, TestAssertions


def main_usage_example():
    """
    Main example showing how to use the refactored algorithm
    """
    print("=" * 60)
    print("REFACTORED SUPPLY & DEMAND ALGORITHM USAGE EXAMPLE")
    print("=" * 60)
    
    # Example 1: Basic usage with default configuration
    print("\n1. BASIC USAGE WITH DEFAULT CONFIGURATION")
    print("-" * 50)
    
    # Create algorithm with default settings
    algorithm = SupplyDemandAlgorithm()
    
    # Create some test data
    candles = create_sample_market_data()
    
    # Analyze the data
    result = algorithm.analyze_market_data(candles)
    
    # Display results
    print(f"Analyzed {len(result.candles)} candles")
    print(f"Found {result.total_zones} zones:")
    print(f"  - Supply zones: {len(result.supply_zones)}")
    print(f"  - Demand zones: {len(result.demand_zones)}")
    
    # Show detailed zone information
    for i, zone in enumerate(result.zones):
        print(f"\nZone {i+1} ({zone.zone_type.value.upper()}):")
        print(f"  Price range: {zone.low:.2f} - {zone.high:.2f}")
        print(f"  Base candles: {len(zone.base_candles)}")
        print(f"  Pattern: LEG({zone.entry_leg_index}) → BASE({zone.base_candles}) → LEG({zone.exit_leg_index})")
    
    
    # Example 2: Custom configuration
    print("\n\n2. CUSTOM CONFIGURATION EXAMPLE")
    print("-" * 50)
    
    # Create custom configuration
    custom_config = AlgorithmConfig(
        body_ratio_threshold=1.5,  # Stricter requirement for LEG candles
        min_base_candles=2,        # Require at least 2 base candles
        max_base_candles=4         # Limit consolidation size
    )
    
    # Create algorithm with custom config
    custom_algorithm = SupplyDemandAlgorithm(custom_config)
    
    # Use more complex test data
    complex_candles = TestDataFactory.create_complex_market_data()
    
    # Analyze with custom settings
    custom_result = custom_algorithm.analyze_market_data(complex_candles)
    
    print(f"Custom analysis of {len(custom_result.candles)} candles:")
    print(f"Found {custom_result.total_zones} zones (vs {result.total_zones} with default config)")
    
    # Get zone summary
    summary = custom_algorithm.get_zone_summary(list(custom_result.zones))
    if summary['count'] > 0:
        print(f"Average zone range: {summary['avg_range']:.2f}")
        print(f"Average base candles per zone: {summary['avg_base_candles']:.1f}")
    
    
    # Example 3: Component testing and validation
    print("\n\n3. COMPONENT TESTING AND VALIDATION")
    print("-" * 50)
    
    demonstrate_component_testing()
    
    
    # Example 4: Real-world usage patterns
    print("\n\n4. REAL-WORLD USAGE PATTERNS")
    print("-" * 50)
    
    demonstrate_real_world_patterns()


def create_sample_market_data() -> List[Candle]:
    """
    Create sample market data for demonstration
    """
    builder = CandleBuilder()
    base_time = datetime(2024, 1, 1, 9, 0)
    
    return [
        # Initial movement
        builder.reset().with_timestamp(base_time).with_ohlc(100, 102, 99, 101).build(),
        
        # Supply zone pattern: Bullish LEG → BASE → Bearish LEG
        builder.reset().with_timestamp(base_time + timedelta(minutes=5))
               .with_ohlc(101, 115, 100, 113).build(),  # Strong bullish LEG
        
        builder.reset().with_timestamp(base_time + timedelta(minutes=10))
               .with_ohlc(113, 115, 111, 113).build(),  # BASE (consolidation)
        
        builder.reset().with_timestamp(base_time + timedelta(minutes=15))
               .with_ohlc(113, 114, 105, 106).build(),  # Strong bearish LEG
        
        # Some continuation
        builder.reset().with_timestamp(base_time + timedelta(minutes=20))
               .with_ohlc(106, 108, 104, 105).build(),
        
        # Demand zone pattern: Bearish LEG → BASE → Bullish LEG
        builder.reset().with_timestamp(base_time + timedelta(minutes=25))
               .with_ohlc(105, 106, 90, 92).build(),    # Strong bearish LEG
        
        builder.reset().with_timestamp(base_time + timedelta(minutes=30))
               .with_ohlc(92, 95, 89, 93).build(),      # BASE (consolidation)
        
        builder.reset().with_timestamp(base_time + timedelta(minutes=35))
               .with_ohlc(93, 107, 92, 105).build(),    # Strong bullish LEG
    ]


def demonstrate_component_testing():
    """
    Demonstrate how the SRP refactoring enables better testing
    """
    print("Testing individual components:")
    
    # Test data factory
    supply_pattern = TestDataFactory.create_simple_supply_pattern()
    demand_pattern = TestDataFactory.create_simple_demand_pattern()
    
    print(f"✓ Created supply pattern with {len(supply_pattern)} candles")
    print(f"✓ Created demand pattern with {len(demand_pattern)} candles")
    
    # Test data validation
    try:
        for candle in supply_pattern:
            TestAssertions.assert_valid_candle(candle)
        print("✓ All supply pattern candles are valid")
        
        for candle in demand_pattern:
            TestAssertions.assert_valid_candle(candle)
        print("✓ All demand pattern candles are valid")
        
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test algorithm with known patterns
    algorithm = SupplyDemandAlgorithm()
    
    supply_result = algorithm.analyze_market_data(supply_pattern)
    if supply_result.total_zones == 1 and len(supply_result.supply_zones) == 1:
        print("✓ Supply pattern correctly detected")
    else:
        print("✗ Supply pattern detection failed")
    
    demand_result = algorithm.analyze_market_data(demand_pattern)
    if demand_result.total_zones == 1 and len(demand_result.demand_zones) == 1:
        print("✓ Demand pattern correctly detected")
    else:
        print("✗ Demand pattern detection failed")


def demonstrate_real_world_patterns():
    """
    Demonstrate realistic trading scenarios
    """
    print("Real-world scenario analysis:")
    
    # Scenario 1: Market with strong trends and clear zones
    print("\nScenario 1: Strong trending market")
    trending_data = create_trending_market_data()
    
    algorithm = SupplyDemandAlgorithm()
    trend_result = algorithm.analyze_market_data(trending_data)
    
    print(f"  Zones found: {trend_result.total_zones}")
    print(f"  Supply zones: {len(trend_result.supply_zones)}")
    print(f"  Demand zones: {len(trend_result.demand_zones)}")
    
    # Scenario 2: Choppy market with many false signals
    print("\nScenario 2: Choppy/ranging market")
    choppy_data = create_choppy_market_data()
    
    # Use stricter configuration for choppy markets
    strict_config = AlgorithmConfig(
        body_ratio_threshold=2.0,  # Require very strong legs
        min_base_candles=3,        # Require longer consolidation
        max_base_candles=6
    )
    
    strict_algorithm = SupplyDemandAlgorithm(strict_config)
    choppy_result = strict_algorithm.analyze_market_data(choppy_data)
    
    print(f"  Zones found with strict config: {choppy_result.total_zones}")
    print(f"  (Filters out weaker signals in choppy conditions)")
    
    # Scenario 3: Performance comparison
    print("\nScenario 3: Configuration comparison")
    test_data = TestDataFactory.create_complex_market_data()
    
    configs = {
        "Conservative": AlgorithmConfig(body_ratio_threshold=2.0, min_base_candles=3),
        "Standard": AlgorithmConfig(body_ratio_threshold=1.0, min_base_candles=1),
        "Aggressive": AlgorithmConfig(body_ratio_threshold=0.5, min_base_candles=1),
    }
    
    for name, config in configs.items():
        algo = SupplyDemandAlgorithm(config)
        result = algo.analyze_market_data(test_data)
        summary = algo.get_zone_summary(list(result.zones))
        
        print(f"  {name:12}: {result.total_zones:2d} zones "
              f"(S:{len(result.supply_zones)}, D:{len(result.demand_zones)})")


def create_trending_market_data() -> List[Candle]:
    """Create data representing a strong trending market"""
    builder = CandleBuilder()
    base_time = datetime(2024, 1, 1, 9, 0)
    candles = []
    
    current_price = 100.0
    
    for i in range(15):
        timestamp = base_time + timedelta(minutes=i * 5)
        
        if i % 4 == 0:  # Strong trend leg every 4 candles
            if i < 8:  # First half: uptrend
                candle = builder.reset().with_timestamp(timestamp).with_ohlc(
                    current_price, current_price + 8, current_price - 1, current_price + 7
                ).build()
                current_price += 7
            else:  # Second half: downtrend
                candle = builder.reset().with_timestamp(timestamp).with_ohlc(
                    current_price, current_price + 1, current_price - 8, current_price - 7
                ).build()
                current_price -= 7
        else:  # Consolidation candles
            candle = builder.reset().with_timestamp(timestamp).with_ohlc(
                current_price, current_price + 2, current_price - 2, current_price + 0.5
            ).build()
            current_price += 0.5
        
        candles.append(candle)
    
    return candles


def create_choppy_market_data() -> List[Candle]:
    """Create data representing a choppy, ranging market"""
    builder = CandleBuilder()
    base_time = datetime(2024, 1, 1, 9, 0)
    candles = []
    
    base_price = 100.0
    
    for i in range(12):
        timestamp = base_time + timedelta(minutes=i * 5)
        
        # Create small, overlapping moves that don't form clear patterns
        price_variation = 2.0 * (i % 3 - 1)  # -2, 0, +2 pattern
        current_price = base_price + price_variation
        
        candle = builder.reset().with_timestamp(timestamp).with_ohlc(
            current_price,
            current_price + 1.5,
            current_price - 1.5,
            current_price + (0.5 if i % 2 else -0.5)
        ).build()
        
        candles.append(candle)
    
    return candles


def benefits_of_refactoring():
    """
    Demonstrate the benefits of the SRP refactoring
    """
    print("\n\n" + "=" * 60)
    print("BENEFITS OF SINGLE RESPONSIBILITY PRINCIPLE REFACTORING")
    print("=" * 60)
    
    print("\n1. MODULARITY & TESTABILITY:")
    print("   - Each class has one clear responsibility")
    print("   - Easy to unit test individual components")
    print("   - Mock objects can replace dependencies for testing")
    print("   - Property-based testing for edge cases")
    
    print("\n2. EXTENSIBILITY:")
    print("   - New classification strategies can be plugged in")
    print("   - Zone detection logic can be extended")
    print("   - Different analysis methods can be added")
    print("   - Configuration-driven behavior")
    
    print("\n3. MAINTAINABILITY:")
    print("   - Clear separation of concerns")
    print("   - Immutable data models prevent side effects")
    print("   - Validation logic is centralized")
    print("   - Easy to debug specific components")
    
    print("\n4. REUSABILITY:")
    print("   - Components can be used independently")
    print("   - Test utilities support multiple scenarios")
    print("   - Data models are framework-agnostic")
    print("   - Business logic separated from data structures")
    
    print("\n5. RELIABILITY:")
    print("   - Comprehensive test coverage")
    print("   - Input validation at model level")
    print("   - Consistent error handling")
    print("   - Property-based testing for edge cases")


if __name__ == "__main__":
    main_usage_example()
    benefits_of_refactoring()
