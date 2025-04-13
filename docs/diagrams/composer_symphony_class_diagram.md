# Composer Symphony Module - Class Diagram

```mermaid
classDiagram
    class SymbolList {
        +List[str] symbols
        +str name
        +__init__(symbols, name)
        +__repr__() str
        +__len__() int
        +__iter__()
        +__getitem__(index)
        +validate_symbols(client) Dict[str, bool]
        +add_symbol(symbol) void
        +remove_symbol(symbol) void
        +intersect(other) SymbolList
        +union(other) SymbolList
    }

    class SymbologyOperator {
        +str name
        +__init__(name)
        +__repr__() str
        +execute(inputs) Any
    }

    class Filter {
        +str name
        +Dict condition
        +__init__(name, condition)
        +execute(symbols, market_data) SymbolList
    }

    class Momentum {
        +str name
        +Dict condition
        +int lookback_days
        +int top_n
        +__init__(name, lookback_days, top_n)
        +execute(symbols, market_data) SymbolList
    }

    class RSIFilter {
        +str name
        +Dict condition
        +float threshold
        +str condition
        +__init__(name, threshold, condition)
        +execute(symbols, market_data, technical_data) SymbolList
    }

    class MovingAverageCrossover {
        +str name
        +Dict condition
        +int fast_period
        +int slow_period
        +str condition
        +__init__(name, fast_period, slow_period, condition)
        +execute(symbols, market_data) SymbolList
    }

    class Allocator {
        +str name
        +str allocation_method
        +__init__(name, allocation_method)
        +execute(symbols) Dict[str, float]
    }

    class EqualWeightAllocator {
        +str name
        +str allocation_method
        +__init__(name)
        +execute(symbols) Dict[str, float]
    }

    class InverseVolatilityAllocator {
        +str name
        +str allocation_method
        +int lookback_days
        +__init__(name, lookback_days)
        +execute(symbols, market_data) Dict[str, float]
    }

    class Symphony {
        +str name
        +str description
        +SymbolList universe
        +List[SymbologyOperator] operators
        +Allocator allocator
        +__init__(name, description, universe)
        +add_operator(operator) void
        +set_allocator(allocator) void
        +execute(client) Dict[str, float]
        +to_dict() Dict
        +to_json() str
        +from_dict(data) Symphony
        +from_json(json_str) Symphony
    }

    class SymphonyBacktester {
        +AlphaVantageClient client
        +__init__(client)
        +get_historical_data(symbols, start_date, end_date) Dict[str, DataFrame]
        +backtest(symphony, start_date, end_date, rebalance_frequency, initial_capital) Dict
        +_mock_execute_symphony(symphony, mock_client, as_of_date) Dict[str, float]
        +_apply_momentum_filter(symbols, data, lookback_days, top_n, as_of_date) List[str]
        +_apply_ma_crossover_filter(symbols, data, fast_period, slow_period, condition, as_of_date) List[str]
        +plot_backtest_results(results, benchmark_symbol) void
    }

    class MockAlphaVantageClient {
        +Dict[str, DataFrame] data
        +__init__(data)
        +get_daily(symbol, **kwargs) DataFrame
        +get_rsi(symbol, **kwargs) DataFrame
        +get_sma(symbol, time_period, **kwargs) DataFrame
        +get_quote(symbol) Dict
    }

    SymbologyOperator <|-- Filter
    SymbologyOperator <|-- Allocator
    Filter <|-- Momentum
    Filter <|-- RSIFilter
    Filter <|-- MovingAverageCrossover
    Allocator <|-- EqualWeightAllocator
    Allocator <|-- InverseVolatilityAllocator
    
    Symphony o-- SymbolList : contains
    Symphony o-- SymbologyOperator : contains
    Symphony o-- Allocator : contains
    SymphonyBacktester -- Symphony : backtests
    MockAlphaVantageClient -- SymphonyBacktester : used by
```

## Class Relationships

### Core Data Structure
- `SymbolList`: Container for managing lists of stock/ETF symbols

### Operator Hierarchy
- `SymbologyOperator`: Abstract base class for all operators
  - `Filter`: Base class for operators that filter symbols
    - `Momentum`: Selects top performing symbols by momentum
    - `RSIFilter`: Filters symbols based on RSI values
    - `MovingAverageCrossover`: Selects symbols based on MA crossovers
  - `Allocator`: Base class for allocation methods
    - `EqualWeightAllocator`: Assigns equal weight to all symbols
    - `InverseVolatilityAllocator`: Weights inversely to volatility

### Strategy Definition
- `Symphony`: Complete trading strategy defined by operators
  - Contains `SymbolList` as the universe
  - Contains multiple `SymbologyOperator` instances
  - Contains an `Allocator` for weighting

### Testing & Simulation
- `SymphonyBacktester`: Backtests Symphony strategies
- `MockAlphaVantageClient`: Mock client for testing/backtesting

## Key Dependencies

- `AlphaVantageClient`: For retrieving market data
- `pandas`: For data manipulation
- `matplotlib`: For visualization

## Architecture Analysis

### Current Design

The module uses an object-oriented design with good separation of concerns:

1. `Symphony` acts as a container for the strategy components
2. The operator hierarchy allows for extensibility
3. `SymphonyBacktester` separates testing from strategy definition

### Potential Issues

1. `Symphony.execute()` method is complex and could be broken down
2. `SymphonyBacktester.backtest()` is very long and does multiple things
3. The mock execution methods contain duplicate logic from the real implementations
4. Error handling is inconsistent across methods

### Refactoring Targets

1. **Symphony.execute()**: Should be decomposed into smaller methods
2. **SymphonyBacktester.backtest()**: Should be split into multiple methods
3. **_mock_execute_symphony()**: Could share logic with actual `execute()` method
4. **Data validation**: Add consistent validation across all methods
