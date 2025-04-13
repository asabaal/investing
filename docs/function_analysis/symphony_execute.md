# Function Specification: Symphony.execute()

## Overview

| Field | Description |
|-------|-------------|
| **Function Name** | execute |
| **Module** | composer_symphony.py |
| **Purpose** | Execute a symphony to generate portfolio allocations based on specified filters and allocator |
| **Current Status** | Functional but with multiple responsibilities and minimal error handling |

## Current Implementation

```python
def execute(self, client: AlphaVantageClient) -> Dict[str, float]:
    """
    Execute the symphony to generate allocations.
    
    Args:
        client: Alpha Vantage client for market data
        
    Returns:
        Dictionary mapping symbols to allocation weights
    """
    # Fetch market data for universe
    market_data = {}
    technical_data = {}
    
    for symbol in self.universe:
        try:
            # Get daily price data
            market_data[symbol] = client.get_daily(symbol)
            
            # Get RSI technical indicator
            technical_data[symbol] = client.get_rsi(symbol)
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {str(e)}")
    
    # Apply operators in sequence
    current_symbols = self.universe
    
    for operator in self.operators:
        if isinstance(operator, Filter):
            if isinstance(operator, RSIFilter):
                current_symbols = operator.execute(current_symbols, market_data, technical_data)
            else:
                current_symbols = operator.execute(current_symbols, market_data)
    
    # Apply allocator to final symbol list
    if isinstance(self.allocator, EqualWeightAllocator):
        allocations = self.allocator.execute(current_symbols)
    else:
        allocations = self.allocator.execute(current_symbols, market_data)
    
    return allocations
```

## Specification

### Inputs
- **client** (AlphaVantageClient): Client for fetching market data, required

### Outputs
- **Dict[str, float]**: Dictionary mapping symbols to allocation weights
  - Keys are symbol strings (e.g., "SPY")
  - Values are float allocation weights (0.0-1.0) that sum to 1.0
- **Possible exceptions**: None specified in current implementation, but could fail if client API calls fail

### Behavior
1. Fetches market and technical data for each symbol in the universe
2. Iterates through defined operators, applying them in sequence
3. Uses appropriate data for each operator type
4. Applies the allocator to the final filtered symbol list
5. Returns the resulting allocation dictionary

## Implementation Assessment

### Current Implementation Issues
1. **Multiple responsibilities**: Fetching data, applying filters, allocation
2. **Error handling**: Minimal error handling for data fetching, none for other operations
3. **Type checking**: Uses isinstance() checks that could be replaced with polymorphism
4. **Validation**: No validation of input or output data
5. **Logging**: Minimal logging for debugging purposes

### Single Responsibility Analysis
This function should be split into:
1. **fetch_market_data()**: Fetch and prepare data needed for execution
2. **apply_filters()**: Apply the filters to the universe
3. **calculate_allocations()**: Apply the allocator to the filtered symbols
4. **execute()**: Orchestrate the above steps

### Dependencies
- Depends on AlphaVantageClient for data
- Depends on Filter, RSIFilter, and Allocator class implementations
- Tight coupling to specific implementations (RSIFilter, EqualWeightAllocator)

## Testing Strategy

### Test Cases
1. **Basic execution**: Test with a simple symphony, verify allocations returned
2. **Empty universe**: Test with empty universe, should return empty allocations
3. **Failed data fetching**: Test with some symbols failing to fetch data
4. **No symbols after filtering**: Test when all symbols are filtered out
5. **Different allocators**: Test with each allocator type

### Mocking Requirements
- Mock AlphaVantageClient to return predictable data
- Mock Filter and Allocator classes to isolate testing

## Refactoring Plan

### Proposed Changes
1. Extract data fetching to a separate method
2. Extract filter application to a separate method
3. Extract allocation calculation to a separate method
4. Add proper validation and error handling
5. Improve logging for debugging
6. Use polymorphism instead of explicit type checking

### New Function Signatures

```python
def fetch_market_data(self, client: AlphaVantageClient) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Fetch market data for all symbols in the universe.
    
    Args:
        client: Alpha Vantage client for market data
        
    Returns:
        Tuple of (market_data, technical_data) dictionaries
    """

def apply_filters(self, current_symbols: SymbolList, market_data: Dict[str, pd.DataFrame], 
                 technical_data: Dict[str, pd.DataFrame]) -> SymbolList:
    """
    Apply all filters in sequence to the symbol list.
    
    Args:
        current_symbols: Current set of symbols
        market_data: Dictionary of market data by symbol
        technical_data: Dictionary of technical data by symbol
        
    Returns:
        Filtered symbol list
    """

def calculate_allocations(self, symbols: SymbolList, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate allocations for the given symbols.
    
    Args:
        symbols: Symbol list to allocate
        market_data: Dictionary of market data by symbol
        
    Returns:
        Dictionary mapping symbols to weights
    """

def execute(self, client: AlphaVantageClient) -> Dict[str, float]:
    """
    Execute the symphony to generate allocations.
    
    Args:
        client: Alpha Vantage client for market data
        
    Returns:
        Dictionary mapping symbols to allocation weights
    """
```

### Backward Compatibility
- The refactored `execute()` method will maintain the same signature
- The internal implementation changes won't affect external callers
- The extracted methods will be used internally by execute()

## Documentation

### Updated Function Docstring

```python
def execute(self, client: AlphaVantageClient) -> Dict[str, float]:
    """
    Execute the symphony to generate allocations based on defined filters and allocator.
    
    This method fetches required market data, applies all defined filters in sequence,
    and then calculates allocations using the configured allocator.
    
    Args:
        client: Alpha Vantage client for market data
        
    Returns:
        Dictionary mapping symbols to allocation weights (0.0-1.0), summing to 1.0
    
    Raises:
        ValueError: If the symphony is improperly configured
        RuntimeError: If all symbols are filtered out and no allocations can be made
    """
```

### Usage Example

```python
# Create a symphony
universe = SymbolList(["SPY", "QQQ", "IWM"])
symphony = Symphony("Sample Symphony", "A test symphony", universe)
symphony.add_operator(Momentum("Momentum Filter", lookback_days=90, top_n=2))
symphony.set_allocator(EqualWeightAllocator())

# Execute symphony
client = AlphaVantageClient(api_key="YOUR_API_KEY")
allocations = symphony.execute(client)

# Use allocations
for symbol, weight in allocations.items():
    print(f"{symbol}: {weight:.2%}")
```

## Implementation Checklist

- [ ] Write tests for current `execute()` behavior
- [ ] Extract `fetch_market_data()` method with tests
- [ ] Extract `apply_filters()` method with tests
- [ ] Extract `calculate_allocations()` method with tests
- [ ] Refactor `execute()` to use the new methods
- [ ] Add validation and improved error handling
- [ ] Update documentation
- [ ] Verify all tests pass
