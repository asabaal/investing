# Alpha Vantage API for Investing Symphony

This guide explains how to use the enhanced Alpha Vantage API client for your Symphony Trading System.

## Alpha Vantage API Issues and Solutions

The Alpha Vantage API has different tiers of access:
- Free: Limited to 5 calls per minute, 500 per day
- Premium: Allows 75+ calls per minute, access to "premium endpoints"

Some endpoints like `TIME_SERIES_DAILY_ADJUSTED` with `outputsize=full` are premium-only. Without premium access, you'll get the error:

```
Premium endpoint required: Thank you for using Alpha Vantage! This is a premium endpoint...
```

## How to Check Your API Access Level

We've added a script to check your API access level:

```bash
export ALPHA_VANTAGE_API_KEY=your_api_key_here
python check_api_access.py
```

This will check both standard and premium endpoints to help you understand what level of access you have.

## Solutions for Non-Premium Users

1. **Fallback Mode**: The client will automatically try to use free endpoints to simulate premium ones
2. **Automatic Data Cleaning**: NaN values in data are handled gracefully
3. **Better Error Handling**: Premium endpoint errors are clearly identified

## Testing the API Client

### 1. Run the Automated Tests

```bash
cd tests
python test_alpha_vantage.py
```

Don't worry if you see error messages like "Premium endpoint required" - this is part of the test to ensure proper error handling.

### 2. Run the Manual Test

```bash
export ALPHA_VANTAGE_API_KEY=your_api_key_here
python test_api_manually.py
```

This will test both data cleaning and actual API access.

### 3. Try Your Symphony Again

```bash
python symphony_cli.py analyze sample_symphony.json
```

## Upgrading to Premium (Optional)

If you need the full capabilities, consider upgrading to a premium Alpha Vantage subscription at [alphavantage.co/premium](https://www.alphavantage.co/premium/).

## Troubleshooting

If you're still having issues:

1. Check your API key is set correctly in your environment
2. Verify your API key access level using `check_api_access.py`
3. Look for specific error messages in the logs

## API Client Features

- Robust rate limiting to avoid exceeding API limits
- Persistent rate limiting across program executions
- Data cleaning to handle missing values
- Fallback mechanisms for premium endpoints
- Comprehensive error handling
- Automatic data parsing and formatting
