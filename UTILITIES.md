# Symphony Trading System Utilities

This document explains the utility scripts available in the Symphony Trading System for debugging, fixing issues, and managing code changes.

## Diagnostic Tools

### diagnostic_forecasts.py

This script helps diagnose issues with the forecasting system by analyzing the data structures and identifying problems.

**Usage:**
```bash
python diagnostic_forecasts.py
```

**What it does:**
- Tests forecasting for a sample symbol (SPY)
- Tests the SymphonyAnalyzer forecasting functionality
- Saves detailed diagnostic information to the `diagnosis` directory
- Analyzes the structure of forecast data to help identify issues

## Fix Scripts

Fix scripts are specialized tools that make targeted changes to specific files to address issues without manually editing code. They create backups before making changes, making it easy to revert if needed.

### fix_forecast_table.py

Addresses issues with the forecast table in reports by modifying how the report generator handles symbols with errors.

**Usage:**
```bash
python fix_forecast_table.py
```

**What it does:**
- Creates a backup of `report_generator.py` with a timestamp
- Removes error-skipping code that might be hiding symbols in the table
- Improves how forecast values are extracted from the data structure
- Makes the table display more tolerant of different data formats

### fix_volume_nan.py

Addresses issues with NaN values in volume data that cause forecasting to fail.

**Usage:**
```bash
python fix_volume_nan.py
```

**What it does:**
- Creates a backup of `prophet_forecasting.py` with a timestamp
- Improves how NaN values are handled in the volume column
- Makes the forecasting module more robust when dealing with incomplete data
- Adds better fallback strategies for when data cleaning fails

## Git Utilities

### revert_commits.sh

A simple script to revert specific commits that might be causing issues.

**Usage:**
```bash
bash revert_commits.sh
```

**What it does:**
- Reverts the three recent commits made to fix forecasting issues
- Returns the codebase to its previous state
- Uses `git revert` which creates new commits rather than erasing history

### safe_git_utils.py

A comprehensive utility for managing Git operations safely, particularly useful during development and debugging.

**Usage:**
```bash
python safe_git_utils.py [command] [args]
```

**Available commands:**
- `snapshot` - Create a timestamped snapshot of the current state
- `revert HASH` - Revert a specific commit
- `revert-last N` - Revert the last N commits
- `backup` - Create a backup branch of the current state
- `reset HASH [--hard]` - Reset to a specific commit
- `list [N]` - List recent commits (default: 10)
- `interactive` - Interactive mode to select and revert commits
- `help` - Show help message

**Examples:**
```bash
# Create a snapshot of current code state
python safe_git_utils.py snapshot "Before testing fixes"

# Revert the last 3 commits
python safe_git_utils.py revert-last 3

# List the 20 most recent commits
python safe_git_utils.py list 20

# Interactive selection for reverting commits
python safe_git_utils.py interactive
```

## Testing Your Fixes

After applying fixes with the above tools, you should test if they worked:

1. Run the backtest script again:
   ```bash
   python symphony_backtester.py your_symphony.json --enhanced-report
   ```

2. Check if the forecast table now displays properly in the report

3. If issues persist, try running the diagnostic script again to see what changed:
   ```bash
   python diagnostic_forecasts.py
   ```

## Best Practices

1. **Always create backups** before making changes (the fix scripts do this automatically)
2. **Run diagnostics first** to understand the problem before applying fixes
3. **Test each fix** before moving on to the next one
4. **Keep track of changes** using Git snapshots or backups
5. **Use the interactive revert tool** if you need to selectively undo changes
