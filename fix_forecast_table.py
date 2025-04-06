#!/usr/bin/env python3
"""
Fix script for forecast table display in Symphony Trading System reports.
This script applies a targeted fix to the report_generator.py file.
"""

import os
import re
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of the original file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{file_path}.{timestamp}.bak"
    
    try:
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Backup created at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def fix_forecast_table_section(file_path="report_generator.py"):
    """
    Apply targeted fix to the forecast table section in report_generator.py.
    
    This fix modifies the _generate_forecast_section method to ensure it
    correctly iterates through symbols and extracts forecast data.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Create backup
    if not backup_file(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern to find the forecast table generation code
        # Looking for the section that iterates through symbols
        pattern = r'(for symbol in symphony_info\[\'symbols\'\]:.*?if symbol in forecasts:.*?symbol_forecast = forecasts\[symbol\].*?)(\n\s+# Skip symbols with errors.*?\n\s+if \'error\' in symbol_forecast:.*?\n\s+continue.*?)'
        
        # The replacement - remove the error check that's skipping symbols
        replacement = r'\1'
        
        # Apply the replacement
        updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Pattern to find how forecast values are extracted
        # Looking for the code that gets forecast_7d and forecast_30d
        value_pattern = r'(forecast_7d = self\._get_nested_value\(symbol_forecast, \[\'7_day\', \'percent_change\'\], 0\.0\).*?forecast_30d = self\._get_nested_value\(symbol_forecast, \[\'30_day\', \'percent_change\'\], 0\.0\))'
        
        # Replacement with direct extraction of forecast values
        value_replacement = r"""# Extract forecast data with fallbacks
                if '7_day' in symbol_forecast and isinstance(symbol_forecast['7_day'], dict) and 'percent_change' in symbol_forecast['7_day']:
                    forecast_7d = float(symbol_forecast['7_day']['percent_change'])
                else:
                    forecast_7d = 0.0
                
                if '30_day' in symbol_forecast and isinstance(symbol_forecast['30_day'], dict) and 'percent_change' in symbol_forecast['30_day']:
                    forecast_30d = float(symbol_forecast['30_day']['percent_change'])
                else:
                    forecast_30d = 0.0"""
        
        # Apply the value extraction replacement
        updated_content = re.sub(value_pattern, value_replacement, updated_content, flags=re.DOTALL)
        
        # Write modified content back to file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Applied forecast table fix to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply fix: {e}")
        return False

def main():
    """Main function to apply the fix."""
    logger.info("Starting forecast table fix script")
    
    # Default path
    file_path = "report_generator.py"
    
    # Allow specifying a different path via command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Apply the fix
    if fix_forecast_table_section(file_path):
        logger.info("Fix applied successfully")
        print("\nForecast table fix applied successfully!")
        print("Please rebuild and run the system to verify the fix works.\n")
    else:
        logger.error("Failed to apply the fix")
        print("\nFailed to apply the forecast table fix. See log for details.\n")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
