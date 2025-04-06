#!/usr/bin/env python3
"""
Fix script for NaN volume handling in Prophet forecasting module.
This script applies a targeted fix to make the forecasting more tolerant of NaN values.
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

def fix_volume_nan_handling(file_path="prophet_forecasting.py"):
    """
    Apply targeted fix to improve NaN handling in volume data.
    
    This fix modifies the prepare_data method to be more robust
    when handling NaN values in the volume column.
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
        
        # Pattern to find the volume handling code in prepare_data method
        pattern = r'(# Add volume as a regressor if available\n\s+if \'volume\' in df_copy\.columns:.*?# Ensure volume is clean and positive before log transform.*?volume = df_copy\[\'volume\'\]\.copy\(\).*?volume = volume\.clip\(lower=1\).*?# Normalize volume to reduce impact of extreme values.*?prophet_df\[\'volume\'\] = np\.log1p\(volume\).*?# Double-check for NaN values.*?if prophet_df\[\'volume\'\]\.isna\(\)\.any\(\):.*?logger\.warning.*?prophet_df\[\'volume\'\] = prophet_df\[\'volume\'\]\.fillna\(prophet_df\[\'volume\'\]\.mean\(\)\))'
        
        # Replacement with more robust NaN handling
        replacement = r"""# Add volume as a regressor if available
        if 'volume' in df_copy.columns:
            # Check for NaN values first before any transformations
            if df_copy['volume'].isna().any():
                logger.info(f"Found NaN values in volume column, filling with median")
                # Use median for filling to be more robust against outliers
                volume_median = df_copy['volume'].median()
                # If median is NaN, use mean. If mean is NaN, use a default value
                if pd.isna(volume_median):
                    volume_mean = df_copy['volume'].mean()
                    if pd.isna(volume_mean):
                        fill_value = 1000000  # Default volume value
                        logger.warning(f"Both median and mean are NaN, using default volume: {fill_value}")
                    else:
                        fill_value = volume_mean
                        logger.info(f"Using mean volume for NaN values: {fill_value}")
                else:
                    fill_value = volume_median
                    logger.info(f"Using median volume for NaN values: {fill_value}")
                
                # Fill NaN values
                df_copy['volume'] = df_copy['volume'].fillna(fill_value)
            
            # Now we can safely work with the volume data
            volume = df_copy['volume'].copy()
            
            # Replace any remaining NaNs or negative/zero values
            if volume.isna().any() or (volume <= 0).any():
                logger.info("Ensuring volume is positive for log transform")
                # Replace any remaining NaNs or non-positive values with 1
                volume = volume.fillna(1).clip(lower=1)
            
            # Normalize volume to reduce impact of extreme values
            prophet_df['volume'] = np.log1p(volume)
            
            # Final safety check
            if prophet_df['volume'].isna().any():
                logger.warning(f"Still found NaN values in volume after cleaning, replacing with mean")
                mean_volume = prophet_df['volume'].mean()
                if pd.isna(mean_volume):
                    mean_volume = 0  # Last resort fallback
                prophet_df['volume'] = prophet_df['volume'].fillna(mean_volume)"""
        
        # Apply the replacement
        updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Fix the fit_prophet_model method to be more tolerant of NaN values
        model_pattern = r'(# Add volume regressor if available and requested\n\s+if include_volume and \'volume\' in df\.columns:.*?# Double-check for any NaN values\n\s+if df\[\'volume\'\]\.isna\(\)\.any\(\):.*?logger\.warning.*?df\[\'volume\'\] = df\[\'volume\'\]\.fillna\(df\[\'volume\'\]\.mean\(\)\))'
        
        model_replacement = r"""# Add volume regressor if available and requested
        if include_volume and 'volume' in df.columns:
            # Handle NaN values in volume robustly
            if df['volume'].isna().any():
                logger.warning("Found NaN in volume column")
                
                # Try median first (more robust to outliers)
                median_val = df['volume'].median()
                if pd.isna(median_val):
                    # Try mean if median is NaN
                    mean_val = df['volume'].mean()
                    if pd.isna(mean_val):
                        # Last resort - use a constant value
                        fill_val = 1.0
                        logger.warning(f"Both median and mean are NaN, using constant {fill_val}")
                    else:
                        fill_val = mean_val
                        logger.info(f"Using mean {fill_val} to fill NaN values")
                else:
                    fill_val = median_val
                    logger.info(f"Using median {fill_val} to fill NaN values")
                    
                # Fill NaN values with chosen strategy
                df['volume'] = df['volume'].fillna(fill_val)
            
            # Verify no NaNs remain
            if df['volume'].isna().any():
                logger.error("Failed to clean volume data, using constant value")
                df['volume'] = 1.0  # Replace everything with a safe value as last resort"""
                
        # Apply the model replacement
        updated_content = re.sub(model_pattern, model_replacement, updated_content, flags=re.DOTALL)
        
        # Also fix the forecast_ensemble method
        ensemble_pattern = r'(# Failed to fit any models for .+?)'
        ensemble_replacement = r"""# Log the specific error for debugging
                logger.warning(f"Model fitting error: {e}")
                # Continue trying with more models instead of failing completely\1"""
        
        # Apply the ensemble replacement
        updated_content = re.sub(ensemble_pattern, ensemble_replacement, updated_content, flags=re.DOTALL)
        
        # Write modified content back to file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Applied volume NaN handling fix to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply fix: {e}")
        return False

def main():
    """Main function to apply the fix."""
    logger.info("Starting volume NaN handling fix script")
    
    # Default path
    file_path = "prophet_forecasting.py"
    
    # Allow specifying a different path via command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Apply the fix
    if fix_volume_nan_handling(file_path):
        logger.info("Fix applied successfully")
        print("\nVolume NaN handling fix applied successfully!")
        print("Please run the backtest again to verify the fix works:\n")
        print("python symphony_backtester.py your_symphony.json --enhanced-report\n")
    else:
        logger.error("Failed to apply the fix")
        print("\nFailed to apply the volume NaN handling fix. See log for details.\n")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
