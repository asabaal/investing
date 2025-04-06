#!/usr/bin/env python3
"""
Test Script for Logging Functionality

This script tests logging functionality in the Symphony Trading System to ensure
logs are properly written to files.
"""

import os
import sys
import logging
from datetime import datetime

def test_basic_logging():
    """Test basic logging functionality to a file"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_dir}/test_logging_{timestamp}.log"
    
    # Configure file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("test_logging")
    
    # Log various messages at different levels
    logger.debug("This is a DEBUG message - should not appear unless DEBUG level is enabled")
    logger.info("This is an INFO message - should appear in logs")
    logger.warning("This is a WARNING message - should appear in logs")
    logger.error("This is an ERROR message - should appear in logs")
    
    # Verify log file was created and has content
    file_exists = os.path.exists(log_filename)
    file_size = os.path.getsize(log_filename) if file_exists else 0
    
    print(f"\nLog file: {log_filename}")
    print(f"File exists: {file_exists}")
    print(f"File size: {file_size} bytes")
    
    if file_exists and file_size > 0:
        print("✅ Basic logging test PASSED")
        return True
    else:
        print("❌ Basic logging test FAILED - log file is empty or doesn't exist")
        return False

def test_improved_logging():
    """Test improved logging setup from symphony_backtester_fix.py"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_dir}/test_improved_logging_{timestamp}.log"
    
    # Configure logging with both file and console handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    
    # Set formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set log level and add handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get a named logger
    logger = logging.getLogger("test_improved_logging")
    
    # Log various messages at different levels
    logger.debug("This is a DEBUG message - should not appear unless DEBUG level is enabled")
    logger.info("This is an INFO message - should appear in logs")
    logger.warning("This is a WARNING message - should appear in logs")
    logger.error("This is an ERROR message - should appear in logs")
    
    # Verify log file was created and has content
    file_exists = os.path.exists(log_filename)
    file_size = os.path.getsize(log_filename) if file_exists else 0
    
    print(f"\nLog file: {log_filename}")
    print(f"File exists: {file_exists}")
    print(f"File size: {file_size} bytes")
    
    if file_exists and file_size > 0:
        print("✅ Improved logging test PASSED")
        return True
    else:
        print("❌ Improved logging test FAILED - log file is empty or doesn't exist")
        return False

def main():
    """Run the logging tests"""
    print("=== TESTING LOGGING FUNCTIONALITY ===")
    
    print("\n1. Testing basic logging configuration...")
    basic_test_passed = test_basic_logging()
    
    print("\n2. Testing improved logging configuration...")
    improved_test_passed = test_improved_logging()
    
    print("\n=== TEST SUMMARY ===")
    print(f"Basic Logging: {'PASSED' if basic_test_passed else 'FAILED'}")
    print(f"Improved Logging: {'PASSED' if improved_test_passed else 'FAILED'}")
    
    if basic_test_passed and improved_test_passed:
        print("\n✅ All tests PASSED")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
