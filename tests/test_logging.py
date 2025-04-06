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

# Add parent directory to path to allow imports from main package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    """Test improved logging setup from symphony_backtester.py"""
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

def test_symphony_backtester_logging():
    """Test the logging setup in the symphony_backtester module"""
    try:
        # Import the setup_logging function from symphony_backtester
        from symphony_backtester import setup_logging
        
        # Use the setup_logging function
        logger = setup_logging()
        
        # Log some test messages
        logger.info("This is a test INFO message from symphony_backtester")
        logger.warning("This is a test WARNING message from symphony_backtester")
        logger.error("This is a test ERROR message from symphony_backtester")
        
        # Success if we got this far without errors
        print("✅ Symphony backtester logging test PASSED")
        return True
    except Exception as e:
        print(f"❌ Symphony backtester logging test FAILED: {str(e)}")
        return False

def main():
    """Run the logging tests"""
    print("=== TESTING LOGGING FUNCTIONALITY ===")
    
    print("\n1. Testing basic logging configuration...")
    basic_test_passed = test_basic_logging()
    
    print("\n2. Testing improved logging configuration...")
    improved_test_passed = test_improved_logging()
    
    print("\n3. Testing symphony_backtester logging...")
    backtester_test_passed = test_symphony_backtester_logging()
    
    print("\n=== TEST SUMMARY ===")
    print(f"Basic Logging: {'PASSED' if basic_test_passed else 'FAILED'}")
    print(f"Improved Logging: {'PASSED' if improved_test_passed else 'FAILED'}")
    print(f"Symphony Backtester Logging: {'PASSED' if backtester_test_passed else 'FAILED'}")
    
    if basic_test_passed and improved_test_passed and backtester_test_passed:
        print("\n✅ All tests PASSED")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
