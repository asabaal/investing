#!/usr/bin/env python3
"""
Log Analyzer Tool

This script analyzes log files from the Symphony Trading System and provides a summary
of message types and examples, making it easier to diagnose problems without being
overwhelmed by the volume of log messages.
"""

import os
import sys
import re
import argparse
import logging
from collections import defaultdict, Counter
from datetime import datetime

def analyze_log_file(log_file, max_examples=3, min_count=1, show_stats=True, filter_level=None, filter_module=None):
    """
    Analyze a log file and group messages by type.
    
    Args:
        log_file (str): Path to log file
        max_examples (int): Maximum number of examples to show per message type
        min_count (int): Minimum count to include a message type
        show_stats (bool): Whether to show statistics
        filter_level (str): Only show messages of this level (INFO, ERROR, etc.)
        filter_module (str): Only show messages from this module
        
    Returns:
        dict: Analysis results
    """
    # Define log levels for coloring
    level_colors = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m', # Magenta
        'DEFAULT': '\033[0m'  # Reset
    }
    
    # Regular expression to parse log lines
    # Format: timestamp - module - level - message
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    
    # Store grouped messages
    message_groups = defaultdict(list)
    message_counts = Counter()
    level_counts = Counter()
    module_counts = Counter()
    
    print(f"Analyzing log file: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            line_count = 0
            filtered_count = 0
            
            for line in f:
                line_count += 1
                match = re.match(log_pattern, line.strip())
                
                if match:
                    timestamp, module, level, message = match.groups()
                    
                    # Apply filters if specified
                    if filter_level and level != filter_level:
                        filtered_count += 1
                        continue
                    
                    if filter_module and module != filter_module:
                        filtered_count += 1
                        continue
                    
                    # Count levels and modules
                    level_counts[level] += 1
                    module_counts[module] += 1
                    
                    # Create a message signature to group similar messages
                    # Replace numbers, dates, variable names with placeholders
                    signature = re.sub(r'\d+\.\d+', '<FLOAT>', message)
                    signature = re.sub(r'\d+', '<NUMBER>', signature)
                    signature = re.sub(r'\b[A-Za-z0-9_]+\.[A-Za-z0-9_.]+\b', '<VARIABLE>', signature)
                    signature = re.sub(r'\b[A-Za-z0-9_]+@[A-Za-z0-9_]+\b', '<EMAIL>', signature)
                    signature = re.sub(r'\b[0-9a-f]{8,}\b', '<HASH>', signature)
                    signature = re.sub(r'(19|20)\d\d-\d\d-\d\d', '<DATE>', signature)
                    
                    # Include level and module in the signature to separate different sources
                    full_signature = f"{level} - {module} - {signature}"
                    
                    # Store the original message with its timestamp, level and module
                    message_groups[full_signature].append((timestamp, level, module, message))
                    message_counts[full_signature] += 1
                else:
                    # Continuation of a previous message or malformed line
                    pass
        
        # Print statistics if requested
        if show_stats:
            print("\n=== LOG STATISTICS ===")
            print(f"Total lines: {line_count}")
            print(f"Filtered lines: {filtered_count}")
            print(f"Unique message types: {len(message_counts)}")
            
            print("\nMessage levels:")
            for level, count in sorted(level_counts.items(), key=lambda x: x[1], reverse=True):
                color = level_colors.get(level, level_colors['DEFAULT'])
                print(f"  {color}{level}{level_colors['DEFAULT']}: {count}")
            
            print("\nModules:")
            for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {module}: {count}")
        
        # Print message groups with examples
        print("\n=== MESSAGE TYPES ===")
        
        # Sort message groups by count (most frequent first)
        sorted_groups = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)
        
        for signature, count in sorted_groups:
            if count < min_count:
                continue
            
            # Extract level and module from signature
            parts = signature.split(' - ', 2)
            level = parts[0]
            module = parts[1]
            message_type = parts[2] if len(parts) > 2 else signature
            
            color = level_colors.get(level, level_colors['DEFAULT'])
            print(f"\n{color}{level}{level_colors['DEFAULT']} - {module} - Count: {count}")
            print(f"  Pattern: {message_type}")
            
            # Print examples (limited to max_examples)
            print("  Examples:")
            examples = message_groups[signature][:max_examples]
            for i, (timestamp, msg_level, msg_module, message) in enumerate(examples, 1):
                print(f"    {i}. [{timestamp}] {message}")
        
        return {
            "line_count": line_count,
            "filtered_count": filtered_count,
            "unique_messages": len(message_counts),
            "level_counts": level_counts,
            "module_counts": module_counts,
            "message_groups": message_groups,
            "message_counts": message_counts
        }
    
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        return None

def get_most_recent_log(log_dir="logs", prefix=None):
    """
    Get the most recent log file in the specified directory.
    
    Args:
        log_dir (str): Directory to search for log files
        prefix (str): Prefix to filter log files
        
    Returns:
        str: Path to the most recent log file
    """
    if not os.path.exists(log_dir):
        return None
    
    log_files = []
    for f in os.listdir(log_dir):
        if prefix and not f.startswith(prefix):
            continue
        
        if f.endswith('.log'):
            file_path = os.path.join(log_dir, f)
            log_files.append((file_path, os.path.getmtime(file_path)))
    
    if not log_files:
        return None
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x[1], reverse=True)
    return log_files[0][0]

def main():
    """Main function to run the log analyzer"""
    parser = argparse.ArgumentParser(description='Analyze Symphony Trading System log files')
    
    parser.add_argument('--file', '-f', help='Path to log file (default: most recent log)')
    parser.add_argument('--dir', '-d', default='logs', help='Directory containing log files')
    parser.add_argument('--prefix', '-p', help='Prefix to filter log files')
    parser.add_argument('--examples', '-e', type=int, default=3, help='Maximum examples per message type')
    parser.add_argument('--min-count', '-m', type=int, default=1, help='Minimum count to include a message type')
    parser.add_argument('--no-stats', action='store_true', help='Do not show statistics')
    parser.add_argument('--level', '-l', help='Filter by log level (INFO, ERROR, etc.)')
    parser.add_argument('--module', help='Filter by module name')
    
    args = parser.parse_args()
    
    # Determine log file to analyze
    log_file = args.file
    if not log_file:
        log_file = get_most_recent_log(args.dir, args.prefix)
    
    if not log_file or not os.path.exists(log_file):
        print(f"Error: Log file not found. Please specify a valid log file.")
        return 1
    
    # Analyze the log file
    analyze_log_file(
        log_file,
        max_examples=args.examples,
        min_count=args.min_count,
        show_stats=not args.no_stats,
        filter_level=args.level,
        filter_module=args.module
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
