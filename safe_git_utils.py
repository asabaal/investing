#!/usr/bin/env python3
"""
Safe Git Utilities for Asabaal Projects

This module provides utilities for safe Git operations, particularly useful
for reverting problematic commits, creating clean snapshots, and other
git operations that are helpful during development and debugging.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_git_command(command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a git command safely and return the result.
    
    Args:
        command: List of command components (e.g. ['git', 'status'])
        capture_output: Whether to capture and return command output
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        logger.debug(f"Running git command: {' '.join(command)}")
        
        if capture_output:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        else:
            # Run the command with output directly to terminal
            result = subprocess.run(command, check=False)
            return result.returncode, "", ""
    except Exception as e:
        logger.error(f"Error running git command: {e}")
        return 1, "", str(e)

def get_repo_root() -> str:
    """
    Get the root directory of the current git repository.
    
    Returns:
        Path to git repository root
    """
    code, stdout, stderr = run_git_command(['git', 'rev-parse', '--show-toplevel'])
    if code != 0:
        logger.error(f"Failed to get repository root: {stderr}")
        return os.getcwd()  # Fall back to current directory
    return stdout.strip()

def is_repo_clean() -> bool:
    """
    Check if the repository has uncommitted changes.
    
    Returns:
        True if repository is clean, False otherwise
    """
    code, stdout, stderr = run_git_command(['git', 'status', '--porcelain'])
    if code != 0:
        logger.error(f"Failed to check repository status: {stderr}")
        return False
    return stdout.strip() == ""

def get_current_branch() -> str:
    """
    Get the name of the current branch.
    
    Returns:
        Name of the current branch
    """
    code, stdout, stderr = run_git_command(['git', 'branch', '--show-current'])
    if code != 0:
        logger.error(f"Failed to get current branch: {stderr}")
        return "unknown"
    return stdout.strip()

def create_snapshot(message: str = None) -> Tuple[bool, str]:
    """
    Create a snapshot commit of the current state.
    
    Args:
        message: Commit message (default: auto-generated timestamp message)
        
    Returns:
        Tuple of (success, commit_hash)
    """
    if not is_repo_clean():
        # First add all changes
        code, _, stderr = run_git_command(['git', 'add', '.'])
        if code != 0:
            logger.error(f"Failed to stage changes: {stderr}")
            return False, ""
        
        # Create commit
        if message is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"SNAPSHOT: {timestamp}"
        
        code, stdout, stderr = run_git_command(['git', 'commit', '-m', message])
        if code != 0:
            logger.error(f"Failed to create snapshot commit: {stderr}")
            return False, ""
        
        # Get the commit hash
        code, stdout, stderr = run_git_command(['git', 'rev-parse', 'HEAD'])
        if code != 0:
            logger.error(f"Failed to get commit hash: {stderr}")
            return True, "unknown"
        
        return True, stdout.strip()
    else:
        logger.info("Repository is clean, no snapshot needed")
        
        # Get the current commit hash
        code, stdout, stderr = run_git_command(['git', 'rev-parse', 'HEAD'])
        if code != 0:
            logger.error(f"Failed to get commit hash: {stderr}")
            return True, "unknown"
        
        return True, stdout.strip()

def revert_commit(commit_hash: str, no_edit: bool = True) -> bool:
    """
    Revert a specific commit.
    
    Args:
        commit_hash: Hash of the commit to revert
        no_edit: Whether to skip the commit message editor
        
    Returns:
        True if successful, False otherwise
    """
    cmd = ['git', 'revert', commit_hash]
    if no_edit:
        cmd.append('--no-edit')
    
    code, stdout, stderr = run_git_command(cmd, capture_output=False)
    if code != 0:
        logger.error(f"Failed to revert commit {commit_hash}: {stderr}")
        return False
    
    logger.info(f"Successfully reverted commit {commit_hash}")
    return True

def revert_last_n_commits(n: int, no_edit: bool = True) -> bool:
    """
    Revert the last N commits in reverse order.
    
    Args:
        n: Number of commits to revert
        no_edit: Whether to skip the commit message editor
        
    Returns:
        True if all reverts were successful, False otherwise
    """
    # Get the last N commit hashes
    code, stdout, stderr = run_git_command(['git', 'log', f'-{n}', '--pretty=format:%H'])
    if code != 0:
        logger.error(f"Failed to get last {n} commits: {stderr}")
        return False
    
    commit_hashes = stdout.strip().split('\n')
    
    # Revert in reverse order (oldest first)
    success = True
    for commit_hash in reversed(commit_hashes):
        if not revert_commit(commit_hash, no_edit):
            success = False
    
    return success

def create_backup_branch() -> Tuple[bool, str]:
    """
    Create a backup branch of the current state.
    
    Returns:
        Tuple of (success, branch_name)
    """
    # Generate branch name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_branch = get_current_branch()
    backup_branch = f"backup/{current_branch}/{timestamp}"
    
    # Create the branch
    code, stdout, stderr = run_git_command(['git', 'branch', backup_branch])
    if code != 0:
        logger.error(f"Failed to create backup branch: {stderr}")
        return False, ""
    
    logger.info(f"Created backup branch: {backup_branch}")
    return True, backup_branch

def reset_to_commit(commit_hash: str, hard: bool = False) -> bool:
    """
    Reset the repository to a specific commit.
    
    Args:
        commit_hash: Hash of the commit to reset to
        hard: Whether to use --hard reset (discards all changes)
        
    Returns:
        True if successful, False otherwise
    """
    # Create backup branch first for safety
    backup_success, backup_branch = create_backup_branch()
    if not backup_success:
        logger.warning("Failed to create backup branch, proceeding with caution")
    
    # Perform the reset
    reset_type = "--hard" if hard else "--mixed"
    
    code, stdout, stderr = run_git_command(['git', 'reset', reset_type, commit_hash], capture_output=False)
    if code != 0:
        logger.error(f"Failed to reset to commit {commit_hash}: {stderr}")
        return False
    
    if hard:
        logger.info(f"Hard reset to commit {commit_hash}")
    else:
        logger.info(f"Soft reset to commit {commit_hash} (changes preserved in working directory)")
    
    return True

def get_recent_commits(n: int = 10) -> List[Dict[str, str]]:
    """
    Get information about recent commits.
    
    Args:
        n: Number of commits to retrieve
        
    Returns:
        List of commit dictionaries with 'hash', 'author', 'date', and 'message'
    """
    format_str = '%H|%an|%ad|%s'  # hash, author name, author date, subject
    code, stdout, stderr = run_git_command(['git', 'log', f'-{n}', f'--pretty=format:{format_str}'])
    if code != 0:
        logger.error(f"Failed to get recent commits: {stderr}")
        return []
    
    commits = []
    for line in stdout.strip().split('\n'):
        if not line:
            continue
        
        parts = line.split('|', 3)
        if len(parts) == 4:
            commits.append({
                'hash': parts[0],
                'author': parts[1],
                'date': parts[2],
                'message': parts[3]
            })
    
    return commits

def cherry_pick_commit(commit_hash: str) -> bool:
    """
    Cherry-pick a specific commit.
    
    Args:
        commit_hash: Hash of the commit to cherry-pick
        
    Returns:
        True if successful, False otherwise
    """
    code, stdout, stderr = run_git_command(['git', 'cherry-pick', commit_hash], capture_output=False)
    if code != 0:
        logger.error(f"Failed to cherry-pick commit {commit_hash}: {stderr}")
        return False
    
    logger.info(f"Successfully cherry-picked commit {commit_hash}")
    return True

def interactive_revert() -> bool:
    """
    Interactive mode to select and revert commits.
    
    Returns:
        True if any commits were successfully reverted
    """
    # Get recent commits
    commits = get_recent_commits(20)  # Show last 20 commits
    if not commits:
        logger.error("No recent commits found")
        return False
    
    # Display commits
    print("\nRecent commits:")
    print("-" * 80)
    for i, commit in enumerate(commits):
        print(f"{i+1:2d}. [{commit['hash'][:8]}] {commit['date']} - {commit['message']}")
    print("-" * 80)
    
    # Get selection from user
    selection = input("Enter commit numbers to revert (comma separated), or 'q' to quit: ")
    if selection.lower() == 'q':
        return False
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_commits = [commits[i]['hash'] for i in indices if 0 <= i < len(commits)]
        
        if not selected_commits:
            logger.error("No valid commits selected")
            return False
        
        # Confirm with user
        print("\nSelected commits to revert:")
        for i, hash in enumerate(selected_commits):
            commit = next((c for c in commits if c['hash'] == hash), None)
            if commit:
                print(f"{i+1}. [{commit['hash'][:8]}] {commit['message']}")
        
        confirm = input("\nProceed with revert? (y/n): ")
        if confirm.lower() != 'y':
            return False
        
        # Create backup branch
        backup_success, backup_branch = create_backup_branch()
        if backup_success:
            print(f"Created backup branch: {backup_branch}")
        
        # Revert selected commits in reverse order
        success = True
        for commit_hash in reversed(selected_commits):
            if not revert_commit(commit_hash):
                success = False
        
        return success
    
    except Exception as e:
        logger.error(f"Error during interactive revert: {e}")
        return False

def print_help() -> None:
    """Print help message with available commands."""
    print("\nSafe Git Utilities")
    print("-" * 80)
    print("Available commands:")
    print("  snapshot                  - Create a snapshot of the current state")
    print("  revert HASH               - Revert a specific commit")
    print("  revert-last N             - Revert the last N commits")
    print("  backup                    - Create a backup branch")
    print("  reset HASH [--hard]       - Reset to a specific commit")
    print("  list [N]                  - List recent commits (default: 10)")
    print("  interactive               - Interactive mode to select and revert commits")
    print("  help                      - Show this help message")
    print("-" * 80)

def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(description="Safe Git Utilities")
    parser.add_argument('command', nargs='?', default='help',
                      help='Command to execute (snapshot, revert, revert-last, backup, reset, list, interactive, help)')
    parser.add_argument('args', nargs='*', help='Additional arguments for the command')
    parser.add_argument('--hard', action='store_true', help='Use hard reset (for reset command)')
    parser.add_argument('--no-edit', action='store_true', help='Skip commit message editor')
    
    args = parser.parse_args()
    
    # Process commands
    if args.command == 'snapshot':
        message = args.args[0] if args.args else None
        success, commit_hash = create_snapshot(message)
        if success:
            print(f"Created snapshot commit: {commit_hash}")
    
    elif args.command == 'revert':
        if not args.args:
            print("Error: Missing commit hash")
            return 1
        
        if revert_commit(args.args[0], args.no_edit):
            print(f"Successfully reverted commit {args.args[0]}")
    
    elif args.command == 'revert-last':
        if not args.args:
            print("Error: Missing number of commits to revert")
            return 1
        
        try:
            n = int(args.args[0])
            if revert_last_n_commits(n, args.no_edit):
                print(f"Successfully reverted last {n} commits")
        except ValueError:
            print("Error: Number of commits must be an integer")
            return 1
    
    elif args.command == 'backup':
        success, branch_name = create_backup_branch()
        if success:
            print(f"Created backup branch: {branch_name}")
    
    elif args.command == 'reset':
        if not args.args:
            print("Error: Missing commit hash")
            return 1
        
        if reset_to_commit(args.args[0], args.hard):
            print(f"Successfully reset to commit {args.args[0]}")
    
    elif args.command == 'list':
        n = 10
        if args.args:
            try:
                n = int(args.args[0])
            except ValueError:
                print("Error: Number of commits must be an integer")
                return 1
        
        commits = get_recent_commits(n)
        if commits:
            print("\nRecent commits:")
            print("-" * 80)
            for i, commit in enumerate(commits):
                print(f"{i+1:2d}. [{commit['hash'][:8]}] {commit['date']} - {commit['message']}")
            print("-" * 80)
    
    elif args.command == 'interactive':
        interactive_revert()
    
    elif args.command == 'help':
        print_help()
    
    else:
        print(f"Unknown command: {args.command}")
        print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
