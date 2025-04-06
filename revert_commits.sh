#!/bin/bash
# Script to revert the recent forecasting fix commits

# Set -e to exit on error
set -e

echo "Starting reversion of recent commits..."

# Revert the most recent commit (report_generator.py changes)
git revert 1f8d60b34e8d4ae5ba39ff875c5e78b56ce2afba --no-edit
echo "Reverted report_generator.py changes"

# Revert the second commit (symphony_analyzer.py changes)
git revert 43b2eab72c4bc254eb3d6bc944048c062f2982b0 --no-edit
echo "Reverted symphony_analyzer.py changes"

# Revert the first commit (prophet_forecasting.py changes)
git revert 24fbe2488dff65e6a7918544ab7915efb0f81692 --no-edit
echo "Reverted prophet_forecasting.py changes"

echo "All commits reverted successfully"
echo "The system should now be back to its previous state"
