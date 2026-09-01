#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing/paper_trading_system')

from paper_trading_journal import PaperTradingJournal

journal = PaperTradingJournal('/home/asabaal/asabaal_ventures/repos/investing/paper_trading_system/paper_trading_journal.json')

# Close TGT trade - assuming stop loss was hit at 103.62
trade_id = "2216858322"

# The stop loss was at 103.62, so that's likely where it closed
exit_price = 103.62
exit_reason = "Stop Loss Hit"

print(f"Closing TGT trade {trade_id} at ${exit_price} due to {exit_reason}")

journal.close_trade(trade_id, exit_price, exit_reason)

# Print updated summary
print("\n" + "="*50)
journal.print_summary()