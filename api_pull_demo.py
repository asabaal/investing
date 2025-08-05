#!/usr/bin/env python3
from simple_data_puller import SimpleSession

# Create session
session = SimpleSession('NVDA')

# ACTUALLY PULL from API (not just read database)
print("🔄 Pulling fresh data from API...")
daily = session.add_daily()                    # Pulls from API 
hourly = session.add_intraday('60min')         # Pulls from API
minute = session.add_intraday('15min')         # Pulls from API

print(f"Fresh Daily: {len(daily)}")
print(f"Fresh Hourly: {len(hourly)}")  
print(f"Fresh Minute: {len(minute)}")

# Or use the update functions directly
from simple_data_puller import update_data
print("\n🔄 Direct API updates...")
update_data('NVDA', 'daily')                   # Direct API call
update_data('NVDA', '60min')                   # Direct API call

print("Done - actually pulled from API!")