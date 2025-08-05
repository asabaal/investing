#!/usr/bin/env python3
from simple_data_puller import SimpleSession

# Create session
session = SimpleSession('SPY')

# Pull data with time windows
daily = session.get_daily('2024-01-01', '2024-12-31')      # 2024 daily data
hourly = session.get_intraday('60min', '2024-01-01')       # Hourly from start of 2024
minute = session.get_intraday('15min', '2024-12-01')       # Minute from Dec 2024

print(f"Daily 2024: {len(daily)}")
print(f"Hourly 2024+: {len(hourly)}")  
print(f"Minute Dec+: {len(minute)}")

# Pull more data - different windows
daily_all = session.get_daily()                            # All daily data
hourly_recent = session.get_intraday('60min', '2025-07-01') # Recent hourly
minute_today = session.get_intraday('15min', '2025-08-01')  # Very recent minute

print(f"Daily all: {len(daily_all)}")
print(f"Hourly recent: {len(hourly_recent)}")
print(f"Minute recent: {len(minute_today)}")

print("Done - got data with different time windows!")