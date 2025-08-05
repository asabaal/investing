#!/usr/bin/env python3
from simple_data_puller import SimpleSession

# Create session
session = SimpleSession('SPY')

# Pull data iteratively - multiple timescales
daily = session.get_daily()           # Get daily data
hourly = session.get_intraday('60min') # Get hourly data  
minute = session.get_intraday('15min') # Get minute data

print(f"Daily: {len(daily)} | Hourly: {len(hourly)} | Minute: {len(minute)}")