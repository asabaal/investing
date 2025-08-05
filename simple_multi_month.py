#!/usr/bin/env python3
from direct_api_puller import get_intraday_data_api
import pandas as pd

# Pull multiple months of 15min data for AAPL
data = get_intraday_data_api('AAPL', '15min', '2025-05-01', '2025-08-05')
print(f"Got {len(data):,} records from May to August")
print(f"Range: {data.index.min()} to {data.index.max()}")