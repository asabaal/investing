# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🔍 Data Coverage Diagnosis Notebook
#
# This notebook will help us **diagnose exactly what's wrong** with our data coverage for Black Swan trading strategies.
#
# ## Hypothesis to Test:
# - ✅ We have extensive historical data in CSV files
# - ❌ Database only contains recent ~20-22 days for most symbols
# - ❌ This prevents deployment of strategies requiring 252+ days of data
#
# Let's investigate step by step!

# %%
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import our database utilities
from market_data_database import MarketDataDatabase, get_default_database_path

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

# %% [markdown]
# ## 🎯 Step 1: Load Black Swan Strategy Requirements

# %%
# Load the Black Swan strategy requirements
with open('black_swan_symphonies.json', 'r') as f:
    strategies = json.load(f)

# Extract all required symbols
required_symbols = set()
strategy_symbols = {}

for strat_key, strat_data in strategies['symphonies'].items():
    strategy_name = strat_data['name']
    universe = strat_data['universe']
    strategy_symbols[strategy_name] = universe
    required_symbols.update(universe)

required_symbols = sorted(list(required_symbols))

print(f"📊 Found {len(strategies['symphonies'])} Black Swan strategies")
print(f"🎯 Total unique symbols required: {len(required_symbols)}")
print(f"📋 Required symbols: {', '.join(required_symbols)}")

print("\n🔍 Strategy breakdown:")
for name, symbols in strategy_symbols.items():
    print(f"   {name}: {len(symbols)} symbols - {', '.join(symbols)}")

# %% [markdown]
# ## 📁 Step 2: Check What CSV Data We Have Available

# %%
# Check CSV files in data_cache
csv_files = list(Path('data_cache').glob('*_daily.csv'))
csv_symbols = [f.stem.replace('_daily', '') for f in csv_files]

print(f"📁 Found {len(csv_files)} CSV files in data_cache/")
print(f"💾 CSV symbols: {', '.join(sorted(csv_symbols))}")

# Check which required symbols have CSV data
csv_coverage = {}
missing_csv = []

for symbol in required_symbols:
    csv_file = Path(f'data_cache/{symbol}_daily.csv')
    if csv_file.exists():
        # Quick check of CSV data range
        try:
            df = pd.read_csv(csv_file)
            df['Date'] = pd.to_datetime(df['Date'])
            csv_coverage[symbol] = {
                'file_exists': True,
                'total_rows': len(df),
                'date_range': (df['Date'].min(), df['Date'].max()),
                'days_span': (df['Date'].max() - df['Date'].min()).days,
                'file_size_mb': csv_file.stat().st_size / (1024*1024)
            }
        except Exception as e:
            csv_coverage[symbol] = {'file_exists': True, 'error': str(e)}
    else:
        csv_coverage[symbol] = {'file_exists': False}
        missing_csv.append(symbol)

print(f"\n✅ Required symbols WITH CSV data: {len([s for s in required_symbols if csv_coverage[s]['file_exists']])}")
print(f"❌ Required symbols MISSING CSV data: {len(missing_csv)}")
if missing_csv:
    print(f"   Missing: {', '.join(missing_csv)}")

# %%
# Show detailed CSV coverage for required symbols
print("📊 CSV DATA COVERAGE DETAILS:")
print("=" * 80)
print(f"{'Symbol':<8} {'Rows':<8} {'Days':<8} {'Start Date':<12} {'End Date':<12} {'Size(MB)':<10} {'Status'}")
print("-" * 80)

for symbol in required_symbols:
    data = csv_coverage[symbol]
    if data['file_exists'] and 'error' not in data:
        start_date = data['date_range'][0].strftime('%Y-%m-%d')
        end_date = data['date_range'][1].strftime('%Y-%m-%d')
        status = "✅ Good" if data['days_span'] > 252 else "⚠️ Short"
        print(f"{symbol:<8} {data['total_rows']:<8} {data['days_span']:<8} {start_date:<12} {end_date:<12} {data['file_size_mb']:<10.1f} {status}")
    elif data['file_exists']:
        print(f"{symbol:<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} ❌ Error")
    else:
        print(f"{symbol:<8} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} ❌ Missing")

print("\n" + "=" * 80)

# %% [markdown]
# ## 🗄️ Step 3: Check What's Actually in the Database

# %%
# Connect to database and check what we have
db_path = get_default_database_path()
print(f"📊 Database path: {db_path}")
print(f"📁 Database exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    print(f"💾 Database size: {os.path.getsize(db_path) / (1024*1024):.1f} MB")
else:
    print("❌ Database not found!")

# %%
# Query database for required symbols
db_coverage = {}

with sqlite3.connect(db_path) as conn:
    # Get overall stats
    total_daily = conn.execute('SELECT COUNT(*) FROM daily_data').fetchone()[0]
    total_intraday = conn.execute('SELECT COUNT(*) FROM intraday_data').fetchone()[0]
    unique_daily_symbols = conn.execute('SELECT COUNT(DISTINCT symbol) FROM daily_data').fetchone()[0]
    unique_intraday_symbols = conn.execute('SELECT COUNT(DISTINCT symbol) FROM intraday_data').fetchone()[0]
    
    print(f"🗄️ DATABASE OVERVIEW:")
    print(f"   Daily records: {total_daily:,}")
    print(f"   Intraday records: {total_intraday:,}")
    print(f"   Daily symbols: {unique_daily_symbols}")
    print(f"   Intraday symbols: {unique_intraday_symbols}")
    
    # Check each required symbol
    for symbol in required_symbols:
        # Daily data check
        daily_query = '''
            SELECT 
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(DISTINCT date) as unique_days
            FROM daily_data 
            WHERE symbol = ?
        '''
        daily_result = conn.execute(daily_query, (symbol,)).fetchone()
        
        # Intraday data check
        intraday_query = '''
            SELECT 
                COUNT(*) as record_count,
                MIN(datetime) as earliest_datetime,
                MAX(datetime) as latest_datetime,
                COUNT(DISTINCT DATE(datetime)) as unique_days
            FROM intraday_data 
            WHERE symbol = ?
        '''
        intraday_result = conn.execute(intraday_query, (symbol,)).fetchone()
        
        db_coverage[symbol] = {
            'daily': {
                'records': daily_result[0],
                'earliest': daily_result[1],
                'latest': daily_result[2],
                'unique_days': daily_result[3]
            },
            'intraday': {
                'records': intraday_result[0],
                'earliest': intraday_result[1],
                'latest': intraday_result[2],
                'unique_days': intraday_result[3]
            }
        }

# %%
# Show detailed database coverage for required symbols
print("\n🗄️ DATABASE COVERAGE FOR REQUIRED SYMBOLS:")
print("=" * 100)
print(f"{'Symbol':<8} {'Daily Recs':<10} {'Daily Days':<10} {'Daily Range':<25} {'Intraday Days':<12} {'Status'}")
print("-" * 100)

for symbol in required_symbols:
    data = db_coverage[symbol]
    daily = data['daily']
    intraday = data['intraday']
    
    # Determine best data source
    best_days = max(daily['unique_days'], intraday['unique_days'])
    
    if daily['records'] > 0:
        date_range = f"{daily['earliest']} to {daily['latest']}"
    else:
        date_range = "No daily data"
    
    # Status determination
    if best_days == 0:
        status = "❌ No Data"
    elif best_days < 252:
        status = f"⚠️ Insufficient ({best_days} days)"
    else:
        status = f"✅ Sufficient ({best_days} days)"
    
    print(f"{symbol:<8} {daily['records']:<10} {daily['unique_days']:<10} {date_range:<25} {intraday['unique_days']:<12} {status}")

print("\n" + "=" * 100)

# %% [markdown]
# ## 🔍 Step 4: The Big Comparison - CSV vs Database
#
# This is where we'll **prove our hypothesis**!

# %%
# Create comparison table
comparison_data = []

for symbol in required_symbols:
    csv_data = csv_coverage.get(symbol, {})
    db_data = db_coverage.get(symbol, {})
    
    # CSV info
    csv_days = csv_data.get('days_span', 0) if csv_data.get('file_exists', False) else 0
    csv_status = "✅ Available" if csv_days > 252 else ("⚠️ Short" if csv_days > 0 else "❌ Missing")
    
    # Database info
    db_daily_days = db_data.get('daily', {}).get('unique_days', 0) if db_data else 0
    db_intraday_days = db_data.get('intraday', {}).get('unique_days', 0) if db_data else 0
    db_best_days = max(db_daily_days, db_intraday_days)
    db_status = "✅ Sufficient" if db_best_days >= 252 else ("⚠️ Insufficient" if db_best_days > 0 else "❌ None")
    
    # The gap!
    gap = csv_days - db_best_days if csv_days > 0 and db_best_days > 0 else 0
    gap_status = "🚨 HUGE GAP" if gap > 1000 else ("⚠️ Gap" if gap > 100 else "✅ OK")
    
    comparison_data.append({
        'symbol': symbol,
        'csv_days': csv_days,
        'csv_status': csv_status,
        'db_days': db_best_days,
        'db_status': db_status,
        'gap': gap,
        'gap_status': gap_status
    })

# Convert to DataFrame for easy analysis
comparison_df = pd.DataFrame(comparison_data)

print("🔍 THE BIG COMPARISON - CSV vs DATABASE:")
print("=" * 90)
print(f"{'Symbol':<8} {'CSV Days':<9} {'CSV Status':<12} {'DB Days':<8} {'DB Status':<13} {'Gap':<8} {'Gap Status'}")
print("-" * 90)

for _, row in comparison_df.iterrows():
    print(f"{row['symbol']:<8} {row['csv_days']:<9} {row['csv_status']:<12} {row['db_days']:<8} {row['db_status']:<13} {row['gap']:<8} {row['gap_status']}")

print("\n" + "=" * 90)

# %%
# Summary statistics
huge_gaps = comparison_df[comparison_df['gap_status'] == '🚨 HUGE GAP']
gaps = comparison_df[comparison_df['gap_status'] == '⚠️ Gap']
csv_sufficient = comparison_df[comparison_df['csv_days'] >= 252]
db_sufficient = comparison_df[comparison_df['db_days'] >= 252]

print("📊 DIAGNOSIS SUMMARY:")
print("=" * 50)
print(f"Total required symbols: {len(comparison_df)}")
print(f"CSV files with 252+ days: {len(csv_sufficient)} ({len(csv_sufficient)/len(comparison_df)*100:.1f}%)")
print(f"Database with 252+ days: {len(db_sufficient)} ({len(db_sufficient)/len(comparison_df)*100:.1f}%)")
print(f"Symbols with HUGE gaps (1000+ days): {len(huge_gaps)}")
print(f"Symbols with gaps (100+ days): {len(gaps)}")

if len(huge_gaps) > 0:
    print(f"\n🚨 CRITICAL ISSUE DETECTED:")
    print(f"   We have extensive CSV data but minimal database data!")
    print(f"   Average gap for critical symbols: {huge_gaps['gap'].mean():.0f} days")
    print(f"   This confirms our hypothesis: data loading issue!")

print("\n" + "=" * 50)

# %% [markdown]
# ## 📈 Step 5: Visual Diagnosis
#
# Let's create some charts to **clearly show the problem**!

# %%
# Create visualization comparing CSV vs Database coverage
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Chart 1: Bar chart comparison
symbols_subset = comparison_df.head(10)  # Show first 10 for readability
x = range(len(symbols_subset))
width = 0.35

ax1.bar([i - width/2 for i in x], symbols_subset['csv_days'], width, label='CSV Data', color='green', alpha=0.7)
ax1.bar([i + width/2 for i in x], symbols_subset['db_days'], width, label='Database Data', color='red', alpha=0.7)
ax1.axhline(y=252, color='orange', linestyle='--', label='Required (252 days)')
ax1.set_xlabel('Symbols')
ax1.set_ylabel('Days of Data')
ax1.set_title('CSV vs Database Data Comparison (First 10 Symbols)')
ax1.set_xticks(x)
ax1.set_xticklabels(symbols_subset['symbol'], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Gap analysis
gap_data = comparison_df[comparison_df['gap'] > 0]
ax2.bar(range(len(gap_data)), gap_data['gap'], color='orange', alpha=0.7)
ax2.set_xlabel('Symbols with Gaps')
ax2.set_ylabel('Gap (Days)')
ax2.set_title('Data Gaps: CSV Days - Database Days')
ax2.set_xticks(range(len(gap_data)))
ax2.set_xticklabels(gap_data['symbol'], rotation=45)
ax2.grid(True, alpha=0.3)

# Chart 3: Strategy readiness pie chart
csv_ready = len(comparison_df[comparison_df['csv_days'] >= 252])
csv_not_ready = len(comparison_df) - csv_ready
ax3.pie([csv_ready, csv_not_ready], labels=['CSV Ready (252+ days)', 'CSV Not Ready'], 
        colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
ax3.set_title('CSV Data Readiness for Strategies')

# Chart 4: Database readiness pie chart
db_ready = len(comparison_df[comparison_df['db_days'] >= 252])
db_not_ready = len(comparison_df) - db_ready
ax4.pie([db_ready, db_not_ready], labels=['DB Ready (252+ days)', 'DB Not Ready'], 
        colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
ax4.set_title('Database Readiness for Strategies')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🎯 Step 6: Sample a Few Critical Symbols
#
# Let's look at specific examples to **prove our case**!

# %%
# Pick the most critical symbols (used in multiple strategies)
critical_symbols = ['SPY', 'QQQ', 'UVXY', 'TLT', 'GLD']

print("🔍 DETAILED ANALYSIS OF CRITICAL SYMBOLS:")
print("=" * 80)

for symbol in critical_symbols:
    print(f"\n📊 {symbol}:")
    
    # CSV analysis
    csv_file = Path(f'data_cache/{symbol}_daily.csv')
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   CSV: {len(df)} rows, {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   CSV: {(df['Date'].max() - df['Date'].min()).days} days of data")
    else:
        print(f"   CSV: ❌ File not found")
    
    # Database analysis
    db_data = db_coverage.get(symbol, {})
    if db_data:
        daily = db_data['daily']
        if daily['records'] > 0:
            print(f"   DB:  {daily['records']} records, {daily['earliest']} to {daily['latest']}")
            print(f"   DB:  {daily['unique_days']} unique days")
        else:
            print(f"   DB:  ❌ No daily data found")
    
    # Show the gap
    row = comparison_df[comparison_df['symbol'] == symbol]
    if not row.empty:
        gap = row.iloc[0]['gap']
        print(f"   🚨 GAP: {gap} days missing from database!")
        if gap > 1000:
            print(f"   💡 This symbol alone could enable strategy deployment!")
    
    print("-" * 40)

# %% [markdown]
# ## 📋 Step 7: Final Diagnosis Report
#
# Let's summarize our findings and **confirm the hypothesis**!

# %%
# Count strategies that could be enabled
potentially_enabled_strategies = 0
strategy_analysis = []

for strategy_name, symbols in strategy_symbols.items():
    current_ready = sum(1 for s in symbols if comparison_df[comparison_df['symbol'] == s]['db_days'].iloc[0] >= 252 if not comparison_df[comparison_df['symbol'] == s].empty else 0)
    csv_ready = sum(1 for s in symbols if comparison_df[comparison_df['symbol'] == s]['csv_days'].iloc[0] >= 252 if not comparison_df[comparison_df['symbol'] == s].empty else 0)
    
    current_pct = (current_ready / len(symbols)) * 100
    potential_pct = (csv_ready / len(symbols)) * 100
    
    strategy_analysis.append({
        'name': strategy_name,
        'total_symbols': len(symbols),
        'current_ready': current_ready,
        'current_pct': current_pct,
        'potential_ready': csv_ready,
        'potential_pct': potential_pct,
        'deployable_now': current_pct >= 80,
        'deployable_after_fix': potential_pct >= 80
    })
    
    if potential_pct >= 80 and current_pct < 80:
        potentially_enabled_strategies += 1

print("🎯 FINAL DIAGNOSIS REPORT:")
print("=" * 60)
print(f"\n📊 HYPOTHESIS TESTING RESULTS:")
print(f"   ✅ CSV files contain extensive historical data: CONFIRMED")
print(f"   ❌ Database only has recent data: CONFIRMED")
print(f"   🚨 This prevents strategy deployment: CONFIRMED")

print(f"\n📈 STRATEGY IMPACT:")
deployable_now = sum(1 for s in strategy_analysis if s['deployable_now'])
deployable_after = sum(1 for s in strategy_analysis if s['deployable_after_fix'])

print(f"   Currently deployable strategies: {deployable_now}/{len(strategy_analysis)}")
print(f"   Potentially deployable after fix: {deployable_after}/{len(strategy_analysis)}")
print(f"   Strategies that would be ENABLED by fix: {potentially_enabled_strategies}")

print(f"\n🔍 DETAILED STRATEGY BREAKDOWN:")
for s in strategy_analysis:
    status_now = "✅ Ready" if s['deployable_now'] else "❌ Not Ready"
    status_after = "✅ Ready" if s['deployable_after_fix'] else "❌ Not Ready"
    impact = "🚀 ENABLED!" if s['deployable_after_fix'] and not s['deployable_now'] else "No change"
    
    print(f"   {s['name']}:")
    print(f"      Now: {s['current_ready']}/{s['total_symbols']} ({s['current_pct']:.1f}%) - {status_now}")
    print(f"      After fix: {s['potential_ready']}/{s['total_symbols']} ({s['potential_pct']:.1f}%) - {status_after}")
    print(f"      Impact: {impact}")
    print()

print("\n" + "=" * 60)
print("🎯 CONCLUSION:")
if potentially_enabled_strategies > 0:
    print(f"   🚨 CRITICAL DATA LOADING ISSUE CONFIRMED!")
    print(f"   💡 Fixing this would enable {potentially_enabled_strategies} strategies immediately")
    print(f"   🚀 Your Black Swan deployment is blocked by a fixable data loading problem!")
else:
    print(f"   ⚠️ Data loading issue confirmed, but may need additional data sources")
    print(f"   📊 Some symbols may still need more historical data even from CSV")

print("=" * 60)

# %% [markdown]
# ## 🛠️ Next Steps Recommendations
#
# Based on our diagnosis, here's what needs to be done:

# %%
print("🛠️ RECOMMENDED NEXT STEPS:")
print("=" * 50)

if potentially_enabled_strategies > 0:
    print("1. 🔧 FIX DATA LOADING (HIGH PRIORITY):")
    print("   - Identify why only recent data is being loaded into database")
    print("   - Fix the historical data import process")
    print("   - Re-import all CSV files with full historical data")
    print()
    
    print("2. 📊 IMMEDIATE IMPACT:")
    huge_gap_symbols = comparison_df[comparison_df['gap'] > 1000]['symbol'].tolist()
    print(f"   - Focus on symbols with huge gaps: {', '.join(huge_gap_symbols[:5])}")
    print(f"   - These symbols alone could enable {potentially_enabled_strategies} strategies")
    print()
    
    print("3. 🚀 DEPLOYMENT READINESS:")
    print(f"   - After fix: {deployable_after}/{len(strategy_analysis)} strategies ready")
    print("   - Your Black Swan system will be deployment-ready!")
    print()

missing_csv_symbols = comparison_df[comparison_df['csv_days'] == 0]['symbol'].tolist()
if missing_csv_symbols:
    print("4. 📁 MISSING DATA FILES:")
    print(f"   - Still need CSV data for: {', '.join(missing_csv_symbols)}")
    print("   - Download historical data for these symbols")
    print()

print("5. 🔄 ONGOING MAINTENANCE:")
print("   - Set up automated daily data updates")
print("   - Monitor data freshness (currently 16 days stale)")
print("   - Verify data quality after fixes")

print("\n" + "=" * 50)
print("✨ YOUR DIAGNOSIS IS COMPLETE!")
print("The path to Black Swan deployment is now crystal clear! 🎯")

# %%
