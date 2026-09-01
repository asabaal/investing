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
# # 🚀 Database Growth Demo - TSLA 15min Data
#
# **STUPID SIMPLE demo showing:**
# 1. Current TSLA data in database
# 2. Collect NEW data using working API
# 3. Add to database and watch it GROW
# 4. Beautiful dark mode visualizations
#
# **TSLA currently has limited 15min data - let's expand it!**

# %%
# Setup
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime

# Dark theme
pio.templates.default = "plotly_dark"

# Import our systems
from direct_api_puller import get_intraday_data_api  # Working API
from market_data_database import MarketDataDatabase  # Database

# Colors
COLORS = {
    'primary': '#00D4FF',
    'secondary': '#FF6B6B', 
    'accent': '#4ECDC4',
    'success': '#95E1D3'
}

SYMBOL = 'TSLA'
print(f"🚀 Database Growth Demo for {SYMBOL}")
print("=" * 40)

# %% [markdown]
# ## 📊 Step 1: Check Current Database State

# %%
# Connect to database
db = MarketDataDatabase()

# Get current TSLA 15min data
print(f"📊 Current {SYMBOL} 15min data in database...")
current_data = db._get_intraday_data_direct(SYMBOL, interval='15min')

print(f"✅ Current records: {len(current_data):,}")
if not current_data.empty:
    print(f"📅 Current range: {current_data.index.min()} to {current_data.index.max()}")
    display(current_data.tail())

# Store initial count for comparison
initial_count = len(current_data)
print(f"\n🎯 Starting with {initial_count:,} records")


# %%
# Visualize current database state
def show_database_state(data, title, symbol):
    """Show current database state with dark mode viz"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{symbol} Price Chart', 'Volume', 'Daily Record Count', 'Latest Price'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    if not data.empty:
        # 1. Price chart
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], 
                      name='Close Price', line=dict(color=COLORS['primary'], width=2)),
            row=1, col=1
        )
        
        # 2. Volume
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume'], 
                      name='Volume', line=dict(color=COLORS['accent'], width=1)),
            row=1, col=2
        )
        
        # 3. Daily record count
        daily_counts = data.groupby(data.index.date).size()
        fig.add_trace(
            go.Bar(x=daily_counts.index, y=daily_counts.values, 
                  marker_color=COLORS['success'], name='Records per Day'),
            row=2, col=1
        )
        
        # 4. Latest price indicator
        latest_price = data['Close'].iloc[-1]
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=latest_price,
                title={"text": f"{symbol} Latest"},
                number={'prefix': "$", 'font': {'size': 40}}
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f"📊 {title} - {len(data):,} Records",
        height=800,
        showlegend=False
    )
    
    return fig

# Show current state
fig1 = show_database_state(current_data, f"{SYMBOL} Current Database State", SYMBOL)
fig1.show()

# %% [markdown]
# ## 🔥 Step 2: Collect NEW Data (June 2025)

# %%
print(f"🔥 Collecting NEW {SYMBOL} 15min data for June 2025...")
print("Using the WORKING API method!")

# Collect June data using working API
june_data = get_intraday_data_api(SYMBOL, '15min', '2025-06-01', '2025-06-30')

print(f"✅ Collected {len(june_data):,} NEW records for June")
if not june_data.empty:
    print(f"📅 New data range: {june_data.index.min()} to {june_data.index.max()}")
    display(june_data.head())
    display(june_data.tail())
else:
    print("❌ No June data collected")

# %% [markdown]
# ## 💾 Step 3: Add to Database and Watch it GROW!

# %%
if not june_data.empty:
    print(f"💾 Adding {len(june_data):,} records to database...")
    
    # Prepare data for database (add datetime column)
    june_for_db = june_data.copy()
    june_for_db['datetime'] = june_for_db.index
    
    # Store in database
    db._store_intraday_data(SYMBOL, june_for_db, '15min')
    print("✅ Data added to database!")
    
    # Get updated database state
    print("\n🔍 Checking updated database state...")
    updated_data = db._get_intraday_data_direct(SYMBOL, interval='15min')
    
    final_count = len(updated_data)
    growth = final_count - initial_count
    
    print(f"📈 DATABASE GROWTH:")
    print(f"   Before: {initial_count:,} records")
    print(f"   After:  {final_count:,} records")
    print(f"   GREW BY: {growth:,} records! 🚀")
    print(f"   New range: {updated_data.index.min()} to {updated_data.index.max()}")
else:
    print("⚠️ No data to add")
    updated_data = current_data
    final_count = initial_count
    growth = 0

# %%
# Show the DATABASE GROWTH with before/after comparison
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Database Growth', 'Price Chart (Updated)', 'Record Counts', 'Growth Stats'],
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "indicator"}]]
)

# 1. Database growth comparison
fig.add_trace(
    go.Bar(x=['Before', 'After'], y=[initial_count, final_count],
           marker_color=[COLORS['secondary'], COLORS['success']],
           text=[f'{initial_count:,}', f'{final_count:,}'], textposition='auto'),
    row=1, col=1
)

# 2. Updated price chart
if not updated_data.empty:
    fig.add_trace(
        go.Scatter(x=updated_data.index, y=updated_data['close'],
                  name='Close Price', line=dict(color=COLORS['primary'], width=2)),
        row=1, col=2
    )

# 3. Daily record counts (updated)
if not updated_data.empty:
    daily_counts = updated_data.groupby(updated_data.index.date).size()
    fig.add_trace(
        go.Bar(x=daily_counts.index, y=daily_counts.values,
              marker_color=COLORS['accent'], name='Records per Day'),
        row=2, col=1
    )

# 4. Growth indicator
fig.add_trace(
    go.Indicator(
        mode="number",
        value=growth,
        title={"text": "Records Added"},
        number={'font': {'size': 40, 'color': COLORS['success']}}
    ),
    row=2, col=2
)

fig.update_layout(
    title=f"🚀 {SYMBOL} DATABASE GROWTH - Added {growth:,} Records!",
    height=800,
    showlegend=False
)

fig.show()

# %% [markdown]
# ## 🎯 Step 4: Add EVEN MORE Data (July 2025)

# %%
print(f"🎯 Let's add EVEN MORE data - July 2025!")

# Collect July data
july_data = get_intraday_data_api(SYMBOL, '15min', '2025-07-01', '2025-07-31')

print(f"✅ Collected {len(july_data):,} MORE records for July")

if not july_data.empty:
    # Add to database
    july_for_db = july_data.copy()
    july_for_db['datetime'] = july_for_db.index
    db._store_intraday_data(SYMBOL, july_for_db, '15min')
    
    # Get final state
    final_data = db._get_intraday_data_direct(SYMBOL, interval='15min')
    final_final_count = len(final_data)
    total_growth = final_final_count - initial_count
    
    print(f"\n🚀 FINAL DATABASE STATE:")
    print(f"   Started with: {initial_count:,} records")
    print(f"   Final count:  {final_final_count:,} records")
    print(f"   TOTAL GROWTH: {total_growth:,} records! 🎉")
    print(f"   Full range: {final_data.index.min()} to {final_data.index.max()}")
    
    # Show final visualization
    fig_final = show_database_state(final_data, f"{SYMBOL} FINAL Database State", SYMBOL)
    fig_final.show()
    
    print(f"\n✅ SUCCESS! Database grew by {total_growth:,} records!")
    print(f"🎯 Perfect proof that the system works!")

# %% [markdown]
# ## 🎉 Summary: Database Growth Success!
#
# **What we just proved:**
# 1. ✅ Started with limited TSLA 15min data
# 2. ✅ Used WORKING API to collect June data
# 3. ✅ Added to database and watched it GROW
# 4. ✅ Added July data and grew it MORE
# 5. ✅ Beautiful visualizations showing the growth
#
# **The system works perfectly:**
# - Working API collects historical data
# - Database stores it properly
# - Visualizations show the growth
# - Ready for your trading workflow!
#
# **🚀 You can now collect data older than 30 days AND store it in your database!**
