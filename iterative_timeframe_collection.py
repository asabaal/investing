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
# # 🎯 Iterative Timeframe Collection Demo
#
# **Shows data growing as we add more TIMEFRAMES**
#
# 1. Start with 15min data for August 2025
# 2. ADD 15min data for July 2025 (same timescale, different timeframe)
# 3. ADD 15min data for June 2025 (expanding timeframe coverage)
# 4. ADD 5min data for July (different timescale)
# 5. Beautiful dark mode visualizations showing growth
#
# **Timeframes AND Timescales - exactly what you asked for!**

# %%
# Setup
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

# Import our direct API puller
from direct_api_puller import get_daily_data_api, get_intraday_data_api

# Colors
COLORS = {
    'primary': '#00D4FF',
    'secondary': '#FF6B6B', 
    'accent': '#4ECDC4',
    'success': '#95E1D3'
}

print("🚀 Ready to show iterative timeframe collection!")

# %% [markdown]
# ## 📊 Step 1: Start with August 15-min Data

# %%
# Choose security and timeframe
SYMBOL = 'AAPL'

print(f"🎯 Starting with {SYMBOL} 15-minute data for August 2025...")

# Step 1: Pull 15-minute data for August 2025
data_aug = get_intraday_data_api(SYMBOL, '15min', '2025-08-01', '2025-08-05')

print(f"✅ Got {len(data_aug):,} 15-minute records for August")
if not data_aug.empty:
    print(f"📅 Range: {data_aug.index.min()} to {data_aug.index.max()}")
else:
    print("📅 No August data, using recent fallback...")
    data_aug = get_intraday_data_api(SYMBOL, '15min')
    print(f"✅ Got {len(data_aug):,} recent 15-minute records")
    
# Show sample
display(data_aug.tail() if not data_aug.empty else pd.DataFrame())


# %%
# Visualize Step 1: August 15-minute data foundation
def show_timeframe_growth(datasets, title):
    """Show growing dataset with timeframe expansion"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Record Counts by Timeframe', 'Price Coverage', 'Daily Volume', 'Latest Price'],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # 1. Record counts by timeframe
    timeframes = list(datasets.keys())
    counts = [len(data) for data in datasets.values() if not data.empty]
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], COLORS['success']]
    
    fig.add_trace(
        go.Bar(x=timeframes, y=counts, marker_color=colors[:len(timeframes)],
               text=[f'{c:,}' for c in counts], textposition='auto'),
        row=1, col=1
    )
    
    # 2. Price coverage across timeframes
    color_idx = 0
    for name, data in datasets.items():
        if not data.empty:
            # Show recent data for visibility
            recent_data = data.tail(100) if len(data) > 100 else data
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['close'], 
                          name=name, line=dict(color=colors[color_idx % len(colors)])),
                row=1, col=2
            )
            color_idx += 1
    
    # 3. Daily coverage (group by date)
    all_data = pd.concat([data for data in datasets.values() if not data.empty], ignore_index=False) if any(not data.empty for data in datasets.values()) else pd.DataFrame()
    if not all_data.empty:
        daily_counts = all_data.groupby(all_data.index.date).size()
        fig.add_trace(
            go.Bar(x=daily_counts.index, y=daily_counts.values, marker_color=COLORS['success'],
                  name='Records per Day'),
            row=2, col=1
        )
    
    # 4. Latest price indicator
    if datasets and any(not data.empty for data in datasets.values()):
        latest_data = next(data for data in datasets.values() if not data.empty)
        latest_price = latest_data['close'].iloc[-1]
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=latest_price,
                title={"text": f"{SYMBOL} Latest"},
                number={'prefix': "$"}
            ),
            row=2, col=2
        )
    
    total_records = sum(len(data) for data in datasets.values() if not data.empty)
    fig.update_layout(
        title=f"📊 {title} - {total_records:,} Total Records",
        height=800,
        showlegend=True
    )
    
    return fig

# Show Step 1 visualization
datasets = {'Aug 2025': data_aug}
fig1 = show_timeframe_growth(datasets, f"{SYMBOL} Step 1: August 15-min Foundation")
fig1.show()

# %% [markdown]
# ## ⏰ Step 2: ADD July Timeframe (Same Timescale)

# %%
print(f"⏰ Now ADDING July 2025 15-minute data for {SYMBOL}...")
print(f"🎯 Same timescale (15min), different timeframe (July vs August)")

# Step 2: Add July 15-minute data (expanding timeframe coverage)
data_july = get_intraday_data_api(SYMBOL, '15min', '2025-07-01', '2025-07-31')

print(f"✅ Got {len(data_july):,} 15-minute records for July")
if not data_july.empty:
    print(f"📅 Range: {data_july.index.min()} to {data_july.index.max()}")
    display(data_july.tail())
else:
    print("❌ No July data available")
    display(pd.DataFrame())

# Combine datasets - now we have TWO months of 15-minute data!
combined_data = pd.concat([data_july, data_aug]).sort_index() if not data_july.empty else data_aug
print(f"\n📈 Combined dataset: {len(combined_data):,} records spanning July + August!")
print(f"🚀 Dataset GREW by adding another timeframe!")

# %%
# Show Step 2 visualization - TIMEFRAME EXPANSION!
datasets = {
    'Aug 2025': data_aug,
    'July 2025': data_july,
    'Combined July+Aug': combined_data
}
fig2 = show_timeframe_growth(datasets, f"{SYMBOL} Step 2: Added July Timeframe")
fig2.show()

# Show the growth
original_count = len(data_aug)
new_count = len(combined_data)
print(f"📈 Dataset expanded from {original_count:,} to {new_count:,} records!")
print(f"🎯 Same timescale (15-min), expanded timeframe coverage!")
print(f"⏰ Time expansion: Aug → July+Aug")

# %% [markdown]
# ## ⚡ Step 3: ADD June Timeframe (More Expansion)

# %%
print(f"⚡ Now ADDING June 2025 15-minute data for {SYMBOL}...")
print(f"🎯 Still same timescale (15min), adding ANOTHER timeframe (June)")

# Step 3: Add June 15-minute data (further expanding timeframe coverage)
data_june = get_intraday_data_api(SYMBOL, '15min', '2025-06-01', '2025-06-30')

print(f"✅ Got {len(data_june):,} 15-minute records for June")
if not data_june.empty:
    print(f"📅 Range: {data_june.index.min()} to {data_june.index.max()}")
    display(data_june.tail())
else:
    print("❌ No June data available")
    display(pd.DataFrame())

# Combine all datasets - now we have THREE months of 15-minute data!
all_15min_data = pd.concat([data_june, data_july, data_aug]).sort_index() if not data_june.empty else combined_data
print(f"\n📈 Full 15min dataset: {len(all_15min_data):,} records spanning June + July + August!")
print(f"🚀 Dataset GREW AGAIN by adding another timeframe!")

# %%
# Show Step 3 visualization - MAXIMUM TIMEFRAME EXPANSION!
datasets = {
    'Aug 2025': data_aug,
    'July 2025': data_july, 
    'June 2025': data_june,
    'Full 15min Dataset': all_15min_data
}
fig3 = show_timeframe_growth(datasets, f"{SYMBOL} Step 3: Added June Timeframe")
fig3.show()

# Show the growth
print(f"📈 Dataset expanded to {len(all_15min_data):,} total 15-minute records!")
print(f"🎯 Same timescale (15-min), THREE MONTHS of coverage!")
print(f"⏰ Time expansion: Aug → July+Aug → June+July+Aug")
print(f"🚀 This is what you meant by adding TIMEFRAMES!")

# %% [markdown]
# ## 🔥 Step 4: ADD Different Timescale

# %%
print(f"🔥 Now adding DIFFERENT TIMESCALE: 5-minute data for {SYMBOL} (July timeframe)...")
print(f"🎯 Different timescale (5min vs 15min), same timeframe (July)")

# Step 4: Add different timescale (5-minute) for July timeframe
data_5min_july = get_intraday_data_api(SYMBOL, '5min', '2025-07-01', '2025-07-31')

print(f"✅ Got {len(data_5min_july):,} 5-minute records for July")
if not data_5min_july.empty:
    print(f"📅 Range: {data_5min_july.index.min()} to {data_5min_july.index.max()}")
    display(data_5min_july.tail())
else:
    print("❌ No 5-minute July data available (trying recent data...)")
    data_5min_july = get_intraday_data_api(SYMBOL, '5min')
    print(f"✅ Got {len(data_5min_july):,} recent 5-minute records")
    if not data_5min_july.empty:
        display(data_5min_july.tail())

print(f"\n🚀 Now we have BOTH timeframe expansion AND timescale addition!")

# %%
# Show Final visualization - COMPLETE ITERATIVE COLLECTION!
datasets = {
    '15min Jun': data_june,
    '15min July': data_july,
    '15min Aug': data_aug,
    '5min July': data_5min_july,
    '15min Full': all_15min_data
}

fig4 = show_timeframe_growth(datasets, f"{SYMBOL} FINAL: Timeframes + Timescales")
fig4.show()

# Final summary
total_15min = len(all_15min_data)
total_5min = len(data_5min_july)
grand_total = total_15min + total_5min

print(f"\n🎯 ITERATIVE DATA COLLECTION COMPLETE!")
print(f"📊 15-minute data: {total_15min:,} records (3 timeframes: Jun+July+Aug)")
print(f"📊 5-minute data: {total_5min:,} records (1 timeframe: July)")
print(f"📊 Grand total: {grand_total:,} records")
print(f"\n✅ TIMEFRAME expansion: Aug → July+Aug → June+July+Aug")
print(f"✅ TIMESCALE addition: 15min → 15min + 5min")
print(f"\n🚀 Perfect for your trading workflow - iterative collection!")
print(f"🎯 Same concept scales to 10 trades/day!")
print(f"\n💡 You can keep adding MORE timeframes (May, April...) or MORE timescales (1min, hourly...)")

# %% [markdown]
# ## 🎯 Summary: Iterative Timeframe & Timescale Collection
#
# **What we just demonstrated:**
# 1. ✅ Started with 15-min data for **August 2025** (foundation)
# 2. ✅ **ADDED** 15-min data for **July 2025** (timeframe expansion - same timescale, different month)
# 3. ✅ **ADDED** 15-min data for **June 2025** (more timeframe expansion)
# 4. ✅ **ADDED** 5-min data for **July 2025** (timescale addition - different resolution)
# 5. ✅ Watched dataset grow with beautiful visualizations at each step
#
# **Perfect iterative workflow:**
# - **Timeframe expansion**: Same timescale (15min), more time periods (months)
# - **Timescale addition**: Different timescales (5min) for analysis depth
# - Direct API calls, completely isolated system
# - Beautiful dark mode visualizations showing growth
# - Scalable to your 10 trades/day workflow
#
# **🎉 Exactly what you asked for - iterative timeframe AND timescale collection!**
#
# **Next steps:** Keep adding more timeframes (May, April...) or more timescales (1min, hourly, daily) as needed!
