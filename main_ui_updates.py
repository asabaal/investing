# Main UI updates for app.py

# Add this code to the main app area of app.py

# Add a persistent status area for showing current operation status
operation_status = st.empty()

# Display fetch status if active
if 'fetch_state' in st.session_state and st.session_state.fetch_state['is_fetching']:
    operation_status.info(f"Active operation in progress: Fetching month {st.session_state.fetch_state['current_month']} of {st.session_state.fetch_state['total_months']}")

# Render fetch status in sidebar if active
render_fetch_status_sidebar()

# When starting API calls for multi-month, use this code:
# Show loading indicator
with operation_status:
    st.info(f"Starting data fetch operation for {ticker}...")

# Update your "Load Stock Data" button handling code to include this
# status visualization at startup

# When creating the multi-month data option, add this disclaimer:
# Display a prominent warning for 24-month case
if months_total > 20:
    st.markdown("""
    <div style="background-color: #ffe8d6; color: #774936; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #d8a48f;">
        <h4>⚠️ Long Operation Warning</h4>
        <p>You've selected a large date range that will require many API calls.</p>
        <p>This operation may take a long time due to API rate limits.</p>
        <p>The app will remain active throughout the process with progress updates.</p>
    </div>
    """, unsafe_allow_html=True)
