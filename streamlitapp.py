# Function to fetch data and update the portfolio assets
def fetch_and_validate_ticker_data(ticker_symbol: str, lookback_years: int):
    """
    Fetches price data for a single ticker, validates it, and returns the data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))

    try:
        # Use yfinance.Ticker to check if the ticker is valid
        ticker_info = yf.Ticker(ticker_symbol).info
        if not ticker_info:
            return None, f"⚠️ Ticker **{ticker_symbol.upper()}** not found or inactive."

        # Fetch the historical data
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

        # Check if the DataFrame is empty (common failure point)
        if data.empty:
            return None, f"⚠️ No price data retrieved for **{ticker_symbol.upper()}** in the selected date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})."

        # Check for sufficient data points (e.g., more than 5 days)
        if len(data) < 5:
            return None, f"⚠️ Only {len(data)} data points retrieved for **{ticker_symbol.upper()}**. Insufficient data for analysis."

        return data, None # Success

    except Exception as e:
        # Catch any other network or API errors
        return None, f"❌ An error occurred while fetching data for **{ticker_symbol.upper()}**: {e}"


# Example of how to integrate this logic where you process the new ticker input
# (This assumes 'new_ticker' is the string input and 'lookback_years' is available)
# --- (Placeholder for your Streamlit UI logic) ---
# new_ticker = st.session_state.new_ticker_input.upper().strip()
# lookback_years = st.session_state.lookback_years

# if new_ticker and st.button("Add Asset"):
#     data, error_message = fetch_and_validate_ticker_data(new_ticker, lookback_years)
#     if data is not None:
#         st.success(f"✅ Successfully added asset: **{new_ticker}**")
#         # Your existing logic to update st.session_state.portfolio_assets and re-run analysis
#         # ...
#     else:
#         st.error(error_message)
