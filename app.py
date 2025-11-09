import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
import plotly.express as px

# --- Page config ---
st.set_page_config(page_title="Fallen2Fly Pro Stock Dashboard", layout="wide")

st.title("Fallen2Fly Pro AI Stock Tracker")
st.write("Track multiple stocks, see historical data, predict next-day prices, and monitor real-time updates!")

# --- Sidebar for user input ---
st.sidebar.header("Stock Dashboard Settings")
tickers_input = st.sidebar.text_area(
    "Enter stock tickers (comma separated, e.g., AAPL, TSLA, MSFT):",
    "AAPL, TSLA, MSFT"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

history_days = st.sidebar.number_input(
    "Number of days to fetch historical data (max 7300 ~ 20 yrs):", min_value=30, max_value=7300, value=365
)

update_interval = st.sidebar.slider(
    "Realtime update interval (seconds):", min_value=10, max_value=600, value=60
)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Historical Chart", "Next Day Prediction", "Company Info"])

# --- Loop through tickers ---
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)

        # Fetch historical data
        hist = stock.history(period=f"{history_days}d", interval="1d")

        if hist.empty:
            st.warning(f"No data found for {ticker}.")
            continue

        # --- Tab 1: Historical Chart ---
        with tab1:
            st.subheader(f"{ticker} - Historical Closing Prices")
            fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} Historical Prices")
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 2: Next Day Prediction ---
        with tab2:
            hist['Day'] = range(len(hist))
            X = hist[['Day']]
            y = hist['Close']
            model = LinearRegression()
            model.fit(X, y)
            next_day = [[len(hist)]]
            prediction = model.predict(next_day)
            st.subheader(f"{ticker} - Next Day Predicted Close")
            st.metric(label="Predicted Price", value=f"${prediction[0]:.2f}", delta=f"${prediction[0]-hist['Close'][-1]:.2f}")

        # --- Tab 3: Company Info ---
        with tab3:
            st.subheader(f"{ticker} - Company Info & Realtime Data")
            info = stock.info
            realtime_price = stock.history(period="1d", interval="1m")['Close'][-1]
            st.write({
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Market Cap": info.get("marketCap"),
                "Previous Close": info.get("previousClose"),
                "Open": info.get("open"),
                "Realtime Price": realtime_price
            })

        st.divider()

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
