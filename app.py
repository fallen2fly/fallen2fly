import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Title and description ---
st.set_page_config(page_title="Fallen2Fly AI Stock Tracker")
st.title("Fallen2Fly AI Stock Tracker")
st.write("Track stocks and predict growth/fall with AI.")

# --- User input for stock ticker ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", "AAPL")

if ticker:
    try:
        # Fetch stock data for the past month
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")

        if hist.empty:
            st.error("No data found for this ticker.")
        else:
            # Show historical price chart
            st.subheader("Price History (Last 30 Days)")
            st.line_chart(hist['Close'])

            # --- Simple AI prediction ---
            hist['Day'] = range(len(hist))
            X = hist[['Day']]
            y = hist['Close']
            model = LinearRegression()
            model.fit(X, y)

            next_day = [[len(hist)]]
            prediction = model.predict(next_day)

            st.subheader("Next Day Predicted Price")
            st.write(f"${prediction[0]:.2f}")

            # Show latest stock info
            st.subheader("Company Info")
            info = stock.info
            st.write({
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "Market Cap": info.get("marketCap"),
                "Previous Close": info.get("previousClose"),
                "Open": info.get("open")
            })

    except Exception as e:
        st.error(f"Error fetching data: {e}")
