import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Fallen2Fly Stock Tracker", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.title("📈 Fallen2Fly Stock Tracker & Predictor")
st.markdown("Real-time data, history, and machine learning forecasts for global stocks.")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Settings")

# Select stocks
default_stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "NVDA", "META", "NFLX", "JPM", "AMD"]
stock_symbols = st.sidebar.text_input(
    "Enter stock tickers (comma-separated):",
    ",".join(default_stocks)
).upper().split(",")

# Select date range
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Prices", "📉 Historical Data", "🤖 Predictions", "🧠 About"])

# ---------------------------
# TAB 1: Live Prices
# ---------------------------
with tab1:
    st.header("📊 Live Stock Prices")
    data_list = []
    for symbol in stock_symbols:
        try:
            ticker = yf.Ticker(symbol.strip())
            info = ticker.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")
            change = None if current_price is None or previous_close is None else round(((current_price - previous_close) / previous_close) * 100, 2)
            data_list.append({
                "Symbol": symbol.strip(),
                "Company": info.get("longName", "Unknown"),
                "Current Price": current_price,
                "Change (%)": change
            })
        except Exception as e:
            st.warning(f"⚠️ Could not fetch data for {symbol.strip()}: {e}")

    if data_list:
        df_live = pd.DataFrame(data_list)
        st.dataframe(df_live, use_container_width=True)
    else:
        st.error("No valid data found. Check your stock symbols.")

# ---------------------------
# TAB 2: Historical Data
# ---------------------------
with tab2:
    st.header("📉 Historical Stock Data")
    selected_stock = st.selectbox("Select a stock:", stock_symbols)
    if selected_stock:
        try:
            hist = yf.download(selected_stock, start=start_date, end=end_date)
            if not hist.empty:
                hist.reset_index(inplace=True)
                fig = px.line(hist, x="Date", y="Close", title=f"{selected_stock} Closing Prices")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(hist.tail(20), use_container_width=True)
            else:
                st.warning("No historical data found for that date range.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# ---------------------------
# TAB 3: Machine Learning Predictions
# ---------------------------
with tab3:
    st.header("🤖 Stock Price Prediction (Linear Regression)")
    symbol = st.selectbox("Pick a stock to predict:", stock_symbols, key="predict_stock")
    days_to_predict = st.slider("Days to predict ahead:", 1, 30, 7)

    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if len(df) > 10:
            df.reset_index(inplace=True)
            df["Day"] = np.arange(len(df))
            X = df[["Day"]]
            y = df["Close"]

            model = LinearRegression()
            model.fit(X, y)

            future_days = np.arange(len(df), len(df) + days_to_predict).reshape(-1, 1)
            predictions = model.predict(future_days)
            future_dates = [df["Date"].iloc[-1] + timedelta(days=i + 1) for i in range(days_to_predict)]

            pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions})
            full_df = pd.concat([df[["Date", "Close"]].rename(columns={"Close": "Price"}), pred_df.rename(columns={"Predicted Close": "Price"})])

            fig_pred = px.line(full_df, x="Date", y="Price", title=f"{symbol} Price Prediction for Next {days_to_predict} Days")
            st.plotly_chart(fig_pred, use_container_width=True)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.warning("Not enough data to predict. Try a longer date range.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ---------------------------
# TAB 4: About
# ---------------------------
with tab4:
    st.header("🧠 About Fallen2Fly")
    st.markdown("""
    **Fallen2Fly** is a real-time stock tracker and predictor web app built using:
    - 📊 **Streamlit** for UI  
    - 🧮 **scikit-learn** for predictions  
    - 💹 **Plotly** for visualizations  
    - 🌎 **Yahoo Finance API (yfinance)** for data  
    ---
    🔥 Designed for learning, analysis, and financial education — not trading advice.
    """)

    st.markdown("Created by [You 🚀] for finance & data science experimentation.")
