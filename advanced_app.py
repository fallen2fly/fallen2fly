import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import feedparser

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(page_title="Fallen2Fly Pro", layout="wide")

# -------------------------------------
# HEADER
# -------------------------------------
st.title("🚀 Fallen2Fly Advanced Dashboard")
st.markdown("Your full-blown finance intelligence hub — real-time stocks, news, predictions, and portfolio tracking.")

# -------------------------------------
# SIDEBAR
# -------------------------------------
st.sidebar.header("Settings")

# Stock selection
default_stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "NVDA", "META", "NFLX", "JPM", "AMD"]
stock_symbols = st.sidebar.text_input(
    "Enter stock tickers (comma-separated):",
    ",".join(default_stocks)
).upper().split(",")

# Date range
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Prices",
    "📉 History",
    "🤖 Predictions",
    "📰 News",
    "💰 Portfolio",
    "ℹ️ About"
])

# -------------------------------------
# TAB 1: Live Prices
# -------------------------------------
with tab1:
    st.header("📊 Live Market Data")
    stock_data = []

    for symbol in stock_symbols:
        try:
            ticker = yf.Ticker(symbol.strip())
            info = ticker.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            prev = info.get("previousClose")
            change = None if price is None or prev is None else round(((price - prev) / prev) * 100, 2)
            stock_data.append({
                "Symbol": symbol.strip(),
                "Company": info.get("longName", "Unknown"),
                "Price": price,
                "Change (%)": change
            })
        except Exception as e:
            st.warning(f"⚠️ Could not load {symbol}: {e}")

    if stock_data:
        df_live = pd.DataFrame(stock_data)
        st.dataframe(df_live, use_container_width=True)
    else:
        st.error("No valid stocks found.")

# -------------------------------------
# TAB 2: History
# -------------------------------------
with tab2:
    st.header("📉 Historical Charts")
    selected = st.selectbox("Choose a stock", stock_symbols)

    try:
        hist = yf.download(selected, start=start_date, end=end_date)
        if not hist.empty:
            hist.reset_index(inplace=True)
            fig = px.line(hist, x="Date", y="Close", title=f"{selected} - Historical Closing Prices")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist.tail(20), use_container_width=True)
        else:
            st.warning("No data found for that range.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# -------------------------------------
# TAB 3: Predictions
# -------------------------------------
with tab3:
    st.header("🤖 Machine Learning Forecasts")
    target = st.selectbox("Predict for:", stock_symbols, key="ml_stock")
    days = st.slider("Predict days ahead:", 1, 30, 7)

    try:
        df = yf.download(target, start=start_date, end=end_date)
        if len(df) > 10:
            df.reset_index(inplace=True)
            df["Day"] = np.arange(len(df))
            X = df[["Day"]]
            y = df["Close"]

            model = LinearRegression()
            model.fit(X, y)

            future_days = np.arange(len(df), len(df) + days).reshape(-1, 1)
            future_preds = model.predict(future_days)
            future_dates = [df["Date"].iloc[-1] + timedelta(days=i + 1) for i in range(days)]

            pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds})
            full_df = pd.concat([
                df[["Date", "Close"]].rename(columns={"Close": "Price"}),
                pred_df.rename(columns={"Predicted Close": "Price"})
            ])

            fig_pred = px.line(full_df, x="Date", y="Price", title=f"{target} - Predicted Next {days} Days")
            st.plotly_chart(fig_pred, use_container_width=True)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.warning("Not enough data to predict.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------
# TAB 4: News
# -------------------------------------
with tab4:
    st.header("📰 Latest Stock Market News")
    try:
        news_feed = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for entry in news_feed.entries[:15]:
            st.markdown(f"### [{entry.title}]({entry.link})")
            st.caption(entry.published)
            st.write(entry.summary[:300] + "...")
            st.markdown("---")
    except Exception as e:
        st.error(f"Error loading news: {e}")

# -------------------------------------
# TAB 5: Portfolio
# -------------------------------------
with tab5:
    st.header("💰 Personal Portfolio Tracker")

    st.markdown("Enter your stock holdings below 👇")

    tickers_input = st.text_area("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA")
    shares_input = st.text_area("Enter shares (comma-separated):", "10, 5, 3")

    try:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        shares = [float(s.strip()) for s in shares_input.split(",")]
        if len(tickers) != len(shares):
            st.error("Number of tickers and shares must match.")
        else:
            data = []
            for t, s in zip(tickers, shares):
                ticker = yf.Ticker(t)
                price = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice")
                total = (price or 0) * s
                data.append({"Symbol": t, "Shares": s, "Price": price, "Value": total})

            df_portfolio = pd.DataFrame(data)
            df_portfolio["Value"] = df_portfolio["Value"].round(2)
            total_value = df_portfolio["Value"].sum()

            st.dataframe(df_portfolio, use_container_width=True)
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")

# -------------------------------------
# TAB 6: About
# -------------------------------------
with tab6:
    st.header("ℹ️ About Fallen2Fly Pro")
    st.markdown("""
    **Fallen2Fly Pro** is a comprehensive financial web app that lets you:
    - 📊 View live stock prices  
    - 📉 Analyze history  
    - 🤖 Predict future prices  
    - 📰 Read the latest financial news  
    - 💰 Track your own portfolio  
    ---
    Built with ❤️ using **Streamlit, Plotly, Scikit-learn, and YFinance**.
    """)
  
