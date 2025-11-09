import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Auto-refresh ---
update_interval = 60
st_autorefresh(interval=update_interval*1000, key="auto_refresh")

# --- Page config ---
st.set_page_config(page_title="Fallen2Fly Ultimate Stock Tracker", layout="wide")
st.title("Fallen2Fly Ultimate AI Stock Dashboard")
st.write("Track multiple stocks, see historical data, technical indicators, predictions, and portfolio performance!")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma separated, up to 10 recommended):",
    "AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, FB, JPM, BAC, NFLX"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")][:10]

history_days = st.sidebar.number_input("Historical data days (max 7300):", 30, 7300, 365)

portfolio_input = st.sidebar.text_area(
    "Portfolio (ticker,shares,buy_price per line):",
    "AAPL,10,150\nTSLA,5,700"
)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Historical Chart", "Indicators", "Next Day Prediction", "Company Info", "Portfolio"])

# --- Portfolio processing ---
portfolio = []
for line in portfolio_input.splitlines():
    try:
        t, shares, buy_price = line.split(",")
        portfolio.append({
            "ticker": t.strip().upper(),
            "shares": float(shares),
            "buy_price": float(buy_price)
        })
    except:
        continue

# --- Loop through tickers ---
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{history_days}d", interval="1d")
        if hist.empty:
            st.warning(f"No data for {ticker}")
            continue

        # --- Technical indicators ---
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        hist['Upper_BB'] = hist['SMA_20'] + 2*hist['Close'].rolling(20).std()
        hist['Lower_BB'] = hist['SMA_20'] - 2*hist['Close'].rolling(20).std()

        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1*delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100/(1+rs))

        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema12 - ema26

        # --- Tab 1: Historical Chart ---
        with tab1:
            st.subheader(f"{ticker} Historical Prices")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="SMA20"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], name="EMA20"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Upper_BB'], name="Upper BB", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Lower_BB'], name="Lower BB", line=dict(dash="dot")))
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 2: Indicators ---
        with tab2:
            st.subheader(f"{ticker} Indicators")
            st.line_chart(hist[['RSI','MACD']])

        # --- Tab 3: Next Day Prediction ---
        with tab3:
            hist['Day'] = range(len(hist))
            model = LinearRegression()
            model.fit(hist[['Day']], hist['Close'])
            pred = model.predict([[len(hist)]])
            st.metric(label=f"{ticker} Predicted Next Close", value=f"${pred[0]:.2f}", delta=f"${pred[0]-hist['Close'][-1]:.2f}")

        # --- Tab 4: Company Info ---
        with tab4:
            st.subheader(f"{ticker} Company Info")
            info = stock.info
            current_price = stock.history(period="1d", interval="1m")['Close'][-1]
            st.write({
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Market Cap": info.get("marketCap"),
                "Open": info.get("open"),
                "Prev Close": info.get("previousClose"),
                "Realtime Price": current_price
            })

        # --- Tab 5: Portfolio ---
        with tab5:
            st.subheader("Portfolio Simulation")
            data = []
            total_value = 0
            total_cost = 0
            for item in portfolio:
                t = item['ticker']
                shares = item['shares']
                buy = item['buy_price']
                stk = yf.Ticker(t)
                cp = stk.history(period="1d", interval="1m")['Close'][-1]
                value = shares*cp
                cost = shares*buy
                total_value += value
                total_cost += cost
                data.append({"Ticker":t,"Shares":shares,"Buy Price":buy,"Current Price":cp,"Value":value,"P/L":value-cost})
            if data:
                st.dataframe(pd.DataFrame(data))
                st.write(f"Total Portfolio Value: ${total_value:.2f}")
                st.write(f"Total P/L: ${total_value-total_cost:.2f}")
            else:
                st.write("No valid portfolio data.")

        st.divider()

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
