import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every N seconds ---
update_interval = 60  # seconds
st_autorefresh(interval=update_interval*1000, key="dashboard_refresh")

# --- Page setup ---
st.set_page_config(page_title="Fallen2Fly Pro Stock Dashboard", layout="wide")
st.title("Fallen2Fly Pro AI Stock Tracker")
st.write("Realtime stock tracker, predictions, and portfolio simulator!")

# --- Sidebar ---
st.sidebar.header("Dashboard Settings")

tickers_input = st.sidebar.text_area(
    "Enter stock tickers (comma separated, e.g., AAPL, TSLA, MSFT):",
    "AAPL, TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

history_days = st.sidebar.number_input(
    "Number of days of historical data (max 7300 ~ 20 yrs):", min_value=30, max_value=7300, value=365
)

portfolio_input = st.sidebar.text_area(
    "Optional portfolio: ticker,shares,buy_price (one per line, e.g., AAPL,10,150):",
    "AAPL,10,150\nTSLA,5,700"
)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Historical Chart", "Next Day Prediction", "Company Info", "Portfolio Simulation"])

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
            st.warning(f"No data found for {ticker}.")
            continue

        # --- Calculate technical indicators ---
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # --- Tab 1: Historical Chart with SMA/EMA ---
        with tab1:
            st.subheader(f"{ticker} - Historical Prices & Indicators")
            fig = px.line(hist, x=hist.index, y=['Close', 'SMA_20', 'EMA_20'],
                          title=f"{ticker} Prices with SMA/EMA")
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

        # --- Tab 3: Company Info & Realtime Price ---
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
                "Realtime Price": realtime_price,
                "RSI": round(hist['RSI'][-1],2)
            })

        st.divider()

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

# --- Tab 4: Portfolio Simulation ---
with tab4:
    st.subheader("Portfolio Simulation")
    portfolio_data = []
    total_value = 0
    total_cost = 0

    for item in portfolio:
        t = item['ticker']
        shares = item['shares']
        buy_price = item['buy_price']
        stock = yf.Ticker(t)
        try:
            current_price = stock.history(period="1d", interval="1m")['Close'][-1]
            value = shares * current_price
            cost = shares * buy_price
            profit_loss = value - cost
            total_value += value
            total_cost += cost
            portfolio_data.append({
                "Ticker": t,
                "Shares": shares,
                "Buy Price": buy_price,
                "Current Price": current_price,
                "Value": round(value,2),
                "Profit/Loss": round(profit_loss,2)
            })
        except:
            continue

    if portfolio_data:
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio)
        st.write(f"Total Portfolio Value: ${total_value:.2f}")
        st.write(f"Total Profit/Loss: ${total_value - total_cost:.2f}")
    else:
        st.write("No valid portfolio data provided.")

