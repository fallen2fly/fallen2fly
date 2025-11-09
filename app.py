import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import requests
from textblob import TextBlob
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh ---
update_interval = 60
st_autorefresh(interval=update_interval*1000, key="auto_refresh")

# --- Page setup ---
st.set_page_config(page_title="Fallen2Fly Ultimate AI Stock Dashboard", layout="wide")
st.title("Fallen2Fly Ultimate AI Stock Dashboard")
st.write("Full-featured AI Stock Research Platform: charts, predictions, news, portfolio, alerts, multi-market support.")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma separated):",
    "AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, FB, JPM, BAC, NFLX"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

history_days = st.sidebar.number_input("Days of historical data (max 10000):", 30, 10000, 365)
portfolio_input = st.sidebar.text_area(
    "Portfolio (ticker,shares,buy_price per line):",
    "AAPL,10,150\nTSLA,5,700"
)

news_api_key = st.secrets.get("NEWS_API_KEY") if "NEWS_API_KEY" in st.secrets else None

# --- Tabs ---
tabs = st.tabs([
    "Historical Chart","Indicators","Next Day Prediction","Multi-Day Forecast",
    "Company Info","Portfolio","News & Sentiment","Alerts","Market Overview","Download Data"
])

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = tabs

# --- Portfolio Parsing ---
portfolio = []
for line in portfolio_input.splitlines():
    try:
        t, shares, buy_price = line.split(",")
        portfolio.append({"ticker": t.strip().upper(), "shares": float(shares), "buy_price": float(buy_price)})
    except:
        continue

# --- Function: Technical Indicators ---
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Upper_BB'] = df['SMA_20'] + 2*df['Close'].rolling(20).std()
    df['Lower_BB'] = df['SMA_20'] - 2*df['Close'].rolling(20).std()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100/(1+rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['Close'] < df['SMA_20'])
    df['Sell_Signal'] = (df['RSI'] > 70) & (df['Close'] > df['SMA_20'])
    return df

# --- Loop through tickers ---
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{history_days}d", interval="1d")
        if hist.empty:
            st.warning(f"No data for {ticker}")
            continue

        hist = calculate_indicators(hist)

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

        # --- Tab 4: Multi-Day Forecast (Simple placeholder for LSTM) ---
        with tab4:
            st.subheader(f"{ticker} Multi-Day Forecast (Next 7 Days)")
            X = np.arange(len(hist)).reshape(-1,1)
            y = hist['Close'].values
            lr = LinearRegression()
            lr.fit(X,y)
            future_days = np.arange(len(hist), len(hist)+7).reshape(-1,1)
            predictions = lr.predict(future_days)
            st.write(pd.DataFrame({"Day": [i+1 for i in range(7)], "Predicted Close": predictions}))

        # --- Tab 5: Company Info ---
        with tab5:
            st.subheader(f"{ticker} Company Info")
            info = stock.info
            try:
                current_price = stock.history(period="1d", interval="1m")['Close'][-1]
            except:
                current_price = hist['Close'][-1]
            st.write({
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Market Cap": info.get("marketCap"),
                "Open": info.get("open"),
                "Prev Close": info.get("previousClose"),
                "Realtime Price": current_price,
                "RSI": round(hist['RSI'][-1],2),
                "Buy Signal": hist['Buy_Signal'][-1],
                "Sell Signal": hist['Sell_Signal'][-1]
            })

        # --- Tab 6: Portfolio Simulation ---
        with tab6:
            st.subheader("Portfolio Simulation")
            data = []
            total_value = 0
            total_cost = 0
            for item in portfolio:
                t = item['ticker']
                shares = item['shares']
                buy = item['buy_price']
                stk = yf.Ticker(t)
                try:
                    cp = stk.history(period="1d", interval="1m")['Close'][-1]
                except:
                    cp = 0
                value = shares*cp
                cost = shares*buy
                total_value += value
                total_cost += cost
                data.append({"Ticker":t,"Shares":shares,"Buy Price":buy,"Current Price":cp,"Value":value,"P/L":value-cost})
            if data:
                st.dataframe(pd.DataFrame(data))
                st.write(f"Total Portfolio Value: ${total_value:.2f}")
                st.write(f"Total P/L: ${total_value-total_cost:.2f}")

        # --- Tab 7: News & Sentiment ---
        if news_api_key:
            with tab7:
                st.subheader(f"{ticker} News with Sentiment")
                url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={news_api_key}"
                try:
                    r = requests.get(url)
                    news = r.json().get("articles", [])
                    for article in news[:10]:
                        title = article['title']
                        sentiment = TextBlob(title).sentiment.polarity
                        st.markdown(f"[{title}]({article['url']}) - {article['source']['name']} (Sentiment: {round(sentiment,2)})")
                except:
                    st.write("Error fetching news. Check API key or connectivity.")

        # --- Tab 8: Alerts ---
        with tab8:
            st.subheader(f"{ticker} Alerts")
            buy_alert = hist['Buy_Signal'][-1]
            sell_alert = hist['Sell_Signal'][-1]
            if buy_alert:
                st.success(f"{ticker}: BUY signal triggered!")
            if sell_alert:
                st.warning(f"{ticker}: SELL signal triggered!")
            if not buy_alert and not sell_alert:
                st.info(f"{ticker}: No alerts currently.")

        # --- Tab 9: Market Overview ---
        with tab9:
            st.subheader("Top Gainers / Losers (NASDAQ)")
            try:
                market = yf.download("^IXIC", period="1d", interval="1d")
                st.line_chart(market['Close'])
            except:
                st.write("Market overview data unavailable.")

        # --- Tab 10: Download Data ---
        with tab10:
            st.subheader("Download Data")
            st.download_button(
                label="Download Historical Data CSV",
                data=hist.to_csv().encode('utf-8'),
                file_name=f"{ticker}_historical.csv",
                mime="text/csv"
            )

        st.divider()

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
