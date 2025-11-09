import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from textblob import TextBlob
import requests

# --- Page config ---
st.set_page_config(page_title="Fallen2Fly Ultimate AI Stock Dashboard", layout="wide")
st.title("Fallen2Fly Ultimate AI Stock Dashboard")
st.write("Full-featured AI Stock Research Platform with charts, predictions, portfolio, news, alerts, and more!")

# --- Sidebar Settings ---
st.sidebar.header("Settings")

# 300+ tickers (NASDAQ100 + NYSE + crypto)
nasdaq100 = ["AAPL","MSFT","TSLA","NVDA","GOOGL","AMZN","FB","PYPL","ADBE","INTC","NFLX","CMCSA","PEP","CSCO","AVGO","TXN","QCOM","COST","AMGN","HON","SBUX","INTU","MDLZ","ISRG","BKNG","FISV","AMAT","GILD","LRCX","CHTR","VRTX","ADP","MU","ADI","SNPS","REGN","CSX","MAR","ATVI","BIIB","EA","KLAC","IDXX","MELI","ADSK","PCAR","ROST","ALGN","EXC","WDAY","NXPI","CDNS","BIDU","SIRI","CTSH","PAYX","XLNX","CTAS","EBAY","WBA","FAST","ORLY","LULU","DOCU","SNAP","MRVL","TTWO","SGEN","ILMN","FIS","KLAC","ANSS","MXIM","CDW","TER","XEL","MCHP","KHC","VRSK","TECH","DLTR","BMRN","SPLK","TCOM","VRSN","INCY","CERN","CPRT","MTCH","IDXX"]
nyse_top = ["JPM","BAC","V","DIS","UNH","KO","CVX","XOM","GS","WMT","HD","MCD","TRV","BA","MMM","CAT","C","AXP","IBM","NKE","BLK","LOW","SO","GE","DOW","DD","AIG","MS","PNC","T","CL","ETN","GM","USB","SCHW"]
crypto = ["BTC-USD","ETH-USD","ADA-USD","BNB-USD","SOL-USD"]

# Combine
tickers = nasdaq100 + nyse_top + crypto

# Optional user input
user_input = st.sidebar.text_area("Enter extra tickers (comma separated):")
if user_input:
    tickers += [t.strip().upper() for t in user_input.split(",")]

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

# --- Parse portfolio ---
portfolio = []
for line in portfolio_input.splitlines():
    try:
        t, shares, buy_price = line.split(",")
        portfolio.append({"ticker": t.strip().upper(), "shares": float(shares), "buy_price": float(buy_price)})
    except:
        continue

# --- Fetch batch data ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, period="1y"):
    return yf.download(tickers, period=period, interval="1d", group_by='ticker', threads=True)

all_data = fetch_data(tickers, period=f"{history_days}d")

# --- Stock selector ---
selected_stock = st.sidebar.selectbox("Select a stock to view", tickers)
hist = all_data[selected_stock].copy()

# --- Technical Indicators ---
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
hist['Buy_Signal'] = (hist['RSI'] < 30) & (hist['Close'] < hist['SMA_20'])
hist['Sell_Signal'] = (hist['RSI'] > 70) & (hist['Close'] > hist['SMA_20'])

# --- Tab 1: Historical Chart ---
with tab1:
    st.subheader(f"{selected_stock} Historical Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="SMA20"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], name="EMA20"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Upper_BB'], name="Upper BB", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Lower_BB'], name="Lower BB", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Indicators ---
with tab2:
    st.subheader(f"{selected_stock} Indicators")
    st.line_chart(hist[['RSI','MACD']])

# --- Tab 3: Next Day Prediction ---
with tab3:
    hist['Day'] = range(len(hist))
    model = LinearRegression()
    model.fit(hist[['Day']], hist['Close'])
    pred = model.predict([[len(hist)]])
    st.metric(label=f"{selected_stock} Predicted Next Close", value=f"${pred[0]:.2f}", delta=f"${pred[0]-hist['Close'][-1]:.2f}")

# --- Tab 4: Multi-Day Forecast ---
with tab4:
    st.subheader(f"{selected_stock} Multi-Day Forecast (Next 7 Days)")
    X = np.arange(len(hist)).reshape(-1,1)
    y = hist['Close'].values
    lr = LinearRegression()
    lr.fit(X,y)
    future_days = np.arange(len(hist), len(hist)+7).reshape(-1,1)
    predictions = lr.predict(future_days)
    st.write(pd.DataFrame({"Day": [i+1 for i in range(7)], "Predicted Close": predictions}))

# --- Tab 5: Company Info ---
with tab5:
    st.subheader(f"{selected_stock} Company Info")
    info = yf.Ticker(selected_stock).info
    try:
        current_price = yf.Ticker(selected_stock).history(period="1d", interval="1m")['Close'][-1]
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
        try:
            df = all_data[t].copy()
            cp = df['Close'][-1]
        except:
            cp = 0
        value = item['shares']*cp
        cost = item['shares']*item['buy_price']
        total_value += value
        total_cost += cost
        data.append({"Ticker":t,"Shares":item['shares'],"Buy Price":item['buy_price'],"Current Price":cp,"Value":value,"P/L":value-cost})
    if data:
        st.dataframe(pd.DataFrame(data))
        st.write(f"Total Portfolio Value: ${total_value:.2f}")
        st.write(f"Total P/L: ${total_value-total_cost:.2f}")

# --- Tab 7: News & Sentiment ---
if news_api_key:
    with tab7:
        st.subheader(f"{selected_stock} News with Sentiment")
        url = f"https://newsapi.org/v2/everything?q={selected_stock}&sortBy=publishedAt&apiKey={news_api_key}"
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
    st.subheader(f"{selected_stock} Alerts")
    buy_alert = hist['Buy_Signal'][-1]
    sell_alert = hist['Sell_Signal'][-1]
    if buy_alert:
        st.success(f"{selected_stock}: BUY signal triggered!")
    if sell_alert:
        st.warning(f"{selected_stock}: SELL signal triggered!")
    if not buy_alert and not sell_alert:
        st.info(f"{selected_stock}: No alerts currently.")

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
        file_name=f"{selected_stock}_historical.csv",
        mime="text/csv"
    )
