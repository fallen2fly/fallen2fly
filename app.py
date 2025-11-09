import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from textblob import TextBlob
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

# --- Page config ---
st.set_page_config(page_title="Fallen2Fly AI Stock Platform", layout="wide")
st.title("Fallen2Fly AI Stock Platform")
st.write("Professional-level stock research platform with charts, AI predictions, portfolio, news, and alerts.")

# --- Auto-refresh ---
count = st_autorefresh(interval=60*1000, limit=None, key="auto_refresh")

# --- Sidebar Settings ---
st.sidebar.header("Settings")

# --- Tick List ---
nasdaq100 = ["AAPL","MSFT","TSLA","NVDA","GOOGL","AMZN","FB","PYPL","ADBE","INTC","NFLX","CMCSA","PEP","CSCO","AVGO","TXN","QCOM","COST","AMGN","HON","SBUX","INTU","MDLZ","ISRG","BKNG","FISV","AMAT","GILD","LRCX","CHTR","VRTX","ADP","MU","ADI","SNPS","REGN","CSX","MAR","ATVI","BIIB","EA","KLAC","IDXX","MELI","ADSK","PCAR","ROST","ALGN","EXC","WDAY","NXPI","CDNS","BIDU","SIRI","CTSH","PAYX","XLNX","CTAS","EBAY","WBA","FAST","ORLY","LULU","DOCU","SNAP","MRVL","TTWO","SGEN","ILMN","FIS","KLAC","ANSS","MXIM","CDW","TER","XEL","MCHP","KHC","VRSK","TECH","DLTR","BMRN","SPLK","TCOM","VRSN","INCY","CERN","CPRT","MTCH","IDXX"]
nyse_top = ["JPM","BAC","V","DIS","UNH","KO","CVX","XOM","GS","WMT","HD","MCD","TRV","BA","MMM","CAT","C","AXP","IBM","NKE","BLK","LOW","SO","GE","DOW","DD","AIG","MS","PNC","T","CL","ETN","GM","USB","SCHW"]
crypto = ["BTC-USD","ETH-USD","ADA-USD","BNB-USD","SOL-USD"]

tickers = nasdaq100 + nyse_top + crypto

# Optional user input
user_input = st.sidebar.text_area("Extra tickers (comma separated):")
if user_input:
    tickers += [t.strip().upper() for t in user_input.split(",")]

history_days = st.sidebar.number_input("Historical days (max 10000):", 30, 10000, 365)

portfolio_input = st.sidebar.text_area("Portfolio (ticker,shares,buy_price per line):", "AAPL,10,150\nTSLA,5,700")

news_api_key = st.secrets.get("NEWS_API_KEY") if "NEWS_API_KEY" in st.secrets else None

# --- Tabs ---
tabs = st.tabs([
    "Historical Chart","Indicators","Next Day Prediction","Multi-Day Forecast",
    "Company Info","Portfolio","News & Sentiment","Alerts","Market Overview","Download Data","Watchlist"
])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = tabs

# --- Portfolio ---
portfolio = []
for line in portfolio_input.splitlines():
    try:
        t, shares, buy_price = line.split(",")
        portfolio.append({"ticker": t.strip().upper(), "shares": float(shares), "buy_price": float(buy_price)})
    except:
        continue

# --- Fetch data ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, period="1y"):
    return yf.download(tickers, period=period, interval="1d", group_by='ticker', threads=True)

all_data = fetch_data(tickers, period=f"{history_days}d")

# --- Stock selector ---
selected_stock = st.sidebar.selectbox("Select a stock", tickers)

# Safely get historical data
try:
    hist = all_data[selected_stock].copy()
except:
    hist = pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    st.warning(f"No data for {selected_stock}")

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
    st.subheader(f"{selected_stock} Next Day Prediction")
    hist_model = hist[['Close']].copy().reset_index()
    hist_model['Day'] = range(len(hist_model))
    hist_model = hist_model.dropna(subset=['Close'])
    if len(hist_model) >= 2:
        model = LinearRegression()
        model.fit(hist_model[['Day']], hist_model['Close'])
        pred = model.predict([[len(hist_model)]])
        st.metric(f"{selected_stock} Predicted Next Close", f"${pred[0]:.2f}", delta=f"${pred[0]-hist_model['Close'].iloc[-1]:.2f}")
    else:
        st.warning("Not enough data for prediction.")

# --- Tab 4: LSTM Multi-Day Forecast ---
with tab4:
    st.subheader(f"{selected_stock} LSTM 7-Day Forecast")
    df_close = hist[['Close']].dropna()
    if len(df_close) >= 50:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_close)
        X, y = [], []
        window = 30
        for i in range(window, len(scaled)):
            X.append(scaled[i-window:i,0])
            y.append(scaled[i,0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1],1)
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
        model_lstm.add(LSTM(50))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)
        last_window = scaled[-window:].reshape(1,window,1)
        preds = []
        for _ in range(7):
            p = model_lstm.predict(last_window)
            preds.append(p[0,0])
            last_window = np.append(last_window[:,1:,:],[[[p[0,0]]]], axis=1)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
        st.write(pd.DataFrame({"Day": range(1,8), "Predicted Close": preds.flatten()}))
    else:
        st.warning("Not enough data for LSTM forecast.")

# --- Tab 5: Company Info ---
with tab5:
    st.subheader(f"{selected_stock} Company Info")
    info = yf.Ticker(selected_stock).info
    try:
        current_price = yf.Ticker(selected_stock).history(period="1d", interval="1m")['Close'].iloc[-1]
    except:
        current_price = hist['Close'].iloc[-1] if not hist.empty else 0
    st.write({
        "Name": info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": info.get("marketCap"),
        "Open": info.get("open"),
        "Prev Close": info.get("previousClose"),
        "Realtime Price": current_price,
        "RSI": round(hist['RSI'].iloc[-1],2) if not hist.empty else None,
        "Buy Signal": hist['Buy_Signal'].iloc[-1] if not hist.empty else False,
        "Sell Signal": hist['Sell_Signal'].iloc[-1] if not hist.empty else False
    })

# --- Tab 6: Portfolio ---
with tab6:
    st.subheader("Portfolio")
    data = []
    total_value = 0
    total_cost = 0
    for item in portfolio:
        t = item['ticker']
        try:
            df = all_data[t].copy()
            cp = df['Close'].iloc[-1]
        except:
            cp = 0
        value = item['shares']*cp
        cost = item['shares']*item['buy_price']
        total_value += value
        total_cost += cost
        data.append({"Ticker":t,"Shares":item['shares'],"Buy Price":item['buy_price'],"Current Price":cp,"Value":value,"P/L":value-cost})
    if data:
        st.dataframe(pd.DataFrame(data))
        st.write(f"Total Value: ${total_value:.2f}, Total P/L: ${total_value-total_cost:.2f}")
        # Pie chart
        df_chart = pd.DataFrame(data)
        fig = go.Figure(go.Pie(labels=df_chart['Ticker'], values=df_chart['Value'], hole=.3))
        st.plotly_chart(fig)

# --- Tab 7: News & Sentiment ---
if news_api_key:
    with tab7:
        st.subheader(f"{selected_stock} News & Sentiment")
        url = f"https://newsapi.org/v2/everything?q={selected_stock}&sortBy=publishedAt&apiKey={news_api_key}"
        try:
            r = requests.get(url)
            news = r.json().get("articles", [])
            for article in news[:10]:
                title = article['title']
                sentiment = TextBlob(title).sentiment.polarity
                st.markdown(f"[{title}]({article['url']}) - {article['source']['name']} (Sentiment: {round(sentiment,2)})")
        except:
            st.write("Error fetching news")

# --- Tab 8: Alerts ---
with tab8:
    st.subheader("Alerts")
    if not hist.empty:
        buy_alert = hist['Buy_Signal'].iloc[-1]
        sell_alert = hist['Sell_Signal'].iloc[-1]
        if buy_alert: st.success(f"{selected_stock} BUY signal!")
        if sell_alert: st.warning(f"{selected_stock} SELL signal!")
        if not buy_alert and not sell_alert: st.info("No alerts.")
    else:
        st.info("No alerts due to missing data.")

# --- Tab 9: Market Overview ---
with tab9:
    st.subheader("Market Overview")
    try:
        market = yf.download("^IXIC", period="1d", interval="1d")
        st.line_chart(market['Close'])
    except:
        st.write("Market overview unavailable.")

# --- Tab 10: Download Data ---
with tab10:
    st.subheader("Download Data")
    if not hist.empty:
        st.download_button("Download CSV", hist.to_csv().encode('utf-8'), file_name=f"{selected_stock}_historical.csv", mime="text/csv")
    else:
        st.info("No data to download.")

# --- Tab 11: Multi-stock Watchlist ---
with tab11:
    st.subheader("Watchlist Overview")
    watchlist = tickers[:20]  # limit for performance
    watch_data = []
    for t in watchlist:
        try:
            df = all_data[t].copy()
            price = df['Close'].iloc[-1]
            rsi = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / (-df['Close'].diff().clip(upper=0).rolling(14).mean()))).iloc[-1])
            buy = (rsi < 30) & (price < df['Close'].rolling(20).mean().iloc[-1])
            sell = (rsi > 70) & (price > df['Close'].rolling(20).mean().iloc[-1])
            watch_data.append({"Ticker": t, "Price": price, "RSI": round(rsi,2), "Buy": buy, "Sell": sell})
        except:
            continue
    st.dataframe(pd.DataFrame(watch_data))
